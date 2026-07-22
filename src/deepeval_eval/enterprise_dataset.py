from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval_eval.config import ensure_dirs
from deepeval_eval.io_utils import download_bytes, download_text

RELEASE_BASE_URL = (
    "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"
)
QUESTIONS_JSONL_URL = "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/main/questions.jsonl"
INGESTOR_TYPE = "enterprise_rag_bench_deepeval"
INGESTOR_NAME = "enterprise-rag-bench-deepeval"
SOURCE_SLICE_COUNTS = {
    "confluence": 2,
    "jira": 2,
    "github": 2,
    "hubspot": 4,
    "fireflies": 3,
    "linear": 8,
    "google_drive": 6,
    "gmail": 25,
    "slack": 58,
}


@dataclass
class EnterpriseDoc:
    doc_id: str
    title: str
    text: str
    source_type: str


@dataclass
class EvalQuestion:
    question_id: str
    user_input: str
    reference: str
    category: str
    source_types: list[str]
    expected_doc_ids: list[str]
    answer_facts: list[Any]


def parse_doc_filename(name: str) -> tuple[str, str] | None:
    base = name.rsplit("/", 1)[-1]
    if not base.startswith("dsid_") or not base.endswith(".txt"):
        return None
    stem = base[:-4]
    if "__" not in stem:
        return None
    doc_id, slug = stem.split("__", 1)
    return doc_id, slug.replace("-", " ")


def load_questions(cache_dir: Path) -> list[EvalQuestion]:
    ensure_dirs(cache_dir)
    raw = download_text(
        QUESTIONS_JSONL_URL, cache_dir / "enterprise_rag_bench_questions.jsonl"
    )
    questions: list[EvalQuestion] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        questions.append(
            EvalQuestion(
                question_id=str(item.get("question_id") or ""),
                user_input=str(item.get("user_input") or item.get("question") or ""),
                reference=str(item.get("reference") or item.get("gold_answer") or ""),
                category=str(
                    item.get("category") or item.get("question_type") or "uncategorized"
                ),
                source_types=list(item.get("source_types") or []),
                expected_doc_ids=list(item.get("expected_doc_ids") or []),
                answer_facts=list(item.get("answer_facts") or []),
            )
        )
    return questions


def select_questions(
    questions: list[EvalQuestion],
    source_types: list[str],
    question_limit: int,
    questions_per_category: int,
) -> list[EvalQuestion]:
    wanted_sources = set(source_types)
    candidates = [
        q
        for q in questions
        if not q.source_types or bool(set(q.source_types) & wanted_sources)
    ]
    selected: list[EvalQuestion] = []
    counts: dict[str, int] = {}
    for question in candidates:
        category = question.category or "uncategorized"
        if counts.get(category, 0) >= questions_per_category:
            continue
        selected.append(question)
        counts[category] = counts.get(category, 0) + 1
        if len(selected) >= question_limit:
            break
    if len(selected) < question_limit:
        seen = {q.question_id for q in selected}
        for question in candidates:
            if question.question_id in seen:
                continue
            selected.append(question)
            if len(selected) >= question_limit:
                break
    return selected


# EnterpriseRAG-Bench is stored as many source-specific zip slices. The
# sampler streams those slices and keeps gold documents ahead of filler docs.
def fetch_documents(
    source_types: list[str],
    limit_per_source: int,
    cache_dir: Path,
    reference_doc_ids: set[str],
) -> list[EnterpriseDoc]:
    ensure_dirs(cache_dir)
    all_docs: list[EnterpriseDoc] = []
    seen_hashes: set[str] = set()
    for source_type in source_types:
        reference_docs: list[EnterpriseDoc] = []
        other_docs: list[EnterpriseDoc] = []
        n_slices = SOURCE_SLICE_COUNTS[source_type]
        print(f"Fetching {source_type}: {n_slices} slice(s)")
        for slice_num in range(1, n_slices + 1):
            zip_name = f"{source_type}_slice_{slice_num:04d}.zip"
            zip_bytes = download_bytes(
                f"{RELEASE_BASE_URL}/{zip_name}", cache_dir / zip_name
            )
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names = [n for n in zf.namelist() if n.endswith(".txt")]
                print(f"  {zip_name}: {len(names)} files")
                for name in names:
                    parsed = parse_doc_filename(name)
                    if not parsed:
                        continue
                    doc_id, fallback_title = parsed
                    raw = zf.read(name).decode("utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)
                    first_line = raw.split("\n", 1)[0].strip()
                    doc = EnterpriseDoc(
                        doc_id=doc_id,
                        title=first_line or fallback_title,
                        text=raw,
                        source_type=source_type,
                    )
                    if doc_id in reference_doc_ids:
                        reference_docs.append(doc)
                    else:
                        other_docs.append(doc)
        selected = (reference_docs + other_docs)[:limit_per_source]
        print(
            f"  selected {len(selected)} docs ({len(reference_docs)} reference docs available)"
        )
        all_docs.extend(selected)
    return all_docs


def to_caipe_payload(
    doc: EnterpriseDoc, datasource_id: str, ingestor_id: str
) -> dict[str, Any]:
    return {
        "page_content": doc.text,
        "type": "Document",
        "metadata": {
            "document_id": doc.doc_id,
            "datasource_id": datasource_id,
            "ingestor_id": ingestor_id,
            "title": doc.title,
            "description": f"EnterpriseRAG-Bench - {doc.source_type}",
            "is_structured_entity": False,
            "document_type": "text",
            "document_ingested_at": None,
            "fresh_until": None,
            "metadata": {
                "source": "enterprise_rag_bench",
                "source_type": doc.source_type,
            },
        },
    }


def write_questions(
    questions: list[EvalQuestion],
    docs_by_id: dict[str, EnterpriseDoc],
    jsonl_path: Path,
    csv_path: Path,
) -> None:
    with (
        jsonl_path.open("w", encoding="utf-8") as jf,
        csv_path.open("w", encoding="utf-8", newline="") as cf,
    ):
        writer = csv.writer(cf)
        writer.writerow(
            [
                "question_id",
                "user_input",
                "reference",
                "category",
                "source_types",
                "expected_doc_ids",
            ]
        )
        for question in questions:
            context = [
                docs_by_id[doc_id].text
                for doc_id in question.expected_doc_ids
                if doc_id in docs_by_id
            ]
            record = {
                "question_id": question.question_id,
                "user_input": question.user_input,
                "reference": question.reference,
                "category": question.category,
                "source_types": question.source_types,
                "expected_doc_ids": question.expected_doc_ids,
                "answer_facts": question.answer_facts,
                "context": context,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow(
                [
                    question.question_id,
                    question.user_input,
                    question.reference,
                    question.category,
                    ";".join(question.source_types),
                    ";".join(question.expected_doc_ids),
                ]
            )


def write_corpus(docs: list[EnterpriseDoc], jsonl_path: Path, csv_path: Path) -> None:
    with (
        jsonl_path.open("w", encoding="utf-8") as jf,
        csv_path.open("w", encoding="utf-8", newline="") as cf,
    ):
        writer = csv.writer(cf)
        writer.writerow(["document_id", "source_type", "title", "text"])
        for doc in docs:
            record = {
                "document_id": doc.doc_id,
                "source_type": doc.source_type,
                "title": doc.title,
                "text": doc.text,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow([doc.doc_id, doc.source_type, doc.title, doc.text])
