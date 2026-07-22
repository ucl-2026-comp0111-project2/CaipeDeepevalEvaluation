from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from deepeval_eval.enterprise_dataset import (
    EnterpriseDoc,
    EvalQuestion,
    parse_doc_filename,
)
from deepeval_eval.enterprise_dataset import (
    load_questions as load_enterprise_questions,
)
from deepeval_eval.enterprise_dataset import (
    select_questions as select_enterprise_questions,
)
from deepeval_eval.enterprise_dataset import (
    to_caipe_payload as enterprise_to_caipe_payload,
)
from deepeval_eval.enterprise_dataset import (
    write_corpus as write_enterprise_corpus,
)
from deepeval_eval.enterprise_dataset import (
    write_questions as write_enterprise_questions,
)
from deepeval_eval.hotpotqa_dataset import (
    load_document_pool,
    read_jsonl_zip,
    resolve_zip,
    unique,
)
from deepeval_eval.hotpotqa_dataset import (
    load_questions as load_hotpotqa_questions,
)
from deepeval_eval.hotpotqa_dataset import (
    select_documents as select_hotpotqa_documents,
)
from deepeval_eval.hotpotqa_dataset import (
    select_questions as select_hotpotqa_questions,
)
from deepeval_eval.hotpotqa_dataset import (
    to_caipe_payload as hotpotqa_to_caipe_payload,
)
from deepeval_eval.hotpotqa_dataset import (
    write_questions as write_hotpotqa_questions,
)


def test_parse_doc_filename_positive() -> None:
    res = parse_doc_filename("folder/dsid_doc123__sample-title.txt")
    assert res == ("dsid_doc123", "sample title")


def test_parse_doc_filename_negative() -> None:
    assert parse_doc_filename("invalid_name.txt") is None
    assert parse_doc_filename("dsid_nodoubleunderscore.txt") is None
    assert parse_doc_filename("dsid_123__test.doc") is None


def test_enterprise_questions_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    questions_jsonl = (
        '{"question_id": "q1", "user_input": "What is X?", "reference": "X is Y", "category": "cat1", "source_types": ["slack"], "expected_doc_ids": ["doc1"]}\n'
        '{"question_id": "q2", "question": "Where is Z?", "gold_answer": "Z is here", "question_type": "cat2", "source_types": ["jira"], "expected_doc_ids": ["doc2"]}\n'
    )
    monkeypatch.setattr(
        "deepeval_eval.enterprise_dataset.download_text",
        lambda url, dest: questions_jsonl,
    )

    q_list = load_enterprise_questions(cache_dir)
    assert len(q_list) == 2
    assert q_list[0].question_id == "q1"
    assert q_list[1].user_input == "Where is Z?"

    selected = select_enterprise_questions(
        q_list, source_types=["slack"], question_limit=1, questions_per_category=1
    )
    assert len(selected) == 1
    assert selected[0].question_id == "q1"


def test_enterprise_to_caipe_payload() -> None:
    doc = EnterpriseDoc(
        doc_id="d1", title="Title", text="Body text", source_type="slack"
    )
    payload = enterprise_to_caipe_payload(doc, datasource_id="ds1", ingestor_id="ing1")
    assert payload["page_content"] == "Body text"
    assert payload["metadata"]["document_id"] == "d1"


def test_fetch_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deepeval_eval.enterprise_dataset import fetch_documents

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "slack_slice_0001/dsid_doc999__first-doc.txt",
            "Doc Title\nBody text content",
        )

    monkeypatch.setattr(
        "deepeval_eval.enterprise_dataset.download_bytes",
        lambda url, dest: buf.getvalue(),
    )
    monkeypatch.setattr(
        "deepeval_eval.enterprise_dataset.SOURCE_SLICE_COUNTS", {"slack": 1}
    )

    docs = fetch_documents(
        source_types=["slack"],
        limit_per_source=5,
        cache_dir=tmp_path,
        reference_doc_ids={"dsid_doc999"},
    )
    assert len(docs) == 1
    assert docs[0].doc_id == "dsid_doc999"


def test_write_enterprise_files(tmp_path: Path) -> None:
    doc = EnterpriseDoc(
        doc_id="d1", title="Title", text="Body text", source_type="slack"
    )
    q = EvalQuestion(
        question_id="q1",
        user_input="Input",
        reference="Ref",
        category="cat",
        source_types=["slack"],
        expected_doc_ids=["d1"],
        answer_facts=[],
    )
    jsonl_p = tmp_path / "q.jsonl"
    csv_p = tmp_path / "q.csv"
    write_enterprise_questions([q], {"d1": doc}, jsonl_p, csv_p)
    assert jsonl_p.exists() and csv_p.exists()

    corpus_j = tmp_path / "c.jsonl"
    corpus_c = tmp_path / "c.csv"
    write_enterprise_corpus([doc], corpus_j, corpus_c)
    assert corpus_j.exists() and corpus_c.exists()


def test_unique_helper() -> None:
    assert unique(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert unique([]) == []


def test_hotpotqa_dataset_helpers(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        jsonl_data = '{"question_id": "hq1", "user_input": "What?", "reference": "Ans", "category": "catA", "expected_doc_ids": ["docA"]}\n'
        zf.writestr("items.jsonl", jsonl_data)
    zip_path.write_bytes(buf.getvalue())

    resolved = resolve_zip(zip_path, "fallback.zip")
    assert resolved == zip_path

    items = read_jsonl_zip(zip_path)
    assert len(items) == 1
    assert items[0]["question_id"] == "hq1"

    questions = load_hotpotqa_questions(zip_path)
    assert len(questions) == 1
    assert questions[0]["question_id"] == "hq1"

    selected_q = select_hotpotqa_questions(
        questions, limit=1, per_category=1, categories=None
    )
    assert len(selected_q) == 1

    pool_buf = io.BytesIO()
    with zipfile.ZipFile(pool_buf, "w") as zf:
        doc_jsonl = (
            '{"document_id": "docA", "title": "Doc A", "content": "Sample content"}\n'
        )
        zf.writestr("docs.jsonl", doc_jsonl)
    pool_zip = tmp_path / "pool.zip"
    pool_zip.write_bytes(pool_buf.getvalue())

    doc_pool = load_document_pool(pool_zip)
    assert "docA" in doc_pool

    selected_docs = select_hotpotqa_documents(
        questions, doc_pool, distractors_per_question=1, max_docs=5
    )
    assert len(selected_docs) == 1
    assert selected_docs[0]["document_id"] == "docA"

    payload = hotpotqa_to_caipe_payload(selected_docs[0], "ds", "ing")
    assert "Doc A\n\nSample content" in payload["page_content"]

    q_j = tmp_path / "hq.jsonl"
    q_c = tmp_path / "hq.csv"
    write_hotpotqa_questions(questions, doc_pool, q_j, q_c)
    assert q_j.exists() and q_c.exists()


def test_hotpotqa_fallbacks() -> None:
    from deepeval_eval.hotpotqa_dataset import select_documents, select_questions

    questions = [
        {"question_id": "q1", "category": "cat1", "expected_doc_ids": ["d1"]},
        {"question_id": "q2", "category": "cat1", "expected_doc_ids": ["d2"]},
    ]
    # Request limit=2 with per_category=1 to trigger fallback loop
    sel_q = select_questions(questions, limit=2, per_category=1, categories=None)
    assert len(sel_q) == 2

    pool = {
        "d1": {"document_id": "d1", "title": "T1", "text": "Text 1"},
        "d2": {"document_id": "d2", "title": "T2", "text": "Text 2"},
        "d3": {"document_id": "d3", "title": "T3", "text": "Text 3"},
    }
    # Request max_docs=3 to trigger filler loop in select_documents
    sel_d = select_documents(questions, pool, distractors_per_question=2, max_docs=3)
    assert len(sel_d) == 3


def test_hotpotqa_resolve_zip_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    non_existent = tmp_path / "non_existent.zip"
    monkeypatch.setattr(
        "deepeval_eval.hotpotqa_dataset.DEFAULT_DOWNLOADS_DIR", tmp_path / "downloads"
    )
    with pytest.raises(FileNotFoundError):
        resolve_zip(non_existent, "fallback.zip")
