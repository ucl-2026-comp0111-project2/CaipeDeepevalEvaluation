from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

from deepeval_eval.config import DEFAULT_DOWNLOADS_DIR

INGESTOR_TYPE = 'hotpotqa_deepeval'
INGESTOR_NAME = 'hotpotqa-deepeval'


def resolve_zip(path: Path, fallback_name: str) -> Path:
    if path.exists():
        return path
    fallback = DEFAULT_DOWNLOADS_DIR / fallback_name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f'Could not find {path} or {fallback}')


def read_jsonl_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.endswith('.jsonl')]
        if not names:
            raise RuntimeError(f'No jsonl file found inside {path}')
        with zf.open(names[0]) as f:
            return [json.loads(line.decode('utf-8')) for line in f if line.strip()]


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    for item in read_jsonl_zip(path):
        questions.append({
            'question_id': str(item.get('question_id') or ''),
            'user_input': str(item.get('user_input') or item.get('input') or ''),
            'reference': str(item.get('reference') or item.get('expected_output') or ''),
            'category': str(item.get('category') or 'uncategorized'),
            'level': str(item.get('level') or ''),
            'expected_doc_ids': unique([str(v) for v in item.get('expected_doc_ids') or []]),
            'source_types': list(item.get('source_types') or ['hotpotqa']),
            'supporting_facts': list(item.get('supporting_facts') or []),
        })
    return questions


def load_document_pool(path: Path) -> dict[str, dict[str, str]]:
    docs = {}
    for item in read_jsonl_zip(path):
        doc_id = str(item.get('document_id') or '')
        if not doc_id:
            continue
        docs[doc_id] = {
            'document_id': doc_id,
            'title': str(item.get('title') or doc_id),
            'text': str(item.get('content') or item.get('text') or ''),
        }
    return docs


def select_questions(questions: list[dict[str, Any]], limit: int, per_category: int, categories: list[str] | None) -> list[dict[str, Any]]:
    wanted = set(categories or [])
    candidates = [q for q in questions if not wanted or q['category'] in wanted]
    selected = []
    counts: dict[str, int] = {}
    for question in candidates:
        category = question['category']
        if per_category and counts.get(category, 0) >= per_category:
            continue
        selected.append(question)
        counts[category] = counts.get(category, 0) + 1
        if len(selected) >= limit:
            return selected
    seen = {q['question_id'] for q in selected}
    for question in candidates:
        if question['question_id'] in seen:
            continue
        selected.append(question)
        if len(selected) >= limit:
            break
    return selected


# Include supporting paragraphs first, then add distractors so retrieval has
# both relevant and irrelevant candidates to rank.
def select_documents(questions: list[dict[str, Any]], pool: dict[str, dict[str, str]], distractors_per_question: int, max_docs: int | None) -> list[dict[str, str]]:
    reference_ids = unique([doc_id for q in questions for doc_id in q['expected_doc_ids']])
    docs = [pool[doc_id] for doc_id in reference_ids if doc_id in pool]
    target = max_docs or (len(docs) + len(questions) * distractors_per_question)
    target = max(target, len(docs))
    selected_ids = {doc['document_id'] for doc in docs}
    for doc in pool.values():
        if len(docs) >= target:
            break
        if doc['document_id'] in selected_ids:
            continue
        selected_ids.add(doc['document_id'])
        docs.append(doc)
    return docs


def to_caipe_payload(doc: dict[str, str], datasource_id: str, ingestor_id: str) -> dict[str, Any]:
    title = doc.get('title', '')
    text = doc.get('text', '')
    return {
        'page_content': f'{title}\n\n{text}',
        'type': 'Document',
        'metadata': {
            'document_id': doc['document_id'],
            'datasource_id': datasource_id,
            'ingestor_id': ingestor_id,
            'title': doc['title'],
            'description': 'HotpotQA Wikipedia paragraph',
            'is_structured_entity': False,
            'document_type': 'text',
            'document_ingested_at': None,
            'fresh_until': None,
            'metadata': {'source': 'hotpotqa', 'source_type': 'hotpotqa'},
        },
    }


def write_corpus(docs: list[dict[str, str]], jsonl_path: Path, csv_path: Path) -> None:
    with jsonl_path.open('w', encoding='utf-8') as jf, csv_path.open('w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['document_id', 'title', 'text'])
        for doc in docs:
            jf.write(json.dumps(doc, ensure_ascii=False) + '\n')
            writer.writerow([doc['document_id'], doc['title'], doc['text']])


def write_questions(questions: list[dict[str, Any]], docs_by_id: dict[str, dict[str, str]], jsonl_path: Path, csv_path: Path) -> None:
    with jsonl_path.open('w', encoding='utf-8') as jf, csv_path.open('w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['question_id', 'user_input', 'reference', 'category', 'level', 'expected_doc_ids'])
        for q in questions:
            context = [docs_by_id[doc_id]['text'] for doc_id in q['expected_doc_ids'] if doc_id in docs_by_id]
            record = dict(q)
            record['context'] = context
            jf.write(json.dumps(record, ensure_ascii=False) + '\n')
            writer.writerow([q['question_id'], q['user_input'], q['reference'], q['category'], q['level'], ';'.join(q['expected_doc_ids'])])
