from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import string
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

WORK_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = WORK_DIR / 'data'
DEFAULT_CACHE_DIR = WORK_DIR / 'cache'
DEFAULT_RESULTS_DIR = WORK_DIR / 'results'
DEFAULT_ENV_FILE = Path('C:/Users/liana/ai-platform-engineering/.env')
DEFAULT_DOWNLOADS_DIR = Path('C:/Users/liana/Downloads')
COMMON_SCRIPT = Path(__file__).resolve().with_name('enterprise_deepeval.py')
INGESTOR_TYPE = 'hotpotqa_deepeval'
INGESTOR_NAME = 'hotpotqa-deepeval'


def load_common():
    spec = importlib.util.spec_from_file_location('enterprise_deepeval_common', COMMON_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load common DeepEval helpers from {COMMON_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_common()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def check_response(resp: requests.Response) -> requests.Response:
    if not resp.ok:
        raise RuntimeError(f'{resp.request.method} {resp.request.url} -> HTTP {resp.status_code}\n{resp.text}')
    return resp


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


class HotpotRagClient:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'

    def reset_datasource(self, datasource_id: str) -> None:
        resp = self.session.delete(f'{self.base_url}/v1/datasource', params={'datasource_id': datasource_id}, timeout=60)
        if resp.status_code not in (200, 204, 404):
            check_response(resp)

    def register_ingestor(self) -> tuple[str, int]:
        resp = check_response(self.session.post(
            f'{self.base_url}/v1/ingestor/heartbeat',
            json={'ingestor_type': INGESTOR_TYPE, 'ingestor_name': INGESTOR_NAME, 'description': 'HotpotQA ingestion for CAIPE DeepEval'},
            timeout=60,
        ))
        data = resp.json()
        return data['ingestor_id'], int(data['max_documents_per_ingest'])

    def upsert_datasource(self, datasource_id: str, name: str, ingestor_id: str) -> None:
        check_response(self.session.post(
            f'{self.base_url}/v1/datasource',
            json={'datasource_id': datasource_id, 'name': name, 'ingestor_id': ingestor_id, 'description': 'HotpotQA sample for DeepEval', 'source_type': INGESTOR_TYPE, 'last_updated': int(time.time())},
            timeout=60,
        ))

    def open_job(self, datasource_id: str, total: int) -> str:
        resp = check_response(self.session.post(
            f'{self.base_url}/v1/job',
            params={'datasource_id': datasource_id, 'job_status': 'in_progress', 'message': 'HotpotQA DeepEval ingestion', 'total': total},
            timeout=60,
        ))
        return resp.json()['job_id']

    def close_job(self, job_id: str) -> None:
        check_response(self.session.patch(
            f'{self.base_url}/v1/job/{job_id}',
            params={'job_status': 'completed', 'message': 'HotpotQA DeepEval ingestion complete'},
            timeout=60,
        ))

    def ingest_batch(self, documents: list[dict[str, Any]], ingestor_id: str, datasource_id: str, job_id: str) -> None:
        check_response(self.session.post(
            f'{self.base_url}/v1/ingest',
            json={'documents': documents, 'ingestor_id': ingestor_id, 'datasource_id': datasource_id, 'job_id': job_id},
            timeout=300,
        ))
        for endpoint in ('increment-document-count', 'increment-progress'):
            resp = self.session.post(f'{self.base_url}/v1/job/{job_id}/{endpoint}', params={'increment': len(documents)}, timeout=60)
            if resp.status_code >= 500:
                check_response(resp)



def to_payload(doc: dict[str, str], datasource_id: str, ingestor_id: str) -> dict[str, Any]:
    return {
        'page_content': f'{doc.get(chr(116)+chr(105)+chr(116)+chr(108)+chr(101), chr(39)+chr(39))}\n\n{doc.get(chr(116)+chr(101)+chr(120)+chr(116), chr(39)+chr(39))}',
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


def run_ingest(args: argparse.Namespace) -> None:
    ensure_dirs(args.data_dir, args.cache_dir, args.results_dir)
    questions_zip = resolve_zip(args.questions_zip, 'hotpotqa_full_questions.jsonl.zip')
    documents_zip = resolve_zip(args.documents_zip, 'hotpotqa_full_document_pool.jsonl.zip')
    questions = load_questions(questions_zip)
    selected = select_questions(questions, args.limit, args.questions_per_category, args.categories)
    print(f'Selected {len(selected)} questions')
    pool = load_document_pool(documents_zip)
    docs = select_documents(selected, pool, args.distractors_per_question, args.max_docs)
    docs_by_id = {doc['document_id']: doc for doc in docs}
    covered = [q for q in selected if set(q['expected_doc_ids']) <= set(docs_by_id.keys())]
    if covered:
        selected = covered
    print(f'Selected {len(docs)} docs')
    print(f'Questions fully covered by ingested docs: {len(covered)}')
    if not args.skip_ingest:
        client = HotpotRagClient(args.rag_url, args.auth_token)
        if args.reset:
            print(f'Resetting datasource {args.datasource_id}')
            client.reset_datasource(args.datasource_id)
        ingestor_id, max_docs_per_batch = client.register_ingestor()
        batch_size = min(args.batch_size, max_docs_per_batch)
        payloads = [to_payload(doc, args.datasource_id, ingestor_id) for doc in docs]
        client.upsert_datasource(args.datasource_id, args.datasource_name, ingestor_id)
        job_id = client.open_job(args.datasource_id, len(payloads))
        print(f'Ingestion job opened: {job_id}')
        for start in range(0, len(payloads), batch_size):
            batch = payloads[start:start + batch_size]
            client.ingest_batch(batch, ingestor_id, args.datasource_id, job_id)
            print(f'  ingested {start + len(batch)}/{len(payloads)} documents')
        client.close_job(job_id)
        print('Ingestion job completed')
    write_corpus(docs, args.data_dir / 'hotpotqa_deepeval_corpus.jsonl', args.data_dir / 'hotpotqa_deepeval_corpus.csv')
    write_questions(selected, docs_by_id, args.data_dir / 'hotpotqa_deepeval_questions.jsonl', args.data_dir / 'hotpotqa_deepeval_questions.csv')
    print(f'Wrote data files to {args.data_dir}')
def load_eval_questions(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_items and len(rows) >= max_items:
                break
    return rows


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punc = ''.join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r'\b(a|an|the)\b', ' ', no_punc)
    return ' '.join(no_articles.split())


def answer_scores(answer: str, reference: str) -> tuple[float, float]:
    answer_norm = normalize_answer(answer)
    ref_norm = normalize_answer(reference)
    if not ref_norm:
        return 0.0, 0.0
    return (1.0 if answer_norm == ref_norm else 0.0, 1.0 if ref_norm in answer_norm else 0.0)


def make_prompt(question: str, contexts: list[str]) -> str:
    context_block = '\n\n'.join(f'[{idx + 1}] {text}' for idx, text in enumerate(contexts))
    return 'Answer the HotpotQA question using only the context below. Keep the answer short. If the context is not enough, say that the answer is not in the provided context.\n\n' + f'Question:\n{question}\n\nContext:\n{context_block}\n\nAnswer:'


def run_eval(args: argparse.Namespace) -> None:
    ensure_dirs(args.results_dir)
    env_values = common.load_dotenv_loose(args.env_file)
    base_url = args.llm_base_url or env_values.get('OPENAI_ENDPOINT') or common.os.environ.get('OPENAI_ENDPOINT')
    api_key = args.llm_api_key or env_values.get('OPENAI_API_KEY') or common.os.environ.get('OPENAI_API_KEY')
    model = args.llm_model or env_values.get('OPENAI_MODEL_NAME') or common.os.environ.get('OPENAI_MODEL_NAME')
    if not base_url or not api_key or not model:
        raise RuntimeError('Missing Cisco LiteLLM settings. Need OPENAI_ENDPOINT, OPENAI_API_KEY, and OPENAI_MODEL_NAME.')
    llm_client = common.OpenAICompatibleClient(model=model, api_key=api_key, base_url=base_url)
    judge = common.DeepEvalJudge('cisco-litellm', model, llm_client).model
    metrics = common.build_metrics(judge)
    rag_client = common.RagServerClient(args.rag_url, args.auth_token)
    from deepeval.test_case import LLMTestCase

    rows = load_eval_questions(args.questions_file, args.max_items)
    results = []
    for idx, row in enumerate(rows, start=1):
        question = row['user_input']
        reference = row.get('reference') or ''
        print(f'Evaluating {idx}/{len(rows)}: {question[:90]}')
        retrieved_raw = rag_client.query(question, args.datasource_id, args.top_k)
        contexts, sources = common.extract_contexts_and_sources(retrieved_raw)
        trimmed_contexts = [text[:args.max_context_chars] for text in contexts]
        answer = str(llm_client.generate(make_prompt(question, trimmed_contexts)))
        test_case = LLMTestCase(input=question, actual_output=answer, expected_output=reference, retrieval_context=trimmed_contexts, context=row.get('context') or [])
        metric_results = {}
        for metric in metrics:
            try:
                metric.measure(test_case)
                metric_results[metric.__class__.__name__] = {'score': metric.score, 'success': metric.success, 'reason': metric.reason}
            except Exception as exc:
                metric_results[metric.__class__.__name__] = {'score': None, 'success': False, 'reason': f'metric failed: {exc}'}
        doc_recall, doc_precision = common.doc_id_scores(sources, list(row.get('expected_doc_ids') or []))
        exact_match, contains_reference = answer_scores(answer, reference)
        results.append({
            'question_id': row.get('question_id'),
            'category': row.get('category'),
            'question': question,
            'reference': reference,
            'actual_output': answer,
            'retrieved_sources': sources,
            'doc_id_recall': doc_recall,
            'doc_id_precision': doc_precision,
            'answer_exact_match': exact_match,
            'answer_contains_reference': contains_reference,
            'metrics': metric_results,
        })
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    json_path = args.results_dir / f'hotpotqa_deepeval_results_{timestamp}.json'
    csv_path = args.results_dir / f'hotpotqa_deepeval_results_{timestamp}.csv'
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question_id', 'category', 'doc_id_recall', 'doc_id_precision', 'answer_exact_match', 'answer_contains_reference', 'answer_relevancy', 'faithfulness', 'contextual_relevancy', 'contextual_precision', 'contextual_recall'])
        for row in results:
            ms = row['metrics']
            writer.writerow([row['question_id'], row['category'], row['doc_id_recall'], row['doc_id_precision'], row['answer_exact_match'], row['answer_contains_reference'], ms.get('AnswerRelevancyMetric', {}).get('score'), ms.get('FaithfulnessMetric', {}).get('score'), ms.get('ContextualRelevancyMetric', {}).get('score'), ms.get('ContextualPrecisionMetric', {}).get('score'), ms.get('ContextualRecallMetric', {}).get('score')])
    print(f'Wrote results:\n  {json_path}\n  {csv_path}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='HotpotQA DeepEval pipeline for CAIPE')
    parser.add_argument('--rag-url', default='http://localhost:9446')
    parser.add_argument('--auth-token', default=None)
    parser.add_argument('--env-file', type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    subparsers = parser.add_subparsers(dest='command', required=True)
    ingest = subparsers.add_parser('ingest')
    ingest.add_argument('--questions-zip', type=Path, default=DEFAULT_CACHE_DIR / 'hotpotqa_full_questions.jsonl.zip')
    ingest.add_argument('--documents-zip', type=Path, default=DEFAULT_CACHE_DIR / 'hotpotqa_full_document_pool.jsonl.zip')
    ingest.add_argument('--datasource-id', default='hotpotqa_deepeval')
    ingest.add_argument('--datasource-name', default='HotpotQA DeepEval')
    ingest.add_argument('--limit', type=int, default=100)
    ingest.add_argument('--questions-per-category', type=int, default=50)
    ingest.add_argument('--categories', nargs='+', default=None, choices=['bridge', 'comparison'])
    ingest.add_argument('--distractors-per-question', type=int, default=8)
    ingest.add_argument('--max-docs', type=int, default=None)
    ingest.add_argument('--batch-size', type=int, default=50)
    ingest.add_argument('--reset', action='store_true')
    ingest.add_argument('--skip-ingest', action='store_true')
    ingest.set_defaults(func=run_ingest)
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--datasource-id', default='hotpotqa_deepeval')
    eval_parser.add_argument('--questions-file', type=Path, default=DEFAULT_DATA_DIR / 'hotpotqa_deepeval_questions.jsonl')
    eval_parser.add_argument('--max-items', type=int, default=10)
    eval_parser.add_argument('--top-k', type=int, default=5)
    eval_parser.add_argument('--max-context-chars', type=int, default=12000)
    eval_parser.add_argument('--llm-base-url', default=None)
    eval_parser.add_argument('--llm-api-key', default=None)
    eval_parser.add_argument('--llm-model', default=None)
    eval_parser.set_defaults(func=run_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    common.load_dotenv_loose(args.env_file)
    args.func(args)


if __name__ == '__main__':
    main()
