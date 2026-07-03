from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import requests
from pydantic import BaseModel

os.environ.setdefault('DEEPEVAL_DISABLE_DOTENV', '1')
os.environ.setdefault('DEEPEVAL_TELEMETRY_OPT_OUT', '1')

WORK_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = WORK_DIR / 'data'
DEFAULT_CACHE_DIR = WORK_DIR / 'cache'
DEFAULT_RESULTS_DIR = WORK_DIR / 'results'
DEFAULT_ENV_FILE = Path(r'C:\Users\liana\ai-platform-engineering\.env')
RELEASE_BASE_URL = 'https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0'
QUESTIONS_JSONL_URL = 'https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/main/questions.jsonl'
INGESTOR_TYPE = 'enterprise_rag_bench_deepeval'
INGESTOR_NAME = 'enterprise-rag-bench-deepeval'
SOURCE_SLICE_COUNTS = {
    'confluence': 2,
    'jira': 2,
    'github': 2,
    'hubspot': 4,
    'fireflies': 3,
    'linear': 8,
    'google_drive': 6,
    'gmail': 25,
    'slack': 58,
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

def load_dotenv_loose(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    raw = path.read_text(encoding='utf-8', errors='ignore')
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip(chr(39)).strip(chr(34))
        values[key] = value
        os.environ.setdefault(key, value)
    return values

def check_response(resp: requests.Response) -> requests.Response:
    if not resp.ok:
        raise RuntimeError(f'{resp.request.method} {resp.request.url} -> HTTP {resp.status_code}\n{resp.text}')
    return resp

def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def parse_doc_filename(name: str) -> tuple[str, str] | None:
    base = name.rsplit('/', 1)[-1]
    if not base.startswith('dsid_') or not base.endswith('.txt'):
        return None
    stem = base[:-4]
    if '__' not in stem:
        return None
    doc_id, slug = stem.split('__', 1)
    return doc_id, slug.replace('-', ' ')

def download_text(url: str, cache_path: Path, timeout: int = 60) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding='utf-8')
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding='utf-8')
    return resp.text

def download_bytes(url: str, cache_path: Path, timeout: int = 180) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return resp.content

def load_questions(cache_dir: Path) -> list[EvalQuestion]:
    ensure_dirs(cache_dir)
    raw = download_text(QUESTIONS_JSONL_URL, cache_dir / 'enterprise_rag_bench_questions.jsonl')
    questions: list[EvalQuestion] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        questions.append(EvalQuestion(
            question_id=str(item.get('question_id') or ''),
            user_input=str(item.get('user_input') or item.get('question') or ''),
            reference=str(item.get('reference') or item.get('gold_answer') or ''),
            category=str(item.get('category') or item.get('question_type') or 'uncategorized'),
            source_types=list(item.get('source_types') or []),
            expected_doc_ids=list(item.get('expected_doc_ids') or []),
            answer_facts=list(item.get('answer_facts') or []),
        ))
    return questions

def select_questions(questions: list[EvalQuestion], source_types: list[str], question_limit: int, questions_per_category: int) -> list[EvalQuestion]:
    wanted_sources = set(source_types)
    candidates = [q for q in questions if not q.source_types or bool(set(q.source_types) & wanted_sources)]
    selected: list[EvalQuestion] = []
    counts: dict[str, int] = {}
    for question in candidates:
        category = question.category or 'uncategorized'
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

def fetch_documents(source_types: list[str], limit_per_source: int, cache_dir: Path, reference_doc_ids: set[str]) -> list[EnterpriseDoc]:
    ensure_dirs(cache_dir)
    all_docs: list[EnterpriseDoc] = []
    seen_hashes: set[str] = set()
    for source_type in source_types:
        reference_docs: list[EnterpriseDoc] = []
        other_docs: list[EnterpriseDoc] = []
        n_slices = SOURCE_SLICE_COUNTS[source_type]
        print(f'Fetching {source_type}: {n_slices} slice(s)')
        for slice_num in range(1, n_slices + 1):
            zip_name = f'{source_type}_slice_{slice_num:04d}.zip'
            zip_bytes = download_bytes(f'{RELEASE_BASE_URL}/{zip_name}', cache_dir / zip_name)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names = [n for n in zf.namelist() if n.endswith('.txt')]
                print(f'  {zip_name}: {len(names)} files')
                for name in names:
                    parsed = parse_doc_filename(name)
                    if not parsed:
                        continue
                    doc_id, fallback_title = parsed
                    raw = zf.read(name).decode('utf-8', errors='replace').strip()
                    if not raw:
                        continue
                    digest = hashlib.md5(raw.encode('utf-8')).hexdigest()
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)
                    first_line = raw.split('\n', 1)[0].strip()
                    doc = EnterpriseDoc(doc_id=doc_id, title=first_line or fallback_title, text=raw, source_type=source_type)
                    if doc_id in reference_doc_ids:
                        reference_docs.append(doc)
                    else:
                        other_docs.append(doc)
        selected = (reference_docs + other_docs)[:limit_per_source]
        print(f'  selected {len(selected)} docs ({len(reference_docs)} reference docs available)')
        all_docs.extend(selected)
    return all_docs

class RagServerClient:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'

    def register_ingestor(self) -> tuple[str, int]:
        resp = check_response(self.session.post(f'{self.base_url}/v1/ingestor/heartbeat', json={'ingestor_type': INGESTOR_TYPE, 'ingestor_name': INGESTOR_NAME, 'description': 'EnterpriseRAG-Bench ingestion for DeepEval'}, timeout=60))
        data = resp.json()
        return data['ingestor_id'], int(data['max_documents_per_ingest'])

    def reset_datasource(self, datasource_id: str) -> None:
        resp = self.session.delete(f'{self.base_url}/v1/datasource', params={'datasource_id': datasource_id}, timeout=60)
        if resp.status_code not in (200, 204, 404):
            check_response(resp)

    def upsert_datasource(self, datasource_id: str, name: str, ingestor_id: str) -> None:
        check_response(self.session.post(f'{self.base_url}/v1/datasource', json={'datasource_id': datasource_id, 'name': name, 'ingestor_id': ingestor_id, 'description': 'EnterpriseRAG-Bench sample for CAIPE DeepEval evaluation', 'source_type': INGESTOR_TYPE, 'last_updated': int(time.time())}, timeout=60))

    def open_job(self, datasource_id: str, total: int) -> str:
        resp = check_response(self.session.post(f'{self.base_url}/v1/job', params={'datasource_id': datasource_id, 'job_status': 'in_progress', 'message': 'EnterpriseRAG-Bench DeepEval ingestion', 'total': total}, timeout=60))
        return resp.json()['job_id']

    def close_job(self, job_id: str) -> None:
        check_response(self.session.patch(f'{self.base_url}/v1/job/{job_id}', params={'job_status': 'completed', 'message': 'EnterpriseRAG-Bench DeepEval ingestion complete'}, timeout=60))

    def ingest_batch(self, documents: list[dict[str, Any]], ingestor_id: str, datasource_id: str, job_id: str) -> None:
        check_response(self.session.post(f'{self.base_url}/v1/ingest', json={'documents': documents, 'ingestor_id': ingestor_id, 'datasource_id': datasource_id, 'job_id': job_id}, timeout=300))
        for endpoint in ('increment-document-count', 'increment-progress'):
            resp = self.session.post(f'{self.base_url}/v1/job/{job_id}/{endpoint}', params={'increment': len(documents)}, timeout=60)
            if resp.status_code >= 500:
                check_response(resp)

    def query(self, question: str, datasource_id: str | None, limit: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {'query': question, 'limit': limit}
        if datasource_id:
            payload['filters'] = {'datasource_id': datasource_id}
        resp = check_response(self.session.post(f'{self.base_url}/v1/query', json=payload, timeout=120))
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get('results') or [])
        return []

def to_caipe_payload(doc: EnterpriseDoc, datasource_id: str, ingestor_id: str) -> dict[str, Any]:
    return {'page_content': doc.text, 'type': 'Document', 'metadata': {'document_id': doc.doc_id, 'datasource_id': datasource_id, 'ingestor_id': ingestor_id, 'title': doc.title, 'description': f'EnterpriseRAG-Bench - {doc.source_type}', 'is_structured_entity': False, 'document_type': 'text', 'document_ingested_at': None, 'fresh_until': None, 'metadata': {'source': 'enterprise_rag_bench', 'source_type': doc.source_type}}}

def write_questions(questions: list[EvalQuestion], docs_by_id: dict[str, EnterpriseDoc], jsonl_path: Path, csv_path: Path) -> None:
    with jsonl_path.open('w', encoding='utf-8') as jf, csv_path.open('w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['question_id', 'user_input', 'reference', 'category', 'source_types', 'expected_doc_ids'])
        for question in questions:
            context = [docs_by_id[doc_id].text for doc_id in question.expected_doc_ids if doc_id in docs_by_id]
            record = {'question_id': question.question_id, 'user_input': question.user_input, 'reference': question.reference, 'category': question.category, 'source_types': question.source_types, 'expected_doc_ids': question.expected_doc_ids, 'answer_facts': question.answer_facts, 'context': context}
            jf.write(json.dumps(record, ensure_ascii=False) + '\n')
            writer.writerow([question.question_id, question.user_input, question.reference, question.category, ';'.join(question.source_types), ';'.join(question.expected_doc_ids)])

def write_corpus(docs: list[EnterpriseDoc], jsonl_path: Path, csv_path: Path) -> None:
    with jsonl_path.open('w', encoding='utf-8') as jf, csv_path.open('w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['document_id', 'source_type', 'title', 'text'])
        for doc in docs:
            record = {'document_id': doc.doc_id, 'source_type': doc.source_type, 'title': doc.title, 'text': doc.text}
            jf.write(json.dumps(record, ensure_ascii=False) + '\n')
            writer.writerow([doc.doc_id, doc.source_type, doc.title, doc.text])

def extract_contexts_and_sources(results: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    for row in results:
        document = row.get('document') if isinstance(row, dict) else None
        if not isinstance(document, dict):
            document = {}
        metadata = document.get('metadata') if isinstance(document.get('metadata'), dict) else {}
        nested = metadata.get('metadata') if isinstance(metadata.get('metadata'), dict) else {}
        text = document.get('page_content') or row.get('page_content') or document.get('content') or row.get('content') or ''
        if not text:
            continue
        contexts.append(text)
        sources.append({'document_id': metadata.get('document_id'), 'title': metadata.get('title'), 'source_type': nested.get('source_type'), 'score': row.get('score')})
    return contexts, sources

def make_generation_prompt(question: str, contexts: list[str]) -> str:
    context_block = '\n\n'.join(f'[{idx + 1}] {text}' for idx, text in enumerate(contexts))
    return 'Answer the question using only the context below. If the context is not enough, say that the answer is not in the provided context.\n\n' + f'Question:\n{question}\n\nContext:\n{context_block}\n\nAnswer:'

class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        if schema is not None:
            prompt = with_json_schema_instruction(prompt, schema)
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0}
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f'{self.base_url}/chat/completions', headers=headers, json=payload)
            response.raise_for_status()
            text = response.json()['choices'][0]['message']['content'] or ''
        if schema is None:
            return text.strip()
        return parse_schema_response(text, schema)

class DeepEvalJudge:
    def __init__(self, provider: str, model: str, client: OpenAICompatibleClient):
        from deepeval.models.base_model import DeepEvalBaseLLM
        class Judge(DeepEvalBaseLLM):
            def __init__(self, provider_name: str, model_name: str, llm_client: OpenAICompatibleClient):
                self.provider_name = provider_name
                self.model_name = model_name
                self.llm_client = llm_client
                super().__init__(model=model_name)
            def load_model(self, *args: Any, **kwargs: Any):
                return self.llm_client
            def get_model_name(self, *args: Any, **kwargs: Any) -> str:
                return f'{self.provider_name}:{self.model_name}'
            def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any):
                return self.model.generate(prompt, schema=schema)
            async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any):
                return await asyncio.to_thread(self.generate, prompt, schema=schema, **kwargs)
        self.model = Judge(provider, model, client)

def with_json_schema_instruction(prompt: str, schema: type[BaseModel]) -> str:
    return f'{prompt}\n\n' + 'Return only valid JSON matching this JSON schema. Do not include markdown fences.\n\n' + f'{json.dumps(schema.model_json_schema(), indent=2)}'

def parse_schema_response(text: str, schema: type[BaseModel]) -> BaseModel:
    cleaned = text.strip()
    fence = chr(96) * 3
    if fence in cleaned:
        parts = [part.strip() for part in cleaned.split(fence) if part.strip()]
        if parts:
            cleaned = parts[-1]
            if cleaned.lower().startswith('json'):
                cleaned = cleaned[4:].strip()
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        match = re.search(r'(\{.*\}|\[.*\])', cleaned, flags=re.DOTALL)
        if not match:
            raise
        return schema.model_validate_json(match.group(1))

def load_eval_questions(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_items and len(rows) >= max_items:
                break
    return rows

def doc_id_scores(retrieved: list[dict[str, Any]], expected_doc_ids: list[str]) -> tuple[float, float]:
    retrieved_ids = {str(item.get('document_id')) for item in retrieved if item.get('document_id') is not None}
    expected = {str(doc_id) for doc_id in expected_doc_ids}
    if not expected:
        return 0.0, 0.0
    hits = retrieved_ids & expected
    recall = len(hits) / len(expected)
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
    return recall, precision

def build_metrics(judge_model: Any) -> list[Any]:
    from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, FaithfulnessMetric
    common = {'threshold': 0.5, 'model': judge_model, 'include_reason': True, 'async_mode': False}
    return [AnswerRelevancyMetric(**common), FaithfulnessMetric(**common), ContextualRelevancyMetric(**common), ContextualPrecisionMetric(**common), ContextualRecallMetric(**common)]

def run_ingest(args: argparse.Namespace) -> None:
    ensure_dirs(args.data_dir, args.cache_dir, args.results_dir)
    questions = load_questions(args.cache_dir)
    selected = select_questions(questions, args.sources, args.num_questions, args.questions_per_category)
    reference_doc_ids = {doc_id for question in selected for doc_id in question.expected_doc_ids}
    print(f'Selected {len(selected)} questions')
    print(f'Reference doc ids to prioritize: {len(reference_doc_ids)}')
    docs = fetch_documents(args.sources, args.limit_per_source, args.cache_dir, reference_doc_ids)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    covered = [q for q in selected if set(q.expected_doc_ids) <= set(docs_by_id.keys())]
    if covered:
        selected = covered
    print(f'Questions fully covered by ingested docs: {len(covered)}')
    if not args.skip_ingest:
        client = RagServerClient(args.rag_url, args.auth_token)
        if args.reset:
            print(f'Resetting datasource {args.datasource_id}')
            client.reset_datasource(args.datasource_id)
        ingestor_id, max_docs = client.register_ingestor()
        batch_size = min(args.batch_size, max_docs)
        payloads = [to_caipe_payload(doc, args.datasource_id, ingestor_id) for doc in docs]
        client.upsert_datasource(args.datasource_id, args.datasource_name, ingestor_id)
        job_id = client.open_job(args.datasource_id, len(payloads))
        print(f'Ingestion job opened: {job_id}')
        for start in range(0, len(payloads), batch_size):
            batch = payloads[start:start + batch_size]
            client.ingest_batch(batch, ingestor_id, args.datasource_id, job_id)
            print(f'  ingested {start + len(batch)}/{len(payloads)} documents')
        client.close_job(job_id)
        print('Ingestion job completed')
    write_corpus(docs, args.data_dir / 'enterprise_deepeval_corpus.jsonl', args.data_dir / 'enterprise_deepeval_corpus.csv')
    write_questions(selected, docs_by_id, args.data_dir / 'enterprise_deepeval_questions.jsonl', args.data_dir / 'enterprise_deepeval_questions.csv')
    print(f'Wrote data files to {args.data_dir}')

def run_eval(args: argparse.Namespace) -> None:
    ensure_dirs(args.results_dir)
    env_values = load_dotenv_loose(args.env_file)
    base_url = args.llm_base_url or env_values.get('OPENAI_ENDPOINT') or os.environ.get('OPENAI_ENDPOINT')
    api_key = args.llm_api_key or env_values.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    model = args.llm_model or env_values.get('OPENAI_MODEL_NAME') or os.environ.get('OPENAI_MODEL_NAME')
    if not base_url or not api_key or not model:
        raise RuntimeError('Missing Cisco LiteLLM settings. Need OPENAI_ENDPOINT, OPENAI_API_KEY, and OPENAI_MODEL_NAME.')
    llm_client = OpenAICompatibleClient(model=model, api_key=api_key, base_url=base_url)
    judge = DeepEvalJudge('cisco-litellm', model, llm_client).model
    metrics = build_metrics(judge)
    rag_client = RagServerClient(args.rag_url, args.auth_token)
    from deepeval.test_case import LLMTestCase
    rows = load_eval_questions(args.questions_file, args.max_items)
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        question = row['user_input']
        print(f'Evaluating {idx}/{len(rows)}: {question[:90]}')
        retrieved_raw = rag_client.query(question, args.datasource_id, args.top_k)
        contexts, sources = extract_contexts_and_sources(retrieved_raw)
        trimmed_contexts = [text[:args.max_context_chars] for text in contexts]
        answer = str(llm_client.generate(make_generation_prompt(question, trimmed_contexts)))
        test_case = LLMTestCase(input=question, actual_output=answer, expected_output=row.get('reference'), retrieval_context=trimmed_contexts, context=row.get('context') or [])
        metric_results: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            try:
                metric.measure(test_case)
                metric_results[metric.__class__.__name__] = {'score': metric.score, 'success': metric.success, 'reason': metric.reason}
            except Exception as exc:
                metric_results[metric.__class__.__name__] = {'score': None, 'success': False, 'reason': f'metric failed: {exc}'}
        doc_recall, doc_precision = doc_id_scores(sources, list(row.get('expected_doc_ids') or []))
        results.append({'question_id': row.get('question_id'), 'question': question, 'reference': row.get('reference'), 'actual_output': answer, 'retrieved_sources': sources, 'doc_id_recall': doc_recall, 'doc_id_precision': doc_precision, 'metrics': metric_results})
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    json_path = args.results_dir / f'enterprise_deepeval_results_{timestamp}.json'
    csv_path = args.results_dir / f'enterprise_deepeval_results_{timestamp}.csv'
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question_id', 'doc_id_recall', 'doc_id_precision', 'answer_relevancy', 'faithfulness', 'contextual_relevancy', 'contextual_precision', 'contextual_recall'])
        for row in results:
            ms = row['metrics']
            writer.writerow([row['question_id'], row['doc_id_recall'], row['doc_id_precision'], ms.get('AnswerRelevancyMetric', {}).get('score'), ms.get('FaithfulnessMetric', {}).get('score'), ms.get('ContextualRelevancyMetric', {}).get('score'), ms.get('ContextualPrecisionMetric', {}).get('score'), ms.get('ContextualRecallMetric', {}).get('score')])
    print(f'Wrote results:\n  {json_path}\n  {csv_path}')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='EnterpriseRAG-Bench DeepEval pipeline for CAIPE')
    parser.add_argument('--rag-url', default='http://localhost:9446')
    parser.add_argument('--auth-token', default=None)
    parser.add_argument('--env-file', type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    subparsers = parser.add_subparsers(dest='command', required=True)
    ingest = subparsers.add_parser('ingest')
    ingest.add_argument('--sources', nargs='+', default=['confluence', 'jira'], choices=sorted(SOURCE_SLICE_COUNTS))
    ingest.add_argument('--datasource-id', default='enterprise_rag_bench_deepeval')
    ingest.add_argument('--datasource-name', default='EnterpriseRAG-Bench DeepEval')
    ingest.add_argument('--limit-per-source', type=int, default=1000)
    ingest.add_argument('--num-questions', type=int, default=10)
    ingest.add_argument('--questions-per-category', type=int, default=3)
    ingest.add_argument('--batch-size', type=int, default=100)
    ingest.add_argument('--reset', action='store_true')
    ingest.add_argument('--skip-ingest', action='store_true')
    ingest.set_defaults(func=run_ingest)
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--datasource-id', default='enterprise_rag_bench_deepeval')
    eval_parser.add_argument('--questions-file', type=Path, default=DEFAULT_DATA_DIR / 'enterprise_deepeval_questions.jsonl')
    eval_parser.add_argument('--max-items', type=int, default=3)
    eval_parser.add_argument('--top-k', type=int, default=5)
    eval_parser.add_argument('--max-context-chars', type=int, default=16000)
    eval_parser.add_argument('--llm-base-url', default=None)
    eval_parser.add_argument('--llm-api-key', default=None)
    eval_parser.add_argument('--llm-model', default=None)
    eval_parser.set_defaults(func=run_eval)
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(args.env_file)
    args.func(args)

if __name__ == '__main__':
    main()
