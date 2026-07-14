from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any
from deepeval_eval.agentic_rag import AgenticRetriever
if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepeval_eval.caipe import CaipeRagClient, extract_contexts_and_sources
from deepeval_eval.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_RESULTS_DIR,
    ensure_dirs,
    load_dotenv_loose,
    resolve_litellm_settings,
)
from deepeval_eval.hotpotqa_dataset import (
    INGESTOR_NAME,
    INGESTOR_TYPE,
    load_document_pool,
    load_questions,
    resolve_zip,
    select_documents,
    select_questions,
    to_caipe_payload,
    write_corpus,
    write_questions,
)
from deepeval_eval.io_utils import load_eval_questions
from deepeval_eval.llm import DeepEvalJudge, OpenAICompatibleClient, make_short_answer_prompt
from deepeval_eval.metrics import answer_scores, build_metrics, doc_id_scores


def run_ingest(args: argparse.Namespace) -> None:
    ensure_dirs(args.data_dir, args.cache_dir, args.results_dir)

    questions_zip = resolve_zip(args.questions_zip, 'hotpotqa_full_questions.jsonl.zip')
    documents_zip = resolve_zip(args.documents_zip, 'hotpotqa_full_document_pool.jsonl.zip')

    questions = load_questions(questions_zip)
    selected = select_questions(questions, args.limit, args.questions_per_category, args.categories)
    print(f'Selected {len(selected)} questions')

    pool = load_document_pool(documents_zip)
    # Gold paragraphs are loaded first; distractors make retrieval less trivial
    # while keeping the local sample small enough for a laptop run.
    docs = select_documents(selected, pool, args.distractors_per_question, args.max_docs)
    docs_by_id = {doc['document_id']: doc for doc in docs}
    covered = [q for q in selected if set(q['expected_doc_ids']) <= set(docs_by_id.keys())]
    if covered:
        selected = covered

    print(f'Selected {len(docs)} docs')
    print(f'Questions fully covered by ingested docs: {len(covered)}')

    if not args.skip_ingest:
        client = CaipeRagClient(args.rag_url, args.auth_token)
        if args.reset:
            print(f'Resetting datasource {args.datasource_id}')
            client.reset_datasource(args.datasource_id)

        ingestor_id, max_docs_per_batch = client.register_ingestor(
            INGESTOR_TYPE,
            INGESTOR_NAME,
            'HotpotQA ingestion for CAIPE DeepEval',
        )
        batch_size = min(args.batch_size, max_docs_per_batch)
        payloads = [to_caipe_payload(doc, args.datasource_id, ingestor_id) for doc in docs]

        client.upsert_datasource(
            args.datasource_id,
            args.datasource_name,
            ingestor_id,
            'HotpotQA sample for CAIPE DeepEval evaluation',
            INGESTOR_TYPE,
        )
        job_id = client.open_job(args.datasource_id, len(payloads), 'HotpotQA DeepEval ingestion')
        print(f'Ingestion job opened: {job_id}')

        for start in range(0, len(payloads), batch_size):
            batch = payloads[start:start + batch_size]
            client.ingest_batch(batch, ingestor_id, args.datasource_id, job_id)
            print(f'  ingested {start + len(batch)}/{len(payloads)} documents')

        client.close_job(job_id, 'HotpotQA DeepEval ingestion complete')
        print('Ingestion job completed')

    write_corpus(
        docs,
        args.data_dir / 'hotpotqa_deepeval_corpus.jsonl',
        args.data_dir / 'hotpotqa_deepeval_corpus.csv',
    )
    write_questions(
        selected,
        docs_by_id,
        args.data_dir / 'hotpotqa_deepeval_questions.jsonl',
        args.data_dir / 'hotpotqa_deepeval_questions.csv',
    )
    print(f'Wrote data files to {args.data_dir}')


def run_eval(args: argparse.Namespace) -> None:
    ensure_dirs(args.results_dir)
    load_dotenv_loose(args.env_file)
    base_url, api_key, model = resolve_litellm_settings(
        args.env_file,
        args.llm_base_url,
        args.llm_api_key,
        args.llm_model,
    )

    llm_client = OpenAICompatibleClient(model=model, api_key=api_key, base_url=base_url)
    judge = DeepEvalJudge('cisco-litellm', model, llm_client).model
    metrics = build_metrics(judge)
    rag_client = CaipeRagClient(args.rag_url, args.auth_token)

    from deepeval.test_case import LLMTestCase

    rows = load_eval_questions(args.questions_file, args.max_items, getattr(args, 'limit_per_category', None))
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        question = row['user_input']
        reference = row.get('reference') or ''
        print(f'Evaluating {idx}/{len(rows)}: {question[:90]}')

        # Reset tokens tracking on the judge client before each question evaluation
        llm_client.reset_tokens()

        agentic_result = None

        if getattr(args, 'agentic', False):
            if not hasattr(args, '_agentic_retriever'):
                args._agentic_retriever = AgenticRetriever(
                    supervisor_url=getattr(args, 'supervisor_url', 'http://localhost:8000'),
                    timeout=200.0,
                    logdir=str(args.results_dir / 'logs'),
                )
            agentic_result = args._agentic_retriever.retrieve(question, k=args.top_k)
            answer = agentic_result.answer
            trimmed_contexts = [c[:args.max_context_chars] for c in agentic_result.contexts]
            sources = []
            for c_idx in range(len(agentic_result.contexts)):
                doc_id = None
                if c_idx < len(args._agentic_retriever.documents_metadata):
                    doc_id = args._agentic_retriever.documents_metadata[c_idx].get("doc_id")
                if not doc_id:
                    doc_id = c_idx
                sources.append({
                    "document_id": doc_id,
                })
        else:
            retrieved_raw = rag_client.query(question, args.datasource_id, args.top_k)
            contexts, sources = extract_contexts_and_sources(retrieved_raw)
            trimmed_contexts = [c[:args.max_context_chars] for c in contexts]
            answer = str(llm_client.generate(make_short_answer_prompt(question, trimmed_contexts)))

        # retrieved_raw = rag_client.query(question, args.datasource_id, args.top_k)
        # contexts, sources = extract_contexts_and_sources(retrieved_raw)
        # trimmed_contexts = [text[:args.max_context_chars] for text in contexts]

        # answer = str(llm_client.generate(make_short_answer_prompt(question, trimmed_contexts)))
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference,
            retrieval_context=trimmed_contexts,
            context=row.get('context') or [],
        )

        metric_results: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            try:
                metric.measure(test_case)
                metric_results[metric.__class__.__name__] = {
                    'score': metric.score,
                    'success': metric.success,
                    'reason': metric.reason,
                }
            except Exception as exc:
                metric_results[metric.__class__.__name__] = {
                    'score': None,
                    'success': False,
                    'reason': f'metric failed: {exc}',
                }

        doc_recall, doc_precision = doc_id_scores(sources, list(row.get('expected_doc_ids') or []))
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
            'input_tokens': agentic_result.input_tokens if agentic_result else 0,
            'output_tokens': agentic_result.output_tokens if agentic_result else 0,
            'total_tokens': agentic_result.total_tokens if agentic_result else 0,
            'evaluator_input_tokens': llm_client.input_tokens,
            'evaluator_output_tokens': llm_client.output_tokens,
            'evaluator_total_tokens': llm_client.total_tokens,
            'latency_ms': agentic_result.latency_ms if agentic_result else 0,
        })

    write_results(args.results_dir, results)


def write_results(results_dir: Path, results: list[dict[str, Any]]) -> None:
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    json_path = results_dir / f'hotpotqa_deepeval_results_{timestamp}.json'
    csv_path = results_dir / f'hotpotqa_deepeval_results_{timestamp}.csv'

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'question_id',
            'category',
            'doc_id_recall',
            'doc_id_precision',
            'answer_exact_match',
            'answer_contains_reference',
            'answer_relevancy',
            'faithfulness',
            'contextual_relevancy',
            'contextual_precision',
            'contextual_recall',
        ])
        for row in results:
            metrics = row['metrics']
            writer.writerow([
                row['question_id'],
                row['category'],
                row['doc_id_recall'],
                row['doc_id_precision'],
                row['answer_exact_match'],
                row['answer_contains_reference'],
                metrics.get('AnswerRelevancyMetric', {}).get('score'),
                metrics.get('FaithfulnessMetric', {}).get('score'),
                metrics.get('ContextualRelevancyMetric', {}).get('score'),
                metrics.get('ContextualPrecisionMetric', {}).get('score'),
                metrics.get('ContextualRecallMetric', {}).get('score'),
            ])

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
    eval_parser.add_argument('--limit-per-category', type=int, default=None)
    eval_parser.add_argument('--top-k', type=int, default=5)
    eval_parser.add_argument('--max-context-chars', type=int, default=12000)
    eval_parser.add_argument('--llm-base-url', default=None)
    eval_parser.add_argument('--llm-api-key', default=None)
    eval_parser.add_argument('--llm-model', default=None)
    eval_parser.add_argument('--agentic', action='store_true',
                         help='Route queries through caipe-supervisor A2A endpoint')
    eval_parser.add_argument('--supervisor-url', default='http://localhost:8000',
                            help='CAIPE supervisor URL for agentic eval')
    eval_parser.set_defaults(func=run_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(args.env_file)
    args.func(args)


if __name__ == '__main__':
    main()
