from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepeval_eval.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GATE_CONFIG,
    DEFAULT_RESULTS_DIR,
    ensure_dirs,
    load_dotenv_loose,
    resolve_litellm_settings,
)
from deepeval_eval.io_utils import load_eval_questions
from deepeval_eval.llm import DeepEvalJudge, OpenAICompatibleClient, make_generation_prompt, make_short_answer_prompt
from deepeval_eval.metrics import answer_scores, build_metrics, doc_id_scores


DEFAULT_QUESTION_FILES = {
    'enterprise': DEFAULT_DATA_DIR / 'enterprise_deepeval_questions.jsonl',
    'hotpotqa': DEFAULT_DATA_DIR / 'hotpotqa_deepeval_questions.jsonl',
}


def build_gold_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected_doc_ids = list(row.get('expected_doc_ids') or [])
    source_types = list(row.get('source_types') or [])
    source_type = source_types[0] if source_types else row.get('benchmark')
    return [
        {
            'document_id': doc_id,
            'title': None,
            'source_type': source_type,
            'score': 1.0,
            'retrieval_mode': 'ground_truth',
        }
        for doc_id in expected_doc_ids
    ]


def context_from_row(row: dict[str, Any], max_context_chars: int) -> list[str]:
    contexts = row.get('context') or []
    if isinstance(contexts, str):
        contexts = [contexts]
    return [str(text)[:max_context_chars] for text in contexts if str(text).strip()]


def make_answer(
    args: argparse.Namespace,
    llm_client: OpenAICompatibleClient,
    question: str,
    reference: str,
    contexts: list[str],
) -> str:
    if args.answer_mode == 'reference':
        return reference
    if args.benchmark == 'hotpotqa':
        return str(llm_client.generate(make_short_answer_prompt(question, contexts)))
    return str(llm_client.generate(make_generation_prompt(question, contexts)))


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

    from deepeval.test_case import LLMTestCase

    questions_file = args.questions_file or DEFAULT_QUESTION_FILES[args.benchmark]
    rows = load_eval_questions(questions_file, args.max_items)
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        question = row['user_input']
        reference = row.get('reference') or ''
        print(f'Evaluating {idx}/{len(rows)}: {question[:90]}')

        contexts = context_from_row(row, args.max_context_chars)
        answer = make_answer(args, llm_client, question, reference, contexts)
        gold_sources = build_gold_sources(row)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference,
            retrieval_context=contexts,
            context=contexts,
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

        doc_recall, doc_precision = doc_id_scores(gold_sources, list(row.get('expected_doc_ids') or []))
        exact_match, contains_reference = answer_scores(answer, reference)
        results.append({
            'question_id': row.get('question_id'),
            'benchmark': args.benchmark,
            'category': row.get('category'),
            'question': question,
            'reference': reference,
            'actual_output': answer,
            'answer_mode': args.answer_mode,
            'retrieval_mode': 'ground_truth_context',
            'retrieved_sources': gold_sources,
            'ground_truth_context_count': len(contexts),
            'doc_id_recall': doc_recall,
            'doc_id_precision': doc_precision,
            'answer_exact_match': exact_match,
            'answer_contains_reference': contains_reference,
            'metrics': metric_results,
        })

    write_results(args.results_dir, args.benchmark, args.answer_mode, results)

    if getattr(args, 'gate', False):
        from deepeval_eval.gate import run_gate_on_results
        passed = run_gate_on_results(results, args.gate_config, args.results_dir)
        if not passed:
            sys.exit(1)


def write_results(results_dir: Path, benchmark: str, answer_mode: str, results: list[dict[str, Any]]) -> None:
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    stem = f'precomputed_deepeval_{benchmark}_{answer_mode}_{timestamp}'
    json_path = results_dir / f'{stem}.json'
    csv_path = results_dir / f'{stem}.csv'

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'question_id',
            'benchmark',
            'category',
            'answer_mode',
            'ground_truth_context_count',
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
                row['benchmark'],
                row['category'],
                row['answer_mode'],
                row['ground_truth_context_count'],
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
    parser = argparse.ArgumentParser(
        description='DeepEval against benchmark ground-truth contexts and reference answers',
    )
    parser.add_argument('--env-file', type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--benchmark', choices=sorted(DEFAULT_QUESTION_FILES), default='hotpotqa')
    parser.add_argument('--questions-file', type=Path, default=None)
    parser.add_argument('--max-items', type=int, default=20)
    parser.add_argument('--max-context-chars', type=int, default=12000)
    parser.add_argument(
        '--answer-mode',
        choices=['reference', 'generate'],
        default='reference',
        help='reference uses the benchmark answer as actual_output; generate answers from gold context using the LLM',
    )
    parser.add_argument('--llm-base-url', default=None)
    parser.add_argument('--llm-api-key', default=None)
    parser.add_argument('--llm-model', default=None)
    parser.add_argument('--gate', action='store_true',
        help='Apply the quality gate after evaluation and exit non-zero if it fails.')
    parser.add_argument('--gate-config', type=Path, default=DEFAULT_GATE_CONFIG,
        help='Path to the gate threshold config (YAML/JSON).')
    parser.set_defaults(func=run_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(args.env_file)
    args.func(args)


if __name__ == '__main__':
    main()
