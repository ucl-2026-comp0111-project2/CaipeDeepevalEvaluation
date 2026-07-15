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

    start_eval_time = time.time()
    for idx, row in enumerate(rows, start=1):
        question = row['user_input']
        reference = row.get('reference') or ''
        print(f'Evaluating {idx}/{len(rows)}: {question[:90]}')

        # Reset tokens tracking on the judge client before each question evaluation
        llm_client.reset_tokens()

        start_time = time.time()
        contexts = context_from_row(row, args.max_context_chars)
        answer = make_answer(args, llm_client, question, reference, contexts)
        gold_sources = build_gold_sources(row)
        latency_sec = time.time() - start_time

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
            'user_input': question,
            'reference': reference,
            'actual_output': answer,
            'answer_mode': args.answer_mode,
            'retrieval_mode': 'ground_truth_context',
            'retrieved_contexts': contexts,
            'retrieved_doc_ids': [str(s.get('document_id')) for s in gold_sources if s.get('document_id') is not None],
            'expected_doc_ids': row.get('expected_doc_ids') or [],
            'ground_truth_context_count': len(contexts),
            'doc_id_recall': doc_recall,
            'doc_id_precision': doc_precision,
            'answer_exact_match': exact_match,
            'answer_contains_reference': contains_reference,
            'metrics': metric_results,
            'evaluator_input_tokens': llm_client.input_tokens,
            'evaluator_output_tokens': llm_client.output_tokens,
            'evaluator_total_tokens': llm_client.total_tokens,
            'latency': latency_sec,
            'latency_ms': latency_sec * 1000.0,
            'total_tokens': 0,
            'log_file': " ",
        })

    eval_time = time.time() - start_eval_time
    config_args = {}
    for k, v in vars(args).items():
        if v is None or k in ('llm_api_key', 'auth_token') or callable(v) or k.startswith('_'):
            continue
        if isinstance(v, Path):
            config_args[k] = str(v)
        elif isinstance(v, (str, int, float, bool, list, dict)):
            config_args[k] = v
        else:
            config_args[k] = str(v)
    write_results(args.results_dir, args.benchmark, args.answer_mode, results, eval_time, config_args)


def write_results(
    results_dir: Path,
    benchmark: str,
    answer_mode: str,
    results: list[dict[str, Any]],
    evaluation_time: float,
    config_args: dict[str, Any],
) -> None:
    import math
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    stem = f'precomputed_deepeval_{benchmark}_{answer_mode}_{timestamp}'
    json_path = results_dir / f'{stem}.json'
    csv_path = results_dir / f'{stem}.csv'
    summary_json_path = results_dir / f'{stem}_summary.json'

    # Compute metric statistics
    n = len(results)
    latencies = [r.get('latency', 0.0) for r in results]
    latencies_sorted = sorted(latencies)

    p50_latency = 0.0
    p95_latency = 0.0
    if latencies_sorted:
        p50_latency = latencies_sorted[len(latencies_sorted) // 2]
        p95_index = max(0, min(len(latencies_sorted) - 1, int(math.ceil((len(latencies_sorted) * 95) / 100)) - 1))
        p95_latency = latencies_sorted[p95_index]

    def get_metric_avg(metric_name):
        vals = [
            r.get('metrics', {}).get(metric_name, {}).get('score')
            for r in results
            if r.get('metrics', {}).get(metric_name, {}).get('score') is not None
        ]
        return sum(vals) / len(vals) if vals else 0.0

    avg_answer_relevancy = get_metric_avg('AnswerRelevancyMetric')
    avg_faithfulness = get_metric_avg('FaithfulnessMetric')
    avg_contextual_relevancy = get_metric_avg('ContextualRelevancyMetric')
    avg_contextual_precision = get_metric_avg('ContextualPrecisionMetric')
    avg_contextual_recall = get_metric_avg('ContextualRecallMetric')

    avg_exact_match = sum(r.get('answer_exact_match', 0.0) for r in results) / n if n else 0.0
    avg_contains_ref = sum(r.get('answer_contains_reference', 0.0) for r in results) / n if n else 0.0
    avg_recall = sum(r.get('doc_id_recall') or 0.0 for r in results) / n if n else 0.0
    avg_precision = sum(r.get('doc_id_precision') or 0.0 for r in results) / n if n else 0.0

    # Calculate failure causes
    failure_counts = {}
    for r in results:
        fc = 'none'
        faith = r.get('metrics', {}).get('FaithfulnessMetric', {}).get('score')
        ctx_recall = r.get('metrics', {}).get('ContextualRecallMetric', {}).get('score')
        ans_rel = r.get('metrics', {}).get('AnswerRelevancyMetric', {}).get('score')

        if faith is not None and faith < 0.5:
            fc = 'hallucination'
        elif ctx_recall is not None and ctx_recall < 0.5:
            fc = 'poor_retrieval'
        elif ans_rel is not None and ans_rel < 0.5:
            fc = 'incorrect_generation'

        r['failure_cause'] = fc
        failure_counts[fc] = failure_counts.get(fc, 0) + 1

    # Evaluator metrics summary
    evaluator_prompt_tokens = sum(r.get('evaluator_input_tokens', 0) for r in results)
    evaluator_completion_tokens = sum(r.get('evaluator_output_tokens', 0) for r in results)
    evaluator_total_tokens = evaluator_prompt_tokens + evaluator_completion_tokens

    # Log summary to console
    print("\n--- RUN CONFIGURATION ---")
    print(f"datasource: {benchmark} (precomputed)")
    for k, v in config_args.items():
        print(f"{k}: {v}")

    print("\n--- OPERATIONAL BEHAVIOR ---")
    print("RAG Pipeline:")
    print(f"  P50 Latency: {p50_latency:.2f}s")
    print(f"  P95 Latency: {p95_latency:.2f}s")
    print(f"  Total Tokens: 0")
    print("\nDeepEval Evaluator:")
    print(f"  Evaluation Time: {evaluation_time:.2f}s")
    print(f"  Prompt Tokens: {evaluator_prompt_tokens}")
    print(f"  Completion Tokens: {evaluator_completion_tokens}")
    print(f"  Total Evaluator Tokens: {evaluator_total_tokens}")

    print("\n--- QUALITY METRICS AVERAGE ---")
    print(f"Average answer_relevancy: {avg_answer_relevancy:.2f}")
    print(f"Average faithfulness: {avg_faithfulness:.2f}")
    print(f"Average contextual_relevancy: {avg_contextual_relevancy:.2f}")
    print(f"Average contextual_precision: {avg_contextual_precision:.2f}")
    print(f"Average contextual_recall: {avg_contextual_recall:.2f}")
    print(f"Average retrieval_recall: {avg_recall:.2f}")
    print(f"Average retrieval_precision: {avg_precision:.2f}")
    print(f"Average answer_exact_match: {avg_exact_match:.2f}")
    print(f"Average answer_contains_reference: {avg_contains_ref:.2f}")

    print("\n--- FAILURE CAUSE ANALYSIS ---")
    for cause, count in failure_counts.items():
        print(f"{cause:<20} {count}")

    # Write detailed JSON results
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    # Ensure config_args is JSON serializable
    serializable_config = {}
    for k, v in config_args.items():
        if k.startswith('_') or k in ('llm_api_key', 'auth_token'):
            continue
        try:
            json.dumps(v)
            serializable_config[k] = v
        except (TypeError, OverflowError):
            serializable_config[k] = str(v)

    # Write companion summary JSON
    summary_data = {
        "experiment_name": stem,
        "datasource": f"{benchmark}_precomputed",
        "config_args": serializable_config,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "total_tokens": 0,
        "metrics": {
            "answer_relevancy": avg_answer_relevancy,
            "faithfulness": avg_faithfulness,
            "contextual_relevancy": avg_contextual_relevancy,
            "contextual_precision": avg_contextual_precision,
            "contextual_recall": avg_contextual_recall,
            "retrieval_recall": avg_recall,
            "retrieval_precision": avg_precision,
            "answer_exact_match": avg_exact_match,
            "answer_contains_reference": avg_contains_ref
        },
        "average_retrieval_recall": avg_recall,
        "average_retrieval_precision": avg_precision,
        "deepeval_evaluator_usage": {
            "evaluation_time_seconds": evaluation_time,
            "prompt_tokens": evaluator_prompt_tokens,
            "completion_tokens": evaluator_completion_tokens,
            "total_tokens": evaluator_total_tokens
        }
    }
    summary_json_path.write_text(json.dumps(summary_data, indent=4, ensure_ascii=False), encoding='utf-8')

    # Write CSV results
    csv_columns = [
        'question_id',
        'benchmark',
        'category',
        'answer_mode',
        'question',
        'user_input',
        'reference',
        'expected_doc_ids',
        'response',
        'retrieved_contexts',
        'retrieved_doc_ids',
        'latency',
        'total_tokens',
        'log_file',
        'answer_exact_match',
        'answer_contains_reference',
        'answer_relevancy',
        'faithfulness',
        'contextual_relevancy',
        'contextual_precision',
        'contextual_recall',
        'answer_relevancy_reason',
        'faithfulness_reason',
        'contextual_relevancy_reason',
        'contextual_precision_reason',
        'contextual_recall_reason',
        'failure_cause',
        'retrieval_recall',
        'retrieval_precision',
        'evaluator_evaluation_time_seconds',
        'evaluator_prompt_tokens',
        'evaluator_completion_tokens',
        'evaluator_total_tokens'
    ]

    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

        for r in results:
            metrics = r.get('metrics', {})
            retrieved_contexts_str = json.dumps(r.get('retrieved_contexts') or [])
            expected_doc_ids_str = ';'.join(r.get('expected_doc_ids') or [])
            retrieved_doc_ids_str = ';'.join(r.get('retrieved_doc_ids') or [])

            writer.writerow([
                r.get('question_id'),
                r.get('benchmark'),
                r.get('category'),
                r.get('answer_mode'),
                r.get('question'),
                r.get('user_input'),
                r.get('reference'),
                expected_doc_ids_str,
                r.get('actual_output'),
                retrieved_contexts_str,
                retrieved_doc_ids_str,
                r.get('latency'),
                r.get('total_tokens'),
                r.get('log_file'),
                r.get('answer_exact_match'),
                r.get('answer_contains_reference'),
                metrics.get('AnswerRelevancyMetric', {}).get('score'),
                metrics.get('FaithfulnessMetric', {}).get('score'),
                metrics.get('ContextualRelevancyMetric', {}).get('score'),
                metrics.get('ContextualPrecisionMetric', {}).get('score'),
                metrics.get('ContextualRecallMetric', {}).get('score'),
                metrics.get('AnswerRelevancyMetric', {}).get('reason'),
                metrics.get('FaithfulnessMetric', {}).get('reason'),
                metrics.get('ContextualRelevancyMetric', {}).get('reason'),
                metrics.get('ContextualPrecisionMetric', {}).get('reason'),
                metrics.get('ContextualRecallMetric', {}).get('reason'),
                r.get('failure_cause'),
                r.get('doc_id_recall'),
                r.get('doc_id_precision'),
                evaluation_time,
                r.get('evaluator_input_tokens'),
                r.get('evaluator_output_tokens'),
                r.get('evaluator_total_tokens')
            ])

        # Write AVERAGE_METRICS row
        summary_row = dict.fromkeys(csv_columns, "")
        summary_row['question'] = "AVERAGE_METRICS"
        summary_row['latency'] = sum(latencies) / n if n else 0.0
        summary_row['total_tokens'] = 0.0
        summary_row['answer_exact_match'] = avg_exact_match
        summary_row['answer_contains_reference'] = avg_contains_ref
        summary_row['answer_relevancy'] = avg_answer_relevancy
        summary_row['faithfulness'] = avg_faithfulness
        summary_row['contextual_relevancy'] = avg_contextual_relevancy
        summary_row['contextual_precision'] = avg_contextual_precision
        summary_row['contextual_recall'] = avg_contextual_recall
        summary_row['failure_cause'] = "N/A"
        summary_row['retrieval_recall'] = avg_recall
        summary_row['retrieval_precision'] = avg_precision
        summary_row['evaluator_evaluation_time_seconds'] = evaluation_time
        summary_row['evaluator_prompt_tokens'] = evaluator_prompt_tokens
        summary_row['evaluator_completion_tokens'] = evaluator_completion_tokens
        summary_row['evaluator_total_tokens'] = evaluator_total_tokens

        writer.writerow([summary_row[col] for col in csv_columns])

    print(f'Wrote results:\n  {json_path}\n  {csv_path}\n  {summary_json_path}')


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
    parser.set_defaults(func=run_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(args.env_file)
    args.func(args)


if __name__ == '__main__':
    main()
