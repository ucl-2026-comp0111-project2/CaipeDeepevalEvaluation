from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepeval_eval.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_RESULTS_DIR,
    ensure_dirs,
    load_dotenv_loose,
    resolve_llm_settings,
)
from deepeval_eval.io_utils import load_eval_questions
from deepeval_eval.llm_client import (
    DeepEvalJudge,
    OpenAICompatibleClient,
)
from deepeval_eval.metrics import answer_scores, build_metrics, doc_id_scores


# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------

BENCHMARK_QUESTION_FILES = {
    "enterprise": DEFAULT_DATA_DIR / "enterprise_deepeval_questions.jsonl",
    "hotpotqa": DEFAULT_DATA_DIR / "hotpotqa_deepeval_questions.jsonl",
}


def _environ_get(key: str, default: str | None = None) -> str | None:
    """Read from os.environ with a fallback default."""
    return os.environ.get(key) or default


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared write_results (unified across all 3 scripts)
# ---------------------------------------------------------------------------


def _write_results(
    results_dir: Path,
    prefix: str,
    results: list[dict[str, Any]],
    evaluation_time: float,
    config_args: dict[str, Any],
) -> None:
    """Write JSON, CSV, and summary JSON. All three scripts share this logic."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"{prefix}_{timestamp}.json"
    csv_path = results_dir / f"{prefix}_{timestamp}.csv"
    summary_json_path = results_dir / f"{prefix}_{timestamp}_summary.json"

    n = len(results)
    latencies = [r.get("latency", 0.0) for r in results]
    latencies_sorted = sorted(latencies)

    p50_latency = 0.0
    p95_latency = 0.0
    if latencies_sorted:
        p50_latency = latencies_sorted[len(latencies_sorted) // 2]
        p95_index = max(
            0,
            min(
                len(latencies_sorted) - 1,
                int(math.ceil((len(latencies_sorted) * 95) / 100)) - 1,
            ),
        )
        p95_latency = latencies_sorted[p95_index]

    total_tokens_sum = sum(r.get("total_tokens", 0) for r in results)

    def get_metric_avg(metric_name: str) -> float:
        vals = [
            r.get("metrics", {}).get(metric_name, {}).get("score")
            for r in results
            if r.get("metrics", {}).get(metric_name, {}).get("score") is not None
        ]
        return sum(vals) / len(vals) if vals else 0.0

    avg_answer_relevancy = get_metric_avg("AnswerRelevancyMetric")
    avg_faithfulness = get_metric_avg("FaithfulnessMetric")
    avg_answer_correctness = get_metric_avg("AnswerCorrectnessMetric")
    avg_contextual_relevancy = get_metric_avg("ContextualRelevancyMetric")
    avg_contextual_precision = get_metric_avg("ContextualPrecisionMetric")
    avg_contextual_recall = get_metric_avg("ContextualRecallMetric")
    avg_mrr = get_metric_avg("MRRMetric")
    avg_ndcg = get_metric_avg("NDCGAtKMetric")

    avg_exact_match = sum(r.get("answer_exact_match", 0.0) for r in results) / n if n else 0.0
    avg_contains_ref = sum(r.get("answer_contains_reference", 0.0) for r in results) / n if n else 0.0
    avg_recall = sum(r.get("doc_id_recall") or 0.0 for r in results) / n if n else 0.0
    avg_precision = sum(r.get("doc_id_precision") or 0.0 for r in results) / n if n else 0.0

    # Failure causes
    failure_counts: dict[str, int] = {}
    for r in results:
        fc = "none"
        faith = r.get("metrics", {}).get("FaithfulnessMetric", {}).get("score")
        ctx_recall = r.get("metrics", {}).get("ContextualRecallMetric", {}).get("score")
        ans_rel = r.get("metrics", {}).get("AnswerRelevancyMetric", {}).get("score")
        if faith is not None and faith < 0.5:
            fc = "hallucination"
        elif ctx_recall is not None and ctx_recall < 0.5:
            fc = "poor_retrieval"
        elif ans_rel is not None and ans_rel < 0.5:
            fc = "incorrect_generation"
        r["failure_cause"] = fc
        failure_counts[fc] = failure_counts.get(fc, 0) + 1

    # Evaluator tokens
    evaluator_prompt_tokens = sum(r.get("evaluator_input_tokens", 0) for r in results)
    evaluator_completion_tokens = sum(r.get("evaluator_output_tokens", 0) for r in results)
    evaluator_total_tokens = evaluator_prompt_tokens + evaluator_completion_tokens

    # ---- Console summary ----
    datasource = config_args.get("datasource", "unknown")
    print("\n--- RUN CONFIGURATION ---")
    print(f"datasource: {datasource}")
    for k, v in config_args.items():
        print(f"{k}: {v}")

    print("\n--- OPERATIONAL BEHAVIOR ---")
    print("RAG Pipeline:")
    print(f"  P50 Latency: {p50_latency:.2f}s")
    print(f"  P95 Latency: {p95_latency:.2f}s")
    print(f"  Total Tokens: {total_tokens_sum}")
    print("\nDeepEval Evaluator:")
    print(f"  Evaluation Time: {evaluation_time:.2f}s")
    print(f"  Prompt Tokens: {evaluator_prompt_tokens}")
    print(f"  Completion Tokens: {evaluator_completion_tokens}")
    print(f"  Total Evaluator Tokens: {evaluator_total_tokens}")

    print("\n--- QUALITY METRICS AVERAGE ---")
    print(f"Average answer_relevancy: {avg_answer_relevancy:.2f}")
    print(f"Average faithfulness: {avg_faithfulness:.2f}")
    print(f"Average answer_correctness: {avg_answer_correctness:.2f}")
    print(f"Average contextual_relevancy: {avg_contextual_relevancy:.2f}")
    print(f"Average contextual_precision: {avg_contextual_precision:.2f}")
    print(f"Average contextual_recall: {avg_contextual_recall:.2f}")
    print(f"Average retrieval_mrr: {avg_mrr:.2f}")
    print(f"Average retrieval_ndcg: {avg_ndcg:.2f}")
    print(f"Average retrieval_recall: {avg_recall:.2f}")
    print(f"Average retrieval_precision: {avg_precision:.2f}")
    print(f"Average answer_exact_match: {avg_exact_match:.2f}")
    print(f"Average answer_contains_reference: {avg_contains_ref:.2f}")

    print("\n--- FAILURE CAUSE ANALYSIS ---")
    for cause, count in failure_counts.items():
        print(f"{cause:<20} {count}")

    # ---- JSON ----
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Serializable config ----
    serializable_config = {}
    for k, v in config_args.items():
        if k.startswith("_") or k in ("llm_api_key", "auth_token"):
            continue
        try:
            json.dumps(v)
            serializable_config[k] = v
        except (TypeError, OverflowError):
            serializable_config[k] = str(v)

    # ---- Summary JSON ----
    summary_data: dict[str, Any] = {
        "experiment_name": csv_path.stem,
        "datasource": datasource,
        "config_args": serializable_config,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "total_tokens": total_tokens_sum,
        "metrics": {
            "answer_relevancy": avg_answer_relevancy,
            "faithfulness": avg_faithfulness,
            "answer_correctness": avg_answer_correctness,
            "contextual_relevancy": avg_contextual_relevancy,
            "contextual_precision": avg_contextual_precision,
            "contextual_recall": avg_contextual_recall,
            "retrieval_mrr": avg_mrr,
            "retrieval_ndcg": avg_ndcg,
            "retrieval_recall": avg_recall,
            "retrieval_precision": avg_precision,
            "answer_exact_match": avg_exact_match,
            "answer_contains_reference": avg_contains_ref,
        },
        "average_retrieval_mrr": avg_mrr,
        "average_retrieval_ndcg": avg_ndcg,
        "average_retrieval_recall": avg_recall,
        "average_retrieval_precision": avg_precision,
        "deepeval_evaluator_usage": {
            "evaluation_time_seconds": evaluation_time,
            "prompt_tokens": evaluator_prompt_tokens,
            "completion_tokens": evaluator_completion_tokens,
            "total_tokens": evaluator_total_tokens,
        },
    }
    summary_json_path.write_text(
        json.dumps(summary_data, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # ---- CSV ----
    csv_columns = [
        "question_id",
        "benchmark",
        "category",
        "level",
        "answer_mode",
        "question",
        "user_input",
        "reference",
        "expected_doc_ids",
        "response",
        "retrieved_contexts",
        "retrieved_doc_ids",
        "latency",
        "latency_ms",
        "total_tokens",
        "log_file",
        "answer_exact_match",
        "answer_contains_reference",
        "answer_relevancy",
        "faithfulness",
        "factual_correctness",
        "contextual_relevancy",
        "contextual_precision",
        "contextual_recall",
        "mrr",
        "ndcg_at_k",
        "answer_relevancy_reason",
        "faithfulness_reason",
        "factual_correctness_reason",
        "contextual_relevancy_reason",
        "contextual_precision_reason",
        "contextual_recall_reason",
        "mrr_reason",
        "ndcg_at_k_reason",
        "failure_cause",
        "retrieval_recall",
        "retrieval_precision",
        "evaluator_evaluation_time_seconds",
        "evaluator_prompt_tokens",
        "evaluator_completion_tokens",
        "evaluator_total_tokens",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        for r in results:
            metrics_dict = r.get("metrics", {})
            retrieved_contexts_str = json.dumps(r.get("retrieved_contexts") or [])
            expected_doc_ids_str = ";".join(r.get("expected_doc_ids") or [])
            retrieved_doc_ids_str = ";".join(r.get("retrieved_doc_ids") or [])
            writer.writerow([
                r.get("question_id"),
                r.get("benchmark", datasource),
                r.get("category"),
                r.get("level"),
                r.get("answer_mode"),
                r.get("question"),
                r.get("user_input"),
                r.get("reference"),
                expected_doc_ids_str,
                r.get("actual_output"),
                retrieved_contexts_str,
                retrieved_doc_ids_str,
                r.get("latency"),
                r.get("latency_ms", r.get("latency", 0.0) * 1000.0),
                r.get("total_tokens"),
                r.get("log_file"),
                r.get("answer_exact_match"),
                r.get("answer_contains_reference"),
                metrics_dict.get("AnswerRelevancyMetric", {}).get("score"),
                metrics_dict.get("FaithfulnessMetric", {}).get("score"),
                metrics_dict.get("AnswerCorrectnessMetric", {}).get("score"),
                metrics_dict.get("ContextualRelevancyMetric", {}).get("score"),
                metrics_dict.get("ContextualPrecisionMetric", {}).get("score"),
                metrics_dict.get("ContextualRecallMetric", {}).get("score"),
                metrics_dict.get("MRRMetric", {}).get("score"),
                metrics_dict.get("NDCGAtKMetric", {}).get("score"),
                metrics_dict.get("AnswerRelevancyMetric", {}).get("reason"),
                metrics_dict.get("FaithfulnessMetric", {}).get("reason"),
                metrics_dict.get("AnswerCorrectnessMetric", {}).get("reason"),
                metrics_dict.get("ContextualRelevancyMetric", {}).get("reason"),
                metrics_dict.get("ContextualPrecisionMetric", {}).get("reason"),
                metrics_dict.get("ContextualRecallMetric", {}).get("reason"),
                metrics_dict.get("MRRMetric", {}).get("reason"),
                metrics_dict.get("NDCGAtKMetric", {}).get("reason"),
                r.get("failure_cause"),
                r.get("doc_id_recall"),
                r.get("doc_id_precision"),
                evaluation_time,
                r.get("evaluator_input_tokens"),
                r.get("evaluator_output_tokens"),
                r.get("evaluator_total_tokens"),
            ])

        # AVERAGE_METRICS row
        summary_row = dict.fromkeys(csv_columns, "")
        summary_row["question"] = "AVERAGE_METRICS"
        summary_row["latency"] = sum(latencies) / n if n else 0.0
        summary_row["latency_ms"] = (sum(latencies) / n if n else 0.0) * 1000.0
        summary_row["total_tokens"] = total_tokens_sum / n if n else 0.0
        summary_row["answer_exact_match"] = avg_exact_match
        summary_row["answer_contains_reference"] = avg_contains_ref
        summary_row["answer_relevancy"] = avg_answer_relevancy
        summary_row["faithfulness"] = avg_faithfulness
        summary_row["answer_correctness"] = avg_answer_correctness
        summary_row["contextual_relevancy"] = avg_contextual_relevancy
        summary_row["contextual_precision"] = avg_contextual_precision
        summary_row["contextual_recall"] = avg_contextual_recall
        summary_row["mrr"] = avg_mrr
        summary_row["ndcg_at_k"] = avg_ndcg
        summary_row["failure_cause"] = "N/A"
        summary_row["retrieval_recall"] = avg_recall
        summary_row["retrieval_precision"] = avg_precision
        summary_row["evaluator_evaluation_time_seconds"] = evaluation_time
        summary_row["evaluator_prompt_tokens"] = evaluator_prompt_tokens
        summary_row["evaluator_completion_tokens"] = evaluator_completion_tokens
        summary_row["evaluator_total_tokens"] = evaluator_total_tokens
        writer.writerow([summary_row[col] for col in csv_columns])

    print(f"Wrote results:\n    {json_path}\n    {csv_path}\n    {summary_json_path}")


# ---------------------------------------------------------------------------
# Shared arg-parsing helpers
# ---------------------------------------------------------------------------


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add common eval arguments to an existing subparser."""
    parser.add_argument("--datasource-id", default=None, help="The target CAIPE datasource")
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--agentic", action="store_true",
        help="Route queries through caipe-supervisor A2A endpoint",
    )
    parser.add_argument(
        "--supervisor-url", default="http://localhost:8000",
        help="CAIPE supervisor URL for agentic eval",
    )
    parser.add_argument(
        "--fail-on-error", action="store_true",
        help="Fail loudly if a query evaluation fails after retries",
    )
    parser.add_argument(
        "--precompute", action="store_true",
        help="Run precomputed benchmark (gold-source retrieval via CAIPE)",
    )


def _build_config_args(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize args into a JSON-serializable dict (hiding secrets)."""
    config = {}
    for k, v in vars(args).items():
        if v is None or k in ("llm_api_key", "auth_token") or callable(v) or k.startswith("_"):
            continue
        if isinstance(v, Path):
            config[k] = str(v)
        elif isinstance(v, (str, int, float, bool, list, dict)):
            config[k] = v
        else:
            config[k] = str(v)
    return config


# ---------------------------------------------------------------------------
# Unified run_eval (works for enterprise and hotpotqa backends)
# ---------------------------------------------------------------------------


def _build_rag_client(args: argparse.Namespace, env_values: dict[str, Any]) -> Any:
    """Factory function to build the appropriate RAG client for the evaluation run."""
    from deepeval_eval.caipe_client import build_caipe_client
    if getattr(args, "precompute", False):
        from deepeval_eval.precomputed_client import PrecomputedRagClient
        caipe_client = build_caipe_client(env_values)
        return PrecomputedRagClient(caipe_client)
    elif getattr(args, "agentic", False):
        from deepeval_eval.rag_client import AgenticRagAdapter
        return AgenticRagAdapter(
            supervisor_url=getattr(args, "supervisor_url", "http://localhost:8000"),
            results_dir=getattr(args, "results_dir", None),
            fail_on_error=getattr(args, "fail_on_error", False),
        )
    else:
        return build_caipe_client(env_values)


def _run_eval(args: argparse.Namespace) -> None:
    """Unified evaluation loop for enterprise/hotpotqa/precomputed backends."""
    ensure_dirs(args.results_dir)
    env_values = load_dotenv_loose(args.env_file)
    base_url, api_key, model = resolve_llm_settings(
        args.env_file, args.llm_base_url, args.llm_api_key, args.llm_model,
    )

    llm_client = OpenAICompatibleClient(model=model, api_key=api_key, base_url=base_url)
    judge = DeepEvalJudge("openai-compatible", model, llm_client).model
    metrics = build_metrics(judge)

    from deepeval.test_case import LLMTestCase

    # Resolve datasource
    datasource_id = getattr(args, "datasource_id", None) or env_values.get("CAIPE_DATASOURCE_ID")

    # Determine benchmark backend
    benchmark = getattr(args, "benchmark", "enterprise")

    # Load questions
    questions_file = getattr(args, "questions_file", None)
    if questions_file is None:
        bf = BENCHMARK_QUESTION_FILES.get(benchmark)
        if bf is not None:
            questions_file = bf
    if questions_file is None:
        raise FileNotFoundError(f"No questions file resolved for benchmark={benchmark}")

    rows = load_eval_questions(
        questions_file,
        args.max_items,
        getattr(args, "limit_per_category", None),
        combine_with_level=(benchmark == "hotpotqa"),
    )

    # Optionally filter by question IDs (enterprise-style)
    question_ids = getattr(args, "question_ids", None)
    if question_ids:
        target_ids = {qid.strip() for qid in question_ids.split(",")}
        rows = [row for row in rows if str(row.get("question_id")) in target_ids]

    # Optionally filter by question indices (enterprise-style)
    question_indices = getattr(args, "question_indices", None)
    if question_indices:
        indices = set()
        for part in question_indices.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start_str, end_str = part.split("-", 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    for i in range(start, end + 1):
                        if 1 <= i <= len(rows):
                            indices.add(i)
                except ValueError:
                    pass
            else:
                try:
                    i = int(part)
                    if 1 <= i <= len(rows):
                        indices.add(i)
                except ValueError:
                    pass
        rows = [rows[i - 1] for i in sorted(indices) if 1 <= i <= len(rows)]

    # Build RAG client for execution
    rag_client = _build_rag_client(args, env_values)

    results: list[dict[str, Any]] = []
    start_eval_time = time.time()

    for idx, row in enumerate(rows, start=1):
        question = row["user_input"]
        reference = row.get("reference") or ""
        print(f"Evaluating {idx}/{len(rows)}: {question[:90]}")

        llm_client.reset_tokens()

        # Query RAG backend polymorphically
        query_res = rag_client.query(
            question=question,
            reference=reference,
            datasource_id=datasource_id,
            top_k=args.top_k,
            answer_mode=getattr(args, "answer_mode", "reference"),
            benchmark=benchmark,
            llm_client=llm_client,
            max_context_chars=args.max_context_chars,
        )

        answer = query_res.answer
        trimmed_contexts = query_res.contexts
        sources = query_res.sources
        current_retrieved_ids = query_res.retrieved_doc_ids

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference,
            retrieval_context=trimmed_contexts,
            context=row.get("context") or [],
            metadata={
                "retrieved_doc_ids": row.get("retrieved_doc_ids", []),
                "expected_doc_ids": row.get("expected_doc_ids", []),
            },
        )

        metric_results: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            try:
                metric.measure(test_case)
                metric_results[metric.__class__.__name__] = {
                    "score": metric.score,
                    "success": metric.success,
                    "reason": metric.reason,
                }
            except Exception as exc:
                metric_results[metric.__class__.__name__] = {
                    "score": None,
                    "success": False,
                    "reason": f"metric failed: {exc}",
                }

        doc_recall, doc_precision = doc_id_scores(sources, list(row.get("expected_doc_ids") or []))
        exact_match, contains_reference = answer_scores(answer, reference)

        results.append({
            "question_id": row.get("question_id"),
            "benchmark": benchmark,
            "category": row.get("category"),
            "level": row.get("level"),
            "answer_mode": getattr(args, "answer_mode", None),
            "question": question,
            "user_input": question,
            "reference": reference,
            "actual_output": answer,
            "retrieved_contexts": trimmed_contexts,
            "retrieved_doc_ids": current_retrieved_ids,
            "expected_doc_ids": row.get("expected_doc_ids") or [],
            "doc_id_recall": doc_recall,
            "doc_id_precision": doc_precision,
            "answer_exact_match": exact_match,
            "answer_contains_reference": contains_reference,
            "metrics": metric_results,
            "input_tokens": query_res.input_tokens,
            "output_tokens": query_res.output_tokens,
            "total_tokens": query_res.total_tokens,
            "evaluator_input_tokens": llm_client.input_tokens,
            "evaluator_output_tokens": llm_client.output_tokens,
            "evaluator_total_tokens": llm_client.total_tokens,
            "latency": query_res.latency_sec,
            "latency_ms": query_res.latency_ms,
            "log_file": query_res.log_file,
        })

    eval_time = time.time() - start_eval_time
    config_args = _build_config_args(args)
    config_args["datasource"] = benchmark


    _write_results(
        args.results_dir,
        prefix=f"{benchmark}_deepeval",
        results=results,
        evaluation_time=eval_time,
        config_args=config_args,
    )


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def _eval_subcommand(args: argparse.Namespace) -> None:
    """Dispatch to the unified evaluation loop."""
    _run_eval(args)


# ---------------------------------------------------------------------------
# Build parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
         description="DeepEval evaluation pipeline supporting enterprise and hotpotqa backends",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- eval subcommand ----
    eval_parser = subparsers.add_parser("eval", help="Run DeepEval evaluation")
    eval_parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_QUESTION_FILES),
         default="enterprise",
        help="Benchmark backend to evaluate against",
    )
    eval_parser.add_argument(
        "--answer-mode",
        choices=["reference", "generate"],
        default="reference",
        help="reference uses the benchmark answer as actual_output; generate answers from gold context using the LLM",
    )
    _add_eval_args(eval_parser)
    eval_parser.set_defaults(func=_eval_subcommand)

    return parser


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(getattr(args, "env_file", DEFAULT_ENV_FILE))
    args.func(args)


if __name__ == "__main__":
    main()
