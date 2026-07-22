from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from deepeval_eval.metrics import get_metric_column_name
from deepeval_eval.sinks.metrics_aggregator import (
    calculate_latency_percentiles,
    categorize_failure_causes,
    compute_metric_averages,
    discover_all_metrics,
)


class FileResultSink:
    """Saves evaluation results to JSON, CSV, and summary JSON files with dynamic metric aggregation."""

    def save(
        self,
        results_dir: Path,
        prefix: str,
        results: list[dict[str, Any]],
        evaluation_time: float,
        config_args: dict[str, Any],
    ) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        json_path = results_dir / f"{prefix}_{timestamp}.json"
        csv_path = results_dir / f"{prefix}_{timestamp}.csv"
        summary_json_path = results_dir / f"{prefix}_{timestamp}_summary.json"

        n = len(results)
        latencies = [r.get("latency", 0.0) for r in results]
        p50_latency, p95_latency = calculate_latency_percentiles(latencies)

        total_tokens_sum = sum(r.get("total_tokens", 0) for r in results)

        # Dynamic metric discovery and score averages
        discovered_metrics = discover_all_metrics(results)
        metric_averages = compute_metric_averages(results, discovered_metrics)

        metric_score_cols = [get_metric_column_name(m) for m in discovered_metrics]
        metric_reason_cols = [
            f"{get_metric_column_name(m)}_reason" for m in discovered_metrics
        ]

        avg_mrr = metric_averages.get("MRRMetric", 0.0)
        avg_ndcg = metric_averages.get("NDCGAtKMetric", 0.0)

        avg_recall = (
            sum(r.get("doc_id_recall") or 0.0 for r in results) / n if n else 0.0
        )
        avg_precision = (
            sum(r.get("doc_id_precision") or 0.0 for r in results) / n if n else 0.0
        )

        # Categorize failure causes
        failure_counts = categorize_failure_causes(results)

        evaluator_prompt_tokens = sum(
            r.get("evaluator_input_tokens", 0) for r in results
        )
        evaluator_completion_tokens = sum(
            r.get("evaluator_output_tokens", 0) for r in results
        )
        evaluator_total_tokens = evaluator_prompt_tokens + evaluator_completion_tokens

        # Console Summary
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
        for m_name, score in metric_averages.items():
            col_name = get_metric_column_name(m_name)
            print(f"Average {col_name}: {score:.2f}")
        print(f"Average retrieval_recall: {avg_recall:.2f}")
        print(f"Average retrieval_precision: {avg_precision:.2f}")

        print("\n--- FAILURE CAUSE ANALYSIS ---")
        for cause, count in failure_counts.items():
            print(f"{cause:<20} {count}")

        # Write JSON results
        json_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        serializable_config = {}
        for k, v in config_args.items():
            if k.startswith("_") or k in ("llm_api_key", "auth_token"):
                continue
            try:
                json.dumps(v)
                serializable_config[k] = v
            except (TypeError, OverflowError):
                serializable_config[k] = str(v)

        summary_metrics: dict[str, float] = {
            get_metric_column_name(m): metric_averages.get(m, 0.0)
            for m in discovered_metrics
        }
        summary_metrics.update(
            {
                "retrieval_mrr": avg_mrr,
                "retrieval_ndcg": avg_ndcg,
                "retrieval_recall": avg_recall,
                "retrieval_precision": avg_precision,
            }
        )

        summary_data: dict[str, Any] = {
            "experiment_name": csv_path.stem,
            "datasource": datasource,
            "config_args": serializable_config,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "total_tokens": total_tokens_sum,
            "metrics": summary_metrics,
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

        # CSV Columns
        csv_columns = (
            [
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
            ]
            + metric_score_cols
            + metric_reason_cols
            + [
                "failure_cause",
                "retrieval_recall",
                "retrieval_precision",
                "evaluator_evaluation_time_seconds",
                "evaluator_prompt_tokens",
                "evaluator_completion_tokens",
                "evaluator_total_tokens",
            ]
        )

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_columns)
            for r in results:
                metrics_dict = r.get("metrics", {})
                retrieved_contexts_str = json.dumps(r.get("retrieved_contexts") or [])
                expected_doc_ids_str = ";".join(r.get("expected_doc_ids") or [])
                retrieved_doc_ids_str = ";".join(r.get("retrieved_doc_ids") or [])
                scores = [
                    metrics_dict.get(m, {}).get("score") for m in discovered_metrics
                ]
                reasons = [
                    metrics_dict.get(m, {}).get("reason") for m in discovered_metrics
                ]
                writer.writerow(
                    [
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
                        *scores,
                        *reasons,
                        r.get("failure_cause"),
                        r.get("doc_id_recall"),
                        r.get("doc_id_precision"),
                        evaluation_time,
                        r.get("evaluator_input_tokens"),
                        r.get("evaluator_output_tokens"),
                        r.get("evaluator_total_tokens"),
                    ]
                )

            # Summary row
            summary_row = dict.fromkeys(csv_columns, "")
            summary_row["question"] = "AVERAGE_METRICS"
            summary_row["latency"] = sum(latencies) / n if n else 0.0
            summary_row["latency_ms"] = (sum(latencies) / n if n else 0.0) * 1000.0
            summary_row["total_tokens"] = total_tokens_sum / n if n else 0.0
            for m in discovered_metrics:
                summary_row[get_metric_column_name(m)] = metric_averages.get(m, 0.0)

            summary_row["failure_cause"] = "N/A"
            summary_row["retrieval_recall"] = avg_recall
            summary_row["retrieval_precision"] = avg_precision
            summary_row["evaluator_evaluation_time_seconds"] = evaluation_time
            summary_row["evaluator_prompt_tokens"] = evaluator_prompt_tokens
            summary_row["evaluator_completion_tokens"] = evaluator_completion_tokens
            summary_row["evaluator_total_tokens"] = evaluator_total_tokens
            writer.writerow([summary_row[col] for col in csv_columns])

        print(
            f"Wrote results:\n    {json_path}\n    {csv_path}\n    {summary_json_path}"
        )
