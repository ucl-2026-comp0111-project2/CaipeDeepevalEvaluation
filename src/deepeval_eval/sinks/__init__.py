from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepeval_eval.sinks.composite_sink import CompositeResultSink
from deepeval_eval.sinks.db_sink import DatabaseResultSink
from deepeval_eval.sinks.file_sink import FileResultSink
from deepeval_eval.sinks.metrics_aggregator import (
    calculate_latency_percentiles,
    categorize_failure_causes,
    compute_all_metric_averages,
    discover_all_metrics,
)
from deepeval_eval.sinks.protocol import ResultSink


def write_evaluation_results(
    results_dir: Path,
    prefix: str,
    results: list[dict[str, Any]],
    evaluation_time: float,
    config_args: dict[str, Any],
    sinks: Sequence[ResultSink] | None = None,
) -> None:
    """Save results using the provided sinks (defaults to FileResultSink)."""
    if sinks is None:
        sinks = [FileResultSink()]
    composite = CompositeResultSink(sinks)
    composite.save(results_dir, prefix, results, evaluation_time, config_args)


__all__ = [
    "ResultSink",
    "FileResultSink",
    "DatabaseResultSink",
    "CompositeResultSink",
    "write_evaluation_results",
    "discover_all_metrics",
    "compute_all_metric_averages",
    "calculate_latency_percentiles",
    "categorize_failure_causes",
]
