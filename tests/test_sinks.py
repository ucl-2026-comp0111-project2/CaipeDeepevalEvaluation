import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepeval_eval.sinks import (
    CompositeResultSink,
    DatabaseResultSink,
    FileResultSink,
    ResultSink,
    calculate_latency_percentiles,
    categorize_failure_causes,
    compute_metric_averages,
    discover_all_metrics,
    write_evaluation_results,
)


def _mock_record(ar_score=0.9, fa_score=0.8, custom_score=0.95):
    return {
        "question_id": "q1",
        "question": "What is AI?",
        "latency": 1.2,
        "total_tokens": 150,
        "metrics": {
            "AnswerRelevancyMetric": {"score": ar_score, "success": True, "reason": "Good"},
            "FaithfulnessMetric": {"score": fa_score, "success": True, "reason": "Faithful"},
            "CustomNewMetric": {"score": custom_score, "success": True, "reason": "Custom ok"},
        },
    }


def test_discover_all_metrics():
    records = [_mock_record()]
    discovered = discover_all_metrics(records)
    assert "AnswerRelevancyMetric" in discovered
    assert "FaithfulnessMetric" in discovered
    assert "CustomNewMetric" in discovered


def test_compute_metric_averages():
    records = [_mock_record(ar_score=0.8), _mock_record(ar_score=1.0)]
    averages = compute_metric_averages(records, ["AnswerRelevancyMetric", "FaithfulnessMetric"])
    assert averages["AnswerRelevancyMetric"] == 0.9
    assert averages["FaithfulnessMetric"] == 0.8


def test_calculate_latency_percentiles():
    p50, p95 = calculate_latency_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert p50 == 3.0
    assert p95 == 5.0
    # Empty case
    assert calculate_latency_percentiles([]) == (0.0, 0.0)


def test_categorize_failure_causes():
    records = [
        {"metrics": {"FaithfulnessMetric": {"score": 0.3}}},
        {"metrics": {"ContextualRecallMetric": {"score": 0.4}}},
        {"metrics": {"AnswerRelevancyMetric": {"score": 0.2}}},
        {"metrics": {"FaithfulnessMetric": {"score": 0.9}}},
    ]
    counts = categorize_failure_causes(records)
    assert counts["hallucination"] == 1
    assert counts["poor_retrieval"] == 1
    assert counts["incorrect_generation"] == 1
    assert counts["none"] == 1


def test_file_result_sink_saves(tmp_path: Path):
    sink = FileResultSink()
    results = [_mock_record() for _ in range(3)]
    config_args = {"datasource": "test_ds", "top_k": 3}
    sink.save(tmp_path, "test_prefix", results, 5.0, config_args)

    all_json = list(tmp_path.glob("test_prefix_*.json"))
    summary_files = list(tmp_path.glob("test_prefix_*_summary.json"))
    json_files = [f for f in all_json if not f.name.endswith("_summary.json")]
    csv_files = list(tmp_path.glob("test_prefix_*.csv"))

    assert len(json_files) == 1
    assert len(csv_files) == 1
    assert len(summary_files) == 1

    summary_data = json.loads(summary_files[0].read_text(encoding="utf-8"))
    assert summary_data["datasource"] == "test_ds"
    assert "metrics" in summary_data


def test_csv_contains_all_metric_scores_and_reasons(tmp_path: Path):
    record = {
        "question_id": "q100",
        "benchmark": "enterprise",
        "question": "What is CAIPE?",
        "user_input": "What is CAIPE?",
        "reference": "CAIPE is an AI platform",
        "actual_output": "CAIPE is an AI platform",
        "metrics": {
            "AnswerRelevancyMetric": {"score": 0.92, "success": True, "reason": "Highly relevant answer"},
            "FaithfulnessMetric": {"score": 0.88, "success": True, "reason": "Faithful to context"},
            "AnswerCorrectnessMetric": {"score": 0.95, "success": True, "reason": "Factually correct"},
            "ContextualRelevancyMetric": {"score": 0.85, "success": True, "reason": "Relevant context"},
            "ContextualPrecisionMetric": {"score": 0.90, "success": True, "reason": "Precise context"},
            "ContextualRecallMetric": {"score": 0.80, "success": True, "reason": "High recall"},
            "MRRMetric": {"score": 1.0, "success": True, "reason": "Top rank match"},
            "NDCGAtKMetric": {"score": 0.99, "success": True, "reason": "High NDCG gain"},
        },
    }

    sink = FileResultSink()
    sink.save(tmp_path, "csv_test", [record], 2.5, {"datasource": "enterprise"})

    csv_files = list(tmp_path.glob("csv_test_*.csv"))
    assert len(csv_files) == 1

    with csv_files[0].open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # First row should be the question result record
    row = rows[0]
    assert row["question_id"] == "q100"
    assert float(row["answer_relevancy"]) == 0.92
    assert row["answer_relevancy_reason"] == "Highly relevant answer"
    assert float(row["faithfulness"]) == 0.88
    assert row["faithfulness_reason"] == "Faithful to context"
    assert float(row["answer_correctness"]) == 0.95
    assert row["answer_correctness_reason"] == "Factually correct"

    # Second row should be the AVERAGE_METRICS row
    avg_row = rows[1]
    assert avg_row["question"] == "AVERAGE_METRICS"
    assert float(avg_row["answer_relevancy"]) == 0.92


class CustomDuckTypedSink:
    """A duck-typed custom sink that does NOT inherit from any base class, satisfying ResultSink Protocol."""

    def __init__(self):
        self.saved = False

    def save(
        self,
        results_dir: Path,
        prefix: str,
        results: list[dict],
        evaluation_time: float,
        config_args: dict,
    ) -> None:
        self.saved = True


def test_custom_duck_typed_sink_with_protocol(tmp_path: Path):
    custom_sink: ResultSink = CustomDuckTypedSink()

    # Structural check - static typing contract verification
    write_evaluation_results(
        results_dir=tmp_path,
        prefix="custom_test",
        results=[_mock_record()],
        evaluation_time=1.0,
        config_args={"datasource": "custom"},
        sinks=[custom_sink],
    )
    assert custom_sink.saved is True


def test_composite_result_sink(tmp_path: Path):
    sink1 = CustomDuckTypedSink()
    sink2 = CustomDuckTypedSink()

    composite = CompositeResultSink([sink1])
    composite.add_sink(sink2)
    composite.save(tmp_path, "comp", [_mock_record()], 1.0, {})

    assert sink1.saved is True
    assert sink2.saved is True


@patch.object(DatabaseResultSink, "_get_connection")
def test_database_result_sink_query_runs(mock_get_conn):
    mock_psycopg2_extras = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"run_id": "run_1", "batch_id": "b1", "config_name": "cfg", "loaded_at": "2026-07-22", "config_json": "{}"}
    ]

    with patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": mock_psycopg2_extras}):
        db_sink = DatabaseResultSink(connection_string="postgresql://user:pass@localhost:5432/db")
        runs = db_sink.query_runs(limit=5)

    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_1"


def test_database_result_sink_missing_psycopg2():
    with patch.dict("sys.modules", {"psycopg2": None, "psycopg2.extras": None}):
        db_sink = DatabaseResultSink(connection_string="postgresql://user:pass@localhost:5432/db")
        runs = db_sink.query_runs(limit=5)
        assert runs == []

        # Ensure save does not crash when psycopg2 is missing
        db_sink.save(Path("/tmp"), "test", [], 1.0, {})
