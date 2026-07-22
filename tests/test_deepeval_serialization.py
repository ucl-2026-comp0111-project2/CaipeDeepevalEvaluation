import json
from pathlib import Path

import pytest

from deepeval_eval.enterprise_deepeval import write_results as write_results_enterprise
from deepeval_eval.hotpotqa_deepeval import write_results as write_results_hotpotqa
from deepeval_eval.precomputed_deepeval import (
    write_results as write_results_precomputed,
)


class DummyObject:
    def __repr__(self):
        return "<DummyObject>"


@pytest.fixture
def sample_results():
    return [
        {
            "question_id": "q1",
            "question": "question 1",
            "user_input": "question 1",
            "reference": "ref 1",
            "expected_doc_ids": ["doc_1"],
            "category": "cat1",
            "actual_output": "ans 1",
            "retrieved_contexts": ["ctx 1"],
            "retrieved_doc_ids": ["doc_1"],
            "doc_id_recall": 1.0,
            "doc_id_precision": 1.0,
            "metrics": {
                "AnswerRelevancyMetric": {"score": 0.9, "reason": "good"},
                "FaithfulnessMetric": {"score": 0.9, "reason": "good"},
            },
            "evaluator_input_tokens": 10,
            "evaluator_output_tokens": 20,
            "evaluator_total_tokens": 30,
            "latency": 5.0,
            "total_tokens": 100,
            "log_file": "log1",
        }
    ]


def test_enterprise_write_results_serialization_positive(
    tmp_path: Path, sample_results
):
    config_args = {"datasource_id": "test_ds", "top_k": 5, "agentic": True}
    # Should write successfully without exceptions
    write_results_enterprise(
        results_dir=tmp_path,
        results=sample_results,
        evaluation_time=10.0,
        config_args=config_args,
        datasource="test_ds",
    )
    # Check that summary file is written and config is correct
    summary_files = list(tmp_path.glob("*_summary.json"))
    assert len(summary_files) == 1
    with open(summary_files[0], encoding="utf-8") as f:
        data = json.load(f)
        assert data["config_args"]["top_k"] == 5


def test_enterprise_write_results_serialization_negative(
    tmp_path: Path, sample_results
):
    # Pass a non-serializable object to trigger the fallback
    config_args = {
        "datasource_id": "test_ds",
        "custom_obj": DummyObject(),
        "_private_val": "secret",
    }
    # Should write successfully without exceptions due to sanitization
    write_results_enterprise(
        results_dir=tmp_path,
        results=sample_results,
        evaluation_time=10.0,
        config_args=config_args,
        datasource="test_ds",
    )
    summary_files = list(tmp_path.glob("*_summary.json"))
    assert len(summary_files) == 1
    with open(summary_files[0], encoding="utf-8") as f:
        data = json.load(f)
        # Verify custom_obj is converted to string
        assert data["config_args"]["custom_obj"] == "<DummyObject>"
        # Verify private values are excluded
        assert "_private_val" not in data["config_args"]


def test_hotpotqa_write_results_serialization_positive(tmp_path: Path, sample_results):
    config_args = {"datasource_id": "test_ds", "top_k": 5}
    # Should write successfully without exceptions
    write_results_hotpotqa(
        results_dir=tmp_path,
        results=sample_results,
        evaluation_time=10.0,
        config_args=config_args,
        datasource="test_ds",
    )
    summary_files = list(tmp_path.glob("*_summary.json"))
    assert len(summary_files) == 1


def test_hotpotqa_write_results_serialization_negative(tmp_path: Path, sample_results):
    config_args = {"custom_obj": DummyObject(), "_private_val": "secret"}
    # Should write successfully without exceptions due to sanitization
    write_results_hotpotqa(
        results_dir=tmp_path,
        results=sample_results,
        evaluation_time=10.0,
        config_args=config_args,
        datasource="test_ds",
    )
    summary_files = list(tmp_path.glob("*_summary.json"))
    assert len(summary_files) == 1
    with open(summary_files[0], encoding="utf-8") as f:
        data = json.load(f)
        assert data["config_args"]["custom_obj"] == "<DummyObject>"
        assert "_private_val" not in data["config_args"]


def test_precomputed_write_results_serialization_positive(
    tmp_path: Path, sample_results
):
    config_args = {"benchmark": "hotpotqa", "answer_mode": "short"}
    # Should write successfully without exceptions
    write_results_precomputed(
        results_dir=tmp_path,
        dataset_name="hotpotqa",
        answer_mode="short",
        results=sample_results,
        evaluation_time=10.0,
        config_args=config_args,
    )
    summary_files = list(tmp_path.glob("*_summary.json"))
    assert len(summary_files) == 1


def test_precomputed_write_results_serialization_negative(
    tmp_path: Path, sample_results
):
    config_args = {"custom_obj": DummyObject(), "_private_val": "secret"}
    # Should write successfully without exceptions due to sanitization
    write_results_precomputed(
        results_dir=tmp_path,
        dataset_name="hotpotqa",
        answer_mode="short",
        results=sample_results,
        evaluation_time=10.0,
        config_args=config_args,
    )
    summary_files = list(tmp_path.glob("*_summary.json"))
    assert len(summary_files) == 1
    with open(summary_files[0], encoding="utf-8") as f:
        data = json.load(f)
        assert data["config_args"]["custom_obj"] == "<DummyObject>"
        assert "_private_val" not in data["config_args"]
