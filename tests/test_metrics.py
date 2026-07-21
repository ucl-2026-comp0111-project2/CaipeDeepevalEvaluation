from __future__ import annotations

from unittest.mock import MagicMock
from deepeval.test_case import LLMTestCase

from deepeval_eval.metrics import (
    AnswerCorrectnessMetric,
    MRRMetric,
    NDCGAtKMetric,
    answer_scores,
    build_metrics,
    doc_id_scores,
    normalize_answer,
)


from deepeval.models.base_model import DeepEvalBaseLLM

class DummyJudge(DeepEvalBaseLLM):
    def __init__(self):
        super().__init__(model="test")
    def load_model(self, *args, **kwargs):
        return None
    def get_model_name(self, *args, **kwargs) -> str:
        return "dummy"
    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        return ""
    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        return ""


def test_build_metrics_positive() -> None:
    mock_model = DummyJudge()
    metrics = build_metrics(mock_model)
    assert len(metrics) == 8
    assert any(isinstance(m, MRRMetric) for m in metrics)
    assert any(isinstance(m, NDCGAtKMetric) for m in metrics)


def test_build_metrics_negative() -> None:
    metrics = build_metrics(None)
    assert len(metrics) == 8


def test_doc_id_scores_positive() -> None:
    retrieved = [{"document_id": "doc1"}, {"document_id": "doc2"}, {"document_id": "doc3"}]
    expected = ["doc2", "doc4"]
    recall, precision = doc_id_scores(retrieved, expected)
    assert recall == 0.5  # 1 hit out of 2 expected
    assert round(precision, 4) == round(1 / 3, 4)  # 1 hit out of 3 retrieved


def test_doc_id_scores_negative() -> None:
    # No expected doc IDs
    recall, precision = doc_id_scores([{"document_id": "doc1"}], [])
    assert recall == 0.0
    assert precision == 0.0

    # Empty retrieved
    recall2, precision2 = doc_id_scores([], ["doc1"])
    assert recall2 == 0.0
    assert precision2 == 0.0


def test_normalize_answer_positive() -> None:
    raw = "The quick brown fox, a simple dog!"
    normalized = normalize_answer(raw)
    assert normalized == "quick brown fox simple dog"


def test_normalize_answer_negative() -> None:
    assert normalize_answer("") == ""
    assert normalize_answer("a an the !!!") == ""


def test_answer_scores_positive() -> None:
    exact, partial = answer_scores("Paris", "Paris")
    assert exact == 1.0
    assert partial == 1.0

    exact2, partial2 = answer_scores("The capital of France is Paris", "Paris")
    assert exact2 == 0.0
    assert partial2 == 1.0


def test_answer_scores_negative() -> None:
    exact, partial = answer_scores("London", "")
    assert exact == 0.0
    assert partial == 0.0

    exact2, partial2 = answer_scores("London", "Paris")
    assert exact2 == 0.0
    assert partial2 == 0.0


def test_answer_correctness_metric_positive() -> None:
    mock_model = DummyJudge()
    metric = AnswerCorrectnessMetric(name="TestCorrectness", model=mock_model)
    metric.geval_judge = MagicMock()
    metric.geval_judge.measure.return_value = 0.9
    metric.geval_judge.is_successful.return_value = True
    metric.geval_judge.reason = "High factual match"

    test_case = LLMTestCase(input="What is 2+2?", actual_output="4", expected_output="4")
    score = metric.measure(test_case)
    assert score == 0.9
    assert metric.is_successful() is True
    assert metric.get_reason() == "High factual match"


def test_answer_correctness_metric_negative() -> None:
    mock_model = DummyJudge()
    metric = AnswerCorrectnessMetric(name="TestCorrectness", model=mock_model)
    metric.geval_judge = MagicMock()
    metric.geval_judge.measure.return_value = 0.1
    metric.geval_judge.is_successful.return_value = False
    metric.geval_judge.reason = "Incorrect answer"

    test_case = LLMTestCase(input="What is 2+2?", actual_output="5", expected_output="4")
    score = metric.measure(test_case)
    assert score == 0.1
    assert metric.is_successful() is False
    assert metric.get_reason() == "Incorrect answer"


def test_mrr_metric_positive() -> None:
    metric = MRRMetric()
    test_case = LLMTestCase(
        input="test",
        actual_output="out",
        metadata={"retrieved_doc_ids": ["docA", "docB", "docC"], "expected_doc_ids": ["docB"]},
    )
    score = metric.measure(test_case)
    assert score == 0.5  # Rank 2 -> 1/2
    assert metric.is_successful() is True
    assert "0.5000" in metric.get_reason()


def test_mrr_metric_negative() -> None:
    metric = MRRMetric()

    # No match
    test_case_no_match = LLMTestCase(
        input="test",
        actual_output="out",
        metadata={"retrieved_doc_ids": ["docA", "docB"], "expected_doc_ids": ["docZ"]},
    )
    assert metric.measure(test_case_no_match) == 0.0
    assert metric.is_successful() is False

    # Missing metadata
    test_case_empty = LLMTestCase(input="test", actual_output="out")
    assert metric.measure(test_case_empty) == 0.0


def test_ndcg_at_k_metric_positive() -> None:
    metric = NDCGAtKMetric(k=3)
    test_case = LLMTestCase(
        input="test",
        actual_output="out",
        metadata={"retrieved_doc_ids": ["doc1", "doc2", "doc3"], "expected_doc_ids": ["doc1", "doc3"]},
    )
    score = metric.measure(test_case)
    assert score > 0.0
    assert metric.is_successful() is True
    assert "nDCG@3" in metric.get_reason()


def test_ndcg_at_k_metric_negative() -> None:
    metric = NDCGAtKMetric(k=3)

    # Empty expected doc IDs
    test_case_empty = LLMTestCase(
        input="test",
        actual_output="out",
        metadata={"retrieved_doc_ids": ["doc1"], "expected_doc_ids": []},
    )
    assert metric.measure(test_case_empty) == 0.0
    assert metric.is_successful() is False

    # No overlapping doc IDs
    test_case_no_match = LLMTestCase(
        input="test",
        actual_output="out",
        metadata={"retrieved_doc_ids": ["doc1"], "expected_doc_ids": ["docX"]},
    )
    assert metric.measure(test_case_no_match) == 0.0
