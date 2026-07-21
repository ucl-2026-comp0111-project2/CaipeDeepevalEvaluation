from __future__ import annotations

import math
import re
import string
from typing import Any, Optional
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)


# Keep metric construction in one place so both benchmark pipelines are judged
# with the same DeepEval settings.
def build_metrics(judge_model: Any) -> list[Any]:
    common = {
        "threshold": 0.5,
        "model": judge_model,
        "include_reason": True,
        "async_mode": False,
    }

    return [
        AnswerRelevancyMetric(**common),
        FaithfulnessMetric(**common),
        AnswerCorrectnessMetric(**common),
        ContextualRelevancyMetric(**common),
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
        MRRMetric(**common),
        NDCGAtKMetric(**common),
    ]


def doc_id_scores(
    retrieved: list[dict[str, Any]], expected_doc_ids: list[str]
) -> tuple[float, float]:
    retrieved_ids = {
        str(item.get("document_id"))
        for item in retrieved
        if item.get("document_id") is not None
    }
    expected = {str(doc_id) for doc_id in expected_doc_ids}
    if not expected:
        return 0.0, 0.0
    hits = retrieved_ids & expected
    recall = len(hits) / len(expected)
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
    return recall, precision


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def answer_scores(answer: str, reference: str) -> tuple[float, float]:
    answer_norm = normalize_answer(answer)
    ref_norm = normalize_answer(reference)
    if not ref_norm:
        return 0.0, 0.0
    return (
        1.0 if answer_norm == ref_norm else 0.0,
        1.0 if ref_norm in answer_norm else 0.0,
    )


class AnswerCorrectnessMetric(BaseMetric):
    """
    Answer Correctness Metric wrapping DeepEval's GEval framework.
    Evaluates generated output factual alignment against the ground truth reference.
    """
    def __init__(self, name: str = "AnswerCorrectnessMetric", model: Any = None, threshold: float = 0.5, **kwargs):
        self.name = name
        
        self.geval_judge = GEval(
            name=name,
            model=model,
            threshold=threshold,
            verbose_mode=kwargs.get("verbose_mode", False),
            async_mode=kwargs.get("async_mode", False),
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT, 
                SingleTurnParams.EXPECTED_OUTPUT
            ],
            evaluation_steps=[
                "Compare the actual output directly with the expected output to verify factual accuracy.",
                "Check if all elements mentioned in the expected output are present and correctly represented in the actual output.",
                "Assess if there are any discrepancies in details, values, or information between the actual and expected outputs."
            ]
        )
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        # Execute the underlying G-Eval logic safely
        self.score = self.geval_judge.measure(test_case)
        self.success = self.geval_judge.is_successful()
        self.reason = self.geval_judge.reason
        return self.score

    def get_reason(self) -> str:
        return self.reason or ""

    def is_successful(self) -> bool:
        return bool(self.success)
    

class MRRMetric(BaseMetric):
    """
    Mean Reciprocal Rank (MRR) for DeepEval retrieval evaluation.
    Calculates 1.0 / rank of the first matching ground-truth document ID.
    """
    # Added **kwargs to gracefully handle standard judge/framework arguments
    def __init__(self, name: str = "MRR", **kwargs):
        self.name = name
        # ensure score is always a float to avoid None comparisons later
        self.score: Optional[float] = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        # Extract fields from the built-in metadata parameter contract
        metadata = test_case.metadata or {}
        
        retrieved_ids = [str(d) for d in metadata.get("retrieved_doc_ids", [])]
        expected_ids = set(str(d) for d in metadata.get("expected_doc_ids", []))

        if not expected_ids or not retrieved_ids:
            self.score = 0.0
            return self.score

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                self.score = 1.0 / rank
                return self.score

        self.score = 0.0
        return self.score

    def get_reason(self) -> str:
        return f"Deterministic MRR ranking quality score: {self.score:.4f}"

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= 0.5


class NDCGAtKMetric(BaseMetric):
    """
    Normalized Discounted Cumulative Gain at k (nDCG@k) for DeepEval.
    Evaluates positional weighting distributions for multi-document retrieval.
    """
    # Added **kwargs to gracefully handle standard judge/framework arguments
    def __init__(self, name: str = "nDCG@k", k: int = 5, **kwargs):
        self.name = name
        self.k = k
        self.score: Optional[float] = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        metadata = test_case.metadata or {}
        retrieved_ids = [str(d) for d in metadata.get("retrieved_doc_ids", [])]
        expected_ids = set(str(d) for d in metadata.get("expected_doc_ids", []))

        if not expected_ids or not retrieved_ids:
            self.score = 0.0
            return self.score

        # Truncate evaluation list to k cutoff boundaries
        retrieved_k = retrieved_ids[:self.k]
        
        # Calculate Discounted Cumulative Gain (DCG@k)
        dcg = sum((1.0 / math.log2(i + 2)) for i, doc_id in enumerate(retrieved_k) if doc_id in expected_ids)
        if math.isclose(dcg, 0.0):
            self.score = 0.0
            return self.score

        # Calculate Ideal Discounted Cumulative Gain (IDCG@k)
        ideal_hits = min(len(expected_ids), self.k)
        idcg = sum((1.0 / math.log2(i + 2)) for i in range(ideal_hits))

        self.score = dcg / idcg if idcg > 0.0 else 0.0
        return self.score

    def get_reason(self) -> str:
        score = self.score if self.score is not None else 0.0
        return f"Deterministic nDCG@{self.k} ranking quality score: {score:.4f}"

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= 0.5