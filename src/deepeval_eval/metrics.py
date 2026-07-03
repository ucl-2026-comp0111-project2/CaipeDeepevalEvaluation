from __future__ import annotations

import re
import string
from typing import Any


# Keep metric construction in one place so both benchmark pipelines are judged
# with the same DeepEval settings.
def build_metrics(judge_model: Any) -> list[Any]:
    from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, FaithfulnessMetric
    common = {'threshold': 0.5, 'model': judge_model, 'include_reason': True, 'async_mode': False}
    return [
        AnswerRelevancyMetric(**common),
        FaithfulnessMetric(**common),
        ContextualRelevancyMetric(**common),
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
    ]


def doc_id_scores(retrieved: list[dict[str, Any]], expected_doc_ids: list[str]) -> tuple[float, float]:
    retrieved_ids = {str(item.get('document_id')) for item in retrieved if item.get('document_id') is not None}
    expected = {str(doc_id) for doc_id in expected_doc_ids}
    if not expected:
        return 0.0, 0.0
    hits = retrieved_ids & expected
    recall = len(hits) / len(expected)
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
    return recall, precision


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punc = ''.join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r'\b(a|an|the)\b', ' ', no_punc)
    return ' '.join(no_articles.split())


def answer_scores(answer: str, reference: str) -> tuple[float, float]:
    answer_norm = normalize_answer(answer)
    ref_norm = normalize_answer(reference)
    if not ref_norm:
        return 0.0, 0.0
    return (1.0 if answer_norm == ref_norm else 0.0, 1.0 if ref_norm in answer_norm else 0.0)
