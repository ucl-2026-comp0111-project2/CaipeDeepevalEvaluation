"""Precomputed benchmark client for DeepEval evaluation.

This module provides ``PrecomputedRagClient``, a thin RAG client interface (matching ``caipe_client.py``)
that executes gold-source oracle retrieval via CAIPE RAG and handles answer generation.
All evaluation loop execution, metrics, and result writing are handled by ``deepeval_evaluator.py``.
"""

from __future__ import annotations

import time

from deepeval_eval.caipe_client import CaipeRagClient, extract_contexts_and_sources
from deepeval_eval.llm_client import OpenAICompatibleClient, make_generation_prompt, make_short_answer_prompt
from deepeval_eval.rag_client import BaseRagClient, RagQueryResult


class PrecomputedRagClient(BaseRagClient):
    """RAG client for precomputed gold-source retrieval and answer building."""


    def __init__(self, caipe_client: CaipeRagClient) -> None:
        self.caipe_client = caipe_client

    def query(
        self,
        question: str,
        reference: str,
        datasource_id: str | None,
        top_k: int = 3,
        answer_mode: str = "reference",
        benchmark: str = "enterprise",
        llm_client: OpenAICompatibleClient | None = None,
        max_context_chars: int = 12000,
    ) -> RagQueryResult:
        """Query CAIPE using oracle (question + reference) and generate/select response."""
        start_time = time.time()
        reference_query = f"{question} {reference}".strip()

        results = self.caipe_client.query_raw(reference_query, datasource_id=datasource_id, limit=top_k)
        contexts, sources = extract_contexts_and_sources(results)
        trimmed_contexts = [c[:max_context_chars] for c in contexts]

        if answer_mode == "reference":
            answer = reference
        elif benchmark == "hotpotqa":
            if llm_client is None:
                raise ValueError("llm_client is required when answer_mode != 'reference'")
            answer = str(llm_client.generate(make_short_answer_prompt(question, trimmed_contexts)))
        else:
            if llm_client is None:
                raise ValueError("llm_client is required when answer_mode != 'reference'")
            answer = str(llm_client.generate(make_generation_prompt(question, trimmed_contexts)))

        latency_sec = time.time() - start_time
        retrieved_ids = [
            str(s.get("document_id"))
            for s in sources
            if s.get("document_id") is not None
        ]

        return RagQueryResult(
            answer=answer,
            contexts=trimmed_contexts,
            sources=sources,
            retrieved_doc_ids=retrieved_ids,
            latency_sec=latency_sec,
            latency_ms=latency_sec * 1000.0,
            log_file=" ",
        )

