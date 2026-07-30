"""Base RAG Client interface and adapter classes for DeepEval evaluation.

This module provides standard abstractions for executing queries against different
RAG backends (Standard CAIPE RAG, Agentic RAG, and Precomputed RAG) and returning
a unified ``RagQueryResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class RagQueryResult:
    """Standardized result returned by RAG client implementations."""

    answer: str
    contexts: list[str]
    sources: list[dict[str, Any]]
    retrieved_doc_ids: list[str]
    latency_sec: float = 0.0
    latency_ms: float = 0.0
    log_file: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class BaseRagClient(ABC):
    """Abstract base class for all evaluation RAG clients."""

    @abstractmethod
    def query(
        self,
        question: str,
        top_k: int = 3,
        **kwargs: Any,
    ) -> RagQueryResult:
        """Execute a query against the RAG system and return a standardized result."""
        pass


class AgenticRagAdapter(BaseRagClient):
    """Adapter wrapping AgenticRetriever for agentic evaluation runs."""

    def __init__(
        self,
        supervisor_url: str | None = None,
        results_dir: Any = None,
        fail_on_error: bool | None = None,
        datasource_id: str | None = None,
        agent_id: str | None = None,
        insecure: bool | None = None,
        trace_log: bool | None = None,
        agentic_settings: Any | None = None,
    ) -> None:
        from deepeval_eval.core.config import AgenticSettings

        settings = (
            agentic_settings
            if isinstance(agentic_settings, AgenticSettings)
            else AgenticSettings()
        )

        agentic_rag_module = __import__(
            "deepeval_eval.engine.agentic_rag", fromlist=["AgenticRetriever"]
        )
        logdir = str(results_dir / "logs") if results_dir else "./logs"
        self.datasource_id = datasource_id
        self.agent_id = agent_id

        resolved_supervisor_url = supervisor_url or settings.supervisor_url
        resolved_trace_log = trace_log if trace_log is not None else settings.trace_log
        retriever_kwargs: dict[str, Any] = {
            "supervisor_url": resolved_supervisor_url,
            "timeout": settings.timeout,
            "logdir": logdir,
            "fail_on_error": (
                fail_on_error if fail_on_error is not None else settings.fail_on_error
            ),
            "datasource_id": datasource_id,
            "agent_id": agent_id,
            "trace_log": resolved_trace_log,
        }
        if insecure is not None:
            retriever_kwargs["insecure"] = insecure

        self.retriever = agentic_rag_module.AgenticRetriever(**retriever_kwargs)

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_context_chars: int = 12000,
        datasource_id: str | None = None,
        **kwargs: Any,
    ) -> RagQueryResult:
        effective_ds_id = (
            datasource_id
            if datasource_id is not None
            else kwargs.get("datasource_id", self.datasource_id)
        )
        agentic_result = self.retriever.retrieve(
            question, k=top_k, datasource_id=effective_ds_id
        )
        answer = agentic_result.answer
        trimmed_contexts = [c[:max_context_chars] for c in agentic_result.contexts]
        sources = []
        for c_idx in range(len(agentic_result.contexts)):
            doc_id = None
            if c_idx < len(self.retriever.documents_metadata):
                doc_id = self.retriever.documents_metadata[c_idx].get("doc_id")
            if not doc_id:
                doc_id = c_idx
            sources.append({"document_id": doc_id})

        latency_sec = agentic_result.latency_ms / 1000.0 if agentic_result else 0.0
        # TODO: Change the hardcoded log file path
        log_file_val = (
            f"logs/query_trace_{agentic_result.task_id}.json" if agentic_result else " "
        )
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
            latency_ms=(
                agentic_result.latency_ms if agentic_result else (latency_sec * 1000.0)
            ),
            log_file=log_file_val,
            input_tokens=agentic_result.input_tokens if agentic_result else 0,
            output_tokens=agentic_result.output_tokens if agentic_result else 0,
            total_tokens=agentic_result.total_tokens if agentic_result else 0,
        )
