from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from deepeval_eval.rag_client import AgenticRagAdapter, RagQueryResult


def test_rag_query_result_dataclass_positive() -> None:
    res = RagQueryResult(
        answer="Paris",
        contexts=["Context A"],
        sources=[{"document_id": "doc1"}],
        retrieved_doc_ids=["doc1"],
        latency_sec=0.5,
        latency_ms=500.0,
    )
    assert res.answer == "Paris"
    assert res.contexts == ["Context A"]
    assert res.retrieved_doc_ids == ["doc1"]
    assert res.latency_sec == 0.5
    assert res.latency_ms == 500.0


def test_rag_query_result_dataclass_negative() -> None:
    res = RagQueryResult(
        answer="",
        contexts=[],
        sources=[],
        retrieved_doc_ids=[],
    )
    assert res.answer == ""
    assert res.latency_sec == 0.0
    assert res.log_file == " "


def test_agentic_rag_adapter_positive(tmp_path: Path) -> None:
    mock_retriever = MagicMock()
    mock_agentic_result = MagicMock()
    mock_agentic_result.answer = "Agentic Answer"
    mock_agentic_result.contexts = ["Context 1 text long enough for testing"]
    mock_agentic_result.latency_ms = 1200.0
    mock_agentic_result.task_id = "task-123"
    mock_agentic_result.input_tokens = 100
    mock_agentic_result.output_tokens = 50
    mock_agentic_result.total_tokens = 150

    mock_retriever.retrieve.return_value = mock_agentic_result
    mock_retriever.documents_metadata = [{"doc_id": "doc_meta_1"}]

    with patch("deepeval_eval.agentic_rag.AgenticRetriever", return_value=mock_retriever):
        adapter = AgenticRagAdapter(supervisor_url="http://localhost:8000", results_dir=tmp_path)
        result = adapter.query("What is the prompt?", top_k=2)

        assert result.answer == "Agentic Answer"
        assert result.contexts == ["Context 1 text long enough for testing"]
        assert result.retrieved_doc_ids == ["doc_meta_1"]
        assert result.latency_ms == 1200.0
        assert result.latency_sec == 1.2
        assert "task-123" in result.log_file


def test_agentic_rag_adapter_negative(tmp_path: Path) -> None:
    mock_retriever = MagicMock()
    mock_agentic_result = MagicMock()
    mock_agentic_result.answer = ""
    mock_agentic_result.contexts = []
    mock_agentic_result.latency_ms = 0.0
    mock_agentic_result.task_id = "empty"
    mock_agentic_result.input_tokens = 0
    mock_agentic_result.output_tokens = 0
    mock_agentic_result.total_tokens = 0

    mock_retriever.retrieve.return_value = mock_agentic_result
    mock_retriever.documents_metadata = []

    with patch("deepeval_eval.agentic_rag.AgenticRetriever", return_value=mock_retriever):
        adapter = AgenticRagAdapter(supervisor_url="http://localhost:8000", results_dir=tmp_path)
        result = adapter.query("Missing query")

        assert result.answer == ""
        assert result.contexts == []
        assert result.retrieved_doc_ids == []
