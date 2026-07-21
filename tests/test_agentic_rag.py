import os
import unittest.mock as mock
import requests
import json

from deepeval_eval.agentic_rag import (
    clean_snippet_markdown,
    _extract_text_from_parts,
    _parse_rag_context_artifact,
    _dedupe_preserve_order,
    AgenticRetriever,
    AgenticRAG,
    default_agentic_rag_client,
)

# ============================================================
# 1. clean_snippet_markdown
# ============================================================

def test_clean_snippet_markdown_positive():
    # Positive: Strip prefix, bold markers, ellipses, and extra whitespace
    raw = "  **Snippet:** ...**CAIPE** uses **nomic-embed-text**...  "
    expected = "CAIPE uses nomic-embed-text"
    assert clean_snippet_markdown(raw) == expected


def test_clean_snippet_markdown_negative():
    # Negative: Handles empty string and None/empty inputs gracefully
    assert clean_snippet_markdown("") == ""
    assert clean_snippet_markdown(None) is None


# ============================================================
# 2. _extract_text_from_parts
# ============================================================

def test_extract_text_from_parts_positive():
    # Positive: Concatenate text parts
    parts = [
        {"kind": "text", "text": "Hello "},
        {"kind": "image", "url": "http://img"},
        {"kind": "text", "text": "world!"},
    ]
    assert _extract_text_from_parts(parts) == "Hello world!"


def test_extract_text_from_parts_negative():
    # Negative: Empty parts or parts without text/kind
    assert _extract_text_from_parts([]) == ""
    assert _extract_text_from_parts([{"kind": "image"}, {"text": "ignored"}]) == ""


# ============================================================
# 3. _parse_rag_context_artifact
# ============================================================

def test_parse_rag_context_artifact_positive_search():
    # Positive: parses search results shape (semantic and keyword results)
    search_data = {
        "semantic_results": [
            {"text_content": "**Snippet:** text1", "document_id": "doc1"},
            {"text_content": "text2", "metadata": {"doc_id": "doc2"}},
        ],
        "keyword_results": [
            {"text_content": "text3", "metadata": {"document_id": 3}},
        ]
    }
    raw = json.dumps(search_data)
    parsed = _parse_rag_context_artifact(raw)
    assert parsed == [
        ("text1", "doc1"),
        ("text2", "doc2"),
        ("text3", "3"),
    ]


def test_parse_rag_context_artifact_positive_fetch():
    # Positive: parses fetch_document list shape
    fetch_data = [
        {"document": {"page_content": "doc_content_1", "document_id": "doc1"}},
        {"document": {"page_content": "doc_content_2", "doc_id": "doc2"}},
    ]
    raw = json.dumps(fetch_data)
    parsed = _parse_rag_context_artifact(raw)
    assert parsed == [
        ("doc_content_1", "doc1"),
        ("doc_content_2", "doc2"),
    ]


def test_parse_rag_context_artifact_negative():
    # Negative: Handles invalid JSON and None
    assert _parse_rag_context_artifact("invalid-json") == []
    assert _parse_rag_context_artifact(None) == []


# ============================================================
# 4. _dedupe_preserve_order
# ============================================================

def test_dedupe_preserve_order_positive():
    # Positive: Dedupes by first element (content) in tuple, preserving order
    items = [
        ("a", "id1"),
        ("b", "id2"),
        ("a", "id3"),
        ("c", "id4"),
    ]
    assert _dedupe_preserve_order(items) == [
        ("a", "id1"),
        ("b", "id2"),
        ("c", "id4"),
    ]


def test_dedupe_preserve_order_negative():
    # Negative: Handles empty list
    assert _dedupe_preserve_order([]) == []


# ============================================================
# 5. AgenticRetriever
# ============================================================

def test_agentic_retriever_init_positive():
    # Positive: Initialize with custom options
    ret = AgenticRetriever(agent_api_url="http://custom", timeout=10.0, insecure=True, use_a2a=True)
    assert ret.agent_api_url == "http://custom"
    assert ret.timeout == 10.0
    assert ret.insecure is True
    assert ret.use_a2a is True


def test_agentic_retriever_init_negative():
    # Negative: Default config fallback
    with mock.patch.dict(os.environ, {"INSECURE_SSL": "false", "CAIPE_USE_A2A": "false"}):
        ret = AgenticRetriever()
        assert ret.agent_api_url is not None
        assert ret.timeout == 120.0
        assert ret.insecure is False
        assert ret.use_a2a is False  # Default to False (gateway API)

    # Env var override
    with mock.patch.dict(os.environ, {"CAIPE_USE_A2A": "true"}):
        ret_env = AgenticRetriever()
        assert ret_env.use_a2a is True


def test_agentic_retriever_fit():
    # Positive/Negative: fit does not crash and updates internal variables
    ret = AgenticRetriever()
    ret.fit(["doc1", "doc2"])
    assert ret.documents == ["doc1", "doc2"]
    assert ret.documents_metadata == [{}, {}]


@mock.patch("requests.post")
def test_agentic_retriever_call_supervisor_positive(mock_post):
    # Positive: successful supervisor call
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"result": "success"}
    mock_post.return_value = mock_resp

    ret = AgenticRetriever(agent_api_url="http://supervisor")
    res = ret._call_supervisor("hello")
    assert res == {"result": "success"}
    mock_post.assert_called_once()


@mock.patch("requests.post")
def test_agentic_retriever_call_supervisor_negative(mock_post):
    # Negative: handles timeout and exceptions
    mock_post.side_effect = requests.Timeout("Timeout error")
    ret = AgenticRetriever()
    assert ret._call_supervisor("hello") is None

    mock_post.side_effect = requests.HTTPError("HTTP error", response=mock.Mock(status_code=500))
    assert ret._call_supervisor("hello") is None

    mock_post.side_effect = Exception("Unexpected error")
    assert ret._call_supervisor("hello") is None


def test_agentic_retriever_get_top_k_positive():
    # Positive: extracts answer and contexts from supervisor response
    ret = AgenticRetriever(use_a2a=True)
    mock_body = {
        "result": {
            "artifacts": [
                {
                    "name": "rag_context",
                    "parts": [{"kind": "text", "text": json.dumps([{"document": {"page_content": "c1", "doc_id": "d1"}}])}]
                },
                {
                    "name": "final_result",
                    "parts": [{"kind": "text", "text": "This is the final answer."}]
                }
            ]
        }
    }
    
    with mock.patch.object(ret, "_call_supervisor", return_value=mock_body):
        res = ret.get_top_k("test query", k=2)
        assert res == [(0, 1.0)]
        assert ret.documents == ["c1"]
        assert ret.documents_metadata == [{"doc_id": "d1"}]
        assert ret.last_answer == "This is the final answer."
        assert ret.last_raw_response == mock_body


def test_agentic_retriever_get_top_k_negative():
    # Negative: empty/failed response
    ret = AgenticRetriever(use_a2a=True)
    with mock.patch.object(ret, "_call_supervisor", return_value=None):
        assert ret.get_top_k("test query") == []
        assert ret.documents == []
        assert ret.documents_metadata == []
        assert ret.last_answer == ""
        assert ret.last_raw_response is None


# ============================================================
# 6. AgenticRAG
# ============================================================

def test_agentic_rag_init_positive():
    # Positive: Initialize with values
    rag = AgenticRAG(agent_api_url="http://supervisor", timeout=50.0, insecure=True, use_a2a=True)
    assert rag.model_name == "agentic"
    assert rag._agentic_retriever.agent_api_url == "http://supervisor"
    assert rag._agentic_retriever.timeout == 50.0
    assert rag._agentic_retriever.insecure is True


def test_agentic_rag_init_negative():
    # Negative: Default init
    rag = AgenticRAG()
    assert rag.model_name == "agentic"
    assert rag._agentic_retriever.agent_api_url is not None


def test_dedupe_and_merge_contexts_dup() -> None:
    from deepeval_eval.agentic_rag import _dedupe_and_merge_contexts
    # Duplicate doc_id with longer content replaces shorter content + invalid items
    items = [
        None,
        "invalid_item",
        ("only_one_elem",),
        ("short", "d1"),
        ("longer content here", "d1"),
        ("no doc id content", None),
    ]
    res = _dedupe_and_merge_contexts(items)
    assert len(res) == 2
    assert res[0] == ("longer content here", "d1")
    assert res[1][1] is None


def test_agentic_rag_usage_artifacts_parsing() -> None:
    from deepeval_eval.agentic_rag import AgenticRAG
    rag = AgenticRAG(agent_api_url="http://localhost:8000")
    rag._agentic_retriever.last_answer = "Ans"
    rag._agentic_retriever.documents = ["Doc"]
    rag._agentic_retriever.documents_metadata = [{"doc_id": "d1"}]
    rag._agentic_retriever.last_raw_response = {
        "result": {
            "artifacts": [
                {"metadata": {"usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}
            ]
        }
    }
    with mock.patch.object(rag._agentic_retriever, "get_top_k", return_value=[(0, 0.9)]):
        res = rag.query("Q?", run_id="r1")
        assert res["usage"]["total_tokens"] == 15


def test_agentic_retriever_error_fallback() -> None:
    from deepeval_eval.agentic_rag import AgenticRetriever
    retriever = AgenticRetriever(agent_api_url="http://localhost:8000")
    with mock.patch.object(retriever, "get_top_k", side_effect=ValueError("Gateway connection error")):
        res = retriever.retrieve("What is X?")
        assert res.error == "Gateway connection error"
        assert res.answer == ""
        assert res.contexts == []


def test_agentic_rag_query_default_run_id() -> None:
    from deepeval_eval.agentic_rag import AgenticRAG

    rag = AgenticRAG(agent_api_url="http://localhost:8000")
    rag._agentic_retriever.last_answer = "Ans"
    rag._agentic_retriever.documents = ["Doc text"]
    rag._agentic_retriever.documents_metadata = [{"doc_id": "d1"}]
    with mock.patch.object(rag._agentic_retriever, "get_top_k", return_value=[(0, 0.9)]):
        res = rag.query("Test question", run_id=None)
        assert res["answer"] == "Ans"
        assert len(res["retrieved_docs"]) == 1


@mock.patch("deepeval_eval.agentic_rag.AgenticRAG.export_traces_to_log")
def test_agentic_rag_query_positive(mock_export):
    # Positive: successful query and usage parsing
    rag = AgenticRAG(use_a2a=True)
    ret = rag._agentic_retriever
    
    mock_body = {
        "result": {
            "metadata": {
                "usage_metadata": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                }
            },
            "artifacts": [
                {
                    "name": "rag_context",
                    "parts": [{"kind": "text", "text": json.dumps([{"document": {"page_content": "context content", "document_id": "doc1"}}])}]
                },
                {
                    "name": "final_result",
                    "parts": [{"kind": "text", "text": "the answer"}]
                }
            ]
        }
    }
    
    mock_export.return_value = "log_path.json"
    with mock.patch.object(ret, "_call_supervisor", return_value=mock_body):
        res = rag.query("question text", top_k=2)
        assert res["answer"] == "the answer"
        assert res["retrieved_doc_ids"] == ["doc1"]
        assert len(res["retrieved_docs"]) == 1
        assert res["retrieved_docs"][0]["content"] == "context content"
        assert res["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        assert res["logs"] == "log_path.json"


@mock.patch("deepeval_eval.agentic_rag.AgenticRAG.export_traces_to_log")
def test_agentic_rag_query_negative(mock_export):
    # Negative: exception during query flow
    rag = AgenticRAG(use_a2a=True)
    ret = rag._agentic_retriever
    
    mock_export.return_value = "error_log_path.json"
    with mock.patch.object(ret, "get_top_k", side_effect=Exception("Retriever failure")):
        res = rag.query("question text")
        assert "Error processing query: Retriever failure" in res["answer"]
        assert res["retrieved_docs"] == []
        assert res["retrieved_doc_ids"] == []
        assert res["usage"] is None
        assert res["logs"] == "error_log_path.json"


# ============================================================
# 7. default_agentic_rag_client
# ============================================================

def test_default_agentic_rag_client_positive():
    # Positive
    rag = default_agentic_rag_client(agent_api_url="http://sup", timeout=10.0, insecure=True, use_a2a=True)
    assert isinstance(rag, AgenticRAG)
    assert rag._agentic_retriever.agent_api_url == "http://sup"
    assert rag._agentic_retriever.timeout == 10.0


def test_default_agentic_rag_client_negative():
    # Negative: default values
    rag = default_agentic_rag_client()
    assert isinstance(rag, AgenticRAG)


# ============================================================
# 8. _query_gateway
# ============================================================

@mock.patch("httpx.stream")
@mock.patch("httpx.post")
def test_agentic_retriever_query_gateway_positive(mock_post, mock_stream):
    # Setup mock for conv_url
    mock_conv_resp = mock.Mock()
    mock_conv_resp.status_code = 201
    mock_conv_resp.json.return_value = {
        "data": {
            "conversation": {
                "_id": "conv-123"
            }
        }
    }
    mock_post.return_value = mock_conv_resp

    # Setup mock for stream_url
    mock_stream_resp = mock.MagicMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.iter_lines.return_value = [
        "event: content",
        'data: {"text": "thinking..."}',
        "event: tool_end",
        'data: {"result": "{\\"semantic_results\\": [{\\"text_content\\": \\"doc content\\", \\"document_id\\": \\"doc-99\\"}]}"}',
        "event: content",
        'data: {"text": "hello"}',
        "event: done",
        ""
    ]
    mock_stream.return_value.__enter__.return_value = mock_stream_resp

    ret = AgenticRetriever(agent_api_url="https://gateway.service", use_a2a=False)
    res = ret._query_gateway("test question", k=1)

    assert res == [("doc content", "doc-99")]
    assert ret.last_answer == "hello"


@mock.patch("httpx.post")
def test_agentic_retriever_query_gateway_negative(mock_post):
    mock_post.side_effect = Exception("Connection error")

    ret = AgenticRetriever(agent_api_url="https://gateway.service", use_a2a=False)
    res = ret._query_gateway("test question", k=1)

    assert res == []
    assert ret.last_answer == ""


# ============================================================
# 9. Trace Logging Tests
# ============================================================

@mock.patch("httpx.stream")
@mock.patch("httpx.post")
def test_agentic_retriever_trace_log_positive(mock_post, mock_stream, tmp_path):
    # Setup mock for conv_url
    mock_conv_resp = mock.Mock()
    mock_conv_resp.status_code = 201
    mock_conv_resp.json.return_value = {
        "data": {
            "conversation": {
                "_id": "conv-123"
            }
        }
    }
    mock_post.return_value = mock_conv_resp

    # Setup mock for stream_url
    mock_stream_resp = mock.MagicMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.iter_lines.return_value = [
        "event: content",
        'data: {"text": "hello"}',
        "event: tool_end",
        'data: {"result": "{\\"semantic_results\\": [{\\"text_content\\": \\"doc content\\", \\"document_id\\": \\"doc-99\\"}]}"}',
    ]
    mock_stream.return_value.__enter__.return_value = mock_stream_resp

    logdir = str(tmp_path / "logs")
    ret = AgenticRetriever(agent_api_url="https://gateway.service", use_a2a=False, trace_log=True, logdir=logdir)
    res = ret._query_gateway("test question", k=1, run_id="test_run_123")

    assert res == [("doc content", "doc-99")]
    # Check log file was created and contains expected content
    log_file_path = os.path.join(logdir, "agentic_run_test_run_123.log")
    assert os.path.exists(log_file_path)
    with open(log_file_path, "r") as f:
        content = f.read()
    assert "[event: content]" in content
    assert '"text": "hello"' in content


@mock.patch("httpx.stream")
@mock.patch("httpx.post")
def test_agentic_retriever_trace_log_negative(mock_post, mock_stream, tmp_path):
    # Setup mock for conv_url
    mock_conv_resp = mock.Mock()
    mock_conv_resp.status_code = 201
    mock_conv_resp.json.return_value = {
        "data": {
            "conversation": {
                "_id": "conv-123"
            }
        }
    }
    mock_post.return_value = mock_conv_resp

    # Setup mock for stream_url
    mock_stream_resp = mock.MagicMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.iter_lines.return_value = [
        "event: content",
        'data: {"text": "hello"}',
    ]
    mock_stream.return_value.__enter__.return_value = mock_stream_resp

    logdir = str(tmp_path / "logs")
    # trace_log explicitly False
    ret = AgenticRetriever(agent_api_url="https://gateway.service", use_a2a=False, trace_log=False, logdir=logdir)
    res = ret._query_gateway("test question", k=1, run_id="test_run_123")

    assert res == []
    # Log file should NOT exist
    log_file_path = os.path.join(logdir, "agentic_run_test_run_123.log")
    assert not os.path.exists(log_file_path)


# ============================================================
# 10. Retry Logic & fail_on_error Tests
# ============================================================

@mock.patch("httpx.post")
def test_agentic_retriever_fail_on_error(mock_post):
    mock_post.side_effect = Exception("Persistent failure")
    ret = AgenticRetriever(
        agent_api_url="https://gateway.service", use_a2a=False, fail_on_error=True
    )
    import pytest
    with pytest.raises(Exception, match="Persistent failure"):
        ret.retrieve("test question")


@mock.patch("httpx.stream")
@mock.patch("httpx.post")
@mock.patch("time.sleep")  # avoid actual sleeping during tests
@mock.patch.object(AgenticRetriever, "_get_oidc_token", return_value="fake-token")
def test_agentic_retriever_retry_success(mock_get_token, mock_sleep, mock_post, mock_stream):
    # First attempt: httpx.post raises an exception (e.g. timeout)
    # Second attempt: succeeds
    mock_conv_resp = mock.Mock()
    mock_conv_resp.status_code = 201
    mock_conv_resp.json.return_value = {
        "data": {
            "conversation": {
                "_id": "conv-456"
            }
        }
    }

    mock_post.side_effect = [Exception("Transient timeout"), mock_conv_resp]

    # Setup mock for stream_url on the second attempt
    mock_stream_resp = mock.MagicMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.iter_lines.return_value = [
        "event: content",
        'data: {"text": "hello on retry"}',
    ]
    mock_stream.return_value.__enter__.return_value = mock_stream_resp

    ret = AgenticRetriever(agent_api_url="https://gateway.service", use_a2a=False)
    res = ret._query_gateway("test question", k=1)

    assert res == []
    assert ret.last_answer == "hello on retry"
    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1

