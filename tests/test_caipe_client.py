from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import requests

from deepeval_eval.caipe_client import (
    CaipeRagClient,
    build_caipe_client,
    check_response,
    extract_contexts_and_sources,
)


def test_check_response_positive() -> None:
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = True
    result = check_response(mock_resp)
    assert result == mock_resp


def test_check_response_negative() -> None:
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = False
    mock_resp.status_code = 404
    mock_resp.request = MagicMock()
    mock_resp.request.method = "GET"
    mock_resp.request.url = "http://example.com"
    mock_resp.text = "Not Found"

    with pytest.raises(RuntimeError, match="HTTP 404"):
        check_response(mock_resp)


def test_extract_contexts_and_sources_positive() -> None:
    raw_results = [
        {
            "document": {
                "page_content": "Paris is the capital of France.",
                "metadata": {
                    "document_id": "doc100",
                    "title": "France Info",
                    "metadata": {"source_type": "pdf"},
                },
            },
            "score": 0.95,
        }
    ]
    contexts, sources = extract_contexts_and_sources(raw_results)
    assert contexts == ["Paris is the capital of France."]
    assert sources == [
        {
            "document_id": "doc100",
            "title": "France Info",
            "source_type": "pdf",
            "score": 0.95,
        }
    ]


def test_extract_contexts_and_sources_negative() -> None:
    # Empty content or invalid structures
    raw_results = [
        {"document": {"metadata": {}}},
        {"invalid": "structure"},
        {},
    ]
    contexts, sources = extract_contexts_and_sources(raw_results)
    assert contexts == []
    assert sources == []


def test_caipe_rag_client_refresh_access_token_positive() -> None:
    client = CaipeRagClient(
        base_url="https://caipe.homelab/api",
        keycloak_url="https://keycloak.caipe.homelab/token",
        client_id="test_client",
        client_secret="test_secret",
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"access_token": "new_access_token", "expires_in": 600}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        client.refresh_access_token()
        assert client.session.headers["Authorization"] == "Bearer new_access_token"


def test_caipe_rag_client_refresh_access_token_negative() -> None:
    client = CaipeRagClient(base_url="https://caipe.homelab/api")
    # Refreshing without keycloak_url should be a no-op
    client.refresh_access_token()


def test_caipe_rag_client_query_raw_positive() -> None:
    client = CaipeRagClient(base_url="https://caipe.homelab/api", token="static_token")
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = [{"document": {"page_content": "Content"}, "score": 0.9}]

    with patch.object(client.session, "post", return_value=mock_resp):
        res = client.query_raw("query text", datasource_id="ds1", limit=3)
        assert len(res) == 1
        assert res[0]["score"] == 0.9


def test_caipe_rag_client_query_positive() -> None:
    client = CaipeRagClient(base_url="https://caipe.homelab/api", token="static_token")
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Generated answer"

    raw_results = [
        {
            "document": {
                "page_content": "Context info",
                "metadata": {"document_id": "doc1"},
            }
        }
    ]

    with patch.object(client, "query_raw", return_value=raw_results):
        res = client.query("What is X?", llm_client=mock_llm)
        assert res.answer == "Generated answer"
        assert res.contexts == ["Context info"]
        assert res.retrieved_doc_ids == ["doc1"]


def test_caipe_rag_client_query_negative() -> None:
    client = CaipeRagClient(base_url="https://caipe.homelab/api", token="static_token")

    with patch.object(client, "query_raw", return_value=[]):
        with pytest.raises(ValueError, match="llm_client is required"):
            client.query("What is X?", llm_client=None)


def test_caipe_rag_client_ingest_endpoints(tmp_path) -> None:
    client = CaipeRagClient(base_url="https://caipe.homelab/api", token="static_token")

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200

    # test register_ingestor
    mock_resp.json.return_value = {"ingestor_id": "ing123", "max_documents_per_ingest": 100}
    with patch.object(client.session, "post", return_value=mock_resp):
        ing_id, max_docs = client.register_ingestor("type", "name", "desc")
        assert ing_id == "ing123"
        assert max_docs == 100

    # test reset_datasource
    mock_resp_del = MagicMock()
    mock_resp_del.status_code = 204
    with patch.object(client.session, "delete", return_value=mock_resp_del):
        client.reset_datasource("ds1")

    # test upsert_datasource
    with patch.object(client.session, "post", return_value=mock_resp):
        client.upsert_datasource("ds1", "Name", "ing123", "desc", "slack")

    # test open_job and close_job
    mock_resp.json.return_value = {"job_id": "job789"}
    with patch.object(client.session, "post", return_value=mock_resp):
        job_id = client.open_job("ds1", 10, "start")
        assert job_id == "job789"

    with patch.object(client.session, "patch", return_value=mock_resp):
        client.close_job("job789", "finish")

    # test ingest_batch
    docs = [{"text": "doc1"}]
    with patch.object(client.session, "post", return_value=mock_resp):
        client.ingest_batch(docs, "ing123", "ds1", "job789")


def test_caipe_rag_client_hotpotqa_query() -> None:
    client = CaipeRagClient(base_url="https://caipe.homelab/api", token="static_token")
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Hotpot Answer"

    raw_results = [{"document": {"page_content": "Hotpot context", "metadata": {"document_id": "hp1"}}}]
    with patch.object(client, "query_raw", return_value=raw_results):
        res = client.query("What is Y?", benchmark="hotpotqa", llm_client=mock_llm)
        assert res.answer == "Hotpot Answer"
        assert res.retrieved_doc_ids == ["hp1"]


def test_build_caipe_client_positive() -> None:
    env_values = {
        "CAIPE_BASE_URL": "http://localhost:8080",
        "CAIPE_AUTH_TOKEN": "token123",
        "INSECURE_SSL": "true",
    }
    client = build_caipe_client(env_values)
    assert client.base_url == "http://localhost:8080"
    assert client.session.verify is False
