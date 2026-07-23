from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from deepeval_eval.api import (
    JobManager,
    JobStatusEnum,
    LocalCacheManager,
    app,
    compute_eval_hash,
    execute_evaluation_job,
    run_server,
    sanitize_config_args,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests for Helper Functions & Cache Management
# ---------------------------------------------------------------------------


def test_sanitize_config_args_positive():
    """Verify sensitive keys and null values are omitted from sanitized configuration output."""
    raw_config = {
        "dataset_name": "enterprise",
        "llm_api_key": "secret-12345",
        "auth_token": "bearer-abc",
        "prompt_style": None,
        "max_items": 10,
    }
    sanitized = sanitize_config_args(raw_config)
    assert "llm_api_key" not in sanitized
    assert "auth_token" not in sanitized
    assert "prompt_style" not in sanitized
    assert sanitized["dataset_name"] == "enterprise"
    assert sanitized["max_items"] == 10


def test_sanitize_config_args_negative():
    """Verify empty dictionary or dict with all sensitive/null keys returns empty dict."""
    raw_config = {
        "llm_api_key": "secret",
        "db_connection_string": "postgres://...",
        "auth_token": None,
    }
    sanitized = sanitize_config_args(raw_config)
    assert sanitized == {}


def test_compute_eval_hash_positive():
    """Verify compute_eval_hash produces deterministic fingerprint."""
    config1 = {"dataset_name": "enterprise", "top_k": 3, "force_rerun": False}
    config2 = {"dataset_name": "enterprise", "top_k": 3, "force_rerun": True}
    h1 = compute_eval_hash(config1)
    h2 = compute_eval_hash(config2)
    assert h1 == h2  # force_rerun should be ignored in hash
    assert len(h1) == 16


def test_compute_eval_hash_negative():
    """Verify compute_eval_hash produces different fingerprint for different inputs."""
    config1 = {"dataset_name": "enterprise", "top_k": 3}
    config2 = {"dataset_name": "hotpotqa", "top_k": 3}
    h1 = compute_eval_hash(config1)
    h2 = compute_eval_hash(config2)
    assert h1 != h2


def test_local_cache_manager_positive(tmp_path: Path):
    """Verify LocalCacheManager set and get within TTL."""
    cm = LocalCacheManager(cache_dir=tmp_path)
    eval_hash = "test_hash_123"
    payload = {"job_id": "job_1", "summary": {"total": 5}}

    cm.set(eval_hash, payload)
    retrieved = cm.get(eval_hash)
    assert retrieved is not None
    assert retrieved["job_id"] == "job_1"


def test_local_cache_manager_negative_expired(tmp_path: Path):
    """Verify LocalCacheManager purges and returns None for expired cache."""
    cm = LocalCacheManager(cache_dir=tmp_path)
    eval_hash = "test_hash_expired"
    payload = {"job_id": "job_2", "timestamp": time.time() - 90000}  # Older than 24h

    cm.set(eval_hash, payload)
    # Manually overwrite file to force old timestamp
    cache_file = tmp_path / f"{eval_hash}.json"
    cache_file.write_text(json.dumps(payload))

    assert cm.get(eval_hash) is None
    purged = cm.purge_expired()
    assert purged >= 0


# ---------------------------------------------------------------------------
# Unit Tests for JobManager
# ---------------------------------------------------------------------------


def test_job_manager_create_and_list(tmp_path: Path):
    """Verify JobManager job creation, retrieval, listing, and deduplication."""
    cm = LocalCacheManager(cache_dir=tmp_path)
    jm = JobManager(cache_manager=cm)

    config = {"dataset_name": "enterprise", "top_k": 3}
    job1 = jm.create_job("hash1", config)
    assert job1["job_id"] is not None
    assert job1["status"] == JobStatusEnum.PENDING
    assert job1["cached"] is False

    # Simulate completed job in cache
    cm.set(
        "hash1",
        {"job_id": job1["job_id"], "status": "completed", "summary": {"score": 1.0}},
    )

    # Create job with same hash -> should return cached job
    job2 = jm.create_job("hash1", config, force_rerun=False)
    assert job2["cached"] is True

    # Force rerun -> should create new non-cached job
    job3 = jm.create_job("hash1", config, force_rerun=True)
    assert job3["cached"] is False
    assert job3["job_id"] != job1["job_id"]

    all_jobs = jm.list_jobs()
    assert len(all_jobs) >= 2


def test_job_manager_get_negative(tmp_path: Path):
    """Verify JobManager returns None for unknown job_id."""
    cm = LocalCacheManager(cache_dir=tmp_path)
    jm = JobManager(cache_manager=cm)
    assert jm.get_job("non_existent_id") is None


# ---------------------------------------------------------------------------
# Unit Tests for Background Job Execution Handler
# ---------------------------------------------------------------------------


@patch("deepeval_eval.api.run_evaluation")
@patch("deepeval_eval.api._build_rag_client")
def test_execute_evaluation_job_positive(mock_build_rag, mock_run_eval, tmp_path: Path):
    """Verify execute_evaluation_job updates job status to COMPLETED."""
    mock_run_eval.return_value = [{"question": "q1", "metrics": {}}]

    from deepeval_eval.api import EvaluationRequest, job_manager

    eval_hash = "exec_hash_pos"
    req = EvaluationRequest(dataset_name="enterprise", max_items=1)
    job = job_manager.create_job(eval_hash, req.model_dump(), force_rerun=True)

    execute_evaluation_job(job["job_id"], req)

    updated_job = job_manager.get_job(job["job_id"])
    assert updated_job["status"] == JobStatusEnum.COMPLETED
    results = job_manager.get_job_results_payload(job["job_id"])
    assert len(results) == 1


@patch("deepeval_eval.api.run_evaluation", side_effect=ValueError("Eval engine error"))
@patch("deepeval_eval.api._build_rag_client")
def test_execute_evaluation_job_negative(mock_build_rag, mock_run_eval):
    """Verify execute_evaluation_job handles failure gracefully."""
    from deepeval_eval.api import EvaluationRequest, job_manager

    eval_hash = "exec_hash_neg"
    req = EvaluationRequest(dataset_name="enterprise")
    job = job_manager.create_job(eval_hash, req.model_dump(), force_rerun=True)

    execute_evaluation_job(job["job_id"], req)

    updated_job = job_manager.get_job(job["job_id"])
    assert updated_job["status"] == JobStatusEnum.FAILED
    assert "Eval engine error" in updated_job["error"]


# ---------------------------------------------------------------------------
# Endpoint Tests using TestClient
# ---------------------------------------------------------------------------


def test_root_and_health_endpoints_positive():
    """Verify GET / and GET /health return 200 OK."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


def test_endpoint_authentication_protection(monkeypatch):
    """Verify endpoints enforce authentication when unauthenticated access is disabled."""
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED_ACCESS", "false")
    monkeypatch.delenv("CAIPE_UNSAFE_RBAC_BYPASS", raising=False)

    # Protected endpoint without token -> 401 Unauthorized
    res_root = client.get("/")
    assert res_root.status_code == 401

    res_jobs = client.post("/eval/jobs", json={"dataset_name": "enterprise"})
    assert res_jobs.status_code == 401

    # Health check endpoint remains unauthenticated -> 200 OK
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Protected endpoint with static API key header -> 200 OK
    monkeypatch.setenv("DEEPEVAL_API_KEY", "test_key_123")
    headers = {"Authorization": "Bearer test_key_123"}
    res_root_auth = client.get("/", headers=headers)
    assert res_root_auth.status_code == 200


def test_swagger_docs_accessible():
    """Verify Swagger UI docs endpoint at /docs returns 200 OK."""
    res = client.get("/docs")
    assert res.status_code == 200
    assert "swagger-ui" in res.text.lower() or "html" in res.text.lower()


@patch("deepeval_eval.api.execute_evaluation_job")
def test_submit_eval_job_positive(mock_execute):
    """Verify POST /eval/jobs returns 202 Accepted and launches background task."""
    payload = {
        "dataset_name": "enterprise",
        "answer_mode": "reference",
        "top_k": 3,
        "max_items": 2,
        "force_rerun": True,
    }
    res = client.post("/eval/jobs", json=payload)
    assert res.status_code == 202
    data = res.json()
    assert "job_id" in data
    assert data["status"] in ("pending", "running", "completed")


def test_submit_eval_job_negative_invalid_body():
    """Verify POST /eval/jobs returns 422 Unprocessable Entity for invalid field types."""
    payload = {
        "max_items": -5,  # Constraint: ge=1
    }
    res = client.post("/eval/jobs", json=payload)
    assert res.status_code == 422


@patch("deepeval_eval.api.execute_evaluation_job")
def test_submit_eval_job_with_upload_positive(mock_execute):
    """Verify POST /eval/jobs/upload accepts multipart dataset file upload."""
    file_content = b'[{"question": "What is CAIPE?"}]'
    files = {"file": ("test_questions.json", file_content, "application/json")}

    res = client.post(
        "/eval/jobs/upload?dataset_name=custom&top_k=2&force_rerun=true",
        files=files,
    )
    assert res.status_code == 202
    data = res.json()
    assert "job_id" in data


def test_submit_eval_job_with_upload_negative_empty_file():
    """Verify POST /eval/jobs/upload rejects empty files with 400 Bad Request."""
    files = {"file": ("empty.json", b"", "application/json")}

    res = client.post("/eval/jobs/upload", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_get_job_status_and_list():
    """Verify GET /jobs and GET /jobs/{job_id} endpoints."""
    # Submit job first
    res_sub = client.post(
        "/eval/jobs", json={"dataset_name": "enterprise", "force_rerun": True}
    )
    job_id = res_sub.json()["job_id"]

    res_get = client.get(f"/jobs/{job_id}")
    assert res_get.status_code == 200
    assert res_get.json()["job_id"] == job_id

    res_list = client.get("/jobs")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1


def test_get_job_status_negative_not_found():
    """Verify GET /jobs/{job_id} returns 404 for unknown job ID."""
    res = client.get("/jobs/unknown_job_id_999")
    assert res.status_code == 404


def test_get_job_results_negative_pending():
    """Verify GET /jobs/{job_id}/results returns 400 if job is not completed yet."""
    from deepeval_eval.api import job_manager

    job = job_manager.create_job(
        "hash_pending", {"dataset_name": "test"}, force_rerun=True
    )
    res = client.get(f"/jobs/{job['job_id']}/results")
    assert res.status_code == 400


def test_get_job_results_positive_completed():
    """Verify GET /jobs/{job_id}/results returns results for completed job."""
    from deepeval_eval.api import JobStatusEnum, cache_manager, job_manager

    job = job_manager.create_job(
        "hash_completed", {"dataset_name": "test"}, force_rerun=True
    )
    job["status"] = JobStatusEnum.COMPLETED
    cache_manager.save_job_payload(job["job_id"], [{"question": "q1"}])

    res = client.get(f"/jobs/{job['job_id']}/results")
    assert res.status_code == 200
    assert len(res.json()["results"]) == 1


@patch("deepeval_eval.api.DatabaseResultSink")
def test_save_job_results_to_db_positive(mock_sink_cls):
    """Verify POST /jobs/{job_id}/save-db calls PostgresResultSink.save."""
    from deepeval_eval.api import JobStatusEnum, cache_manager, job_manager

    mock_sink_instance = MagicMock()
    mock_sink_cls.return_value = mock_sink_instance

    job = job_manager.create_job(
        "hash_db_save", {"dataset_name": "test"}, force_rerun=True
    )
    job["status"] = JobStatusEnum.COMPLETED
    cache_manager.save_job_payload(job["job_id"], [{"question": "q1"}])

    res = client.post(f"/jobs/{job['job_id']}/save-db")
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    mock_sink_instance.save.assert_called_once()


def test_save_job_results_to_db_negative_not_completed():
    """Verify POST /jobs/{job_id}/save-db returns 400 for incomplete jobs."""
    from deepeval_eval.api import job_manager

    job = job_manager.create_job(
        "hash_db_save_neg", {"dataset_name": "test"}, force_rerun=True
    )

    res = client.post(f"/jobs/{job['job_id']}/save-db")
    assert res.status_code == 400


def test_query_db_evaluation_runs_positive():
    """Verify GET /results/db queries PostgreSQL database via DatabaseResultSink."""
    mock_psycopg2 = MagicMock()
    mock_extras = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchall.return_value = [
        {
            "run_id": "run_1",
            "batch_id": "batch_1",
            "config_name": "enterprise",
            "loaded_at": "2026-07-22",
            "config_json": {},
        }
    ]

    with patch.dict(
        "sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_extras}
    ):
        res = client.get("/results/db?limit=5")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 1
        assert data["runs"][0]["run_id"] == "run_1"


def test_query_db_evaluation_runs_negative():
    """Verify GET /results/db handles connection failures cleanly with 500 error."""
    mock_psycopg2 = MagicMock()
    mock_psycopg2.connect.side_effect = Exception("DB Connection Error")

    with patch.dict(
        "sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()}
    ):
        res = client.get("/results/db")
        assert res.status_code == 500
        assert "DB Connection Error" in res.json()["detail"]


@patch("uvicorn.run")
def test_run_server_positive(mock_uvicorn_run):
    """Verify run_server calls uvicorn.run."""
    run_server(host="127.0.0.1", port=9000)
    mock_uvicorn_run.assert_called_once_with(
        "deepeval_eval.api:app", host="127.0.0.1", port=9000, reload=False
    )


def test_purge_expired_corrupted_and_unwriteable_files(tmp_path: Path):
    """Verify purge_expired handles invalid JSON files and write failures."""
    cm = LocalCacheManager(cache_dir=tmp_path)

    # Invalid JSON file
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("invalid_json_content")

    purged = cm.purge_expired()
    assert purged == 1

    # Unreadable cache test
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_bytes(b"\x80abc")
    assert cm.get("unreadable") is None


def test_get_job_results_additional_negative_cases():
    """Verify get_job_results for 404 not found and 500 failed jobs."""
    from deepeval_eval.api import JobStatusEnum, job_manager

    # Job not found
    res1 = client.get("/jobs/non_existent_9999/results")
    assert res1.status_code == 404

    # Job failed
    failed_job = job_manager.create_job(
        "hash_failed", {"dataset_name": "test"}, force_rerun=True
    )
    failed_job["status"] = JobStatusEnum.FAILED
    failed_job["error"] = "Custom error message"

    res2 = client.get(f"/jobs/{failed_job['job_id']}/results")
    assert res2.status_code == 500
    assert "Custom error message" in res2.json()["detail"]


@patch("deepeval_eval.api.DatabaseResultSink")
def test_save_job_results_to_db_additional_negative_cases(mock_sink_cls):
    """Verify save_job_results_to_db for not found, empty results, and sink errors."""
    from deepeval_eval.api import JobStatusEnum, cache_manager, job_manager

    # 404 Job not found
    res1 = client.post("/jobs/non_existent_8888/save-db")
    assert res1.status_code == 404

    # Empty results list
    empty_job = job_manager.create_job(
        "hash_empty_results", {"dataset_name": "test"}, force_rerun=True
    )
    empty_job["status"] = JobStatusEnum.COMPLETED
    empty_job["results"] = []

    res2 = client.post(f"/jobs/{empty_job['job_id']}/save-db")
    assert res2.status_code == 400
    assert "No evaluation results" in res2.json()["detail"]

    # Exception during sink.save
    mock_instance = MagicMock()
    mock_instance.save.side_effect = Exception("Sink write error")
    mock_sink_cls.return_value = mock_instance

    valid_job = job_manager.create_job(
        "hash_sink_err", {"dataset_name": "test"}, force_rerun=True
    )
    valid_job["status"] = JobStatusEnum.COMPLETED
    cache_manager.save_job_payload(valid_job["job_id"], [{"question": "q"}])

    res3 = client.post(f"/jobs/{valid_job['job_id']}/save-db")
    assert res3.status_code == 500
    assert "Sink write error" in res3.json()["detail"]


def test_submit_eval_job_with_upload_cached_positive(tmp_path: Path):
    """Verify upload endpoint returns cached job response when hash matches."""
    from deepeval_eval.api import cache_manager, job_manager

    with (
        patch("tempfile.mkdtemp", return_value=str(tmp_path)),
        patch("deepeval_eval.api.execute_evaluation_job"),
    ):
        file_content = b'[{"question": "What is CAIPE upload cache?"}]'
        files = {"file": ("cached_questions.json", file_content, "application/json")}

        res1 = client.post(
            "/eval/jobs/upload?dataset_name=cached_up&force_rerun=true", files=files
        )
        assert res1.status_code == 202
        job_id = res1.json()["job_id"]

        # Complete and cache job
        job = job_manager.get_job(job_id)
        job["status"] = JobStatusEnum.COMPLETED
        cache_manager.set(job["eval_hash"], job)

        # Submit second upload job with same file and config
        res2 = client.post(
            "/eval/jobs/upload?dataset_name=cached_up&force_rerun=false", files=files
        )
        assert res2.status_code == 202
        assert res2.json()["cached"] is True


def test_get_job_results_csv_positive():
    """Verify GET /jobs/{job_id}/results format=csv returns CSV content."""
    from deepeval_eval.api import cache_manager, job_manager

    job = job_manager.create_job(
        "hash_csv_test", {"dataset_name": "enterprise"}, force_rerun=True
    )
    job["status"] = JobStatusEnum.COMPLETED
    sample_results = [
        {
            "question_id": "q1",
            "question": "What is CAIPE?",
            "actual_output": "CAIPE is an enterprise RAG platform.",
            "latency": 1.25,
            "total_tokens": 150,
            "metrics": {"AnswerRelevancyMetric": {"score": 1.0, "reason": "Relevant"}},
        }
    ]
    cache_manager.save_job_payload(job["job_id"], sample_results)

    response = client.get(f"/jobs/{job['job_id']}/results?format=csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    disp = response.headers["content-disposition"]
    assert f"attachment; filename=job_{job['job_id']}_results.csv" in disp
    assert "question_id,benchmark" in response.text
    assert "What is CAIPE?" in response.text
    assert "AVERAGE_METRICS" in response.text


def test_get_job_results_invalid_format():
    """Verify GET /jobs/{job_id}/results with unsupported format returns HTTP 400."""
    from deepeval_eval.api import job_manager

    job = job_manager.create_job(
        "hash_invalid_format_test", {"dataset_name": "enterprise"}, force_rerun=True
    )
    job["status"] = JobStatusEnum.COMPLETED

    response = client.get(f"/jobs/{job['job_id']}/results?format=invalid_fmt")
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]
