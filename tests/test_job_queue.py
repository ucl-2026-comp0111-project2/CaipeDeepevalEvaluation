from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from deepeval_eval.config import get_max_concurrent_jobs
from deepeval_eval.job_queue import DatabaseManager, PersistentJobQueue


@pytest.fixture
def unconfigured_db(monkeypatch: pytest.MonkeyPatch) -> DatabaseManager:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LANGGRAPH_CHECKPOINT_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    return DatabaseManager()


def test_database_manager_unconfigured(unconfigured_db: DatabaseManager) -> None:
    assert unconfigured_db.is_postgres() is False
    with pytest.raises(RuntimeError, match="PostgreSQL database is not configured"):
        unconfigured_db.get_connection()


def test_database_manager_postgres_connection_string() -> None:
    db_mgr = DatabaseManager(
        connection_string="postgresql://user:pass@localhost:5432/db"
    )
    assert db_mgr.is_postgres() is True


def test_persistent_job_queue_in_memory_enqueue_and_get(
    unconfigured_db: DatabaseManager,
) -> None:
    queue = PersistentJobQueue(unconfigured_db)
    job_record = queue.enqueue("job-123", "hash123", {"dataset_name": "test"})
    assert job_record["job_id"] == "job-123"
    assert job_record["status"] == "pending"

    fetched = queue.get_job("job-123")
    assert fetched is not None
    assert fetched["job_id"] == "job-123"
    assert fetched["eval_hash"] == "hash123"
    assert fetched["config_args"] == {"dataset_name": "test"}

    jobs = queue.list_jobs()
    assert len(jobs) >= 1
    assert any(j["job_id"] == "job-123" for j in jobs)


def test_persistent_job_queue_worker_execution(
    unconfigured_db: DatabaseManager,
) -> None:
    executed_jobs: list[str] = []

    def dummy_task(job_id: str, config: dict) -> None:
        time.sleep(0.05)
        executed_jobs.append(job_id)

    queue = PersistentJobQueue(unconfigured_db)
    queue.set_task_executor(dummy_task)
    queue.start()

    try:
        queue.enqueue("job-worker-1", "h1", {"dataset_name": "test1"})
        queue.enqueue("job-worker-2", "h2", {"dataset_name": "test2"})

        timeout = time.time() + 3.0
        while len(executed_jobs) < 2 and time.time() < timeout:
            time.sleep(0.05)

        assert "job-worker-1" in executed_jobs
        assert "job-worker-2" in executed_jobs

        j1 = queue.get_job("job-worker-1")
        assert j1 is not None and j1["status"] == "completed"
    finally:
        queue.stop()


def test_persistent_job_queue_postgres_mode() -> None:
    db_mgr = DatabaseManager(
        connection_string="postgresql://user:pass@localhost:5432/db"
    )
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchall.return_value = [
        {
            "job_id": "pg-job-1",
            "eval_hash": "hash_pg",
            "status": "pending",
            "config_json": '{"dataset_name": "pg_test"}',
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
    ]

    with patch.dict(
        "sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()}
    ):
        queue = PersistentJobQueue(db_mgr)
        job = queue.get_job("pg-job-1")
        assert job is not None
        assert job["job_id"] == "pg-job-1"


def test_get_max_concurrent_jobs_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_MAX_CONCURRENT_JOBS", "5")
    assert get_max_concurrent_jobs() == 5

    monkeypatch.setenv("EVAL_MAX_CONCURRENT_JOBS", "invalid")
    assert get_max_concurrent_jobs() == 1


def test_sanitize_config_dict_redacts_secrets(
    unconfigured_db: DatabaseManager,
) -> None:
    queue = PersistentJobQueue(unconfigured_db)
    raw_config = {
        "dataset_name": "test_secret",
        "llm_api_key": "sk-secret-123",
        "auth_token": "bearer-token-xyz",
        "rag_auth_token": "rag-secret",
    }
    job = queue.enqueue("job-sec-1", "hashsec", raw_config)
    assert job["config_args"]["llm_api_key"] == "***REDACTED***"
    assert job["config_args"]["auth_token"] == "***REDACTED***"
    assert job["config_args"]["rag_auth_token"] == "***REDACTED***"
    assert job["config_args"]["dataset_name"] == "test_secret"
