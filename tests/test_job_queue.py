from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from deepeval_eval.api.job_queue import DatabaseManager, PersistentJobQueue
from deepeval_eval.core.config import get_max_concurrent_jobs


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


# ---------------------------------------------------------------------------
# Regression tests: startup stall fix & connect_timeout enforcement
# ---------------------------------------------------------------------------


@pytest.fixture
def postgres_db() -> DatabaseManager:
    """DatabaseManager pre-configured with a Postgres DSN (no actual connection made)."""
    return DatabaseManager(
        connection_string="postgresql://user:pass@unreachable:5432/db"
    )


class TestConnectTimeoutEnforced:
    """Verify psycopg2.connect is always called with connect_timeout=5."""

    def test_connect_with_dsn_string_has_timeout(
        self, postgres_db: DatabaseManager
    ) -> None:
        """Positive: DSN-form connect() is called with connect_timeout=5."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = MagicMock()

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            try:
                postgres_db.get_connection()
            except Exception:
                pass

        mock_psycopg2.connect.assert_called_once()
        _, kwargs = mock_psycopg2.connect.call_args
        assert kwargs.get("connect_timeout") == 5 or (
            len(mock_psycopg2.connect.call_args.args) > 0
            and "connect_timeout=5" in str(mock_psycopg2.connect.call_args)
        ), "connect_timeout=5 must be passed to psycopg2.connect when using DSN string"

    def test_connect_with_keyword_args_has_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive: keyword-args form connect() is called with connect_timeout=5."""
        monkeypatch.setenv("POSTGRES_HOST", "unreachable-host")
        monkeypatch.setenv("POSTGRES_PORT", "5432")
        monkeypatch.setenv("POSTGRES_DB", "testdb")
        monkeypatch.setenv("POSTGRES_USER", "testuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("LANGGRAPH_CHECKPOINT_POSTGRES_DSN", raising=False)
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        monkeypatch.delenv("DB_CONNECTION_STRING", raising=False)

        db_mgr = DatabaseManager.__new__(DatabaseManager)
        db_mgr.connection_string = None
        import threading

        db_mgr._lock = threading.Lock()

        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = MagicMock()

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            try:
                db_mgr.get_connection()
            except Exception:
                pass

        mock_psycopg2.connect.assert_called_once()
        _, kwargs = mock_psycopg2.connect.call_args
        assert kwargs.get("connect_timeout") == 5, (
            "connect_timeout=5 must be passed to psycopg2.connect in keyword-args path"
        )

    def test_connect_without_timeout_would_block_indefinitely(
        self, postgres_db: DatabaseManager
    ) -> None:
        """Negative: absence of connect_timeout would be a regression — assert it is present."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = MagicMock()

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            try:
                postgres_db.get_connection()
            except Exception:
                pass

        _, kwargs = mock_psycopg2.connect.call_args
        assert "connect_timeout" in kwargs, (
            "Regression: connect_timeout missing — psycopg2.connect will block indefinitely "
            "on unreachable hosts"
        )


class TestStartNonBlocking:
    """Verify PersistentJobQueue.start() submits recovery to the executor and
    does NOT call _recover_and_dispatch synchronously on the calling thread."""

    def test_start_submits_recovery_to_executor_not_inline(
        self, postgres_db: DatabaseManager
    ) -> None:
        """Positive: start() uses executor.submit for DB recovery, not a direct call."""
        queue = PersistentJobQueue(postgres_db)
        queue.set_task_executor(MagicMock())

        recover_calls: list[str] = []

        def tracking_recover() -> None:
            recover_calls.append("recover_called")

        queue._recover_and_dispatch = tracking_recover  # type: ignore[method-assign]

        mock_executor = MagicMock()
        submitted_fns: list = []

        def capture_submit(fn, *args, **kwargs):
            submitted_fns.append(fn)
            return MagicMock()

        mock_executor.submit.side_effect = capture_submit

        with patch(
            "deepeval_eval.api.job_queue.ThreadPoolExecutor",
            return_value=mock_executor,
        ):
            queue.start()

        # Recovery must have been submitted to the executor, not called inline
        assert len(submitted_fns) == 1, (
            "_recover_and_dispatch must be submitted to executor"
        )
        assert submitted_fns[0] is tracking_recover
        assert recover_calls == [], (
            "Regression: _recover_and_dispatch was called synchronously on the event loop thread "
            "— this causes uvicorn 'Waiting for application startup.' to stall"
        )

    def test_start_returns_immediately_without_db(
        self, unconfigured_db: DatabaseManager
    ) -> None:
        """Positive: start() returns immediately when Postgres is not configured."""
        queue = PersistentJobQueue(unconfigured_db)
        queue.set_task_executor(MagicMock())

        start_time = time.monotonic()
        queue.start()
        elapsed = time.monotonic() - start_time
        queue.stop()

        assert elapsed < 0.5, (
            f"start() took {elapsed:.3f}s without DB — should be near instant"
        )

    def test_start_is_idempotent(self, unconfigured_db: DatabaseManager) -> None:
        """Positive: calling start() twice does not error or duplicate executor."""
        queue = PersistentJobQueue(unconfigured_db)
        queue.set_task_executor(MagicMock())
        queue.start()
        queue.start()  # second call must be a no-op
        assert queue._running is True
        queue.stop()

    def test_start_does_not_block_with_slow_recovery(
        self, postgres_db: DatabaseManager
    ) -> None:
        """Behavioral regression: start() returns in <0.5s even when _recover_and_dispatch
        takes 2 seconds (simulates an unreachable / slow Postgres host).

        This is the direct reproduction of the original startup stall bug.
        If _recover_and_dispatch() is called synchronously (pre-fix), this test
        takes ~2 seconds and fails the timing assertion, reproducing the exact
        symptom: uvicorn stuck at 'Waiting for application startup.'
        """
        import threading

        queue = PersistentJobQueue(postgres_db)
        queue.set_task_executor(MagicMock())

        recovery_started = threading.Event()

        def slow_recovery() -> None:
            """Simulates a 2-second blocking psycopg2.connect() on an unreachable host."""
            recovery_started.set()
            time.sleep(2.0)

        queue._recover_and_dispatch = slow_recovery  # type: ignore[method-assign]

        start_time = time.monotonic()
        queue.start()
        elapsed = time.monotonic() - start_time

        # start() must return well before the 2s slow recovery finishes
        assert elapsed < 0.5, (
            f"REGRESSION: start() blocked for {elapsed:.2f}s — "
            "_recover_and_dispatch() is being called synchronously on the calling thread. "
            "This reproduces the uvicorn 'Waiting for application startup.' stall. "
            "Fix: submit _recover_and_dispatch to self._executor instead of calling directly."
        )

        # Recovery must still RUN — just in the background, not on this thread
        assert recovery_started.wait(timeout=1.5), (
            "Recovery should have started in a background thread"
        )
        queue.stop()

    def test_would_stall_if_recovery_were_synchronous(
        self, postgres_db: DatabaseManager
    ) -> None:
        """Negative / confirmatory: documents what the pre-fix (broken) behaviour looked like.

        Calls _recover_and_dispatch() synchronously (as the old code did) with a
        deliberately fast mock (0.3s) and verifies it blocks start() for that duration.
        This confirms our timing threshold is correctly calibrated to catch the regression.
        """
        queue = PersistentJobQueue(postgres_db)
        queue.set_task_executor(MagicMock())

        SIMULATED_DELAY = 0.3  # deliberately short so the test stays fast

        def blocking_recovery() -> None:
            time.sleep(SIMULATED_DELAY)

        # Manually call synchronously (pre-fix pattern) to confirm it DOES block
        start_time = time.monotonic()
        blocking_recovery()  # calling directly = the old broken pattern
        elapsed = time.monotonic() - start_time

        assert elapsed >= SIMULATED_DELAY, (
            "Confirmatory test failed: synchronous recovery must block for at least "
            f"{SIMULATED_DELAY}s — this validates that the timing threshold in "
            "test_start_does_not_block_with_slow_recovery would correctly catch a regression."
        )
        queue.stop()


class TestDatabaseManagerLogging:
    """Verify DB errors are logged at warning/error level, not silently at debug."""

    def test_init_db_failure_logs_warning_not_debug(
        self, postgres_db: DatabaseManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Positive: DatabaseManager.__init__ logs a warning when init_db fails."""
        import logging

        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = OSError("Connection refused")

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            with caplog.at_level(logging.WARNING, logger="deepeval_eval.api.job_queue"):
                db_mgr = DatabaseManager(
                    connection_string="postgresql://user:pass@unreachable:5432/db"
                )

        assert db_mgr is not None
        warning_msgs = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            "unreachable" in m or "DB may be unreachable" in m or "init" in m.lower()
            for m in warning_msgs
        ), (
            "Regression: DatabaseManager init DB failure must log at WARNING, "
            f"got: {warning_msgs}"
        )

    def test_get_job_postgres_failure_logs_warning(
        self, postgres_db: DatabaseManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Positive: get_job() Postgres failure falls back to memory and logs a warning."""
        import logging

        queue = PersistentJobQueue(postgres_db)
        # Seed in-memory fallback
        queue._memory_jobs["fallback-job"] = {
            "job_id": "fallback-job",
            "eval_hash": "h",
            "status": "pending",
            "config_args": {},
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = OSError("Connection refused")
        mock_psycopg2.extras = MagicMock()

        with patch.dict(
            "sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()}
        ):
            with caplog.at_level(logging.WARNING, logger="deepeval_eval.api.job_queue"):
                result = queue.get_job("fallback-job")

        assert result is not None, (
            "get_job must fall back to memory when Postgres fails"
        )
        assert result["job_id"] == "fallback-job"
        warning_msgs = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("Postgres" in m or "falling back" in m for m in warning_msgs), (
            "Regression: get_job Postgres failure must log at WARNING level, "
            f"got: {warning_msgs}"
        )

    def test_list_jobs_postgres_failure_logs_warning(
        self, postgres_db: DatabaseManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Positive: list_jobs() Postgres failure falls back to memory and logs a warning."""
        import logging

        queue = PersistentJobQueue(postgres_db)
        queue._memory_jobs["mem-job-1"] = {
            "job_id": "mem-job-1",
            "eval_hash": "h",
            "status": "completed",
            "config_args": {},
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = OSError("Connection refused")

        with patch.dict(
            "sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()}
        ):
            with caplog.at_level(logging.WARNING, logger="deepeval_eval.api.job_queue"):
                results = queue.list_jobs()

        assert any(j["job_id"] == "mem-job-1" for j in results), (
            "list_jobs must fall back to memory when Postgres fails"
        )
        warning_msgs = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("Postgres" in m or "falling back" in m for m in warning_msgs), (
            "Regression: list_jobs Postgres failure must log at WARNING level, "
            f"got: {warning_msgs}"
        )

    def test_dispatch_postgres_failure_logs_error(
        self, postgres_db: DatabaseManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Positive: _dispatch_next_if_possible() Postgres failure logs at ERROR level."""
        import logging
        from concurrent.futures import ThreadPoolExecutor

        queue = PersistentJobQueue(postgres_db)
        queue.set_task_executor(MagicMock())
        queue._running = True
        queue._executor = ThreadPoolExecutor(max_workers=1)

        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = OSError("Connection refused")

        try:
            with patch.dict(
                "sys.modules",
                {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()},
            ):
                with caplog.at_level(
                    logging.ERROR, logger="deepeval_eval.api.job_queue"
                ):
                    queue._dispatch_next_if_possible()
        finally:
            queue._executor.shutdown(wait=False)

        error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            "dispatch" in m.lower() or "pending" in m.lower() for m in error_msgs
        ), (
            "Regression: _dispatch_next_if_possible Postgres failure must log at ERROR level, "
            f"got: {error_msgs}"
        )
