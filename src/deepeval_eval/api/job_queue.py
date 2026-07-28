from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from deepeval_eval.core.config import get_max_concurrent_jobs

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Standardized PostgreSQL database manager."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string
        self._lock = threading.Lock()
        if self.is_postgres():
            try:
                self.init_db()
            except Exception as exc:
                logger.debug(f"PostgreSQL init_db deferred: {exc}")

    def is_postgres(self) -> bool:
        conn_str = (
            self.connection_string
            or os.environ.get("DATABASE_URL")
            or os.environ.get("LANGGRAPH_CHECKPOINT_POSTGRES_DSN")
            or os.environ.get("POSTGRES_DSN")
        )
        if conn_str:
            return conn_str.startswith("postgresql://") or conn_str.startswith(
                "postgres://"
            )
        host = (
            os.environ.get("POSTGRES_HOST")
            or os.environ.get("PGHOST")
            or os.environ.get("DB_HOST")
        )
        return bool(host)

    def get_connection(self) -> Any:
        if not self.is_postgres():
            raise RuntimeError("PostgreSQL database is not configured.")

        import psycopg2

        conn_str = (
            self.connection_string
            or os.environ.get("DATABASE_URL")
            or os.environ.get("LANGGRAPH_CHECKPOINT_POSTGRES_DSN")
            or os.environ.get("POSTGRES_DSN")
        )
        if conn_str:
            return psycopg2.connect(conn_str)

        host = (
            os.environ.get("POSTGRES_HOST")
            or os.environ.get("PGHOST")
            or os.environ.get("DB_HOST", "localhost")
        )
        port = (
            os.environ.get("POSTGRES_PORT")
            or os.environ.get("PGPORT")
            or os.environ.get("DB_PORT", "5432")
        )
        dbname = (
            os.environ.get("POSTGRES_DB")
            or os.environ.get("PGDATABASE")
            or os.environ.get("DB_NAME", "caipe_eval")
        )
        user = (
            os.environ.get("POSTGRES_USER")
            or os.environ.get("PGUSER")
            or os.environ.get("DB_USER", "postgres")
        )
        password = (
            os.environ.get("POSTGRES_PASSWORD")
            or os.environ.get("PGPASSWORD")
            or os.environ.get("DB_PASSWORD", "")
        )
        sslmode = os.environ.get("PGSSLMODE", "prefer")
        return psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=sslmode,
        )

    def init_db(self) -> None:
        """Initialize PostgreSQL schema tables if not present."""
        if not self.is_postgres():
            return
        with self._lock:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS eval_job_queue (
                            job_id       TEXT PRIMARY KEY,
                            eval_hash    TEXT NOT NULL,
                            status       TEXT NOT NULL,
                            config_json  TEXT NOT NULL,
                            created_at   DOUBLE PRECISION NOT NULL,
                            started_at   DOUBLE PRECISION,
                            completed_at DOUBLE PRECISION,
                            error        TEXT
                        );
                        CREATE TABLE IF NOT EXISTS batches (
                            batch_id    TEXT PRIMARY KEY,
                            created_at  TIMESTAMP NOT NULL DEFAULT now(),
                            description TEXT
                        );
                        CREATE TABLE IF NOT EXISTS runs (
                            run_id       TEXT PRIMARY KEY,
                            batch_id     TEXT NOT NULL,
                            config_name  TEXT NOT NULL,
                            config_json  JSONB,
                            started_at   TIMESTAMP,
                            finished_at  TIMESTAMP,
                            loaded_at    TIMESTAMP NOT NULL DEFAULT now()
                        );
                        CREATE TABLE IF NOT EXISTS eval_results (
                            id         BIGSERIAL PRIMARY KEY,
                            run_id     TEXT NOT NULL,
                            batch_id   TEXT NOT NULL,
                            question   TEXT,
                            row_data   JSONB
                        );
                        CREATE TABLE IF NOT EXISTS run_summary (
                            run_id        TEXT PRIMARY KEY,
                            p50_latency   DOUBLE PRECISION,
                            p95_latency   DOUBLE PRECISION,
                            summary_json  JSONB
                        );
                        CREATE INDEX IF NOT EXISTS idx_eval_job_queue_status_created ON eval_job_queue (status, created_at);
                        """
                    )
                conn.commit()
            except Exception as exc:
                if conn is not None and hasattr(conn, "rollback"):
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                logger.warning(f"PostgreSQL schema initialization skipped: {exc}")
            finally:
                if conn is not None and not getattr(conn, "closed", False):
                    conn.close()

    def execute_write(self, query: str, params: tuple[Any, ...]) -> None:
        with self._lock:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
            except Exception:
                if conn is not None and hasattr(conn, "rollback"):
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn is not None and not getattr(conn, "closed", False):
                    conn.close()

    def query_all(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self.get_connection()
            try:
                from psycopg2.extras import RealDictCursor

                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
            except Exception:
                if conn is not None and hasattr(conn, "rollback"):
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn is not None and not getattr(conn, "closed", False):
                    conn.close()


def sanitize_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive credentials before storing or returning job configuration dicts."""
    sanitized = dict(config_dict)
    for key in ("llm_api_key", "auth_token", "rag_auth_token", "api_key", "password"):
        if key in sanitized:
            sanitized[key] = "***REDACTED***"
    return sanitized


class PersistentJobQueue:
    """Evaluation job queue with PostgreSQL persistence and in-memory fallback."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.max_workers = get_max_concurrent_jobs()
        self._executor: ThreadPoolExecutor | None = None
        self._running = False
        self._lock = threading.Lock()
        self._task_fn: Callable[[str, dict[str, Any]], None] | None = None
        self._active_jobs: set[str] = set()
        # In-memory fallback storage when PostgreSQL is not configured
        self._memory_jobs: dict[str, dict[str, Any]] = {}
        self._memory_queue: deque[str] = deque()
        self._max_memory_jobs = 1000

    def _evict_old_memory_jobs(self) -> None:
        """Evict completed/failed jobs from memory storage if exceeding maximum capacity."""
        if len(self._memory_jobs) <= self._max_memory_jobs:
            return
        candidates = [
            jid
            for jid, jdata in self._memory_jobs.items()
            if jdata.get("status") in ("completed", "failed")
            and jid not in self._active_jobs
        ]
        for jid in candidates[: len(self._memory_jobs) - self._max_memory_jobs]:
            self._memory_jobs.pop(jid, None)

    def set_task_executor(self, task_fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Register the job execution function."""
        self._task_fn = task_fn

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self.max_workers = get_max_concurrent_jobs()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="eval_worker"
                )
            if self.db_manager.is_postgres():
                self._recover_and_dispatch()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None
            self._active_jobs.clear()

    def enqueue(
        self, job_id: str, eval_hash: str, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        now = time.time()
        sanitized_config = sanitize_config_dict(config_dict)
        job_record = {
            "job_id": job_id,
            "eval_hash": eval_hash,
            "status": "pending",
            "config_args": sanitized_config,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

        if self.db_manager.is_postgres():
            config_json = json.dumps(sanitized_config, ensure_ascii=False)
            self.db_manager.execute_write(
                "INSERT INTO eval_job_queue (job_id, eval_hash, status, config_json, created_at) VALUES (%s, %s, %s, %s, %s)",
                (job_id, eval_hash, "pending", config_json, now),
            )
        else:
            with self._lock:
                self._memory_jobs[job_id] = job_record
                self._memory_queue.append(job_id)
                self._evict_old_memory_jobs()

        self._dispatch_next_if_possible()
        return job_record

    def update_status(
        self,
        job_id: str,
        status: str,
        error: str | None = None,
        started_at: float | None = None,
        completed_at: float | None = None,
    ) -> None:
        now = time.time()
        if self.db_manager.is_postgres():
            if status == "running":
                s_at = started_at or now
                self.db_manager.execute_write(
                    "UPDATE eval_job_queue SET status=%s, started_at=%s WHERE job_id=%s",
                    (status, s_at, job_id),
                )
            elif status in ("completed", "failed"):
                c_at = completed_at or now
                self.db_manager.execute_write(
                    "UPDATE eval_job_queue SET status=%s, completed_at=%s, error=%s WHERE job_id=%s",
                    (status, c_at, error, job_id),
                )
        else:
            with self._lock:
                j = self._memory_jobs.get(job_id)
                if j:
                    j["status"] = status
                    if status == "running":
                        j["started_at"] = started_at or now
                    elif status in ("completed", "failed"):
                        j["completed_at"] = completed_at or now
                        j["error"] = error

        if status in ("completed", "failed"):
            with self._lock:
                self._active_jobs.discard(job_id)
            self._dispatch_next_if_possible()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if self.db_manager.is_postgres():
            try:
                rows = self.db_manager.query_all(
                    "SELECT * FROM eval_job_queue WHERE job_id=%s",
                    (job_id,),
                )
                if not rows:
                    return None
                r = rows[0]
                try:
                    cfg = json.loads(r["config_json"])
                except Exception:
                    cfg = {}
                return {
                    "job_id": r["job_id"],
                    "eval_hash": r["eval_hash"],
                    "status": r["status"],
                    "config_args": cfg,
                    "created_at": r["created_at"],
                    "started_at": r.get("started_at"),
                    "completed_at": r.get("completed_at"),
                    "error": r.get("error"),
                }
            except Exception as e:
                logger.debug(f"Failed to query job '{job_id}' from Postgres: {e}")

        with self._lock:
            j = self._memory_jobs.get(job_id)
            return dict(j) if j else None

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        if self.db_manager.is_postgres():
            try:
                rows = self.db_manager.query_all(
                    "SELECT job_id, eval_hash, status, config_json, created_at, started_at, completed_at, error "
                    "FROM eval_job_queue ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                results = []
                for r in rows:
                    try:
                        cfg = json.loads(r["config_json"])
                    except Exception:
                        cfg = {}
                    results.append(
                        {
                            "job_id": r["job_id"],
                            "eval_hash": r["eval_hash"],
                            "status": r["status"],
                            "config_args": cfg,
                            "created_at": r["created_at"],
                            "started_at": r.get("started_at"),
                            "completed_at": r.get("completed_at"),
                            "error": r.get("error"),
                        }
                    )
                return results
            except Exception as e:
                logger.debug(f"Failed to list jobs from Postgres: {e}")

        with self._lock:
            sorted_jobs = sorted(
                [dict(j) for j in self._memory_jobs.values()],
                key=lambda j: j["created_at"],
                reverse=True,
            )
            return sorted_jobs[:limit]

    def _recover_and_dispatch(self) -> None:
        """On startup, reset interrupted running jobs to pending and dispatch tasks."""
        if not self.db_manager.is_postgres():
            return
        try:
            self.db_manager.execute_write(
                "UPDATE eval_job_queue SET status='pending' WHERE status IN ('running', 'queued')",
                (),
            )
        except Exception as exc:
            logger.warning(f"PostgreSQL job recovery skipped: {exc}")
        self._dispatch_next_if_possible()

    def _dispatch_next_if_possible(self) -> None:
        with self._lock:
            if not self._running or not self._executor or not self._task_fn:
                return
            available_slots = self.max_workers - len(self._active_jobs)
            if available_slots <= 0:
                return

        if self.db_manager.is_postgres():
            try:
                rows = self.db_manager.query_all(
                    "SELECT job_id, config_json FROM eval_job_queue WHERE status='pending' ORDER BY created_at ASC LIMIT %s",
                    (available_slots,),
                )
                for r in rows:
                    job_id = r["job_id"]
                    with self._lock:
                        if job_id in self._active_jobs:
                            continue
                        self._active_jobs.add(job_id)
                    try:
                        cfg = json.loads(r["config_json"])
                    except Exception:
                        cfg = {}
                    try:
                        self._executor.submit(self._run_job_wrapper, job_id, cfg)
                    except Exception as exc:
                        with self._lock:
                            self._active_jobs.discard(job_id)
                        logger.error(
                            f"Failed to submit job {job_id} to worker pool: {exc}"
                        )
            except Exception as exc:
                logger.debug(f"Failed to query pending jobs from Postgres: {exc}")
        else:
            with self._lock:
                while self._memory_queue:
                    if len(self._active_jobs) >= self.max_workers:
                        break
                    job_id = self._memory_queue.popleft()
                    if job_id in self._active_jobs:
                        continue
                    job_rec = self._memory_jobs.get(job_id)
                    if job_rec and job_rec["status"] == "pending":
                        self._active_jobs.add(job_id)
                        cfg = job_rec.get("config_args", {})
                        try:
                            self._executor.submit(self._run_job_wrapper, job_id, cfg)
                        except Exception as exc:
                            self._active_jobs.discard(job_id)
                            logger.error(
                                f"Failed to submit job {job_id} to worker pool: {exc}"
                            )

    def _run_job_wrapper(self, job_id: str, config_dict: dict[str, Any]) -> None:
        if not self._task_fn:
            return
        self.update_status(job_id, "running")
        try:
            self._task_fn(job_id, config_dict)
            self.update_status(job_id, "completed")
        except Exception as exc:
            logger.error(f"Execution of job '{job_id}' failed: {exc}", exc_info=True)
            self.update_status(job_id, "failed", error=str(exc))
