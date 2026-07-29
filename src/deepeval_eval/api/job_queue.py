from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from deepeval_eval.core.config import get_max_concurrent_jobs
from deepeval_eval.db import DatabaseManager

logger = logging.getLogger(__name__)


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
        with self._lock:
            candidates = [
                jid
                for jid, jdata in list(self._memory_jobs.items())
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
                # Submit recovery to the thread pool so start() returns
                # immediately and does not block the async event loop during
                # uvicorn lifespan startup.
                self._executor.submit(self._recover_and_dispatch)

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
            raw_config_json = json.dumps(config_dict, ensure_ascii=False)
            self.db_manager.execute_write(
                "INSERT INTO eval_job_queue (job_id, eval_hash, status, config_json, created_at) VALUES (%s, %s, %s, %s, %s)",
                (job_id, eval_hash, "pending", raw_config_json, now),
            )
        else:
            with self._lock:
                self._memory_jobs[job_id] = {**job_record, "raw_config": config_dict}
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
                logger.warning(
                    f"Failed to query job '{job_id}' from Postgres, falling back to memory: {e}"
                )

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
                logger.warning(
                    f"Failed to list jobs from Postgres, falling back to memory: {e}"
                )

        with self._lock:
            recent_jobs = list(self._memory_jobs.values())
            recent_jobs.sort(key=lambda j: j["created_at"], reverse=True)
            return [dict(j) for j in recent_jobs[:limit]]

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
                logger.error(
                    f"Failed to query pending jobs from Postgres during dispatch: {exc}"
                )
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
                        cfg = job_rec.get("raw_config") or job_rec.get(
                            "config_args", {}
                        )
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
