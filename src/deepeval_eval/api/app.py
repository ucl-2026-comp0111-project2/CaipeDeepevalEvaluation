from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deepeval_eval.api.auth import UserContext, get_current_user
from deepeval_eval.api.job_queue import DatabaseManager, PersistentJobQueue
from deepeval_eval.api.question_sets import router as question_sets_router
from deepeval_eval.api.telemetry import (
    setup_otlp_tracing,
    telemetry_metrics,
    telemetry_router,
)
from deepeval_eval.core.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_GATE_CONFIG,
    DEFAULT_RESULTS_DIR,
    EvalConfig,
    get_eval_config,
)
from deepeval_eval.core.io_utils import sanitize_path
from deepeval_eval.core.prompt_style import DEFAULT_PROMPT_STYLE
from deepeval_eval.engine.eval_engine import (
    _build_rag_client,
    run_evaluation,
)
from deepeval_eval.sinks import PostgresResultSink
from deepeval_eval.sinks.file_sink import format_results_as_csv

logger = logging.getLogger(__name__)

# Server-level configuration read from EvalConfig singleton at startup
SERVER_PROMPT_CONFIG: Path | None = (
    get_eval_config().prompt_config.resolve()
    if get_eval_config().prompt_config
    else None
)

# ---------------------------------------------------------------------------
# Pydantic Request & Response Models (DTOs)
# ---------------------------------------------------------------------------


class JobStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRequest(BaseModel):
    question_set_id: int | None = Field(
        default=None,
        description="ID of a Question Set stored in Question Set Manager to evaluate",
    )
    dataset_name: str = Field(
        default="enterprise",
        description="Dataset name (e.g. enterprise, hotpotqa) or custom benchmark",
    )
    answer_mode: str = Field(
        default="generate",
        description="Evaluation answer mode: 'generate' or 'ground_truth'",
    )
    oracle_testing: bool = Field(
        default=False,
        description="Shortcut flag to enable oracle_retrieval and ground_truth answer mode",
    )
    datasource_id: str | None = Field(
        default=None, description="Target CAIPE datasource ID"
    )
    prompt_style: str | None = Field(
        default=DEFAULT_PROMPT_STYLE,
        description="Prompt style (e.g. generation, short, or custom)",
    )
    max_items: int | None = Field(
        default=None, ge=1, description="Maximum number of items to evaluate"
    )
    limit_per_category: int | None = Field(
        default=None, ge=1, description="Limit items per category"
    )
    top_k: int = Field(
        default=3, ge=1, description="Number of context documents to retrieve"
    )
    max_context_chars: int = Field(
        default=12000, ge=100, description="Max context characters to pass to evaluator"
    )
    llm_base_url: str | None = Field(
        default=None, description="Custom LLM API base URL"
    )
    llm_api_key: str | None = Field(default=None, description="Custom LLM API key")
    llm_model: str | None = Field(default=None, description="Custom LLM model name")
    agentic: bool = Field(
        default=False, description="Route queries through CAIPE supervisor A2A endpoint"
    )
    trace_log: bool = Field(
        default=False,
        description="Save detailed agentic stream and query trace logs to disk",
    )
    agent_id: str | None = Field(
        default=None,
        description="Optional CAIPE agent ID for agentic RAG evaluations",
    )
    supervisor_url: str | None = Field(default=None, description="CAIPE supervisor URL")
    fail_on_error: bool = Field(
        default=False, description="Fail loudly if a query evaluation fails"
    )
    oracle_retrieval: bool = Field(
        default=False, description="Enable oracle (question + reference) retrieval"
    )
    gate: bool = Field(default=False, description="Apply quality gate after evaluation")
    save_to_db: bool = Field(
        default=False, description="Persist evaluation results to PostgreSQL DB"
    )
    force_rerun: bool = Field(
        default=False,
        description="Bypass evaluation deduplication cache and force rerun",
    )
    question_ids: list[str] | None = Field(
        default=None, description="List of specific question IDs to evaluate"
    )
    question_indices: list[int] | None = Field(
        default=None, description="List of specific question indices to evaluate"
    )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    created_at: float
    completed_at: float | None = None
    cached: bool = False
    eval_hash: str
    error: str | None = None
    user_info: dict[str, Any] | None = Field(
        default=None, description="Authenticated user/client identity details"
    )


class EvaluationResultsResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    created_at: float
    completed_at: float | None = None
    cached: bool = False
    eval_hash: str
    evaluation_time: float = 0.0
    config_args: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)
    saved_to_db: bool = False
    user_info: dict[str, Any] | None = Field(
        default=None, description="Authenticated user/client identity details"
    )


class EvaluationSummaryResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    created_at: float
    completed_at: float | None = None
    cached: bool = False
    eval_hash: str
    evaluation_time: float = 0.0
    config_args: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    saved_to_db: bool = False
    user_info: dict[str, Any] | None = Field(
        default=None, description="Authenticated user/client identity details"
    )


# ---------------------------------------------------------------------------
# Deduplication Hashing & Cache Management (Cache-Aside Pattern)
# ---------------------------------------------------------------------------


def validate_safe_path(user_path: str | Path | None) -> Path | None:
    """Validate that specified file path resides strictly within approved sandbox directories."""
    if not user_path:
        return None
    path_obj = Path(user_path).expanduser().resolve()
    allowed_roots = [
        Path(tempfile.gettempdir()).resolve(),
        DEFAULT_DATA_DIR.resolve(),
        (DEFAULT_DATA_DIR.parent / "evals").resolve(),
    ]
    is_safe = any(
        path_obj == root or root in path_obj.parents for root in allowed_roots
    )
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Access to file path '{user_path}' is restricted: path is outside allowed sandbox directories.",
        )
    return path_obj


def sanitize_config_args(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Sanitize configuration fields to prevent credential leakage in outputs."""
    sensitive_keys = {
        "llm_api_key",
        "auth_token",
        "client_secret",
        "db_connection_string",
    }
    path_keys = {"questions_file", "results_dir", "log_file"}
    sanitized = {}
    for k, v in config_dict.items():
        if k in sensitive_keys or v is None:
            continue
        if k in path_keys and isinstance(v, str):
            sanitized[k] = sanitize_path(v)
        else:
            sanitized[k] = v
    return sanitized


def compute_eval_hash(
    config_dict: dict[str, Any], dataset_bytes: bytes | None = None
) -> str:
    """Compute a deterministic SHA-256 fingerprint for evaluation parameters."""
    hash_obj = hashlib.sha256()

    # Filter out transient non-config keys
    ignored_keys = {
        "force_rerun",
        "llm_api_key",
        "auth_token",
        "client_secret",
        "db_connection_string",
    }
    normalized_config = {
        k: str(v)
        for k, v in sorted(config_dict.items())
        if v is not None and k not in ignored_keys
    }
    hash_obj.update(json.dumps(normalized_config, sort_keys=True).encode("utf-8"))

    if dataset_bytes:
        hash_obj.update(dataset_bytes)

    return hash_obj.hexdigest()[:16]


class LocalCacheManager:
    """Manages local 24-hour file cache for evaluation results."""

    CACHE_TTL_SECONDS = 86400  # 24 hours

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR / "eval_results"):
        self.cache_dir = cache_dir
        self.job_payloads_dir = cache_dir / "job_payloads"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.job_payloads_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, eval_hash: str) -> Path:
        return self.cache_dir / f"{eval_hash}.json"

    def _get_job_payload_path(self, job_id: str) -> Path:
        return self.job_payloads_dir / f"{job_id}.json"

    def _get_job_meta_path(self, job_id: str) -> Path:
        return self.job_payloads_dir / f"{job_id}_meta.json"

    def get(self, eval_hash: str) -> dict[str, Any] | None:
        """Retrieve cached result if present and within 24-hour TTL."""
        path = self._get_cache_path(eval_hash)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            timestamp = data.get("timestamp", 0.0)
            if time.time() - timestamp > self.CACHE_TTL_SECONDS:
                path.unlink(missing_ok=True)
                return None
            return data
        except Exception:
            return None

    def set(self, eval_hash: str, job_data: dict[str, Any]) -> None:
        """Store evaluation metadata in cache with current timestamp."""
        status = job_data.get("status")
        if (
            status and status not in ("completed", JobStatusEnum.COMPLETED)
        ) or job_data.get("error"):
            return
        path = self._get_cache_path(eval_hash)
        payload = dict(job_data)
        job_id = payload.get("job_id", eval_hash)
        # Store results payload separately to keep cache metadata lean
        results = payload.pop("results", None)
        if results is not None:
            self.save_job_payload(job_id, results)
        payload["timestamp"] = time.time()
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # Maintain O(1) job_id -> eval_hash lookup index
            self._get_job_meta_path(job_id).write_text(eval_hash, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write to evaluation cache: {e}")

    def save_job_payload(self, job_id: str, results: list[dict[str, Any]]) -> None:
        """Persist full job evaluation results array to disk cache."""
        path = self._get_job_payload_path(job_id)
        try:
            path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to write job payload to disk: {e}")

    def get_job_payload(self, job_id: str) -> list[dict[str, Any]]:
        """Load full evaluation results array from disk cache."""
        path = self._get_job_payload_path(job_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def get_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        """O(1) index search for evaluation cache entry by job_id."""
        meta_path = self._get_job_meta_path(job_id)
        if meta_path.exists():
            try:
                eval_hash = meta_path.read_text(encoding="utf-8").strip()
                return self.get(eval_hash)
            except Exception as e:
                logger.debug(f"Failed to read cache index for job '{job_id}': {e}")
                return None
        return None

    def purge_expired(self) -> int:
        """Purge entries older than 24 hours or unparseable corrupted cache files."""
        purged = 0
        now = time.time()
        for p in self.cache_dir.glob("*.json"):
            try:
                if now - p.stat().st_mtime > self.CACHE_TTL_SECONDS:
                    p.unlink(missing_ok=True)
                    purged += 1
                else:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if now - data.get("timestamp", 0.0) > self.CACHE_TTL_SECONDS:
                        p.unlink(missing_ok=True)
                        purged += 1
            except Exception:
                p.unlink(missing_ok=True)
                purged += 1
        return purged


# ---------------------------------------------------------------------------
# Job Queue & Execution Manager
# ---------------------------------------------------------------------------


class JobManager:
    """In-memory state machine and manager for background evaluation jobs."""

    MAX_IN_MEMORY_JOBS = 1000

    def __init__(self, cache_manager: LocalCacheManager):
        self.jobs: dict[str, dict[str, Any]] = {}
        self.hash_to_job_id: dict[str, str] = {}
        self.cache_manager = cache_manager
        self._lock = threading.Lock()

    def create_job(
        self,
        eval_hash: str,
        config_dict: dict[str, Any],
        force_rerun: bool = False,
        user: UserContext | None = None,
    ) -> dict[str, Any]:
        user_info = (
            {
                "subject": user.subject,
                "email": user.email,
                "role": user.role,
                "client_id": user.client_id,
            }
            if user
            else None
        )
        with self._lock:
            # Evict oldest finished jobs if in-memory limit is reached
            if len(self.jobs) >= self.MAX_IN_MEMORY_JOBS:
                finished_ids = [
                    jid
                    for jid, j in self.jobs.items()
                    if j["status"] in (JobStatusEnum.COMPLETED, JobStatusEnum.FAILED)
                ]
                for jid in finished_ids[:200]:
                    evicted_job = self.jobs.pop(jid, None)
                    if (
                        evicted_job
                        and evicted_job.get("eval_hash") in self.hash_to_job_id
                    ):
                        del self.hash_to_job_id[evicted_job["eval_hash"]]

            # Check cache deduplication first
            if not force_rerun:
                cached_data = self.cache_manager.get(eval_hash)
                if cached_data:
                    telemetry_metrics.record_cache_hit()
                    cached_job_id = cached_data.get("job_id", str(uuid.uuid4()))
                    cached_job = {
                        "job_id": cached_job_id,
                        "status": JobStatusEnum.COMPLETED,
                        "created_at": cached_data.get("created_at", time.time()),
                        "completed_at": cached_data.get("completed_at", time.time()),
                        "cached": True,
                        "eval_hash": eval_hash,
                        "evaluation_time": cached_data.get("evaluation_time", 0.0),
                        "config_args": cached_data.get("config_args", config_dict),
                        "summary": cached_data.get("summary", {}),
                        "results": [],
                        "saved_to_db": cached_data.get("saved_to_db", False),
                        "user_info": cached_data.get("user_info", user_info),
                        "error": None,
                    }
                    self.jobs[cached_job_id] = cached_job
                    self.hash_to_job_id[eval_hash] = cached_job_id
                    return cached_job

            telemetry_metrics.record_cache_miss()

            job_id = str(uuid.uuid4())
            job = {
                "job_id": job_id,
                "status": JobStatusEnum.PENDING,
                "created_at": time.time(),
                "completed_at": None,
                "cached": False,
                "eval_hash": eval_hash,
                "evaluation_time": 0.0,
                "config_args": sanitize_config_args(config_dict),
                "summary": {},
                "results": [],
                "saved_to_db": config_dict.get("save_to_db", False),
                "user_info": user_info,
                "error": None,
            }
            self.jobs[job_id] = job
            self.hash_to_job_id[eval_hash] = job_id
            return job

    def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Thread-safely update fields on an existing job."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.update(updates)
                return dict(job)
            return None

    def mark_saved_to_db(self, job_id: str) -> None:
        """Thread-safely mark a job as persisted to PostgreSQL DB."""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["saved_to_db"] = True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                db_job = persistent_job_queue.get_job(job_id)
                if db_job:
                    job["status"] = db_job["status"]
                    if db_job.get("started_at"):
                        job["started_at"] = db_job["started_at"]
                    if db_job.get("completed_at"):
                        job["completed_at"] = db_job["completed_at"]
                    if db_job.get("error"):
                        job["error"] = db_job["error"]
                return dict(job)
        db_job = persistent_job_queue.get_job(job_id)
        if db_job:
            return {
                "job_id": job_id,
                "status": db_job["status"],
                "created_at": db_job.get("created_at", time.time()),
                "completed_at": db_job.get("completed_at"),
                "cached": False,
                "eval_hash": db_job.get("eval_hash", ""),
                "evaluation_time": 0.0,
                "config_args": db_job.get("config_args", {}),
                "summary": {},
                "results": [],
                "saved_to_db": False,
                "user_info": None,
                "error": db_job.get("error"),
            }
        cached_job = self.cache_manager.get_by_job_id(job_id)
        if cached_job:
            return {
                "job_id": job_id,
                "status": JobStatusEnum.COMPLETED,
                "created_at": cached_job.get("created_at", time.time()),
                "completed_at": cached_job.get("completed_at", time.time()),
                "cached": True,
                "eval_hash": cached_job.get("eval_hash", ""),
                "evaluation_time": cached_job.get("evaluation_time", 0.0),
                "config_args": cached_job.get("config_args", {}),
                "summary": cached_job.get("summary", {}),
                "results": [],
                "saved_to_db": cached_job.get("saved_to_db", False),
                "user_info": cached_job.get("user_info"),
                "error": None,
            }
        return None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            local_jobs = {j["job_id"]: dict(j) for j in self.jobs.values()}
        db_jobs = persistent_job_queue.list_jobs()
        for dj in db_jobs:
            jid = dj["job_id"]
            if jid in local_jobs:
                local_jobs[jid]["status"] = dj["status"]
                if dj.get("completed_at"):
                    local_jobs[jid]["completed_at"] = dj["completed_at"]
                if dj.get("error"):
                    local_jobs[jid]["error"] = dj["error"]
            else:
                local_jobs[jid] = {
                    "job_id": jid,
                    "status": dj["status"],
                    "created_at": dj.get("created_at", time.time()),
                    "completed_at": dj.get("completed_at"),
                    "cached": False,
                    "eval_hash": dj.get("eval_hash", ""),
                    "evaluation_time": 0.0,
                    "config_args": dj.get("config_args", {}),
                    "summary": {},
                    "results": [],
                    "saved_to_db": False,
                    "user_info": None,
                    "error": dj.get("error"),
                }
        return sorted(local_jobs.values(), key=lambda j: j["created_at"], reverse=True)

    def get_job_results_payload(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job and job.get("results"):
                return job["results"]
        return self.cache_manager.get_job_payload(job_id)


# Initialize global cache, DB manager, job manager and persistent queue
cache_manager = LocalCacheManager()
job_manager = JobManager(cache_manager)
db_manager = DatabaseManager()
persistent_job_queue = PersistentJobQueue(db_manager)

# ---------------------------------------------------------------------------
# Background Task Execution
# ---------------------------------------------------------------------------


def _build_job_summary(
    results: list[dict[str, Any]], eval_time: float
) -> dict[str, Any]:
    """Compute metrics aggregation stats and summary for job output."""
    from deepeval_eval.sinks import (
        calculate_latency_percentiles,
        categorize_failure_causes,
        compute_all_metric_averages,
    )

    latencies = [r.get("latency", 0.0) for r in results if "latency" in r]
    p50_latency, p95_latency = calculate_latency_percentiles(latencies)
    total_tokens_sum = sum(r.get("total_tokens", 0) for r in results)
    all_metric_averages = compute_all_metric_averages(results)
    failure_counts = categorize_failure_causes(results)

    evaluator_prompt_tokens = sum(r.get("evaluator_input_tokens", 0) for r in results)
    evaluator_completion_tokens = sum(
        r.get("evaluator_output_tokens", 0) for r in results
    )
    evaluator_total_tokens = evaluator_prompt_tokens + evaluator_completion_tokens

    return {
        "total_items": len(results),
        "evaluation_time_seconds": round(eval_time, 2),
        "p50_latency": round(p50_latency, 4),
        "p95_latency": round(p95_latency, 4),
        "total_tokens": total_tokens_sum,
        "metrics": all_metric_averages,
        "failure_causes": failure_counts,
        "deepeval_evaluator_usage": {
            "evaluation_time_seconds": round(eval_time, 2),
            "prompt_tokens": evaluator_prompt_tokens,
            "completion_tokens": evaluator_completion_tokens,
            "total_tokens": evaluator_total_tokens,
        },
    }


def execute_evaluation_job(
    job_id: str, req: EvaluationRequest, temp_file_path: str | None = None
) -> None:
    job = job_manager.get_job(job_id)
    if not job:
        return

    job_manager.update_job(job_id, {"status": JobStatusEnum.RUNNING})
    start_time = time.time()

    try:
        raw_qfile = getattr(req, "questions_file", None) or temp_file_path
        q_file = validate_safe_path(raw_qfile) if raw_qfile else None
        p_config = SERVER_PROMPT_CONFIG
        results_dir = DEFAULT_RESULTS_DIR
        g_config = DEFAULT_GATE_CONFIG

        q_ids_str = (
            ",".join(req.question_ids)
            if isinstance(req.question_ids, list)
            else req.question_ids
        )
        q_idx_str = (
            ",".join(str(i) for i in req.question_indices)
            if isinstance(req.question_indices, list)
            else req.question_indices
        )

        eval_config = EvalConfig(
            dataset_name=req.dataset_name,
            answer_mode=req.answer_mode,
            datasource_id=req.datasource_id,
            data_dir=DEFAULT_DATA_DIR,
            questions_file=q_file,
            prompt_style=req.prompt_style,
            prompt_config=p_config,
            max_items=req.max_items,
            limit_per_category=req.limit_per_category,
            top_k=req.top_k,
            max_context_chars=req.max_context_chars,
            llm_base_url=req.llm_base_url,
            llm_api_key=req.llm_api_key,
            llm_model=req.llm_model,
            agentic=req.agentic,
            trace_log=req.trace_log,
            agent_id=req.agent_id,
            supervisor_url=req.supervisor_url,
            fail_on_error=req.fail_on_error,
            oracle_retrieval=req.oracle_retrieval,
            oracle_testing=req.oracle_testing,
            gate=req.gate,
            gate_config=g_config,
            results_dir=results_dir,
            question_ids=q_ids_str,
            question_indices=q_idx_str,
            save_to_db=req.save_to_db,
        )

        rag_client = _build_rag_client(eval_config)

        results = run_evaluation(eval_config, rag_client=rag_client)

        end_time = time.time()
        eval_time = end_time - start_time
        telemetry_metrics.record_evaluation(eval_time)

        summary = _build_job_summary(results, eval_time)

        updated_job = job_manager.update_job(
            job_id,
            {
                "status": JobStatusEnum.COMPLETED,
                "completed_at": end_time,
                "evaluation_time": eval_time,
                "results": results,
                "summary": summary,
            },
        )

        if eval_config.save_to_db and results:
            try:
                sink = PostgresResultSink(db_manager=db_manager)
                sink.save(
                    results_dir=Path(DEFAULT_RESULTS_DIR),
                    prefix=eval_config.dataset_name or "enterprise",
                    results=results,
                    evaluation_time=eval_time,
                    config_args=eval_config.to_config_args(),
                )
                job_manager.mark_saved_to_db(job_id)
                logger.info(
                    f"Auto-saved completed evaluation job '{job_id}' to PostgreSQL DB."
                )
            except Exception as db_err:
                logger.warning(
                    f"Auto-saving results for job '{job_id}' to PostgreSQL DB failed: {db_err}"
                )

        if updated_job:
            cache_manager.set(updated_job["eval_hash"], updated_job)
            job_manager.update_job(job_id, {"results": []})

    except Exception as e:
        job_manager.update_job(
            job_id,
            {
                "status": JobStatusEnum.FAILED,
                "completed_at": time.time(),
                "error": str(e),
            },
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                parent_dir = Path(temp_file_path).parent.resolve()
                system_temp = Path(tempfile.gettempdir()).resolve()
                if (
                    parent_dir.exists()
                    and (system_temp in parent_dir.parents or parent_dir == system_temp)
                    and parent_dir.name.startswith("eval_upload_")
                ):
                    shutil.rmtree(parent_dir, ignore_errors=True)
            except Exception as cleanup_err:
                logger.warning(
                    f"Failed to clean up temporary upload directory: {cleanup_err}"
                )


def _run_queued_evaluation(job_id: str, config_dict: dict[str, Any]) -> None:
    temp_file_str = config_dict.get("questions_file")
    temp_file_path = Path(temp_file_str) if temp_file_str else None
    req = EvaluationRequest(
        **{k: v for k, v in config_dict.items() if k in EvaluationRequest.model_fields}
    )
    try:
        execute_evaluation_job(job_id, req, temp_file_path=temp_file_path)
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                parent_dir = temp_file_path.parent
                system_temp = Path(tempfile.gettempdir())
                if (
                    parent_dir.exists()
                    and (system_temp in parent_dir.parents or parent_dir == system_temp)
                    and parent_dir.name.startswith("eval_upload_")
                ):
                    shutil.rmtree(parent_dir, ignore_errors=True)
                elif temp_file_path.is_file():
                    temp_file_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning(
                    f"Failed to clean up temporary upload directory in queued task: {cleanup_err}"
                )


persistent_job_queue.set_task_executor(_run_queued_evaluation)


@asynccontextmanager
async def lifespan(app: FastAPI):
    persistent_job_queue.start()
    try:
        yield
    finally:
        persistent_job_queue.stop()


# ---------------------------------------------------------------------------
# FastAPI Application Definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CAIPE DeepEval REST API Evaluation Service",
    description=(
        "REST API service to trigger evaluation pipelines, submit datasets, "
        "manage async evaluation jobs, poll execution results, query PostgreSQL "
        "evaluation runs, and leverage 24-hour evaluation caching."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Initialize CAIPE OpenTelemetry tracing exporter if configured via environment
setup_otlp_tracing(app)


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    endpoint = request.url.path
    telemetry_metrics.record_http_request(endpoint, response.status_code)
    return response


@app.get("/", summary="Root Endpoint", include_in_schema=False)
def root_endpoint(
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return {
        "service": "CAIPE DeepEval REST API Evaluation Service",
        "version": "0.1.0",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


# Mount Health & Telemetry APIRouter (/healthz, /livez, /readyz, /health, /metrics)
app.include_router(telemetry_router)
app.include_router(question_sets_router, prefix="/api/v1")


def _prepare_job_from_question_set(
    set_id: int, request: EvaluationRequest, user: UserContext
) -> JobResponse:
    qset = db_manager.questions.get_question_set(set_id)
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set with ID={set_id} not found in database.",
        )

    res = db_manager.questions.list_questions(set_id=set_id, page=1, limit=10000)
    questions = res["items"]
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Question set ID={set_id} contains no questions.",
        )

    lines: list[str] = []
    for q in questions:
        row = {
            "question_id": q.get("question_id"),
            "user_input": q.get("input"),
            "reference": q.get("expected_output"),
            "category": q.get("category"),
            "level": q.get("level"),
            "expected_doc_ids": q.get("expected_doc_ids") or [],
        }
        if q.get("context"):
            row["context"] = q["context"]
        if q.get("extra") and isinstance(q["extra"], dict):
            row.update(q["extra"])
        lines.append(json.dumps(row) + "\n")

    jsonl_bytes = "".join(lines).encode("utf-8")

    temp_dir = tempfile.mkdtemp(prefix="eval_question_set_")
    set_hash = hashlib.sha256(jsonl_bytes).hexdigest()[:16]
    temp_file_path = os.path.join(temp_dir, f"qset_{set_id}_{set_hash}.jsonl")
    validate_safe_path(temp_file_path)
    with open(temp_file_path, "wb") as f:
        f.write(jsonl_bytes)

    if request.dataset_name == "enterprise" and qset.get("name"):
        request.dataset_name = qset["name"]

    request.question_set_id = set_id

    config_dict = request.model_dump()
    config_dict["questions_file"] = temp_file_path
    eval_hash = compute_eval_hash(config_dict, dataset_bytes=jsonl_bytes)

    job = job_manager.create_job(
        eval_hash, config_dict, force_rerun=request.force_rerun, user=user
    )

    if job["cached"]:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return JobResponse(**job)

    persistent_job_queue.enqueue(job["job_id"], eval_hash, config_dict)
    return JobResponse(**job)


@app.post(
    "/eval/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Evaluation Job",
    tags=["Evaluation Jobs"],
)
def submit_eval_job(
    request: EvaluationRequest,
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Submit an evaluation job asynchronously using JSON request parameters."""
    if request.question_set_id is not None:
        return _prepare_job_from_question_set(request.question_set_id, request, user)

    config_dict = request.model_dump()
    eval_hash = compute_eval_hash(config_dict)

    job = job_manager.create_job(
        eval_hash, config_dict, force_rerun=request.force_rerun, user=user
    )

    if job["cached"]:
        return JobResponse(**job)

    persistent_job_queue.enqueue(job["job_id"], eval_hash, config_dict)
    return JobResponse(**job)


@app.post(
    "/eval/jobs/question-sets/{set_id}",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Evaluation Job for Question Set",
    tags=["Evaluation Jobs"],
)
def submit_eval_job_for_question_set(
    set_id: int,
    request: EvaluationRequest | None = None,
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Submit an evaluation job targeting a Question Set stored in Question Set Manager."""
    req = request or EvaluationRequest()
    req.question_set_id = set_id
    return _prepare_job_from_question_set(set_id, req, user)


@app.post(
    "/eval/jobs/upload",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Evaluation Job with Dataset File Upload",
    tags=["Evaluation Jobs"],
)
async def submit_eval_job_with_upload(
    file: UploadFile = File(..., description="Dataset file (JSON/CSV)"),
    dataset_name: str = Query("custom_upload", description="Dataset name"),
    answer_mode: str = Query(
        "generate", description="Answer mode: generate or ground_truth"
    ),
    oracle_testing: bool = Query(
        False,
        description="Shortcut flag to enable oracle_retrieval and ground_truth answer mode",
    ),
    datasource_id: str | None = Query(None, description="Target CAIPE datasource ID"),
    max_items: int | None = Query(None, description="Maximum items to evaluate"),
    limit_per_category: int | None = Query(
        None, description="Limit items per category"
    ),
    top_k: int = Query(3, description="Top-k documents"),
    max_context_chars: int = Query(12000, description="Max context characters"),
    agentic: bool = Query(
        False, description="Route queries through CAIPE supervisor A2A endpoint"
    ),
    supervisor_url: str | None = Query(None, description="CAIPE supervisor URL"),
    save_to_db: bool = Query(False, description="Persist results to DB"),
    force_rerun: bool = Query(False, description="Force rerun ignoring cache"),
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Submit an evaluation job by uploading a dataset file (multipart/form-data)."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    temp_dir = tempfile.mkdtemp(prefix="eval_upload_")
    ext = (
        Path(file.filename).suffix.lower()
        if file.filename
        and Path(file.filename).suffix.lower() in (".json", ".csv", ".jsonl")
        else ".json"
    )
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    temp_file_path = os.path.join(temp_dir, f"upload_{file_hash}{ext}")
    validate_safe_path(temp_file_path)
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)

    req = EvaluationRequest(
        dataset_name=dataset_name,
        answer_mode=answer_mode,
        datasource_id=datasource_id,
        max_items=max_items,
        limit_per_category=limit_per_category,
        top_k=top_k,
        max_context_chars=max_context_chars,
        agentic=agentic,
        supervisor_url=supervisor_url,
        save_to_db=save_to_db,
        force_rerun=force_rerun,
        oracle_testing=oracle_testing,
    )
    config_dict = req.model_dump()
    config_dict["questions_file"] = temp_file_path
    eval_hash = compute_eval_hash(config_dict, dataset_bytes=file_bytes)

    job = job_manager.create_job(
        eval_hash, config_dict, force_rerun=force_rerun, user=user
    )

    if job["cached"]:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return JobResponse(**job)

    persistent_job_queue.enqueue(job["job_id"], eval_hash, config_dict)
    return JobResponse(**job)


@app.get(
    "/jobs",
    response_model=list[JobResponse],
    summary="List Evaluation Jobs",
    tags=["Evaluation Jobs"],
)
def list_jobs(
    user: UserContext = Depends(get_current_user),
) -> list[JobResponse]:
    """List all submitted evaluation jobs and their current status."""
    return [JobResponse(**j) for j in job_manager.list_jobs()]


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Poll Job Status",
    tags=["Evaluation Jobs"],
)
def get_job_status(
    job_id: str,
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Retrieve status and metadata for a specific job ID."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobResponse(**job)


@app.get(
    "/jobs/{job_id}/results",
    summary="Get Evaluation Job Results",
    tags=["Evaluation Results"],
)
def get_job_results(
    job_id: str,
    format: str = Query("json", description="Output format: 'json' or 'csv'"),
    user: UserContext = Depends(get_current_user),
) -> Any:
    """Retrieve evaluation results for a completed job in JSON or CSV format."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] == JobStatusEnum.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Job '{job_id}' failed with error: {job.get('error')}",
        )

    if job["status"] != JobStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is still in status '{job['status']}'",
        )

    results = job_manager.get_job_results_payload(job_id)

    requested_format = format.lower()
    if requested_format not in ("json", "csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Supported: 'json', 'csv'.",
        )

    if requested_format == "csv":
        datasource = job.get("config_args", {}).get("dataset_name", "enterprise")
        evaluation_time = job.get("evaluation_time", 0.0)
        csv_content = format_results_as_csv(
            results=results,
            evaluation_time=evaluation_time,
            datasource=datasource,
        )
        headers = {
            "Content-Disposition": f"attachment; filename=job_{job_id}_results.csv"
        }

        def generate_csv_chunks() -> Iterator[bytes]:
            yield csv_content.encode("utf-8")

        return StreamingResponse(
            generate_csv_chunks(),
            media_type="text/csv",
            headers=headers,
        )

    job_data = dict(job)
    safe_results = results or []
    if safe_results and (
        not job_data.get("summary") or "metrics" not in job_data.get("summary", {})
    ):
        job_data["summary"] = _build_job_summary(
            safe_results, job.get("evaluation_time", 0.0)
        )

    def generate_json_chunks() -> Iterator[bytes]:
        meta = {
            "job_id": job_data.get("job_id"),
            "status": job_data.get("status"),
            "created_at": job_data.get("created_at"),
            "completed_at": job_data.get("completed_at"),
            "cached": job_data.get("cached", False),
            "eval_hash": job_data.get("eval_hash", ""),
            "evaluation_time": job_data.get("evaluation_time", 0.0),
            "config_args": job_data.get("config_args", {}),
            "summary": job_data.get("summary", {}),
            "saved_to_db": job_data.get("saved_to_db", False),
            "user_info": job_data.get("user_info"),
        }
        safe_meta = jsonable_encoder(meta)
        meta_json = json.dumps(safe_meta, ensure_ascii=False)
        prefix = (
            meta_json[:-1] + ',"results":['
            if meta_json.endswith("}")
            else meta_json + ',"results":['
        )
        yield prefix.encode("utf-8")

        for idx, item in enumerate(safe_results):
            chunk = ("," if idx > 0 else "") + json.dumps(
                item, ensure_ascii=False, default=str
            )
            yield chunk.encode("utf-8")

        yield b"]}"

    return StreamingResponse(
        generate_json_chunks(),
        media_type="application/json",
    )


def format_summary_as_csv(job_id: str, job_data: dict[str, Any]) -> str:
    """Format evaluation summary metadata and aggregated metrics into CSV string representation."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    summary = job_data.get("summary", {})
    metrics = summary.get("metrics", {})

    headers = [
        "job_id",
        "status",
        "evaluation_time_seconds",
        "total_items",
        "p50_latency",
        "p95_latency",
        "total_tokens",
    ] + list(metrics.keys())

    values = [
        job_id,
        job_data.get("status", ""),
        job_data.get("evaluation_time", 0.0),
        summary.get("total_items", 0),
        summary.get("p50_latency", 0.0),
        summary.get("p95_latency", 0.0),
        summary.get("total_tokens", 0),
    ] + [metrics[k] for k in metrics]

    writer.writerow(headers)
    writer.writerow(values)
    return output.getvalue()


@app.get(
    "/jobs/{job_id}/summary",
    summary="Get Evaluation Job Summary Only",
    tags=["Evaluation Results"],
)
def get_job_summary(
    job_id: str,
    format: str = Query("json", description="Output format: 'json' or 'csv'"),
    user: UserContext = Depends(get_current_user),
) -> Any:
    """Retrieve only the summary metadata and aggregated metrics for a completed job in JSON or CSV format."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] == JobStatusEnum.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Job '{job_id}' failed with error: {job.get('error')}",
        )

    if job["status"] != JobStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is still in status '{job['status']}'",
        )

    requested_format = format.lower()
    if requested_format not in ("json", "csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Supported: 'json', 'csv'.",
        )

    results = job_manager.get_job_results_payload(job_id)
    job_data = dict(job)
    job_data.pop("results", None)

    if results and (
        not job_data.get("summary") or "metrics" not in job_data.get("summary", {})
    ):
        job_data["summary"] = _build_job_summary(
            results, job.get("evaluation_time", 0.0)
        )

    if requested_format == "csv":
        csv_content = format_summary_as_csv(job_id, job_data)
        headers = {
            "Content-Disposition": f"attachment; filename=job_{job_id}_summary.csv"
        }
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers=headers,
        )

    return EvaluationSummaryResponse(**job_data)


@app.post(
    "/jobs/{job_id}/save-db",
    summary="Save Completed Job Results to Database",
    tags=["Evaluation Results"],
)
def save_job_results_to_db(
    job_id: str,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Persist completed job results to PostgreSQL database on demand."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] != JobStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=400, detail=f"Job '{job_id}' is in status '{job['status']}'"
        )

    results = job_manager.get_job_results_payload(job_id)
    if not results:
        raise HTTPException(
            status_code=400, detail="No evaluation results found for job"
        )

    try:
        sink = PostgresResultSink(db_manager=db_manager)
        sink.save(
            results_dir=Path(DEFAULT_RESULTS_DIR),
            prefix=job["config_args"].get("dataset_name", "enterprise"),
            results=results,
            evaluation_time=job.get("evaluation_time", 0.0),
            config_args=job["config_args"],
        )
        job_manager.mark_saved_to_db(job_id)
        job["saved_to_db"] = True
        cache_manager.set(job["eval_hash"], job)
        return {
            "job_id": job_id,
            "status": "success",
            "message": "Evaluation results successfully saved to PostgreSQL database",
        }
    except Exception as e:
        logger.exception(f"Failed to persist results for job '{job_id}' to DB: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to persist results to PostgreSQL DB: {e}"
        )


@app.get(
    "/results/db",
    summary="Query Database Evaluation Runs",
    tags=["Evaluation Results"],
)
def query_db_evaluation_runs(
    limit: int = Query(10, ge=1, le=100),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Query recent evaluation experiment runs stored in PostgreSQL database."""
    try:
        sink = PostgresResultSink(db_manager=db_manager)
        runs = sink.query_runs(limit=limit)

        return {"count": len(runs), "runs": runs}
    except Exception as e:
        logger.exception(f"Failed to query database evaluation runs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to query database evaluation runs: {e}"
        )


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """CLI launcher for starting the Uvicorn ASGI server."""
    import uvicorn

    uvicorn.run("deepeval_eval.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
