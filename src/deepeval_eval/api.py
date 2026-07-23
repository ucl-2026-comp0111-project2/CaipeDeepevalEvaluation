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
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from deepeval_eval.auth import UserContext, get_current_user
from deepeval_eval.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GATE_CONFIG,
    DEFAULT_RESULTS_DIR,
    load_dotenv_loose,
)
from deepeval_eval.eval_engine import EvalConfig, _build_rag_client, run_evaluation
from deepeval_eval.io_utils import sanitize_path
from deepeval_eval.prompt_style import DEFAULT_PROMPT_STYLE
from deepeval_eval.sinks import DatabaseResultSink
from deepeval_eval.sinks.file_sink import format_results_as_csv

logger = logging.getLogger(__name__)

# Server-level configuration read from environment at startup
_env_prompt_config = os.environ.get("DEEPEVAL_PROMPT_CONFIG") or os.environ.get(
    "PROMPT_CONFIG"
)
SERVER_PROMPT_CONFIG: Path | None = (
    Path(_env_prompt_config).resolve() if _env_prompt_config else None
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


# ---------------------------------------------------------------------------
# Deduplication Hashing & Cache Management (Cache-Aside Pattern)
# ---------------------------------------------------------------------------


def validate_safe_path(user_path: str | Path | None) -> Path | None:
    """Validate that server temporary files reside within the system temporary directory."""
    if not user_path:
        return None
    path_obj = Path(user_path).expanduser().resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if path_obj != temp_dir and temp_dir not in path_obj.parents:
        raise HTTPException(
            status_code=400,
            detail=f"Access to file path '{user_path}' is restricted for security.",
        )
    return path_obj


def sanitize_config_args(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Sanitize configuration fields to prevent credential leakage in outputs."""
    sensitive_keys = {
        "llm_api_key",
        "auth_token",
        "client_secret",
        "db_connection_string",
        "env_file",
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
                    self.jobs.pop(jid, None)

            # Check cache deduplication first
            if not force_rerun:
                cached_data = self.cache_manager.get(eval_hash)
                if cached_data:
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
                return dict(job)
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
            return [
                dict(j)
                for j in sorted(
                    self.jobs.values(), key=lambda j: j["created_at"], reverse=True
                )
            ]

    def get_job_results_payload(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job and job.get("results"):
                return job["results"]
        return self.cache_manager.get_job_payload(job_id)


# Initialize global cache and job manager
cache_manager = LocalCacheManager()
job_manager = JobManager(cache_manager)

# ---------------------------------------------------------------------------
# Background Task Execution
# ---------------------------------------------------------------------------


def execute_evaluation_job(
    job_id: str, req: EvaluationRequest, temp_file_path: str | None = None
) -> None:
    job = job_manager.get_job(job_id)
    if not job:
        return

    job_manager.update_job(job_id, {"status": JobStatusEnum.RUNNING})
    start_time = time.time()

    try:
        q_file = validate_safe_path(temp_file_path) if temp_file_path else None
        p_config = SERVER_PROMPT_CONFIG
        env_file = DEFAULT_ENV_FILE
        results_dir = DEFAULT_RESULTS_DIR
        g_config = DEFAULT_GATE_CONFIG

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
            supervisor_url=req.supervisor_url,
            fail_on_error=req.fail_on_error,
            oracle_retrieval=req.oracle_retrieval,
            oracle_testing=req.oracle_testing,
            gate=req.gate,
            gate_config=g_config,
            env_file=env_file,
            results_dir=results_dir,
            question_ids=req.question_ids,
            question_indices=req.question_indices,
            save_to_db=req.save_to_db,
        )

        env_values = load_dotenv_loose(eval_config.env_file)
        rag_client = _build_rag_client(eval_config, env_values)

        results = run_evaluation(eval_config, rag_client=rag_client)

        end_time = time.time()
        eval_time = end_time - start_time

        updated_job = job_manager.update_job(
            job_id,
            {
                "status": JobStatusEnum.COMPLETED,
                "completed_at": end_time,
                "evaluation_time": eval_time,
                "results": results,
                "summary": {
                    "total_items": len(results),
                    "evaluation_time_seconds": round(eval_time, 2),
                },
            },
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
)


@app.get("/", summary="Root Endpoint")
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


@app.get("/health", summary="Health Check")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/eval/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Evaluation Job",
)
def submit_eval_job(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Submit an evaluation job asynchronously using JSON request parameters."""
    config_dict = request.model_dump()
    eval_hash = compute_eval_hash(config_dict)

    job = job_manager.create_job(
        eval_hash, config_dict, force_rerun=request.force_rerun, user=user
    )

    if job["cached"]:
        return JobResponse(**job)

    background_tasks.add_task(execute_evaluation_job, job["job_id"], request)
    return JobResponse(**job)


@app.post(
    "/eval/jobs/upload",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Evaluation Job with Dataset File Upload",
)
async def submit_eval_job_with_upload(
    background_tasks: BackgroundTasks,
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
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)

    req = EvaluationRequest(
        dataset_name=dataset_name,
        answer_mode=answer_mode,
        datasource_id=datasource_id,
        questions_file=temp_file_path,
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
    eval_hash = compute_eval_hash(config_dict, dataset_bytes=file_bytes)

    job = job_manager.create_job(
        eval_hash, config_dict, force_rerun=force_rerun, user=user
    )

    if job["cached"]:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return JobResponse(**job)

    background_tasks.add_task(
        execute_evaluation_job, job["job_id"], req, temp_file_path
    )
    return JobResponse(**job)


@app.get(
    "/jobs",
    response_model=list[JobResponse],
    summary="List Evaluation Jobs",
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
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers=headers,
        )

    job_data = dict(job)
    job_data["results"] = results
    return EvaluationResultsResponse(**job_data)


@app.post(
    "/jobs/{job_id}/save-db",
    summary="Save Completed Job Results to Database",
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
        sink = DatabaseResultSink()
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
)
def query_db_evaluation_runs(
    limit: int = Query(10, ge=1, le=100),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Query recent evaluation experiment runs stored in PostgreSQL database."""
    try:
        load_dotenv_loose(DEFAULT_ENV_FILE)
        sink = DatabaseResultSink()
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

    uvicorn.run("deepeval_eval.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
