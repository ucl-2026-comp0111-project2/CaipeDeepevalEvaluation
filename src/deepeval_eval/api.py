from __future__ import annotations

import hashlib
import json
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
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

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
        default="reference",
        description="Evaluation answer mode: 'reference' or 'generate'",
    )
    datasource_id: str | None = Field(
        default=None, description="Target CAIPE datasource ID"
    )
    questions_file: str | None = Field(
        default=None, description="Path to custom questions dataset file"
    )
    prompt_style: str | None = Field(
        default=DEFAULT_PROMPT_STYLE,
        description="Prompt style (e.g. generation, short, or custom)",
    )
    prompt_config: str | None = Field(
        default=None, description="Path to custom prompt style YAML/JSON config"
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
    precompute: bool = Field(
        default=False, description="Run precomputed benchmark (gold retrieval)"
    )
    gate: bool = Field(default=False, description="Apply quality gate after evaluation")
    gate_config: str | None = Field(
        default=None, description="Path to quality gate YAML config"
    )
    save_to_db: bool = Field(
        default=False, description="Persist evaluation results to PostgreSQL DB"
    )
    force_rerun: bool = Field(
        default=False,
        description="Bypass evaluation deduplication cache and force rerun",
    )
    env_file: str | None = Field(
        default=None, description="Path to .env configuration file"
    )
    results_dir: str | None = Field(
        default=None, description="Directory to store result artifacts"
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


# ---------------------------------------------------------------------------
# Deduplication Hashing & Cache Management (Cache-Aside Pattern)
# ---------------------------------------------------------------------------


def sanitize_config_args(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Sanitize configuration fields to prevent credential leakage and remove clutter in API outputs."""
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
    """Compute a deterministic SHA-256 fingerprint for evaluation parameters and dataset."""
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
    """Manages local 24-hour file cache for evaluation results and disk-backed job payloads."""

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
        # Store results payload separately to keep cache metadata lean
        results = payload.pop("results", None)
        if results is not None:
            self.save_job_payload(payload.get("job_id", eval_hash), results)
        payload["timestamp"] = time.time()
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            print(f"Warning: Failed to write to evaluation cache: {e}")

    def save_job_payload(self, job_id: str, results: list[dict[str, Any]]) -> None:
        """Persist full job evaluation results array to disk cache."""
        path = self._get_job_payload_path(job_id)
        try:
            path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            print(f"Warning: Failed to write job payload to disk: {e}")

    def get_job_payload(self, job_id: str) -> list[dict[str, Any]]:
        """Load full evaluation results array from disk cache."""
        path = self._get_job_payload_path(job_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def purge_expired(self) -> int:
        """Purge entries older than 24 hours."""
        purged = 0
        now = time.time()
        for p in self.cache_dir.glob("*.json"):
            try:
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

    def __init__(self, cache_manager: LocalCacheManager):
        self.jobs: dict[str, dict[str, Any]] = {}
        self.hash_to_job_id: dict[str, str] = {}
        self.cache_manager = cache_manager
        self._lock = threading.Lock()

    def create_job(
        self, eval_hash: str, config_dict: dict[str, Any], force_rerun: bool = False
    ) -> dict[str, Any]:
        with self._lock:
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
                "error": None,
            }
            self.jobs[job_id] = job
            self.hash_to_job_id[eval_hash] = job_id
            return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

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

    job["status"] = JobStatusEnum.RUNNING
    start_time = time.time()

    try:
        q_file = (
            Path(temp_file_path)
            if temp_file_path
            else (Path(req.questions_file) if req.questions_file else None)
        )
        p_config = Path(req.prompt_config) if req.prompt_config else None
        env_file = Path(req.env_file) if req.env_file else DEFAULT_ENV_FILE
        results_dir = Path(req.results_dir) if req.results_dir else DEFAULT_RESULTS_DIR
        g_config = Path(req.gate_config) if req.gate_config else DEFAULT_GATE_CONFIG

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
            precompute=req.precompute,
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

        job["status"] = JobStatusEnum.COMPLETED
        job["completed_at"] = end_time
        job["evaluation_time"] = eval_time
        job["results"] = results
        job["summary"] = {
            "total_items": len(results),
            "evaluation_time_seconds": round(eval_time, 2),
        }

        # Persist full results array to disk cache to keep in-memory footprint lean
        cache_manager.set(job["eval_hash"], job)
        job["results"] = []
        with job_manager._lock:
            job_manager.jobs[job_id] = dict(job)

    except Exception as e:
        job["status"] = JobStatusEnum.FAILED
        job["completed_at"] = time.time()
        job["error"] = str(e)
        with job_manager._lock:
            job_manager.jobs[job_id] = dict(job)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                parent_dir = os.path.dirname(temp_file_path)
                if (
                    os.path.exists(parent_dir)
                    and tempfile.gettempdir() in parent_dir
                    and "eval_upload_" in os.path.basename(parent_dir)
                ):
                    shutil.rmtree(parent_dir, ignore_errors=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# FastAPI Application Definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CAIPE DeepEval REST API Evaluation Service",
    description=(
        "REST API service to trigger evaluation pipelines, submit datasets, manage async evaluation jobs, "
        "poll execution results, query PostgreSQL evaluation runs, and leverage 24-hour evaluation caching with deduplication."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", summary="Root Endpoint")
def root_endpoint() -> dict[str, Any]:
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
) -> JobResponse:
    """Submit an evaluation job asynchronously using JSON request parameters."""
    config_dict = request.model_dump()
    eval_hash = compute_eval_hash(config_dict)

    job = job_manager.create_job(
        eval_hash, config_dict, force_rerun=request.force_rerun
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
        "reference", description="Answer mode: reference or generate"
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
    env_file: str | None = Query(None, description="Path to .env configuration file"),
    save_to_db: bool = Query(False, description="Persist results to DB"),
    force_rerun: bool = Query(False, description="Force rerun ignoring cache"),
) -> JobResponse:
    """Submit an evaluation job by uploading a dataset file (multipart/form-data)."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename or "dataset.json")
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
        env_file=env_file,
        save_to_db=save_to_db,
        force_rerun=force_rerun,
    )
    config_dict = req.model_dump()
    eval_hash = compute_eval_hash(config_dict, dataset_bytes=file_bytes)

    job = job_manager.create_job(eval_hash, config_dict, force_rerun=force_rerun)

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
def list_jobs() -> list[JobResponse]:
    """List all submitted evaluation jobs and their current status."""
    return [JobResponse(**j) for j in job_manager.list_jobs()]


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Poll Job Status",
)
def get_job_status(job_id: str) -> JobResponse:
    """Retrieve status and metadata for a specific job ID."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobResponse(**job)


@app.get(
    "/jobs/{job_id}/results",
    response_model=EvaluationResultsResponse,
    summary="Get Evaluation Job Results",
)
def get_job_results(job_id: str) -> EvaluationResultsResponse:
    """Retrieve evaluation results and detailed metrics for a completed job."""
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

    job_data = dict(job)
    job_data["results"] = job_manager.get_job_results_payload(job_id)
    return EvaluationResultsResponse(**job_data)


@app.post(
    "/jobs/{job_id}/save-db",
    summary="Save Completed Job Results to Database",
)
def save_job_results_to_db(job_id: str) -> dict[str, Any]:
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
        job["saved_to_db"] = True
        cache_manager.set(job["eval_hash"], job)
        return {
            "job_id": job_id,
            "status": "success",
            "message": "Evaluation results successfully saved to PostgreSQL database",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to persist results to PostgreSQL DB: {e}"
        )


@app.get(
    "/results/db",
    summary="Query Database Evaluation Runs",
)
def query_db_evaluation_runs(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    """Query recent evaluation experiment runs stored in PostgreSQL database."""
    try:
        load_dotenv_loose(DEFAULT_ENV_FILE)
        sink = DatabaseResultSink()
        runs = sink.query_runs(limit=limit)
        return {"count": len(runs), "runs": runs}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to query database evaluation runs: {e}"
        )


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """CLI launcher for starting the Uvicorn ASGI server."""
    import uvicorn

    uvicorn.run("deepeval_eval.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
