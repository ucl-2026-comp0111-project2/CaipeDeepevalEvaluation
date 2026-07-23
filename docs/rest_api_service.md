# REST API Evaluation & Results Management Service (`api.py`)

The REST API Evaluation Service provides an asynchronous HTTP REST interface and OpenAPI/Swagger documentation (`/docs`, `/redoc`) for running evaluation pipelines, submitting custom datasets, managing background evaluation jobs, querying results, downloading CSV reports, and persisting evaluation runs in PostgreSQL.

---

## Architectural Overview & Design Patterns

The service is implemented in [`src/deepeval_eval/api.py`](file:///Users/alexanghh/development/CaipeDeepevalEvaluation/src/deepeval_eval/api.py) using **FastAPI** and **ASGI (`uvicorn`)**. It incorporates key enterprise software design patterns:

### 1. Repository / Sink Pattern (Data Abstraction)
- **Concept**: Decouples evaluation business logic from persistent storage mechanisms.
- **Implementation**: Uses `ResultSink` Protocol with `FileResultSink` (local 24-hour JSON/CSV artifact generation) and `DatabaseResultSink` (direct PostgreSQL persistence).
- **Benefit**: Endpoints querying evaluation outputs operate on uniform result objects regardless of storage backend.

### 2. Job Queue & State Machine Pattern (Asynchronous Operations)
- **Concept**: Long-running LLM evaluations execute asynchronously via background tasks (`fastapi.BackgroundTasks`) without blocking client HTTP connections or risking gateway timeouts.
- **State Machine Transitions**:
  $$\text{PENDING} \longrightarrow \text{RUNNING} \longrightarrow \text{COMPLETED} \quad / \quad \text{FAILED}$$
- **Workflow**: `JobManager` manages job states in memory and disk (`.cache/eval_results/job_payloads`), returning an immediate `202 Accepted` response containing a unique `job_id`.

### 3. Cache-Aside / Memoization Pattern (Evaluation Deduplication)
- **Concept**: Prevents redundant LLM evaluation calls when parameters and dataset content are unchanged.
- **Fingerprinting Algorithm**:
  $$\text{eval\_hash} = \text{SHA256}(\text{normalized\_config\_json} + \text{dataset\_bytes})[:16]$$
- **Workflow**:
  - `JobManager` checks `LocalCacheManager` (24-hour TTL) for matching `eval_hash`.
  - If a valid cache entry exists and `force_rerun=False`, the API immediately returns a completed job response (`cached=True`).
  - Setting `force_rerun=True` invalidates cache and forces fresh evaluation.

### 4. DTO (Data Transfer Object) & Schema Validation Pattern
- **Concept**: Strongly typed Pydantic request/response DTO models (`EvaluationRequest`, `JobResponse`, `EvaluationResultsResponse`).
- **Benefit**: Enforces strict input validation constraints (`max_items >= 1`, `top_k >= 1`, `max_context_chars >= 100`) and auto-generates interactive Swagger UI at `/docs` and ReDoc at `/redoc`.

---

## Security & Authentication

The API integrates with [`deepeval_eval.auth`](file:///Users/alexanghh/development/CaipeDeepevalEvaluation/src/deepeval_eval/auth.py) for token verification, static key validation, and role-based access control.

### Supported Authentication Methods

| Auth Header | Format / Parameter | Description |
| :--- | :--- | :--- |
| `Authorization` | `Bearer <TOKEN>` | OIDC JWT token or static API key. |
| `X-API-Key` | `<API_KEY>` | Static API key string matching `DEEPEVAL_API_KEY` or `API_KEY` in environment. |

### Environment Configuration

- **Static Key Auth**: Set `DEEPEVAL_API_KEY=your_secret_key` in `.env`.
- **OIDC JWT Auth**: Set `OIDC_ISSUER_URL` and `OIDC_AUDIENCE`.
- **Local Dev Bypass**: Set `ALLOW_UNAUTHENTICATED_ACCESS=true` (default in local dev when no auth key is configured).

---

## Data Transfer Objects & Schemas

### `JobStatusEnum`

Enum representing evaluation job status:
- `"pending"`: Job queued, waiting for background task execution.
- `"running"`: Evaluation pipeline currently executing.
- `"completed"`: Job completed successfully. Results available.
- `"failed"`: Execution failed due to an exception. Error message recorded.

---

### `EvaluationRequest` (Request Body Schema)

JSON payload used when submitting evaluation jobs via `POST /eval/jobs`.

| Field Name | Type | Default | Validation / Constraint | Description |
| :--- | :--- | :--- | :--- | :--- |
| `dataset_name` | `str` | `"enterprise"` | - | Benchmark dataset name (e.g. `"enterprise"`, `"hotpotqa"`, or custom). |
| `answer_mode` | `str` | `"generate"` | `"generate"` \| `"ground_truth"` | Mode for answer generation. |
| `oracle_testing` | `bool` | `false` | - | Enables oracle retrieval and ground truth answer mode. |
| `datasource_id` | `str \| null` | `null` | - | Target CAIPE RAG datasource ID. |
| `prompt_style` | `str \| null` | `"generation"` | - | Prompt style (`"generation"`, `"short"`, custom). |
| `max_items` | `int \| null` | `null` | `ge=1` | Maximum total evaluation items to process. |
| `limit_per_category` | `int \| null` | `null` | `ge=1` | Limit items per dataset category. |
| `top_k` | `int` | `3` | `ge=1` | Number of context documents to retrieve from RAG server. |
| `max_context_chars` | `int` | `12000` | `ge=100` | Max context characters passed to LLM evaluator. |
| `llm_base_url` | `str \| null` | `null` | - | Custom LLM OpenAI-compatible base URL. |
| `llm_api_key` | `str \| null` | `null` | - | Custom LLM API key. |
| `llm_model` | `str \| null` | `null` | - | Custom LLM model name (e.g., `"gpt-4o"`). |
| `agentic` | `bool` | `false` | - | Route queries through CAIPE supervisor agent (A2A). |
| `supervisor_url` | `str \| null` | `null` | - | CAIPE supervisor endpoint URL. |
| `fail_on_error` | `bool` | `false` | - | Fail job loudly if a single query fails. |
| `oracle_retrieval` | `bool` | `false` | - | Enable question + reference ground truth retrieval. |
| `gate` | `bool` | `false` | - | Apply pass/fail quality gate check to results. |
| `save_to_db` | `bool` | `false` | - | Persist evaluation results to PostgreSQL DB automatically on completion. |
| `force_rerun` | `bool` | `false` | - | Bypass 24-hour deduplication cache and force rerun. |
| `question_ids` | `list[str] \| null` | `null` | - | List of specific question IDs to evaluate. |
| `question_indices` | `list[int] \| null` | `null` | - | List of specific dataset question indices to evaluate. |

#### Request JSON Example

```json
{
  "dataset_name": "enterprise",
  "answer_mode": "generate",
  "top_k": 3,
  "max_items": 5,
  "max_context_chars": 6000,
  "save_to_db": true,
  "force_rerun": false
}
```

---

### `JobResponse` (Response Schema)

Returned by `POST /eval/jobs`, `POST /eval/jobs/upload`, `GET /jobs`, and `GET /jobs/{job_id}`.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `job_id` | `str` | UUID v4 string identifying the evaluation job. |
| `status` | `JobStatusEnum` | Current status (`"pending"`, `"running"`, `"completed"`, `"failed"`). |
| `created_at` | `float` | Unix timestamp of job creation. |
| `completed_at` | `float \| null` | Unix timestamp of job completion (or `null` if pending/running). |
| `cached` | `bool` | `true` if job result was retrieved from 24-hour deduplication cache. |
| `eval_hash` | `str` | 16-character SHA-256 evaluation fingerprint. |
| `error` | `str \| null` | Error traceback or message if job status is `"failed"`. |
| `user_info` | `dict \| null` | Authenticated user context (`subject`, `email`, `role`). |

#### Response JSON Example

```json
{
  "job_id": "c7a8b9e0-1234-4567-89ab-cdef01234567",
  "status": "pending",
  "created_at": 1784683200.0,
  "completed_at": null,
  "cached": false,
  "eval_hash": "a1b2c3d4e5f67890",
  "error": null,
  "user_info": {
    "subject": "service-account-key",
    "email": "service-account@deepeval",
    "role": "admin"
  }
}
```

---

### `EvaluationResultsResponse` (Full Results Schema)

Returned by `GET /jobs/{job_id}/results?format=json`.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `job_id` | `str` | Job UUID. |
| `status` | `JobStatusEnum` | Job status (`"completed"`). |
| `created_at` | `float` | Job creation timestamp. |
| `completed_at` | `float \| null` | Job completion timestamp. |
| `cached` | `bool` | Cache hit boolean flag. |
| `eval_hash` | `str` | Evaluation fingerprint hash. |
| `evaluation_time` | `float` | Total duration of LLM evaluation run in seconds. |
| `config_args` | `dict` | Sanitized evaluation parameters used for run. |
| `summary` | `dict` | Aggregated evaluation metrics summary (overall score, pass rates, latency). |
| `results` | `list[dict]` | Per-question detail objects (query, actual output, expected output, metrics). |
| `saved_to_db` | `bool` | Whether results were saved to PostgreSQL DB. |
| `user_info` | `dict \| null` | Identity details of submitter. |

#### Response JSON Example

```json
{
  "job_id": "c7a8b9e0-1234-4567-89ab-cdef01234567",
  "status": "completed",
  "created_at": 1784683200.0,
  "completed_at": 1784683245.5,
  "cached": false,
  "eval_hash": "a1b2c3d4e5f67890",
  "evaluation_time": 45.5,
  "config_args": {
    "dataset_name": "enterprise",
    "answer_mode": "generate",
    "top_k": 3,
    "max_items": 5
  },
  "summary": {
    "total_questions": 5,
    "passed_questions": 5,
    "failed_questions": 0,
    "pass_rate": 1.0,
    "average_faithfulness": 0.95,
    "average_answer_relevance": 0.92,
    "average_context_precision": 0.88
  },
  "results": [
    {
      "question_id": "q1",
      "category": "confluence",
      "question": "What is the security compliance process?",
      "actual_output": "The security compliance process involves quarterly audits...",
      "expected_output": "Quarterly audits and SOC2 verification.",
      "success": true,
      "metrics": {
        "faithfulness": 0.96,
        "answer_relevance": 0.94
      }
    }
  ],
  "saved_to_db": true,
  "user_info": {
    "subject": "service-account-key",
    "email": "service-account@deepeval",
    "role": "admin"
  }
}
```

---

## API Endpoints Reference

| Method | Path | Summary | Auth Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health Check | No | Check server health status (`{"status": "healthy"}`). |
| `POST` | `/eval/jobs` | Submit Evaluation Job | Yes | Submit async evaluation job with JSON request body. |
| `POST` | `/eval/jobs/upload` | Submit Job with Dataset File | Yes | Upload dataset file (`.json`, `.csv`, `.jsonl`) via `multipart/form-data`. |
| `GET` | `/jobs` | List Evaluation Jobs | Yes | List all submitted jobs and status. |
| `GET` | `/jobs/{job_id}` | Poll Job Status | Yes | Poll execution state of a specific job ID. |
| `GET` | `/jobs/{job_id}/results` | Get Job Results | Yes | Download evaluation results in JSON or CSV format (`?format=json\|csv`). |
| `POST` | `/jobs/{job_id}/save-db` | Save Results to Database | Yes | Manually persist completed job results to PostgreSQL DB on demand. |
| `GET` | `/results/db` | Query Database Runs | Yes | Retrieve recent evaluation runs stored in PostgreSQL DB. |

---

### Endpoint Details & Curl Examples

#### 1. Submit Evaluation Job (JSON)

- **HTTP Method**: `POST`
- **Path**: `/eval/jobs`
- **Headers**: `Content-Type: application/json`, `Authorization: Bearer <TOKEN>` (or `X-API-Key`)

```bash
curl -X POST "http://localhost:8000/eval/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: secret-key-123" \
  -d '{
    "dataset_name": "enterprise",
    "answer_mode": "generate",
    "top_k": 3,
    "max_items": 5,
    "save_to_db": false,
    "force_rerun": false
  }'
```

---

#### 2. Submit Job with Dataset File Upload

- **HTTP Method**: `POST`
- **Path**: `/eval/jobs/upload`
- **Content-Type**: `multipart/form-data`
- **Query Parameters**:
  - `dataset_name` (string, default: `"custom_upload"`)
  - `top_k` (int, default: `3`)
  - `max_items` (int, optional)
  - `save_to_db` (bool, default: `false`)
  - `force_rerun` (bool, default: `false`)

```bash
curl -X POST "http://localhost:8000/eval/jobs/upload?dataset_name=my_benchmark&top_k=3&save_to_db=true" \
  -H "X-API-Key: secret-key-123" \
  -F "file=@my_questions.json"
```

---

#### 3. List Evaluation Jobs

- **HTTP Method**: `GET`
- **Path**: `/jobs`

```bash
curl -H "X-API-Key: secret-key-123" "http://localhost:8000/jobs"
```

---

#### 4. Poll Job Status

- **HTTP Method**: `GET`
- **Path**: `/jobs/{job_id}`

```bash
curl -H "X-API-Key: secret-key-123" \
  "http://localhost:8000/jobs/c7a8b9e0-1234-4567-89ab-cdef01234567"
```

---

#### 5. Retrieve Job Results (JSON / CSV)

- **HTTP Method**: `GET`
- **Path**: `/jobs/{job_id}/results`
- **Query Parameter**: `format=json` (default) or `format=csv`

##### Fetch JSON Results:

```bash
curl -H "X-API-Key: secret-key-123" \
  "http://localhost:8000/jobs/c7a8b9e0-1234-4567-89ab-cdef01234567/results?format=json"
```

##### Download CSV Report:

```bash
curl -H "X-API-Key: secret-key-123" \
  -o evaluation_report.csv \
  "http://localhost:8000/jobs/c7a8b9e0-1234-4567-89ab-cdef01234567/results?format=csv"
```

**CSV Response Headers**:
`Content-Disposition: attachment; filename=job_c7a8b9e0-1234-4567-89ab-cdef01234567_results.csv`
`Content-Type: text/csv`

---

#### 6. Save Results to PostgreSQL DB

- **HTTP Method**: `POST`
- **Path**: `/jobs/{job_id}/save-db`

```bash
curl -X POST -H "X-API-Key: secret-key-123" \
  "http://localhost:8000/jobs/c7a8b9e0-1234-4567-89ab-cdef01234567/save-db"
```

##### Response JSON:

```json
{
  "job_id": "c7a8b9e0-1234-4567-89ab-cdef01234567",
  "status": "success",
  "message": "Evaluation results successfully saved to PostgreSQL database"
}
```

---

#### 7. Query PostgreSQL Database Runs

- **HTTP Method**: `GET`
- **Path**: `/results/db`
- **Query Parameter**: `limit=10` (default: `10`, range: `1..100`)

```bash
curl -H "X-API-Key: secret-key-123" \
  "http://localhost:8000/results/db?limit=5"
```

##### Response JSON:

```json
{
  "count": 1,
  "runs": [
    {
      "experiment_id": "exp-2026-07-23-001",
      "timestamp": "2026-07-23T18:00:00Z",
      "datasource": "enterprise",
      "evaluation_time": 45.5,
      "total_questions": 5,
      "passed_questions": 5,
      "pass_rate": 1.0,
      "avg_faithfulness": 0.95
    }
  ]
}
```

---

## Error Handling & HTTP Status Codes

The API returns standard HTTP status codes and JSON error objects formatted as `{"detail": "<error message>"}`.

| Code | Reason | Common Causes |
| :--- | :--- | :--- |
| `200 OK` | Request succeeded | Results returned or DB query succeeded. |
| `202 Accepted` | Job accepted | Asynchronous job submitted and queued in background. |
| `400 Bad Request` | Invalid input | Validation error in parameters (`max_items < 1`), unsupported format, or empty file upload. |
| `401 Unauthorized` | Authentication failed | Missing or invalid Bearer token / `X-API-Key`. |
| `404 Not Found` | Resource not found | Specified `job_id` does not exist. |
| `500 Internal Server Error` | Backend execution error | Job evaluation failed or PostgreSQL DB query error. |

---

## Python Integration SDK Example

You can interact with the REST API using Python's `httpx` or `requests` library:

```python
import time
import httpx

API_BASE_URL = "http://localhost:8000"
API_KEY = "secret-key-123"
headers = {"X-API-Key": API_KEY}

with httpx.Client(base_url=API_BASE_URL, headers=headers) as client:
    # 1. Submit evaluation job
    response = client.post("/eval/jobs", json={
        "dataset_name": "enterprise",
        "top_k": 3,
        "max_items": 5,
        "save_to_db": True
    })
    job_info = response.json()
    job_id = job_info["job_id"]
    print(f"Submitted job {job_id}, status: {job_info['status']}")

    # 2. Poll until completed or failed
    while True:
        status_resp = client.get(f"/jobs/{job_id}").json()
        current_status = status_resp["status"]
        print(f"Job {job_id} status: {current_status}")

        if current_status in ("completed", "failed"):
            break
        time.sleep(2)

    # 3. Fetch full JSON results
    if current_status == "completed":
        results_resp = client.get(f"/jobs/{job_id}/results?format=json").json()
        print("Summary:", results_resp["summary"])
```
