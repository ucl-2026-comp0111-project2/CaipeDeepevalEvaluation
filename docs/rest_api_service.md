# REST API Evaluation & Results Management Service

The REST API Evaluation Service provides an asynchronous HTTP interface and OpenAPI/Swagger documentation (`/docs`) for running evaluation pipelines, submitting datasets, managing background evaluation jobs, querying results, and storing evaluation runs in PostgreSQL.

---

## Architectural Overview & Design Patterns

The service is built using FastAPI and ASGI (`uvicorn`) and follows several enterprise software design patterns:

### 1. Repository / Sink Pattern (Data Abstraction)
- **Concept**: Separates evaluation business logic from persistent storage mechanisms.
- **Implementation**: Implements `ResultSink` Protocol interface with `FileResultSink` (for local 24-hour JSON/CSV artifact generation) and `DatabaseResultSink` (for direct PostgreSQL persistence).
- **Benefit**: Endpoints querying evaluation outputs operate on standard result objects regardless of whether storage is local disk or PostgreSQL database.

### 2. Job Queue & State Machine Pattern (Asynchronous Operations)
- **Concept**: Long-running LLM evaluations run asynchronously in background tasks without blocking client HTTP requests or incurring server request timeouts.
- **State Machine Transitions**:
  $$\text{PENDING} \longrightarrow \text{RUNNING} \longrightarrow \text{COMPLETED} \quad / \quad \text{FAILED}$$
- **Implementation**: `JobManager` handles job state transitions and background execution (`fastapi.BackgroundTasks`), returning an immediate `202 Accepted` response with a unique `job_id`.

### 3. Strategy / Adapter Pattern (Dataset Ingestion)
- **Concept**: Unifies diverse dataset sources into standard benchmark formats consumed by `EvalEngine`.
- **Supported Strategies**:
  - Built-in benchmarks (`enterprise`, `hotpotqa`) loaded via `data_loader.py`.
  - Direct file upload (`POST /eval/jobs/upload` multipart form) storing temporary dataset JSON/CSV files.
  - Custom dataset files (`questions_file` JSON path).

### 4. Cache-Aside / Memoization Pattern (Evaluation Deduplication)
- **Concept**: Prevents redundant LLM evaluation runs with identical configurations and inputs.
- **Fingerprinting Algorithm**:
  $$\text{eval\_hash} = \text{SHA256}(\text{normalized\_config\_json} + \text{dataset\_bytes})[:16]$$
- **Workflow**:
  - When a job is submitted, `JobManager` checks `LocalCacheManager` (24-hour TTL) for a matching `eval_hash`.
  - If a cached result exists and `force_rerun=False`, the API instantly returns the completed job (`cached=True`).
  - Setting `force_rerun=True` invalidates cache and triggers fresh evaluation.

### 5. DTO (Data Transfer Object) & Schema Validation Pattern
- **Concept**: Strongly typed Pydantic request/response models (`EvaluationRequest`, `JobResponse`, `EvaluationResultsResponse`).
- **Benefit**: Guarantees input validation constraints (`max_items >= 1`, `top_k >= 1`) and auto-generates interactive Swagger UI docs at `/docs` and ReDoc at `/redoc`.

---

## Configuration & Environment Resolution

The service automatically loads configuration settings on startup:
1. Loads `.env` file via `load_dotenv_loose(DEFAULT_ENV_FILE)`.
2. Overrides with system environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `OPENAI_API_KEY`, `CAIPE_SUPERVISOR_URL`).
3. Merges per-request parameter overrides from `EvaluationRequest`.

---

## API Endpoints Reference

| HTTP Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root endpoint with service status and Swagger links | `200 OK` |
| `GET` | `/health` | Health check endpoint | `200 OK` |
| `GET` | `/docs` | Interactive Swagger UI documentation | `200 OK` |
| `GET` | `/redoc` | ReDoc API documentation | `200 OK` |
| `POST` | `/eval/jobs` | Submit asynchronous evaluation job (JSON payload) | `202 Accepted` |
| `POST` | `/eval/jobs/upload` | Submit job with dataset file upload (`multipart/form-data`) | `202 Accepted` |
| `GET` | `/jobs` | List all submitted evaluation jobs | `200 OK` |
| `GET` | `/jobs/{job_id}` | Poll evaluation job status (`pending`, `running`, `completed`, `failed`) | `200 OK` / `404` |
| `GET` | `/jobs/{job_id}/results` | Retrieve evaluation results summary and metrics | `200 OK` / `400` / `500` |
| `POST` | `/jobs/{job_id}/save-db` | Persist completed job results to PostgreSQL DB on demand | `200 OK` / `400` / `500` |
| `GET` | `/results/db` | Query evaluation runs stored in PostgreSQL database | `200 OK` / `500` |

---

## Usage Guide & Examples

### Starting the REST API Server

```bash
uv run python -m deepeval_eval.api
# Or via uvicorn directly:
uv run uvicorn deepeval_eval.api:app --host 0.0.0.0 --port 8000
```

Access Swagger UI in your browser at:
`http://localhost:8000/docs`

### Submitting an Evaluation Job (JSON)

```bash
curl -X POST "http://localhost:8000/eval/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "enterprise",
    "answer_mode": "reference",
    "top_k": 3,
    "max_items": 5,
    "save_to_db": false,
    "force_rerun": false
  }'
```

Response:
```json
{
  "job_id": "c7a8b9e0-1234-4567-89ab-cdef01234567",
  "status": "pending",
  "created_at": 1784683200.0,
  "completed_at": null,
  "cached": false,
  "eval_hash": "a1b2c3d4e5f67890",
  "error": null
}
```

### Polling Job Status

```bash
curl "http://localhost:8000/jobs/c7a8b9e0-1234-4567-89ab-cdef01234567"
```

### Retrieving Evaluation Results

```bash
curl "http://localhost:8000/jobs/c7a8b9e0-1234-4567-89ab-cdef01234567/results"
```

### Submitting Job via File Upload

```bash
curl -X POST "http://localhost:8000/eval/jobs/upload?dataset_name=custom&top_k=3" \
  -F "file=@my_questions.json"
```

### Persisting Results to PostgreSQL Database

```bash
curl -X POST "http://localhost:8000/jobs/c7a8b9e0-1234-4567-89ab-cdef01234567/save-db"
```
