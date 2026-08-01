from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deepeval_eval.api.auth import UserContext, get_current_user
from deepeval_eval.db.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/question-sets", tags=["Question Sets"])

# ---------------------------------------------------------------------------
# Pydantic DTOs
# ---------------------------------------------------------------------------


class QuestionSetCreate(BaseModel):
    name: str = Field(..., description="Name of the question set")
    description: str | None = Field(
        default=None, description="Description of the question set"
    )
    source_format: str | None = Field(
        default=None, description="Source format (e.g. jsonl, csv)"
    )


class QuestionSetUpdate(BaseModel):
    name: str | None = Field(default=None, description="Updated name of question set")
    description: str | None = Field(default=None, description="Updated description")
    source_format: str | None = Field(default=None, description="Updated source format")


class QuestionSetResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    source_format: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    question_count: int = 0
    categories: dict[str, int] | None = None


class QuestionSetListResponse(BaseModel):
    items: list[QuestionSetResponse]
    total: int
    page: int
    limit: int
    total_pages: int


class QuestionCreate(BaseModel):
    question_id: str | None = Field(
        default=None, description="Custom question string identifier"
    )
    input: str = Field(..., description="User question / prompt input text")
    expected_output: str | None = Field(
        default=None, description="Expected ground truth answer reference"
    )
    category: str | None = Field(default=None, description="Category / domain tag")
    level: str | None = Field(default=None, description="Difficulty level")
    expected_doc_ids: list[str] = Field(
        default_factory=list, description="Expected source document IDs"
    )
    context: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Context payload"
    )
    extra: dict[str, Any] | None = Field(default=None, description="Extra metadata")


class QuestionUpdate(BaseModel):
    question_id: str | None = None
    input: str | None = None
    expected_output: str | None = None
    category: str | None = None
    level: str | None = None
    expected_doc_ids: list[str] | None = None
    context: dict[str, Any] | list[Any] | None = None
    extra: dict[str, Any] | None = None


class QuestionResponse(BaseModel):
    id: int
    question_set_id: int
    question_id: str | None = None
    input: str
    expected_output: str | None = None
    category: str | None = None
    level: str | None = None
    expected_doc_ids: list[str] = Field(default_factory=list)
    context: Any = None
    extra: Any = None
    created_at: str | None = None
    updated_at: str | None = None


class QuestionListResponse(BaseModel):
    items: list[QuestionResponse]
    total: int
    page: int
    limit: int
    total_pages: int


class BatchDeleteRequest(BaseModel):
    question_ids: list[str | int] = Field(
        ..., description="List of internal DB IDs or external question_ids to delete"
    )


class BatchDeleteResponse(BaseModel):
    deleted_count: int


# ---------------------------------------------------------------------------
# Helper Ingestion Functions
# ---------------------------------------------------------------------------


def parse_questions_file_content(
    content: bytes, filename: str | None
) -> list[dict[str, Any]]:
    """Parse JSON, JSONL, or CSV bytes into a list of normalized question dicts."""
    filename_lower = (filename or "").lower()
    items: list[dict[str, Any]] = []

    if filename_lower.endswith(".jsonl") or not filename_lower.endswith(
        (".json", ".csv")
    ):
        # Try JSONL parsing line-by-line
        text = content.decode("utf-8", errors="replace")
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                if isinstance(item, dict):
                    items.append(item)
            except json.JSONDecodeError:
                pass

        if items:
            return items

    if filename_lower.endswith(".csv"):
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            doc_ids_raw = row.get("expected_doc_ids") or row.get("doc_ids") or ""
            doc_ids: list[str] = []
            if doc_ids_raw:
                if doc_ids_raw.startswith("[") and doc_ids_raw.endswith("]"):
                    try:
                        doc_ids = json.loads(doc_ids_raw)
                    except Exception:
                        doc_ids = [
                            d.strip()
                            for d in doc_ids_raw.strip("[]").split(",")
                            if d.strip()
                        ]
                else:
                    doc_ids = [d.strip() for d in doc_ids_raw.split(",") if d.strip()]

            items.append(
                {
                    "question_id": row.get("question_id"),
                    "input": row.get("input") or row.get("user_input") or "",
                    "expected_output": row.get("expected_output")
                    or row.get("reference")
                    or "",
                    "category": row.get("category"),
                    "level": row.get("level"),
                    "expected_doc_ids": doc_ids,
                }
            )
        return items

    if filename_lower.endswith(".json"):
        text = content.decode("utf-8", errors="replace")
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        elif isinstance(data, dict):
            if "questions" in data and isinstance(data["questions"], list):
                return [d for d in data["questions"] if isinstance(d, dict)]
            return [data]

    return items


def get_db_manager() -> DatabaseManager:
    return DatabaseManager()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@router.post(
    "", response_model=QuestionSetResponse, status_code=status.HTTP_201_CREATED
)
async def create_question_set(
    name: str = Form(...),
    description: str | None = Form(default=None),
    source_format: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """Create a new question set (blank or initialized via JSONL/CSV file upload)."""
    detected_format = source_format
    if file and file.filename:
        if file.filename.endswith(".jsonl"):
            detected_format = detected_format or "jsonl"
        elif file.filename.endswith(".csv"):
            detected_format = detected_format or "csv"
        elif file.filename.endswith(".json"):
            detected_format = detected_format or "json"

    qset = db.questions.create_question_set(
        name=name.strip(),
        description=description.strip() if description else None,
        source_format=detected_format,
    )

    if file:
        content = await file.read()
        questions = parse_questions_file_content(content, file.filename)
        if questions:
            db.questions.add_questions(qset["id"], questions)
            qset = db.questions.get_question_set(qset["id"]) or qset

    return qset


@router.get("", response_model=QuestionSetListResponse)
def list_question_sets(
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page"),
    query: str | None = Query(
        default=None, description="Search term for name or description"
    ),
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """List question sets with pagination and search filtering."""
    return db.questions.list_question_sets(page=page, limit=limit, query=query)


@router.get("/{set_id}", response_model=QuestionSetResponse)
def get_question_set(
    set_id: int,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """Get question set details and summary stats by ID."""
    qset = db.questions.get_question_set(set_id)
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )
    return qset


@router.put("/{set_id}", response_model=QuestionSetResponse)
def update_question_set(
    set_id: int,
    payload: QuestionSetUpdate,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """Update question set metadata."""
    qset = db.questions.update_question_set(
        set_id=set_id,
        name=payload.name,
        description=payload.description,
        source_format=payload.source_format,
    )
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )
    return qset


@router.delete("/{set_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_question_set(
    set_id: int,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> None:
    """Delete a question set and all its associated questions."""
    success = db.questions.delete_question_set(set_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )


@router.post(
    "/{set_id}/questions",
    response_model=list[QuestionResponse],
    status_code=status.HTTP_201_CREATED,
)
def add_questions_to_set(
    set_id: int,
    payload: list[QuestionCreate] | QuestionCreate,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> list[dict[str, Any]]:
    """Add questions to a set via JSON request body (single question object or list of objects)."""
    qset = db.questions.get_question_set(set_id)
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )

    if isinstance(payload, list):
        questions_to_add = [item.model_dump(exclude_unset=True) for item in payload]
    else:
        questions_to_add = [payload.model_dump(exclude_unset=True)]

    if not questions_to_add:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No questions provided in request body.",
        )

    try:
        return db.questions.add_questions(set_id, questions_to_add)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )


@router.post(
    "/{set_id}/questions/upload",
    response_model=list[QuestionResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_questions_file_to_set(
    set_id: int,
    file: UploadFile = File(...),
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> list[dict[str, Any]]:
    """Upload new questions to a question set via file upload (.jsonl, .csv, .json)."""
    qset = db.questions.get_question_set(set_id)
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )

    content = await file.read()
    questions_to_add = parse_questions_file_content(content, file.filename)

    if not questions_to_add:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse valid questions from uploaded file '{file.filename}'.",
        )

    try:
        return db.questions.add_questions(set_id, questions_to_add)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )


@router.get("/{set_id}/questions", response_model=QuestionListResponse)
def list_questions_in_set(
    set_id: int,
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page"),
    category: str | None = Query(default=None, description="Filter by category"),
    level: str | None = Query(default=None, description="Filter by difficulty level"),
    query: str | None = Query(
        default=None, description="Search query in input or question_id"
    ),
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """List questions in a question set with pagination and filters."""
    qset = db.questions.get_question_set(set_id)
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )

    return db.questions.list_questions(
        set_id=set_id,
        page=page,
        limit=limit,
        category=category,
        level=level,
        query=query,
    )


@router.get(
    "/{set_id}/questions/{question_identifier}", response_model=QuestionResponse
)
def get_question(
    set_id: int,
    question_identifier: str,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """Get a single question by DB id or question_id string."""
    q = db.questions.get_question(set_id, question_identifier)
    if not q:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{question_identifier}' not found in question set {set_id}.",
        )
    return q


@router.put(
    "/{set_id}/questions/{question_identifier}", response_model=QuestionResponse
)
def update_question(
    set_id: int,
    question_identifier: str,
    payload: QuestionUpdate,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """Edit a question in a question set."""
    updated = db.questions.update_question(
        set_id, question_identifier, payload.model_dump(exclude_unset=True)
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{question_identifier}' not found in question set {set_id}.",
        )
    return updated


@router.delete(
    "/{set_id}/questions/{question_identifier}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_question(
    set_id: int,
    question_identifier: str,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> None:
    """Delete a question from a question set."""
    success = db.questions.delete_question(set_id, question_identifier)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{question_identifier}' not found in question set {set_id}.",
        )


@router.post("/{set_id}/questions/batch-delete", response_model=BatchDeleteResponse)
def batch_delete_questions(
    set_id: int,
    payload: BatchDeleteRequest,
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, int]:
    """Batch delete multiple questions in a set."""
    qset = db.questions.get_question_set(set_id)
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )

    deleted_count = db.questions.batch_delete_questions(set_id, payload.question_ids)
    return {"deleted_count": deleted_count}


@router.get("/{set_id}/export")
def export_question_set(
    set_id: int,
    format: str = Query(
        default="jsonl",
        pattern="^(jsonl|csv)$",
        description="Export format: jsonl or csv",
    ),
    user: UserContext = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager),
) -> StreamingResponse:
    """Export question set as JSONL or CSV file download."""
    qset = db.questions.get_question_set(set_id)
    if not qset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question set {set_id} not found.",
        )

    # Fetch all questions (up to max limit)
    result = db.questions.list_questions(set_id=set_id, page=1, limit=10000)
    questions = result["items"]

    name_clean = (
        "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in qset["name"]
        ).strip("_")
        or "question_set"
    )

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "question_id",
                "input",
                "expected_output",
                "category",
                "level",
                "expected_doc_ids",
            ],
        )
        writer.writeheader()
        for q in questions:
            writer.writerow(
                {
                    "question_id": q.get("question_id"),
                    "input": q.get("input"),
                    "expected_output": q.get("expected_output"),
                    "category": q.get("category"),
                    "level": q.get("level"),
                    "expected_doc_ids": json.dumps(q.get("expected_doc_ids") or []),
                }
            )
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{name_clean}.csv"'},
        )
    else:  # jsonl
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

        return StreamingResponse(
            iter(["".join(lines)]),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{name_clean}.jsonl"'
            },
        )
