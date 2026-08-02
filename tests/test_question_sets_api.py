from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from deepeval_eval.api.app import app
from deepeval_eval.api.question_sets import parse_questions_file_content
from deepeval_eval.db.db_manager import DatabaseManager
from deepeval_eval.db.question_db_manager import QuestionDBManager

client = TestClient(app)


@pytest.fixture(autouse=True)
def enable_unauthenticated_access_for_tests():
    os.environ["ALLOW_UNAUTHENTICATED_ACCESS"] = "true"
    yield
    os.environ.pop("ALLOW_UNAUTHENTICATED_ACCESS", None)


# ---------------------------------------------------------------------------
# Unit Tests for Data Ingestion Helper
# ---------------------------------------------------------------------------


def test_parse_questions_file_content_jsonl():
    """Verify parsing valid JSONL dataset content with user_input and reference keys."""
    jsonl_content = (
        b'{"question_id": "q1", "user_input": "What is A?", "reference": "Ans A", "category": "cat1"}\n'
        b'{"question_id": "q2", "user_input": "What is B?", "reference": "Ans B", "category": "cat2"}\n'
    )
    items = parse_questions_file_content(jsonl_content, "data.jsonl")
    assert len(items) == 2
    assert items[0]["question_id"] == "q1"
    assert items[0]["user_input"] == "What is A?"
    assert items[1]["user_input"] == "What is B?"


def test_parse_questions_file_content_csv():
    """Verify parsing CSV dataset content into normalized dict items."""
    csv_content = (
        b"question_id,input,expected_output,category,level,expected_doc_ids\n"
        b'q1,Prompt 1,Ans 1,cat1,easy,"doc1,doc2"\n'
    )
    items = parse_questions_file_content(csv_content, "data.csv")
    assert len(items) == 1
    assert items[0]["question_id"] == "q1"
    assert items[0]["input"] == "Prompt 1"
    assert items[0]["expected_doc_ids"] == ["doc1", "doc2"]


def test_parse_questions_file_content_json_array():
    """Verify parsing JSON array dataset content."""
    json_content = json.dumps(
        [
            {"question_id": "q1", "input": "In 1", "expected_output": "Out 1"},
            {"question_id": "q2", "input": "In 2", "expected_output": "Out 2"},
        ]
    ).encode("utf-8")
    items = parse_questions_file_content(json_content, "data.json")
    assert len(items) == 2
    assert items[0]["question_id"] == "q1"


def test_parse_questions_file_content_invalid():
    """Verify empty array returned for unparseable file content."""
    items = parse_questions_file_content(b"not valid json or csv", "data.txt")
    assert items == []


# ---------------------------------------------------------------------------
# Unit Tests for API Endpoints using Mock Database Manager
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_manager():
    """Create a mock DatabaseManager with simulated QuestionDBManager behavior."""
    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.is_postgres.return_value = True

    # In-memory storage for test data
    question_sets_store: dict[int, dict] = {}
    questions_store: dict[int, list[dict]] = {}
    set_id_counter = [1]
    question_id_counter = [1]

    mock_q_db = MagicMock(spec=QuestionDBManager)

    def create_set(name, description=None, source_format=None):
        sid = set_id_counter[0]
        set_id_counter[0] += 1
        record = {
            "id": sid,
            "name": name,
            "description": description,
            "source_format": source_format,
            "created_at": "2026-08-02T00:00:00+00:00",
            "updated_at": "2026-08-02T00:00:00+00:00",
            "question_count": 0,
            "categories": {},
        }
        question_sets_store[sid] = record
        questions_store[sid] = []
        return record

    def list_sets(page=1, limit=50, query=None):
        items = list(question_sets_store.values())
        if query:
            items = [i for i in items if query.lower() in i["name"].lower()]
        total = len(items)
        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": 1 if total > 0 else 0,
        }

    def get_set(set_id):
        if set_id not in question_sets_store:
            return None
        rec = dict(question_sets_store[set_id])
        q_list = questions_store.get(set_id, [])
        rec["question_count"] = len(q_list)
        cats = {}
        for q in q_list:
            c = q.get("category") or "uncategorized"
            cats[c] = cats.get(c, 0) + 1
        rec["categories"] = cats
        return rec

    def update_set(set_id, name=None, description=None, source_format=None):
        if set_id not in question_sets_store:
            return None
        rec = question_sets_store[set_id]
        if name is not None:
            rec["name"] = name
        if description is not None:
            rec["description"] = description
        if source_format is not None:
            rec["source_format"] = source_format
        return get_set(set_id)

    def delete_set(set_id):
        if set_id in question_sets_store:
            del question_sets_store[set_id]
            questions_store.pop(set_id, None)
            return True
        return False

    def add_q(set_id, questions_data):
        if set_id not in question_sets_store:
            raise ValueError(f"Question set {set_id} does not exist")
        inserted = []
        for idx, item in enumerate(questions_data, start=1):
            qid_num = question_id_counter[0]
            question_id_counter[0] += 1
            inp = item.get("input") or item.get("user_input")
            if not inp:
                raise ValueError("Missing input")
            qid_str = item.get("question_id") or f"q-{qid_num}"
            q_rec = {
                "id": qid_num,
                "question_set_id": set_id,
                "question_id": qid_str,
                "input": inp,
                "expected_output": item.get("expected_output") or item.get("reference"),
                "category": item.get("category"),
                "level": item.get("level"),
                "expected_doc_ids": item.get("expected_doc_ids") or [],
                "context": item.get("context"),
                "extra": item.get("extra"),
                "created_at": "2026-08-02T00:00:00+00:00",
                "updated_at": "2026-08-02T00:00:00+00:00",
            }
            questions_store[set_id].append(q_rec)
            inserted.append(q_rec)
        return inserted

    def list_q(set_id, page=1, limit=50, category=None, level=None, query=None):
        if set_id not in question_sets_store:
            return {
                "items": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "total_pages": 0,
            }
        q_list = list(questions_store.get(set_id, []))
        if category:
            q_list = [q for q in q_list if q.get("category") == category]
        if level:
            q_list = [q for q in q_list if q.get("level") == level]
        if query:
            q_list = [q for q in q_list if query.lower() in q["input"].lower()]
        total = len(q_list)
        return {
            "items": q_list,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": 1 if total > 0 else 0,
        }

    def get_q(set_id, identifier):
        q_list = questions_store.get(set_id, [])
        ident_str = str(identifier)
        for q in q_list:
            if str(q["id"]) == ident_str or q.get("question_id") == ident_str:
                return q
        return None

    def update_q(set_id, identifier, data):
        q = get_q(set_id, identifier)
        if not q:
            return None
        for k, v in data.items():
            if k in q and v is not None:
                q[k] = v
        return q

    def delete_q(set_id, identifier):
        q_list = questions_store.get(set_id, [])
        ident_str = str(identifier)
        for i, q in enumerate(q_list):
            if str(q["id"]) == ident_str or q.get("question_id") == ident_str:
                del q_list[i]
                return True
        return False

    def batch_delete_q(set_id, identifiers):
        cnt = 0
        for ident in identifiers:
            if delete_q(set_id, ident):
                cnt += 1
        return cnt

    def stream_q(set_id, batch_size=1000):
        yield from questions_store.get(set_id, [])

    mock_q_db.create_question_set.side_effect = create_set
    mock_q_db.list_question_sets.side_effect = list_sets
    mock_q_db.get_question_set.side_effect = get_set
    mock_q_db.update_question_set.side_effect = update_set
    mock_q_db.delete_question_set.side_effect = delete_set
    mock_q_db.add_questions.side_effect = add_q
    mock_q_db.list_questions.side_effect = list_q
    mock_q_db.stream_questions.side_effect = stream_q
    mock_q_db.get_question_by_id.side_effect = get_q
    mock_q_db.get_question_by_question_id.side_effect = get_q
    mock_q_db.update_question_by_id.side_effect = update_q
    mock_q_db.update_question_by_question_id.side_effect = update_q
    mock_q_db.delete_question_by_id.side_effect = delete_q
    mock_q_db.delete_question_by_question_id.side_effect = delete_q
    mock_q_db.batch_delete_questions.side_effect = (
        lambda set_id, ids=None, question_ids=None: batch_delete_q(
            set_id, (ids or []) + (question_ids or [])
        )
    )
    mock_q_db.batch_delete_questions_by_ids.side_effect = batch_delete_q
    mock_q_db.batch_delete_questions_by_question_ids.side_effect = batch_delete_q

    mock_db.questions = mock_q_db
    return mock_db


def test_question_sets_full_crud_lifecycle(mock_db_manager):
    """Test full CRUD lifecycle of Question Sets and Questions via REST API endpoints."""
    from deepeval_eval.api.question_sets import get_db_manager

    app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
    try:
        # 1. Create blank Question Set
        res = client.post(
            "/api/v1/question-sets",
            data={"name": "HotpotQA Benchmark", "description": "Multi-hop QA test set"},
        )
        assert res.status_code == 201
        data = res.json()
        set_id = data["id"]
        assert data["name"] == "HotpotQA Benchmark"
        assert data["question_count"] == 0

        # 2. List Question Sets
        res = client.get("/api/v1/question-sets")
        assert res.status_code == 200
        sets_data = res.json()
        assert sets_data["total"] == 1
        assert sets_data["items"][0]["id"] == set_id

        # 3. Add single question via REST JSON payload
        q_payload = {
            "question_id": "q-101",
            "input": "Where was the 2024 Summer Olympics held?",
            "expected_output": "Paris, France",
            "category": "sports",
            "level": "easy",
            "expected_doc_ids": ["doc_olympics_2024"],
        }
        res = client.post(f"/api/v1/question-sets/{set_id}/questions", json=q_payload)
        assert res.status_code == 201
        q_res = res.json()
        assert len(q_res) == 1
        assert q_res[0]["question_id"] == "q-101"

        # 4. Add batch of questions via JSON array REST payload
        batch_payload = [
            {
                "question_id": "q-102",
                "input": "What is 2+2?",
                "expected_output": "4",
                "category": "math",
            },
        ]
        res = client.post(
            f"/api/v1/question-sets/{set_id}/questions", json=batch_payload
        )
        assert res.status_code == 201
        assert len(res.json()) == 1

        # 4b. Upload questions via JSONL file upload endpoint
        file_bytes = b'{"question_id": "q-103", "user_input": "What is light speed?", "reference": "299,792,458 m/s", "category": "physics"}\n'
        res = client.post(
            f"/api/v1/question-sets/{set_id}/questions/upload",
            files={"file": ("batch.jsonl", file_bytes, "application/x-ndjson")},
        )
        assert res.status_code == 201
        assert res.json()[0]["question_id"] == "q-103"

        # 5. List questions in set
        res = client.get(f"/api/v1/question-sets/{set_id}/questions")
        assert res.status_code == 200
        q_list_data = res.json()
        assert q_list_data["total"] == 3

        # 6. Get single question by question_id
        res = client.get(f"/api/v1/question-sets/{set_id}/questions/question-id/q-101")
        assert res.status_code == 200
        assert res.json()["input"] == "Where was the 2024 Summer Olympics held?"

        # 6b. Get single question by id
        res = client.get(f"/api/v1/question-sets/{set_id}/questions/id/1")
        assert res.status_code == 200

        # 7. Edit a question via explicit sub-path
        res = client.put(
            f"/api/v1/question-sets/{set_id}/questions/question-id/q-101",
            json={"level": "hard", "expected_output": "Paris"},
        )
        assert res.status_code == 200
        assert res.json()["level"] == "hard"

        # 8. Delete a single question via explicit sub-path
        res = client.delete(
            f"/api/v1/question-sets/{set_id}/questions/question-id/q-102"
        )
        assert res.status_code == 204

        # 9. Batch delete questions (via question_ids and ids)
        res = client.post(
            f"/api/v1/question-sets/{set_id}/questions/batch-delete",
            json={"question_ids": ["q-103"]},
        )
        assert res.status_code == 200
        assert res.json()["deleted_count"] == 1

        res_db_ids = client.post(
            f"/api/v1/question-sets/{set_id}/questions/batch-delete",
            json={"ids": [103]},
        )
        assert res_db_ids.status_code == 200

        # 10. Verify export endpoints (JSONL & CSV)
        res_jsonl = client.get(f"/api/v1/question-sets/{set_id}/export?format=jsonl")
        assert res_jsonl.status_code == 200
        assert "Paris" in res_jsonl.text

        res_csv = client.get(f"/api/v1/question-sets/{set_id}/export?format=csv")
        assert res_csv.status_code == 200
        assert "Paris" in res_csv.text

        # 11. Submit evaluation job for Question Set ID
        res = client.post(
            f"/eval/jobs/question-sets/{set_id}",
            json={"eval_name": "qset_eval"},
        )
        assert res.status_code == 202

        # 12. Delete entire question set
        res = client.delete(f"/api/v1/question-sets/{set_id}")
        assert res.status_code == 204

        # 13. Confirm non-existent set returns 404
        res = client.get(f"/api/v1/question-sets/{set_id}")
        assert res.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_question_sets_negative_cases(mock_db_manager):
    """Test negative cases (404 / 400 / 422) for question set endpoints."""
    from deepeval_eval.api.question_sets import get_db_manager

    app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
    try:
        # 1. Non-existent question set -> 404
        res = client.get("/api/v1/question-sets/999")
        assert res.status_code == 404

        # 2. Update non-existent set -> 404
        res = client.put("/api/v1/question-sets/999", json={"name": "Ghost"})
        assert res.status_code == 404

        # 3. Delete non-existent set -> 404
        res = client.delete("/api/v1/question-sets/999")
        assert res.status_code == 404

        # 4. Add questions to non-existent set -> 404
        res = client.post(
            "/api/v1/question-sets/999/questions",
            json=[{"input": "Test"}],
        )
        assert res.status_code == 404

        # 5. Upload file to non-existent set -> 404
        res = client.post(
            "/api/v1/question-sets/999/questions/upload",
            files={
                "file": ("test.jsonl", b'{"user_input": "x"}\n', "application/x-ndjson")
            },
        )
        assert res.status_code == 404

        # 6. GET questions in non-existent set -> 404
        res = client.get("/api/v1/question-sets/999/questions")
        assert res.status_code == 404

        # 7. Batch delete in non-existent set -> 404
        res = client.post(
            "/api/v1/question-sets/999/questions/batch-delete",
            json={"question_ids": ["q-101"]},
        )
        assert res.status_code == 404

        # 8. Batch delete with empty payload -> 422 validation error
        res = client.post(
            "/api/v1/question-sets/1/questions/batch-delete",
            json={},
        )
        assert res.status_code == 422

        # 9. Export non-existent set -> 404
        res = client.get("/api/v1/question-sets/999/export")
        assert res.status_code == 404

        # 10. Submit eval job for non-existent set -> 404
        res = client.post("/eval/jobs/question-sets/999")
        assert res.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_question_sets_file_upload_size_limit(mock_db_manager):
    """Test that file uploads exceeding MAX_UPLOAD_SIZE_BYTES trigger HTTP 413."""
    from unittest.mock import patch

    from deepeval_eval.api.question_sets import get_db_manager

    app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
    try:
        # Create question set first
        create_res = client.post(
            "/api/v1/question-sets", data={"name": "Size Limit Test Set"}
        )
        assert create_res.status_code == 201
        set_id = create_res.json()["id"]

        # Patch MAX_UPLOAD_SIZE_BYTES to 20 bytes for fast test execution
        with patch("deepeval_eval.api.question_sets.MAX_UPLOAD_SIZE_BYTES", 20):
            file_bytes = b'{"question_id": "q-101", "user_input": "Very long payload text exceeding max bytes limit"}\n'
            res = client.post(
                f"/api/v1/question-sets/{set_id}/questions/upload",
                files={"file": ("batch.jsonl", file_bytes, "application/x-ndjson")},
            )
            assert res.status_code == 413
            assert "exceeds maximum allowed size" in res.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_question_sets_batch_delete_max_length_limit(mock_db_manager):
    """Test that batch delete payloads exceeding 1000 items trigger HTTP 422 validation error."""
    from deepeval_eval.api.question_sets import get_db_manager

    app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
    try:
        # 1. Single array with 1001 items -> 422
        oversized_ids = list(range(1001))
        res = client.post(
            "/api/v1/question-sets/1/questions/batch-delete",
            json={"ids": oversized_ids},
        )
        assert res.status_code == 422

        # 2. Combined arrays (600 ids + 600 question_ids = 1200 total) -> 422
        res_combined = client.post(
            "/api/v1/question-sets/1/questions/batch-delete",
            json={
                "ids": list(range(600)),
                "question_ids": [f"q-{i}" for i in range(600)],
            },
        )
        assert res_combined.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_question_sets_export_key_collision_and_rfc5987_headers(mock_db_manager):
    """Test export endpoint handles reserved key collisions in extra metadata and complies with RFC 5987 content-disposition header."""
    import json

    from deepeval_eval.api.question_sets import get_db_manager

    # Setup custom question stream returning colliding 'extra' keys
    def stream_q_collision(set_id):
        yield {
            "id": 1,
            "question_set_id": set_id,
            "question_id": "orig-qid",
            "input": "orig-input",
            "expected_output": "orig-ref",
            "category": "eval",
            "level": "easy",
            "expected_doc_ids": ["doc-1"],
            "extra": {
                "question_id": "malicious-qid",
                "user_input": "malicious-input",
                "custom_field": "safe-value",
            },
        }

    mock_db_manager.questions.stream_questions.side_effect = stream_q_collision
    app.dependency_overrides[get_db_manager] = lambda: mock_db_manager
    try:
        create_res = client.post(
            "/api/v1/question-sets", data={"name": "Benchmark Set 1"}
        )
        assert create_res.status_code == 201
        set_id = create_res.json()["id"]

        res = client.get(f"/api/v1/question-sets/{set_id}/export?format=jsonl")
        assert res.status_code == 200

        # Check RFC 5987 Content-Disposition header format
        cd_header = res.headers["content-disposition"]
        assert "attachment; filename=" in cd_header
        assert "filename*=UTF-8''" in cd_header

        # Parse exported line
        exported_data = json.loads(res.text.strip())
        assert exported_data["question_id"] == "orig-qid"
        assert exported_data["user_input"] == "orig-input"
        assert exported_data["custom_field"] == "safe-value"
        assert exported_data["extra_question_id"] == "malicious-qid"
        assert exported_data["extra_user_input"] == "malicious-input"
    finally:
        app.dependency_overrides.clear()
