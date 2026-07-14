import json
from pathlib import Path
import pytest
from deepeval_eval.io_utils import load_eval_questions


def test_load_eval_questions_basic(tmp_path: Path):
    # Prepare a mock jsonl file
    questions_file = tmp_path / "questions.jsonl"
    data = [
        {"user_input": "q1", "reference": "r1", "category": "cat1"},
        {"user_input": "q2", "reference": "r2", "category": "cat2"},
        {"user_input": "q3", "reference": "r3", "category": "cat1"},
        {"user_input": "q4", "reference": "r4", "category": "cat2"},
        {"user_input": "q5", "reference": "r5", "category": "cat3"},
    ]
    with open(questions_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # 1. Test basic loading without limits
    rows = load_eval_questions(questions_file, max_items=None, limit_per_category=None)
    assert len(rows) == 5
    assert [r["user_input"] for r in rows] == ["q1", "q2", "q3", "q4", "q5"]

    # 2. Test max_items limit
    rows = load_eval_questions(questions_file, max_items=3, limit_per_category=None)
    assert len(rows) == 3
    assert [r["user_input"] for r in rows] == ["q1", "q2", "q3"]

    # 3. Test limit_per_category limit
    # Expected: "cat1" has q1 (kept), q3 (kept if limit >= 2). With limit 1, we keep q1, q2, q4 (since cat2 limit is 1, keep q2 and drop q4), q5.
    # Let's check with limit_per_category = 1
    rows = load_eval_questions(questions_file, max_items=None, limit_per_category=1)
    assert len(rows) == 3
    assert [r["user_input"] for r in rows] == ["q1", "q2", "q5"]

    # 4. Test combination of both max_items and limit_per_category
    rows = load_eval_questions(questions_file, max_items=2, limit_per_category=1)
    assert len(rows) == 2
    assert [r["user_input"] for r in rows] == ["q1", "q2"]


def test_load_eval_questions_missing_category(tmp_path: Path):
    questions_file = tmp_path / "questions.jsonl"
    data = [
        {"user_input": "q1", "reference": "r1"},
        {"user_input": "q2", "reference": "r2"},
    ]
    with open(questions_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # With missing category, it defaults to 'basic'
    rows = load_eval_questions(questions_file, max_items=None, limit_per_category=1)
    assert len(rows) == 1
    assert rows[0]["user_input"] == "q1"
