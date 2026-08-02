from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deepeval_eval.db.db_manager import DatabaseManager
from deepeval_eval.db.evaluation_db_manager import EvaluationDBManager
from deepeval_eval.db.question_db_manager import QuestionDBManager

# ---------------------------------------------------------------------------
# Unit Tests for QuestionDBManager (Direct Database Methods)
# ---------------------------------------------------------------------------


def test_question_db_manager_init_tables_non_postgres():
    """Verify init_tables returns early when PostgreSQL is not configured."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_base_db.is_postgres.return_value = False

    manager = QuestionDBManager(mock_base_db)
    manager.init_tables()

    mock_base_db.get_connection.assert_not_called()


def test_question_db_manager_init_tables_positive():
    """Verify init_tables executes schema creation SQL statements on PostgreSQL connection."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_base_db.is_postgres.return_value = True

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    manager.init_tables()

    assert mock_cur.execute.called
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


def test_question_db_manager_create_question_set_positive():
    """Verify create_question_set inserts record and returns formatted dictionary."""
    mock_base_db = MagicMock(spec=DatabaseManager)

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (
        1,
        "Enterprise RAG Bench",
        "Test dataset description",
        "jsonl",
        MagicMock(isoformat=lambda: "2026-08-02T00:00:00+00:00"),
        MagicMock(isoformat=lambda: "2026-08-02T00:00:00+00:00"),
    )
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    res = manager.create_question_set(
        name="Enterprise RAG Bench",
        description="Test dataset description",
        source_format="jsonl",
    )

    assert res["id"] == 1
    assert res["name"] == "Enterprise RAG Bench"
    assert res["description"] == "Test dataset description"
    assert res["source_format"] == "jsonl"
    assert res["question_count"] == 0
    mock_conn.commit.assert_called_once()


def test_question_db_manager_get_question_set_negative():
    """Verify get_question_set returns None for non-existent set ID."""
    mock_base_db = MagicMock(spec=DatabaseManager)

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    res = manager.get_question_set(set_id=999)

    assert res is None


def test_question_db_manager_update_question_set_negative():
    """Verify update_question_set returns None when target set_id does not exist."""
    mock_base_db = MagicMock(spec=DatabaseManager)

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    res = manager.update_question_set(set_id=999, name="Non-existent")

    assert res is None


def test_question_db_manager_delete_question_set_negative():
    """Verify delete_question_set returns False when deleting non-existent set ID."""
    mock_base_db = MagicMock(spec=DatabaseManager)

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    success = manager.delete_question_set(set_id=999)

    assert success is False


def test_question_db_manager_add_questions_negative_non_existent_set():
    """Verify add_questions raises ValueError when set_id does not exist."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    with pytest.raises(ValueError, match="does not exist"):
        manager.add_questions(set_id=999, questions_data=[{"input": "Test"}])


def test_question_db_manager_add_questions_negative_missing_input():
    """Verify add_questions raises ValueError when question item lacks input field."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (1,)
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    with pytest.raises(ValueError, match="missing required field 'input'"):
        manager.add_questions(set_id=1, questions_data=[{"category": "test"}])


def test_question_db_manager_get_question_negative():
    """Verify get_question returns None for non-existent question identifier."""
    mock_base_db = MagicMock(spec=DatabaseManager)

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    res = manager.get_question_by_question_id(set_id=1, question_id="non_existent_q")

    assert res is None


def test_question_db_manager_batch_delete_questions_negative_empty():
    """Verify batch_delete methods return 0 when empty list of identifiers is provided."""
    mock_base_db = MagicMock(spec=DatabaseManager)

    manager = QuestionDBManager(mock_base_db)
    assert manager.batch_delete_questions_by_ids(set_id=1, ids=[]) == 0
    assert (
        manager.batch_delete_questions_by_question_ids(set_id=1, question_ids=[]) == 0
    )


def test_question_db_manager_batch_delete_questions_by_ids_positive():
    """Verify batch_delete_questions_by_ids executes correct query for integer PKs."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [(1,), (2,)]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    count = manager.batch_delete_questions_by_ids(set_id=10, ids=[1, 2])

    assert count == 2
    executed_sql = mock_cur.execute.call_args_list[0][0][0]
    assert "id = ANY(%s)" in executed_sql
    assert "WHERE question_set_id = %s" in executed_sql


def test_question_db_manager_batch_delete_questions_by_question_ids_positive():
    """Verify batch_delete_questions_by_question_ids executes correct query for string question_ids."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [(101,)]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    manager = QuestionDBManager(mock_base_db)
    count = manager.batch_delete_questions_by_question_ids(
        set_id=5, question_ids=["q_alpha", "q_beta"]
    )

    assert count == 1
    executed_sql = mock_cur.execute.call_args_list[0][0][0]
    assert "question_id = ANY(%s)" in executed_sql
    assert "WHERE question_set_id = %s" in executed_sql


def test_question_db_manager_batch_delete_oversized_payload_raises_value_error():
    """Verify batch_delete_questions raises ValueError when payload exceeds 1000 items."""
    import pytest

    mock_base_db = MagicMock(spec=DatabaseManager)
    manager = QuestionDBManager(mock_base_db)

    oversized_ids = list(range(1001))
    with pytest.raises(
        ValueError,
        match="Batch delete payload exceeds the maximum limit of 1,000 total items",
    ):
        manager.batch_delete_questions(set_id=1, ids=oversized_ids)


# ---------------------------------------------------------------------------
# Unit Tests for EvaluationDBManager
# ---------------------------------------------------------------------------


def test_evaluation_db_manager_init_tables_non_postgres():
    """Verify EvaluationDBManager.init_tables returns early when non-PostgreSQL."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_base_db.is_postgres.return_value = False

    eval_manager = EvaluationDBManager(mock_base_db)
    eval_manager.init_tables()

    mock_base_db.get_connection.assert_not_called()


def test_evaluation_db_manager_init_tables_positive():
    """Verify EvaluationDBManager.init_tables creates evaluation schema tables."""
    mock_base_db = MagicMock(spec=DatabaseManager)
    mock_base_db.is_postgres.return_value = True

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_base_db.get_connection.return_value = mock_conn

    eval_manager = EvaluationDBManager(mock_base_db)
    eval_manager.init_tables()

    assert mock_cur.execute.called
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()
