from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

try:
    from psycopg2.extras import execute_values
except ImportError:
    execute_values = None

if TYPE_CHECKING:
    from deepeval_eval.db.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

MAX_BATCH_DELETE_ITEMS = 1000


class QuestionDBManager:
    """PostgreSQL database manager for Question Sets and Questions."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def init_tables(self) -> None:
        """Initialize PostgreSQL schema for question_sets and questions."""
        if not self.db_manager.is_postgres():
            return

        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS question_sets (
                        id            BIGSERIAL PRIMARY KEY,
                        name          TEXT NOT NULL,
                        description   TEXT,
                        source_format TEXT,
                        created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                        updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
                    );

                    CREATE TABLE IF NOT EXISTS questions (
                        id               BIGSERIAL PRIMARY KEY,
                        question_set_id  BIGINT NOT NULL REFERENCES question_sets(id) ON DELETE CASCADE,
                        question_id      TEXT,
                        input            TEXT NOT NULL,
                        expected_output  TEXT,
                        category         TEXT,
                        level            TEXT,
                        expected_doc_ids TEXT[] NOT NULL DEFAULT '{}'::text[],
                        context          JSONB,
                        extra            JSONB,
                        created_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                        updated_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                        CONSTRAINT questions_question_set_id_question_id_key UNIQUE (question_set_id, question_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_question_sets_name ON question_sets (name);
                    CREATE INDEX IF NOT EXISTS idx_questions_set_category ON questions (question_set_id, category);
                    CREATE INDEX IF NOT EXISTS idx_questions_set_id ON questions (question_set_id, id);
                    """
                )
            conn.commit()
        except Exception as exc:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.warning(f"QuestionDBManager table initialization failed: {exc}")
            raise
        finally:
            conn.close()

    def create_question_set(
        self,
        name: str,
        description: str | None = None,
        source_format: str | None = None,
    ) -> dict[str, Any]:
        """Create a new question set."""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO question_sets (name, description, source_format)
                    VALUES (%s, %s, %s)
                    RETURNING id, name, description, source_format, created_at, updated_at;
                    """,
                    (name, description, source_format),
                )
                row = cur.fetchone()
                conn.commit()
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "source_format": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "updated_at": row[5].isoformat() if row[5] else None,
                    "question_count": 0,
                }
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def list_question_sets(
        self, page: int = 1, limit: int = 50, query: str | None = None
    ) -> dict[str, Any]:
        """List question sets with pagination and search."""
        offset = (max(1, page) - 1) * limit
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                where_clause = ""
                params: list[Any] = []
                if query and query.strip():
                    where_clause = "WHERE qs.name ILIKE %s OR qs.description ILIKE %s"
                    search_pat = f"%{query.strip()}%"
                    params.extend([search_pat, search_pat])

                # Count total matching rows
                count_sql = f"SELECT COUNT(*) FROM question_sets qs {where_clause};"
                cur.execute(count_sql, tuple(params))
                total = cur.fetchone()[0]

                # Fetch paginated question sets using correlated subquery for count aggregation
                fetch_sql = f"""
                    SELECT 
                        qs.id, 
                        qs.name, 
                        qs.description, 
                        qs.source_format, 
                        qs.created_at, 
                        qs.updated_at,
                        (SELECT COUNT(*) FROM questions q WHERE q.question_set_id = qs.id) AS question_count
                    FROM question_sets qs
                    {where_clause}
                    ORDER BY qs.id DESC
                    LIMIT %s OFFSET %s;
                """
                fetch_params = params + [limit, offset]
                cur.execute(fetch_sql, tuple(fetch_params))
                rows = cur.fetchall()

                items = [
                    {
                        "id": r[0],
                        "name": r[1],
                        "description": r[2],
                        "source_format": r[3],
                        "created_at": r[4].isoformat() if r[4] else None,
                        "updated_at": r[5].isoformat() if r[5] else None,
                        "question_count": r[6],
                    }
                    for r in rows
                ]

                return {
                    "items": items,
                    "total": total,
                    "page": page,
                    "limit": limit,
                    "total_pages": (total + limit - 1) // limit if limit > 0 else 0,
                }
        finally:
            conn.close()

    def get_question_set(self, set_id: int) -> dict[str, Any] | None:
        """Get details and summary stats of a question set by ID."""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        qs.id, 
                        qs.name, 
                        qs.description, 
                        qs.source_format, 
                        qs.created_at, 
                        qs.updated_at,
                        COUNT(q.id) AS question_count
                    FROM question_sets qs
                    LEFT JOIN questions q ON qs.id = q.question_set_id
                    WHERE qs.id = %s
                    GROUP BY qs.id;
                    """,
                    (set_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                # Fetch category distribution stats
                cur.execute(
                    """
                    SELECT category, COUNT(*) 
                    FROM questions 
                    WHERE question_set_id = %s 
                    GROUP BY category;
                    """,
                    (set_id,),
                )
                cat_rows = cur.fetchall()
                categories = {
                    (cat if cat is not None else "uncategorized"): cnt
                    for cat, cnt in cat_rows
                }

                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "source_format": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "updated_at": row[5].isoformat() if row[5] else None,
                    "question_count": row[6],
                    "categories": categories,
                }
        finally:
            conn.close()

    def update_question_set(
        self,
        set_id: int,
        name: str | None = None,
        description: str | None = None,
        source_format: str | None = None,
    ) -> dict[str, Any] | None:
        """Update metadata of a question set."""
        updates: list[str] = []
        params: list[Any] = []

        if name is not None:
            updates.append("name = %s")
            params.append(name)
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        if source_format is not None:
            updates.append("source_format = %s")
            params.append(source_format)

        if not updates:
            return self.get_question_set(set_id)

        updates.append("updated_at = now()")
        params.append(set_id)

        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE question_sets SET {', '.join(updates)} WHERE id = %s RETURNING id;",
                    tuple(params),
                )
                if cur.fetchone() is None:
                    if hasattr(conn, "rollback"):
                        conn.rollback()
                    return None
                conn.commit()
            return self.get_question_set(set_id)
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def delete_question_set(self, set_id: int) -> bool:
        """Delete a question set and all associated questions."""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM question_sets WHERE id = %s RETURNING id;", (set_id,)
                )
                row = cur.fetchone()
                conn.commit()
                return row is not None
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def add_questions(
        self, set_id: int, questions_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Batch insert questions into a question set."""
        if not questions_data:
            return []

        conn = self.db_manager.get_connection()
        inserted_rows: list[dict[str, Any]] = []
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM question_sets WHERE id = %s;", (set_id,))
                if not cur.fetchone():
                    raise ValueError(f"Question set with id={set_id} does not exist.")

                prepared_params: list[tuple[Any, ...]] = []
                for idx, item in enumerate(questions_data, start=1):
                    # Key mappings for alias compatibility (user_input -> input, reference -> expected_output)
                    inp = item.get("input") or item.get("user_input")
                    if not inp:
                        raise ValueError(
                            f"Question item at index {idx} missing required field 'input' or 'user_input'."
                        )

                    qid = item.get("question_id")
                    if not qid or not str(qid).strip():
                        qid = f"q-{idx}"

                    exp_out = item.get("expected_output") or item.get("reference")
                    category = item.get("category")
                    level = item.get("level")
                    doc_ids = item.get("expected_doc_ids") or []
                    if not isinstance(doc_ids, list):
                        doc_ids = [str(doc_ids)]
                    else:
                        doc_ids = [str(d) for d in doc_ids]

                    ctx = item.get("context")
                    extra = item.get("extra")

                    # If extra metadata fields exist in item outside known schema fields, capture in extra dict
                    known_keys = {
                        "question_id",
                        "input",
                        "user_input",
                        "expected_output",
                        "reference",
                        "category",
                        "level",
                        "expected_doc_ids",
                        "context",
                        "extra",
                    }
                    leftover_keys = {
                        k: v for k, v in item.items() if k not in known_keys
                    }
                    if leftover_keys:
                        if extra is None:
                            extra = leftover_keys
                        elif isinstance(extra, dict):
                            extra = {**extra, **leftover_keys}

                    ctx_json = json.dumps(ctx) if ctx is not None else None
                    extra_json = json.dumps(extra) if extra is not None else None

                    prepared_params.append(
                        (
                            set_id,
                            qid,
                            inp,
                            exp_out,
                            category,
                            level,
                            doc_ids,
                            ctx_json,
                            extra_json,
                        )
                    )

                batch_sql = """
                    INSERT INTO questions (
                        question_set_id, question_id, input, expected_output,
                        category, level, expected_doc_ids, context, extra
                    )
                    VALUES %s
                    ON CONFLICT (question_set_id, question_id) 
                    DO UPDATE SET
                        input = EXCLUDED.input,
                        expected_output = EXCLUDED.expected_output,
                        category = EXCLUDED.category,
                        level = EXCLUDED.level,
                        expected_doc_ids = EXCLUDED.expected_doc_ids,
                        context = EXCLUDED.context,
                        extra = EXCLUDED.extra,
                        updated_at = now()
                    RETURNING id, question_set_id, question_id, input, expected_output,
                              category, level, expected_doc_ids, context, extra, created_at, updated_at;
                """

                single_sql = """
                    INSERT INTO questions (
                        question_set_id, question_id, input, expected_output,
                        category, level, expected_doc_ids, context, extra
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (question_set_id, question_id) 
                    DO UPDATE SET
                        input = EXCLUDED.input,
                        expected_output = EXCLUDED.expected_output,
                        category = EXCLUDED.category,
                        level = EXCLUDED.level,
                        expected_doc_ids = EXCLUDED.expected_doc_ids,
                        context = EXCLUDED.context,
                        extra = EXCLUDED.extra,
                        updated_at = now()
                    RETURNING id, question_set_id, question_id, input, expected_output,
                              category, level, expected_doc_ids, context, extra, created_at, updated_at;
                """

                use_batch = execute_values is not None
                if use_batch:
                    try:
                        fetched = execute_values(
                            cur, batch_sql, prepared_params, fetch=True
                        )
                        for r in fetched:
                            inserted_rows.append(
                                {
                                    "id": r[0],
                                    "question_set_id": r[1],
                                    "question_id": r[2],
                                    "input": r[3],
                                    "expected_output": r[4],
                                    "category": r[5],
                                    "level": r[6],
                                    "expected_doc_ids": r[7],
                                    "context": r[8],
                                    "extra": r[9],
                                    "created_at": r[10].isoformat() if r[10] else None,
                                    "updated_at": r[11].isoformat() if r[11] else None,
                                }
                            )
                    except Exception:
                        use_batch = False

                if not use_batch:
                    for row_params in prepared_params:
                        cur.execute(single_sql, row_params)
                        r = cur.fetchone()
                        if r:
                            inserted_rows.append(
                                {
                                    "id": r[0],
                                    "question_set_id": r[1],
                                    "question_id": r[2],
                                    "input": r[3],
                                    "expected_output": r[4],
                                    "category": r[5],
                                    "level": r[6],
                                    "expected_doc_ids": r[7],
                                    "context": r[8],
                                    "extra": r[9],
                                    "created_at": r[10].isoformat() if r[10] else None,
                                    "updated_at": r[11].isoformat() if r[11] else None,
                                }
                            )

                # Touch updated_at on question set
                cur.execute(
                    "UPDATE question_sets SET updated_at = now() WHERE id = %s;",
                    (set_id,),
                )
                conn.commit()
                return inserted_rows
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_questions(
        self,
        set_id: int,
        page: int = 1,
        limit: int = 50,
        category: str | None = None,
        level: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """List questions in a question set with pagination and filters."""
        offset = (max(1, page) - 1) * limit
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                where_clauses = ["question_set_id = %s"]
                params: list[Any] = [set_id]

                if category and category.strip():
                    where_clauses.append("category = %s")
                    params.append(category.strip())
                if level and level.strip():
                    where_clauses.append("level = %s")
                    params.append(level.strip())
                if query and query.strip():
                    where_clauses.append(
                        "(input ILIKE %s OR expected_output ILIKE %s OR question_id ILIKE %s)"
                    )
                    pat = f"%{query.strip()}%"
                    params.extend([pat, pat, pat])

                where_sql = " WHERE " + " AND ".join(where_clauses)

                # Total count
                cur.execute(
                    f"SELECT COUNT(*) FROM questions{where_sql};", tuple(params)
                )
                total = cur.fetchone()[0]

                # Fetch rows
                fetch_sql = f"""
                    SELECT 
                        id, question_set_id, question_id, input, expected_output,
                        category, level, expected_doc_ids, context, extra, created_at, updated_at
                    FROM questions
                    {where_sql}
                    ORDER BY id ASC
                    LIMIT %s OFFSET %s;
                """
                fetch_params = params + [limit, offset]
                cur.execute(fetch_sql, tuple(fetch_params))
                rows = cur.fetchall()

                items = [
                    {
                        "id": r[0],
                        "question_set_id": r[1],
                        "question_id": r[2],
                        "input": r[3],
                        "expected_output": r[4],
                        "category": r[5],
                        "level": r[6],
                        "expected_doc_ids": r[7] if r[7] is not None else [],
                        "context": r[8],
                        "extra": r[9],
                        "created_at": r[10].isoformat() if r[10] else None,
                        "updated_at": r[11].isoformat() if r[11] else None,
                    }
                    for r in rows
                ]

                return {
                    "items": items,
                    "total": total,
                    "page": page,
                    "limit": limit,
                    "total_pages": (total + limit - 1) // limit if limit > 0 else 0,
                }
        finally:
            conn.close()

    def stream_questions(
        self, set_id: int, batch_size: int = 1000
    ) -> Iterator[dict[str, Any]]:
        """Stream questions in batches using keyset pagination (WHERE id > last_id) for zero offset drift."""
        conn = self.db_manager.get_connection()
        try:
            last_id = 0
            while True:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, question_set_id, question_id, input, expected_output,
                               category, level, expected_doc_ids, context, extra, created_at, updated_at
                        FROM questions
                        WHERE question_set_id = %s AND id > %s
                        ORDER BY id ASC
                        LIMIT %s;
                        """,
                        (set_id, last_id, batch_size),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        break
                    for r in rows:
                        yield {
                            "id": r[0],
                            "question_set_id": r[1],
                            "question_id": r[2],
                            "input": r[3],
                            "expected_output": r[4],
                            "category": r[5],
                            "level": r[6],
                            "expected_doc_ids": r[7] if r[7] is not None else [],
                            "context": r[8],
                            "extra": r[9],
                            "created_at": r[10].isoformat() if r[10] else None,
                            "updated_at": r[11].isoformat() if r[11] else None,
                        }
                    last_id = rows[-1][0]
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn is not None and hasattr(conn, "close"):
                conn.close()

    def _execute_question_update(
        self,
        set_id: int,
        q_identifier: int | str,
        data: dict[str, Any],
        key_name: str = "id",
    ) -> dict[str, Any] | None:
        """Helper to execute SQL update for a question by key (id or question_id)."""
        allowed_keys = {"id", "question_id"}
        if key_name not in allowed_keys:
            raise ValueError(f"Invalid column update key: {key_name}")

        updates: list[str] = []
        params: list[Any] = []

        if "question_id" in data and data["question_id"] is not None:
            updates.append("question_id = %s")
            params.append(data["question_id"])
        if "input" in data and data["input"] is not None:
            updates.append("input = %s")
            params.append(data["input"])
        elif "user_input" in data and data["user_input"] is not None:
            updates.append("input = %s")
            params.append(data["user_input"])
        if "expected_output" in data:
            updates.append("expected_output = %s")
            params.append(data["expected_output"])
        elif "reference" in data:
            updates.append("expected_output = %s")
            params.append(data["reference"])
        if "category" in data:
            updates.append("category = %s")
            params.append(data["category"])
        if "level" in data:
            updates.append("level = %s")
            params.append(data["level"])
        if "expected_doc_ids" in data and data["expected_doc_ids"] is not None:
            doc_ids = data["expected_doc_ids"]
            if not isinstance(doc_ids, list):
                doc_ids = [str(doc_ids)]
            else:
                doc_ids = [str(d) for d in doc_ids]
            updates.append("expected_doc_ids = %s")
            params.append(doc_ids)
        if "context" in data:
            updates.append("context = %s")
            params.append(
                json.dumps(data["context"]) if data["context"] is not None else None
            )
        if "extra" in data:
            updates.append("extra = %s")
            params.append(
                json.dumps(data["extra"]) if data["extra"] is not None else None
            )

        if not updates:
            return (
                self.get_question_by_id(set_id, int(q_identifier))
                if key_name == "id"
                else self.get_question_by_question_id(set_id, str(q_identifier))
            )

        updates.append("updated_at = now()")
        params.extend([set_id, q_identifier])

        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                if key_name == "id":
                    sql = f"UPDATE questions SET {', '.join(updates)} WHERE question_set_id = %s AND id = %s RETURNING id;"
                else:
                    sql = f"UPDATE questions SET {', '.join(updates)} WHERE question_set_id = %s AND question_id = %s RETURNING id;"
                cur.execute(sql, tuple(params))
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    "UPDATE question_sets SET updated_at = now() WHERE id = %s;",
                    (set_id,),
                )
                conn.commit()
                return self.get_question_by_id(set_id, row[0])
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def get_question_by_id(self, set_id: int, id: int) -> dict[str, Any] | None:
        """Get a single question by id."""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, question_set_id, question_id, input, expected_output,
                           category, level, expected_doc_ids, context, extra, created_at, updated_at
                    FROM questions
                    WHERE question_set_id = %s AND id = %s;
                    """,
                    (set_id, id),
                )
                r = cur.fetchone()
                if not r:
                    return None
                return {
                    "id": r[0],
                    "question_set_id": r[1],
                    "question_id": r[2],
                    "input": r[3],
                    "expected_output": r[4],
                    "category": r[5],
                    "level": r[6],
                    "expected_doc_ids": r[7] if r[7] is not None else [],
                    "context": r[8],
                    "extra": r[9],
                    "created_at": r[10].isoformat() if r[10] else None,
                    "updated_at": r[11].isoformat() if r[11] else None,
                }
        finally:
            conn.close()

    def get_question_by_question_id(
        self, set_id: int, question_id: str
    ) -> dict[str, Any] | None:
        """Get a single question by question_id."""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, question_set_id, question_id, input, expected_output,
                           category, level, expected_doc_ids, context, extra, created_at, updated_at
                    FROM questions
                    WHERE question_set_id = %s AND question_id = %s;
                    """,
                    (set_id, str(question_id).strip()),
                )
                r = cur.fetchone()
                if not r:
                    return None
                return {
                    "id": r[0],
                    "question_set_id": r[1],
                    "question_id": r[2],
                    "input": r[3],
                    "expected_output": r[4],
                    "category": r[5],
                    "level": r[6],
                    "expected_doc_ids": r[7] if r[7] is not None else [],
                    "context": r[8],
                    "extra": r[9],
                    "created_at": r[10].isoformat() if r[10] else None,
                    "updated_at": r[11].isoformat() if r[11] else None,
                }
        finally:
            conn.close()

    def update_question_by_id(
        self, set_id: int, id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update a specific question by id."""
        return self._execute_question_update(set_id, id, data, key_name="id")

    def update_question_by_question_id(
        self, set_id: int, question_id: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update a specific question by question_id."""
        return self._execute_question_update(
            set_id, str(question_id).strip(), data, key_name="question_id"
        )

    def delete_question_by_id(self, set_id: int, id: int) -> bool:
        """Delete a single question from a set by id."""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM questions WHERE question_set_id = %s AND id = %s RETURNING id;",
                    (set_id, id),
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "UPDATE question_sets SET updated_at = now() WHERE id = %s;",
                        (set_id,),
                    )
                conn.commit()
                return row is not None
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def delete_question_by_question_id(self, set_id: int, question_id: str) -> bool:
        """Delete a single question from a set by question_id."""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM questions WHERE question_set_id = %s AND question_id = %s RETURNING id;",
                    (set_id, str(question_id).strip()),
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "UPDATE question_sets SET updated_at = now() WHERE id = %s;",
                        (set_id,),
                    )
                conn.commit()
                return row is not None
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def batch_delete_questions(
        self,
        set_id: int,
        ids: list[int] | None = None,
        question_ids: list[str] | None = None,
    ) -> int:
        """Atomically delete questions by db_ids, question_ids, or both in a single query."""
        ids_clean: list[int] = []
        if ids:
            parsed_ids: list[int] = []
            for i in ids:
                try:
                    parsed_ids.append(int(i))
                except (ValueError, TypeError):
                    pass
            ids_clean = list(dict.fromkeys(parsed_ids))

        qids_clean: list[str] = []
        if question_ids:
            raw_qids = [
                str(q).strip() for q in question_ids if q is not None and str(q).strip()
            ]
            qids_clean = list(dict.fromkeys(raw_qids))

        if len(ids_clean) + len(qids_clean) > MAX_BATCH_DELETE_ITEMS:
            raise ValueError(
                f"Batch delete payload exceeds the maximum limit of {MAX_BATCH_DELETE_ITEMS:,} total items."
            )

        if not ids_clean and not qids_clean:
            return 0

        where_conditions: list[str] = []
        params: list[Any] = [set_id]

        if ids_clean:
            where_conditions.append("id = ANY(%s)")
            params.append(ids_clean)

        if qids_clean:
            where_conditions.append("question_id = ANY(%s)")
            params.append(qids_clean)

        sql = f"""
            DELETE FROM questions
            WHERE question_set_id = %s
              AND ({" OR ".join(where_conditions)})
            RETURNING id;
        """

        if len(params) != 1 + len(where_conditions):
            raise RuntimeError(
                "Mismatched parameter bindings during batch delete SQL construction."
            )

        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                deleted_rows = cur.fetchall()
                deleted_count = len(deleted_rows)
                if deleted_count > 0:
                    cur.execute(
                        "UPDATE question_sets SET updated_at = now() WHERE id = %s;",
                        (set_id,),
                    )
                conn.commit()
                return deleted_count
        except Exception:
            if conn is not None and hasattr(conn, "rollback"):
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def batch_delete_questions_by_ids(self, set_id: int, ids: list[int]) -> int:
        """Delete multiple questions in a set by IDs."""
        return self.batch_delete_questions(set_id, ids=ids)

    def batch_delete_questions_by_question_ids(
        self, set_id: int, question_ids: list[str]
    ) -> int:
        """Delete multiple questions in a set by question_ids."""
        return self.batch_delete_questions(set_id, question_ids=question_ids)
