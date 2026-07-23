from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseResultSink:
    """Persists evaluation run directly to PostgreSQL tables."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string

    def _get_connection(self) -> Any:
        import psycopg2

        conn_str = self.connection_string or os.environ.get("DATABASE_URL")
        if conn_str:
            return psycopg2.connect(conn_str)

        host = (
            os.environ.get("POSTGRES_HOST")
            or os.environ.get("PGHOST")
            or os.environ.get("DB_HOST", "localhost")
        )
        port = (
            os.environ.get("POSTGRES_PORT")
            or os.environ.get("PGPORT")
            or os.environ.get("DB_PORT", "5432")
        )
        dbname = (
            os.environ.get("POSTGRES_DB")
            or os.environ.get("PGDATABASE")
            or os.environ.get("DB_NAME", "caipe_eval")
        )
        user = (
            os.environ.get("POSTGRES_USER")
            or os.environ.get("PGUSER")
            or os.environ.get("DB_USER", "postgres")
        )
        password = (
            os.environ.get("POSTGRES_PASSWORD")
            or os.environ.get("PGPASSWORD")
            or os.environ.get("DB_PASSWORD", "")
        )
        sslmode = os.environ.get("PGSSLMODE", "prefer")

        return psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=sslmode,
        )

    def query_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Query recent evaluation runs stored in PostgreSQL database."""
        try:
            from psycopg2.extras import RealDictCursor
        except ImportError:
            logger.warning("psycopg2 is not installed; skipping database query.")
            return []

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT run_id, batch_id, config_name, loaded_at, config_json
                    FROM runs
                    ORDER BY loaded_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            if conn is not None and not conn.closed:
                conn.close()

    def save(
        self,
        results_dir: Path,
        prefix: str,
        results: list[dict[str, Any]],
        evaluation_time: float,
        config_args: dict[str, Any],
    ) -> None:
        try:
            from psycopg2.extras import execute_values
        except ImportError:
            logger.warning("psycopg2 is not installed; skipping database persistence.")
            return

        batch_id = config_args.get("batch_id") or f"batch_{time.strftime('%Y%m%d')}"
        run_id = (
            config_args.get("run_id") or f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        config_name = config_args.get("config_name") or prefix

        try:
            conn = self._get_connection()
        except Exception as exc:
            logger.warning(
                f"Failed to connect to database for direct persistence: {exc}"
            )
            return

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO batches (batch_id)
                    VALUES (%s)
                    ON CONFLICT (batch_id) DO NOTHING
                    """,
                    (batch_id,),
                )
                cur.execute(
                    """
                    INSERT INTO runs (run_id, batch_id, config_name, config_json)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE
                        SET config_json = EXCLUDED.config_json,
                            loaded_at   = now()
                    """,
                    (
                        run_id,
                        batch_id,
                        config_name,
                        json.dumps(config_args, default=str),
                    ),
                )

                cur.execute("DELETE FROM eval_results WHERE run_id = %s", (run_id,))

                rows = [
                    (
                        run_id,
                        batch_id,
                        r.get("question"),
                        json.dumps(r, default=str),
                    )
                    for r in results
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO eval_results (run_id, batch_id, question, row_data)
                    VALUES %s
                    """,
                    rows,
                    template="(%s, %s, %s, %s::jsonb)",
                )
                conn.commit()
            logger.info(
                f"Persisted run '{run_id}' ({len(results)} rows) to Postgres DB."
            )
        except Exception:
            if conn is not None and not conn.closed:
                conn.rollback()
            logger.exception("Failed to insert eval results into database.")
        finally:
            if conn is not None and not conn.closed:
                conn.close()
