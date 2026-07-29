from __future__ import annotations

import logging
import os
import threading
from typing import Any

from pydantic import SecretStr

from deepeval_eval.core.config import DatabaseSettings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Standardized PostgreSQL database manager."""

    def __init__(
        self,
        connection_string: str | Any | None = None,
        db_settings: DatabaseSettings | None = None,
    ):
        raw_conn = (
            connection_string.get_secret_value()
            if isinstance(connection_string, SecretStr)
            else (str(connection_string) if connection_string is not None else None)
        )
        if (
            raw_conn is None
            and db_settings is not None
            and getattr(db_settings, "connection_string", None)
        ):
            setting_conn = db_settings.connection_string
            raw_conn = (
                setting_conn.get_secret_value()
                if isinstance(setting_conn, SecretStr)
                else (str(setting_conn) if setting_conn is not None else None)
            )
        self._explicit_connection_string = raw_conn
        self._db_settings = db_settings
        self._lock = threading.Lock()
        if self.is_postgres():
            try:
                self.init_db()
            except Exception as exc:
                logger.warning(
                    f"PostgreSQL schema init failed on startup (DB may be unreachable): {exc}"
                )

    @property
    def connection_string(self) -> str | None:
        if (
            getattr(self, "_explicit_connection_string", None)
            and self._explicit_connection_string.strip() != ""
        ):
            return self._explicit_connection_string

        settings = getattr(self, "_db_settings", None) or DatabaseSettings()
        conn = (
            settings.connection_string.get_secret_value()
            if isinstance(settings.connection_string, SecretStr)
            else (
                str(settings.connection_string) if settings.connection_string else None
            )
        )
        return conn if conn and conn.strip() != "" else None

    @connection_string.setter
    def connection_string(self, val: str | None) -> None:
        self._explicit_connection_string = val

    @property
    def postgres_host(self) -> str | None:
        settings = getattr(self, "_db_settings", None) or DatabaseSettings()
        return settings.postgres_host

    def is_postgres(self) -> bool:
        conn_str = self.connection_string
        if conn_str:
            return conn_str.startswith("postgresql://") or conn_str.startswith(
                "postgres://"
            )
        return bool(self.postgres_host)

    def get_connection(self) -> Any:
        if not self.is_postgres():
            raise RuntimeError("PostgreSQL database is not configured.")

        import sys

        psycopg2 = sys.modules.get("psycopg2") or __import__("psycopg2")

        conn_str = (
            self.connection_string
            or os.environ.get("DATABASE_URL")
            or os.environ.get("LANGGRAPH_CHECKPOINT_POSTGRES_DSN")
            or os.environ.get("POSTGRES_DSN")
        )
        if conn_str:
            return psycopg2.connect(conn_str, connect_timeout=5)

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
            connect_timeout=5,
        )

    def init_db(self) -> None:
        """Initialize PostgreSQL schema tables if not present."""
        if not self.is_postgres():
            return
        with self._lock:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS eval_job_queue (
                            job_id       TEXT PRIMARY KEY,
                            eval_hash    TEXT NOT NULL,
                            status       TEXT NOT NULL,
                            config_json  TEXT NOT NULL,
                            created_at   DOUBLE PRECISION NOT NULL,
                            started_at   DOUBLE PRECISION,
                            completed_at DOUBLE PRECISION,
                            error        TEXT
                        );
                        CREATE TABLE IF NOT EXISTS batches (
                            batch_id    TEXT PRIMARY KEY,
                            created_at  TIMESTAMP NOT NULL DEFAULT now(),
                            description TEXT
                        );
                        CREATE TABLE IF NOT EXISTS runs (
                            run_id       TEXT PRIMARY KEY,
                            batch_id     TEXT NOT NULL,
                            config_name  TEXT NOT NULL,
                            config_json  JSONB,
                            started_at   TIMESTAMP,
                            finished_at  TIMESTAMP,
                            loaded_at    TIMESTAMP NOT NULL DEFAULT now()
                        );
                        CREATE TABLE IF NOT EXISTS eval_results (
                            id         BIGSERIAL PRIMARY KEY,
                            run_id     TEXT NOT NULL,
                            batch_id   TEXT NOT NULL,
                            question   TEXT,
                            row_data   JSONB
                        );
                        CREATE TABLE IF NOT EXISTS run_summary (
                            run_id        TEXT PRIMARY KEY,
                            p50_latency   DOUBLE PRECISION,
                            p95_latency   DOUBLE PRECISION,
                            summary_json  JSONB
                        );
                        CREATE TABLE IF NOT EXISTS evaluation_runs (
                            run_id VARCHAR(255) PRIMARY KEY,
                            dataset_name VARCHAR(255),
                            total_questions INT,
                            completed_questions INT DEFAULT 0,
                            total_duration_seconds FLOAT DEFAULT 0,
                            p50_latency_sec FLOAT DEFAULT 0,
                            p95_latency_sec FLOAT DEFAULT 0,
                            metrics JSONB DEFAULT '{}'::jsonb,
                            failure_causes JSONB DEFAULT '{}'::jsonb,
                            evaluator_usage JSONB DEFAULT '{}'::jsonb,
                            status VARCHAR(50) DEFAULT 'RUNNING',
                            config JSONB DEFAULT '{}'::jsonb,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE TABLE IF NOT EXISTS evaluation_results (
                            id SERIAL PRIMARY KEY,
                            run_id VARCHAR(255) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                            question_id VARCHAR(255),
                            question TEXT,
                            user_input TEXT,
                            reference TEXT,
                            actual_output TEXT,
                            contexts JSONB,
                            doc_ids JSONB,
                            metrics JSONB,
                            latency_sec FLOAT,
                            pipeline_usage JSONB,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS idx_eval_job_queue_status_created ON eval_job_queue (status, created_at);
                        """
                    )
                conn.commit()
            except Exception as exc:
                if conn is not None and hasattr(conn, "rollback"):
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                logger.warning(f"PostgreSQL schema initialization skipped: {exc}")
            finally:
                if conn is not None and not getattr(conn, "closed", False):
                    conn.close()

    def execute_write(self, query: str, params: tuple[Any, ...]) -> None:
        with self._lock:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
            except Exception:
                if conn is not None and hasattr(conn, "rollback"):
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn is not None and not getattr(conn, "closed", False):
                    conn.close()

    def query_all(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self.get_connection()
            try:
                from psycopg2.extras import RealDictCursor

                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
            except Exception:
                if conn is not None and hasattr(conn, "rollback"):
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn is not None and not getattr(conn, "closed", False):
                    conn.close()
