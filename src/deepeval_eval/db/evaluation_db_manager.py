from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepeval_eval.db.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class EvaluationDBManager:
    """PostgreSQL database manager for Evaluation jobs, runs, and results."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def init_tables(self) -> None:
        """Initialize PostgreSQL schema tables for evaluation jobs and results."""
        if not self.db_manager.is_postgres():
            return

        conn = self.db_manager.get_connection()
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
            logger.warning(f"EvaluationDBManager schema initialization skipped: {exc}")
            raise
        finally:
            conn.close()
