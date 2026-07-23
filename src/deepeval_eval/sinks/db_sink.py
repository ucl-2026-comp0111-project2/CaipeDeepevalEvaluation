from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PipelineTokenUsage(BaseModel):
    """Token usage metrics for the RAG pipeline generation under test."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class EvaluatorUsage(BaseModel):
    """Token and execution metrics for the DeepEval evaluator judge calls."""

    evaluation_time_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RunSummaryPayload(BaseModel):
    """Structured, validated schema for evaluation run summaries persisted to PostgreSQL."""

    experiment_name: str
    datasource: str = "unknown"
    config_args: dict[str, Any] = Field(default_factory=dict)
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    total_tokens: int = 0
    rag_pipeline_token_usage: PipelineTokenUsage = Field(
        default_factory=PipelineTokenUsage
    )
    total_results: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    failure_causes: dict[str, int] = Field(default_factory=dict)
    deepeval_evaluator_usage: EvaluatorUsage = Field(default_factory=EvaluatorUsage)


class DatabaseResultSink:
    """Persists evaluation run directly to PostgreSQL tables."""

    def __init__(
        self,
        connection_string: str | None = None,
        auto_init: bool = True,
    ):
        self.connection_string = connection_string
        if auto_init:
            try:
                self.init_db()
            except Exception as exc:
                logger.debug(
                    f"Deferred DB schema initialization on sink creation: {exc}"
                )

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

    def init_db(self, conn: Any | None = None) -> None:
        """Initialize database schema tables if they do not already exist."""
        close_conn = False
        if conn is None:
            conn = self._get_connection()
            close_conn = True

        schema_sql = """
        CREATE TABLE IF NOT EXISTS batches (
            batch_id    TEXT PRIMARY KEY,
            created_at  TIMESTAMP NOT NULL DEFAULT now(),
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            batch_id     TEXT NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
            config_name  TEXT NOT NULL,
            config_json  JSONB,
            started_at   TIMESTAMP,
            finished_at  TIMESTAMP,
            loaded_at    TIMESTAMP NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id         BIGSERIAL PRIMARY KEY,
            run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            batch_id   TEXT NOT NULL,
            question   TEXT,
            row_data   JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_results_row_data ON eval_results USING GIN (row_data);

        CREATE TABLE IF NOT EXISTS run_summary (
            run_id        TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            p50_latency   DOUBLE PRECISION,
            p95_latency   DOUBLE PRECISION,
            summary_json  JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_runs_batch_id      ON runs(batch_id);
        CREATE INDEX IF NOT EXISTS idx_runs_config_name   ON runs(config_name);
        CREATE INDEX IF NOT EXISTS idx_results_run_id     ON eval_results(run_id);
        """
        try:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            if close_conn:
                conn.commit()
        finally:
            if close_conn and conn is not None and not conn.closed:
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
            self.init_db(conn)
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

                # Populate run_summary
                from deepeval_eval.sinks.metrics_aggregator import (
                    calculate_latency_percentiles,
                    categorize_failure_causes,
                    compute_all_metric_averages,
                )

                latencies = [
                    r.get("latency", 0.0)
                    for r in results
                    if r.get("latency") is not None
                ]
                p50_latency, p95_latency = calculate_latency_percentiles(latencies)
                all_metric_averages = compute_all_metric_averages(results)
                rag_prompt_tokens = sum(
                    r.get("prompt_tokens") or r.get("input_tokens") or 0
                    for r in results
                )
                rag_completion_tokens = sum(
                    r.get("completion_tokens")
                    or r.get("output_tokens")
                    or r.get("generation_tokens")
                    or 0
                    for r in results
                )
                rag_total_tokens = sum(
                    r.get("total_tokens")
                    or (
                        (r.get("prompt_tokens") or r.get("input_tokens") or 0)
                        + (
                            r.get("completion_tokens")
                            or r.get("output_tokens")
                            or r.get("generation_tokens")
                            or 0
                        )
                    )
                    for r in results
                )
                failure_counts = categorize_failure_causes(results)
                evaluator_prompt_tokens = sum(
                    r.get("evaluator_input_tokens", 0) for r in results
                )
                evaluator_completion_tokens = sum(
                    r.get("evaluator_output_tokens", 0) for r in results
                )
                evaluator_total_tokens = (
                    evaluator_prompt_tokens + evaluator_completion_tokens
                )

                serializable_config = {}
                for k, v in config_args.items():
                    if k.startswith("_") or k in ("llm_api_key", "auth_token"):
                        continue
                    try:
                        json.dumps(v)
                        serializable_config[k] = v
                    except (TypeError, OverflowError):
                        serializable_config[k] = str(v)

                summary_payload = RunSummaryPayload(
                    experiment_name=run_id,
                    datasource=str(config_args.get("datasource", "unknown")),
                    config_args=serializable_config,
                    p50_latency=p50_latency,
                    p95_latency=p95_latency,
                    total_tokens=rag_total_tokens,
                    rag_pipeline_token_usage=PipelineTokenUsage(
                        prompt_tokens=rag_prompt_tokens,
                        completion_tokens=rag_completion_tokens,
                        total_tokens=rag_total_tokens,
                    ),
                    total_results=len(results),
                    metrics=all_metric_averages,
                    failure_causes=failure_counts,
                    deepeval_evaluator_usage=EvaluatorUsage(
                        evaluation_time_seconds=evaluation_time,
                        prompt_tokens=evaluator_prompt_tokens,
                        completion_tokens=evaluator_completion_tokens,
                        total_tokens=evaluator_total_tokens,
                    ),
                )

                cur.execute(
                    """
                    INSERT INTO run_summary (run_id, p50_latency, p95_latency, summary_json)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE
                        SET p50_latency  = EXCLUDED.p50_latency,
                            p95_latency  = EXCLUDED.p95_latency,
                            summary_json = EXCLUDED.summary_json
                    """,
                    (
                        run_id,
                        p50_latency,
                        p95_latency,
                        summary_payload.model_dump_json(),
                    ),
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
