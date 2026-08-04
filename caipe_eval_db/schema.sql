-- CAIPE DeepEval results schema
-- Read-after-complete design: rows are inserted once a run's CSV/JSON output
-- exists on disk. No partial-run / streaming writes assumed.

CREATE TABLE IF NOT EXISTS batches (
    batch_id    TEXT PRIMARY KEY,       -- e.g. batch timestamp
    created_at  TIMESTAMP NOT NULL DEFAULT now(),
    description TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,      -- batch timestamp + config name
    batch_id     TEXT NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
    config_name  TEXT NOT NULL,
    config_json  JSONB,                 -- full sweep parameters for this run
    started_at   TIMESTAMP,
    finished_at  TIMESTAMP,
    loaded_at    TIMESTAMP NOT NULL DEFAULT now()
);

-- row_data holds whatever per-question/per-metric columns the underlying
-- eval script produced (these differ between precomputed_deepeval.py,
-- hotpotqa_deepeval.py, etc.) so the loader doesn't need to hardcode them.
-- Query with Postgres JSONB operators, e.g.:
--   SELECT run_id, row_data->>'faithfulness_score' FROM eval_results;
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

-- Indexes for the UI's grouping/filtering queries
CREATE INDEX IF NOT EXISTS idx_runs_batch_id      ON runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_runs_config_name   ON runs(config_name);
CREATE INDEX IF NOT EXISTS idx_results_run_id     ON eval_results(run_id);

-- Question sets and evaluation questions schema
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

CREATE OR REPLACE FUNCTION set_default_question_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.question_id IS NULL OR TRIM(NEW.question_id) = '' THEN
        NEW.question_id := NEW.id::text;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_default_question_id ON questions;

CREATE TRIGGER trg_set_default_question_id
BEFORE INSERT ON questions
FOR EACH ROW EXECUTE FUNCTION set_default_question_id();