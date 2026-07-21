from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepeval_eval import deepeval_evaluator
from deepeval_eval.caipe_client import CaipeRagClient
from deepeval_eval.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GATE_CONFIG,
    DEFAULT_RESULTS_DIR,
    ensure_dirs,
    load_dotenv_loose,
)
from deepeval_eval.hotpotqa_dataset import (
    INGESTOR_NAME,
    INGESTOR_TYPE,
    load_document_pool,
    load_questions,
    resolve_zip,
    select_documents,
    select_questions,
    to_caipe_payload,
    write_corpus,
    write_questions,
)


def run_ingest(args: argparse.Namespace) -> None:
    ensure_dirs(args.data_dir, args.cache_dir, args.results_dir)

    questions_zip = resolve_zip(args.questions_zip, "hotpotqa_full_questions.jsonl.zip")
    documents_zip = resolve_zip(
        args.documents_zip, "hotpotqa_full_document_pool.jsonl.zip"
    )

    questions = load_questions(questions_zip)
    selected = select_questions(
        questions, args.limit, args.questions_per_category, args.categories
    )
    print(f"Selected {len(selected)} questions")

    pool = load_document_pool(documents_zip)
    # Gold paragraphs are loaded first; distractors make retrieval less trivial
    # while keeping the local sample small enough for a laptop run.
    docs = select_documents(
        selected, pool, args.distractors_per_question, args.max_docs
    )
    docs_by_id = {doc["document_id"]: doc for doc in docs}
    covered = [
        q for q in selected if set(q["expected_doc_ids"]) <= set(docs_by_id.keys())
    ]
    if covered:
        selected = covered

    print(f"Selected {len(docs)} docs")
    print(f"Questions fully covered by ingested docs: {len(covered)}")

    if not args.skip_ingest:
        client = CaipeRagClient(args.rag_url, args.auth_token)
        if args.reset:
            print(f"Resetting datasource {args.datasource_id}")
            client.reset_datasource(args.datasource_id)

        ingestor_id, max_docs_per_batch = client.register_ingestor(
            INGESTOR_TYPE,
            INGESTOR_NAME,
            "HotpotQA ingestion for CAIPE DeepEval",
        )
        batch_size = min(args.batch_size, max_docs_per_batch)
        payloads = [
            to_caipe_payload(doc, args.datasource_id, ingestor_id) for doc in docs
        ]

        client.upsert_datasource(
            args.datasource_id,
            args.datasource_name,
            ingestor_id,
            "HotpotQA sample for CAIPE DeepEval evaluation",
            INGESTOR_TYPE,
        )
        job_id = client.open_job(
            args.datasource_id, len(payloads), "HotpotQA DeepEval ingestion"
        )
        print(f"Ingestion job opened: {job_id}")

        for start in range(0, len(payloads), batch_size):
            batch = payloads[start : start + batch_size]
            client.ingest_batch(batch, ingestor_id, args.datasource_id, job_id)
            print(f"  ingested {start + len(batch)}/{len(payloads)} documents")

        client.close_job(job_id, "HotpotQA DeepEval ingestion complete")
        print("Ingestion job completed")

    write_corpus(
        docs,
        args.data_dir / "hotpotqa_deepeval_corpus.jsonl",
        args.data_dir / "hotpotqa_deepeval_corpus.csv",
    )
    write_questions(
        selected,
        docs_by_id,
        args.data_dir / "hotpotqa_deepeval_questions.jsonl",
        args.data_dir / "hotpotqa_deepeval_questions.csv",
    )
    print(f"Wrote data files to {args.data_dir}")


def run_eval(args: argparse.Namespace) -> None:
    warnings.warn(
        "hotpotqa_deepeval.py eval is deprecated. Please use 'deepeval_evaluator.py eval --benchmark hotpotqa' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    args.benchmark = "hotpotqa"
    deepeval_evaluator._run_eval(args)


def write_results(
    results_dir: Path,
    results: list[dict[str, Any]],
    evaluation_time: float,
    config_args: dict[str, Any],
    datasource: str = "hotpotqa",
) -> None:
    config = dict(config_args)
    config["datasource"] = datasource
    deepeval_evaluator._write_results(
        results_dir, f"hotpotqa_deepeval_{datasource}", results, evaluation_time, config
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HotpotQA DeepEval pipeline for CAIPE")
    parser.add_argument("--rag-url", default="http://localhost:9446")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument(
        "--questions-zip",
        type=Path,
        default=DEFAULT_CACHE_DIR / "hotpotqa_full_questions.jsonl.zip",
    )
    ingest.add_argument(
        "--documents-zip",
        type=Path,
        default=DEFAULT_CACHE_DIR / "hotpotqa_full_document_pool.jsonl.zip",
    )
    ingest.add_argument("--datasource-id", default="hotpotqa_deepeval")
    ingest.add_argument("--datasource-name", default="HotpotQA DeepEval")
    ingest.add_argument("--limit", type=int, default=100)
    ingest.add_argument("--questions-per-category", type=int, default=50)
    ingest.add_argument(
        "--categories", nargs="+", default=None, choices=["bridge", "comparison"]
    )
    ingest.add_argument("--distractors-per-question", type=int, default=8)
    ingest.add_argument("--max-docs", type=int, default=None)
    ingest.add_argument("--batch-size", type=int, default=50)
    ingest.add_argument("--reset", action="store_true")
    ingest.add_argument("--skip-ingest", action="store_true")
    ingest.set_defaults(func=run_ingest)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--datasource-id", default="hotpotqa_deepeval")
    eval_parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_DATA_DIR / "hotpotqa_deepeval_questions.jsonl",
    )
    eval_parser.add_argument("--max-items", type=int, default=None)
    eval_parser.add_argument("--limit-per-category", type=int, default=None)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--max-context-chars", type=int, default=12000)
    eval_parser.add_argument("--llm-base-url", default=None)
    eval_parser.add_argument("--llm-api-key", default=None)
    eval_parser.add_argument("--llm-model", default=None)
    eval_parser.add_argument(
        "--agentic",
        action="store_true",
        help="Route queries through caipe-supervisor A2A endpoint",
    )
    eval_parser.add_argument(
        "--supervisor-url",
        default="http://localhost:8000",
        help="CAIPE supervisor URL for agentic eval",
    )
    eval_parser.add_argument(
        "--gate",
        action="store_true",
        help="Apply the quality gate after evaluation and exit non-zero if it fails.",
    )
    eval_parser.add_argument(
        "--gate-config",
        type=Path,
        default=DEFAULT_GATE_CONFIG,
        help="Path to the gate threshold config (YAML/JSON).",
    )
    eval_parser.set_defaults(func=run_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
