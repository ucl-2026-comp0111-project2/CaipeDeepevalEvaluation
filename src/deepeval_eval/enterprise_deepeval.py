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
    DEFAULT_RESULTS_DIR,
    ensure_dirs,
    load_dotenv_loose,
)
from deepeval_eval.enterprise_dataset import (
    INGESTOR_NAME,
    INGESTOR_TYPE,
    SOURCE_SLICE_COUNTS,
    fetch_documents,
    load_questions,
    select_questions,
    to_caipe_payload,
    write_corpus,
    write_questions,
)


def run_ingest(args: argparse.Namespace) -> None:
    ensure_dirs(args.data_dir, args.cache_dir, args.results_dir)

    questions = load_questions(args.cache_dir)
    selected = select_questions(
        questions, args.sources, args.num_questions, args.questions_per_category
    )
    reference_doc_ids = {
        doc_id for question in selected for doc_id in question.expected_doc_ids
    }

    print(f"Selected {len(selected)} questions")
    print(f"Reference doc ids to prioritize: {len(reference_doc_ids)}")

    # Small local runs are only useful if the gold documents are present, so the
    # dataset loader puts expected document IDs before random same-source docs.
    docs = fetch_documents(
        args.sources, args.limit_per_source, args.cache_dir, reference_doc_ids
    )
    docs_by_id = {doc.doc_id: doc for doc in docs}
    covered = [q for q in selected if set(q.expected_doc_ids) <= set(docs_by_id.keys())]
    if covered:
        selected = covered
    print(f"Questions fully covered by ingested docs: {len(covered)}")

    if not args.skip_ingest:
        client = CaipeRagClient(args.rag_url, args.auth_token)
        if args.reset:
            print(f"Resetting datasource {args.datasource_id}")
            client.reset_datasource(args.datasource_id)

        ingestor_id, max_docs = client.register_ingestor(
            INGESTOR_TYPE,
            INGESTOR_NAME,
            "EnterpriseRAG-Bench ingestion for DeepEval",
        )
        batch_size = min(args.batch_size, max_docs)
        payloads = [
            to_caipe_payload(doc, args.datasource_id, ingestor_id) for doc in docs
        ]

        client.upsert_datasource(
            args.datasource_id,
            args.datasource_name,
            ingestor_id,
            "EnterpriseRAG-Bench sample for CAIPE DeepEval evaluation",
            INGESTOR_TYPE,
        )
        job_id = client.open_job(
            args.datasource_id, len(payloads), "EnterpriseRAG-Bench DeepEval ingestion"
        )
        print(f"Ingestion job opened: {job_id}")

        for start in range(0, len(payloads), batch_size):
            batch = payloads[start : start + batch_size]
            client.ingest_batch(batch, ingestor_id, args.datasource_id, job_id)
            print(f"  ingested {start + len(batch)}/{len(payloads)} documents")

        client.close_job(job_id, "EnterpriseRAG-Bench DeepEval ingestion complete")
        print("Ingestion job completed")

    write_corpus(
        docs,
        args.data_dir / "enterprise_deepeval_corpus.jsonl",
        args.data_dir / "enterprise_deepeval_corpus.csv",
    )
    write_questions(
        selected,
        docs_by_id,
        args.data_dir / "enterprise_deepeval_questions.jsonl",
        args.data_dir / "enterprise_deepeval_questions.csv",
    )
    print(f"Wrote data files to {args.data_dir}")


def parse_indices(indices_str: str, max_len: int) -> set[int]:
    """Parse string representation of indices (e.g., '1,2,5-8') into a set of 1-based indices."""
    indices = set()
    for part in indices_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                for i in range(start, end + 1):
                    if 1 <= i <= max_len:
                        indices.add(i)
            except ValueError:
                pass
        else:
            try:
                i = int(part)
                if 1 <= i <= max_len:
                    indices.add(i)
            except ValueError:
                pass
    return indices


def run_eval(args: argparse.Namespace) -> None:
    warnings.warn(
        "enterprise_deepeval.py eval is deprecated. Please use 'deepeval_evaluator.py eval --benchmark enterprise' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    args.benchmark = "enterprise"
    deepeval_evaluator._run_eval(args)


def write_results(
    results_dir: Path,
    results: list[dict[str, Any]],
    evaluation_time: float,
    config_args: dict[str, Any],
    datasource: str = "enterprise",
) -> None:
    config = dict(config_args)
    config["datasource"] = datasource
    deepeval_evaluator._write_results(
        results_dir,
        f"enterprise_deepeval_{datasource}",
        results,
        evaluation_time,
        config,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EnterpriseRAG-Bench DeepEval pipeline for CAIPE"
    )
    parser.add_argument("--rag-url", default="http://localhost:9446")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument(
        "--sources",
        nargs="+",
        default=["confluence", "jira"],
        choices=sorted(SOURCE_SLICE_COUNTS),
    )
    ingest.add_argument("--datasource-id", default="enterprise_rag_bench_deepeval")
    ingest.add_argument("--datasource-name", default="EnterpriseRAG-Bench DeepEval")
    ingest.add_argument("--limit-per-source", type=int, default=1000)
    ingest.add_argument("--num-questions", type=int, default=10)
    ingest.add_argument("--questions-per-category", type=int, default=3)
    ingest.add_argument("--batch-size", type=int, default=100)
    ingest.add_argument("--reset", action="store_true")
    ingest.add_argument("--skip-ingest", action="store_true")
    ingest.set_defaults(func=run_ingest)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--datasource-id", default="enterprise_rag_bench_deepeval")
    eval_parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_DATA_DIR / "enterprise_deepeval_questions.jsonl",
    )
    eval_parser.add_argument("--max-items", type=int, default=None)
    eval_parser.add_argument("--limit-per-category", type=int, default=None)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--max-context-chars", type=int, default=16000)
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
        "--question-ids",
        default=None,
        help="Comma-separated list of specific question IDs to run/retry",
    )
    eval_parser.add_argument(
        "--question-indices",
        default=None,
        help="Comma-separated list or range of 1-based question indices to run/retry (e.g. 57, 57-60, 1,3,5)",
    )
    eval_parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Fail loudly and raise an exception if a query evaluation fails after retries",
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
