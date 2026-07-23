from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepeval_eval.caipe_client import CaipeRagClient
from deepeval_eval.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_RESULTS_DIR,
    ensure_dirs,
    load_dotenv_loose,
)


def run_enterprise_ingest(args: argparse.Namespace) -> None:
    """Ingest EnterpriseRAG-Bench documents and questions into CAIPE datasource."""
    from deepeval_eval.enterprise_dataset import (
        INGESTOR_NAME,
        INGESTOR_TYPE,
        fetch_documents,
        load_questions,
        select_questions,
        to_caipe_payload,
        write_corpus,
        write_questions,
    )

    data_dir = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    cache_dir = getattr(args, "cache_dir", DEFAULT_CACHE_DIR)
    results_dir = getattr(args, "results_dir", DEFAULT_RESULTS_DIR)
    ensure_dirs(data_dir, cache_dir, results_dir)

    questions = load_questions(cache_dir)
    sources = getattr(args, "sources", None) or ["confluence", "jira"]
    num_questions = getattr(args, "num_questions", 10)
    questions_per_category = getattr(args, "questions_per_category", 3)
    selected = select_questions(
        questions, sources, num_questions, questions_per_category
    )
    reference_doc_ids = {doc_id for q in selected for doc_id in q.expected_doc_ids}

    print(f"Selected {len(selected)} questions")
    print(f"Reference doc ids to prioritize: {len(reference_doc_ids)}")

    limit_per_source = getattr(args, "limit_per_source", 1000)
    docs = fetch_documents(sources, limit_per_source, cache_dir, reference_doc_ids)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    covered = [q for q in selected if set(q.expected_doc_ids) <= set(docs_by_id.keys())]
    if covered:
        selected = covered
    print(f"Questions fully covered by ingested docs: {len(covered)}")

    if not getattr(args, "skip_ingest", False):
        rag_url = getattr(args, "rag_url", "http://localhost:9446")
        auth_token = getattr(args, "auth_token", None)
        client = CaipeRagClient(rag_url, auth_token)

        datasource_id = (
            getattr(args, "datasource_id", None) or "enterprise_rag_bench_deepeval"
        )
        datasource_name = (
            getattr(args, "datasource_name", None) or "EnterpriseRAG-Bench DeepEval"
        )

        if getattr(args, "reset", False):
            print(f"Resetting datasource {datasource_id}")
            client.reset_datasource(datasource_id)

        ingestor_id, max_docs = client.register_ingestor(
            INGESTOR_TYPE, INGESTOR_NAME, "EnterpriseRAG-Bench ingestion for DeepEval"
        )
        batch_size = min(getattr(args, "batch_size", 100), max_docs)
        payloads = [to_caipe_payload(doc, datasource_id, ingestor_id) for doc in docs]

        client.upsert_datasource(
            datasource_id,
            datasource_name,
            ingestor_id,
            "EnterpriseRAG-Bench sample for CAIPE DeepEval evaluation",
            INGESTOR_TYPE,
        )
        job_id = client.open_job(
            datasource_id, len(payloads), "EnterpriseRAG-Bench DeepEval ingestion"
        )
        print(f"Ingestion job opened: {job_id}")

        for start in range(0, len(payloads), batch_size):
            batch = payloads[start : start + batch_size]
            client.ingest_batch(batch, ingestor_id, datasource_id, job_id)
            print(f"  ingested {start + len(batch)}/{len(payloads)} documents")

        client.close_job(job_id, "EnterpriseRAG-Bench DeepEval ingestion complete")
        print("Ingestion job completed")

    write_corpus(
        docs,
        data_dir / "enterprise_deepeval_corpus.jsonl",
        data_dir / "enterprise_deepeval_corpus.csv",
    )
    write_questions(
        selected,
        docs_by_id,
        data_dir / "enterprise_deepeval_questions.jsonl",
        data_dir / "enterprise_deepeval_questions.csv",
    )
    print(f"Wrote data files to {data_dir}")


def run_hotpotqa_ingest(args: argparse.Namespace) -> None:
    """Ingest HotpotQA documents and questions into CAIPE datasource."""
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

    data_dir = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    cache_dir = getattr(args, "cache_dir", DEFAULT_CACHE_DIR)
    results_dir = getattr(args, "results_dir", DEFAULT_RESULTS_DIR)
    ensure_dirs(data_dir, cache_dir, results_dir)

    questions_zip = resolve_zip(
        getattr(args, "questions_zip", None), "hotpotqa_full_questions.jsonl.zip"
    )
    documents_zip = resolve_zip(
        getattr(args, "documents_zip", None), "hotpotqa_full_document_pool.jsonl.zip"
    )

    questions = load_questions(questions_zip)
    limit = getattr(args, "limit", 100)
    questions_per_category = getattr(args, "questions_per_category", 50)
    categories = getattr(args, "categories", None)
    selected = select_questions(questions, limit, questions_per_category, categories)
    print(f"Selected {len(selected)} questions")

    pool = load_document_pool(documents_zip)
    distractors_per_question = getattr(args, "distractors_per_question", 8)
    max_docs = getattr(args, "max_docs", None)
    docs = select_documents(selected, pool, distractors_per_question, max_docs)
    docs_by_id = {doc["document_id"]: doc for doc in docs}
    covered = [
        q for q in selected if set(q["expected_doc_ids"]) <= set(docs_by_id.keys())
    ]
    if covered:
        selected = covered

    print(f"Selected {len(docs)} docs")
    print(f"Questions fully covered by ingested docs: {len(covered)}")

    if not getattr(args, "skip_ingest", False):
        rag_url = getattr(args, "rag_url", "http://localhost:9446")
        auth_token = getattr(args, "auth_token", None)
        client = CaipeRagClient(rag_url, auth_token)

        datasource_id = getattr(args, "datasource_id", None) or "hotpotqa_deepeval"
        datasource_name = getattr(args, "datasource_name", None) or "HotpotQA DeepEval"

        if getattr(args, "reset", False):
            print(f"Resetting datasource {datasource_id}")
            client.reset_datasource(datasource_id)

        ingestor_id, max_docs_per_batch = client.register_ingestor(
            INGESTOR_TYPE, INGESTOR_NAME, "HotpotQA ingestion for CAIPE DeepEval"
        )
        batch_size = min(getattr(args, "batch_size", 50), max_docs_per_batch)
        payloads = [to_caipe_payload(doc, datasource_id, ingestor_id) for doc in docs]

        client.upsert_datasource(
            datasource_id,
            datasource_name,
            ingestor_id,
            "HotpotQA sample for CAIPE DeepEval evaluation",
            INGESTOR_TYPE,
        )
        job_id = client.open_job(
            datasource_id, len(payloads), "HotpotQA DeepEval ingestion"
        )
        print(f"Ingestion job opened: {job_id}")

        for start in range(0, len(payloads), batch_size):
            batch = payloads[start : start + batch_size]
            client.ingest_batch(batch, ingestor_id, datasource_id, job_id)
            print(f"  ingested {start + len(batch)}/{len(payloads)} documents")

        client.close_job(job_id, "HotpotQA DeepEval ingestion complete")
        print("Ingestion job completed")

    write_corpus(
        docs,
        data_dir / "hotpotqa_deepeval_corpus.jsonl",
        data_dir / "hotpotqa_deepeval_corpus.csv",
    )
    write_questions(
        selected,
        docs_by_id,
        data_dir / "hotpotqa_deepeval_questions.jsonl",
        data_dir / "hotpotqa_deepeval_questions.csv",
    )
    print(f"Wrote data files to {data_dir}")


def run_ingest(args: argparse.Namespace) -> None:
    """Dispatch to dataset-specific ingestion handlers."""
    ds = (
        getattr(args, "dataset_name", None) or getattr(args, "benchmark", "enterprise")
    ).lower()
    if ds in ("enterprise", "enterprise_rag_bench"):
        run_enterprise_ingest(args)
    elif ds == "hotpotqa":
        run_hotpotqa_ingest(args)
    else:
        raise ValueError(f"Unsupported dataset for ingestion: {ds}")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for standalone dataset ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Dataset ingestion CLI for CAIPE evaluation benchmarks",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    parser.add_argument(
        "--dataset-name",
        "--dataset",
        "--benchmark",
        dest="dataset_name",
        default="enterprise",
        help="Dataset name to ingest (default: enterprise)",
    )
    parser.add_argument("--rag-url", default="http://localhost:9446")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--datasource-id", default=None)
    parser.add_argument("--datasource-name", default=None)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--limit-per-source", type=int, default=1000)
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--questions-per-category", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--distractors-per-question", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--questions-zip", type=Path, default=None)
    parser.add_argument("--documents-zip", type=Path, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(getattr(args, "env_file", DEFAULT_ENV_FILE))
    run_ingest(args)


if __name__ == "__main__":
    main()
