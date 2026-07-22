from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepeval_eval import deepeval_evaluator
from deepeval_eval.caipe_client import CaipeRagClient, extract_contexts_and_sources
from deepeval_eval.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GATE_CONFIG,
    DEFAULT_RESULTS_DIR,
    load_dotenv_loose,
)
from deepeval_eval.llm_client import (
    OpenAICompatibleClient,
    make_generation_prompt,
    make_short_answer_prompt,
)

# Ensure your custom metrics are imported properly from metrics.py


def build_gold_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected_doc_ids = list(row.get("expected_doc_ids") or [])
    source_types = list(row.get("source_types") or [])
    source_type = source_types[0] if source_types else (row.get("dataset_name") or row.get("benchmark"))
    return [
        {
            "document_id": doc_id,
            "title": None,
            "source_type": source_type,
            "score": 1.0,
            "retrieval_mode": "ground_truth",
        }
        for doc_id in expected_doc_ids
    ]


def context_from_row(row: dict[str, Any], max_context_chars: int) -> list[str]:
    contexts = row.get("context") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    return [str(text)[:max_context_chars] for text in contexts if str(text).strip()]


def retrieve_live_context_and_sources(
    caipe_client: CaipeRagClient,
    datasource_id: str | None,
    question: str,
    reference: str,
    top_k: int = 3,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Retrieves context and gold sources live using a combined Oracle reference query."""
    reference_query = f"{question} {reference}".strip()

    # Run query against CAIPE RAG endpoint
    results = caipe_client.query(
        reference_query, datasource_id=datasource_id, limit=top_k
    )

    # Standard helper parsing directly from caipe_client.py
    return extract_contexts_and_sources(results)


def make_answer(
    args: argparse.Namespace,
    llm_client: OpenAICompatibleClient,
    question: str,
    reference: str,
    contexts: list[str],
) -> str:
    if args.answer_mode == "reference":
        return reference
    ds_name = getattr(args, "dataset_name", None) or getattr(args, "benchmark", "hotpotqa")
    prompt_style = getattr(args, "prompt_style", None)
    if prompt_style == "short" or (prompt_style is None and ds_name == "hotpotqa"):
        return str(llm_client.generate(make_short_answer_prompt(question, contexts)))
    return str(llm_client.generate(make_generation_prompt(question, contexts)))


def run_eval(args: argparse.Namespace) -> None:
    warnings.warn(
        "precomputed_deepeval.py eval is deprecated. Please use 'deepeval_evaluator.py eval --precompute' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    args.precompute = True
    deepeval_evaluator._run_eval(args)


def write_results(
    results_dir: Path,
    dataset_name: str = "enterprise",
    answer_mode: str = "reference",
    results: list[dict[str, Any]] | None = None,
    evaluation_time: float = 0.0,
    config_args: dict[str, Any] | None = None,
) -> None:
    if results is None:
        results = []
    if config_args is None:
        config_args = {}
    config = dict(config_args)
    config["datasource"] = f"{dataset_name}_precomputed"
    prefix = f"precomputed_deepeval_{dataset_name}_{answer_mode}"
    deepeval_evaluator._write_results(
        results_dir, prefix, results, evaluation_time, config
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepEval against benchmark ground-truth contexts and reference answers",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--dataset-name",
        "--dataset",
        "--benchmark",
        dest="dataset_name",
        default="hotpotqa",
        help="Dataset name to evaluate against",
    )
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument(
        "--answer-mode",
        choices=["reference", "generate"],
        default="reference",
        help="reference uses the benchmark answer as actual_output; generate answers from gold context using the LLM",
    )
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--datasource-id", type=str, default=None, help="The target CAIPE datasource"
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Apply the quality gate after evaluation and exit non-zero if it fails.",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=DEFAULT_GATE_CONFIG,
        help="Path to the gate threshold config (YAML/JSON).",
    )
    parser.set_defaults(func=run_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
