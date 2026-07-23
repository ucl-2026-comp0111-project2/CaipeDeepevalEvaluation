from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepeval_eval.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GATE_CONFIG,
    DEFAULT_RESULTS_DIR,
    load_dotenv_loose,
)
from deepeval_eval.eval_engine import (
    EvalConfig,
    QualityGateError,
    _build_rag_client,
    run_evaluation,
)
from deepeval_eval.metrics import build_metrics as build_metrics
from deepeval_eval.sinks import write_evaluation_results


def _environ_get(key: str, default: str | None = None) -> str | None:
    """Read from os.environ with a fallback default."""
    return os.environ.get(key) or default


def _write_results(
    results_dir: Path,
    prefix: str,
    results: list[dict[str, Any]],
    evaluation_time: float,
    config_args: dict[str, Any],
) -> None:
    """Delegates to write_evaluation_results in sinks for dynamic metric aggregation."""
    write_evaluation_results(
        results_dir=results_dir,
        prefix=prefix,
        results=results,
        evaluation_time=evaluation_time,
        config_args=config_args,
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add common eval arguments to an existing subparser."""
    parser.add_argument(
        "--datasource-id", default=None, help="The target CAIPE datasource"
    )
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument(
        "--prompt-style",
        default=None,
        help="Prompt style for answer generation (e.g. 'generation', 'short', or custom style name)",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=None,
        help="Path to custom prompt style configuration file (JSON/YAML)",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of documents to retrieve"
    )
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Route queries through caipe-supervisor A2A endpoint",
    )
    parser.add_argument(
        "--agent-id",
        default=None,
        help="CAIPE agent ID for agentic eval (defaults to CAIPE_AGENT_ID env var or hello-world)",
    )
    parser.add_argument(
        "--supervisor-url",
        default="http://localhost:8000",
        help="CAIPE supervisor URL for agentic eval",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Fail loudly if a query evaluation fails after retries",
    )
    parser.add_argument(
        "--oracle-retrieval",
        action="store_true",
        help="Enable oracle retrieval (querying CAIPE search using question + reference)",
    )
    parser.add_argument(
        "--oracle-testing",
        action="store_true",
        help="Shortcut to enable oracle_retrieval and ground_truth answer mode",
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
        help="Path to quality gate threshold YAML config",
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Directly persist evaluation results to PostgreSQL database",
    )


def _build_config_args(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize args into a JSON-serializable dict (hiding secrets)."""
    config = {}
    for k, v in vars(args).items():
        if (
            v is None
            or k in ("llm_api_key", "auth_token")
            or callable(v)
            or k.startswith("_")
        ):
            continue
        if isinstance(v, Path):
            config[k] = str(v)
        elif isinstance(v, (str, int, float, bool, list, dict)):
            config[k] = v
        else:
            config[k] = str(v)
    return config


def _run_eval(args: argparse.Namespace) -> None:
    """CLI handler that builds an EvalConfig and dispatches to eval_engine.run_evaluation."""
    ds_name = (
        getattr(args, "dataset_name", None)
        or getattr(args, "dataset", None)
        or getattr(args, "benchmark", "enterprise")
    )
    config = EvalConfig(
        dataset_name=ds_name,
        answer_mode=getattr(args, "answer_mode", "generate"),
        datasource_id=getattr(args, "datasource_id", None),
        data_dir=getattr(args, "data_dir", DEFAULT_DATA_DIR),
        questions_file=getattr(args, "questions_file", None),
        prompt_style=getattr(args, "prompt_style", None),
        prompt_config=getattr(args, "prompt_config", None),
        max_items=getattr(args, "max_items", None),
        limit_per_category=getattr(args, "limit_per_category", None),
        top_k=getattr(args, "top_k", 3),
        max_context_chars=getattr(args, "max_context_chars", 12000),
        llm_base_url=getattr(args, "llm_base_url", None),
        llm_api_key=getattr(args, "llm_api_key", None),
        llm_model=getattr(args, "llm_model", None),
        agentic=getattr(args, "agentic", False),
        agent_id=getattr(args, "agent_id", None),
        supervisor_url=getattr(args, "supervisor_url", "http://localhost:8000"),
        fail_on_error=getattr(args, "fail_on_error", False),
        oracle_retrieval=getattr(args, "oracle_retrieval", False),
        oracle_testing=getattr(args, "oracle_testing", False),
        gate=getattr(args, "gate", False),
        gate_config=getattr(args, "gate_config", DEFAULT_GATE_CONFIG),
        env_file=getattr(args, "env_file", DEFAULT_ENV_FILE),
        results_dir=getattr(args, "results_dir", DEFAULT_RESULTS_DIR),
        question_ids=getattr(args, "question_ids", None),
        question_indices=getattr(args, "question_indices", None),
        save_to_db=getattr(args, "save_to_db", False),
    )
    env_values = load_dotenv_loose(config.env_file)
    rag_client = _build_rag_client(config, env_values)
    try:
        run_evaluation(config, rag_client=rag_client)
    except QualityGateError as err:
        import sys

        sys.stderr.write(f"Quality gate error: {err}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def _eval_subcommand(args: argparse.Namespace) -> None:
    """Dispatch to the unified evaluation loop."""
    _run_eval(args)


# ---------------------------------------------------------------------------
# Build parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepEval evaluation pipeline supporting arbitrary datasets",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- eval subcommand ----
    eval_parser = subparsers.add_parser("eval", help="Run DeepEval evaluation")
    eval_parser.add_argument(
        "--dataset-name",
        "--dataset",
        "--benchmark",
        dest="dataset_name",
        default="enterprise",
        help="Dataset name to evaluate against (default: enterprise)",
    )
    eval_parser.add_argument(
        "--answer-mode",
        choices=["generate", "ground_truth"],
        default="generate",
        help="ground_truth uses benchmark ground-truth answer; generate synthesizes answers via LLM (default: generate)",
    )
    _add_eval_args(eval_parser)
    eval_parser.set_defaults(func=_eval_subcommand)

    return parser


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_loose(getattr(args, "env_file", DEFAULT_ENV_FILE))
    args.func(args)


if __name__ == "__main__":
    main()
