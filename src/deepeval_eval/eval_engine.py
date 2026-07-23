from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepeval_eval.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_GATE_CONFIG,
    DEFAULT_RESULTS_DIR,
    ensure_dirs,
    load_dotenv_loose,
    resolve_llm_settings,
)
from deepeval_eval.data_loader import BaseDataLoader, FileDataLoader
from deepeval_eval.io_utils import sanitize_path
from deepeval_eval.llm_client import DeepEvalJudge, OpenAICompatibleClient
from deepeval_eval.metrics import build_metrics, doc_id_scores
from deepeval_eval.prompt_style import DEFAULT_PROMPT_STYLE
from deepeval_eval.sinks import (
    DatabaseResultSink,
    FileResultSink,
    ResultSink,
    write_evaluation_results,
)

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Raised when evaluation results fail quality gate thresholds."""

    pass


@dataclass
class EvalConfig:
    dataset_name: str = "enterprise"
    answer_mode: str = "generate"
    datasource_id: str | None = None
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    questions_file: Path | None = None
    prompt_style: str | None = DEFAULT_PROMPT_STYLE
    prompt_config: Path | None = None
    combine_with_level: bool | None = None
    max_items: int | None = None
    limit_per_category: int | None = None
    top_k: int = 3
    max_context_chars: int = 12000
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    agentic: bool = False
    supervisor_url: str | None = None
    fail_on_error: bool = False
    oracle_retrieval: bool = False
    gate: bool = False
    gate_config: Path = field(default_factory=lambda: DEFAULT_GATE_CONFIG)
    env_file: Path = field(default_factory=lambda: DEFAULT_ENV_FILE)
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)
    question_ids: str | None = None
    question_indices: str | None = None
    batch_id: str | None = None
    run_id: str | None = None
    oracle_testing: bool = False
    save_to_db: bool = False
    db_connection_string: str | None = None

    def __post_init__(self) -> None:
        if self.oracle_testing:
            self.oracle_retrieval = True
            self.answer_mode = "ground_truth"

        if self.answer_mode not in ("generate", "ground_truth"):
            raise ValueError(
                f"Invalid answer_mode: '{self.answer_mode}'. Must be 'generate' or 'ground_truth'"
            )

    def to_config_args(self) -> dict[str, Any]:
        """Convert dataclass fields to serializable configuration dictionary."""
        config = {}
        for k, v in self.__dict__.items():
            if (
                v is None
                or k in ("llm_api_key", "db_connection_string")
                or k.startswith("_")
            ):
                continue
            if isinstance(v, Path):
                config[k] = str(v)
            else:
                config[k] = v
        return config


def _build_rag_client(config: EvalConfig, env_values: dict[str, Any]) -> Any:
    """Factory function to build the appropriate RAG client for the evaluation run."""
    from deepeval_eval.caipe_client import build_caipe_client

    supervisor_url = (
        getattr(config, "supervisor_url", None)
        or env_values.get("CAIPE_SUPERVISOR_URL")
        or env_values.get("SUPERVISOR_URL")
        or __import__("os").getenv("CAIPE_SUPERVISOR_URL")
        or "http://localhost:8000"
    )

    datasource_id = getattr(config, "datasource_id", None)
    if datasource_id:
        __import__("os").environ["CAIPE_DATASOURCE_ID"] = datasource_id

    if getattr(config, "oracle_retrieval", False):
        from deepeval_eval.oracle_client import OracleRagClient

        caipe_client = build_caipe_client(env_values)
        return OracleRagClient(caipe_client)
    elif getattr(config, "agentic", False):
        from deepeval_eval.rag_client import AgenticRagAdapter

        return AgenticRagAdapter(
            supervisor_url=supervisor_url,
            results_dir=getattr(config, "results_dir", None),
            fail_on_error=getattr(config, "fail_on_error", False),
            datasource_id=datasource_id,
        )
    else:
        return build_caipe_client(env_values)


def run_evaluation(
    config: EvalConfig,
    data_loader: BaseDataLoader | None = None,
    rag_client: Any | None = None,
    metrics: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute evaluation run according to EvalConfig settings."""
    ensure_dirs(config.results_dir)
    if config.prompt_config:
        from deepeval_eval.prompt_style import load_prompt_styles_from_config

        load_prompt_styles_from_config(config.prompt_config)
    env_values = load_dotenv_loose(config.env_file)
    base_url, api_key, model = resolve_llm_settings(
        config.env_file, config.llm_base_url, config.llm_api_key, config.llm_model
    )

    llm_client = OpenAICompatibleClient(model=model, api_key=api_key, base_url=base_url)
    if metrics is None:
        judge = DeepEvalJudge("openai-compatible", model, llm_client).model
        metrics = build_metrics(judge)

    from deepeval.test_case import LLMTestCase

    datasource_id = config.datasource_id or env_values.get("CAIPE_DATASOURCE_ID")
    dataset_name = config.dataset_name

    if data_loader is None:
        data_loader = FileDataLoader(
            questions_file=config.questions_file,
            dataset_name=dataset_name,
            data_dir=config.data_dir,
        )

    combine_with_level = (
        config.combine_with_level
        if config.combine_with_level is not None
        else (dataset_name == "hotpotqa")
    )

    rows = data_loader.load(
        max_items=config.max_items,
        limit_per_category=config.limit_per_category,
        combine_with_level=combine_with_level,
    )

    if config.question_ids:
        target_ids = {qid.strip() for qid in config.question_ids.split(",")}
        rows = [row for row in rows if str(row.get("question_id")) in target_ids]

    if config.question_indices:
        indices = set()
        for part in config.question_indices.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start_str, end_str = part.split("-", 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    for i in range(start, end + 1):
                        if 1 <= i <= len(rows):
                            indices.add(i)
                except ValueError:
                    pass
            else:
                try:
                    i = int(part)
                    if 1 <= i <= len(rows):
                        indices.add(i)
                except ValueError:
                    pass
        rows = [rows[i - 1] for i in sorted(indices) if 1 <= i <= len(rows)]

    if rag_client is None:
        rag_client = _build_rag_client(config, env_values)

    results: list[dict[str, Any]] = []
    start_eval_time = time.time()

    for idx, row in enumerate(rows, start=1):
        question = row["user_input"]
        reference = row.get("reference") or ""
        logger.info(f"Evaluating {idx}/{len(rows)}: {question[:90]}")

        llm_client.reset_tokens()

        query_res = rag_client.query(
            question=question,
            reference=reference,
            datasource_id=datasource_id,
            top_k=config.top_k,
            answer_mode=config.answer_mode,
            dataset_name=dataset_name,
            prompt_style=config.prompt_style,
            llm_client=llm_client,
            max_context_chars=config.max_context_chars,
        )

        answer = query_res.answer
        trimmed_contexts = query_res.contexts
        sources = query_res.sources
        current_retrieved_ids = query_res.retrieved_doc_ids or row.get(
            "retrieved_doc_ids", []
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference,
            retrieval_context=trimmed_contexts,
            context=row.get("context") or [],
            metadata={
                "retrieved_doc_ids": current_retrieved_ids,
                "expected_doc_ids": row.get("expected_doc_ids", []),
            },
        )

        metric_results: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            metric_name = metric.__class__.__name__
            try:
                metric.measure(test_case)
                reason = getattr(metric, "reason", None)
                if reason is None:
                    get_reason_fn = getattr(metric, "get_reason", None)
                    if callable(get_reason_fn):
                        try:
                            reason = get_reason_fn()
                        except Exception:
                            reason = None
                metric_results[metric_name] = {
                    "score": metric.score,
                    "success": metric.is_successful(),
                    "reason": reason,
                }
            except Exception as exc:
                logger.warning(
                    f"Metric '{metric_name}' failed for question '{question[:40]}...': {exc}"
                )
                if config.fail_on_error:
                    raise exc
                metric_results[metric_name] = {
                    "score": None,
                    "success": False,
                    "reason": f"metric failed: {exc}",
                }

        recall_score = metric_results.get("RetrievalRecallMetric", {}).get("score")
        precision_score = metric_results.get("RetrievalPrecisionMetric", {}).get(
            "score"
        )
        if recall_score is None or precision_score is None:
            fallback_recall, fallback_precision = doc_id_scores(
                sources, list(row.get("expected_doc_ids") or [])
            )
            doc_recall = recall_score if recall_score is not None else fallback_recall
            doc_precision = (
                precision_score if precision_score is not None else fallback_precision
            )
        else:
            doc_recall = recall_score
            doc_precision = precision_score

        results.append(
            {
                "question_id": row.get("question_id"),
                "dataset_name": dataset_name,
                "benchmark": dataset_name,
                "category": row.get("category"),
                "level": row.get("level"),
                "answer_mode": config.answer_mode,
                "question": question,
                "user_input": question,
                "reference": reference,
                "actual_output": answer,
                "retrieved_contexts": trimmed_contexts,
                "retrieved_doc_ids": current_retrieved_ids,
                "expected_doc_ids": row.get("expected_doc_ids") or [],
                "doc_id_recall": doc_recall,
                "doc_id_precision": doc_precision,
                "metrics": metric_results,
                "input_tokens": query_res.input_tokens,
                "output_tokens": query_res.output_tokens,
                "total_tokens": query_res.total_tokens,
                "evaluator_input_tokens": llm_client.input_tokens,
                "evaluator_output_tokens": llm_client.output_tokens,
                "evaluator_total_tokens": llm_client.total_tokens,
                "latency": query_res.latency_sec,
                "log_file": sanitize_path(query_res.log_file),
            }
        )

    eval_time = time.time() - start_eval_time
    config_args = config.to_config_args()
    config_args["datasource"] = dataset_name

    sinks: list[ResultSink] = [FileResultSink()]
    if config.save_to_db:
        sinks.append(DatabaseResultSink(connection_string=config.db_connection_string))

    write_evaluation_results(
        results_dir=config.results_dir,
        prefix=f"deepeval_{dataset_name}",
        results=results,
        evaluation_time=eval_time,
        config_args=config_args,
        sinks=sinks,
    )

    if config.gate:
        from deepeval_eval.gate import run_gate_on_results

        passed = run_gate_on_results(results, config.gate_config, config.results_dir)
        if not passed:
            raise QualityGateError("Evaluation results failed quality gate thresholds.")

    return results
