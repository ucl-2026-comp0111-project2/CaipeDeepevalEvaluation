from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepeval_eval.deepeval_evaluator import (
    _build_config_args,
    _build_rag_client,
    _run_eval,
)
from deepeval_eval.deepeval_evaluator import (
    build_parser as build_evaluator_parser,
)
from deepeval_eval.rag_client import RagQueryResult


def test_build_config_args_positive(tmp_path: Path) -> None:
    args = argparse.Namespace(
        env_file=tmp_path / ".env",
        llm_api_key="secret",
        top_k=3,
        agentic=True,
        _private="hidden",
    )
    res = _build_config_args(args)
    assert "llm_api_key" not in res
    assert "_private" not in res
    assert res["top_k"] == 3
    assert res["agentic"] is True
    assert res["env_file"] == str(tmp_path / ".env")


def test_build_config_args_negative() -> None:
    args = argparse.Namespace(llm_api_key=None, auth_token=None)
    res = _build_config_args(args)
    assert res == {}


def test_build_rag_client_positive(tmp_path: Path) -> None:
    env_values = {"CAIPE_BASE_URL": "http://localhost:8080"}

    # Oracle RAG client
    args_oracle = argparse.Namespace(oracle_retrieval=True, agentic=False)
    client1 = _build_rag_client(args_oracle, env_values)
    assert client1.__class__.__name__ == "OracleRagClient"

    # Agentic RAG client
    args_agentic = argparse.Namespace(
        precompute=False,
        agentic=True,
        supervisor_url="http://localhost:8000",
        results_dir=tmp_path,
        fail_on_error=False,
    )
    with patch("deepeval_eval.agentic_rag.AgenticRetriever"):
        client2 = _build_rag_client(args_agentic, env_values)
        assert client2.__class__.__name__ == "AgenticRagAdapter"

    # Standard CAIPE RAG client
    args_std = argparse.Namespace(precompute=False, agentic=False)
    client3 = _build_rag_client(args_std, env_values)
    assert client3.__class__.__name__ == "CaipeRagClient"


def test_build_parsers_positive() -> None:
    p1 = build_evaluator_parser()
    assert p1 is not None


def test_run_eval_positive(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_ENDPOINT=http://localhost\nOPENAI_API_KEY=k\nOPENAI_MODEL_NAME=m\n"
    )

    questions_file = tmp_path / "questions.jsonl"
    questions_file.write_text(
        '{"question_id": "q1", "user_input": "What is X?", "reference": "X is Y", "expected_doc_ids": ["doc1"]}\n'
    )

    args = argparse.Namespace(
        results_dir=tmp_path / "results",
        env_file=env_file,
        llm_base_url="http://localhost",
        llm_api_key="k",
        llm_model="m",
        datasource_id="ds1",
        dataset_name="enterprise",
        questions_file=questions_file,
        max_items=1,
        limit_per_category=None,
        question_ids=None,
        question_indices=None,
        top_k=3,
        answer_mode="ground_truth",
        max_context_chars=1000,
        precompute=False,
        agentic=False,
    )

    mock_rag = MagicMock()
    mock_rag.query.return_value = RagQueryResult(
        answer="X is Y",
        contexts=["Context text"],
        sources=[{"document_id": "doc1"}],
        retrieved_doc_ids=["doc1"],
        latency_sec=0.1,
    )

    with (
        patch(
            "deepeval_eval.deepeval_evaluator._build_rag_client", return_value=mock_rag
        ),
        patch("deepeval_eval.deepeval_evaluator.build_metrics", return_value=[]),
    ):
        _run_eval(args)

    results_dir = tmp_path / "results"
    assert results_dir.exists()


def test_deepeval_evaluator_prompt_config_cli(tmp_path: Path) -> None:
    parser = build_evaluator_parser()
    prompt_cfg = tmp_path / "custom_prompts.yaml"
    prompt_cfg.write_text(
        "prompt_styles:\n  my_style: 'Q: {question} C: {context}'\n", encoding="utf-8"
    )

    args = parser.parse_args(
        [
            "eval",
            "--prompt-style",
            "my_style",
            "--prompt-config",
            str(prompt_cfg),
        ]
    )
    assert args.prompt_style == "my_style"
    assert args.prompt_config == prompt_cfg
