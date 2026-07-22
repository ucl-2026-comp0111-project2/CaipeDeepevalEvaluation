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
from deepeval_eval.enterprise_deepeval import build_parser as build_enterprise_parser
from deepeval_eval.hotpotqa_deepeval import build_parser as build_hotpotqa_parser
from deepeval_eval.precomputed_deepeval import build_parser as build_precomputed_parser
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

    # Precomputed RAG client
    args_precompute = argparse.Namespace(precompute=True, agentic=False)
    client1 = _build_rag_client(args_precompute, env_values)
    assert client1.__class__.__name__ == "PrecomputedRagClient"

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

    p2 = build_enterprise_parser()
    assert p2 is not None

    p3 = build_hotpotqa_parser()
    assert p3 is not None

    p4 = build_precomputed_parser()
    assert p4 is not None


def test_enterprise_deepeval_cli(tmp_path: Path) -> None:
    from deepeval_eval import enterprise_deepeval

    with patch("sys.argv", ["enterprise_deepeval", "--help"]):
        try:
            enterprise_deepeval.main()
        except SystemExit:
            pass


def test_hotpotqa_deepeval_cli(tmp_path: Path) -> None:
    from deepeval_eval import hotpotqa_deepeval

    with patch("sys.argv", ["hotpotqa_deepeval", "--help"]):
        try:
            hotpotqa_deepeval.main()
        except SystemExit:
            pass


def test_enterprise_deepeval_subcommands(tmp_path: Path, monkeypatch) -> None:
    from deepeval_eval import enterprise_deepeval

    parser = enterprise_deepeval.build_parser()
    monkeypatch.setattr(
        "deepeval_eval.caipe_client.build_caipe_client", lambda env: MagicMock()
    )

    # Test ingest subcommand
    args_ing = parser.parse_args(["--data-dir", str(tmp_path), "ingest"])
    assert args_ing.data_dir == tmp_path

    # Test eval subcommand with mock
    monkeypatch.setattr("deepeval_eval.deepeval_evaluator._run_eval", lambda args: None)
    args_eval = parser.parse_args(
        ["--results-dir", str(tmp_path), "eval", "--top-k", "3"]
    )
    args_eval.func(args_eval)


def test_enterprise_deepeval_full_ingest(tmp_path: Path, monkeypatch) -> None:
    from deepeval_eval import enterprise_deepeval

    parser = enterprise_deepeval.build_parser()
    args = parser.parse_args(
        ["--data-dir", str(tmp_path), "ingest", "--reset", "--batch-size", "1"]
    )

    mock_q = MagicMock(expected_doc_ids=["d1"])
    mock_doc = MagicMock(
        text="Doc text", doc_id="d1", title="Title", source_type="slack"
    )
    monkeypatch.setattr(
        "deepeval_eval.enterprise_dataset.load_questions", lambda c: [mock_q]
    )
    monkeypatch.setattr(
        "deepeval_eval.enterprise_dataset.fetch_documents",
        lambda s, limit, c, r: [mock_doc],
    )

    mock_client = MagicMock()
    mock_client.register_ingestor.return_value = ("ing1", 10)
    mock_client.open_job.return_value = "job1"
    monkeypatch.setattr(
        "deepeval_eval.enterprise_deepeval.CaipeRagClient", lambda *a, **kw: mock_client
    )

    args.func(args)
    assert (tmp_path / "enterprise_deepeval_corpus.jsonl").exists()


def test_hotpotqa_deepeval_full_ingest_mocked(tmp_path: Path, monkeypatch) -> None:
    from deepeval_eval import hotpotqa_deepeval

    parser = hotpotqa_deepeval.build_parser()
    dummy_zip = tmp_path / "dummy.zip"
    dummy_zip.write_text("dummy")

    args = parser.parse_args(
        [
            "--data-dir",
            str(tmp_path),
            "ingest",
            "--reset",
            "--batch-size",
            "1",
            "--questions-zip",
            str(dummy_zip),
            "--documents-zip",
            str(dummy_zip),
        ]
    )

    mock_q = [
        {
            "question_id": "hq1",
            "user_input": "Q",
            "reference": "A",
            "category": "cat",
            "level": "easy",
            "expected_doc_ids": ["d1"],
        }
    ]
    mock_pool = {"d1": {"document_id": "d1", "title": "T", "text": "Content"}}

    monkeypatch.setattr(
        "deepeval_eval.hotpotqa_dataset.read_jsonl_zip", lambda p: mock_q
    )
    monkeypatch.setattr(
        "deepeval_eval.hotpotqa_dataset.load_questions", lambda p: mock_q
    )
    monkeypatch.setattr(
        "deepeval_eval.hotpotqa_dataset.load_document_pool", lambda p: mock_pool
    )
    monkeypatch.setattr(
        "deepeval_eval.hotpotqa_dataset.resolve_zip", lambda p, fallback: dummy_zip
    )
    monkeypatch.setattr(
        "deepeval_eval.hotpotqa_deepeval.select_documents",
        lambda q, p, d, m: [{"document_id": "d1", "title": "T", "text": "Content"}],
    )

    mock_client = MagicMock()
    mock_client.register_ingestor.return_value = ("ing1", 10)
    mock_client.open_job.return_value = "job1"
    monkeypatch.setattr(
        "deepeval_eval.hotpotqa_deepeval.CaipeRagClient", lambda *a, **kw: mock_client
    )

    args.func(args)
    mock_client.ingest_batch.assert_called_once()


def test_precomputed_deepeval_make_answer_and_helpers(
    tmp_path: Path, monkeypatch
) -> None:
    from deepeval_eval.precomputed_deepeval import (
        build_gold_sources,
        context_from_row,
        main,
        make_answer,
        retrieve_live_context_and_sources,
        run_eval,
        write_results,
    )

    row = {"expected_doc_ids": ["d1"], "source_types": ["slack"]}
    sources = build_gold_sources(row)
    assert sources[0]["document_id"] == "d1"

    row_ctx = {"context": ["Context string"]}
    assert context_from_row(row_ctx, 100) == ["Context string"]

    row_str_ctx = {"context": "Single string context"}
    assert context_from_row(row_str_ctx, 100) == ["Single string context"]

    mock_client = MagicMock()
    mock_client.query.return_value = [
        {"document": {"page_content": "Ctx text"}, "metadata": {"document_id": "d1"}}
    ]
    c_list, s_list = retrieve_live_context_and_sources(mock_client, "ds1", "Q?", "Ref!")
    assert c_list == ["Ctx text"]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Generated answer"
    args_ref = MagicMock(answer_mode="reference", dataset_name="enterprise")
    assert make_answer(args_ref, mock_llm, "Q?", "Ref!", ["C"]) == "Ref!"

    args_gen = MagicMock(answer_mode="generated", dataset_name="enterprise")
    assert make_answer(args_gen, mock_llm, "Q?", "Ref!", ["C"]) == "Generated answer"

    args_short = MagicMock(
        answer_mode="generated", dataset_name="hotpotqa", prompt_style="short"
    )
    assert make_answer(args_short, mock_llm, "Q?", "Ref!", ["C"]) == "Generated answer"

    # Test write_results
    write_results(tmp_path)
    summary_files = list(
        tmp_path.glob("precomputed_deepeval_enterprise_reference_*_summary.json")
    )
    assert len(summary_files) == 1

    # Test run_eval deprecation warning
    mock_args = MagicMock()
    monkeypatch.setattr("deepeval_eval.deepeval_evaluator._run_eval", lambda a: None)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_eval(mock_args)
        assert len(w) == 1
        assert "deprecated" in str(w[0].message)

    # Test main CLI call with --help
    with patch("sys.argv", ["precomputed_deepeval", "--help"]):
        try:
            main()
        except SystemExit:
            pass


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
        answer_mode="reference",
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
