from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepeval_eval.ingest import (
    build_parser,
    main,
    run_enterprise_ingest,
    run_hotpotqa_ingest,
    run_ingest,
)


def test_ingest_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.dataset_name == "enterprise"
    assert args.rag_url == "http://localhost:9446"
    assert args.reset is False
    assert args.skip_ingest is False


def test_ingest_build_parser_custom_args(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-name",
            "hotpotqa",
            "--rag-url",
            "http://caipe:9000",
            "--reset",
            "--skip-ingest",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert args.dataset_name == "hotpotqa"
    assert args.rag_url == "http://caipe:9000"
    assert args.reset is True
    assert args.skip_ingest is True
    assert args.data_dir == tmp_path


def test_run_ingest_dispatch_positive(tmp_path: Path) -> None:
    """Verify run_ingest dispatches to enterprise and hotpotqa ingestion functions."""
    args_ent = argparse.Namespace(dataset_name="enterprise_rag_bench", skip_ingest=True)
    args_hot = argparse.Namespace(dataset_name="hotpotqa", skip_ingest=True)

    with (
        patch("deepeval_eval.ingest.run_enterprise_ingest") as mock_ent,
        patch("deepeval_eval.ingest.run_hotpotqa_ingest") as mock_hot,
    ):
        run_ingest(args_ent)
        mock_ent.assert_called_once_with(args_ent)

        run_ingest(args_hot)
        mock_hot.assert_called_once_with(args_hot)


def test_run_ingest_unsupported_dataset_negative() -> None:
    args = argparse.Namespace(dataset_name="invalid_dataset")
    with pytest.raises(
        ValueError, match="Unsupported dataset for ingestion: invalid_dataset"
    ):
        run_ingest(args)


def test_run_enterprise_ingest_skip_ingest(tmp_path: Path) -> None:
    args = argparse.Namespace(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        sources=["confluence"],
        num_questions=2,
        questions_per_category=1,
        limit_per_source=10,
        skip_ingest=True,
    )

    mock_q = MagicMock(expected_doc_ids=["doc1"])
    mock_doc = MagicMock(doc_id="doc1")

    with (
        patch("deepeval_eval.enterprise_dataset.load_questions", return_value=[mock_q]),
        patch(
            "deepeval_eval.enterprise_dataset.select_questions", return_value=[mock_q]
        ),
        patch(
            "deepeval_eval.enterprise_dataset.fetch_documents", return_value=[mock_doc]
        ),
        patch("deepeval_eval.enterprise_dataset.write_corpus") as mock_write_corpus,
        patch("deepeval_eval.enterprise_dataset.write_questions") as mock_write_q,
    ):
        run_enterprise_ingest(args)
        mock_write_corpus.assert_called_once()
        mock_write_q.assert_called_once()


def test_run_enterprise_ingest_with_caipe_ingest_positive(tmp_path: Path) -> None:
    """Verify run_enterprise_ingest executes full CAIPE ingestion workflow when skip_ingest=False."""
    args = argparse.Namespace(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        sources=["confluence"],
        num_questions=2,
        questions_per_category=1,
        limit_per_source=10,
        skip_ingest=False,
        reset=True,
        rag_url="http://localhost:9446",
        auth_token="token_123",
        datasource_id="ds_123",
        datasource_name="DS 123",
        batch_size=50,
    )

    mock_q = MagicMock(expected_doc_ids=["doc1"])
    mock_doc = MagicMock(doc_id="doc1")
    mock_client = MagicMock()
    mock_client.register_ingestor.return_value = ("ingestor_1", 100)
    mock_client.open_job.return_value = "job_1"

    with (
        patch("deepeval_eval.enterprise_dataset.load_questions", return_value=[mock_q]),
        patch(
            "deepeval_eval.enterprise_dataset.select_questions", return_value=[mock_q]
        ),
        patch(
            "deepeval_eval.enterprise_dataset.fetch_documents", return_value=[mock_doc]
        ),
        patch("deepeval_eval.ingest.CaipeRagClient", return_value=mock_client),
        patch("deepeval_eval.enterprise_dataset.write_corpus"),
        patch("deepeval_eval.enterprise_dataset.write_questions"),
    ):
        run_enterprise_ingest(args)
        mock_client.reset_datasource.assert_called_once_with("ds_123")
        mock_client.register_ingestor.assert_called_once()
        mock_client.upsert_datasource.assert_called_once()
        mock_client.open_job.assert_called_once()
        mock_client.ingest_batch.assert_called_once()
        mock_client.close_job.assert_called_once_with(
            "job_1", "EnterpriseRAG-Bench DeepEval ingestion complete"
        )


def test_run_hotpotqa_ingest_skip_ingest(tmp_path: Path) -> None:
    args = argparse.Namespace(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        questions_zip=tmp_path / "q.zip",
        documents_zip=tmp_path / "d.zip",
        limit=10,
        questions_per_category=5,
        categories=None,
        distractors_per_question=2,
        max_docs=20,
        skip_ingest=True,
    )

    mock_q = {"question_id": "q1", "expected_doc_ids": ["doc1"]}
    mock_doc = {"document_id": "doc1"}

    with (
        patch(
            "deepeval_eval.hotpotqa_dataset.resolve_zip",
            side_effect=lambda p, fb: tmp_path / fb,
        ),
        patch("deepeval_eval.hotpotqa_dataset.load_questions", return_value=[mock_q]),
        patch("deepeval_eval.hotpotqa_dataset.select_questions", return_value=[mock_q]),
        patch(
            "deepeval_eval.hotpotqa_dataset.load_document_pool", return_value=[mock_doc]
        ),
        patch(
            "deepeval_eval.hotpotqa_dataset.select_documents", return_value=[mock_doc]
        ),
        patch("deepeval_eval.hotpotqa_dataset.write_corpus") as mock_write_corpus,
        patch("deepeval_eval.hotpotqa_dataset.write_questions") as mock_write_q,
    ):
        run_hotpotqa_ingest(args)
        mock_write_corpus.assert_called_once()
        mock_write_q.assert_called_once()


def test_run_hotpotqa_ingest_with_caipe_ingest_positive(tmp_path: Path) -> None:
    """Verify run_hotpotqa_ingest executes full CAIPE ingestion workflow when skip_ingest=False."""
    args = argparse.Namespace(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        questions_zip=tmp_path / "q.zip",
        documents_zip=tmp_path / "d.zip",
        limit=10,
        questions_per_category=5,
        categories=None,
        distractors_per_question=2,
        max_docs=20,
        skip_ingest=False,
        reset=True,
        rag_url="http://localhost:9446",
        auth_token="token_456",
        datasource_id="ds_hotpot",
        datasource_name="DS Hotpot",
        batch_size=20,
    )

    mock_q = {"question_id": "q1", "expected_doc_ids": ["doc1"]}
    mock_doc = {"document_id": "doc1", "title": "Doc Title", "text": "Doc text"}
    mock_client = MagicMock()
    mock_client.register_ingestor.return_value = ("ingestor_2", 50)
    mock_client.open_job.return_value = "job_2"

    with (
        patch(
            "deepeval_eval.hotpotqa_dataset.resolve_zip",
            side_effect=lambda p, fb: tmp_path / fb,
        ),
        patch("deepeval_eval.hotpotqa_dataset.load_questions", return_value=[mock_q]),
        patch("deepeval_eval.hotpotqa_dataset.select_questions", return_value=[mock_q]),
        patch(
            "deepeval_eval.hotpotqa_dataset.load_document_pool", return_value=[mock_doc]
        ),
        patch(
            "deepeval_eval.hotpotqa_dataset.select_documents", return_value=[mock_doc]
        ),
        patch("deepeval_eval.ingest.CaipeRagClient", return_value=mock_client),
        patch("deepeval_eval.hotpotqa_dataset.write_corpus"),
        patch("deepeval_eval.hotpotqa_dataset.write_questions"),
    ):
        run_hotpotqa_ingest(args)
        mock_client.reset_datasource.assert_called_once_with("ds_hotpot")
        mock_client.register_ingestor.assert_called_once()
        mock_client.upsert_datasource.assert_called_once()
        mock_client.open_job.assert_called_once()
        mock_client.ingest_batch.assert_called_once()
        mock_client.close_job.assert_called_once_with(
            "job_2", "HotpotQA DeepEval ingestion complete"
        )


def test_main_cli_execution_positive() -> None:
    """Verify main function parses arguments and triggers ingestion."""
    with (
        patch("sys.argv", ["ingest", "--skip-ingest"]),
        patch("deepeval_eval.ingest.load_dotenv_loose") as mock_dotenv,
        patch("deepeval_eval.ingest.run_ingest") as mock_run_ingest,
    ):
        main()
        mock_dotenv.assert_called_once()
        mock_run_ingest.assert_called_once()
