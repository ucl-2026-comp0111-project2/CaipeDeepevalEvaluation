from pathlib import Path
from unittest.mock import MagicMock

from deepeval_eval.data_loader import InMemoryDataLoader
from deepeval_eval.eval_engine import EvalConfig, run_evaluation


def test_eval_config_defaults():
    config = EvalConfig()
    assert config.dataset_name == "enterprise"
    assert config.answer_mode == "reference"
    assert config.top_k == 3
    assert not config.agentic
    assert not config.save_to_db


def test_eval_config_to_config_args():
    config = EvalConfig(
        dataset_name="hotpotqa",
        top_k=5,
        questions_file=Path("/tmp/questions.jsonl"),
        save_to_db=True,
    )
    config_dict = config.to_config_args()
    assert config_dict["dataset_name"] == "hotpotqa"
    assert config_dict["top_k"] == 5
    assert config_dict["questions_file"] == "/tmp/questions.jsonl"
    assert config_dict["save_to_db"] is True


def test_run_evaluation_with_in_memory_data_loader(tmp_path: Path):
    config = EvalConfig(
        dataset_name="custom_ds",
        results_dir=tmp_path / "results",
    )
    dataset = [
        {
            "question_id": "q1",
            "user_input": "What is CAIPE?",
            "reference": "CAIPE is an AI platform.",
        }
    ]
    loader = InMemoryDataLoader(dataset)
    mock_rag_client = MagicMock()
    mock_query_res = MagicMock()
    mock_query_res.answer = "CAIPE is an AI platform."
    mock_query_res.contexts = ["CAIPE is an AI platform."]
    mock_query_res.sources = []
    mock_query_res.retrieved_doc_ids = []
    mock_query_res.input_tokens = 10
    mock_query_res.output_tokens = 10
    mock_query_res.total_tokens = 20
    mock_query_res.latency_sec = 0.5
    mock_query_res.latency_ms = 500
    mock_query_res.log_file = None
    mock_rag_client.query.return_value = mock_query_res

    res = run_evaluation(config=config, data_loader=loader, rag_client=mock_rag_client, metrics=[])
    assert len(res) == 1
    assert res[0]["dataset_name"] == "custom_ds"
    assert res[0]["question"] == "What is CAIPE?"


def test_run_evaluation_question_ids_and_indices(tmp_path: Path):
    dataset = [
        {"question_id": "q101", "user_input": "Q1", "reference": "A1"},
        {"question_id": "q102", "user_input": "Q2", "reference": "A2"},
        {"question_id": "q103", "user_input": "Q3", "reference": "A3"},
    ]
    loader = InMemoryDataLoader(dataset)

    mock_rag_client = MagicMock()
    mock_query_res = MagicMock()
    mock_query_res.answer = "Ans"
    mock_query_res.contexts = []
    mock_query_res.sources = []
    mock_query_res.retrieved_doc_ids = []
    mock_query_res.input_tokens = 5
    mock_query_res.output_tokens = 5
    mock_query_res.total_tokens = 10
    mock_query_res.latency_sec = 0.1
    mock_query_res.latency_ms = 100
    mock_query_res.log_file = None
    mock_rag_client.query.return_value = mock_query_res

    # Filter by question_ids
    config_ids = EvalConfig(results_dir=tmp_path / "res1", question_ids="q102")
    res_ids = run_evaluation(config=config_ids, data_loader=loader, rag_client=mock_rag_client, metrics=[])
    assert len(res_ids) == 1
    assert res_ids[0]["question_id"] == "q102"

    # Filter by question_indices (range and single index)
    config_idx = EvalConfig(results_dir=tmp_path / "res2", question_indices="1-2, 3")
    res_idx = run_evaluation(config=config_idx, data_loader=loader, rag_client=mock_rag_client, metrics=[])
    assert len(res_idx) == 3


def test_run_evaluation_prompt_config_and_db(tmp_path: Path):
    prompt_yaml = tmp_path / "prompt_config.yaml"
    prompt_yaml.write_text("styles:\n  custom:\n    system_prompt: sys\n    user_template: '{context} {question}'\n", encoding="utf-8")

    dataset = [{"question_id": "q1", "user_input": "Test prompt config?", "reference": "Ref"}]
    loader = InMemoryDataLoader(dataset)

    mock_rag_client = MagicMock()
    mock_query_res = MagicMock()
    mock_query_res.answer = "Ans"
    mock_query_res.contexts = []
    mock_query_res.sources = []
    mock_query_res.retrieved_doc_ids = []
    mock_query_res.input_tokens = 1
    mock_query_res.output_tokens = 1
    mock_query_res.total_tokens = 2
    mock_query_res.latency_sec = 0.1
    mock_query_res.latency_ms = 100
    mock_query_res.log_file = None
    mock_rag_client.query.return_value = mock_query_res

    config = EvalConfig(
        results_dir=tmp_path / "res_prompt",
        prompt_config=prompt_yaml,
        save_to_db=True,
        db_connection_string="sqlite:///:memory:",
    )
    res = run_evaluation(config=config, data_loader=loader, rag_client=mock_rag_client, metrics=[])
    assert len(res) == 1


class MockGoodMetric:
    def __init__(self):
        self.score = 0.95
        self.reason = "Good response"

    def measure(self, test_case):
        pass

    def is_successful(self):
        return True


class MockBrokenGetReasonMetric:
    def __init__(self):
        self.score = 0.85
        self.reason = "Valid reason"

    def measure(self, test_case):
        pass

    def is_successful(self):
        return True

    def get_reason(self):
        raise AttributeError("'MockBrokenGetReasonMetric' object has no attribute 'get_reason'")


def test_run_evaluation_preserves_metric_scores_and_reasons(tmp_path: Path):
    dataset = [{"question_id": "q_metric", "user_input": "Test metric?", "reference": "Ref"}]
    loader = InMemoryDataLoader(dataset)

    mock_rag_client = MagicMock()
    mock_query_res = MagicMock()
    mock_query_res.answer = "Ans"
    mock_query_res.contexts = []
    mock_query_res.sources = []
    mock_query_res.retrieved_doc_ids = []
    mock_query_res.input_tokens = 1
    mock_query_res.output_tokens = 1
    mock_query_res.total_tokens = 2
    mock_query_res.latency_sec = 0.1
    mock_query_res.latency_ms = 100
    mock_query_res.log_file = None
    mock_rag_client.query.return_value = mock_query_res

    good_metric = MockGoodMetric()
    broken_metric = MockBrokenGetReasonMetric()

    config = EvalConfig(results_dir=tmp_path / "res_metrics")
    res = run_evaluation(
        config=config,
        data_loader=loader,
        rag_client=mock_rag_client,
        metrics=[good_metric, broken_metric],
    )
    assert len(res) == 1
    metrics_res = res[0]["metrics"]
    assert metrics_res["MockGoodMetric"]["score"] == 0.95
    assert metrics_res["MockGoodMetric"]["reason"] == "Good response"
    assert metrics_res["MockGoodMetric"]["success"] is True

    # Check that score was PRESERVED despite get_reason throwing AttributeError
    assert metrics_res["MockBrokenGetReasonMetric"]["score"] == 0.85
    assert metrics_res["MockBrokenGetReasonMetric"]["reason"] == "Valid reason"
    assert metrics_res["MockBrokenGetReasonMetric"]["success"] is True



