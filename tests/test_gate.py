from deepeval_eval.gate import evaluate_gate, render_markdown, resolve_metric_class_name


def _metric(score, success=None):
    if score is None:
        return {'score': None, 'success': False, 'reason': 'metric failed: boom'}
    if success is None:
        success = score >= 0.5
    return {'score': score, 'success': success, 'reason': 'ok'}


def _record(ar=0.9, fa=0.9, cr=0.8, cp=0.75, cre=0.7, dr=0.8, dp=0.6):
    return {
        'question_id': 'q',
        'doc_id_recall': dr,
        'doc_id_precision': dp,
        'metrics': {
            'AnswerRelevancyMetric': _metric(ar),
            'FaithfulnessMetric': _metric(fa),
            'ContextualRelevancyMetric': _metric(cr),
            'ContextualPrecisionMetric': _metric(cp),
            'ContextualRecallMetric': _metric(cre),
        },
    }


HARD_CONFIG = {
    'metrics': {
        'answer_relevancy': {'mean': 0.70, 'severity': 'hard'},
        'faithfulness': {'mean': 0.80, 'pass_rate': 0.90, 'severity': 'hard'},
    },
    'retrieval': {'doc_id_recall': {'mean': 0.60, 'severity': 'hard'}},
    'error_tolerance': 0.10,
}


def test_passes_when_all_above_threshold():
    report = evaluate_gate([_record() for _ in range(5)], HARD_CONFIG)
    assert report.passed
    assert report.hard_violations == []


def test_hard_mean_violation_fails():
    report = evaluate_gate([_record(fa=0.4) for _ in range(5)], HARD_CONFIG)
    assert not report.passed
    names = {v.name for v in report.hard_violations}
    assert 'faithfulness' in names


def test_soft_violation_does_not_fail():
    config = {'metrics': {'faithfulness': {'mean': 0.80, 'severity': 'soft'}}}
    report = evaluate_gate([_record(fa=0.4) for _ in range(3)], config)
    assert report.passed
    assert len(report.soft_violations) == 1


def test_error_rate_over_tolerance_fails():
    # Faithfulness errors on every record → 20% overall error rate > 10% tolerance.
    report = evaluate_gate([_record(fa=None) for _ in range(5)], HARD_CONFIG)
    assert not report.passed
    assert any(v.name == 'error_rate' for v in report.hard_violations)


def test_empty_results_is_hard_failure():
    report = evaluate_gate([], HARD_CONFIG)
    assert not report.passed
    assert any(v.name == 'no_results' for v in report.hard_violations)


def test_absent_metric_is_skipped_not_failed():
    # Config references a metric that never appears in the records.
    records = [{'metrics': {}, 'doc_id_recall': 0.9} for _ in range(3)]
    config = {'metrics': {'answer_relevancy': {'mean': 0.7, 'severity': 'hard'}}}
    report = evaluate_gate(records, config)
    assert report.passed
    assert report.metric_aggs['answer_relevancy'] is None


def test_render_markdown_reflects_status():
    passed = render_markdown(evaluate_gate([_record() for _ in range(3)], HARD_CONFIG))
    assert 'PASSED' in passed
    failed = render_markdown(evaluate_gate([_record(fa=0.3) for _ in range(3)], HARD_CONFIG))
    assert 'FAILED' in failed


def test_resolve_metric_class_name():
    assert resolve_metric_class_name("answer_relevancy") == "AnswerRelevancyMetric"
    assert resolve_metric_class_name("answer_correctness") == "AnswerCorrectnessMetric"
    assert resolve_metric_class_name("custom_eval") == "CustomEvalMetric"


def test_load_thresholds_json_and_validation(tmp_path):
    import json
    from deepeval_eval.gate import load_thresholds
    import pytest

    json_file = tmp_path / "thresholds.json"
    json_file.write_text(json.dumps({"error_tolerance": 0.05}), encoding="utf-8")
    data = load_thresholds(json_file)
    assert data["error_tolerance"] == 0.05

    bad_file = tmp_path / "invalid.json"
    bad_file.write_text("12345", encoding="utf-8")
    with pytest.raises(ValueError, match="Gate config must be a mapping"):
        load_thresholds(bad_file)


def test_run_gate_on_results_and_main(tmp_path, monkeypatch):
    import json
    from deepeval_eval.gate import main, run_gate_on_results

    config_path = tmp_path / "config.yaml"
    config_path.write_text("error_tolerance: 0.1\n", encoding="utf-8")

    results = [_record() for _ in range(2)]
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()

    gh_step = tmp_path / "github_step_summary.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh_step))

    passed = run_gate_on_results(results, config_path, summary_dir)
    assert passed
    assert (summary_dir / "gate_summary.md").exists()
    assert gh_step.exists()

    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(results), encoding="utf-8")

    exit_code = main(["--results", str(results_file), "--config", str(config_path)])
    assert exit_code == 0


