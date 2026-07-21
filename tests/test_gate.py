from deepeval_eval.gate import evaluate_gate, render_markdown


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
