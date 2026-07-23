from __future__ import annotations

from deepeval_eval.telemetry import TelemetryMetrics, setup_otlp_tracing


def test_telemetry_metrics_initialization():
    """Verify TelemetryMetrics initializes and returns valid uptime."""
    tm = TelemetryMetrics()
    assert tm.get_uptime_seconds() >= 0.0


def test_telemetry_metrics_recording():
    """Verify metric recording methods produce valid Prometheus exposition output."""
    tm = TelemetryMetrics()
    tm.record_cache_hit()
    tm.record_cache_miss()
    tm.record_evaluation(1.5)
    tm.record_http_request("/health", 200)

    content, media_type = tm.export_prometheus()
    prom_text = content.decode("utf-8")
    assert "text/plain" in media_type
    assert "deepeval_uptime_seconds" in prom_text
    assert "deepeval_cache_hits_total" in prom_text
    assert "deepeval_cache_misses_total" in prom_text
    assert "deepeval_evaluations_total" in prom_text
    assert 'deepeval_http_requests_total{endpoint="/health",status="200"}' in prom_text


def test_setup_otlp_tracing():
    """Verify OTLP tracing setup function handles missing collector gracefully."""
    result = setup_otlp_tracing()
    assert isinstance(result, bool)


def test_trace_evaluation_span():
    """Verify trace_evaluation_span creates context manager without raising exceptions."""
    from deepeval_eval.telemetry import trace_evaluation_span

    with trace_evaluation_span(
        "enterprise", {"answer_mode": "generate", "max_items": 5}
    ):
        pass
