"""API application, authentication, job queue, and telemetry."""

from deepeval_eval.api.app import app, run_server

__all__ = ["app", "run_server"]
