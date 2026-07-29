from __future__ import annotations

from pathlib import Path

import pytest

from deepeval_eval.core.config import (
    AgenticSettings,
    CaipeClientSettings,
    DatabaseSettings,
    LLMSettings,
    ensure_dirs,
    get_eval_config,
    resolve_llm_settings,
)


def test_ensure_dirs_positive(tmp_path: Path) -> None:
    dir1 = tmp_path / "a" / "b"
    dir2 = tmp_path / "c"
    ensure_dirs(dir1, dir2)
    assert dir1.exists() and dir1.is_dir()
    assert dir2.exists() and dir2.is_dir()


def test_ensure_dirs_negative(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    ensure_dirs(existing)
    assert existing.exists()


def test_llm_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OPENAI_ENDPOINT",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_MODEL_NAME",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = LLMSettings(_env_file=None)
    assert settings.base_url == "http://localhost:8000/v1"
    assert settings.api_key.get_secret_value() == "mock-key"
    assert settings.model == "gpt-4o-mini"


def test_llm_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ENDPOINT", "http://custom-endpoint/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-token")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "custom-llm")

    settings = LLMSettings(_env_file=None)
    assert settings.base_url == "http://custom-endpoint/v1"  # Trailing slash normalized
    assert settings.api_key.get_secret_value() == "secret-token"
    assert settings.model == "custom-llm"


def test_agentic_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CAIPE_AGENT_ID",
        "AGENT_ID",
        "CAIPE_SUPERVISOR_URL",
        "SUPERVISOR_URL",
        "INSECURE_SSL",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = AgenticSettings(_env_file=None)
    assert settings.agent_id == "hello-world"
    assert settings.supervisor_url == "http://localhost:8000"
    assert settings.insecure is True


def test_agentic_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAIPE_AGENT_ID", "custom-agent")
    monkeypatch.setenv("CAIPE_SUPERVISOR_URL", "http://supervisor:9000/")
    monkeypatch.setenv("INSECURE_SSL", "false")

    settings = AgenticSettings()
    assert settings.agent_id == "custom-agent"
    assert (
        settings.supervisor_url == "http://supervisor:9000"
    )  # Trailing slash normalized
    assert settings.insecure is False


def test_caipe_client_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAIPE_BASE_URL", "https://api.caipe.com/rag/")
    monkeypatch.setenv("CAIPE_AUTH_TOKEN", "bearer-123")

    settings = CaipeClientSettings()
    assert settings.base_url == "https://api.caipe.com/rag"
    assert settings.auth_token is not None
    assert settings.auth_token.get_secret_value() == "bearer-123"


def test_database_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/evaldb")

    settings = DatabaseSettings()
    assert settings.connection_string is not None
    assert (
        settings.connection_string.get_secret_value()
        == "postgresql://user:pass@localhost:5432/evaldb"
    )


def test_eval_config_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o")
    cfg1 = get_eval_config()
    assert cfg1.llm.model == "gpt-4o"

    # Reset cache
    get_eval_config.cache_clear()
    monkeypatch.setenv("OPENAI_MODEL_NAME", "claude-3-5-sonnet")
    cfg2 = get_eval_config()
    assert cfg2.llm.model == "claude-3-5-sonnet"


def test_resolve_llm_settings_legacy_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ENDPOINT", "http://localhost:8000")
    monkeypatch.setenv("OPENAI_API_KEY", "testkey")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "testmodel")

    url, key, model = resolve_llm_settings()
    assert url == "http://localhost:8000"
    assert key == "testkey"
    assert model == "testmodel"
