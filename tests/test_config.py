from __future__ import annotations

from pathlib import Path
import pytest

from deepeval_eval.config import ensure_dirs, load_dotenv_loose, resolve_llm_settings


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


def test_load_dotenv_loose_positive(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        "OPENAI_ENDPOINT=https://example.com/v1\n"
        "OPENAI_API_KEY=\"secret_key\"\n"
        "OPENAI_MODEL_NAME='gpt-4o'\n"
        "INVALID_LINE_WITHOUT_EQUALS\n",
        encoding="utf-8",
    )
    result = load_dotenv_loose(env_file)
    assert result["OPENAI_ENDPOINT"] == "https://example.com/v1"
    assert result["OPENAI_API_KEY"] == "secret_key"
    assert result["OPENAI_MODEL_NAME"] == "gpt-4o"
    assert "INVALID_LINE_WITHOUT_EQUALS" not in result


def test_load_dotenv_loose_negative(tmp_path: Path) -> None:
    non_existent = tmp_path / "non_existent.env"
    result = load_dotenv_loose(non_existent)
    assert result == {}


def test_resolve_llm_settings_positive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_ENDPOINT=http://localhost:8000\n"
        "OPENAI_API_KEY=testkey\n"
        "OPENAI_MODEL_NAME=testmodel\n",
        encoding="utf-8",
    )
    url, key, model = resolve_llm_settings(env_file, None, None, None)
    assert url == "http://localhost:8000"
    assert key == "testkey"
    assert model == "testmodel"

    url2, key2, model2 = resolve_llm_settings(env_file, "http://override", "overridekey", "overridemodel")
    assert url2 == "http://override"
    assert key2 == "overridekey"
    assert model2 == "overridemodel"


def test_resolve_llm_settings_negative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)

    with pytest.raises(RuntimeError, match="Missing LLM settings"):
        resolve_llm_settings(env_file, None, None, None)
