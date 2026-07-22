from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepeval_eval.prompt_style import (
    DEFAULT_PROMPT_STYLE,
    PromptStyle,
    build_prompt,
    load_prompt_styles_from_config,
    register_prompt_style,
)


def test_default_prompt_style_constant() -> None:
    assert DEFAULT_PROMPT_STYLE == "generation"
    assert DEFAULT_PROMPT_STYLE == PromptStyle.GENERATION.value


def test_prompt_style_enum_positive() -> None:
    assert PromptStyle.GENERATION == "generation"
    assert PromptStyle.SHORT == "short"
    assert PromptStyle("generation") == PromptStyle.GENERATION
    assert PromptStyle("SHORT") == PromptStyle.SHORT


def test_build_prompt_default_and_builtin_positive() -> None:
    question = "What is capital of France?"
    contexts = ["Paris is the capital of France."]

    # Default (None) uses generation
    prompt_default = build_prompt(None, question, contexts)
    assert "Answer the question using only the context below" in prompt_default
    assert question in prompt_default

    # Enum GENERATION
    prompt_gen = build_prompt(PromptStyle.GENERATION, question, contexts)
    assert prompt_gen == prompt_default

    # Enum SHORT
    prompt_short = build_prompt(PromptStyle.SHORT, question, contexts)
    assert "Keep the answer short" in prompt_short

    # String "short"
    prompt_short_str = build_prompt("short", question, contexts)
    assert prompt_short_str == prompt_short


def test_register_prompt_style_positive() -> None:
    register_prompt_style("custom_style", "Summary of {question}:\n{context}")
    result = build_prompt(
        "custom_style", "Explain quantum computing", ["Qubits superpose."]
    )
    assert "Summary of Explain quantum computing:" in result
    assert "[1] Qubits superpose." in result


def test_load_prompt_styles_from_config_json_positive(tmp_path: Path) -> None:
    config_file = tmp_path / "prompts.json"
    config_data = {
        "prompt_styles": {
            "json_style": "JSON context: {context} -> question: {question}"
        }
    }
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    loaded = load_prompt_styles_from_config(config_file)
    assert "json_style" in loaded

    prompt = build_prompt("json_style", "Why sky blue?", ["Rayleigh scattering."])
    assert "JSON context: [1] Rayleigh scattering. -> question: Why sky blue?" in prompt


def test_load_prompt_styles_from_config_yaml_positive(tmp_path: Path) -> None:
    config_file = tmp_path / "prompts.yaml"
    config_data = """
prompt_styles:
  yaml_style: "YAML context: {context} Question: {question}"
"""
    config_file.write_text(config_data, encoding="utf-8")

    loaded = load_prompt_styles_from_config(config_file)
    assert "yaml_style" in loaded

    prompt = build_prompt("yaml_style", "Test Q", ["Test C"])
    assert "YAML context: [1] Test C Question: Test Q" in prompt


def test_build_prompt_unknown_style_negative() -> None:
    with pytest.raises(ValueError, match="Unknown prompt style: 'non_existent'"):
        build_prompt("non_existent", "Question?", ["Context"])


def test_load_prompt_styles_missing_file_negative() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompt_styles_from_config("non_existent_file.json")
