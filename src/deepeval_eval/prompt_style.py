from __future__ import annotations

import json
import os
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from deepeval_eval.llm_client import make_generation_prompt, make_short_answer_prompt


class PromptStyle(str, Enum):
    """Enum representing supported prompt styles for LLM answer generation."""

    GENERATION = "generation"
    SHORT = "short"

    @classmethod
    def _missing_(cls, value: object) -> PromptStyle | None:
        if isinstance(value, str):
            val_lower = value.lower()
            for member in cls:
                if member.value.lower() == val_lower:
                    return member
        return None


DEFAULT_PROMPT_STYLE: str = PromptStyle.GENERATION.value


PromptBuilder = Callable[[str, list[str]], str]

# Registry mapping style keys to prompt builder functions or formatting strings
_PROMPT_REGISTRY: dict[str, PromptBuilder | str] = {
    PromptStyle.GENERATION.value: make_generation_prompt,
    PromptStyle.SHORT.value: make_short_answer_prompt,
}


def register_prompt_style(
    style_name: str | PromptStyle,
    template_or_builder: str | PromptBuilder,
) -> None:
    """Register a new prompt style or overwrite an existing prompt style."""
    key = (
        style_name.value
        if isinstance(style_name, PromptStyle)
        else str(style_name).strip().lower()
    )
    _PROMPT_REGISTRY[key] = template_or_builder


def load_prompt_styles_from_config(config_path: str | Path) -> dict[str, str]:
    """Load prompt style templates from a JSON or YAML configuration file.

    Configuration file format example (JSON/YAML):
    {
      "prompt_styles": {
        "concise": "Answer concisely:\nQuestion: {question}\nContext:\n{context}",
        "detailed": "Provide a detailed answer:\nQuestion: {question}\nContext:\n{context}"
      }
    }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt style config file not found: {path}")

    content = path.read_text(encoding="utf-8")
    data: dict[str, Any] = {}

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(content) or {}
        except ImportError:
            raise RuntimeError("PyYAML is required to load YAML config files")
    else:
        data = json.loads(content)

    styles = data.get("prompt_styles", data)
    loaded: dict[str, str] = {}
    if isinstance(styles, dict):
        for name, template in styles.items():
            if isinstance(template, str):
                register_prompt_style(name, template)
                loaded[name] = template

    return loaded


def build_prompt(
    style: str | PromptStyle | None,
    question: str,
    contexts: list[str],
    config_path: str | Path | None = None,
) -> str:
    """Build a prompt string given a prompt style enum/string, question, and context list."""
    if config_path:
        load_prompt_styles_from_config(config_path)
    elif os.environ.get("PROMPT_STYLES_CONFIG"):
        config_env = os.environ["PROMPT_STYLES_CONFIG"]
        if Path(config_env).exists():
            load_prompt_styles_from_config(config_env)

    if style is None:
        style_key = DEFAULT_PROMPT_STYLE
    elif isinstance(style, PromptStyle):
        style_key = style.value
    else:
        style_key = str(style).strip().lower()

    builder_or_template = _PROMPT_REGISTRY.get(style_key)

    if builder_or_template is None:
        valid_styles = list(_PROMPT_REGISTRY.keys())
        raise ValueError(
            f"Unknown prompt style: '{style}'. Available styles: {valid_styles}"
        )

    if callable(builder_or_template):
        return builder_or_template(question, contexts)

    context_block = "\n\n".join(
        f"[{idx + 1}] {text}" for idx, text in enumerate(contexts)
    )
    return builder_or_template.format(
        question=question,
        context=context_block,
        contexts=context_block,
    )
