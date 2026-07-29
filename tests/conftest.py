from __future__ import annotations

import pytest

from deepeval_eval.core.config import get_eval_config


@pytest.fixture(autouse=True)
def clear_config_cache() -> None:
    """Clear lru_cache on get_eval_config before and after each test."""
    get_eval_config.cache_clear()
    yield
    get_eval_config.cache_clear()
