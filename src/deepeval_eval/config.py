from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('DEEPEVAL_DISABLE_DOTENV', '1')
os.environ.setdefault('DEEPEVAL_TELEMETRY_OPT_OUT', '1')

WORK_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = WORK_DIR / 'data'
DEFAULT_CACHE_DIR = WORK_DIR / 'cache'
DEFAULT_RESULTS_DIR = WORK_DIR / 'results'
DEFAULT_ENV_FILE = Path('C:/Users/liana/ai-platform-engineering/.env')
DEFAULT_DOWNLOADS_DIR = Path('C:/Users/liana/Downloads')


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# The evaluation scripts reuse CAIPE local .env values, but shell-provided
# values should win when running experiments with a different model or key.
def load_dotenv_loose(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    raw = path.read_text(encoding='utf-8', errors='ignore')
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip(chr(39)).strip(chr(34))
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def resolve_litellm_settings(env_file: Path, base_url: str | None, api_key: str | None, model: str | None) -> tuple[str, str, str]:
    env_values = load_dotenv_loose(env_file)
    resolved_base_url = base_url or env_values.get('OPENAI_ENDPOINT') or os.environ.get('OPENAI_ENDPOINT')
    resolved_api_key = api_key or env_values.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    resolved_model = model or env_values.get('OPENAI_MODEL_NAME') or os.environ.get('OPENAI_MODEL_NAME')
    if not resolved_base_url or not resolved_api_key or not resolved_model:
        raise RuntimeError('Missing Cisco LiteLLM settings. Need OPENAI_ENDPOINT, OPENAI_API_KEY, and OPENAI_MODEL_NAME.')
    return resolved_base_url, resolved_api_key, resolved_model
