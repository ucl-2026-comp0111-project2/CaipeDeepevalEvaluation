from __future__ import annotations

import json
from pathlib import Path

import requests


def download_text(url: str, cache_path: Path, timeout: int = 60) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding='utf-8')
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding='utf-8')
    return resp.text


def download_bytes(url: str, cache_path: Path, timeout: int = 180) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return resp.content


# Evaluation uses generated JSONL question files so ingestion and scoring stay
# connected to the same sampled corpus.
def load_eval_questions(path: Path, max_items: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_items and len(rows) >= max_items:
                break
    return rows
