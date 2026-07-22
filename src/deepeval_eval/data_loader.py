from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from deepeval_eval.config import DEFAULT_DATA_DIR


def resolve_questions_file(
    dataset_name: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    questions_file: Path | None = None,
) -> Path:
    """Resolve the questions dataset file path via explicit input or naming convention."""
    if questions_file is not None:
        resolved = Path(questions_file)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Specified questions file does not exist: {questions_file}")

    candidates = [
        data_dir / f"{dataset_name}_deepeval_questions.jsonl",
        data_dir / f"{dataset_name}_questions.jsonl",
        data_dir / f"{dataset_name}.jsonl",
        data_dir / f"{dataset_name}_deepeval_questions.csv",
        data_dir / f"{dataset_name}_questions.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No valid questions file found for dataset_name='{dataset_name}' in data_dir='{data_dir}'"
    )


class BaseDataLoader(ABC):
    """Abstract base class for evaluation dataset loaders."""

    @abstractmethod
    def load(
        self,
        max_items: Optional[int] = None,
        limit_per_category: Optional[int] = None,
        combine_with_level: bool = False,
    ) -> list[dict[str, Any]]:
        """Load evaluation questions as a list of dictionaries."""
        pass


class FileDataLoader(BaseDataLoader):
    """Data loader that reads evaluation question items from a JSONL or CSV file on disk."""

    def __init__(
        self,
        questions_file: Optional[Path] = None,
        dataset_name: str = "enterprise",
        data_dir: Path = DEFAULT_DATA_DIR,
    ) -> None:
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.questions_file = questions_file

    def resolve_file(self) -> Path:
        return resolve_questions_file(
            dataset_name=self.dataset_name,
            data_dir=self.data_dir,
            questions_file=self.questions_file,
        )

    def load(
        self,
        max_items: Optional[int] = None,
        limit_per_category: Optional[int] = None,
        combine_with_level: bool = False,
    ) -> list[dict[str, Any]]:
        path = self.resolve_file()
        rows: list[dict[str, Any]] = []
        category_counts: dict[tuple[str, str | None] | str, int] = {}

        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    cat = item.get("category", "basic") or "basic"
                    if limit_per_category is not None:
                        key = (cat, item.get("level")) if combine_with_level else cat
                        count = category_counts.get(key, 0)
                        if count >= limit_per_category:
                            continue
                        category_counts[key] = count + 1
                    rows.append(item)
                    if max_items and len(rows) >= max_items:
                        break
        elif path.suffix == ".csv":
            import csv

            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for item in reader:
                    cat = item.get("category", "basic") or "basic"
                    if limit_per_category is not None:
                        key = (cat, item.get("level")) if combine_with_level else cat
                        count = category_counts.get(key, 0)
                        if count >= limit_per_category:
                            continue
                        category_counts[key] = count + 1
                    rows.append(dict(item))
                    if max_items and len(rows) >= max_items:
                        break
        else:
            raise ValueError(f"Unsupported file format for evaluation questions: {path.suffix}")

        return rows


class InMemoryDataLoader(BaseDataLoader):
    """Data loader that wraps an in-memory list of evaluation question dicts."""

    def __init__(self, dataset: list[dict[str, Any]]) -> None:
        self._dataset = dataset

    def load(
        self,
        max_items: Optional[int] = None,
        limit_per_category: Optional[int] = None,
        combine_with_level: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        category_counts: dict[tuple[str, str | None] | str, int] = {}

        for item in self._dataset:
            cat = item.get("category", "basic") or "basic"
            if limit_per_category is not None:
                key = (cat, item.get("level")) if combine_with_level else cat
                count = category_counts.get(key, 0)
                if count >= limit_per_category:
                    continue
                category_counts[key] = count + 1
            rows.append(item)
            if max_items and len(rows) >= max_items:
                break
        return rows


class DatabaseDataLoader(BaseDataLoader):
    """Data loader that fetches evaluation questions from a database (e.g. PostgreSQL or MongoDB)."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_or_collection: str = "eval_questions",
    ) -> None:
        self.connection_string = connection_string
        self.table_or_collection = table_or_collection

    def load(
        self,
        max_items: Optional[int] = None,
        limit_per_category: Optional[int] = None,
        combine_with_level: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.connection_string:
            raise ValueError("connection_string is required for DatabaseDataLoader")
        raise NotImplementedError("DatabaseDataLoader query execution requires active DB connection.")
