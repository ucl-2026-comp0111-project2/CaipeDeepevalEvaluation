from __future__ import annotations

from pathlib import Path
import pytest

from deepeval_eval.data_loader import (
    DatabaseDataLoader,
    FileDataLoader,
    InMemoryDataLoader,
    resolve_questions_file,
)


def test_resolve_questions_file_explicit(tmp_path: Path):
    q_file = tmp_path / "custom_questions.jsonl"
    q_file.write_text('{"user_input": "test"}', encoding="utf-8")

    res = resolve_questions_file("custom", questions_file=q_file)
    assert res == q_file


def test_resolve_questions_file_by_convention(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    q_file = data_dir / "my_ds_deepeval_questions.jsonl"
    q_file.write_text('{"user_input": "hello"}', encoding="utf-8")

    res = resolve_questions_file("my_ds", data_dir=data_dir)
    assert res == q_file


def test_resolve_questions_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_questions_file("nonexistent", data_dir=tmp_path)


def test_in_memory_data_loader():
    items = [
        {"user_input": "q1", "category": "cat1"},
        {"user_input": "q2", "category": "cat1"},
        {"user_input": "q3", "category": "cat2"},
    ]
    loader = InMemoryDataLoader(items)
    loaded = loader.load(max_items=2)
    assert len(loaded) == 2
    assert loaded[0]["user_input"] == "q1"

    limited = loader.load(limit_per_category=1)
    assert len(limited) == 2  # 1 from cat1, 1 from cat2


def test_file_data_loader(tmp_path: Path):
    q_file = tmp_path / "sample_deepeval_questions.jsonl"
    q_file.write_text('{"user_input": "q1"}\n{"user_input": "q2"}\n', encoding="utf-8")

    loader = FileDataLoader(questions_file=q_file)
    rows = loader.load()
    assert len(rows) == 2
    assert rows[0]["user_input"] == "q1"


def test_database_data_loader_requires_connection_string():
    loader = DatabaseDataLoader()
    with pytest.raises(ValueError, match="connection_string is required"):
        loader.load()


def test_resolve_questions_file_explicit_nonexistent(tmp_path: Path):
    q_file = tmp_path / "nonexistent.jsonl"
    with pytest.raises(FileNotFoundError, match="Specified questions file does not exist"):
        resolve_questions_file("custom", questions_file=q_file)


def test_file_data_loader_csv_and_limits(tmp_path: Path):
    csv_file = tmp_path / "questions.csv"
    csv_file.write_text(
        "question_id,category,level,user_input\n"
        "q1,catA,L1,What is Python?\n"
        "q2,catA,L1,What is UV?\n"
        "q3,catA,L2,What is pytest?\n"
        "q4,catB,L1,What is Docker?\n",
        encoding="utf-8",
    )

    loader = FileDataLoader(questions_file=csv_file)
    rows = loader.load(limit_per_category=1, combine_with_level=True)
    assert len(rows) == 3  # (catA, L1), (catA, L2), (catB, L1)

    rows_limited = loader.load(max_items=2)
    assert len(rows_limited) == 2


def test_file_data_loader_unsupported_format(tmp_path: Path):
    txt_file = tmp_path / "questions.txt"
    txt_file.write_text("user_input: q1\n", encoding="utf-8")

    loader = FileDataLoader(questions_file=txt_file)
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load()


def test_file_data_loader_jsonl_limits(tmp_path: Path):
    jsonl_file = tmp_path / "questions.jsonl"
    jsonl_file.write_text(
        '{"category": "c1", "level": "l1", "user_input": "q1"}\n'
        '\n'
        '{"category": "c1", "level": "l1", "user_input": "q2"}\n'
        '{"category": "c1", "level": "l2", "user_input": "q3"}\n',
        encoding="utf-8",
    )
    loader = FileDataLoader(questions_file=jsonl_file)
    rows = loader.load(limit_per_category=1, combine_with_level=False)
    assert len(rows) == 1

