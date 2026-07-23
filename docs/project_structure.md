# Project Structure

This document explains the important files and directories in the repository.

## Top-Level Files

| Path | Purpose |
| --- | --- |
| README.md | Main project overview and quick start. |
| pyproject.toml | Python package metadata and dependency list. |
| .env.example | Example environment variables for local model settings. |
| .gitignore | Excludes local secrets, caches, generated data, results, and runtime artefacts. |

## Entry Points

| Path | Type | Purpose |
| --- | --- | --- |
| src/deepeval_eval/deepeval_evaluator.py | Python CLI | Unified entrypoint for DeepEval evaluation. |
| src/deepeval_eval/ingest.py | Python CLI | Standalone entrypoint for dataset ingestion. |
| scripts/ingest_enterprise.cmd | Windows CMD wrapper | Runs EnterpriseRAG-Bench ingestion with project defaults. |
| scripts/eval_enterprise.cmd | Windows CMD wrapper | Runs EnterpriseRAG-Bench evaluation with project defaults. |
| scripts/ingest_hotpotqa.cmd | Windows CMD wrapper | Runs HotpotQA ingestion with project defaults. |
| scripts/eval_hotpotqa.cmd | Windows CMD wrapper | Runs HotpotQA evaluation with project defaults. |
| scripts/ingest_enterprise.sh | POSIX shell wrapper | Runs EnterpriseRAG-Bench ingestion with project defaults. |
| scripts/eval_enterprise.sh | POSIX shell wrapper | Runs EnterpriseRAG-Bench evaluation with project defaults. |
| scripts/ingest_hotpotqa.sh | POSIX shell wrapper | Runs HotpotQA ingestion with project defaults. |
| scripts/eval_hotpotqa.sh | POSIX shell wrapper | Runs HotpotQA evaluation with project defaults. |

All wrapper scripts resolve the repository root from the script location and pass extra command-line arguments through to the Python CLI.

## Shared Utility Modules

| Path | Responsibility |
| --- | --- |
| src/deepeval_eval/caipe_client.py | CAIPE rag-server client, ingestion job operations, query calls, and retrieval response parsing. |
| src/deepeval_eval/config.py | Default paths, generated folder locations, environment loading, and LiteLLM setting resolution. |
| src/deepeval_eval/io_utils.py | Cached download helpers and JSONL evaluation question loading. |
| src/deepeval_eval/llm_client.py | OpenAI compatible LLM client, DeepEval model adapter, and prompt builders. |
| src/deepeval_eval/metrics.py | DeepEval metric construction, document ID scoring, and HotpotQA short-answer scoring. |

## Dataset-Specific Modules

| Path | Dataset | Responsibility |
| --- | --- | --- |
| src/deepeval_eval/enterprise_dataset.py | EnterpriseRAG-Bench | Load questions, download zip slices, sample documents, build CAIPE document payloads, and write generated data files. |
| src/deepeval_eval/hotpotqa_dataset.py | HotpotQA | Read preprocessed zip files, select questions, select gold documents plus distractors, build CAIPE payloads, and write generated data files. |

## Generated Local Directories

These directories are ignored by Git and are produced during local runs.

| Directory | Created by | Contents |
| --- | --- | --- |
| cache | Dataset loading | Downloaded EnterpriseRAG-Bench files or copied HotpotQA zip files. |
| data | Ingestion commands | Generated corpus and question files in JSONL and CSV. |
| results | Evaluation commands | Timestamped JSON and CSV evaluation results. |
| .deepeval | DeepEval runtime | Tool-generated DeepEval artefacts, if created. |
| __pycache__ | Python runtime | Compiled Python bytecode. |

## File Ownership by Workflow

| Workflow stage | Primary files |
| --- | --- |
| Configure model and paths | config.py, .env.example |
| Load dataset | enterprise_dataset.py, hotpotqa_dataset.py, io_utils.py |
| Ingest to CAIPE | enterprise_deepeval.py, hotpotqa_deepeval.py, caipe_client.py |
| Retrieve from CAIPE | caipe_client.py |
| Generate answers | llm_client.py |
| Score outputs | metrics.py, enterprise_deepeval.py, hotpotqa_deepeval.py |
| Run from terminal | scripts/*.cmd, scripts/*.sh, or direct Python CLI |

## Notes for New Contributors

- Keep generated data and result files out of Git.
- Add new dataset logic in a separate dataset module rather than expanding the entry point files.
- Shared behaviour should go into caipe_client.py, llm_client.py, metrics.py, config.py, or io_utils.py.
- If a new command changes outputs or arguments, update README.md and docs/setup_and_usage.md.
