# Setup and Usage

This guide explains how to set up the Python environment and run both evaluation pipelines.

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.10 or later | Required by pyproject.toml. |
| CAIPE rag-server | Expected at http://localhost:9446 by default. |
| Cisco LiteLLM or compatible endpoint | Used for answer generation and DeepEval judge calls. |
| EnterpriseRAG-Bench network access | Needed when downloading EnterpriseRAG-Bench files. |
| HotpotQA preprocessed files | Required for HotpotQA ingestion. |

Network access is required during EnterpriseRAG-Bench dataset download unless files already exist in cache.

## Python Environment Setup

Create a virtual environment:

~~~powershell
py -3.10 -m venv .venv
~~~

Install the project in editable mode:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e .
~~~

The dependencies are declared in pyproject.toml:

| Dependency | Used for |
| --- | --- |
| deepeval | Evaluation metrics and test case objects. |
| httpx | OpenAI compatible LiteLLM requests. |
| requests | CAIPE and dataset download requests. |
| pydantic | Structured schema handling in the LLM adapter. |

## Environment Variables

The scripts load environment values from a configured env file and from the current shell. Shell values take priority when already set.

Required model settings:

~~~text
OPENAI_API_KEY=replace-with-cisco-litellm-key
OPENAI_ENDPOINT=https://llm-proxy.dev.outshift.ai/v1
OPENAI_MODEL_NAME=azure/gpt-5.4
~~~

The repository includes .env.example as a template. The default env file path used by config.py is:

~~~text
~/ai-platform-engineering/.env
~~~

To use another env file, pass --env-file to the Python entry point or append it after a wrapper script name.

## CAIPE Configuration

The default CAIPE rag-server URL is:

~~~text
http://localhost:9446
~~~

Override it with:

~~~powershell
python src\deepeval_eval\enterprise_deepeval.py eval --rag-url http://localhost:9446
~~~

The code supports an optional auth token:

~~~powershell
python src\deepeval_eval\enterprise_deepeval.py eval --auth-token YOUR_TOKEN
~~~

Automatic auth token retrieval is not implemented and is to be confirmed.

## Wrapper Scripts

The scripts directory contains platform wrappers around the Python entry points.

| Platform | Scripts |
| --- | --- |
| Windows | scripts\*.cmd |
| Linux or macOS | scripts/*.sh |

The wrappers include the repository default options and pass through any extra CLI arguments. Because extra arguments are appended last, they can override defaults where argparse accepts repeated options.

Windows example:

~~~powershell
.\scripts\eval_enterprise.cmd --max-items 1
~~~

Linux or macOS example:

~~~bash
./scripts/eval_enterprise.sh --max-items 1
~~~

## EnterpriseRAG-Bench Usage

Run ingestion on Windows:

~~~powershell
.\scripts\ingest_enterprise.cmd
~~~

Run evaluation on Windows:

~~~powershell
.\scripts\eval_enterprise.cmd
~~~

Run a one-question smoke test on Windows:

~~~powershell
.\scripts\eval_enterprise.cmd --max-items 1 --top-k 3 --max-context-chars 6000
~~~

Linux or macOS equivalents:

~~~bash
./scripts/ingest_enterprise.sh
./scripts/eval_enterprise.sh
./scripts/eval_enterprise.sh --max-items 1 --top-k 3 --max-context-chars 6000
~~~

Use direct Python if preferred:

~~~powershell
python src\deepeval_eval\enterprise_deepeval.py ingest --sources confluence jira github hubspot fireflies linear google_drive gmail slack --limit-per-source 1000 --num-questions 10 --questions-per-category 3 --batch-size 50
python src\deepeval_eval\enterprise_deepeval.py eval --max-items 10 --top-k 3 --max-context-chars 6000
~~~

## HotpotQA Usage

Place these files in cache or ~/Downloads:

~~~text
hotpotqa_full_questions.jsonl.zip
hotpotqa_full_document_pool.jsonl.zip
~~~

Run ingestion on Windows:

~~~powershell
.\scripts\ingest_hotpotqa.cmd
~~~

Run evaluation on Windows:

~~~powershell
.\scripts\eval_hotpotqa.cmd
~~~

Run a one-question smoke test on Windows:

~~~powershell
.\scripts\eval_hotpotqa.cmd --max-items 1 --top-k 5 --max-context-chars 12000
~~~

Linux or macOS equivalents:

~~~bash
./scripts/ingest_hotpotqa.sh
./scripts/eval_hotpotqa.sh
./scripts/eval_hotpotqa.sh --max-items 1 --top-k 5 --max-context-chars 12000
~~~

Use direct Python if preferred:

~~~powershell
python src\deepeval_eval\hotpotqa_deepeval.py ingest --limit 100 --questions-per-category 50 --max-docs 1000 --batch-size 50
python src\deepeval_eval\hotpotqa_deepeval.py eval --max-items 10 --top-k 5 --max-context-chars 12000
~~~

## Common Options

| Option | Applies to | Purpose |
| --- | --- | --- |
| --rag-url | ingest and eval | Override the CAIPE rag-server URL. |
| --auth-token | ingest and eval | Send a Bearer token to CAIPE. |
| --env-file | ingest and eval | Load model settings from a different env file. |
| --data-dir | ingest and eval | Override generated data folder. |
| --cache-dir | ingest and eval | Override cache folder. |
| --results-dir | eval | Override results folder. |
| --reset | ingest | Clear datasource before ingestion. |
| --skip-ingest | ingest | Generate local data files without sending documents to CAIPE. |

## Troubleshooting

| Problem | Likely cause | What to check |
| --- | --- | --- |
| Connection error to localhost:9446 | CAIPE rag-server is not running. | Start CAIPE and confirm rag-server is reachable. |
| Missing OPENAI settings | Env file missing or variables not set. | Check OPENAI_API_KEY, OPENAI_ENDPOINT, and OPENAI_MODEL_NAME. |
| HotpotQA file not found | Preprocessed zip files are missing. | Place the two zip files in cache or ~/Downloads. |
| Ingestion is slow | Large source count or batch size. | Reduce source list, limit per source, max docs, or question count. |
| Wrapper script cannot run | Shell, Python, or file permissions are not configured for the platform. | Use direct Python commands, confirm Python is on PATH, or use the platform-specific wrapper. |
| Evaluation question file missing | Ingestion has not generated data files. | Run the relevant ingest command first. |
