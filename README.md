# CAIPE DeepEval Evaluation

This repository contains DeepEval based evaluation pipelines for CAIPE RAG over two benchmark datasets:

- EnterpriseRAG-Bench
- HotpotQA

The project supports controlled document ingestion into CAIPE, evaluation question generation, retrieval checks against expected document IDs, answer generation from retrieved context, and metric scoring with DeepEval.

## Problem This Project Solves

CAIPE can retrieve documents from a knowledge base through its rag-server. This repository provides a repeatable way to check whether CAIPE retrieves the expected source documents and whether answers generated from those retrieved contexts are relevant and faithful.

The current implementation evaluates CAIPE rag-server retrieval and context-grounded answer generation. Full CAIPE agentic chat evaluation is to be confirmed.

## Main Features

| Area | What is implemented |
| --- | --- |
| Dataset ingestion | Ingests bounded samples from EnterpriseRAG-Bench and HotpotQA into CAIPE. |
| Retrieval evaluation | Compares CAIPE retrieved document IDs with expected document IDs. |
| Answer generation | Uses an OpenAI compatible Cisco LiteLLM endpoint to answer from retrieved context. |
| DeepEval scoring | Runs AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualPrecisionMetric, and ContextualRecallMetric. |
| HotpotQA checks | Adds normalized exact match and contains reference checks for short answers. |
| Output files | Writes JSON and CSV outputs under data and results. |

## Repository Structure

~~~text
caipe_deepeval_evaluation/
|-- README.md
|-- pyproject.toml
|-- .env.example
|-- .gitignore
|-- scripts/
|   |-- ingest_enterprise.cmd
|   |-- eval_enterprise.cmd
|   |-- ingest_hotpotqa.cmd
|   |-- eval_hotpotqa.cmd
|   |-- ingest_enterprise.sh
|   |-- eval_enterprise.sh
|   |-- ingest_hotpotqa.sh
|   |-- eval_hotpotqa.sh
|-- src/
|   |-- deepeval_eval/
|       |-- caipe.py
|       |-- config.py
|       |-- enterprise_dataset.py
|       |-- enterprise_deepeval.py
|       |-- hotpotqa_dataset.py
|       |-- hotpotqa_deepeval.py
|       |-- io_utils.py
|       |-- llm.py
|       |-- metrics.py
|-- docs/
    |-- architecture.md
    |-- evaluation_pipeline.md
    |-- enterprise_rag_bench.md
    |-- hotpotqa.md
    |-- project_structure.md
    |-- setup_and_usage.md
~~~

Generated local folders are ignored by Git:

~~~text
cache/
data/
results/
~~~

## Quick Start

From the repository root:

~~~powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
~~~

Make sure CAIPE rag-server is running at the default URL:

~~~text
http://localhost:9446
~~~

Create a local .env file or use the CAIPE .env file configured in src/deepeval_eval/config.py.

## Configuration

The evaluation code reads model settings from environment variables or from the configured CAIPE .env file.

| Variable | Required | Purpose |
| --- | --- | --- |
| OPENAI_API_KEY | Yes | API key for the OpenAI compatible Cisco LiteLLM endpoint. |
| OPENAI_ENDPOINT | Yes | Base URL for the LiteLLM endpoint. |
| OPENAI_MODEL_NAME | Yes | Model name passed to answer generation and DeepEval judge calls. |

Default paths and folders are defined in src/deepeval_eval/config.py.

| Setting | Default |
| --- | --- |
| CAIPE rag-server URL | http://localhost:9446 |
| Data directory | data |
| Cache directory | cache |
| Results directory | results |
| Default env file | ~/ai-platform-engineering/.env |

## Wrapper Scripts

Windows users can run the .cmd scripts. Linux and macOS users can run the .sh scripts. Each script sets the repository root from its own location and then calls the correct Python entry point.

Extra CLI options can be appended after the script name. For example, this runs only one EnterpriseRAG-Bench evaluation item:

~~~powershell
.\scripts\eval_enterprise.cmd --max-items 1
~~~

The shell scripts accept the same extra options:

~~~bash
./scripts/eval_enterprise.sh --max-items 1
~~~

## EnterpriseRAG-Bench Evaluation

Run ingestion on Windows:

~~~powershell
.\scripts\ingest_enterprise.cmd
~~~

Run evaluation on Windows:

~~~powershell
.\scripts\eval_enterprise.cmd
~~~

Linux or macOS equivalents:

~~~bash
./scripts/ingest_enterprise.sh
./scripts/eval_enterprise.sh
~~~

Direct Python commands are also supported:

~~~powershell
python src\deepeval_eval\enterprise_deepeval.py ingest --sources confluence jira github hubspot fireflies linear google_drive gmail slack --limit-per-source 1000 --num-questions 10 --questions-per-category 3 --batch-size 50
python src\deepeval_eval\enterprise_deepeval.py eval --max-items 10 --top-k 3 --max-context-chars 6000
~~~

## HotpotQA Evaluation

HotpotQA ingestion expects these preprocessed zip files in cache, with fallback lookup in ~/Downloads:

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

Linux or macOS equivalents:

~~~bash
./scripts/ingest_hotpotqa.sh
./scripts/eval_hotpotqa.sh
~~~

Direct Python commands are also supported:

~~~powershell
python src\deepeval_eval\hotpotqa_deepeval.py ingest --limit 100 --questions-per-category 50 --max-docs 1000 --batch-size 50
python src\deepeval_eval\hotpotqa_deepeval.py eval --max-items 10 --top-k 5 --max-context-chars 12000
~~~

## Outputs and Results

Ingestion writes generated dataset files to data:

| Pipeline | Files |
| --- | --- |
| EnterpriseRAG-Bench | enterprise_deepeval_corpus.jsonl, enterprise_deepeval_corpus.csv, enterprise_deepeval_questions.jsonl, enterprise_deepeval_questions.csv |
| HotpotQA | hotpotqa_deepeval_corpus.jsonl, hotpotqa_deepeval_corpus.csv, hotpotqa_deepeval_questions.jsonl, hotpotqa_deepeval_questions.csv |

Evaluation writes timestamped result files to results:

~~~text
enterprise_deepeval_results_timestamp.json
enterprise_deepeval_results_timestamp.csv
hotpotqa_deepeval_results_timestamp.json
hotpotqa_deepeval_results_timestamp.csv
~~~

## Documentation

Detailed documentation is available in the docs folder:

- [Architecture](docs/architecture.md)
- [Evaluation Pipeline](docs/evaluation_pipeline.md)
- [EnterpriseRAG-Bench](docs/enterprise_rag_bench.md)
- [HotpotQA](docs/hotpotqa.md)
- [Project Structure](docs/project_structure.md)
- [Setup and Usage](docs/setup_and_usage.md)

## Notes

- Generated folders are intentionally excluded from Git.
- Authentication token support exists through CLI options, but token retrieval is not implemented in this repository.
- Full CAIPE agentic chat evaluation is to be confirmed.
