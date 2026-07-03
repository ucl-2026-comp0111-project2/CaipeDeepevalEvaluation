# CAIPE DeepEval Evaluation

This repository contains DeepEval-based evaluation pipelines for CAIPE RAG over two benchmark datasets:

- EnterpriseRAG-Bench
- HotpotQA

The implementation supports controlled dataset ingestion into CAIPE, generation of evaluation-ready question and corpus files, retrieval quality checks using expected document IDs, answer generation from retrieved context, and metric evaluation with DeepEval.

## Evaluation Flow

~~~text
benchmark question
-> CAIPE rag-server retrieval through /v1/query
-> retrieved context passages
-> answer generation using Cisco LiteLLM
-> DeepEval judging
-> JSON and CSV result files
~~~

This evaluates CAIPE retrieval behaviour and the quality of answers generated from retrieved context. It is not a full CAIPE agentic-chat evaluation.

## Requirements

- CAIPE rag-server running locally at http://localhost:9446
- CAIPE configured with Cisco LiteLLM for embeddings and LLM calls
- Python 3.10+
- Dependencies from requirements.txt
- Local CAIPE .env file available at C:\Users\liana\ai-platform-engineering\.env

The evaluation scripts read LLM settings from the CAIPE .env file by default:

~~~text
OPENAI_API_KEY=...
OPENAI_ENDPOINT=...
OPENAI_MODEL_NAME=...
~~~

Secrets and generated files are intentionally excluded from Git.

## Project Structure

~~~text
caipe_deepeval_evaluation/
+-- README.md
+-- pyproject.toml
+-- requirements.txt
+-- .env.example
+-- .gitignore
+-- scripts/
|   +-- ingest_enterprise.cmd
|   +-- eval_enterprise.cmd
|   +-- ingest_hotpotqa.cmd
|   +-- eval_hotpotqa.cmd
+-- src/
    +-- deepeval_eval/
        +-- __init__.py
        +-- caipe.py
        +-- config.py
        +-- enterprise_dataset.py
        +-- hotpotqa_dataset.py
        +-- io_utils.py
        +-- llm.py
        +-- metrics.py
        +-- enterprise_deepeval.py
        +-- hotpotqa_deepeval.py
~~~

Module responsibilities:

- caipe.py: CAIPE rag-server client and response parsing.
- config.py: local paths, environment loading, and Cisco LiteLLM settings.
- enterprise_dataset.py: EnterpriseRAG-Bench loading, sampling, and CAIPE payload conversion.
- hotpotqa_dataset.py: HotpotQA loading, sampling, and CAIPE payload conversion.
- io_utils.py: shared download and JSONL helpers.
- llm.py: OpenAI-compatible Cisco LiteLLM client and DeepEval judge adapter.
- metrics.py: retrieval checks, short-answer checks, and DeepEval metric setup.

The following folders are generated locally and are not committed:

~~~text
cache/
data/
results/
~~~

## EnterpriseRAG-Bench

src/deepeval_eval/enterprise_deepeval.py implements ingestion and evaluation for EnterpriseRAG-Bench.

Supported source types:

~~~text
confluence jira github hubspot fireflies linear google_drive gmail slack
~~~

The default ingestion command uses a bounded sample to avoid ingesting the full corpus:

- up to 1,000 documents per source
- 10 selected evaluation questions
- up to 3 questions per category
- expected document IDs prioritised during document selection
- batches of 50 documents per ingest request

Run ingestion from the repository root:

~~~powershell
.\scripts\ingest_enterprise.cmd
~~~

Run evaluation from the repository root:

~~~powershell
.\scripts\eval_enterprise.cmd
~~~

Generated dataset files:

~~~text
data/enterprise_deepeval_corpus.jsonl
data/enterprise_deepeval_corpus.csv
data/enterprise_deepeval_questions.jsonl
data/enterprise_deepeval_questions.csv
~~~

Generated result files:

~~~text
results/enterprise_deepeval_results_timestamp.json
results/enterprise_deepeval_results_timestamp.csv
~~~

## HotpotQA

src/deepeval_eval/hotpotqa_deepeval.py implements ingestion and evaluation for HotpotQA.

The HotpotQA pipeline expects preprocessed zip files:

~~~text
cache/hotpotqa_full_questions.jsonl.zip
cache/hotpotqa_full_document_pool.jsonl.zip
~~~

If those files are not present in cache/, the script falls back to C:\Users\liana\Downloads.

The default ingestion command uses:

- 100 selected questions
- up to 1,000 documents
- up to 50 questions per HotpotQA category
- gold supporting documents included where possible
- batches of 50 documents per ingest request

Run ingestion from the repository root:

~~~powershell
.\scripts\ingest_hotpotqa.cmd
~~~

Run evaluation from the repository root:

~~~powershell
.\scripts\eval_hotpotqa.cmd
~~~

Generated dataset files:

~~~text
data/hotpotqa_deepeval_corpus.jsonl
data/hotpotqa_deepeval_corpus.csv
data/hotpotqa_deepeval_questions.jsonl
data/hotpotqa_deepeval_questions.csv
~~~

Generated result files:

~~~text
results/hotpotqa_deepeval_results_timestamp.json
results/hotpotqa_deepeval_results_timestamp.csv
~~~

## Metrics

### Retrieval Checks

Both pipelines calculate retrieval checks from expected document IDs:

| Metric | Meaning |
| --- | --- |
| doc_id_recall | Fraction of expected documents retrieved by CAIPE. |
| doc_id_precision | Fraction of retrieved documents that are expected documents. |

### DeepEval Metrics

Both pipelines run:

| Metric | Meaning |
| --- | --- |
| AnswerRelevancyMetric | Whether the generated answer addresses the question. |
| FaithfulnessMetric | Whether the answer is grounded in the retrieved context. |
| ContextualRelevancyMetric | Whether the retrieved context is relevant to the question. |
| ContextualPrecisionMetric | Whether relevant retrieved context is ranked highly. |
| ContextualRecallMetric | Whether the retrieved context covers the expected answer. |

### HotpotQA Short-Answer Checks

HotpotQA references are often very short, such as yes, no, or an entity name. The HotpotQA pipeline therefore also records:

| Metric | Meaning |
| --- | --- |
| answer_exact_match | Normalised generated answer exactly matches the reference. |
| answer_contains_reference | Normalised generated answer contains the reference answer. |

## Command Reference

The wrapper commands use the shared virtual environment at D:\summer project\caipe-deepeval-venv\Scripts\python.exe.

Run one-item smoke tests from the repository root:

~~~powershell
.\scripts\eval_enterprise.cmd --max-items 1 --top-k 3 --max-context-chars 6000
.\scripts\eval_hotpotqa.cmd --max-items 1 --top-k 5 --max-context-chars 12000
~~~

## Troubleshooting

If evaluation fails with a connection error, check that CAIPE rag-server is running at http://localhost:9446.

If evaluation fails because a question file is missing, rerun ingestion or restore the generated files under data/.

If DeepEval cannot call the judge model, check OPENAI_API_KEY, OPENAI_ENDPOINT, and OPENAI_MODEL_NAME in the CAIPE .env file.

If ingestion is slow, reduce source limits or question limits before scaling back up.
