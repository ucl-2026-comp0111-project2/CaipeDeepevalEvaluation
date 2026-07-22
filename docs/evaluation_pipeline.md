# Evaluation Pipeline

This document describes the full evaluation flow implemented in the repository.

## Pipeline Overview

~~~mermaid
sequenceDiagram
    participant Dataset
    participant LocalData as data folder
    participant CAIPE as CAIPE rag-server
    participant LLM as OpenAI Compatible LLM
    participant Eval as DeepEval
    participant Results as results folder

    Dataset->>LocalData: write corpus and questions
    LocalData->>CAIPE: ingest corpus documents
    LocalData->>CAIPE: send question to /v1/query
    CAIPE-->>LocalData: retrieved contexts and source metadata
    LocalData->>LLM: prompt with question and retrieved contexts
    LLM-->>LocalData: generated answer
    LocalData->>Eval: test case with input, answer, context, expected output
    Eval-->>Results: metric scores and reasons
~~~

## 1. Ingestion Step

Ingestion is implemented separately for each dataset.

| Pipeline | Entry point | Dataset module |
| --- | --- | --- |
| EnterpriseRAG-Bench | enterprise_deepeval.py ingest | enterprise_dataset.py |
| HotpotQA | hotpotqa_deepeval.py ingest | hotpotqa_dataset.py |

The ingestion step:

1. Loads or downloads dataset inputs.
2. Selects a bounded set of evaluation questions.
3. Selects documents to ingest, prioritising expected document IDs when available.
4. Registers an ingestor with CAIPE.
5. Creates or updates a datasource.
6. Opens an ingestion job.
7. Sends documents to /v1/ingest in batches.
8. Writes generated corpus and question files to data.

Generated question files are important because evaluation reads from them later.

## 2. Retrieval Step

Evaluation calls CAIPE rag-server through /v1/query.

The request contains:

| Field | Meaning |
| --- | --- |
| query | Evaluation question text. |
| limit | Maximum number of retrieved results. |
| filters.datasource_id | Datasource filter when a datasource ID is supplied. |

The response is parsed by extract_contexts_and_sources in caipe.py. The parser extracts:

| Output | Meaning |
| --- | --- |
| contexts | Retrieved document text passed to answer generation and DeepEval. |
| sources | Document ID, title, source type, and score metadata used for retrieval checks. |

## 3. LLM Answer Generation

Answer generation is implemented in llm_client.py.

The prompt asks the model to answer using only the retrieved context. EnterpriseRAG-Bench uses make_generation_prompt. HotpotQA uses make_short_answer_prompt because HotpotQA references are often short answers.

The code uses:

| Setting | Source |
| --- | --- |
| OPENAI_ENDPOINT | Environment variable or env file. |
| OPENAI_API_KEY | Environment variable or env file. |
| OPENAI_MODEL_NAME | Environment variable or env file. |

## 4. DeepEval Metrics

metrics.py builds five DeepEval metrics for both datasets.

| Metric | Purpose |
| --- | --- |
| AnswerRelevancyMetric | Checks whether the answer addresses the input question. |
| FaithfulnessMetric | Checks whether the answer is grounded in retrieved context. |
| ContextualRelevancyMetric | Checks whether retrieved context is relevant to the question. |
| ContextualPrecisionMetric | Checks whether relevant context is ranked highly. |
| ContextualRecallMetric | Checks whether retrieved context covers expected output. |

Each metric is configured with:

| Option | Value in code |
| --- | --- |
| threshold | 0.5 |
| include_reason | True |
| async_mode | False |

### Context Window Management & DeepEval Integration

According to the official [DeepEval Metrics Documentation](https://docs.confident-ai.com/docs/metrics-introduction), DeepEval formats `retrieval_context` strings directly into judge LLM prompts (such as for [`FaithfulnessMetric`](https://docs.confident-ai.com/docs/metrics-faithfulness) and [`AnswerRelevancyMetric`](https://docs.confident-ai.com/docs/metrics-answer-relevancy)) via [`LLMTestCase`](https://docs.confident-ai.com/docs/evaluation-test-cases) objects.

* **No Built-In Sliding Window**: DeepEval does not automatically perform sliding-window chunking or multi-step windowing across large context items during metric evaluation.
* **Role of `--max-context-chars`**: To prevent evaluator LLM context window overflow (or token ceiling errors) when processing massive enterprise search results, our client pre-truncates retrieved contexts (`c[:max_context_chars]`) before constructing test cases.
* **When it Impacts Results**: `--max-context-chars` (defaulting to 12,000–16,000 characters / ~3,000–4,000 tokens per chunk) only affects evaluation scores if a single retrieved document chunk exceeds this length. For standard RAG pipelines with typical chunk sizes (~500–2,000 tokens), it serves purely as a protective ceiling without altering metric precision.

## 5. Additional Checks

Both pipelines compute retrieval checks from expected document IDs:

| Check | Formula |
| --- | --- |
| doc_id_recall | Expected document IDs retrieved divided by expected document IDs. |
| doc_id_precision | Expected document IDs retrieved divided by retrieved document IDs. |

HotpotQA also records short answer checks:

| Check | Purpose |
| --- | --- |
| answer_exact_match | Normalized answer equals normalized reference. |
| answer_contains_reference | Normalized answer contains normalized reference. |

## 6. Output Files

Ingestion outputs are written to data.

Evaluation outputs are written to results as timestamped files:

~~~text
enterprise_deepeval_results_timestamp.json
enterprise_deepeval_results_timestamp.csv
hotpotqa_deepeval_results_timestamp.json
hotpotqa_deepeval_results_timestamp.csv
~~~

The JSON files include detailed per-question results and metric reasons. The CSV files include compact score columns for quick inspection.
