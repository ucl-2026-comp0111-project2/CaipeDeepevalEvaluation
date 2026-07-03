# Architecture

This project is a small evaluation harness around CAIPE rag-server. It does not implement CAIPE itself. It prepares benchmark data, sends documents to CAIPE ingestion endpoints, queries CAIPE retrieval, generates answers from retrieved context, and scores the result with DeepEval.

## High-Level Architecture

~~~mermaid
flowchart LR
    subgraph Benchmarks
        A[EnterpriseRAG-Bench]
        B[HotpotQA]
    end

    subgraph DatasetModules
        C[enterprise_dataset.py]
        D[hotpotqa_dataset.py]
    end

    subgraph LocalOutputs
        E[data corpus files]
        F[data question files]
    end

    subgraph CAIPE
        G[rag-server ingest endpoints]
        H[rag-server query endpoint]
    end

    subgraph Evaluation
        I[retrieved contexts]
        J[Cisco LiteLLM answer generation]
        K[DeepEval judge]
        L[metrics.py]
        M[results JSON and CSV]
    end

    A --> C
    B --> D
    C --> E
    C --> F
    D --> E
    D --> F
    E --> G
    F --> H
    H --> I
    I --> J
    J --> K
    I --> L
    K --> L
    L --> M
~~~

## Runtime Components

| Component | File | Responsibility |
| --- | --- | --- |
| Enterprise command entry point | src/deepeval_eval/enterprise_deepeval.py | CLI for EnterpriseRAG-Bench ingestion and evaluation. |
| HotpotQA command entry point | src/deepeval_eval/hotpotqa_deepeval.py | CLI for HotpotQA ingestion and evaluation. |
| CAIPE client | src/deepeval_eval/caipe.py | Wraps rag-server REST calls and extracts retrieved contexts and source metadata. |
| Configuration | src/deepeval_eval/config.py | Defines default paths, environment loading, and LiteLLM setting resolution. |
| LLM adapter | src/deepeval_eval/llm.py | Calls an OpenAI compatible LiteLLM endpoint and adapts it to DeepEval. |
| Shared metrics | src/deepeval_eval/metrics.py | Builds DeepEval metrics and computes document ID and short answer checks. |
| Enterprise dataset logic | src/deepeval_eval/enterprise_dataset.py | Downloads and samples EnterpriseRAG-Bench questions and source slices. |
| HotpotQA dataset logic | src/deepeval_eval/hotpotqa_dataset.py | Reads preprocessed HotpotQA zip files and selects gold documents plus distractors. |
| IO helpers | src/deepeval_eval/io_utils.py | Downloads cached files and reads generated JSONL question files. |

## CAIPE Interaction

The CAIPE client uses these rag-server endpoints:

| Endpoint | Used for |
| --- | --- |
| POST /v1/ingestor/heartbeat | Register the ingestion source and obtain batch limits. |
| POST /v1/datasource | Create or update a datasource record. |
| DELETE /v1/datasource | Reset a datasource when requested. |
| POST /v1/job | Open an ingestion job. |
| POST /v1/ingest | Send document batches into CAIPE. |
| POST /v1/job/{job_id}/increment-document-count | Update CAIPE job document count after each batch. |
| POST /v1/job/{job_id}/increment-progress | Update CAIPE job progress after each batch. |
| PATCH /v1/job/{job_id} | Mark ingestion complete. |
| POST /v1/query | Retrieve contexts for each evaluation question. |

Authentication is optional in the code. If an auth token is supplied, it is sent as a Bearer token. Automatic token fetching is not implemented and is to be confirmed.

## LLM and DeepEval Interaction

The evaluation step uses two model-facing roles:

| Role | Implementation |
| --- | --- |
| Answer generation | OpenAICompatibleClient sends a prompt containing the question and retrieved contexts. |
| DeepEval judge | DeepEvalJudge adapts the same OpenAI compatible client to DeepEval expected model interface. |

Both use the resolved OPENAI_ENDPOINT, OPENAI_API_KEY, and OPENAI_MODEL_NAME values.

## Data Flow Summary

1. Dataset-specific modules build a bounded local corpus and a matching question set.
2. The generated corpus is ingested into CAIPE as a datasource.
3. Evaluation reads generated questions from data.
4. For each question, CAIPE is queried through /v1/query.
5. Retrieved contexts are passed to the LLM to generate an answer.
6. DeepEval metrics and retrieval checks are computed.
7. Results are written to timestamped JSON and CSV files.
