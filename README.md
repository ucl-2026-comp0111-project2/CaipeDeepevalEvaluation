# CAIPE DeepEval Evaluation

DeepEval evaluation scripts for CAIPE RAG using two benchmark datasets:

- EnterpriseRAG-Bench
- HotpotQA

This repository is the DeepEval counterpart to the RAGAS evaluation work. It uses actual CAIPE rag-server ingestion, queries CAIPE for retrieved context, generates answers from retrieved context, and evaluates the outputs with DeepEval metrics.

## Scope

- EnterpriseRAG-Bench sample ingestion and evaluation.
- HotpotQA preprocessed document/question ingestion and evaluation.
- Retrieval checks using expected document IDs.
- Short-answer checks for HotpotQA, where references are often very short.

## Requirements

- CAIPE rag-server running locally at http://localhost:9446.
- Cisco LiteLLM/OpenAI-compatible settings available in the CAIPE env file.
- Python environment with deepeval, requests, httpx, and pydantic.

The scripts default to reading LLM settings from C:/Users/liana/ai-platform-engineering/.env. This can be overridden with --env-file.

## Repository Layout

`	ext
caipe_deepeval_evaluation/
  README.md
  pyproject.toml
  requirements.txt
  .env.example
  scripts/
    ingest_enterprise.cmd
    eval_enterprise.cmd
    ingest_hotpotqa.cmd
    eval_hotpotqa.cmd
  src/deepeval_eval/
    enterprise_deepeval.py
    hotpotqa_deepeval.py
`

Generated cache, data, and results folders are intentionally gitignored.

## EnterpriseRAG-Bench

Ingest a controlled sample:

`powershell
cd 