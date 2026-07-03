# Results Summary

These are the latest small-sample results from the local CAIPE DeepEval runs.

## EnterpriseRAG-Bench

Run setup:

- 9 source types
- 1,000 documents per source
- 10 questions
- Cisco LiteLLM for answer generation and DeepEval judging

Average scores from the latest recorded run:

| Metric | Average |
| --- | ---: |
| doc_id_recall | 0.60 |
| doc_id_precision | 0.20 |
| answer_relevancy | 0.80 |
| faithfulness | 0.78 |
| contextual_relevancy | 0.330 |
| contextual_precision | 0.733 |
| contextual_recall | 0.60 |

## HotpotQA

Run setup:

- 100 question sample
- 1,000 ingested documents
- 10 evaluated questions
- Cisco LiteLLM for answer generation and DeepEval judging

Average scores from the latest checked run:

| Metric | Average |
| --- | ---: |
| doc_id_recall | 0.90 |
| doc_id_precision | 0.37 |
| answer_exact_match | 0.50 |
| answer_contains_reference | 0.70 |
| answer_relevancy | 0.90 |
| faithfulness | 0.80 |
| contextual_relevancy | 0.20 |
| contextual_precision | 0.883 |
| contextual_recall | 0.667 |
