# CI/CD Quality Gate

Use DeepEval evaluation as a **quality gate** in a CI/CD pipeline. After scoring,
the gate compares aggregated metrics against configured thresholds and exits
non-zero when a hard threshold is missed, so a pipeline can block a merge or
deployment on a quality regression.

---

## How it works

```
run evaluation → gate aggregates metrics → compare against thresholds
                                    │
                          any hard violation? ──yes→ exit 1 → build fails
                                    └──────────────no→ exit 0 → build passes
```

The gate is just a process that returns a non-zero exit code on failure, so it
works in any CI system (GitHub Actions, GitLab CI, Jenkins …) — you
only need to run the command and let the pipeline react to the exit code.



## Components

| File | Purpose |
| --- | --- |
| `src/deepeval_eval/gate.py` | Gate core: aggregates scores, compares thresholds, renders a summary, sets the exit code |
| `gate_thresholds.yaml` | Threshold configuration |

---

## Running locally

Add `--gate` to any `eval` command (works for the `enterprise`, `hotpotqa`, and
`precomputed` pipelines):

```bash
# Precomputed 
python src/deepeval_eval/precomputed_deepeval.py \
  --benchmark hotpotqa --answer-mode generate --max-items 30 --gate

# Full RAG 
python src/deepeval_eval/hotpotqa_deepeval.py eval \
  --max-items 50 --top-k 5 --gate
```

Point at a different config with `--gate-config path/to/config.yaml`.

Re-apply the gate to an existing results file without re-running the evaluation:

```bash
python -m deepeval_eval.gate \
  --results results/hotpotqa_deepeval_results_TIMESTAMP.json \
  --config gate_thresholds.yaml
```

---

## Configuration

`gate_thresholds.yaml` at the repository root:

```yaml
metrics:
  answer_relevancy:  { mean: 0.70, pass_rate: 0.90, severity: soft }
  faithfulness:      { mean: 0.80, pass_rate: 0.90, severity: soft }
  # contextual_relevancy / contextual_precision / contextual_recall ...

retrieval:
  doc_id_recall:     { mean: 0.60, severity: soft }
  doc_id_precision:  { mean: 0.50, severity: soft }

error_tolerance: 0.10
```

| Field | Meaning |
| --- | --- |
| `mean` | The metric's average score must be ≥ this value |
| `pass_rate` | Of the scored cases, the fraction passing the metric threshold must be ≥ this value |
| `severity` | `hard` = fail the build; `soft` = warn only |
| `retrieval.*` | Retrieval metrics — meaningful only when questions carry ground-truth `expected_doc_ids` |
| `error_tolerance` | Max fraction of metric evaluations allowed to error (e.g. LLM timeouts); exceeding it is a hard failure |

**Decision rule:** any **hard** violation fails the gate (exit 1); only soft
violations pass with a warning (exit 0). An error rate above `error_tolerance`,
or an empty result set, is also a hard failure (so a broken or empty run can
never be mistaken for a passing one).
