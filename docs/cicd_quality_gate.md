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

---

## Pytest Integration for CI/CD Gatekeeping

You can run quality gate checks as part of your `pytest` suite in CI/CD pipelines to ensure code changes pass quality thresholds before merging.

### 1. Running Unit & Integration Gate Tests

Run the full pytest suite (including gate and metric tests):

```bash
uv run pytest
```

Run only gate and evaluator tests:

```bash
uv run pytest tests/test_gate.py tests/test_eval_engine.py -v
```

### 2. Writing Custom Pytest Quality Gate Tests

You can write pytest test cases in your repository using the `evaluate_gate` function to enforce threshold policies programmatically:

```python
import json
from pathlib import Path
import pytest
from deepeval_eval.engine.gate import evaluate_gate, load_thresholds

def test_evaluation_quality_gate():
    # Load evaluation result artifact from previous step or mock payload
    results_path = Path("results/latest_eval_results.json")
    if not results_path.exists():
        pytest.skip("Evaluation results file not found.")

    with open(results_path) as f:
        data = json.load(f)

    # Load gate thresholds configuration
    config = load_thresholds(Path("gate_thresholds.yaml"))

    # Evaluate gate
    report = evaluate_gate(data["results"], config)

    # Fail pytest test case if any hard violations exist
    assert report.passed, f"Quality gate failed with hard violations: {report.hard_violations}"
```

---

## GitHub Actions CI/CD Integration Example

Here is a complete `.github/workflows/quality_gate.yml` workflow example that runs unit tests via `pytest` and evaluates benchmark RAG quality gates on every Pull Request:

```yaml
name: CI/CD Quality Gate & Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-and-gate:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install uv and dependencies
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          uv venv
          uv pip install -e .[dev]

      - name: Step 1 - Syntax & Pytest Unit Tests
        run: |
          uv run pytest -v --cov=src

      - name: Step 2 - Precomputed Evaluation & Quality Gate
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_ENDPOINT: ${{ secrets.OPENAI_ENDPOINT }}
          OPENAI_MODEL_NAME: "gpt-4o"
        run: |
          uv run python src/deepeval_eval/engine/deepeval_evaluator.py eval \
            --benchmark hotpotqa \
            --precompute \
            --answer-mode generate \
            --max-items 20 \
            --gate \
            --gate-config gate_thresholds.yaml
```

