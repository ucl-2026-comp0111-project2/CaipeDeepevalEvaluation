import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_FILE = Path(__file__).resolve().parent / "batch_configs.json"

with open(CONFIGS_FILE) as f:
    configs = json.load(f)

for cfg in configs:
    print(f"\n=== Running config: {cfg['name']} ===")
    subprocess.run(
        [
            "python",
            str(REPO_ROOT / "src/deepeval_eval/deepeval_evaluator.py"),
            "eval",
            "--dataset-name",
            "hotpotqa",
            "--oracle-testing",
            "--max-items",
            "14",
            "--max-context-chars",
            str(cfg["max_context_chars"]),
            "--answer-mode",
            cfg["answer_mode"],
        ],
        cwd=REPO_ROOT,
    )
