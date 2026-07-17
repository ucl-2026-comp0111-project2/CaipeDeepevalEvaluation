import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

configs = [
    {"name": "topk3", "top_k": 3, "max_context_chars": 8000},
    {"name": "topk5", "top_k": 5, "max_context_chars": 8000},
    {"name": "topk7", "top_k": 7, "max_context_chars": 8000},
]

for cfg in configs:
    print(f"\n=== Running config: {cfg['name']} ===")
    subprocess.run([
        "python", str(REPO_ROOT / "src/deepeval_eval/hotpotqa_deepeval.py"), "eval",
        "--max-items", "50",
        "--top-k", str(cfg["top_k"]),
        "--max-context-chars", str(cfg["max_context_chars"]),
    ], cwd=REPO_ROOT)
