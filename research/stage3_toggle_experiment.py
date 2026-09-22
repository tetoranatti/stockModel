"""Entry point for the Stage 3 toggle experiment."""

import sys

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

from stage3_exp.experiment_runner import run_experiment

if __name__ == "__main__":
    run_experiment()
