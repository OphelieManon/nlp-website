"""Smoke test: the training stub runs and prints the expected header line.

Invoked as a subprocess so it exercises the same entry point a grader uses.
"""

import subprocess
import sys


def test_train_script_runs_and_prints_row_count():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_classifier"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr was:\n{result.stderr}"
    assert "Loaded " in result.stdout
