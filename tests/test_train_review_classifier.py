"""Smoke test: the train_review_classifier script runs on a tiny CSV and saves artifacts."""

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_train_review_classifier_produces_artifacts(tmp_path: Path):
    """Invoke the script with --corpus and --out overrides on a synthetic tiny CSV."""
    corpus = tmp_path / "tiny_corpus.csv"
    rows = []
    for i in range(20):
        rows.append({
            "review_text": f"loved it a lot sample {i}",
            "review_title": "great product",
            "review_rating": 5,
            "price": 100.0,
            "is_a_buyer": True,
        })
    for i in range(20):
        rows.append({
            "review_text": f"terrible quality waste sample {i}",
            "review_title": "bad product",
            "review_rating": 1,
            "price": 100.0,
            "is_a_buyer": False,
        })
    pd.DataFrame(rows).to_csv(corpus, index=False)

    out = tmp_path / "models"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_review_classifier",
         "--corpus", str(corpus), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "fused_acc" in result.stdout
    assert (out / "text_model.joblib").exists()
    assert (out / "title_model.joblib").exists()
    assert (out / "numeric_model.joblib").exists()
