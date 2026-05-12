"""Scaffold stub: load the Milestone-I training corpus, print a sample, exit 0.

Real model training lives in the notebook(s) under ``notebooks/`` and
will land alongside Task 2 of the assignment. This script exists so the
training-side import path and data discovery are exercised at scaffold time.
"""

from pathlib import Path

import pandas as pd

from app.nlp.preprocessing import tokenize

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_CSV = _REPO_ROOT / "knowledge" / "cosmetics_beauty_products_reviews.csv"


def main() -> int:
    df = pd.read_csv(_TRAINING_CSV)
    print(f"Loaded {len(df)} rows from knowledge/cosmetics_beauty_products_reviews.csv")
    sample_text = str(df.iloc[0]["review_text"])
    tokens = tokenize(sample_text)
    print(f"Sample tokens: {tokens[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
