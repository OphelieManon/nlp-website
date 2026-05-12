"""Train the Milestone 2 three-head review classifier (Task 2).

This is the CLI training entry point. Markers run it once before
launching Flask to produce the four artifacts under ``models/`` that
the webapp loads at startup. Without these artifacts, the
``/product/<id>/review`` route returns 503 because there's no model
to call.

This script reproduces:
  - Milestone 1 Task 1 preprocessing (regex tokenize, lowercase,
    length ≥ 2, NLTK stopwords) — by importing
    ``app/nlp/preprocessing.py:tokenize``.
  - Milestone 1 Task 1 §1.3 vocabulary construction (hapax + top-20
    filters, alphabetical sort) — inlined in ``_build_vocab``.
  - Milestone 1 Task 2 §2.1 Bag-of-Words count vectors using that
    vocabulary — via sklearn's ``CountVectorizer``.
  - Milestone 1 Task 3 Q1 §3.4 LogisticRegression baseline — for
    each of the three heads.
  - Milestone 1 Task 3 Q2 §3.5 title + numeric extensions — by
    adding the title head (B) and numeric head (C).
  - Milestone 2 fusion: arithmetic mean of ``predict_proba``,
    threshold 0.5.

Artifacts written under ``--out``:
  - ``vocab.joblib``         : dict[str, int] - the M1 vocabulary
  - ``text_model.joblib``    : sklearn Pipeline (CountVectorizer + LR)
  - ``title_model.joblib``   : sklearn Pipeline (CountVectorizer + LR)
  - ``numeric_model.joblib`` : sklearn Pipeline (StandardScaler + LR)

The webapp loads these via ``ReviewClassifier.load(models_dir)``.

References:
  knowledge/milestone1_task1.ipynb     - preprocessing + vocab construction
  knowledge/milestone1_task2_3.ipynb   - BoW vectors + LR baseline (Q1)
                                       - title + metadata additions (Q2)
  knowledge/milestone1_spec.pdf        - the formal Task 1 requirements

Usage:
  python -m scripts.train_review_classifier
  python -m scripts.train_review_classifier --corpus path.csv --out models/

The default corpus is ``knowledge/cosmetics_beauty_products_reviews.csv``.
The default output dir is ``models/``. Both are overridable for tests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.nlp.preprocessing import tokenize

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CORPUS = _REPO_ROOT / "knowledge" / "cosmetics_beauty_products_reviews.csv"
_DEFAULT_OUT = _REPO_ROOT / "models"

_RANDOM_STATE = 42
_TOP_N_TO_DROP = 20  # M1 Task 1 step 7: drop top-20 most frequent by doc frequency


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS,
                        help="Path to the training CSV.")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help="Directory to write artifacts into.")
    return parser.parse_args()


def _build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    """Build the M1 vocabulary from a list of tokenized documents.

    Implements Milestone 1 Task 1 steps 6-9:
      - Drop hapax legomena (terms with total term-frequency < 2).
      - Drop the top-20 terms by document frequency.
      - Sort the survivors alphabetically and assign indices from 0.
    """
    tf = Counter()
    df = Counter()
    for tokens in token_lists:
        tf.update(tokens)
        df.update(set(tokens))

    top_n = {term for term, _ in df.most_common(_TOP_N_TO_DROP)}
    keep = sorted(t for t, count in tf.items() if count >= 2 and t not in top_n)
    if not keep:
        # Defensive fallback for tiny synthetic corpora (e.g. test fixtures)
        # where every term gets eaten by the hapax or top-20 filter.
        keep = ["__empty__"]
    return {term: i for i, term in enumerate(keep)}


def _numeric_features(rating, price) -> np.ndarray:
    """Build the 2-column numeric design matrix [rating, log1p(price)]."""
    r = np.asarray(rating, dtype=float).reshape(-1, 1)
    lp = np.log1p(np.asarray(price, dtype=float)).reshape(-1, 1)
    return np.hstack([r, lp])


def _make_vec(vocab: dict[str, int]) -> CountVectorizer:
    """A CountVectorizer that respects pre-tokenized whitespace-joined input."""
    return CountVectorizer(
        vocabulary=vocab, tokenizer=str.split, lowercase=False,
        token_pattern=None,
    )


def _train(corpus_path: Path, out_dir: Path) -> dict[str, float]:
    """Train the three heads and write 4 artifacts. Return test metrics."""
    print(f"Loading corpus: {corpus_path}")
    df = pd.read_csv(corpus_path)
    print(f"Loaded {len(df):,} rows.")

    required = ["review_text", "review_title", "review_rating", "price", "is_a_buyer"]
    before = len(df)
    df = df.dropna(subset=required).copy()
    print(f"Dropped {before - len(df):,} NaN rows; {len(df):,} remain.")
    df["review_text"] = df["review_text"].astype(str)
    df["review_title"] = df["review_title"].astype(str)
    y = df["is_a_buyer"].astype(int).to_numpy()

    print("Tokenizing review_text and review_title (M1 pipeline)...")
    text_tokens = [tokenize(t) for t in df["review_text"].tolist()]
    title_tokens = [tokenize(t) for t in df["review_title"].tolist()]

    # 80/20 stratified split BEFORE building the vocab, so the vocab is
    # derived from training data only (no test-set leakage).
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2,
        random_state=_RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    y_train, y_test = y[idx_train], y[idx_test]

    text_train_tokens = [text_tokens[i] for i in idx_train]
    title_train_tokens = [title_tokens[i] for i in idx_train]
    text_test_tokens = [text_tokens[i] for i in idx_test]
    title_test_tokens = [title_tokens[i] for i in idx_test]

    print("Building vocabulary (hapax + top-20 filters)...")
    vocab = _build_vocab(text_train_tokens)
    print(f"Vocabulary size: {len(vocab):,}")

    text_train_strs = [" ".join(t) for t in text_train_tokens]
    title_train_strs = [" ".join(t) for t in title_train_tokens]
    text_test_strs = [" ".join(t) for t in text_test_tokens]
    title_test_strs = [" ".join(t) for t in title_test_tokens]

    print("Fitting Head A (text BoW + LR)...")
    text_pipe = Pipeline([
        ("vec", _make_vec(vocab)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    text_pipe.fit(text_train_strs, y_train)

    print("Fitting Head B (title BoW + LR)...")
    title_pipe = Pipeline([
        ("vec", _make_vec(vocab)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    title_pipe.fit(title_train_strs, y_train)

    print("Fitting Head C (numeric + LR)...")
    X_num_train = _numeric_features(
        df.iloc[idx_train]["review_rating"], df.iloc[idx_train]["price"]
    )
    X_num_test = _numeric_features(
        df.iloc[idx_test]["review_rating"], df.iloc[idx_test]["price"]
    )
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    numeric_pipe.fit(X_num_train, y_train)

    print("Evaluating on the held-out 20% test split...")
    p_text = text_pipe.predict_proba(text_test_strs)[:, 1]
    p_title = title_pipe.predict_proba(title_test_strs)[:, 1]
    p_num = numeric_pipe.predict_proba(X_num_test)[:, 1]
    p_fused = (p_text + p_title + p_num) / 3.0
    y_hat = (p_fused >= 0.5).astype(int)

    metrics = {
        "text_acc": float(accuracy_score(y_test, (p_text >= 0.5).astype(int))),
        "title_acc": float(accuracy_score(y_test, (p_title >= 0.5).astype(int))),
        "numeric_acc": float(accuracy_score(y_test, (p_num >= 0.5).astype(int))),
        "fused_acc": float(accuracy_score(y_test, y_hat)),
        "fused_f1": float(f1_score(y_test, y_hat, zero_division=0)),
    }
    for k, v in metrics.items():
        print(f"  {k:>12}: {v:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab,        out_dir / "vocab.joblib")
    joblib.dump(text_pipe,    out_dir / "text_model.joblib")
    joblib.dump(title_pipe,   out_dir / "title_model.joblib")
    joblib.dump(numeric_pipe, out_dir / "numeric_model.joblib")
    print(f"Saved 4 artifacts under {out_dir}/")
    return metrics


def main() -> int:
    args = _parse_args()
    _train(args.corpus, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
