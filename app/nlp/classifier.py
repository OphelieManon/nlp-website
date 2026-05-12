"""Three-head fusion classifier — Milestone 2 Task 2.

The model behind the "Would buy / Would not buy" prediction shown to
the user after they submit a review. The Milestone 2 spec for Task 2:
*"using the review description (and/or other information), the
website should generate a binary label to predict whether the
customer would buy or not buy the item."*

Public interface (consumed by app/__init__.py:review_form):
  - ReviewClassifier()
  - clf.train(df)              -> dict of test-set metrics
  - clf.predict(title, rating, text, price) -> (label, fused_proba)
  - clf.save(target_dir)
  - ReviewClassifier.load(source_dir) -> ReviewClassifier

Internal architecture (aligned with Milestone 1):
  Head A: CountVectorizer(M1 vocab) + LogisticRegression on review_text
  Head B: CountVectorizer(M1 vocab) + LogisticRegression on review_title
  Head C: StandardScaler + LogisticRegression on [rating, log1p(price)]
  Fusion: arithmetic mean of predict_proba, threshold 0.5

This satisfies the Milestone 2 DI/HD criterion verbatim:
*"at least two/three different models, which use different type of
data, must be built and fused for final result"* — three LRs, three
data types (free text, short text, numeric), fused.

References to Milestone 1:
  knowledge/milestone1_task1.ipynb   - preprocessing -> app/nlp/preprocessing.py:tokenize()
  knowledge/milestone1_task2_3.ipynb - BoW + LR baseline (Q1, §3.4),
                                       title + metadata extension (Q2, §3.5)
"""

from __future__ import annotations

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

_ARTIFACT_FILES = (
    "vocab.joblib",
    "text_model.joblib",
    "title_model.joblib",
    "numeric_model.joblib",
)
_TOP_N_TO_DROP = 20


def _build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    """Mirror scripts/train_review_classifier.py:_build_vocab.

    Hapax + top-20 filters per Milestone 1 Task 1 steps 6-7, alphabetical
    sort, indices from 0. Falls back to a sentinel token if the corpus
    is too small for any term to survive the filters (test fixtures).
    """
    tf = Counter()
    df_count = Counter()
    for tokens in token_lists:
        tf.update(tokens)
        df_count.update(set(tokens))
    top_n = {term for term, _ in df_count.most_common(_TOP_N_TO_DROP)}
    keep = sorted(t for t, count in tf.items() if count >= 2 and t not in top_n)
    if not keep:
        keep = ["__empty__"]
    return {term: i for i, term in enumerate(keep)}


def _numeric_features(rating, price) -> np.ndarray:
    """[rating, log1p(price)] as a 2-column float matrix."""
    r = np.asarray(rating, dtype=float).reshape(-1, 1)
    lp = np.log1p(np.asarray(price, dtype=float)).reshape(-1, 1)
    return np.hstack([r, lp])


def _vec_factory(vocab: dict[str, int]) -> CountVectorizer:
    """A CountVectorizer that respects pre-tokenized whitespace-joined input."""
    return CountVectorizer(
        vocabulary=vocab, tokenizer=str.split, lowercase=False,
        token_pattern=None,
    )


class ReviewClassifier:
    """Three-head fusion classifier with the public interface used by Flask."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] | None = None
        self.text_model: Pipeline | None = None
        self.title_model: Pipeline | None = None
        self.numeric_model: Pipeline | None = None

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Fit all three heads. ``df`` must have columns
        review_text, review_title, review_rating, price, is_a_buyer.

        Returns test-set metrics on a held-out 20% stratified split.
        """
        df = df.dropna(
            subset=["review_text", "review_title", "review_rating",
                    "price", "is_a_buyer"]
        ).copy()
        df["review_text"] = df["review_text"].astype(str)
        df["review_title"] = df["review_title"].astype(str)
        y = df["is_a_buyer"].astype(int).to_numpy()

        text_tokens = [tokenize(t) for t in df["review_text"].tolist()]
        title_tokens = [tokenize(t) for t in df["review_title"].tolist()]

        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        y_train, y_test = y[idx_train], y[idx_test]

        text_train_strs = [" ".join(text_tokens[i]) for i in idx_train]
        text_test_strs = [" ".join(text_tokens[i]) for i in idx_test]
        title_train_strs = [" ".join(title_tokens[i]) for i in idx_train]
        title_test_strs = [" ".join(title_tokens[i]) for i in idx_test]

        self.vocab = _build_vocab([text_tokens[i] for i in idx_train])

        self.text_model = Pipeline([
            ("vec", _vec_factory(self.vocab)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.text_model.fit(text_train_strs, y_train)

        self.title_model = Pipeline([
            ("vec", _vec_factory(self.vocab)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.title_model.fit(title_train_strs, y_train)

        X_num_train = _numeric_features(
            df.iloc[idx_train]["review_rating"], df.iloc[idx_train]["price"]
        )
        X_num_test = _numeric_features(
            df.iloc[idx_test]["review_rating"], df.iloc[idx_test]["price"]
        )
        self.numeric_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.numeric_model.fit(X_num_train, y_train)

        p_text = self.text_model.predict_proba(text_test_strs)[:, 1]
        p_title = self.title_model.predict_proba(title_test_strs)[:, 1]
        p_num = self.numeric_model.predict_proba(X_num_test)[:, 1]
        p_fused = (p_text + p_title + p_num) / 3.0
        y_hat = (p_fused >= 0.5).astype(int)

        return {
            "text_acc":    float(accuracy_score(y_test, (p_text >= 0.5).astype(int))),
            "title_acc":   float(accuracy_score(y_test, (p_title >= 0.5).astype(int))),
            "numeric_acc": float(accuracy_score(y_test, (p_num >= 0.5).astype(int))),
            "fused_acc":   float(accuracy_score(y_test, y_hat)),
            "fused_f1":    float(f1_score(y_test, y_hat, zero_division=0)),
        }

    def predict(self, title: str, rating: int, review_text: str,
                price: float) -> tuple[int, float]:
        """Return (label in {0,1}, fused_probability_of_buy)."""
        if (self.text_model is None or self.title_model is None
                or self.numeric_model is None):
            raise RuntimeError(
                "ReviewClassifier not trained; call .train() or .load() first."
            )
        text_str = " ".join(tokenize(review_text))
        title_str = " ".join(tokenize(title))
        p_text = self.text_model.predict_proba([text_str])[0, 1]
        p_title = self.title_model.predict_proba([title_str])[0, 1]
        p_num = self.numeric_model.predict_proba(
            _numeric_features([rating], [price])
        )[0, 1]
        proba = float((p_text + p_title + p_num) / 3.0)
        return (1 if proba >= 0.5 else 0, proba)

    def save(self, target_dir: Path | str) -> None:
        if (self.vocab is None or self.text_model is None
                or self.title_model is None or self.numeric_model is None):
            raise RuntimeError("Cannot save an untrained classifier.")
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vocab,         target / _ARTIFACT_FILES[0])
        joblib.dump(self.text_model,    target / _ARTIFACT_FILES[1])
        joblib.dump(self.title_model,   target / _ARTIFACT_FILES[2])
        joblib.dump(self.numeric_model, target / _ARTIFACT_FILES[3])

    @classmethod
    def load(cls, source_dir: Path | str) -> "ReviewClassifier":
        source = Path(source_dir)
        missing = [f for f in _ARTIFACT_FILES if not (source / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifact(s) {missing} in {source}. "
                f"Run: python -m scripts.train_review_classifier"
            )
        clf = cls()
        clf.vocab         = joblib.load(source / _ARTIFACT_FILES[0])
        clf.text_model    = joblib.load(source / _ARTIFACT_FILES[1])
        clf.title_model   = joblib.load(source / _ARTIFACT_FILES[2])
        clf.numeric_model = joblib.load(source / _ARTIFACT_FILES[3])
        return clf
