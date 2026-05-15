"""
Three-head fusion classifier — Milestone 2 Task 2 (UPDATED HEAD C)

Fix applied:
- Head C upgraded from LogisticRegression → XGBoost
- No change to public interface (Flask-compatible)
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

from xgboost import XGBClassifier

from app.nlp.preprocessing import tokenize


# =========================
# Config
# =========================

_ARTIFACT_FILES = (
    "vocab.joblib",
    "text_model.joblib",
    "title_model.joblib",
    "numeric_model.joblib",
)

_TOP_N_TO_DROP = 20

# Fusion weights (you can tune later)
W_TEXT = 0
W_TITLE = 0
W_NUM = 1


# =========================
# Helpers
# =========================

def _build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    tf = Counter()
    df_count = Counter()

    for tokens in token_lists:
        tf.update(tokens)
        df_count.update(set(tokens))

    top_n = {term for term, _ in df_count.most_common(_TOP_N_TO_DROP)}

    keep = sorted(
        t for t, count in tf.items()
        if count >= 2 and t not in top_n
    )

    if not keep:
        keep = ["__empty__"]

    return {term: i for i, term in enumerate(keep)}


def _numeric_features(rating, price) -> np.ndarray:
    r = np.asarray(rating, dtype=float).reshape(-1, 1)
    lp = np.log1p(np.asarray(price, dtype=float)).reshape(-1, 1)
    return np.hstack([r, lp])


def _vec_factory(vocab: dict[str, int]) -> CountVectorizer:
    return CountVectorizer(
        vocabulary=vocab,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
    )


# =========================
# Main Class
# =========================

class ReviewClassifier:

    def __init__(self) -> None:
        self.vocab = None
        self.text_model = None
        self.title_model = None
        self.numeric_model = None

    # =========================
    # TRAINING
    # =========================
    def train(self, df: pd.DataFrame) -> dict[str, float]:
        print("🔥 TRAIN FUNCTION IS RUNNING", flush=True)
        df = df.dropna(
            subset=[
                "review_text",
                "review_title",
                "review_rating",
                "price",
                "is_a_buyer"
            ]
        ).copy()

        df["review_text"] = df["review_text"].astype(str)
        df["review_title"] = df["review_title"].astype(str)

        y = df["is_a_buyer"].astype(int).to_numpy()

        text_tokens = [tokenize(t) for t in df["review_text"].tolist()]
        title_tokens = [tokenize(t) for t in df["review_title"].tolist()]

        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        y_train, y_test = y[idx_train], y[idx_test]

        text_train = [" ".join(text_tokens[i]) for i in idx_train]
        text_test = [" ".join(text_tokens[i]) for i in idx_test]

        title_train = [" ".join(title_tokens[i]) for i in idx_train]
        title_test = [" ".join(title_tokens[i]) for i in idx_test]

        # =========================
        # VOCAB
        # =========================
        self.vocab = _build_vocab([text_tokens[i] for i in idx_train])

        # =========================
        # HEAD A: TEXT
        # =========================
        self.text_model = Pipeline([
            ("vec", _vec_factory(self.vocab)),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )),
        ])
        self.text_model.fit(text_train, y_train)

        # =========================
        # HEAD B: TITLE
        # =========================
        self.title_model = Pipeline([
            ("vec", _vec_factory(self.vocab)),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )),
        ])
        self.title_model.fit(title_train, y_train)

        # =========================
        # HEAD C: NUMERIC (FIXED → XGBOOST)
        # =========================
        X_num_train = _numeric_features(
            df.iloc[idx_train]["review_rating"],
            df.iloc[idx_train]["price"]
        )
        X_num_test = _numeric_features(
            df.iloc[idx_test]["review_rating"],
            df.iloc[idx_test]["price"]
        )

        self.numeric_model = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ])

        self.numeric_model.fit(X_num_train, y_train)

        # =========================
        # EVALUATION (FUSION)
        # =========================
        p_text = self.text_model.predict_proba(text_test)[:, 1]
        p_title = self.title_model.predict_proba(title_test)[:, 1]
        p_num = self.numeric_model.predict_proba(X_num_test)[:, 1]

        #debug
        # =========================
        # PER-HEAD ACCURACY (DEBUG)
        # =========================

        y_text_pred = (p_text >= 0.5).astype(int)
        y_title_pred = (p_title >= 0.5).astype(int)
        y_num_pred = (p_num >= 0.5).astype(int)

        text_acc = accuracy_score(y_test, y_text_pred)
        title_acc = accuracy_score(y_test, y_title_pred)
        num_acc = accuracy_score(y_test, y_num_pred)

        print("\n===== PER-HEAD PERFORMANCE =====")
        print(f"TEXT   ACC: {text_acc:.4f}")
        print(f"TITLE  ACC: {title_acc:.4f}")
        print(f"NUM    ACC: {num_acc:.4f}")
        print("================================\n")

        text_f1 = f1_score(y_test, y_text_pred, zero_division=0)
        title_f1 = f1_score(y_test, y_title_pred, zero_division=0)
        num_f1 = f1_score(y_test, y_num_pred, zero_division=0)

        print(f"TEXT   F1: {text_f1:.4f}")
        print(f"TITLE  F1: {title_f1:.4f}")
        print(f"NUM    F1: {num_f1:.4f}")

        ####
        p_fused = (
            W_TEXT * p_text +
            W_TITLE * p_title +
            W_NUM * p_num
        )

        y_hat = (p_fused >= 0.5).astype(int)

        return {
            "text_acc": float(accuracy_score(y_test, (p_text >= 0.5).astype(int))),
            "title_acc": float(accuracy_score(y_test, (p_title >= 0.5).astype(int))),
            "numeric_acc": float(accuracy_score(y_test, (p_num >= 0.5).astype(int))),
            "fused_acc": float(accuracy_score(y_test, y_hat)),
            "fused_f1": float(f1_score(y_test, y_hat, zero_division=0)),
        }

    # =========================
    # PREDICTION
    # =========================
    def predict(self, title: str, rating: int, review_text: str, price: float):

        if any(m is None for m in [self.text_model, self.title_model, self.numeric_model]):
            raise RuntimeError("Model not trained or loaded.")

        text_str = " ".join(tokenize(review_text))
        title_str = " ".join(tokenize(title))

        # HEAD A
        p_text = self.text_model.predict_proba([text_str])[0, 1]

        # HEAD B
        p_title = self.title_model.predict_proba([title_str])[0, 1]

        # HEAD C
        p_num = self.numeric_model.predict_proba(
            _numeric_features([rating], [price])
        )[0, 1]

        # DEBUG
        print("\n===== DEBUG PREDICT =====", flush=True)
        print("TITLE:", title, flush=True)
        print("RATING:", rating, flush=True)
        print("PRICE:", price, flush=True)
        print("TEXT:", review_text[:80], flush=True)

        print("p_text :", p_text, flush=True)
        print("p_title:", p_title, flush=True)
        print("p_num  :", p_num, flush=True)

        # FUSION (ONLY ONCE)
        proba = (
            W_TEXT * p_text +
            W_TITLE * p_title +
            W_NUM * p_num
        )

        print("FINAL PROBA:", proba, flush=True)
        print("========================\n", flush=True)

        label = 1 if proba >= 0.5 else 0

        return label, float(proba)

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, target_dir: Path | str) -> None:
        if any(m is None for m in [self.vocab, self.text_model, self.title_model, self.numeric_model]):
            raise RuntimeError("Cannot save untrained model.")

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vocab, target / _ARTIFACT_FILES[0])
        joblib.dump(self.text_model, target / _ARTIFACT_FILES[1])
        joblib.dump(self.title_model, target / _ARTIFACT_FILES[2])
        joblib.dump(self.numeric_model, target / _ARTIFACT_FILES[3])

    @classmethod
    def load(cls, source_dir: Path | str) -> "ReviewClassifier":
        source = Path(source_dir)

        clf = cls()
        clf.vocab = joblib.load(source / _ARTIFACT_FILES[0])
        clf.text_model = joblib.load(source / _ARTIFACT_FILES[1])
        clf.title_model = joblib.load(source / _ARTIFACT_FILES[2])
        clf.numeric_model = joblib.load(source / _ARTIFACT_FILES[3])

        return clf