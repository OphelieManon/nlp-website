"""
Three-head fusion classifier — Milestone 2 Task 2 (UPDATED HEAD C)

Fix applied:
- Head C upgraded from LogisticRegression → XGBoost
- No change to public interface (Flask-compatible)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from app.nlp.review_text_title_model.buyer_pipeline import predict as buyer_title_text_predict
from app.nlp.review_text_title_rating_price_model.buyer_pipeline_text_numeric import predict as buyer_text_numeric_predict


# =========================
# Config
# =========================

_ARTIFACT_FILES = (
    "numeric_model.joblib",
)

# Fusion weights (you can tune later)
W_TEXT_NUMERICAL = 0.70  # strongest, most comprehensive
W_NUM            = 0.20  # second strongest (numeric-only is very good)
W_TEXT           = 0.10  # weakest — text+title alone lags behind


# =========================
# Helpers
# =========================

def _numeric_features(rating, price) -> np.ndarray:
    r = np.asarray(rating, dtype=float).reshape(-1, 1)
    lp = np.log1p(np.asarray(price, dtype=float)).reshape(-1, 1)
    return np.hstack([r, lp])


# =========================
# Main Class
# =========================

class ReviewClassifier:

    def __init__(self) -> None:
        self.numeric_model = None

    # =========================
    # TRAINING
    # =========================
    def train(self, df: pd.DataFrame) -> dict[str, float]:
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

        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        y_train, y_test = y[idx_train], y[idx_test]

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
        p_buyer_text_numeric = np.array([
            buyer_text_numeric_predict(
                df.iloc[i]["review_text"],
                df.iloc[i]["review_title"],
                df.iloc[i]["review_rating"],
                df.iloc[i]["price"],
            )[1][1]
            for i in idx_test
        ])
        p_buyer_title_text = np.array([
            buyer_title_text_predict(
                df.iloc[i]["review_text"],
                df.iloc[i]["review_title"],
            )[1][1]
            for i in idx_test
        ])
        p_num = self.numeric_model.predict_proba(X_num_test)[:, 1]

        _w_sum = W_TEXT_NUMERICAL + W_TEXT + W_NUM
        p_fused = (
            W_TEXT_NUMERICAL * p_buyer_text_numeric +
            W_TEXT * p_buyer_title_text +
            W_NUM * p_num
        ) / _w_sum

        y_hat = (p_fused >= 0.5).astype(int)

        return {
            "text_acc": float(accuracy_score(y_test, (p_buyer_text_numeric >= 0.5).astype(int))),
            "title_acc": float(accuracy_score(y_test, (p_buyer_title_text >= 0.5).astype(int))),
            "numeric_acc": float(accuracy_score(y_test, (p_num >= 0.5).astype(int))),
            "fused_acc": float(accuracy_score(y_test, y_hat)),
            "fused_f1": float(f1_score(y_test, y_hat, zero_division=0)),
        }

    # =========================
    # PREDICTION
    # =========================
    def predict(self, title: str, rating: int, review_text: str, price: float):

        if self.numeric_model is None:
            raise RuntimeError("Model not trained or loaded.")

        # HEAD A
        _, _proba_text_numerical = buyer_text_numeric_predict(review_text, title, rating, price)
        p_buyer_text_numeric = float(_proba_text_numerical[1])

        # HEAD B
        _, _proba_text = buyer_title_text_predict(review_text, title)
        p_buyer_title_text = float(_proba_text[1])

        # HEAD C
        p_num = self.numeric_model.predict_proba(
            _numeric_features([rating], [price])
        )[0, 1]

        _w_sum = W_TEXT_NUMERICAL + W_TEXT + W_NUM
        proba = (
            W_TEXT_NUMERICAL * p_buyer_text_numeric +
            W_TEXT * p_buyer_title_text +
            W_NUM * p_num
        ) / _w_sum

        label = 1 if proba >= 0.5 else 0

        print(
            f"[classifier] Head A (text+title+rating+price) : {p_buyer_text_numeric:.4f}\n"
            f"[classifier] Head B (text+title)              : {p_buyer_title_text:.4f}\n"
            f"[classifier] Head C (rating+price)            : {p_num:.4f}\n"
            f"[classifier] Fused                            : {proba:.4f}  →  label={label}"
        )

        return label, float(proba)

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, target_dir: Path | str) -> None:
        if self.numeric_model is None:
            raise RuntimeError("Cannot save untrained model.")

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.numeric_model, target / _ARTIFACT_FILES[0])

    @classmethod
    def load(cls, source_dir: Path | str) -> "ReviewClassifier":
        source = Path(source_dir)

        clf = cls()
        clf.numeric_model = joblib.load(source / _ARTIFACT_FILES[0])

        return clf
