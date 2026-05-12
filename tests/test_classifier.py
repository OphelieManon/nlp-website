"""Unit tests for ReviewClassifier — a 3-model fusion classifier.

Tests use a synthetic 20-row DataFrame mixing labels so the LogReg heads
have something to learn. Not testing accuracy here — that's the training
script's job. Just verifying the API contract.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.nlp.classifier import ReviewClassifier


@pytest.fixture
def tiny_df():
    """Half-positive / half-negative synthetic corpus."""
    rows = []
    for i in range(10):
        rows.append({
            "review_text": f"absolutely loved it sample {i}",
            "review_title": "amazing product",
            "review_rating": 5,
            "price": 100.0,
            "is_a_buyer": True,
        })
    for i in range(10):
        rows.append({
            "review_text": f"terrible bad waste of money sample {i}",
            "review_title": "do not buy",
            "review_rating": 1,
            "price": 100.0,
            "is_a_buyer": False,
        })
    return pd.DataFrame(rows)


def test_train_returns_metrics_dict(tiny_df):
    clf = ReviewClassifier()
    metrics = clf.train(tiny_df)
    for key in ("text_acc", "title_acc", "numeric_acc", "fused_acc", "fused_f1"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_predict_returns_label_and_proba(tiny_df):
    clf = ReviewClassifier()
    clf.train(tiny_df)
    label, proba = clf.predict("great product", 5, "loved it", 19.99)
    assert label in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_predict_returns_positive_on_strongly_positive_training():
    """18 positive + 2 negative — model should strongly predict positive
    for input that matches the majority pattern."""
    rows = [
        {"review_text": f"good item loved {i}", "review_title": "great",
         "review_rating": 5, "price": 50.0, "is_a_buyer": True}
        for i in range(18)
    ]
    rows += [
        {"review_text": f"bad item awful {i}", "review_title": "terrible",
         "review_rating": 1, "price": 50.0, "is_a_buyer": False}
        for i in range(2)
    ]
    clf = ReviewClassifier()
    clf.train(pd.DataFrame(rows))
    label, _ = clf.predict("loved it good", 5, "great", 50.0)
    assert label == 1


def test_predict_returns_negative_on_strongly_negative_training():
    """18 negative + 2 positive — model should predict negative for
    input that matches the majority pattern."""
    rows = [
        {"review_text": f"bad item awful {i}", "review_title": "terrible",
         "review_rating": 1, "price": 50.0, "is_a_buyer": False}
        for i in range(18)
    ]
    rows += [
        {"review_text": f"good item loved {i}", "review_title": "great",
         "review_rating": 5, "price": 50.0, "is_a_buyer": True}
        for i in range(2)
    ]
    clf = ReviewClassifier()
    clf.train(pd.DataFrame(rows))
    label, _ = clf.predict("terrible awful", 1, "do not buy", 50.0)
    assert label == 0


def test_save_load_round_trip(tiny_df, tmp_path: Path):
    clf = ReviewClassifier()
    clf.train(tiny_df)
    before = clf.predict("good", 5, "great", 99.0)
    clf.save(tmp_path)
    loaded = ReviewClassifier.load(tmp_path)
    after = loaded.predict("good", 5, "great", 99.0)
    assert before[0] == after[0]
    assert abs(before[1] - after[1]) < 1e-9


def test_load_missing_dir_raises_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="train_review_classifier"):
        ReviewClassifier.load(tmp_path)  # empty dir
