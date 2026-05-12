"""Pytest fixtures for webapp tests."""

from pathlib import Path

import pandas as pd
import pytest

from app.nlp.classifier import ReviewClassifier
from app import create_app


@pytest.fixture(scope="session")
def trained_classifier():
    """A tiny ReviewClassifier trained on synthetic data, shared across the whole session.

    Tests don't care about classifier accuracy — only that calling .predict()
    returns a valid (label, proba) tuple so the routes can render correctly.
    """
    rows = []
    for i in range(15):
        rows.append({
            "review_text": f"loved this {i}",
            "review_title": "great product",
            "review_rating": 5, "price": 50.0, "is_a_buyer": True,
        })
    for i in range(15):
        rows.append({
            "review_text": f"terrible quality {i}",
            "review_title": "bad",
            "review_rating": 1, "price": 50.0, "is_a_buyer": False,
        })
    clf = ReviewClassifier()
    clf.train(pd.DataFrame(rows))
    return clf


@pytest.fixture
def client(trained_classifier, tmp_path: Path):
    """Function-scoped so review-write tests get an isolated data_dir per test."""
    (tmp_path / "data").mkdir()
    app = create_app(classifier=trained_classifier, data_dir=tmp_path / "data")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
