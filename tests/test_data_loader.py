"""Unit tests for data_loader."""

from pathlib import Path

import pandas as pd

from app.nlp.data_loader import load_products, load_reviews


def test_load_reviews_injects_empty_title_when_missing():
    """knowledge/reviews.csv has no `title` column; loader must add an empty one
    so downstream code (templates, the review store) can rely on it existing.
    """
    df = load_reviews()
    assert "title" in df.columns
    # Existing seed reviews don't have titles
    assert (df["title"] == "").all()


def test_load_reviews_respects_data_dir_override(tmp_path: Path):
    """When `data_dir` points at a directory containing reviews.csv, the loader
    reads from there instead of the repo's runtime data/."""
    fake = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {"review_id": 999, "product_id": 103, "user_id": "U_test",
             "rating": 5, "review_text": "test", "review_date": "2026-05-12",
             "title": "Test title"},
        ]
    ).to_csv(fake, index=False)
    df = load_reviews(data_dir=tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Test title"
