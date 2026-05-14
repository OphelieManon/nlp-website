"""CSV loaders for the catalogue and seed-review data.

Used by:
- ``app/__init__.py``    — loads products + reviews at startup so the
                           HTTP routes can serve them.
- ``scripts/*.py``       — same loaders, same paths.
- ``notebooks/_build.py``— the Milestone 2 notebook loads directly
                           from KNOWLEDGE_DIR rather than via this
                           helper (so the marker sees the file path
                           explicitly).

Path resolution rule: prefer the runtime ``data/`` directory if a
matching file exists there, otherwise fall back to the immutable
``knowledge/`` directory.

The reason: ``knowledge/`` is the assignment-supplied source-of-truth
data and must never be modified. When the user posts a new review
through the Task 2 form, ``ReviewStore.append`` writes the updated
``reviews.csv`` into ``data/`` — and the next load picks that file
up automatically so the new review survives across requests
(spec requirement: "Upon confirmation, the review should be included
on the website and be accessible via URL").
"""

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_KNOWLEDGE_DIR = _REPO_ROOT / "knowledge"


def _resolve(filename: str, data_dir: Path | None = None) -> Path:
    """Return the runtime path if populated, else the knowledge fallback.

    ``data_dir`` lets tests pass a per-test ``tmp_path`` so the
    write-then-read cycle for the review-submission tests stays
    isolated.
    """
    runtime = (data_dir or _DATA_DIR) / filename
    if runtime.exists():
        return runtime
    return _KNOWLEDGE_DIR / filename


def load_products(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the 1,000-product display catalogue.

    Columns: ``product_id, product_name, category, price, image_path``.
    This is the data the Flask app shows to users. Search (Task 1),
    recommendations (Task 3), and aspect extraction (Task 4) all
    operate over these rows.
    """
    return pd.read_csv(_resolve("products.csv", data_dir))


_USER_REVIEWS_COLUMNS = [
    "review_id", "product_id", "user_id", "rating", "review_text", "title", "predicted_label", "final_label",
]


def load_user_reviews(data_dir: Path | None = None) -> pd.DataFrame:
    """Load user-submitted reviews from data/user_analysis_classification.csv.

    Creates the file with correct headers on cold start if it doesn't exist,
    so admin pages work before any review has been submitted.
    """
    path = (data_dir or _DATA_DIR) / "user_analysis_classification.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_USER_REVIEWS_COLUMNS).to_csv(path, index=False)
    return pd.read_csv(path)


def load_product_stats() -> pd.DataFrame:
    """Return per-product rating stats from the full cosmetics dataset.

    Columns returned: product_id, avg_product_rating, product_rating_count.
    Each product appears once (metadata is the same across all review rows
    for a given product, so we deduplicate by product_id).
    """
    path = _KNOWLEDGE_DIR / "cosmetics_beauty_products_reviews.csv"
    df = pd.read_csv(path, usecols=["product_id", "avg_product_rating", "product_rating_count"])
    return df.drop_duplicates(subset="product_id").reset_index(drop=True)


def load_reviews(data_dir):
    df1 = pd.read_csv(data_dir / "reviews.csv")

    new_path = data_dir / "user_analysis_classification.csv"
    if new_path.exists():
        df2 = pd.read_csv(new_path)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1

    return df
