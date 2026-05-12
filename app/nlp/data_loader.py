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


def load_reviews(data_dir: Path | None = None) -> pd.DataFrame:
    """Load reviews.csv (seed + any posted-via-the-form reviews).

    Injects an empty ``title`` column if the source CSV doesn't
    have one — the seed `knowledge/reviews.csv` predates the title
    field added in Milestone 2, so we normalise the schema at read
    time. This means downstream code (templates, AspectExtractor,
    ReviewStore) can rely on the title column always existing.
    """
    df = pd.read_csv(_resolve("reviews.csv", data_dir))
    if "title" not in df.columns:
        df["title"] = ""
    return df
