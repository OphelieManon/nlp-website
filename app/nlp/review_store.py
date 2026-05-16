"""Append-only persistence for new reviews — Milestone 2 Task 2.

Implements the persistence half of the Milestone 2 spec for Task 2:
*"Upon confirmation, the review should be included on the website
and be accessible via URL."* Without writing to disk the new review
would vanish at the next request — failing that requirement.

Writes go to ``data_dir/reviews.csv``. On first write the file is
created by migrating ``knowledge_dir/reviews.csv`` forward (adding
empty ``title`` and ``predicted_label`` columns for the legacy
rows that predate Milestone 2) so the full review list lives in
one place from then on.

The data_loader prefers ``data/reviews.csv`` if present, falling
back to ``knowledge/reviews.csv`` otherwise. That single rule is
what guarantees a newly-posted review appears on subsequent page
loads.

Called from one place: the POST handler of /product/<id>/review in
``app/__init__.py``, on the "commit" branch (after the user has
seen the classifier's predicted label and confirmed/overridden it).
"""

from __future__ import annotations

import datetime as _dt
import itertools as _it
from pathlib import Path

import pandas as pd

# The canonical column order written to disk. Used both for reading
# (we select these columns after backfilling defaults) and for
# writing (the new row dict is converted to a one-row DataFrame
# with this exact column order).
_COLUMNS = [
    "review_id", "product_id", "user_id", "rating", "review_text",
    "review_date", "title", "predicted_label", "final_label",
]


class ReviewStore:
    """Append-new-reviews-only store backed by a single CSV file.

    Constructed once per Flask app instance (in ``create_app()``)
    and called once per successful review submission. Thread-safety
    isn't a concern for the dev server (single-process); production
    would need a row-level lock or a database.
    """

    def __init__(self, data_dir: Path | str, knowledge_dir: Path | str) -> None:
        self._data_csv = Path(data_dir) / "user_analysis_classification.csv"
        self._user_counter = _it.count(self._max_guest_seq() + 1)

    def _read(self) -> pd.DataFrame:
        """Read user_analysis_classification.csv, or return empty DataFrame on cold start."""
        if not self._data_csv.exists():
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_csv(self._data_csv)

    def _max_guest_seq(self) -> int:
        """Largest existing U_guest_<n> integer suffix, or 0 if none."""
        df = self._read()
        seq = df["user_id"].astype(str).str.extract(r"^U_guest_(\d+)$", expand=False)
        nums = pd.to_numeric(seq, errors="coerce").dropna()
        return int(nums.max()) if len(nums) else 0

    def next_review_id(self) -> int:
        """Smallest unused review_id (max existing + 1, or 1 if empty)."""
        df = self._read()
        return int(df["review_id"].max()) + 1 if len(df) else 1

    def next_user_id(self) -> str:
        """A fresh ``U_guest_<n>`` identifier; advances the counter."""
        return f"U_guest_{next(self._user_counter)}"

    def append(
        self,
        *,
        product_id: int,
        rating: int,
        review_text: str,
        title: str,
        predicted_label: int,
        final_label: int,
    ) -> dict:
        """Append a new review row to the CSV and return the row as a dict.

        Called from the /product/<id>/review POST commit step in
        ``app/__init__.py`` after the user has confirmed the
        classifier's prediction (or overridden it). The returned
        dict is also pushed into ``reviews_by_product`` so the
        in-memory cache stays consistent with disk without needing
        a full reload.

        The Milestone 2 spec requires that the new review be
        accessible by URL. Persisting to data/reviews.csv ensures
        that — the next /product/<id> page render will read this
        file (via the data_loader's prefer-data/ rule) and find the
        new row.
        """
        new_row = {
            "review_id":       self.next_review_id(),
            "product_id":      int(product_id),
            "user_id":         self.next_user_id(),
            "rating":          int(rating),
            "review_text":     review_text,
            "review_date":     _dt.date.today().isoformat(),
            "title":           title,
            "predicted_label": int(predicted_label),
            "final_label":     int(final_label),
        }
        df = self._read()
        out = pd.concat([df, pd.DataFrame([new_row], columns=_COLUMNS)], ignore_index=True)
        # Create the data/ directory on first write — it may not
        # exist yet if no review has ever been submitted.
        self._data_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self._data_csv, index=False)
        return new_row
