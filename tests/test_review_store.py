"""Unit tests for ReviewStore (append-only persistence for new reviews)."""

from pathlib import Path

import pandas as pd

from app.nlp.review_store import ReviewStore


def _knowledge_dir_with_seed(tmp_path: Path) -> Path:
    """Build a stub knowledge/ dir with a minimal reviews.csv (no title column)."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    pd.DataFrame([
        {"review_id": 1, "product_id": 103, "user_id": "U001",
         "rating": 5, "review_text": "Great", "review_date": "2026-04-01"},
        {"review_id": 2, "product_id": 103, "user_id": "U002",
         "rating": 4, "review_text": "OK", "review_date": "2026-04-02"},
    ]).to_csv(kdir / "reviews.csv", index=False)
    return kdir


def test_append_to_fresh_store_creates_data_csv(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    kdir = _knowledge_dir_with_seed(tmp_path)
    store = ReviewStore(data_dir=data, knowledge_dir=kdir)
    new = store.append(
        product_id=103, rating=5, review_text="loved it", title="amazing",
        predicted_label=1,
    )
    assert (data / "reviews.csv").exists()
    df = pd.read_csv(data / "reviews.csv")
    # Seed (2 rows) + the new one
    assert len(df) == 3
    assert "title" in df.columns
    assert "predicted_label" in df.columns
    assert new["product_id"] == 103
    assert new["title"] == "amazing"


def test_next_review_id_is_one_plus_max(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    kdir = _knowledge_dir_with_seed(tmp_path)
    store = ReviewStore(data_dir=data, knowledge_dir=kdir)
    # Seed has max review_id = 2
    assert store.next_review_id() == 3


def test_next_user_id_increments_per_call(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    kdir = _knowledge_dir_with_seed(tmp_path)
    store = ReviewStore(data_dir=data, knowledge_dir=kdir)
    first = store.next_user_id()
    second = store.next_user_id()
    assert first != second
    assert first.startswith("U_guest_")
    assert second.startswith("U_guest_")


def test_user_counter_survives_recreation(tmp_path: Path):
    """A second ReviewStore against the same data_dir must NOT reset U_guest_ to 1."""
    data = tmp_path / "data"
    data.mkdir()
    kdir = _knowledge_dir_with_seed(tmp_path)

    store1 = ReviewStore(data_dir=data, knowledge_dir=kdir)
    store1.append(product_id=103, rating=5, review_text="ok", title="t1", predicted_label=1)
    store1.append(product_id=103, rating=4, review_text="ok", title="t2", predicted_label=1)

    # Simulate process restart by building a fresh store
    store2 = ReviewStore(data_dir=data, knowledge_dir=kdir)
    third = store2.append(product_id=103, rating=3, review_text="ok", title="t3", predicted_label=0)

    # All three guest user_ids must be distinct
    df = pd.read_csv(data / "reviews.csv")
    guest_rows = df[df["user_id"].astype(str).str.startswith("U_guest_")]
    assert len(set(guest_rows["user_id"])) == 3
    # The third one (post-restart) must be U_guest_3, not U_guest_1
    assert third["user_id"] == "U_guest_3"


def test_append_returns_dict_with_autofilled_fields(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    kdir = _knowledge_dir_with_seed(tmp_path)
    store = ReviewStore(data_dir=data, knowledge_dir=kdir)
    new = store.append(
        product_id=103, rating=3, review_text="meh", title="so-so",
        predicted_label=0,
    )
    assert new["review_id"] == 3
    assert new["user_id"].startswith("U_guest_")
    assert new["review_date"]  # non-empty ISO date
    assert new["predicted_label"] == 0
