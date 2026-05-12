"""Unit tests for SearchIndex.

A 5-row fixture (2 Maybelline + 3 distractors across categories) gives
deterministic assertions for the algorithm. One slow-ish test at the
bottom (test_real_data_typo_finds_maybelline_products) exercises the
real 1000-row catalogue — that's the headline rubric guarantee.
"""

import pandas as pd
import pytest

from app.nlp.data_loader import load_products
from app.nlp.search import SearchIndex


@pytest.fixture
def small_index():
    df = pd.DataFrame(
        [
            {"product_id": 1, "product_name": "Maybelline Volumizing Mascara",   "category": "Makeup",    "price": 18.0, "image_path": ""},
            {"product_id": 2, "product_name": "Maybelline Gentle Body Mist",     "category": "Fragrance", "price": 22.5, "image_path": ""},
            {"product_id": 3, "product_name": "CeraVe Glow Face Wash",           "category": "Skincare",  "price": 88.0, "image_path": ""},
            {"product_id": 4, "product_name": "L'Oréal Repairing Highlighter",   "category": "Makeup",    "price": 93.7, "image_path": ""},
            {"product_id": 5, "product_name": "Estee Lauder Gentle Night Cream", "category": "Skincare",  "price": 33.5, "image_path": ""},
        ]
    )
    return SearchIndex(df)


def _names(results):
    return [r["product_name"] for r in results]


def test_exact_brand_match(small_index):
    names = _names(small_index.query("Maybelline"))
    assert "Maybelline Volumizing Mascara" in names
    assert "Maybelline Gentle Body Mist" in names


def test_typo_brand_match(small_index):
    """The assignment example: 'Maybeline' (one l) must match 'Maybelline' (two ls)."""
    names = _names(small_index.query("Maybeline"))
    assert "Maybelline Volumizing Mascara" in names
    assert "Maybelline Gentle Body Mist" in names


def test_typo_plus_extra_tokens(small_index):
    """'maybeline New York' must still return the Maybelline rows."""
    names = _names(small_index.query("maybeline New York"))
    assert "Maybelline Volumizing Mascara" in names
    assert "Maybelline Gentle Body Mist" in names


def test_category_match(small_index):
    names = _names(small_index.query("Skincare"))
    assert "CeraVe Glow Face Wash" in names
    assert "Estee Lauder Gentle Night Cream" in names


def test_gibberish_returns_empty(small_index):
    assert small_index.query("xyzzyplugh") == []


def test_empty_or_whitespace_query_returns_empty(small_index):
    assert small_index.query("") == []
    assert small_index.query("   ") == []


def test_results_have_score_and_are_sorted_desc(small_index):
    results = small_index.query("Maybelline")
    assert all("score" in r for r in results)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_real_data_typo_finds_maybelline_products():
    """Real 1000-row catalogue: 'Maybeline' (typo) returns ≥3 Maybelline products in top 10."""
    index = SearchIndex(load_products())
    top_10 = index.query("Maybeline")[:10]
    matched = [r for r in top_10 if "Maybelline" in r["product_name"]]
    assert len(matched) >= 3, (
        f"only {len(matched)} Maybelline products in top 10: {_names(top_10)}"
    )


def test_top_n_limits_result_count(small_index):
    """top_n caps the result list, including the edge case top_n=0."""
    assert len(small_index.query("Maybelline", top_n=1)) == 1
    assert small_index.query("Maybelline", top_n=0) == []


def test_similar_returns_top_n_excluding_target(small_index):
    """similar() returns top_n products, none of which is the target itself."""
    results = small_index.similar(product_id=1, top_n=3)
    assert len(results) == 3
    assert all(r["product_id"] != 1 for r in results)


def test_similar_results_have_score_and_sorted_desc(small_index):
    results = small_index.similar(product_id=1, top_n=4)
    assert all("score" in r for r in results)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_similar_unknown_product_id_raises(small_index):
    import pytest as _pytest
    with _pytest.raises(KeyError):
        small_index.similar(product_id=99999)


def test_similar_prefers_same_category_on_tie():
    """Hand-built 4-row index: two products with similar names, one of which
    shares the target's category. The same-category one must rank higher."""
    df = pd.DataFrame([
        {"product_id": 1, "product_name": "Maybelline Mascara",          "category": "Makeup",   "price": 10.0, "image_path": ""},
        {"product_id": 2, "product_name": "Maybelline Mascara Twin",     "category": "Makeup",   "price": 11.0, "image_path": ""},
        {"product_id": 3, "product_name": "Maybelline Mascara Lookalike","category": "Skincare", "price": 12.0, "image_path": ""},
        {"product_id": 4, "product_name": "Estee Lauder Cream",          "category": "Skincare", "price": 30.0, "image_path": ""},
    ])
    idx = SearchIndex(df)
    results = idx.similar(product_id=1, top_n=3)
    # The same-category "Twin" should outrank the cross-category "Lookalike".
    rank = {r["product_id"]: i for i, r in enumerate(results)}
    assert rank[2] < rank[3], f"expected same-category to rank higher: {rank}"
