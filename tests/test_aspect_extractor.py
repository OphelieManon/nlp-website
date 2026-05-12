"""Unit tests for AspectExtractor."""

from app.nlp.aspect_extractor import AspectExtractor


def _make_reviews_by_product():
    return {
        1: [
            {"review_text": "amazing fragrance, lasts all day, lovely scent"},
            {"review_text": "the fragrance is divine, lasts long"},
            {"review_text": "best perfume, gorgeous smell"},
        ],
        2: [
            {"review_text": "matte finish, lightweight, blends well"},
            {"review_text": "lightweight and matte coverage, great"},
        ],
        3: [
            {"review_text": "shampoo smells lovely and soft"},
        ],
    }


def test_top_terms_returns_n_strings_for_product_with_reviews():
    ax = AspectExtractor(_make_reviews_by_product(), min_df=1)
    terms = ax.top_terms(product_id=1, top_n=3)
    assert isinstance(terms, list)
    assert len(terms) == 3
    assert all(isinstance(t, str) for t in terms)


def test_top_terms_empty_for_unknown_product():
    ax = AspectExtractor(_make_reviews_by_product(), min_df=1)
    assert ax.top_terms(product_id=999) == []


def test_top_terms_excludes_english_stopwords():
    ax = AspectExtractor(_make_reviews_by_product(), min_df=1)
    terms = ax.top_terms(product_id=1, top_n=10)
    for stop in ("the", "and", "is", "a", "all"):
        assert stop not in terms, f"stop-word '{stop}' should not appear"


def test_distinctive_term_outranks_common_term():
    """'fragrance' appears only in product 1; 'lovely' appears in 1 and 3.
    IDF should rank 'fragrance' higher than 'lovely' for product 1."""
    ax = AspectExtractor(_make_reviews_by_product(), min_df=1)
    terms = ax.top_terms(product_id=1, top_n=10)
    # Both should appear in the candidate set for this small fixture
    assert "fragrance" in terms
    if "lovely" in terms:
        assert terms.index("fragrance") < terms.index("lovely")
