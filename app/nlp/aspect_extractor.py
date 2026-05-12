"""Per-product TF-IDF aspect extraction — Milestone 2 Task 4.

Powers the "Customers mention …" pill row on the product-detail page.
This is the additional functionality we chose for the M2 spec's
Task 4 ("propose and implement at least one additional functionality
that enhances the usability or effectiveness of the system").

How it works:
- Concatenate each product's review texts into one document.
- Build a word-level TF-IDF matrix across all products.
- Expose ``top_terms(product_id, top_n)`` that returns the most
  distinctive words for a given product.

Why TF-IDF over per-product documents?
The IDF is computed over the population of products. Words that
appear across many products (generic praise: "good", "nice",
"great") are weighted down, while product-specific vocabulary
(e.g. "fragrance", "matte", "moisture", "shampoo") rises to the
top. This converts the raw review pile into a human-readable
"Customers mention…" pill row that highlights what is actually
distinctive about each item.

Relationship to the other Milestone 2 tasks:
- Task 1's search uses char-n-gram TF-IDF over product name + category.
- Task 3's recommendations reuse that same matrix.
- Task 2's classifier uses word-level BoW (count vectors, not TF-IDF)
  on review_text and review_title with the Milestone 1 vocabulary.
- THIS task uses a fourth, separate representation: word-level
  TF-IDF over an entirely different corpus (per-product aggregated
  review text). That makes it a genuinely new feature
  representation — the four tasks span char and word features,
  TF-IDF and BoW, product-string and review-text inputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class AspectExtractor:
    """Build a per-product TF-IDF matrix once; expose top-K queries.

    Constructed once at app startup over the seed review corpus
    (the reviews displayed on the live app, not the Nykaa training
    corpus used by the classifier). The constructor work is O(N) on
    the review count, so 500 reviews finish in milliseconds.

    The vectorizer parameters:
    - ``stop_words="english"`` — drop common function words ("the",
      "a", "is", …) that would otherwise dominate TF without
      carrying meaning.
    - ``min_df=2`` — a term must appear in at least 2 products to
      be considered. Filters one-off proper nouns and review-specific
      quirks.
    - ``analyzer="word"`` + ``ngram_range=(1,1)`` — unigrams only;
      bigram aspects ("long lasting") are documented as future work.
    """

    def __init__(
        self,
        reviews_by_product: dict[int, list[dict]],
        min_df: int = 2,
    ) -> None:
        # Stable order so row indices align with self._product_ids.
        self._product_ids = sorted(reviews_by_product.keys())
        # Each product's "document" is the whitespace-joined
        # concatenation of all its review_text fields.
        documents = [
            " ".join(str(r.get("review_text") or "") for r in reviews_by_product[pid])
            for pid in self._product_ids
        ]
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            min_df=min_df,
            stop_words="english",
            lowercase=True,
        )
        # Defensive: if every product had empty reviews, sklearn would
        # raise. Detect the corner case and degrade to "no aspects".
        if any(d.strip() for d in documents):
            self._matrix = self._vectorizer.fit_transform(documents)
            self._feature_names = self._vectorizer.get_feature_names_out()
        else:
            self._matrix = None
            self._feature_names = np.array([])

    def top_terms(self, product_id: int, top_n: int = 5) -> list[str]:
        """Return the top-``top_n`` distinctive TF-IDF terms for ``product_id``.

        Called once per /product/<id> render. The route passes the
        result to the template, which renders each term as a pill in
        the "Customers mention" row.

        Returns ``[]`` if the product has no reviews or no terms
        survive the ``min_df`` floor — the template then hides the
        pill row entirely (cleaner than showing an empty section).
        """
        if self._matrix is None or product_id not in self._product_ids:
            return []
        row = self._product_ids.index(product_id)
        weights = self._matrix.getrow(row).toarray().flatten()
        if not weights.any():
            return []
        top_indices = weights.argsort()[::-1][:top_n]
        return [self._feature_names[i] for i in top_indices if weights[i] > 0]
