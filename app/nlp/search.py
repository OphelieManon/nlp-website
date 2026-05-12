"""Search + recommendation index — Milestone 2 Tasks 1 and 3.

This single class powers BOTH Milestone 2 Task 1 (typo-tolerant item
search) and Task 3 (similar-item recommendations). They share the
TF-IDF matrix because the underlying question is the same:
"how close are two product strings in vector space?".

The vectorizer, the sparse matrix, and the DataFrame must stay paired
so their row indices line up. Keeping them in one object makes that
invariant local and obvious: build once, query many times.

----------------------------------------------------------------------
Why character n-grams (`char_wb`, sizes 3-5) instead of word tokens?
----------------------------------------------------------------------
The assignment example requires "Maybeline" (one l) to return the same
results as "Maybelline" (two ls). Word-level matching can't bridge
that — they're different tokens. But character 3-grams overlap heavily
between the two spellings (may, ayb, ybe, bel, …), so cosine
similarity stays high. `char_wb` keeps n-grams within word boundaries,
which avoids artificial matches that span spaces.

This is the M2 Task 1 spec's *"search keyword strings in similar
forms"* requirement.

----------------------------------------------------------------------
Why the same matrix powers Task 3
----------------------------------------------------------------------
Task 3 asks for similar items given a product. The most informative
features about a product in our catalogue are its name and category,
which is exactly what the search index already vectorises. Reusing
the matrix is DRY and ensures search and recommendations agree on
what "close" means.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_MAX_QUERY_LEN = 200


class SearchIndex:
    """Character-n-gram TF-IDF index for product search + similarity.

    Public API used by app/__init__.py:
      - ``query(q, top_n, threshold)`` — Task 1 ranked search.
      - ``similar(product_id, top_n)`` — Task 3 recommendations.

    Usage:
        index = SearchIndex(load_products())
        results = index.query("Maybeline")          # list[dict] with "score" added
        sims    = index.similar(101, top_n=6)
    """

    def __init__(self, products: pd.DataFrame) -> None:
        # reset_index so iloc-based lookups line up with the matrix
        # row order.
        self._products = products.reset_index(drop=True)
        # Document text for each row: "<name> <category>". Both
        # signals matter — name is the primary match but category
        # disambiguates brand-shared products across surfaces.
        documents = [
            f"{row.product_name} {row.category}"
            for row in self._products.itertuples(index=False)
        ]
        # char_wb: character n-grams restricted to within word
        # boundaries (no n-grams spanning a space). ngram_range=(3,5)
        # gives a useful range — trigrams catch most typos, 4- and
        # 5-grams discriminate between similar-looking brands.
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,  # keep even rare features — the catalogue is small.
        )
        self._matrix = self._vectorizer.fit_transform(documents)

    def query(
        self,
        q: str,
        top_n: int = 30,
        threshold: float = 0.05,
    ) -> list[dict]:
        """Milestone 2 Task 1 entry point: ranked keyword search.

        Return up to ``top_n`` product dicts above ``threshold``,
        ranked descending by cosine similarity to ``q``.

        Each dict is the original DataFrame row plus a float
        ``"score"`` key. Empty / whitespace-only queries return ``[]``
        (the route then renders "0 results matched"). Queries longer
        than ``_MAX_QUERY_LEN`` are truncated silently to protect
        against pathological input.

        The 0.05 threshold filters totally unrelated queries (e.g.
        "asdf") so the UI doesn't show garbage results.
        """
        q = (q or "").strip()[:_MAX_QUERY_LEN]
        if not q:
            return []
        q_vec = self._vectorizer.transform([q])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        ranked = scores.argsort()[::-1]
        out: list[dict] = []
        for i in ranked:
            score = float(scores[i])
            if score < threshold:
                break
            if len(out) >= top_n:
                break
            row = self._products.iloc[int(i)].to_dict()
            row["score"] = score
            out.append(row)
        return out

    def similar(self, product_id: int, top_n: int = 6) -> list[dict]:
        """Milestone 2 Task 3 entry point: similar-item recommendations.

        Return up to ``top_n`` products most similar to ``product_id``,
        excluding the target itself.

        Similarity = cosine over the existing char-n-gram TF-IDF
        matrix, plus a 0.1 boost when the candidate shares the
        target's category. The boost is small enough that strong
        textual matches still win, but large enough to break ties
        in favour of same-category items (someone looking at a
        Maybelline mascara probably wants other makeup, not a random
        skincare product with overlapping n-grams).

        Raises ``KeyError`` if ``product_id`` is not in the index.
        """
        mask = self._products["product_id"] == product_id
        if not mask.any():
            raise KeyError(f"product_id {product_id} not in index")
        target_pos = int(mask.idxmax())
        target_category = self._products.iloc[target_pos]["category"]

        scores = cosine_similarity(
            self._matrix[target_pos], self._matrix
        ).flatten()
        same_cat = (self._products["category"] == target_category).to_numpy()
        scores = scores + 0.1 * same_cat.astype(float)
        scores[target_pos] = -1.0  # exclude self

        ranked = scores.argsort()[::-1]
        out: list[dict] = []
        for i in ranked:
            if len(out) >= top_n:
                break
            row = self._products.iloc[int(i)].to_dict()
            row["score"] = float(scores[i])
            out.append(row)
        return out
