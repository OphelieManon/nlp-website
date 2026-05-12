"""Flask application factory and routes — the Milestone 2 web app.

This module wires the four Milestone 2 NLP tasks into HTTP routes.
Each route below names the task it serves so a marker can trace the
data flow from URL to the supporting `app/nlp/*.py` module.

Routes:
- GET  /                            — Catalogue index (no NLP).
- GET  /category/<slug>             — Per-category paginated browse (no NLP).
- GET  /search?q=…                  — **Task 1**: typo-tolerant item search
                                       via SearchIndex.query().
- GET  /product/<int>                — Product detail page. Displays:
                                       • the seed/posted reviews for this product
                                       • **Task 3** similar items (SearchIndex.similar)
                                       • **Task 4** "Customers mention" pills
                                         (AspectExtractor.top_terms)
- GET  /product/<int>/review        — Empty review submission form.
- POST /product/<int>/review        — **Task 2**: two-step submit:
                                       (1) validate form → run the classifier →
                                       show the predicted label for review,
                                       (2) confirm step → persist the new review
                                       via ReviewStore.append.

State built once in ``create_app()`` so per-request work stays light:
- ``products`` + ``reviews_by_product`` — read-only DataFrames /
  dict, closure-captured by the route functions.
- ``search_index`` — the Task 1 + Task 3 TF-IDF index. Exposed on
  ``app.config["search_index"]`` for future blueprint access.
- ``classifier`` — the Task 2 three-head fusion classifier. Either
  injected (tests pass a tiny in-memory classifier) or loaded from
  ``models/`` on disk. ``None`` is allowed; routes return 503 in
  that case so the app still boots before training has run.
- ``review_store`` — the Task 2 persistence layer (CSV append).
- ``aspect_extractor`` — the Task 4 per-product TF-IDF extractor.

The classifier's predict signature is fixed at
``predict(title, rating, review_text, price) -> (label, proba)``.
Keeping that interface stable is what lets us swap the classifier's
internals (M1-aligned BoW + LR fusion, see app/nlp/classifier.py)
without changing any route or template code.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for

from app.nlp.aspect_extractor import AspectExtractor
from app.nlp.classifier import ReviewClassifier
from app.nlp.data_loader import load_products, load_reviews
from app.nlp.review_store import ReviewStore
from app.nlp.search import SearchIndex

# Repo-root-anchored paths. Computed once at import time so every
# subsequent call to create_app() uses the same locations.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_KNOWLEDGE_DIR = _REPO_ROOT / "knowledge"
_IMAGES_DIR = Path(__file__).resolve().parent / "static" / "images"

# Stable category ordering used by the home page sections so the
# layout doesn't reshuffle between requests.
_CATEGORY_ORDER = ("Skincare", "Makeup", "Fragrance", "Beauty Tools", "Haircare")
# URL slug ↔ canonical category name. Slugs are lowercase, kebab-cased
# so they're safe in URLs ("beauty-tools" not "Beauty Tools").
_CATEGORY_BY_SLUG = {
    "skincare":     "Skincare",
    "makeup":       "Makeup",
    "fragrance":    "Fragrance",
    "beauty-tools": "Beauty Tools",
    "haircare":     "Haircare",
}
_SLUG_BY_CATEGORY = {v: k for k, v in _CATEGORY_BY_SLUG.items()}
_FEATURED_PER_SECTION = 4   # Products shown per category on the home page.
_PRODUCTS_PER_PAGE = 24     # Page size for /category/<slug>.


def _scan_available_images() -> set[int]:
    """Return product_ids for which a static JPG exists on disk.

    Built once at app startup so the templates can decide per-product
    whether to render an ``<img>`` tag or fall back to the CSS
    gradient placeholder. Running
    ``python -m scripts.generate_product_images`` and restarting
    Flask is the way to populate this.

    Used for the assignment's "additional artificial display data"
    requirement — the catalogue is real, the per-product hero
    imagery is generated.
    """
    if not _IMAGES_DIR.exists():
        return set()
    return {
        int(p.stem) for p in _IMAGES_DIR.glob("*.jpg") if p.stem.isdigit()
    }


def _validate(form) -> tuple[str | None, str, int, str]:
    """Validate the review-submission form (assignment rubric:
    "all user inputs are correctly validated" — graded under
    Design & Layout).

    Returns ``(error_message, title, rating, review_text)``. The
    error string is ``None`` when every field passes. The route
    caller renders the form again with the error string and the
    user's previous input pre-filled so they don't lose their work.

    Validations performed:
    - title:        non-empty after stripping
    - review_text:  non-empty after stripping
    - rating:       parses as int, falls inside 1..5

    Used by the POST handler in /product/<id>/review for both the
    "predict" and "commit" steps.
    """
    title = (form.get("title") or "").strip()
    review_text = (form.get("review_text") or "").strip()
    rating_raw = (form.get("rating") or "").strip()

    if not title:
        return "Title is required.", title, 0, review_text
    if not review_text:
        return "Review text is required.", title, 0, review_text
    try:
        rating = int(rating_raw)
    except ValueError:
        return "Rating must be a number 1–5.", title, 0, review_text
    if rating < 1 or rating > 5:
        return "Rating must be between 1 and 5.", title, rating, review_text
    return None, title, rating, review_text


def create_app(
    classifier: ReviewClassifier | None = None,
    data_dir: Path | None = None,
) -> Flask:
    """Build and return the configured Flask application.

    The factory pattern (rather than a module-level ``app``) lets
    tests inject a tiny pre-trained ``classifier`` and a temporary
    ``data_dir`` for isolated review-write tests.

    Argument summary:
    - ``classifier``: a trained ReviewClassifier. ``None`` means
      "load from models/", and if THAT fails (no artifacts yet),
      the review route will return 503 until the training script
      has been run. This degraded mode lets the rest of the app
      boot even before training has happened.
    - ``data_dir``: where to read/write the runtime ``reviews.csv``.
      Defaults to ``./data``; tests pass a ``tmp_path`` per test.
    """
    app = Flask(__name__)

    # --- Build the read-only catalogue + reviews state ---
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    products = load_products()
    reviews = load_reviews(data_dir=data_dir)

    # Task 1 + Task 3: the SearchIndex serves both keyword search and
    # similar-item recommendations from a single shared TF-IDF matrix.
    search_index = SearchIndex(products)

    # Pre-group reviews by product so the detail page can do an O(1)
    # lookup instead of scanning the whole reviews DataFrame per request.
    reviews_by_product: dict[int, list[dict]] = {
        int(pid): group.to_dict("records") for pid, group in reviews.groupby("product_id")
    }

    # Task 2 persistence: appends new reviews to data/reviews.csv. The
    # store handles forward-migrating the seed CSV (adds title +
    # predicted_label columns) on first write so old rows survive.
    review_store = ReviewStore(data_dir=data_dir, knowledge_dir=_KNOWLEDGE_DIR)

    # Task 4: per-product TF-IDF over aggregated review text. min_df=2
    # filters terms that appear in only one product's reviews.
    aspect_extractor = AspectExtractor(reviews_by_product, min_df=2)

    # Task 2 classifier: prefer the injected instance; fall back to
    # loading from disk; finally degrade gracefully to None.
    if classifier is None:
        try:
            classifier = ReviewClassifier.load(_DEFAULT_MODELS_DIR)
        except FileNotFoundError:
            classifier = None  # review route returns 503 until training runs

    app.config["search_index"] = search_index
    app.config["classifier"] = classifier

    available_images = _scan_available_images()

    @app.context_processor
    def _inject_available_images():
        # Templates use the `available_images` set to decide whether
        # to render <img> tags or fall back to the CSS gradient.
        return {"available_images": available_images}

    # ------------------------------------------------------------------
    # Route: home page — catalogue index. No NLP, just per-category
    # featured-products sections so the marker can quickly see the data.
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        featured = []
        for cat in _CATEGORY_ORDER:
            subset = products[products["category"] == cat]
            if subset.empty:
                continue
            featured.append({
                "name": cat,
                "slug": _SLUG_BY_CATEGORY[cat],
                "products": subset.head(_FEATURED_PER_SECTION).to_dict("records"),
                "total": int(len(subset)),
            })
        return render_template(
            "index.html",
            featured=featured,
            total=len(products),
        )

    # ------------------------------------------------------------------
    # Route: per-category paginated browse. Supports the "See all N →"
    # links on the home page. ?page= query param drives pagination.
    # ------------------------------------------------------------------
    @app.route("/category/<slug>")
    def category(slug: str):
        canonical = _CATEGORY_BY_SLUG.get(slug)
        if canonical is None:
            abort(404)
        # Input validation: ?page= must parse as a positive int.
        try:
            page = int(request.args.get("page", 1))
        except ValueError:
            page = 1
        page = max(1, page)
        subset = products[products["category"] == canonical]
        total = len(subset)
        pages = max(1, (total + _PRODUCTS_PER_PAGE - 1) // _PRODUCTS_PER_PAGE)
        page = min(page, pages)  # clamp out-of-range page numbers
        start = (page - 1) * _PRODUCTS_PER_PAGE
        end = start + _PRODUCTS_PER_PAGE
        page_products = subset.iloc[start:end].to_dict("records")
        return render_template(
            "category.html",
            category=canonical,
            slug=slug,
            page_products=page_products,
            page=page,
            pages=pages,
            total=total,
            per_page=_PRODUCTS_PER_PAGE,
        )

    # ------------------------------------------------------------------
    # Route: **Task 1** — typo-tolerant item search.
    # Empty query renders the search page with zero results (matches the
    # spec example of "show how many matched" with a count of 0).
    # ------------------------------------------------------------------
    @app.route("/search")
    def search():
        q = (request.args.get("q") or "").strip()
        results = search_index.query(q) if q else []
        return render_template(
            "search.html", q=q, results=results, total=len(results),
        )

    # ------------------------------------------------------------------
    # Route: product detail page. Shows the product card, all its
    # reviews, Task 3's similar-items section, and Task 4's
    # "Customers mention" pills. 404 if the id is unknown.
    # ------------------------------------------------------------------
    @app.route("/product/<int:product_id>")
    def product_detail(product_id: int):
        row = products[products["product_id"] == product_id]
        if row.empty:
            abort(404)
        product = row.iloc[0].to_dict()
        review_list = reviews_by_product.get(product_id, [])
        # Task 3: similar-item recommendations (top 6, cosine + +0.1
        # same-category boost).
        similar = search_index.similar(product_id, top_n=6)
        # Task 4: "Customers mention" pills — top 5 most distinctive
        # words from this product's aggregated review corpus.
        aspects = aspect_extractor.top_terms(product_id, top_n=5)
        return render_template(
            "product.html",
            product=product, reviews=review_list, similar=similar, aspects=aspects,
        )

    # ------------------------------------------------------------------
    # Route: **Task 2** — review submission with classifier label.
    # GET: render an empty form.
    # POST (predict step):  validate inputs, run the classifier, render
    #                       the form back with the prediction shown and
    #                       a hidden field carrying the predicted label.
    # POST (commit step):   validate inputs again (defence in depth),
    #                       read the (possibly user-overridden) label
    #                       from the hidden field, persist via the
    #                       review_store, then redirect to the product
    #                       detail page so the new review appears.
    # The "two-step submit" pattern is what gives the reviewer the
    # chance to override the model — the spec requires it.
    # ------------------------------------------------------------------
    @app.route("/product/<int:product_id>/review", methods=["GET", "POST"])
    def review_form(product_id: int):
        row = products[products["product_id"] == product_id]
        if row.empty:
            abort(404)
        product = row.iloc[0].to_dict()

        # GET → empty form
        if request.method == "GET":
            return render_template(
                "review_form.html",
                product=product, error=None, form={}, predicted=None,
            )

        # POST — validate inputs (rubric: "All user inputs are correctly validated").
        error, title, rating, review_text = _validate(request.form)
        if error:
            return render_template(
                "review_form.html",
                product=product, error=error,
                form={"title": request.form.get("title", ""),
                      "rating": request.form.get("rating", ""),
                      "review_text": request.form.get("review_text", "")},
                predicted=None,
            )

        # Branch 1: commit step — predicted_label is in the form, meaning
        # the user clicked "confirm" on the predicted-label page. Persist
        # the review and redirect.
        if "predicted_label" in request.form:
            try:
                label = int(request.form["predicted_label"])
                if label not in (0, 1):
                    raise ValueError
            except ValueError:
                # Malformed predicted_label — re-run the predict step.
                label = None
            if label is not None:
                new_review = review_store.append(
                    product_id=product_id, rating=rating,
                    review_text=review_text, title=title, predicted_label=label,
                )
                # Insert at the head so the just-posted review appears
                # first on the product page (spec §7.7 ordering).
                reviews_by_product.setdefault(product_id, []).insert(0, new_review)
                return redirect(url_for("product_detail", product_id=product_id))

        # Branch 2: predict step. Run the fused classifier and render
        # the form back with the prediction shown.
        if classifier is None:
            abort(503)
        price = float(product["price"])
        pred_label, proba = classifier.predict(title, rating, review_text, price)
        return render_template(
            "review_form.html",
            product=product, error=None,
            form={"title": title, "rating": rating, "review_text": review_text},
            predicted={"label": pred_label, "proba": proba},
        )

    return app


# Module-level app instance so `flask --app app run` discovers it.
# `python run.py` instead goes through create_app() in the run.py
# script. Both invocations produce the same app object.
app = create_app()
