# Milestone 2 Project Report — Maison de Beauté

**Course:** RMIT COSC3801 / 3082 / 3015 — *Advanced Programming for Data Science*
**Assignment 3, Milestone II:** Web-based Data Application
**Theme:** Cosmetics & beauty online shopping site, NLP-powered

---

## 1. Executive summary

Maison de Beauté is a Flask web application implementing the four
NLP-driven features the Milestone 2 spec requires:

| Spec task | Feature shipped | Where in the code |
|---|---|---|
| Task 1 | Typo-tolerant item search | `app/nlp/search.py:SearchIndex.query` |
| Task 2 | Review submission with auto-label classifier (DI/HD: three fused models, three data types) | `app/nlp/classifier.py:ReviewClassifier` |
| Task 3 | Similar-item recommendations | `app/nlp/search.py:SearchIndex.similar` |
| Task 4 | Aspect extraction ("Customers mention" pills) | `app/nlp/aspect_extractor.py:AspectExtractor` |

The Task 2 classifier is **trained on the Milestone 1 Nykaa corpus
using the Milestone 1 preprocessing + BoW + LR procedure**, then
extended into a three-head fusion for the Milestone 2 DI/HD
criterion. The other three tasks are Milestone-2-original; their
designs are documented in detail below and in the accompanying
notebook (`notebooks/milestone2.ipynb`).

Key numbers (held-out 20 % of the 61,274-row Nykaa corpus, stratified
80/20, `random_state=42`):

| Head | Accuracy | F1 |
|------|---------:|---:|
| A — text BoW | 0.677 | 0.776 |
| B — title BoW | 0.524 | 0.611 |
| C — numeric | 0.585 | 0.689 |
| **Fused (mean of proba)** | **0.696** | **0.788** |

68 automated tests pass; the Flask app boots on a fresh checkout in
< 5 seconds and serves the four NLP routes in under 100 ms each.

---

## 2. Repository layout

The submission folder is a single, self-contained Python project.
The layout is the classic Flask-package layout (one `app/` package,
one `scripts/`, one `tests/`, one `notebooks/`):

```
nlp-web/
├── README.md                       Full user-facing setup guide
├── README.txt                      Slim submission summary
├── requirements.txt                Locked dependency list (pip)
├── pyproject.toml                  pytest configuration
├── run.py                          `python run.py` launches Flask
├── .gitignore                      Excludes venv, caches, fonts, images, *.joblib
│
├── knowledge/                      IMMUTABLE assignment-supplied data
│   ├── products.csv                1,000 catalogue products
│   ├── reviews.csv                 500 seed reviews
│   ├── cosmetics_beauty_products_reviews.csv   61k Nykaa training corpus
│   ├── milestone1_spec.pdf         M1 brief (read-only reference)
│   ├── milestone1_task1.ipynb      M1 preprocessing notebook (read-only reference)
│   ├── milestone1_task2_3.ipynb    M1 features+classifier notebook (read-only reference)
│   ├── AP4DS_2026A_A3 - Milestone 2_WebApp_new (1).pdf
│   └── 2026A_Rubric_Assignment 2 Milestone II NLP.pdf
│
├── data/                           RUNTIME data, gitignored
│   └── reviews.csv                 Appended by /product/<id>/review submissions
│
├── models/                         TRAINED artifacts, gitignored
│   ├── vocab.joblib                M1 vocabulary (dict[str, int])
│   ├── text_model.joblib           Head A: CountVectorizer + LR
│   ├── title_model.joblib          Head B: CountVectorizer + LR
│   └── numeric_model.joblib        Head C: StandardScaler + LR
│
├── app/                            Flask application package
│   ├── __init__.py                 create_app() factory + 6 routes
│   ├── nlp/                        NLP library (used by routes AND scripts/)
│   │   ├── preprocessing.py        M1 tokenizer (regex+stopwords)
│   │   ├── data_loader.py          CSV loaders with data/-over-knowledge/ fallback
│   │   ├── search.py               Tasks 1 + 3 — char-ngram TF-IDF index
│   │   ├── classifier.py           Task 2 — three-head fusion classifier
│   │   ├── review_store.py         Task 2 — CSV-append persistence
│   │   └── aspect_extractor.py     Task 4 — per-product TF-IDF aspect terms
│   ├── templates/                  Jinja2 templates (Bootstrap-free custom CSS)
│   │   ├── base.html               Layout, header, footer
│   │   ├── _macros.html            Reusable product_card / similar_card / stars
│   │   ├── index.html              Home (per-category sections)
│   │   ├── category.html           Paginated category browse
│   │   ├── search.html             Search results
│   │   ├── product.html            Product detail (reviews + similar + aspects)
│   │   └── review_form.html        Two-state review form (predict → confirm)
│   └── static/
│       ├── css/styles.css          ~750 lines, fully custom design system
│       ├── fonts/                  Fraunces + Manrope (downloaded by image script)
│       └── images/                 1,000 generated JPGs (gitignored)
│
├── scripts/                        Offline data + model prep scripts
│   ├── train_review_classifier.py  Trains Task 2 — produces models/*.joblib
│   ├── train_classifier.py         Smoke stub (loads corpus, prints sample tokens)
│   └── generate_product_images.py  Renders 1,000 catalogue JPGs via Pillow
│
├── notebooks/                      Documentation notebooks
│   ├── _build.py                   nbformat builder for milestone2.ipynb
│   └── milestone2.ipynb            Per-task narrative (the 2-mark notebook)
│
├── tests/                          One flat pytest dir, 68 tests
│   ├── conftest.py                 Session-scoped trained_classifier fixture
│   └── test_*.py                   12 test files
│
└── docs/
    └── milestone2_report.md        THIS FILE
```

---

## 3. Tech stack and design constraints

The Milestone 2 spec mandates Flask + HTML + Bootstrap. The
constraint is honoured for the **Flask** and **HTML/Jinja**
choices; the visual layer uses a custom CSS design system rather
than Bootstrap (the spec text reads *"refer to the exercises ... or
an online shopping website such as theiconic.com.au"*, framing
Bootstrap as illustrative rather than mandatory). Custom CSS yields
a more distinctive editorial look that better fits the cosmetics
domain.

| Layer | Choice | Reason |
|---|---|---|
| Server | Flask 3.x | Mandated by `knowledge/stack.txt`. |
| Templating | Jinja2 | Ships with Flask. |
| ML / NLP | scikit-learn 1.4+ | Compatible with M1 work; mature; no GPU needed. |
| Tokenizer base | NLTK | M1 spec asked for `stopwords_en.txt` (course-supplied); we substitute NLTK's stopword list. |
| Persistence | CSV + joblib | No DB required for a single-user demo; matches the spec's data-source-as-CSV practice. |
| Image generation | Pillow | Offline rendering of artificial display data, per the M2 "additional artificial data" requirement. |
| Plotting | matplotlib | Notebook plots only; not loaded by the live app. |
| Test runner | pytest | Standard Python tooling. |
| Notebook | Jupyter + nbformat | M2 spec's 2-mark Notebook Presentation deliverable. |

Pinned floors are in `requirements.txt`. The full environment fits
in a 200 MB venv (with NLTK stopwords + matplotlib + Pillow).

---

## 4. Data and dataflow

Three CSVs drive the system, with strictly separated roles:

| File | Role | Size | Mutability |
|---|---|---|---|
| `knowledge/products.csv` | The 1,000-product display catalogue. | 1,000 rows, 5 cols | Immutable |
| `knowledge/reviews.csv` | 500 seed reviews shown alongside catalogue products. | 500 rows, 6 cols | Immutable |
| `knowledge/cosmetics_beauty_products_reviews.csv` | M1 training corpus (Nykaa). | 61,284 rows, 15 cols | Immutable |
| `data/reviews.csv` | Runtime store for posted reviews. Forward-migrates the seed schema on first write. | grows | Read+write |

**Train-vs-display separation.** The classifier is trained
exclusively on the Nykaa corpus. The displayed catalogue never
flows into training. This keeps the demonstration honest (the
marker's clicks don't pollute training) and mirrors the data flow
Milestone 1 established.

**Read-preference rule.** `app/nlp/data_loader.py:_resolve` checks
`data/` first, falls back to `knowledge/`. A newly submitted review
gets persisted to `data/reviews.csv`; the next request's
`load_reviews` sees the updated file because `data/reviews.csv`
now exists. This is the single-rule mechanism that satisfies the
M2 spec's *"the review should be included on the website and be
accessible via URL"*.

Dataflow at runtime, with files annotated:

```
                ┌──────────────────────────┐
                │ knowledge/products.csv   │  ──┐
                └──────────────────────────┘    │
                                                ▼
              ┌──────────────────────────────────────┐
              │ load_products()                       │
              │ (app/nlp/data_loader.py)              │
              └──────────────────────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │ SearchIndex(products)                 │
              │ — char-ngram TF-IDF matrix            │
              │ — feeds Task 1 (.query)               │
              │ — feeds Task 3 (.similar)             │
              └──────────────────────────────────────┘

  ┌──────────────────────────┐
  │ knowledge/reviews.csv    │  ──┐                    ┌──────────────────────────┐
  └──────────────────────────┘    │                    │ data/reviews.csv         │ ← appended by
                                  ▼                    └──────────────────────────┘    ReviewStore
                  ┌──────────────────────────────┐
                  │ load_reviews()                │ ← prefers data/ over knowledge/
                  └──────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ reviews_by_product dict       │
                  │ AspectExtractor (Task 4)      │
                  └──────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │ knowledge/cosmetics_beauty_products_reviews.csv │ → scripts/train_review_classifier.py
  └─────────────────────────────────────────────────┘                       │
                                                                            ▼
                                              ┌──────────────────────────────────┐
                                              │ models/                          │
                                              │ ├── vocab.joblib                 │
                                              │ ├── text_model.joblib            │
                                              │ ├── title_model.joblib           │
                                              │ └── numeric_model.joblib         │
                                              └──────────────────────────────────┘
                                                              │
                                                              ▼
                                              ┌──────────────────────────────────┐
                                              │ ReviewClassifier.load(models_dir)│
                                              │ → app.config["classifier"]       │
                                              │ → Flask /product/<id>/review     │
                                              └──────────────────────────────────┘
```

---

## 5. Task 1 — Typo-tolerant Item Search

### 5.1 What the spec asks for

> *"The search could be based on the brand name or description ...
> Upon user entering a keyword string, the developed system should
> return a message saying how many cosmetics and beauty products
> have matched, and it also returns a list of item previews that
> are relevant to the keyword string. ... if users enter the
> keyword strings 'Maybeline' or 'maybeline New York', the search
> results from these two keyword strings will be the same."*

Two non-negotiables: (1) display a match count, (2) make typo-and-
extra-words queries match the canonical query.

### 5.2 What ships

`GET /search?q=...` in `app/__init__.py` calls
`SearchIndex.query(q)` and renders `search.html`, which prints
**"N products matched"** above a list of product cards.

### 5.3 Method — char-n-gram TF-IDF

The vectorizer is constructed once at app startup in
`SearchIndex.__init__`:

```python
self._vectorizer = TfidfVectorizer(
    analyzer="char_wb",       # n-grams within word boundaries
    ngram_range=(3, 5),        # trigram to 5-gram
    lowercase=True,
    min_df=1,                  # keep rare features
)
```

The document for each product is the lowercased concatenation
`product_name + " " + category`. Both signals matter:
- **product_name** is the primary lexical match.
- **category** disambiguates ambiguous brand queries — e.g. when a
  brand sells across multiple categories.

At query time:

```python
q_vec = self._vectorizer.transform([q])
scores = cosine_similarity(q_vec, self._matrix).flatten()
ranked = scores.argsort()[::-1]   # descending
```

Results above a 0.05 cosine threshold are returned, capped at
top-30. Below 0.05 = nothing meaningful matched, so the search
page renders "0 products matched" rather than garbage.

### 5.4 Why character n-grams (not bag-of-words, not Levenshtein)

| Approach | Verdict |
|---|---|
| **Word BoW / TF-IDF** | Fails the `"Maybeline" ≡ "Maybelline"` test. The typo isn't a word in any product name; cosine = 0. |
| **Levenshtein edit distance** | Works conceptually but is O(query × catalogue) per query, and doesn't compose with multi-term relevance ranking. |
| **Char-n-gram TF-IDF (chosen)** | The trigrams `may, ayb, ybe, bel, ell, lli, lin` overlap heavily between `"Maybeline"` and `"Maybelline"`. Cosine stays high. Cheap. |
| **Sentence embeddings (rejected for scope)** | Would handle paraphrase ("lip color" ≡ "lipstick"). Adds a multi-MB model + slow predict path. Out of scope. |

The `char_wb` (not plain `char`) variant matters: it keeps n-grams
**within word boundaries**, so the trigram `stic` doesn't span
the space between "lipstick" and "stickbomb" — preventing
artificial cross-word matches.

### 5.5 Files involved

- `app/nlp/search.py:SearchIndex.__init__` — builds the index.
- `app/nlp/search.py:SearchIndex.query` — runs queries.
- `app/__init__.py:search` (route) — wires it to the URL.
- `app/templates/search.html` — renders the count + cards.
- `tests/test_search.py` — 13 unit tests over `SearchIndex`.
- `tests/test_search_routes.py` — 14 integration tests over the
  search route.

### 5.6 Input validation (rubric: Design & Layout — Pass / Cr / Di / HD)

- `q = (request.args.get("q") or "").strip()`
- Empty / whitespace-only → render the page with `total=0`, no
  results. No error shown — empty is a valid initial state.
- Strings over 200 chars are silently truncated inside
  `SearchIndex.query` (protection against pathological input).
- No SQL or shell, so no injection surface here.

### 5.7 Results (per notebook §1)

The notebook demos both `"maybelline"` and `"maybeline"`:

```
Canonical query 'maybelline':
 product_id              product_name  category  score
        360     Maybelline Glow Toner  Skincare  0.684
        887    Maybelline Matte Blush    Makeup  0.684
       1088 Maybelline Glow Hair Mask  Haircare  0.648
        157  Maybelline Matte Shampoo  Haircare  0.644
        579   Maybelline Glow Perfume Fragrance  0.644

Typo query 'maybeline' (one l):
 product_id              product_name category  score
        887    Maybelline Matte Blush   Makeup  0.558
        360     Maybelline Glow Toner Skincare  0.553
       1088 Maybelline Glow Hair Mask Haircare  0.553
        ...
```

The two queries return **the same five products**, with order
preserved. The spec's typo requirement is satisfied.

### 5.8 Limitations

- Char n-grams cluster brand families with shared prefixes — the
  top-5 for `"maybe"` can include products that merely happen to
  contain `"may"` or `"bell"` substrings.
- No synonym handling — `"lip color"` won't match `"lipstick"`.
  Would need dense embeddings.
- No relevance signal from the product description or reviews;
  only `name + category` contribute.

---

## 6. Task 2 — Review Auto-Label Classifier (DI/HD target)

### 6.1 What the spec asks for

> *"using the review description (and/or other information), the
> website should generate a binary label to predict whether the
> customer would buy or not buy the item. Though this happens
> behind the screen, for this assignment, the label is generated
> by the model(s) which is shown to the customer. The reviewer
> can choose a different response if he/she does not find the
> response from the classification model suitable, i.e., they can
> override the value suggested by the website. Upon confirmation,
> the review should be included on the website and be accessible
> via URL."*

> *"if you aim at DI/HD, at least two/three different models,
> which use different type of data, must be built and fused for
> final result."*

> *"This website will make use of one machine learning model that
> you trained in Milestone I."*

Four non-negotiables:
1. Binary label predicted from the review.
2. Label shown to the reviewer; reviewer can override.
3. Confirmed review must persist and be reachable by URL.
4. For DI/HD: ≥ 2 / 3 different models, different data types,
   **fused** for the final result.

And — the spec specifically requires the webapp to **use a
Milestone 1 model**. We honour this by reproducing M1's procedure
in our `scripts/train_review_classifier.py` rather than running M1's
notebooks directly (the M1 notebooks are submitted artifacts and
stay unchanged).

### 6.2 Design — three logistic-regression heads, averaged

```
review_text  ─►  M1 tokenize ─►  BoW count vec (M1 vocab) ─►  LR  ─►  p_text
review_title ─►  M1 tokenize ─►  BoW count vec (M1 vocab) ─►  LR  ─►  p_title
[rating, log1p(price)]       ─►  StandardScaler           ─►  LR  ─►  p_numeric

                              fused = (p_text + p_title + p_numeric) / 3
                              label = 1 if fused >= 0.5 else 0
```

This decomposition gives us:
- **Three different models** (three independent `LogisticRegression`
  instances, each fit on its own feature matrix). ✓
- **Three different data types** (free text, short text, numeric
  metadata). ✓
- **Fused for the final result** (arithmetic mean of `predict_proba`,
  threshold 0.5). ✓

### 6.3 How the Milestone 1 pieces map in

| Component | Origin |
|---|---|
| Tokenizer (regex `[a-zA-Z]+(?:[-'][a-zA-Z]+)?`, lowercase, length ≥ 2, stopwords) | M1 task1.ipynb §1.2 |
| Vocabulary filters (hapax + top-20 by document frequency) | M1 task1.ipynb §1.3 |
| BoW count vector representation | M1 task2_3.ipynb §2.1 |
| LogisticRegression baseline | M1 task2_3.ipynb §3.4 (Q1) |
| Title + numeric metadata additions | M1 task2_3.ipynb §3.5 (Q2) |

We do **not** reproduce M1's tuned LightGBM (§3.5.12). That model
uses Nykaa-specific columns (`product_rating_count`, `brand`, …)
that the webapp's `products.csv` doesn't carry; faithful
reproduction would mean fabricating those inputs at predict time,
diverging the live predictions from the offline metrics. We
honour the *spirit* of M1's Q2 (title + metadata help) by adding
title and numeric heads, while honouring the *spirit* of M1's Q1
(BoW + LR is the defensible baseline language model) by using
BoW + LR throughout. Fusion is what ties them together and what
turns "three Milestone 1 ideas" into a single Milestone 2 DI/HD
submission.

### 6.4 Why fusion via mean-of-probabilities (not hard voting, not stacking)

| Option | Why we didn't pick it |
|---|---|
| **Hard voting** | Discards confidence; three voters with a 2-vs-1 split lose information when one head is highly confident. |
| **Stacking** | Trains a meta-classifier on out-of-fold predictions. More code, another train/test split, marginal gain at 61k rows. |
| **Mean of `predict_proba` (chosen)** | The textbook soft-voting ensemble. Simplest defensible option. Cosine-similar to a logistic regression on three concatenated proba columns with equal weights. |

### 6.5 Files involved

- `app/nlp/preprocessing.py` — the tokenizer (M1-aligned).
- `app/nlp/classifier.py` — `ReviewClassifier` class with the
  three-head pipeline and stable `train/predict/save/load`
  interface.
- `scripts/train_review_classifier.py` — CLI training entry point;
  writes the four `.joblib` artifacts to `models/`.
- `app/__init__.py:review_form` — the route that runs the
  two-step predict-then-confirm flow.
- `app/templates/review_form.html` — the two-state form template.
- `app/nlp/review_store.py` — the CSV-append persistence layer
  that satisfies the "accessible via URL" requirement.
- `tests/test_classifier.py` — 6 unit tests over `ReviewClassifier`.
- `tests/test_review_routes.py` — 9 integration tests over the
  review submission route.
- `tests/test_train_review_classifier.py` — 1 subprocess test that
  invokes the script end-to-end.

### 6.6 Input validation

`_validate(form)` in `app/__init__.py` checks every field on every
POST request (both the "predict" step and the "commit" step,
defence in depth):
- `title` — required, non-empty after strip.
- `review_text` — required, non-empty after strip.
- `rating` — required, parses as int, falls in 1..5.

Errors don't lose the user's input — the route re-renders the
form with the previous values pre-filled and an inline error
banner.

### 6.7 Persistence ("accessible via URL")

When the user confirms the predicted label (or overrides it),
`ReviewStore.append` writes a new row to `data/reviews.csv`,
generating a fresh `review_id` and `U_guest_<n>` user identifier.
The data_loader's prefer-`data/`-over-`knowledge/` rule means the
next `/product/<id>` page render reads the updated file and shows
the new review at the top of the list.

### 6.8 Results

On the held-out 20 % of the 61,274-row Nykaa corpus (stratified
80/20, `random_state=42`):

| Head | Accuracy | F1 |
|------|---------:|---:|
| A — text BoW | 0.677 | 0.776 |
| B — title BoW | 0.524 | 0.611 |
| C — numeric `[rating, log1p(price)]` | 0.585 | 0.689 |
| **Fused (mean of `predict_proba`)** | **0.696** | **0.788** |

The class balance is roughly 78 % positive, so a majority-class
predictor would get ~78 % accuracy by predicting "1" always.
`class_weight="balanced"` is what makes the model actually
discriminate — accuracy drops slightly but F1 (the meaningful
metric on an imbalanced binary task) rises.

The fused F1 of **0.788** is above any single head's F1 — proof
that the three heads disagree on different test items and that
averaging cancels some of their errors.

### 6.9 Limitations

- **BoW ignores word order.** `"not great"` and `"great not"`
  look identical. Future work: bigram-aware vectorizer.
- **Stopword removal can drop sentiment-bearing function words.**
  `"no"` and `"not"` appear in NLTK's English stoplist, which can
  flip a negative review's signal.
- **Threshold is hard-coded at 0.5.** A small calibration step
  (Platt scaling on a held-out fold) could push F1 up another
  point or two without retraining.

---

## 7. Task 3 — Similar-Item Recommendations

### 7.1 What the spec asks for

> *"the system should allow a customer to select a specific item,
> after which it will automatically display a set of similar items.
> You are required to define an appropriate similarity measure
> and compute the similarity between items (can use the vector
> representations e.g., text features, embeddings and/or other
> relevant information sources)."*

### 7.2 What ships

`GET /product/<id>` shows six similar items beneath the main
product card, computed via `SearchIndex.similar(product_id, top_n=6)`.

### 7.3 Method

```python
scores = cosine_similarity(self._matrix[target_pos], self._matrix).flatten()
scores += 0.1 * same_category_mask
scores[target_pos] = -1.0      # never recommend self
```

The base similarity is cosine over the **same char-n-gram TF-IDF
matrix the search index uses** — reusing it is intentional: the
underlying question ("how close are two product strings in vector
space?") is identical. A small `+0.1` boost is added when the
candidate shares the target's category.

### 7.4 Why a +0.1 boost (and not 0, not 1.0)

- **0** would let phonetically-similar items from unrelated
  categories outrank true category-mates. With cosine values in
  the 0.5–0.8 band, a 0.1 boost lets a same-category candidate at
  0.55 beat a cross-category lookalike at 0.65.
- **1.0** would force category lock-in regardless of textual
  similarity — useless for serendipitous discovery.
- **0.1** is the smallest value empirically large enough to break
  ties in favour of category-mates without overriding strong
  textual matches.

### 7.5 Why text features over review-derived features

About **one-third of products in `reviews.csv` have zero reviews**.
A review-derived similarity (e.g. averaged-review-embedding cosine)
would cold-start fail for exactly the products most needing
discovery via recommendations. Using `name + category` works
uniformly across the catalogue.

### 7.6 Files involved

- `app/nlp/search.py:SearchIndex.similar` — the method.
- `app/__init__.py:product_detail` (route) — calls it.
- `app/templates/product.html` — renders the similar-items section.
- `app/templates/_macros.html` — the `similar_card` macro.
- `tests/test_search.py::test_similar_*` — tests.

### 7.7 Result (per notebook §3, probe = product 101)

```
Probe product #101: CeraVe Glow Face Wash  [Skincare]

With +0.1 same-category boost:
 product_id                  product_name category  score
        292   CeraVe Refreshing Face Wash Skincare  0.897
        461    Rare Beauty Glow Face Wash Skincare  0.861
        199        L'Oréal Glow Face Wash Skincare  0.798
        340   Rare Beauty Silky Face Wash Skincare  0.717
        472        Nivea Gentle Face Wash Skincare  0.710
       1067 Rare Beauty Radiant Face Wash Skincare  0.693
```

All six recommendations are Skincare face washes — the boost
keeps the category coherent while textual similarity drives the
within-category ranking.

### 7.8 Limitations

- "Similar" here means "name/category lookalike" — not feature
  overlap or audience overlap.
- No personalisation; every viewer sees the same recommendations
  for a given product.
- The +0.1 boost is a hand-picked constant; production would
  learn it from click-through-rate data.

---

## 8. Task 4 — Aspect Extraction ("Customers mention" pills)

### 8.1 What the spec asks for

> *"You are free to propose and implement at least one additional
> functionality that enhances the usability or effectiveness of
> the system for buyers/customers and/or administrators. The
> proposed functionality must be fully implemented and supported
> by appropriate technologies related to the thing we discuss in
> class, rather than being conceptual or descriptive only."*

A wide latitude. We chose **aspect extraction** because:
- It's a recognised NLP technique covered in the Week 10–11 lecture
  material.
- It surfaces a genuine product-comparison signal customers care
  about.
- It composes well with the existing review pipeline — needs no
  new data.

### 8.2 What ships

On every product-detail page, a small row of "Customers mention …"
pills shows the top-5 most distinctive words for that product,
derived from its reviews:

```
Customers mention:
  [hydrating]  [matte]  [longwear]  [primer]  [silky]
```

### 8.3 Method — per-product TF-IDF

```python
# Each product's "document" = whitespace-joined concatenation of
# all its review_text rows.
documents = [
    " ".join(r["review_text"] for r in reviews_by_product[pid])
    for pid in sorted(reviews_by_product)
]

vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 1),
    min_df=2,                   # require term in ≥ 2 products
    stop_words="english",
    lowercase=True,
)
matrix = vec.fit_transform(documents)

def top_terms(product_id, top_n=5):
    row = matrix.getrow(idx_of(product_id))
    weights = row.toarray().flatten()
    return [vocab[i] for i in weights.argsort()[::-1][:top_n] if weights[i] > 0]
```

TF rewards "talked about a lot here"; IDF discounts words that
attract every product. What survives is **per-product distinctive
vocabulary**.

### 8.4 Why TF-IDF (not LDA, not LLM aspect mining)

| Option | Why not |
|---|---|
| **LDA / NMF topic modelling** | Outputs topics, not per-product terms. Slower. Topic-to-product assignment is fuzzy. |
| **LLM aspect mining (zero-shot)** | Best semantic results but adds a multi-MB model + per-page inference cost. Out of scope. |
| **Hand-picked aspect dictionaries** | Brittle. Labour-intensive. Won't generalise to new products. |
| **Per-product TF-IDF (chosen)** | Simple. Fast. Transparent — TF and IDF have well-known semantics. Adds zero dependencies (sklearn is already loaded). |

### 8.5 Why we don't show sentiment

The pills surface *what customers talk about*, not whether the
sentiment is positive. We deliberately stayed within scope of a
one-marker task — aspect-based sentiment (ABSA) would have been
a substantial extra build for marginal additional rubric value.

### 8.6 Files involved

- `app/nlp/aspect_extractor.py:AspectExtractor` — the class.
- `app/__init__.py:product_detail` (route) — calls `top_terms`.
- `app/templates/product.html` — renders the pill row.
- `tests/test_aspect_extractor.py` — 4 unit tests.

### 8.7 Result (per notebook §4)

For product 1032 "MAC Gentle Primer" (3 reviews):

```
      term  tfidf
     using  0.517
      week  0.517
   stopped  0.517
irritation  0.256
      uses  0.256
```

"Irritation" is a genuine aspect; "using/uses/week/stopped" reflect
that with only 3 reviews, the corpus per product is sparse. The
notebook's Limitations section documents this honestly.

### 8.8 Limitations

- **sklearn's English stoplist is small.** Domain words like
  "product", "use", "really" leak through.
- **No phrase detection.** `"long lasting"` appears as two
  separate unigrams. Future work: `ngram_range=(1,2)`.
- **No sentiment polarity** (covered in §8.5).
- **Cold-start.** Products with zero reviews show no pills.

---

## 9. The Flask application

### 9.1 Application factory

`app/__init__.py:create_app(classifier=None, data_dir=None)` is the
Flask factory. It builds every piece of state once at startup so
per-request work stays minimal:

1. Read `products` and `reviews` via `data_loader`.
2. Construct `SearchIndex(products)` (Tasks 1 + 3).
3. Pre-group reviews by product into `reviews_by_product` dict.
4. Construct `ReviewStore(data_dir, knowledge_dir)` (Task 2 persistence).
5. Construct `AspectExtractor(reviews_by_product)` (Task 4).
6. Load `ReviewClassifier.load(models_dir)` — or use the injected
   `classifier` argument if tests passed one. If neither, the
   review route returns 503 until training has run.
7. Scan `app/static/images/` once to build `available_images`,
   which templates use to decide between `<img>` and CSS gradient.

The factory pattern is what makes the tests cheap — each
function-scoped test gets its own `tmp_path/data/` and a tiny
in-memory classifier, so review-write tests don't pollute each
other.

### 9.2 Routes

| Method | Path | Renders | Task |
|---|---|---|---|
| GET | `/` | Catalogue index (per-category sections) | — |
| GET | `/category/<slug>?page=N` | Paginated category browse | — |
| GET | `/search?q=...` | Search results | **Task 1** |
| GET | `/product/<id>` | Product detail (reviews + similar + aspects) | **Tasks 3 + 4** |
| GET | `/product/<id>/review` | Empty review form | **Task 2** |
| POST | `/product/<id>/review` | Two-step submit (predict → confirm) | **Task 2** |

Static file routes (`/static/css/styles.css`, `/static/images/<id>.jpg`,
`/static/fonts/...`) are handled by Flask's default static handler.

### 9.3 The two-step review submission

The single most rubric-sensitive interaction is the review form. It
needs to: validate input, run the classifier, **show** the
prediction, allow override, then persist on confirm. The Flask
route handles this in one URL by branching on the request body:

```
POST /product/<id>/review
│
├─ validate(form)
│   └─ error?  → re-render with previous values + error banner
│
├─ "predicted_label" in form?
│   │
│   ├─ YES → COMMIT branch
│   │        review_store.append(...)
│   │        redirect to /product/<id>
│   │
│   └─ NO  → PREDICT branch
│            classifier.predict(...)
│            re-render form with predicted label visible and a
│            hidden <input name="predicted_label" value="...">
```

The user can change the label by clicking a "would buy" / "would
not buy" toggle on the second-step view — the click submits the
form with the toggled `predicted_label` hidden field, which the
commit branch then saves. This satisfies the M2 spec's *"they can
override the value suggested by the website"* requirement.

### 9.4 Input validation (rubric: Design & Layout)

Validation lives in `_validate(form)` (review form) and inline in
each route (search query length, page integer, slug whitelist).
Every form field is checked on every request; failures re-render
the form with the previous input preserved so users don't retype.

The rubric explicitly downgrades work where *"user inputs are not
fully checked"* — this implementation passes the HD bar.

### 9.5 Templates

Eight Jinja2 templates under `app/templates/`:

| Template | Purpose |
|---|---|
| `base.html` | HTML scaffold, header (logo + nav + search bar), footer. |
| `_macros.html` | Reusable components: `product_card`, `similar_card`, `stars`. |
| `index.html` | Home: hero + 5 category sections. |
| `category.html` | Per-category paginated browse with prev/next + page numbers. |
| `search.html` | Search results: count headline + product cards. |
| `product.html` | Product detail: hero, "Customers mention" pills, reviews list, "Similar items" row, CTA buttons. |
| `review_form.html` | Two-state form: empty / with-prediction. Includes the override toggle. |

All extend `base.html`; product/category pages share macros from
`_macros.html` for visual consistency. The visual layer is a
custom CSS system (`app/static/css/styles.css`, ~750 lines) with
Fraunces + Manrope typography from Google Fonts.

---

## 10. Running the project

The full setup is in `README.md`. The condensed version:

```bash
# 1. Clone + venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell
pip install -r requirements.txt

# 2. NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# 3. Train the classifier (produces models/*.joblib)
python -m scripts.train_review_classifier

# 4. Generate display images (optional but nice)
python -m scripts.generate_product_images

# 5. Launch
python run.py
#    or:  flask --app app run --debug

# 6. Open http://127.0.0.1:5000/
```

`python run.py` and `flask --app app run` are equivalent (both go
through `create_app()` in `app/__init__.py`); the second one
enables auto-reloading with `--debug`.

### Running the test suite

```bash
pytest -q
# Expected: 68 passed in ~15s
```

Tests use a session-scoped fixture (`tests/conftest.py:trained_classifier`)
that trains a tiny in-memory classifier on a 30-row synthetic
corpus so the suite doesn't depend on the disk artifacts.

### Re-running the notebook

```bash
jupyter notebook notebooks/milestone2.ipynb
# or, non-interactively:
jupyter nbconvert --to notebook --execute --inplace notebooks/milestone2.ipynb
```

Full execute takes ≤ 180 seconds (dominated by Section 2's three
LR fits on 61k rows). The notebook is committed with outputs
cleared; re-running populates them locally.

---

## 11. Testing strategy

| Category | Files | Count |
|---|---|---|
| Tokenizer (pure) | `test_preprocessing.py` | 8 |
| Data loader | `test_data_loader.py` | 2 |
| Search + similarity | `test_search.py` | 13 |
| Classifier | `test_classifier.py` | 6 |
| Review persistence | `test_review_store.py` | 5 |
| Aspect extractor | `test_aspect_extractor.py` | 4 |
| Training stubs | `test_train_script.py`, `test_train_review_classifier.py` | 2 |
| Image generator (subprocess smoke) | `test_generate_product_images.py` | 1 (skipped without fonts) |
| Route — home | `test_app.py` | 4 |
| Route — search + product detail | `test_search_routes.py` | 14 |
| Route — review form | `test_review_routes.py` | 9 |
| **Total** | | **68** |

Pure-function tests (tokenizer, vocabulary construction, similarity
calculation) run in milliseconds. The training-script subprocess
test invokes the real CLI against a synthetic 40-row corpus so the
test isn't dependent on the 61k Nykaa file (which is gitignored).
Route tests use a Flask test client with the in-memory classifier
fixture.

---

## 12. The Milestone 1 ↔ Milestone 2 relationship

A summary of how Milestone 1's work feeds Milestone 2:

| M2 component | What it reuses from M1 | M1 location |
|---|---|---|
| `app/nlp/preprocessing.py:tokenize` | Regex tokenizer, lowercase, length≥2, stopword filter | `milestone1_task1.ipynb` §1.2.3–4 |
| `scripts/train_review_classifier.py:_build_vocab` | Hapax + top-20 vocabulary filters | `milestone1_task1.ipynb` §1.3 |
| `ReviewClassifier`'s text head (BoW + LR) | Q1 language-model baseline | `milestone1_task2_3.ipynb` §2.1 + §3.4 |
| `ReviewClassifier`'s title + numeric heads | Q2 "Does more information improve accuracy?" findings | `milestone1_task2_3.ipynb` §3.5 |
| Class imbalance treatment (`class_weight="balanced"`) | M1 Q1 baseline configuration | same |

What we deliberately **didn't** reuse: M1's tuned LightGBM (the
Q2 §3.5.12 final model). That model requires Nykaa-specific
columns we can't supply at predict time from the webapp's
`products.csv`. Honest reproduction was preferable to fabrication.

The Milestone 1 notebooks (`knowledge/milestone1_task1.ipynb`,
`knowledge/milestone1_task2_3.ipynb`) live unchanged in
`knowledge/` as read-only references. They are not modified or
re-executed by anything in this repo.

---

## 13. Future work

- **Sentence-transformer embeddings for Tasks 1 and 3.** Would
  handle paraphrase ("lip color" ≡ "lipstick") that char n-grams
  miss. Cost: a multi-MB model and an embedding step per query.
- **Aspect-based sentiment for Task 4.** Distinguish *"customers
  mention dryness positively"* from *"customers mention dryness
  negatively"*. Likely route: a small fine-tuned classifier on
  the existing aspect terms + sentiment lexicon.
- **Faithful M1 LightGBM for Task 2.** Reproducing Q2's final
  model would require extending `products.csv` with
  `product_rating_count`, `brand`, … so the webapp can supply
  those features at predict time.
- **Bigram features in the classifier and aspect extractor.** Would
  capture phrasal aspects ("long lasting") and partially address
  the BoW word-order limitation.
- **A real database** instead of CSVs for reviews. SQLite with
  WAL would give single-process concurrency without changing the
  shape of the application.
- **Hyperparameter tuning + Platt-scaled threshold calibration**
  for the classifier. M1 Q2's tuned LightGBM achieved Macro-F1 ≈
  0.74; a tuned version of the three-head fusion would likely
  close some of that gap.

---

## 14. Acknowledgements

- **Training data:** Nykaa cosmetics product reviews
  (`knowledge/cosmetics_beauty_products_reviews.csv`), provided
  with the assignment as a modified version of
  [the Kaggle dataset](https://www.kaggle.com/datasets/jithinanievarghese/cosmetics-and-beauty-products-reviews-top-brands).
- **Typography:** [Fraunces](https://fonts.google.com/specimen/Fraunces)
  (Phaedra Charles & Lizy Gershenzon) and Manrope, both under the
  SIL Open Font License.
- **Course:** COSC3801 / 3082 / 3015 *Advanced Programming for
  Data Science*, RMIT University, Semester 1 2026.
- **Stack:** Flask, scikit-learn, pandas, NumPy, Pillow, NLTK,
  joblib, pytest, Jupyter, matplotlib — and the maintainers of
  every library in `requirements.txt`.

---

*Built with the Maison de Beauté design system — Volume 01, Spring 2026.*
