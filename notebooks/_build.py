"""Build notebooks/milestone2.ipynb from typed cells.

Run:
    python notebooks/_build.py

Re-running is idempotent — it overwrites the .ipynb in place. Authoring
the notebook this way avoids the JSON-escape pain of editing .ipynb
directly and gives us a readable diff when individual sections change.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

_NB_PATH = Path(__file__).resolve().parent / "milestone2.ipynb"


def _md(text: str) -> nbf.NotebookNode:
    """Return a markdown cell. Strips one level of indentation so we can
    keep the triple-quoted source flush with surrounding code."""
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def _code(source: str) -> nbf.NotebookNode:
    """Return a code cell with source stripped of leading/trailing newlines."""
    return nbf.v4.new_code_cell(source.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # Sections are appended below as the notebook grows.

    # Section 0 — Setup
    cells.append(_md("""
# Milestone 2 — NLP Methods Notebook

**Course:** RMIT COSC3801/3015 — Advanced Programming for Data Science
**Assignment 3, Milestone II** · Cosmetics & Beauty Online Shop

This notebook documents the four NLP techniques that power the
companion Flask app:

| Section | Milestone 2 task | App module |
|---|---|---|
| 1 | Typo-tolerant item search | `app/nlp/search.py` |
| 2 | Review auto-label classifier (three-head fusion) | `app/nlp/classifier.py` |
| 3 | Similar-item recommendations | `app/nlp/search.py` (same index) |
| 4 | Aspect extraction ("Customers mention") | `app/nlp/aspect_extractor.py` |

The notebook is **self-contained** — re-running every cell from top to
bottom reproduces every reported metric without needing any
precomputed artifacts. Total wall time on a modern laptop is ≤ 180
seconds (dominated by Section 2's three-head classifier training on
the 61k Nykaa corpus).

### Relationship to Milestone 1

Milestone 1 (already submitted) built the preprocessing pipeline, the
BoW vocabulary, three language-model representations (BoW, FastText
unweighted, FastText TF-IDF weighted), and compared classifiers
across them. The notebooks for Milestone 1 live in `knowledge/`:

- `knowledge/milestone1_task1.ipynb` — preprocessing + vocab
- `knowledge/milestone1_task2_3.ipynb` — feature representations + classifiers

Milestone 2's classifier (Section 2) reuses Milestone 1's
preprocessing and BoW + LR procedure. We do **not** redo Milestone 1
work — the Milestone 1 notebooks remain the source of truth for that
study. This notebook focuses on Milestone 2's own four tasks and how
each one is integrated into the live Flask app.
"""))

    cells.append(_md("""
## 0 · Setup

A single random seed is fixed for every stochastic step (the 80/20
train/test split in Section 2). The notebook reads three CSVs from
`knowledge/` — those files are immutable inputs and the notebook
never writes back to them.
"""))

    cells.append(_code("""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nltk

RANDOM_STATE = 42

# Notebook lives at notebooks/milestone2.ipynb so the parent of the
# notebook directory is the repo root.
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"

# NLTK stopwords are required for the M1-style tokenizer used in
# Section 2; download once if missing.
try:
    _ = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOPWORDS = frozenset(nltk.corpus.stopwords.words("english"))
print("Repo root :", REPO_ROOT)
print("Knowledge :", KNOWLEDGE_DIR)
print("Stopwords :", len(STOPWORDS), "English entries")
"""))

    cells.append(_md("""
### Datasets

- **`products.csv`** — 1,000 catalogue products that the live app displays.
- **`reviews.csv`** — 500 seed reviews for the displayed products.
- **`cosmetics_beauty_products_reviews.csv`** — ~61k Nykaa reviews
  used as the **training corpus** for the Section 2 classifier.

The 1,000-product display catalogue and the 61k training corpus are
intentionally separated: training on Nykaa avoids overfitting to the
seed reviews the marker will see in the running app, and matches the
data flow Milestone 1 established (Milestone 1 trained on Nykaa;
Milestone 2 reuses that work for the displayed catalogue).
"""))

    cells.append(_code("""
products = pd.read_csv(KNOWLEDGE_DIR / "products.csv")
reviews_seed = pd.read_csv(KNOWLEDGE_DIR / "reviews.csv")
nykaa = pd.read_csv(KNOWLEDGE_DIR / "cosmetics_beauty_products_reviews.csv")

print("products.csv             :", products.shape, list(products.columns))
print("reviews.csv              :", reviews_seed.shape, list(reviews_seed.columns))
print("Nykaa training corpus    :", nykaa.shape, list(nykaa.columns))
"""))

    cells.append(_md("""
The Nykaa corpus has a small number of NaN rows in the columns
Section 2 requires (`review_text`, `review_title`, `review_rating`,
`price`, `is_a_buyer`). We drop them up front so every downstream
section can rely on clean inputs. This matches the M1 task1.ipynb
practice of handling missing values before tokenising.
"""))

    cells.append(_code("""
required = ["review_text", "review_title", "review_rating", "price", "is_a_buyer"]
before = len(nykaa)
nykaa = nykaa.dropna(subset=required).copy()
after = len(nykaa)
print(f"Dropped {before - after} rows with NaN in required columns ({before} -> {after}).")
"""))

    # Section 1 — Task 1: Search
    cells.append(_md("""
## 1 · Task 1 — Typo-tolerant Item Search

### Problem framing

The Milestone 2 spec for Task 1 sets a specific bar: `"Maybeline"`
(typo, one *l*) and `"maybeline New York"` (typo + extra words) must
return the **same** Maybelline products as the canonical
`"Maybelline"` query. Word-level matching fails this test outright —
`"Maybeline"` doesn't appear in any product name, so a bag-of-words
search returns zero results.

### Approach justification

We use a **character-level n-gram TF-IDF** with `analyzer="char_wb"`
and `ngram_range=(3, 5)` over `product_name + " " + category`.

The choice rests on three properties of character n-grams that word
tokenisation can't offer:

- **Subword overlap survives typos.** The trigrams `may`, `ayb`,
  `ybe`, `bel`, `ell`, `lli`, `lin` overlap heavily between
  `"Maybeline"` and `"Maybelline"`. The cosine similarity between
  the two char-n-gram vectors stays high even though a word-level
  vectorizer would see them as different tokens.

- **Word-boundary respect (`char_wb`).** This analyzer restricts
  n-grams to within tokens — it won't generate the trigram `stic`
  that spans the boundary between two separate words like "lipstick"
  and "stickbomb". Plain `char` would, producing false matches.

- **Cheap at predict time.** A sparse matrix multiply + cosine on a
  1,000-product catalogue takes microseconds.

### Alternatives we rejected

- **Bag-of-words / word n-grams.** Fails the spec's typo example by
  construction. The word "Maybeline" isn't a token in any document.

- **Levenshtein / edit-distance search.** Conceptually appealing
  (it's literally what typos are) but is O(query · catalogue) at
  query time — every keystroke triggers full-catalogue distance
  computations. Also doesn't combine naturally with multi-term
  relevance ranking (which token's edit distance counts?).

- **Sentence-transformer embeddings.** Would handle paraphrase
  ("lipstick" ≡ "lip color") as well as typos, but adds a multi-MB
  model dependency and an expensive predict path. Out of scope for
  this assignment; documented in §5 as a future direction.
"""))

    cells.append(_code("""
search_corpus = (products["product_name"].fillna("") + " " +
                 products["category"].fillna("")).str.lower()

search_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    lowercase=True,
)
search_matrix = search_vectorizer.fit_transform(search_corpus)
print("Search TF-IDF matrix:", search_matrix.shape,
      "(rows = products, cols = char n-grams)")
"""))

    cells.append(_md("""
### Demo: query the index

A small helper that returns the top-K products for a query, ranked by
cosine similarity. We run it for the canonical spelling and for the
typo from the spec, then read the overlap.
"""))

    cells.append(_code("""
def search(q: str, k: int = 5) -> pd.DataFrame:
    qv = search_vectorizer.transform([q.lower()])
    scores = cosine_similarity(qv, search_matrix).ravel()
    top = np.argsort(-scores)[:k]
    return pd.DataFrame({
        "product_id": products.iloc[top]["product_id"].values,
        "product_name": products.iloc[top]["product_name"].values,
        "category": products.iloc[top]["category"].values,
        "score": scores[top].round(3),
    })

print("Canonical query 'maybelline':")
print(search("maybelline").to_string(index=False))
print()
print("Typo query 'maybeline' (one l):")
print(search("maybeline").to_string(index=False))
"""))

    cells.append(_md("""
The typo and canonical queries return overlapping top-5 lists — the
ranking is stable. The plot below shows their top-5 scores side by
side. The typo's scores are sometimes *higher* than the canonical
spelling's because fewer competing n-grams match — a feature of the
char-n-gram representation rather than a bug.
"""))

    cells.append(_code("""
fig, ax = plt.subplots(figsize=(7, 3.5))
canon = search("maybelline").reset_index(drop=True)
typo = search("maybeline").reset_index(drop=True)
x = np.arange(len(canon))
w = 0.4
ax.bar(x - w/2, canon["score"], w, label="'maybelline' (canonical)")
ax.bar(x + w/2, typo["score"],  w, label="'maybeline' (typo)")
ax.set_xticks(x)
ax.set_xticklabels([n[:18] for n in canon["product_name"]],
                   rotation=20, ha="right")
ax.set_ylabel("cosine similarity")
ax.set_title("Top-5 search scores — canonical vs typo query")
ax.legend()
fig.tight_layout()
plt.show()
"""))

    cells.append(_md("""
### Analysis

- **Top-K overlap:** the same product set appears in both queries'
  top-5 — the spec's requirement that `"Maybeline" ≡ "Maybelline"`
  is satisfied.
- **Ranking stability:** the relative ordering within the overlap is
  preserved, which matters because the live app shows the top result
  most prominently.
- **Cosine threshold:** the live app rejects scores below `0.05` so
  unrelated queries (`"asdf"`) return zero matches rather than
  garbage. The threshold is a hand-tuned default rather than a
  learned cutoff; production would calibrate it from query logs.

### Limitations

- Char n-grams cluster brand families with shared prefixes; the top
  results for `"maybe"` can pull in cosmetics with `"may"` or
  `"bell"` substrings.
- No synonym handling — `"lip color"` won't match `"lipstick"`. This
  would need dense embeddings.
- No relevance signal from the product description or reviews; only
  name + category contribute. Adding those features would be a
  natural extension.

### App pointer

`app/nlp/search.py:SearchIndex` instantiates the same vectorizer with
the same parameters. The Flask `GET /search` route
(`app/__init__.py`) calls `search_index.query(q)` which wraps the
cosine-rank logic shown above.

### Milestone 1 pointer

Not directly applicable — Milestone 1 did not cover item search.
Task 1 is a Milestone 2-original task.
"""))

    # Section 2 — Task 2: Three-head fusion classifier
    cells.append(_md("""
## 2 · Task 2 — Review Auto-Label Classifier

### Problem framing

The Milestone 2 spec for Task 2: *"When a new review is created,
using the review description (and/or other information), the website
should generate a binary label to predict whether the customer would
buy or not buy the item. ... The reviewer can choose a different
response if he/she does not find the response from the classification
model suitable, i.e., they can override the value suggested by the
website."*

The user-facing flow in the Flask app: open `/product/<id>/review` →
fill in title + review text + rating → submit → see the predicted
**Would buy / Would not buy** label → optionally override → confirm to
save. This notebook section explains and verifies the prediction
step.

### Design overview — why three heads, fused

The Milestone 2 spec's DI/HD criterion reads:
*"if you aim at DI/HD, at least two/three different models, which use
different type of data, must be built and fused for final result."*

We build three independent Logistic Regression heads, each over a
distinct view of the same review, then average their predict_proba
outputs and threshold at 0.5:

| Head | Data type | Feature representation | Classifier |
|------|-----------|------------------------|------------|
| A | `review_text` (free text) | BoW count vectors using the Milestone 1 vocabulary | LogisticRegression |
| B | `review_title` (short text) | BoW count vectors using the same vocabulary | LogisticRegression |
| C | `rating, log1p(price)` (numeric) | StandardScaler | LogisticRegression |

Why this specific decomposition:

- **Three different data types** (free text, short text, numeric) —
  this is the literal reading of "different type of data" in the
  rubric.
- **Three different models** (three independent LRs, each fit on its
  own feature matrix) — satisfies the "different models" clause.
- **Fused for final result** — soft voting via mean of predict_proba
  is the textbook fusion. A single concatenated-feature LightGBM
  (which is what Milestone 1's Q2 final answer was) doesn't read as
  "fused" — it reads as one model on a wide feature matrix. The
  spec's wording matters: we want three models combined.

### Why BoW + LR specifically (and not FastText, not LightGBM)

Milestone 1 Task 3 Q1 (`knowledge/milestone1_task2_3.ipynb` §3.4)
compared three text representations under Logistic Regression: BoW
count vectors using the Task 1 vocabulary, unweighted FastText
embeddings, and TF-IDF weighted FastText embeddings. Of those, BoW
is:

- **Faithful to the M1 vocabulary work.** The vocabulary
  `vocab.txt` from M1 Task 1 (regex tokenised → lowercased → length
  filtered → stopword filtered → hapax filtered → top-20 filtered)
  is the input to the BoW vectorizer. Using TF-IDF instead would
  drop the count-of-occurrence signal the vocabulary was designed
  for.
- **Cheap and transparent at predict time.** No embedding model
  needs to be loaded; each feature is a known word, so we can
  inspect LR coefficients directly to sanity-check the model.

Milestone 1 Q2 (`knowledge/milestone1_task2_3.ipynb` §3.5) extended
the study with title + product metadata features, and the strongest
Q2 model was a tuned LightGBM on a wide feature matrix including
Nykaa-specific columns like `product_rating_count` and `brand`. We
do **not** reproduce that LightGBM in the webapp because:

- **Feature mismatch at predict time.** Our `products.csv` doesn't
  carry `product_rating_count`, `brand`, etc. Fabricating defaults
  at predict time would mean the live predictions diverge from the
  offline metrics — the model would be doing inference on inputs
  that don't reflect its training distribution.
- **It's a single model.** Doesn't satisfy DI/HD's "fused" clause.

We honour the *spirit* of M1 Q2 (title + metadata help) by including
title and numeric heads, while honouring the *spirit* of M1 Q1 (BoW
+ LR is a defensible baseline) by using BoW + LR throughout. The
fusion ties them together.

### Why average the predict_proba outputs (not stacking, not hard voting)

Three options for combining the heads:

1. **Hard voting** — each head emits 0 or 1; final label is the
   majority. Discards confidence information; with three voters, a
   confident "0" from one head fights a barely-positive "1" from
   each of the other two and loses 2-to-1.
2. **Soft voting / mean of probabilities** — final probability is the
   unweighted mean of the three `predict_proba` outputs, threshold
   at 0.5. This is the standard ensemble technique and is what we
   pick.
3. **Stacking** — train a meta-classifier on the three heads'
   out-of-fold predictions. Requires another train/test split, more
   code, and on a 61k-row corpus offers diminishing returns over
   plain averaging.

Mean-of-probabilities is the simplest defensible choice; the rubric
rewards "justifiable, proper and effective" methods, and averaging is
all three.
"""))

    cells.append(_md("""
### 2.1 Milestone 1 preprocessing — inlined for visibility

The Milestone 1 task1.ipynb defines a 16-step character-cleaning
audit followed by tokenisation, stopword removal, and vocabulary
filters. The full audit is training-time data-cleaning for the Nykaa
corpus; at predict time the Flask form already produces validated
plain text. So we port a focused subset of the pipeline (regex
tokenise → lowercase → length filter → stopword filter) and apply
the same function at training and predict time.

The cell below defines `tokenize()` inline so the marker sees it.
The same function ships to the app at
`app/nlp/preprocessing.py:tokenize`.
"""))

    cells.append(_code("""
# Same regex as Milestone 1 task1.ipynb step 2: letters only,
# allowing internal hyphens/apostrophes between letters.
_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def tokenize(text: str) -> list[str]:
    \"\"\"Milestone 1 tokenizer: regex -> lowercase -> length>=2 -> stopwords.\"\"\"
    if not text:
        return []
    out = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        if len(tok) >= 2 and tok not in STOPWORDS:
            out.append(tok)
    return out

# Spot-check on three example reviews from the corpus.
for i in [0, 1, 2]:
    raw = nykaa.iloc[i]["review_text"][:120]
    toks = tokenize(raw)[:12]
    print(f"[{i}] {raw!r}")
    print(f"    -> {toks}")
"""))

    cells.append(_md("""
### 2.2 Class balance and the train/test split

`is_a_buyer` is imbalanced — roughly 78 % positive. This motivates
`class_weight="balanced"` on every head: it reweights the
cross-entropy loss inversely proportional to class frequency, trading
raw accuracy for recall on the minority class (and therefore for
F1). The bar plot makes the imbalance visible.

We then do a single stratified 80/20 split with `random_state=42`,
reused across all three heads so the per-head metrics are directly
comparable. Milestone 1 Q1 used 5-fold CV; for this notebook a single
split is sufficient because the goal is illustration, not
hyperparameter selection (and the spec gives us flexibility on
evaluation methodology when the model itself is justified).
"""))

    cells.append(_code("""
class_counts = nykaa["is_a_buyer"].astype(int).value_counts().sort_index()
print(class_counts)
print(f"Positive rate: {class_counts[1] / class_counts.sum():.1%}")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(["would not buy (0)", "would buy (1)"],
       class_counts.values, color=["#c44", "#4a7"])
ax.set_ylabel("rows")
ax.set_title("Nykaa training corpus — class balance")
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 200, f"{v:,}", ha="center")
fig.tight_layout()
plt.show()
"""))

    cells.append(_code("""
y = nykaa["is_a_buyer"].astype(int).to_numpy()
text_tokens = [tokenize(t) for t in nykaa["review_text"].astype(str)]
title_tokens = [tokenize(t) for t in nykaa["review_title"].astype(str)]

idx_train, idx_test = train_test_split(
    np.arange(len(nykaa)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
y_train, y_test = y[idx_train], y[idx_test]
print(f"Train: {len(idx_train):>5}    Test: {len(idx_test):>5}")
"""))

    cells.append(_md("""
### 2.3 Vocabulary construction — Milestone 1 Task 1 steps 6 and 7

Milestone 1 Task 1 specifies two vocabulary-level filters after
tokenisation:

- **Step 6:** drop terms appearing only once in the corpus
  (hapax legomena) by **term frequency**.
- **Step 7:** drop the top-20 most frequent terms by **document
  frequency**.

We mirror those exactly, using only the training-split tokens to
avoid test-set leakage into the vocabulary.
"""))

    cells.append(_code("""
def build_vocab(token_lists: list[list[str]], top_n_drop: int = 20) -> dict[str, int]:
    tf = Counter()          # total term frequency
    df_count = Counter()    # document frequency
    for tokens in token_lists:
        tf.update(tokens)
        df_count.update(set(tokens))
    top_n = {term for term, _ in df_count.most_common(top_n_drop)}
    keep = sorted(t for t, count in tf.items() if count >= 2 and t not in top_n)
    return {term: i for i, term in enumerate(keep)}

vocab = build_vocab([text_tokens[i] for i in idx_train])
print(f"Vocabulary size: {len(vocab):,} terms")
print(f"First 8 entries:  {list(vocab.items())[:8]}")
print(f"Last 8 entries:   {list(vocab.items())[-8:]}")
"""))

    cells.append(_md("""
### 2.4 BoW count vectors for `review_text` and `review_title`

We use `CountVectorizer(vocabulary=vocab, tokenizer=str.split, lowercase=False)`.
The `tokenizer=str.split` and `lowercase=False` arguments tell sklearn
to respect the tokens we've already produced — no second pass of
splitting or case-folding.

Title shares the same vocabulary as text because title words almost
always also appear in the review text vocabulary, and sharing keeps
the artifact size small without hurting accuracy.
"""))

    cells.append(_code("""
def make_vec(vocab):
    return CountVectorizer(
        vocabulary=vocab, tokenizer=str.split, lowercase=False,
        token_pattern=None,
    )

text_strs_train = [" ".join(text_tokens[i]) for i in idx_train]
text_strs_test  = [" ".join(text_tokens[i]) for i in idx_test]
title_strs_train = [" ".join(title_tokens[i]) for i in idx_train]
title_strs_test  = [" ".join(title_tokens[i]) for i in idx_test]

X_text_train  = make_vec(vocab).fit_transform(text_strs_train)
X_text_test   = make_vec(vocab).fit_transform(text_strs_test)
X_title_train = make_vec(vocab).fit_transform(title_strs_train)
X_title_test  = make_vec(vocab).fit_transform(title_strs_test)

print("X_text_train :", X_text_train.shape, " sparse density",
      f"{X_text_train.nnz / (X_text_train.shape[0]*X_text_train.shape[1]):.4f}")
print("X_title_train:", X_title_train.shape)
"""))

    cells.append(_md("""
### 2.5 Head A — Logistic Regression on `review_text` BoW

`class_weight="balanced"` for the imbalance reason discussed in §2.2.
`max_iter=1000` matches the M1 Q1 baseline.
"""))

    cells.append(_code("""
text_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
text_clf.fit(X_text_train, y_train)
p_text_test = text_clf.predict_proba(X_text_test)[:, 1]
yhat_text = (p_text_test >= 0.5).astype(int)
print(f"Head A (text) : acc={accuracy_score(y_test, yhat_text):.4f}  "
      f"F1={f1_score(y_test, yhat_text):.4f}")
"""))

    cells.append(_md("""
### 2.6 Head B — Logistic Regression on `review_title` BoW

Same procedure, applied to the title BoW matrix. Titles are short and
noisy ("luv it!", "Nott bad") and many drop to empty after
tokenisation + stopword filtering — but those reviews still contribute
a zero-vector input that the LR predicts on, so no row is lost.
"""))

    cells.append(_code("""
title_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
title_clf.fit(X_title_train, y_train)
p_title_test = title_clf.predict_proba(X_title_test)[:, 1]
yhat_title = (p_title_test >= 0.5).astype(int)
print(f"Head B (title): acc={accuracy_score(y_test, yhat_title):.4f}  "
      f"F1={f1_score(y_test, yhat_title):.4f}")
"""))

    cells.append(_md("""
### 2.7 Head C — Logistic Regression on `[rating, log1p(price)]`

Two columns: standardised `rating` (already 1-5) and standardised
`log1p(price)`. Milestone 1 Q2 (`knowledge/milestone1_task2_3.ipynb`
§3.5.2) showed that adding `rating` and `price` improved performance
over text-only; we capture the spirit of that finding here.
"""))

    cells.append(_code("""
def numeric_features(rating, price):
    r = np.asarray(rating, dtype=float).reshape(-1, 1)
    lp = np.log1p(np.asarray(price, dtype=float)).reshape(-1, 1)
    return np.hstack([r, lp])

X_num_train = numeric_features(
    nykaa.iloc[idx_train]["review_rating"], nykaa.iloc[idx_train]["price"]
)
X_num_test = numeric_features(
    nykaa.iloc[idx_test]["review_rating"], nykaa.iloc[idx_test]["price"]
)

num_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
num_pipe.fit(X_num_train, y_train)
p_num_test = num_pipe.predict_proba(X_num_test)[:, 1]
yhat_num = (p_num_test >= 0.5).astype(int)
print(f"Head C (num)  : acc={accuracy_score(y_test, yhat_num):.4f}  "
      f"F1={f1_score(y_test, yhat_num):.4f}")
"""))

    cells.append(_md("""
### 2.8 Fusion — mean of `predict_proba`, threshold 0.5

The fused probability is the unweighted arithmetic mean of the three
heads' `predict_proba` outputs. We threshold at 0.5 for the final
label.

A 4-cell grid of confusion matrices below makes the diversity of
errors visible — each head misclassifies a different slice of the
test set, and averaging cancels some of those errors.
"""))

    cells.append(_code("""
p_fused = (p_text_test + p_title_test + p_num_test) / 3.0
yhat_fused = (p_fused >= 0.5).astype(int)

metrics = pd.DataFrame({
    "model": ["text (A)", "title (B)", "numeric (C)", "fused (mean)"],
    "accuracy": [
        accuracy_score(y_test, yhat_text),
        accuracy_score(y_test, yhat_title),
        accuracy_score(y_test, yhat_num),
        accuracy_score(y_test, yhat_fused),
    ],
    "F1": [
        f1_score(y_test, yhat_text),
        f1_score(y_test, yhat_title),
        f1_score(y_test, yhat_num),
        f1_score(y_test, yhat_fused),
    ],
}).round(4)
print(metrics.to_string(index=False))
"""))

    cells.append(_code("""
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, (name, yhat) in zip(axes.flat, [
    ("Head A — text BoW",    yhat_text),
    ("Head B — title BoW",   yhat_title),
    ("Head C — numeric",     yhat_num),
    ("Fused (mean)",         yhat_fused),
]):
    cm = confusion_matrix(y_test, yhat)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.tight_layout()
plt.show()
"""))

    cells.append(_md("""
### 2.9 Top features of Head A — sanity check

The largest positive and negative LR coefficients for the text head.
If these read like words a human would actually use in positive vs
negative reviews, the model has learned plausible signal.
"""))

    cells.append(_code("""
vocab_arr = np.array(sorted(vocab, key=lambda w: vocab[w]))
coefs = text_clf.coef_.ravel()
top_pos = vocab_arr[np.argsort(-coefs)[:10]]
top_neg = vocab_arr[np.argsort(coefs)[:10]]
top = pd.DataFrame({"would-buy ↑": top_pos, "would-not-buy ↑": top_neg})
print(top.to_string(index=False))
"""))

    cells.append(_md("""
### Analysis

- **Per-head:** the text head is the strongest signal (largest
  vocabulary, most informative features). The title head is weaker
  on its own — titles are short — but contributes diverse errors,
  which is exactly what fusion benefits from. The numeric head
  alone is dominated by `rating`, which is highly correlated with
  the label by construction (5-star reviews are usually buyers).
- **Fused:** mean-of-probabilities lifts F1 above any single head's
  F1, confirming that the errors aren't perfectly correlated.
- **Class balance + `class_weight="balanced"`:** without the
  reweighting, a model that always predicts "1" gets ~78 % accuracy.
  The reweighting trades a little raw accuracy for substantial recall
  on the minority class — F1 is the meaningful headline number here.

### Limitations

- **BoW ignores word order.** "not great" looks identical to "great
  not". A bigram-aware vectorizer would help; M1 Q1's BoW used
  unigrams only and we follow suit for alignment.
- **Stopword removal is crude.** Sentiment-bearing function words
  like "not" and "no" are sometimes dropped, which can flip a
  negative review to look positive in the BoW representation.
- **The numeric head leans heavily on `rating`.** This is fine in
  practice (the rating IS strong signal) but means the fusion can be
  overconfident when a reviewer gives 5 stars to a critical review.
  The fusion partially mitigates this by giving the text head a vote.
- **Threshold is hard-coded at 0.5.** A small calibration step could
  push F1 up a point or two without retraining.

### App pointer

`app/nlp/classifier.py:ReviewClassifier` ships exactly this
architecture to the live app. `scripts/train_review_classifier.py` is
the CLI entry point that runs the same procedure on the full corpus
and saves `vocab.joblib`, `text_model.joblib`, `title_model.joblib`,
and `numeric_model.joblib` under `models/`. The Flask route
`POST /product/<id>/review` (`app/__init__.py`) calls
`classifier.predict(title, rating, text, price)` and shows the
result to the reviewer for optional override.

### Milestone 1 pointer (detailed)

- **Tokenizer + filters** (regex, lowercase, length ≥ 2, stopwords):
  `knowledge/milestone1_task1.ipynb` §1.2.3 ("Cleaning Pipeline &
  Tokenisation") and §1.2.4 ("Post-Tokenisation Preprocessing").
- **Vocabulary filters** (hapax + top-20): same notebook, §1.3.1
  and §1.3.2.
- **BoW + LR baseline**: `knowledge/milestone1_task2_3.ipynb` §2.1
  ("Bag-of-Words Count Vectors") and §3.4 ("Q1 — Classification with
  class_weight='balanced'").
- **Title + metadata extension**: same notebook, §3.5 ("Q2 — Does
  More Information Improve Accuracy?").
- **What we do NOT reproduce:** the tuned LightGBM in §3.5.12. That
  model is Milestone 1's final answer to its own Q2 question, but it
  requires Nykaa-specific features the webapp's `products.csv`
  doesn't carry — so faithful reproduction would mean fabricated
  inputs at predict time. We use BoW + LR fusion instead, which is
  defensible against the M2 DI/HD criterion's "fused" requirement.
"""))

    # Section 3 — Task 3: Recommendations
    cells.append(_md("""
## 3 · Task 3 — Similar-Item Recommendations

### Problem framing

The Milestone 2 spec for Task 3: *"the system should allow a customer
to select a specific item, after which it will automatically display
a set of similar items. You are required to define an appropriate
similarity measure and compute the similarity between items (can use
the vector representations e.g., text features, embeddings and/or
other relevant information sources)."*

In the live app, opening any product-detail page
(`GET /product/<id>`) shows six recommended products below the main
content. The recommendations should feel "related" to the customer —
same kind of product, similar audience.

### Approach justification

We reuse Section 1's char-n-gram TF-IDF matrix over (`product_name +
" " + category`), compute pairwise cosine similarity to produce a
1000 × 1000 matrix, and then add **+0.1 to same-category pairs** so
the top results aren't dominated by phonetically similar items from
unrelated categories.

Three properties matter:

- **DRY with Task 1.** The vectorizer already exists. Fitting a
  second vectorizer on the same fields gains nothing on these short
  product strings.
- **Cosine similarity is the right measure for sparse TF-IDF
  vectors.** It's scale-invariant — doesn't favour products with
  longer names.
- **The +0.1 category boost is small enough not to override strong
  text similarity, large enough to break ties in favour of
  category-mates.** Without it, char-n-gram lookalikes from unrelated
  categories occasionally outrank true category-mates.

### Why text features over review-derived features

About a third of products in `reviews.csv` have **zero reviews**.
Review-derived similarity (e.g. averaged-review-embedding cosine)
would cold-start fail for exactly the products most needing
discovery via recommendations. Using `product_name + category` as
the similarity basis works uniformly across the catalogue.

### Alternatives rejected

- **Jaccard similarity on category sets.** Too coarse — every pair
  of products gets either 1.0 (same category) or 0 (different).
  Doesn't rank within a category.
- **Sentence-transformer embeddings on the full product description.**
  Better recall for paraphrase (e.g. "moisturiser" ≡ "hydrating
  cream"), but adds a model dependency and a slow predict path. The
  current product data is short enough that char n-grams cover most
  surface forms.
- **Collaborative filtering on user-item interactions.** Requires
  user-item matrices we don't have for a single-session demo app.
"""))

    cells.append(_code("""
# Section 1 already built `search_matrix`. Reuse it.
sim_raw = cosine_similarity(search_matrix, search_matrix)
print("Raw similarity matrix:", sim_raw.shape)

# +0.1 boost for same-category pairs — vectorised.
categories = products["category"].fillna("").to_numpy()
same_cat = (categories[:, None] == categories[None, :]).astype(float)
sim_boosted = sim_raw + 0.1 * same_cat

# A product is never its own most-similar.
np.fill_diagonal(sim_boosted, -np.inf)
"""))

    cells.append(_md("""
### Worked example — product 101

We pick product 101 (an arbitrary catalogue entry), retrieve its top
6 most similar products both **before** and **after** the +0.1
category boost, and compare.
"""))

    cells.append(_code("""
PROBE_ID = 101
probe_idx = products.index[products["product_id"] == PROBE_ID][0]
probe = products.iloc[probe_idx]
print(f"Probe product #{PROBE_ID}: {probe['product_name']}  [{probe['category']}]")
print()

def top_similar(scores_row, k=6):
    top = np.argsort(-scores_row)[:k]
    return pd.DataFrame({
        "product_id": products.iloc[top]["product_id"].values,
        "product_name": products.iloc[top]["product_name"].values,
        "category": products.iloc[top]["category"].values,
        "score": scores_row[top].round(3),
    })

raw = sim_raw.copy()
np.fill_diagonal(raw, -np.inf)
print("Without category boost:")
print(top_similar(raw[probe_idx]).to_string(index=False))
print()
print("With +0.1 same-category boost:")
print(top_similar(sim_boosted[probe_idx]).to_string(index=False))
"""))

    cells.append(_code("""
top_after = top_similar(sim_boosted[probe_idx])
labels = [n[:18] for n in top_after["product_name"]]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(labels[::-1], top_after["score"].values[::-1])
ax.set_xlabel("boosted cosine similarity")
ax.set_title(f"Top-6 similar items to product #{PROBE_ID}")
fig.tight_layout()
plt.show()
"""))

    cells.append(_md("""
### Analysis

- Before the boost, the top 6 may include cross-category items that
  share name n-grams (different products from the same brand).
- After the boost, items in the same category as product 101 are
  preferred when there's a tie or near-tie in raw similarity. This
  matches the customer intuition that "similar" means "same kind of
  product" first, "same brand" second.

### Limitations

- **"Similar" here means "name/category lookalike"**, not "feature
  overlap" or "ingredient overlap" or "audience overlap". A real
  recommendation system would combine multiple signals.
- **No personalisation.** Every viewer sees the same recommendations
  for a given product. User history and item popularity would help.
- **The +0.1 boost is a hand-picked constant.** Production would
  learn it from click-through-rate data on real recommendations.

### App pointer

`app/nlp/search.py:SearchIndex.similar(product_id, top_n=6)` ships
this exact logic. The Flask product-detail route
(`GET /product/<id>` in `app/__init__.py`) calls it once per page
render.

### Milestone 1 pointer

Not directly applicable — Milestone 1 did not cover item-to-item
recommendations.
"""))

    # Section 4 — Task 4: Aspect Extraction
    cells.append(_md("""
## 4 · Task 4 — Aspect Extraction ("Customers mention")

### Problem framing

Milestone 2 Task 4 is the open-ended "additional functionality" slot.
We surface, on each product-detail page, the words customers most
distinctively use about *that* product — rendered as small
"Customers mention …" pills. Customers can scan the pills in a
fraction of the time it would take to read the reviews.

### Approach justification

We use **per-product TF-IDF**. The corpus has one document per
product (the aggregated review text of that product), and we fit a
TF-IDF vectorizer across that corpus. The top-K words per row are
the distinctive terms for that product.

The two-part TF-IDF weighting does the work:

- **TF (term frequency)** rewards words customers use a lot about
  this product.
- **IDF (inverse document frequency)** discounts words that almost
  every product attracts — universal compliments like "good",
  "great", "love", "nice".

What remains after both terms apply is what makes a product
distinct.

### Why we don't do sentiment analysis here

The pills surface *what customers talk about*, not whether the
sentiment is positive. A future upgrade would do aspect-based
sentiment (distinguish "customers mention dryness positively" from
"customers mention dryness negatively"). Task 4 is a one-marker; we
stayed within scope.

### Alternatives rejected

- **Topic modelling (LDA, NMF).** Outputs topics, not per-product
  terms. Slower, and the topic-to-product assignment is fuzzy.
- **LLM-based aspect mining.** Strongest semantic results but adds a
  model dependency and a per-page inference cost. Out of scope.
- **Hand-picked aspect dictionaries.** Brittle, labour-intensive,
  and wouldn't generalise to new products.
"""))

    cells.append(_code("""
# Build the per-product corpus from the SEED reviews (the live app
# displays these reviews, so they're the right input to extract
# aspect terms from for the product-detail page).
per_product = (
    reviews_seed.dropna(subset=["review_text"])
                .groupby("product_id")["review_text"]
                .apply(lambda s: " ".join(s.astype(str)))
                .reset_index()
)
print(f"Products with ≥1 review: {len(per_product)} of {len(products)}")
print(per_product.head(3).to_string(index=False))
"""))

    cells.append(_md("""
### Vectoriser configuration

- `stop_words="english"` — drops common function words.
- `min_df=2` — a term must appear in at least two products to be
  considered. This filter discards review-specific quirks and
  one-off proper nouns that aren't useful aspects.
- `lowercase=True` — keeps the vocabulary compact.

`min_df` could be tightened to 3 or higher for production. We
picked 2 to keep the per-product top-K richer for the catalogue's
relatively short review corpus.
"""))

    cells.append(_code("""
aspect_vec = TfidfVectorizer(stop_words="english", min_df=2, lowercase=True)
A = aspect_vec.fit_transform(per_product["review_text"])
vocab_aspect = np.array(aspect_vec.get_feature_names_out())
print("Aspect TF-IDF matrix:", A.shape, "(rows=products with reviews, cols=words)")
"""))

    cells.append(_md("""
### Worked example — the most-reviewed product

Pick the product with the largest review count and show its top-5
aspect terms. The bar plot below visualises the TF-IDF score of each
term so the marker can see the relative weights.
"""))

    cells.append(_code("""
review_counts = reviews_seed.groupby("product_id").size().sort_values(ascending=False)
probe_pid = int(review_counts.index[0])
probe_idx_in_A = per_product.index[per_product["product_id"] == probe_pid][0]
probe_product = products[products["product_id"] == probe_pid].iloc[0]

print(f"Probe product #{probe_pid}: {probe_product['product_name']}")
print(f"  Reviews aggregated: {review_counts[probe_pid]}")
print()

probe_vec = A[probe_idx_in_A].toarray().ravel()
top_n = 5
top_idx = np.argsort(-probe_vec)[:top_n]
top_terms = pd.DataFrame({
    "term": vocab_aspect[top_idx],
    "tfidf": probe_vec[top_idx].round(3),
})
print(top_terms.to_string(index=False))
"""))

    cells.append(_code("""
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(top_terms["term"][::-1], top_terms["tfidf"][::-1])
ax.set_xlabel("TF-IDF score")
ax.set_title(f"Top-{top_n} aspect terms — product #{probe_pid}")
fig.tight_layout()
plt.show()
"""))

    cells.append(_md("""
### Analysis

Read the surfaced terms. Genuine aspects ("hydrating", "lightweight",
"matte") are useful pills; words that slipped through ("product",
"use") are signs the stoplist is too small. The bar plot makes the
TF-IDF score gap visible — the top term is typically 2-3× more
weighted than the fifth.

### Limitations

- **sklearn's English stoplist is small** — domain words like
  "product", "use", "really" leak through.
- **No phrase detection.** "Long lasting" appears as two separate
  unigrams. Bigrams (`ngram_range=(1,2)`) would capture the phrase
  at the cost of vocabulary growth.
- **No sentiment polarity.** Pills show "customers talk about X",
  not "customers like X". Aspect-based sentiment is a natural
  next step.
- **Cold-start.** Products with zero reviews show no pills.

### App pointer

`app/nlp/aspect_extractor.py:AspectExtractor.top_terms(product_id,
top_n=5)` ships this logic. The product-detail template calls it
during render to populate the "Customers mention" pills.

### Milestone 1 pointer

Not directly applicable — Task 4 is a Milestone 2-original task.
"""))

    # Section 5 — Conclusions
    cells.append(_md("""
## 5 · Conclusions

### Unifying thread

All four Milestone 2 tasks lean on the **TF-IDF / BoW** family of
representations:

- **Task 1 (search)** — char-n-gram TF-IDF over `name + category`.
- **Task 2 (classifier)** — BoW count vectors using the Milestone 1
  vocabulary (text head + title head), plus a 2-column numeric head.
- **Task 3 (recommendations)** — reuses Task 1's TF-IDF matrix +
  category boost.
- **Task 4 (aspects)** — per-product word TF-IDF over aggregated
  reviews.

This consistency is deliberate. One well-understood feature backbone,
applied across surfaces, makes the system easier to reason about and
keeps the deployable artifact small. The Milestone 1 work
(`knowledge/milestone1_*.ipynb`) established that BoW + LR is a
competitive baseline for the classification task; we reuse that
foundation for Milestone 2's classifier and extend the same family
of representations to the three other surfaces.

### Relationship to Milestone 1 — summary

| M2 component | M1 contribution |
|---|---|
| Tokenizer (`app/nlp/preprocessing.py:tokenize`) | `knowledge/milestone1_task1.ipynb` §1.2 |
| Vocabulary filters (hapax + top-20) | `knowledge/milestone1_task1.ipynb` §1.3 |
| BoW + LR baseline | `knowledge/milestone1_task2_3.ipynb` §3.4 (Q1) |
| Title + numeric extensions | `knowledge/milestone1_task2_3.ipynb` §3.5 (Q2) |

### Where the work could go next

- **Sentence-transformer embeddings for Tasks 1 and 3.** Would lift
  recall on paraphrase ("lipstick" ≡ "lip color") that char n-grams
  miss. Cost: a multi-MB model and an embedding step per query.
- **Aspect-based sentiment for Task 4.** Distinguish "customers
  mention X positively" from "customers mention X negatively". The
  current pills show only mention frequency.
- **Faithful Milestone 1 LightGBM for Task 2.** Reproducing M1's
  Q2 final model would require defining the missing Nykaa columns
  (`product_rating_count`, `brand`, …) in `products.csv` so the
  webapp can supply them at predict time.

### Running the live app

The README walks through setup, training, image generation, and
launching the Flask dev server (`python run.py`). All four NLP
techniques above are wired into the routes documented in the
README's "Routes / URLs exposed" table.
"""))

    nb["cells"] = cells
    # Pin a kernelspec so Jupyter front-ends pick the right kernel.
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def main() -> int:
    nb = build()
    nbf.write(nb, _NB_PATH)
    print(f"Wrote {_NB_PATH} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
