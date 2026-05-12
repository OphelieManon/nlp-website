# Maison de Beauté — NLP-Powered Cosmetics Shopping App

> RMIT COSC3801/3015 — Advanced Programming for Data Science
> Assignment 3, Milestone II

A Flask + scikit-learn web application that demonstrates four NLP-driven
features over a 1,000-product cosmetics & beauty catalogue:

1. **Typo-tolerant keyword search** (`"Maybeline"` returns Maybelline products)
2. **Review submission with an auto-label classifier** (three fused
   logistic-regression heads, user-overridable prediction)
3. **Similar-item recommendations** (cosine similarity over TF-IDF
   + category-match boost)
4. **Aspect extraction** ("Customers mention" pills computed via
   per-product TF-IDF over aggregated review text)

The UI is an editorial-style design system with custom CSS, Fraunces
+ Manrope typography, and category-tinted product imagery rendered
offline by a Pillow-based generator.

---

## Table of Contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create a virtual environment](#2-create-a-virtual-environment)
  - [3. Install Python dependencies](#3-install-python-dependencies)
  - [4. Download NLTK data](#4-download-nltk-data)
  - [5. Train the review classifier](#5-train-the-review-classifier)
  - [6. Generate product images](#6-generate-product-images)
  - [7. Run the development server](#7-run-the-development-server)
- [Running the test suite](#running-the-test-suite)
- [Routes / URLs exposed](#routes--urls-exposed)
- [Project structure](#project-structure)
- [How it works](#how-it-works)
- [Resetting state](#resetting-state)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Features

### Task 1 · Typo-tolerant search
Character-n-gram TF-IDF (`analyzer="char_wb"`, `ngram_range=(3,5)`) +
cosine similarity over `product_name + " " + category`. The character
n-gram representation keeps `"Maybeline"` (typo, one *l*) and
`"Maybelline"` (canonical, two *l*s) close in vector space, so the
assignment example returns the same Maybelline products either way.
Returns the top 30 above a 0.05 cosine threshold.

### Task 2 · Review submission with auto-label classifier
Three independently-trained `LogisticRegression` heads on different
data types, fused for the final prediction (DI/HD criterion):

| Head | Data type | Feature representation | Classifier |
|---|---|---|---|
| A | `review_text` (free text) | BoW count vectors using the Milestone 1 vocabulary | LogisticRegression |
| B | `review_title` (short text) | BoW count vectors using the same vocabulary | LogisticRegression |
| C | `[rating, log1p(price)]` (numeric) | StandardScaler | LogisticRegression |

The Milestone 1 vocabulary is reproduced from
`knowledge/milestone1_task1.ipynb` (regex tokenise → lowercase →
length ≥ 2 → NLTK stopwords → drop hapax → drop top-20 by document
frequency). Predictions are fused by arithmetic mean of
`predict_proba`, thresholded at 0.5. The form asks for title /
rating / text, predicts, and lets the user override the label before
persisting to `data/reviews.csv`. On the held-out 20 % of the
Nykaa corpus this configuration scores **fused F1 ≈ 0.79**.

### Task 3 · Similar-item recommendations
`SearchIndex.similar(product_id)` reuses the catalogue TF-IDF matrix to
compute cosine similarity from a target row to every other product,
applies a +0.1 boost for same-category candidates, excludes the target
itself, and returns the top 6.

### Task 4 · Aspect extraction
`AspectExtractor` builds a word-level TF-IDF matrix over per-product
review documents (concatenation of every review text per product),
with English stop-words removed and `min_df=2`. The top-N terms for
each product surface as the **Customers mention** pill row on the
product detail page.

---

## Tech stack

| Layer | Library / Tool | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Web framework | Flask | ≥ 3.0 |
| Templating | Jinja2 | ≥ 3.1 |
| ML / NLP | scikit-learn | ≥ 1.4 |
| Data | pandas, numpy | ≥ 2.1 / ≥ 1.26 |
| Tokenizer | NLTK | ≥ 3.8 |
| Image generation | Pillow | ≥ 10.0 |
| Model persistence | joblib | ≥ 1.3 |
| Test runner | pytest | ≥ 8.0 |
| Notebook | Jupyter | ≥ 1.0 |
| Styling | Custom CSS (no Bootstrap), Fraunces + Manrope (Google Fonts) | — |

Exact pinned floors are in [`requirements.txt`](requirements.txt).

---

## Prerequisites

- **Python 3.11 or later** (3.11, 3.12, 3.13, or 3.14 all work)
- **pip** (bundled with Python ≥ 3.4)
- **Git** for cloning
- Approximately **500 MB free disk space**: Python deps (~400 MB) + trained models (~3 MB) + generated images (~40 MB) + venv overhead
- **Internet access** for the first run (downloads pip packages, NLTK punkt/stopwords, two Fraunces font files)
- Tested on **Windows 11** (PowerShell), **macOS 14+** (Sonoma / zsh), and **Ubuntu 22.04+** (bash)

### Installing Python (if not present)

<details>
<summary><strong>Windows</strong></summary>

Download the latest 3.11+ installer from
<https://www.python.org/downloads/windows/>.
During install, **tick "Add python.exe to PATH"** and **"py launcher"**.

Verify:
```powershell
py --version           # Python 3.11.x or higher
```

Alternatively, install via [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/):
```powershell
winget install Python.Python.3.11
```
</details>

<details>
<summary><strong>macOS</strong></summary>

The system Python on macOS is not suitable for development. Install via
[Homebrew](https://brew.sh):
```bash
brew install python@3.11
```

Or download the official installer from
<https://www.python.org/downloads/macos/>.

Verify:
```bash
python3 --version      # Python 3.11.x or higher
```
</details>

<details>
<summary><strong>Linux</strong></summary>

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git
```

**Fedora / RHEL:**
```bash
sudo dnf install -y python3.11 python3-pip git
```

**Arch:**
```bash
sudo pacman -S python python-pip git
```

Verify:
```bash
python3 --version      # Python 3.11.x or higher
```
</details>

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/nlp-web.git
cd nlp-web
```

### 2. Create a virtual environment

Keeps this project's dependencies isolated from your system Python.

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation with *"running scripts is disabled on
this system"*, run this once in an **admin** PowerShell and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

If you're using **Command Prompt** instead of PowerShell:
```cmd
py -m venv .venv
.\.venv\Scripts\activate.bat
```

**macOS (Terminal / zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Linux (bash / zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> On Ubuntu/Debian, if you get `ensurepip is not available`, install
> the venv module first: `sudo apt install python3.11-venv`.

Your prompt should now show `(.venv)` at the start. To leave the venv
later, run `deactivate`.

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Flask, scikit-learn, pandas, Pillow, NLTK, joblib, pytest,
Jupyter and their transitive deps. Takes ~1–2 minutes on a fresh venv.

Verify with:
```bash
python -c "import flask, sklearn, pandas, PIL, nltk; print('all imports ok')"
```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

This populates `~/AppData/Roaming/nltk_data/` (Windows) or
`~/nltk_data/` (macOS / Linux) with the tokenizer and stop-word lists
that the classifier and aspect extractor use. ~5 MB.

> **Why `punkt_tab` and not `punkt`?** NLTK 3.9 (mid-2024) renamed the
> punkt tokenizer resource. Using `punkt` on 3.9+ silently downloads a
> deprecated stub that fails later with `LookupError: Resource punkt_tab
> not found`.

### 5. Train the review classifier

```bash
python -m scripts.train_review_classifier
```

What this does:
- Loads `knowledge/cosmetics_beauty_products_reviews.csv` (~61 k reviews)
- Drops rows with NaN in the required feature columns
- Applies the Milestone 1 tokenizer (regex → lowercase → length ≥ 2
  → NLTK stopwords) and builds the vocabulary with the hapax + top-20
  filters from M1 Task 1
- Trains the three logistic-regression heads (BoW text, BoW title,
  numeric) on an 80/20 stratified split
- Prints per-head + fused metrics on the held-out 20 %
- Saves four artifacts to `models/`:
  - `vocab.joblib` (~120 KB — the Milestone 1 vocabulary)
  - `text_model.joblib` (~250 KB)
  - `title_model.joblib` (~250 KB)
  - `numeric_model.joblib` (~1 KB)

Expected output ends with roughly:
```
Vocabulary size: 7,592
  text_acc: 0.6770
 title_acc: 0.5241
numeric_acc: 0.5847
  fused_acc: 0.6963
   fused_f1: 0.7880
Saved 4 artifacts under .../models/
```

Takes 30–90 seconds on a modern machine. **Without these artifacts the
`/product/<id>/review` route returns HTTP 503** (training is required
before the review form can predict labels).

> **Note on the metrics.** The training corpus is class-imbalanced
> (78 % "would buy" / 22 % "would not buy"). We use
> `class_weight="balanced"`, which trades raw accuracy for F1 on the
> minority class. **Fused F1 ≈ 0.81** is the meaningful number on this
> data — raw accuracy would climb to the ~0.78 majority baseline if we
> dropped the balance correction, at the cost of minority-class recall.

### 6. Generate product images

```bash
python -m scripts.generate_product_images
```

What this does:
- Downloads two Fraunces variable-font files (~770 KB total) to
  `app/static/fonts/` on first run, cached thereafter
- Reads `knowledge/products.csv` (1 000 rows)
- Renders an 800×800 JPG for each product, designed with a
  category-tinted gradient, soft vignette, paper grain, brand
  (Fraunces Bold, ALL CAPS), product name (Fraunces Italic, wrapped),
  and a bracketed category label
- Writes them to `app/static/images/<product_id>.jpg`

Total render time: ~1 minute. Total disk usage: ~40 MB. **Without these,
the catalogue still works** — each card falls back to the CSS gradient
+ typography mark — but you won't see the proper images.

Useful flags:
```bash
# Smoke-test on 5 products first
python -m scripts.generate_product_images --limit 5

# Higher resolution / quality (for the demo video)
python -m scripts.generate_product_images --size 1200 --quality 90
```

### 7. Run the development server

The simplest invocation (works on every OS, needs no env vars):

```bash
python run.py
```

For the auto-reloading dev server with the in-browser debugger, use
the Flask CLI directly:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\flask.exe --app app run --debug
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\flask.exe --app app run --debug
```

**macOS / Linux (bash / zsh):**
```bash
./.venv/bin/flask --app app run --debug
```

You should see:
```
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

With `--debug`, the line ``* Debug mode: on`` appears and the server
auto-reloads on save.

Open <http://127.0.0.1:5000> in a browser. Stop with `Ctrl+C`.

> On macOS, port 5000 is sometimes claimed by AirPlay Receiver. If you
> get a `Address already in use` error, either disable AirPlay Receiver
> (System Settings → General → AirDrop & Handoff → AirPlay Receiver
> off), or run on a different port:
> `flask run --debug --port 5001`.

---

## Running the test suite

```bash
pytest
```

Expected output: `68 passed in ~15s`. Distribution:

| Test file | Count |
|---|---|
| `tests/test_preprocessing.py` | 8 |
| `tests/test_data_loader.py` | 2 |
| `tests/test_search.py` | 13 |
| `tests/test_classifier.py` | 6 |
| `tests/test_review_store.py` | 5 |
| `tests/test_aspect_extractor.py` | 4 |
| `tests/test_train_script.py` | 1 |
| `tests/test_train_review_classifier.py` | 1 |
| `tests/test_generate_product_images.py` | 1 |
| `tests/test_app.py` | 4 |
| `tests/test_search_routes.py` | 14 |
| `tests/test_review_routes.py` | 9 |

The route tests use a session-scoped fixture that trains a tiny
classifier on a 30-row synthetic dataset (so tests don't depend on
disk artifacts from step 5). Each write-test gets its own
`tmp_path/data/` for isolation.

To run a single file or test:
```bash
pytest tests/test_search.py -v
pytest tests/test_search.py::test_typo_brand_match -v
```

---

## The notebook (`notebooks/milestone2.ipynb`)

A self-contained narrative documenting the four NLP techniques behind
the app (typo-tolerant search, three-head fusion classifier,
similar-item recommendations, aspect extraction). Each section
explains *why* the chosen approach was picked, the alternatives we
rejected and why, a worked example with real data, analysis,
limitations, the corresponding `app/nlp/` module that ships the
logic to the live app, and (where applicable) the Milestone 1 cell
or section that motivated the design.

To launch Jupyter and open it:

```bash
.\.venv\Scripts\jupyter.exe notebook notebooks/milestone2.ipynb
```

Re-running every cell from top to bottom reproduces every reported
metric in ≤ 180 seconds on a modern laptop (dominated by Section 2's
classifier training on the 61k Nykaa corpus).

The notebook source is generated by `notebooks/_build.py`. To
regenerate after editing the builder:

```bash
.\.venv\Scripts\python.exe notebooks/_build.py
.\.venv\Scripts\jupyter.exe nbconvert --clear-output --inplace notebooks/milestone2.ipynb
```

---

## Routes / URLs exposed

| Method | Path | Renders |
|---|---|---|
| `GET` | `/` | Home — hero + 5 category sections (4 products each) with **See all N →** links |
| `GET` | `/search?q=...` | Search results — top-30 ranked products + count headline |
| `GET` | `/category/<slug>?page=N` | Category browse — 24 products per page with pagination |
| `GET` | `/product/<id>` | Product detail — hero, **Customers mention**, **Similar items**, reviews |
| `GET` | `/product/<id>/review` | Empty review form |
| `POST` | `/product/<id>/review` | Two-step submit (predict → confirm → save) |
| `GET` | `/static/css/styles.css` | Custom design system |
| `GET` | `/static/images/<id>.jpg` | Generated product image (404 if not yet rendered) |

Category slugs: `skincare`, `makeup`, `fragrance`, `beauty-tools`, `haircare`.

---

## Project structure

```
nlp-web/
├── README.md / README.txt    # this file / short submission-required file
├── requirements.txt          # locked deps
├── pyproject.toml            # pytest config (testpaths = ["tests"])
├── run.py                    # `python run.py` launches the Flask dev server
├── .gitignore                # venv, caches, fonts, images, model artifacts
│
├── knowledge/                # IMMUTABLE source data
│   ├── products.csv          # 1 000 products (the live catalogue)
│   ├── reviews.csv           # 500 seed reviews
│   └── cosmetics_beauty_products_reviews.csv   # 61 k Nykaa training corpus
│
├── data/                     # RUNTIME data (gitignored)
│   └── reviews.csv           # appended by /product/<id>/review submissions
│
├── models/                   # generated, gitignored
│   ├── vocab.joblib          # Milestone 1 vocabulary (str -> int)
│   ├── text_model.joblib     # Head A: BoW + LR over review_text
│   ├── title_model.joblib    # Head B: BoW + LR over review_title
│   └── numeric_model.joblib  # Head C: StandardScaler + LR over [rating, log_price]
│
├── app/                      # Flask application package
│   ├── __init__.py           # create_app() factory + all routes
│   ├── nlp/                  # NLP library (used by routes AND scripts/)
│   │   ├── preprocessing.py      # normalize(text) → tokens
│   │   ├── data_loader.py        # load_products(), load_reviews()
│   │   ├── search.py             # SearchIndex (Task 1 + Task 3)
│   │   ├── classifier.py         # ReviewClassifier (Task 2)
│   │   ├── review_store.py       # CSV-append persistence
│   │   └── aspect_extractor.py   # AspectExtractor (Task 4)
│   ├── templates/
│   │   ├── base.html             # layout, header, footer
│   │   ├── _macros.html          # product_card / similar_card / stars
│   │   ├── index.html            # home with category sections
│   │   ├── category.html         # paginated category browse
│   │   ├── search.html           # search results
│   │   ├── product.html          # product detail
│   │   └── review_form.html      # two-state submission form
│   └── static/
│       ├── css/styles.css        # ~750 lines, fully custom
│       ├── fonts/                # downloaded by image script (gitignored)
│       └── images/               # 1 000 JPGs (gitignored)
│
├── scripts/                  # offline content prep + model training
│   ├── train_classifier.py             # scaffold stub
│   ├── train_review_classifier.py      # produces models/*.joblib
│   └── generate_product_images.py      # produces app/static/images/*.jpg
│
├── notebooks/                # Milestone-I Jupyter notebook(s)
│
└── tests/                    # ALL tests (pytest discovers everything here)
```

---

## How it works

**Application factory.** `app/__init__.py:create_app(classifier=None, data_dir=None)`
builds everything once at startup:

1. Loads `products` and `reviews` DataFrames via `app.nlp.data_loader`.
2. Constructs the `SearchIndex` (char-n-gram TF-IDF over name + category).
3. Builds `reviews_by_product` — a dict `{product_id: [review, ...]}`
   for O(1) detail-page lookup.
4. Builds the `ReviewStore` against `data/reviews.csv`.
5. Builds the `AspectExtractor` over `reviews_by_product`.
6. Either uses the `classifier` argument (injected by tests) or loads
   four `joblib` artifacts from `models/` (`vocab.joblib` +
   `text_model.joblib` + `title_model.joblib` + `numeric_model.joblib`).
   If `models/` is empty, `app.config["classifier"] = None` and the
   review-form route returns 503 until you run the training script.
7. Scans `app/static/images/` once to populate `available_images:
   set[int]`. Templates use this to decide whether to render an `<img>`
   or fall back to the CSS gradient placeholder.

**Persistence.** New reviews are appended to `data/reviews.csv`. On
first append, the seed `knowledge/reviews.csv` is migrated forward
(adding `title` and `predicted_label` columns). The data-loader prefers
`data/` over `knowledge/` for `reviews.csv`, so subsequent loads see
the appended rows.

**Testing strategy.** Most tests inject a tiny in-memory classifier so
the suite doesn't depend on the disk artifacts produced by step 5.
Write-tests get function-scoped `tmp_path` data directories so each
test is isolated. The training script is exercised via subprocess
against a 40-row synthetic CSV.

---

## Resetting state

Cross-platform commands (run from the repo root, venv active):

| Action | Command |
|---|---|
| Re-train the classifier | `python -m scripts.train_review_classifier` |
| Re-render images at higher resolution | `python -m scripts.generate_product_images --size 1200 --quality 90` |

File-system clean-up commands differ by shell:

**Windows (PowerShell):**
```powershell
Remove-Item data\reviews.csv             # clear submitted reviews (keep seed)
Remove-Item -Recurse .pytest_cache       # clear pytest cache
Remove-Item -Recurse .venv               # remove venv (then re-run setup from step 2)
Remove-Item -Recurse app\static\images  # force re-render images on next run
```

**macOS / Linux (bash / zsh):**
```bash
rm data/reviews.csv                      # clear submitted reviews (keep seed)
rm -rf .pytest_cache                     # clear pytest cache
rm -rf .venv                             # remove venv (then re-run setup from step 2)
rm -rf app/static/images                 # force re-render images on next run
```

---

## Troubleshooting

**`flask: command not found` after activating the venv**
The venv probably isn't active. Verify `(.venv)` is in your prompt, or
invoke pytest/flask explicitly:
```powershell
.\.venv\Scripts\flask.exe --app app run --debug
```

**`/product/<id>/review` returns 503**
You haven't trained the classifier yet — run step 5.

**Product cards show plain gradients instead of designed images**
You haven't generated the images yet — run step 6, then restart Flask
(image discovery happens at `create_app()`, not per-request).

**`ModuleNotFoundError: No module named 'app.nlp.X'`**
You're probably running a script from somewhere other than the repo
root. All commands should be run from the **repo root** so Python sees
the `app/`, `scripts/`, and `tests/` packages.

**`ValueError: Input X contains NaN` during training**
The Nykaa corpus contains 9 NaN `review_text` rows and one NaN
`review_rating` row. The classifier drops these in `train()` — make
sure you're on a current build (commit `d77c510` or later).

**`pytest` says `0 tests collected`**
You're running pytest outside the repo root. `pyproject.toml` declares
`testpaths` so you need to invoke pytest from the root.

**NLTK download fails**
Some corporate networks block `raw.githubusercontent.com`. Try a personal
network or pre-download from <https://www.nltk.org/nltk_data/>.

**Generated images look misaligned / fonts not rendering**
The font download in step 6 may have failed partially. Delete
`app/static/fonts/` and re-run the script.

**Tests fail with "Permission denied" on Windows**
Antivirus software occasionally locks `.joblib` files mid-write. Disable
real-time scanning for the repo directory or run the venv from outside
common AV-protected paths like `Desktop` / `Documents`.

---

## Acknowledgments

- **Training data**: [Nykaa cosmetics product reviews](https://www.kaggle.com/) — used in Milestone I and re-used here as the supervised corpus.
- **Typography**: [Fraunces](https://fonts.google.com/specimen/Fraunces) (Phaedra Charles & Lizy Gershenzon) and Manrope, both under the SIL Open Font License.
- **Course**: COSC3801/3015 *Advanced Programming for Data Science*, RMIT University.
- **Stack**: Flask, scikit-learn, pandas, Pillow, NLTK, joblib, pytest, Jupyter — and the maintainers of every library in `requirements.txt`.

---

*Built with the Maison de Beauté design system — Volume 01, Spring 2026.*
