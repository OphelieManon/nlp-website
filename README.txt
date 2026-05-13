Maison de Beauté — NLP-Powered Cosmetics Shopping App
======================================================

RMIT COSC3801/3015 Assignment 3, Milestone II.

This README.txt is the brief submission summary. For the full,
GitHub-rendered guide (setup, architecture, troubleshooting, etc.) see
README.md in the same folder, or visit the repository's GitHub page.


Group members
-------------
- TODO: Name 1 — sxxxxxxx
- TODO: Name 2 — sxxxxxxx
- TODO: Name 3 — sxxxxxxx
- TODO: Name 4 — sxxxxxxx

OneDrive (files > 50 MB, if any)
--------------------------------
TODO: paste the shared link here, or write "N/A" if everything is
already in this zip.


Requirements
------------
Python 3.11 or later.

All other dependencies are declared in requirements.txt and installed
via `pip install -r requirements.txt`.


Quick start (Windows — PowerShell)
----------------------------------
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
python -m scripts.train_review_classifier
python -m scripts.generate_product_images
python run.py

If PowerShell rejects the activation script with "running scripts is
disabled on this system", run this once in an admin PowerShell:
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Then open http://127.0.0.1:5000/ in a browser.


Quick start (Windows — Command Prompt)
--------------------------------------
py -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
python -m scripts.train_review_classifier
python -m scripts.generate_product_images
python run.py


Quick start (macOS — Terminal / zsh)
------------------------------------
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
python -m scripts.train_review_classifier
python -m scripts.generate_product_images
python run.py

Python install via Homebrew if not present:
   brew install python@3.11

On macOS port 5000 may be claimed by AirPlay Receiver. If you see
"Address already in use", disable AirPlay Receiver (System Settings ->
General -> AirDrop & Handoff -> AirPlay Receiver) OR run on a different
port: flask --app app run --port 5001


Quick start (Linux — bash / zsh)
--------------------------------
# Ubuntu / Debian:
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Fedora / RHEL:
# sudo dnf install -y python3.11 python3-pip git

# Arch:
# sudo pacman -S python python-pip git

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
python -m scripts.train_review_classifier
python -m scripts.generate_product_images
python run.py


Run tests
---------
pytest

Expected: 68 passed in about 15 seconds.


Open the notebook
-----------------
.\.venv\Scripts\jupyter.exe notebook notebooks/milestone2.ipynb

The notebook narrates all four M2 NLP techniques (search,
classifier, recommendations, aspect extraction) with detailed
methodology, M1 references, and per-section limitations. Total
run time ≤ 180 seconds.


Features implemented (the four scored tasks)
--------------------------------------------
1. Typo-tolerant keyword search — `/search?q=...`
   Character-n-gram TF-IDF over product_name + category.
   Try "Maybeline" (typo) — it returns Maybelline products.

2. Review submission with auto-label classifier — `/product/<id>/review`
   Three fused logistic-regression heads on different data types
   (BoW on review_text, BoW on review_title, numeric features
   [rating, log_price]). Uses the Milestone 1 preprocessing and
   vocabulary procedure. User can override the predicted label
   before saving.

3. Similar-item recommendations — `/product/<id>`
   Cosine similarity over the catalogue TF-IDF matrix + a 0.1 boost
   for same-category items. Six related products on every detail page.

4. Aspect extraction ("Customers mention") — `/product/<id>`
   Per-product word TF-IDF over aggregated review text picks out the
   most distinctive words customers use about each item.


Notes on artifacts (not committed)
----------------------------------
The submission zip includes everything needed to run, but the repo on
GitHub gitignores three categories of generated content:

- models/*.joblib        Produced by step 5 (train_review_classifier).
                          Required for the review-form prediction.
- app/static/images/     Produced by step 6 (generate_product_images).
                          Optional — cards fall back to CSS gradients.
- app/static/fonts/      Downloaded automatically by step 6.

Re-running steps 5 and 6 regenerates everything from the data in
knowledge/. No external services or API keys are required.


For more
--------
See README.md (full setup, architecture, troubleshooting, route map).
See docs/milestone2_report.md for the design write-up of the four tasks.
