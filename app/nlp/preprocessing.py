"""Text preprocessing — Milestone 2 Task 2 support module.

Ports the Milestone 1 tokenizer from `knowledge/milestone1_task1.ipynb`
§1.2 into a function the Milestone 2 webapp can call at both
training time and predict time.

The same tokenizer is used at:
  - **training time** by ``scripts/train_review_classifier.py``
    when it builds the BoW vocabulary from the Nykaa corpus.
  - **predict time** by ``app/nlp/classifier.py:ReviewClassifier.predict``
    when it converts a freshly-submitted review into a BoW vector.

Using the same tokenizer in both phases is critical — any drift
would mean the predict-time tokens look out-of-vocabulary to the
model. The CountVectorizer's ``vocabulary=`` argument is what
freezes the vocabulary between training and predict; this module's
``tokenize()`` is what freezes the *segmentation* of input strings.

Milestone 1 reference: ``knowledge/milestone1_task1.ipynb`` §1.2.3
"Cleaning Pipeline & Tokenisation" and §1.2.4.3 "Removing stop words".
The hapax + top-20 vocabulary filters from §1.3 of that same M1
notebook live in the training script and the classifier, not here
— they are corpus-level filters and don't belong in a per-document
tokenizer.
"""

from __future__ import annotations

import re
from functools import lru_cache

import nltk

# The exact regex required by the M1 spec (Task 1 step 2). Letters only;
# allows internal apostrophes and hyphens between letters (so "well-known"
# stays one token, "don't" stays one token).
_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """English stopwords from NLTK, cached after first call.

    The M1 spec used a course-supplied stopwords_en.txt that we don't
    have access to in this repo. NLTK's English stopword list is a
    close substitute (it shares the same Snowball-era function-word
    base). This trade-off is explicitly called out in the notebook.
    """
    try:
        return frozenset(nltk.corpus.stopwords.words("english"))
    except LookupError:
        # First-time setup: download once and retry.
        nltk.download("stopwords", quiet=True)
        return frozenset(nltk.corpus.stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    """Tokenize a single string using the Milestone 1 pipeline.

    Steps (matching M1 task1.ipynb §1.2.3-4):
    1. Extract letter-only tokens via _TOKEN_RE.
    2. Lowercase.
    3. Drop tokens shorter than 2 characters.
    4. Drop stopwords.

    Note that vocabulary-construction filters (hapax + top-20 by
    document frequency) are applied separately in the training script
    when building the vocab — those are corpus-level filters, not
    document-level filters, so they don't belong here.
    """
    if not text:
        return []
    tokens = (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
    stopwords = _stopwords()
    return [t for t in tokens if len(t) >= 2 and t not in stopwords]
