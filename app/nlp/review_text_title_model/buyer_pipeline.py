import html
import os
import re
import string
import unicodedata

import contractions as _contractions_lib
import emoji as _emoji_lib
import joblib
import nltk
import numpy as np
import pandas as pd
from gensim.models import FastText as _GensimFastText
from nltk.corpus import stopwords, wordnet
from nltk.corpus import words as nltk_words
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tag import pos_tag

try:
    import wordsegment as _wordsegment
    _wordsegment.load()
    _HAS_WORDSEGMENT = True
except ImportError:
    _HAS_WORDSEGMENT = False

nltk.download("wordnet",                        quiet=True)
nltk.download("omw-1.4",                        quiet=True)
nltk.download("averaged_perceptron_tagger",     quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt",                          quiet=True)
nltk.download("stopwords",                      quiet=True)
nltk.download("words",                          quiet=True)

_HERE = os.path.dirname(os.path.abspath(__file__))

FT_MODEL_PATH   = os.path.join(_HERE, "fasttext_ft_300_sg_w10_mc2_e15_n3_6.model")
LGBM_MODEL_PATH = os.path.join(_HERE, "best_lgbm_unweighted_ft.joblib")
STOPWORDS_PATH  = os.path.join(_HERE, "stopwords_en.txt")
EMBEDDING_DIM   = 300

# ── String-level helpers ───────────────────────────────────────────────────
_REPEATED_CHARS = re.compile(r"(.)\1{2,}")
_PUNCT_TABLE    = str.maketrans(
    string.punctuation.replace("-", "").replace("'", ""),
    " " * len(string.punctuation.replace("-", "").replace("'", "")),
)

def _decode_html_entities(text):
    return html.unescape(str(text)) if text is not None else ""

def _expand_contractions(text):
    return _contractions_lib.fix(str(text)) if text is not None else ""

def _normalize_repeated_chars(text):
    return _REPEATED_CHARS.sub(r"\1\1", str(text)) if text is not None else ""

def _lowercase(text):
    return str(text).lower() if text is not None else ""

def _replace_tags(text):
    return re.sub(r"@\w+", " ", str(text)) if text is not None else ""

def _replace_hashtags(text):
    return str(text).replace("#", " ") if text is not None else ""

def _remove_digits(text):
    return re.sub(r"\d+", "", str(text)) if text is not None else ""

def _remove_punctuation(text):
    return str(text).translate(_PUNCT_TABLE) if text is not None else ""

def _remove_diacritics(text):
    if text is None:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text))
        if not unicodedata.combining(c)
    )

def _remove_round_brackets(text):
    return str(text).replace("(", "").replace(")", "") if text is not None else ""

def _remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", str(text)) if text is not None else ""

def _normalize_unicode(text):
    if text is None:
        return ""
    return (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
    )

def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"): return wordnet.ADJ
    if treebank_tag.startswith("V"): return wordnet.VERB
    if treebank_tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

_lemmatizer  = WordNetLemmatizer()
_lemma_cache: dict = {}

def _safe_lemmatise(word, pos):
    key = (word, pos)
    if key not in _lemma_cache:
        lemma = _lemmatizer.lemmatize(word, pos)
        if len(word) <= 3:
            lemma = word
        if len(lemma) == 1 and len(word) > 1:
            lemma = word
        if len(word) >= 4 and len(lemma) <= len(word) - 2:
            lemma = word
        _lemma_cache[key] = lemma
    return _lemma_cache[key]

# ── Domain constants ───────────────────────────────────────────────────────
SENTIMENT_WORDS: set = {
    "not", "no", "never",
    "good", "love", "loved", "like", "nice", "great",
    "amazing", "best", "excellent", "perfect",
    "bad", "worst", "hate", "disappointed", "disappointing", "poor", "waste",
    "recommend", "recommended", "worth", "okay", "average",
}

DOMAIN_PRESERVE: set = {
    "acne", "blush", "concealer", "contour", "exfoliant", "exfoliate",
    "exfoliator", "foundation", "fragrance", "highlighter", "hydrating",
    "hydration", "hyaluronic", "lipstick", "mascara", "matte", "moisturize",
    "moisturizer", "nourishing", "oily", "pigmentation", "primer", "pore",
    "retinol", "serum", "shade", "skincare", "spf", "sunscreen", "toner",
    "vitamin", "waterproof", "squalane",
}

HYPHEN_PRESERVE: set = {
    "acne-prone", "alcohol-free", "anti-aging", "anti-blemish",
    "cruelty-free", "fast-absorbing", "fragrance-free", "full-coverage",
    "long-lasting", "long-wearing", "lightweight", "non-comedogenic",
    "non-greasy", "oil-control", "oil-free", "paraben-free",
    "semi-matte", "sensitive-skin", "skin-friendly", "sulfate-free",
    "travel-size", "ultra-hydrating", "water-proof", "water-resistant",
}

BRAND_CORRECTION_MAP: dict = {
    "maybelene": "maybelline", "maybeline": "maybelline", "maybellin": "maybelline",
    "fragnance": "fragrance",  "frangrance": "fragrance",
    "nyka": "nykaa",           "nykaaa": "nykaa",          "nykka": "nykaa",
}

PROTECTED_WORDS: set = SENTIMENT_WORDS | {
    "but", "very", "really", "too", "so",
    "skin", "hair", "shade", "face", "lip", "cream", "smell",
    "price", "quality", "dry", "smooth",
} | DOMAIN_PRESERVE

TOKEN_RE = re.compile(r"[a-zA-Z]{3,}(?:[-'][a-zA-Z]+)?")
_stemmer = SnowballStemmer("english")
_nltk_word_set: set = {w.lower() for w in nltk_words.words() if w.isalpha()}
_VOWELS: frozenset   = frozenset("aeiou")

# ── Token-level helpers ────────────────────────────────────────────────────
def _handle_hyphens(tokens):
    out = []
    for tok in tokens:
        if "-" in tok:
            out.extend([tok] if tok in HYPHEN_PRESERVE else tok.split("-"))
        else:
            out.append(tok)
    return out

def _apply_brand_corrections(tokens):
    return [BRAND_CORRECTION_MAP.get(tok, tok) for tok in tokens]

def _is_chat_abbreviation(tok, known_vocab=frozenset()):
    if not tok.isalpha() or len(tok) > 6:
        return False
    if tok in _nltk_word_set or tok in known_vocab:
        return False
    return sum(1 for c in tok if c in _VOWELS) / len(tok) < 0.30

def _stem_oov(lemma):
    if lemma not in _nltk_word_set and lemma not in DOMAIN_PRESERVE:
        stemmed = _stemmer.stem(lemma)
        if len(stemmed) >= 3 and len(stemmed) < len(lemma):
            return stemmed
    return lemma

def _load_stopwords(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return frozenset(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return frozenset(stopwords.words("english"))

_raw_stopwords = _load_stopwords(STOPWORDS_PATH)
STOPWORDS      = _raw_stopwords - SENTIMENT_WORDS

def preprocess_review(text: str) -> list:
    """Apply the full preprocessing pipeline to a single review string."""
    text = _decode_html_entities(text)
    text = _expand_contractions(text)
    text = _normalize_repeated_chars(text)
    text = _replace_tags(text)
    text = _replace_hashtags(text)
    text = _remove_round_brackets(text)
    text = _emoji_lib.demojize(str(text), delimiters=(" ", " "))
    text = re.sub(r"_+", " ", text)
    text = _remove_urls(text)
    text = _remove_digits(text)
    text = _remove_diacritics(text)
    text = _normalize_unicode(text)
    text = _remove_punctuation(text)
    text = _lowercase(text)
    text = re.sub(r"'s\b", " ", text)

    tokens = TOKEN_RE.findall(text)

    if _HAS_WORDSEGMENT:
        expanded = []
        for tok in tokens:
            if tok not in _nltk_word_set and tok not in DOMAIN_PRESERVE and len(tok) > 8:
                parts = _wordsegment.segment(tok)
                expanded.extend(parts if len(parts) > 1 else [tok])
            else:
                expanded.append(tok)
        tokens = expanded

    tokens = _handle_hyphens(tokens)
    tokens = _apply_brand_corrections(tokens)
    tokens = [t for t in tokens if t not in STOPWORDS or t in DOMAIN_PRESERVE]
    tokens = [t for t in tokens if not _is_chat_abbreviation(t, DOMAIN_PRESERVE | PROTECTED_WORDS)]
    tokens = [w for w in tokens if len(w) >= 3]
    tokens = [_stem_oov(_safe_lemmatise(w, _get_wordnet_pos(tag)))
              for w, tag in pos_tag(tokens)]
    tokens = [t for t in tokens
              if (t not in STOPWORDS or t in DOMAIN_PRESERVE) and len(t) >= 3]
    return tokens

# ── Load models ────────────────────────────────────────────────────────────
_ft           = _GensimFastText.load(FT_MODEL_PATH)
ft_wv         = _ft.wv
ft_is_subword = hasattr(ft_wv, "vectors_ngrams")

clf = joblib.load(LGBM_MODEL_PATH)

def _embed(tokens: list) -> np.ndarray:
    """Average FastText vectors for a token list."""
    vecs = (
        [ft_wv[w] for w in tokens]
        if ft_is_subword
        else [ft_wv[w] for w in tokens if w in ft_wv]
    )
    return np.mean(vecs, axis=0).astype("float32") if vecs else np.zeros(EMBEDDING_DIM, dtype="float32")


def predict(review_text: str, review_title: str = "") -> tuple:
    """Predict is_a_buyer for a single review.

    Returns
    -------
    label : int   — 1 = Buyer, 0 = Not Buyer
    proba : ndarray of shape (2,) — [P(Not Buyer), P(Buyer)]
    """
    title_tokens = preprocess_review(review_title or "")
    text_tokens  = preprocess_review(review_text  or "")
    combined     = title_tokens + text_tokens
    vec          = _embed(combined).reshape(1, -1)
    label        = int(clf.predict(vec)[0])
    proba        = clf.predict_proba(vec)[0]
    return label, proba


def predict_batch(records: list) -> pd.DataFrame:
    """Predict for a list of dicts with keys 'review_text' and 'review_title'.

    Returns a DataFrame with columns: review_text, review_title, label, label_name,
    p_not_buyer, p_buyer.
    """
    rows = []
    for r in records:
        label, proba = predict(r.get("review_text", ""), r.get("review_title", ""))
        rows.append({
            "review_title" : r.get("review_title", ""),
            "review_text"  : r.get("review_text",  "")[:80] + "…",
            "label"        : label,
            "label_name"   : "Buyer" if label == 1 else "Not Buyer",
            "p_not_buyer"  : round(float(proba[0]), 4),
            "p_buyer"      : round(float(proba[1]), 4),
        })
    return pd.DataFrame(rows)
