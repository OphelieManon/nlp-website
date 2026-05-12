"""Tests for the Milestone 1-aligned tokenizer."""

from app.nlp.preprocessing import tokenize


def test_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_lowercases_words():
    assert "hello" in tokenize("Hello world WORLD")
    assert "Hello" not in tokenize("Hello world")


def test_drops_tokens_shorter_than_two_chars():
    # "I" length 1 → dropped; "am" length 2 BUT is a stopword → dropped.
    # "loved" survives (not a stopword, length 5).
    out = tokenize("I am loved")
    assert "i" not in out
    assert "am" not in out
    assert "loved" in out


def test_drops_stopwords():
    # "the", "and", "is" are stopwords; "lipstick" / "gloss" / "great" are not.
    out = tokenize("the lipstick and the gloss is great")
    assert "the" not in out
    assert "and" not in out
    assert "is" not in out
    assert "lipstick" in out
    assert "gloss" in out
    assert "great" in out


def test_regex_keeps_hyphenated_words_intact():
    # M1 regex: [a-zA-Z]+(?:[-'][a-zA-Z]+)?
    # "well-known" → one token "well-known"
    out = tokenize("This product is well-known")
    assert "well-known" in out


def test_regex_keeps_apostrophe_words_intact():
    # M1 regex preserves a single internal apostrophe between letters.
    # We test with a non-stopword token so the stopword filter doesn't
    # mask the regex behaviour.
    out = tokenize("The customer's favourite")
    assert "customer's" in out


def test_regex_strips_digits_and_punctuation():
    out = tokenize("Bought 3 items for $20 — amazing!")
    assert "3" not in out
    assert "20" not in out
    assert "$" not in out
    assert "bought" in out
    assert "amazing" in out


def test_idempotent_on_already_tokenized_input():
    assert tokenize("lipstick") == ["lipstick"]
