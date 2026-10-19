"""
Text preprocessing utilities for the Cranfield IR project.

Provides consistent text cleaning, tokenization, stopword removal,
and stemming for documents and search queries.
"""

import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Initialize the stemmer once and reuse it.
_STEMMER = PorterStemmer()

# Use scikit-learn's built-in English stopword list.
_STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """
    Normalize text by converting it to lowercase and removing
    punctuation and unnecessary characters.

    Args:
        text: Input text.

    Returns:
        Cleaned text as a string.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"Expected text to be a string, got {type(text).__name__}."
        )

    text = text.lower()

    # Replace non-alphanumeric characters with spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


@lru_cache(maxsize=10000)
def preprocess_text(text: str) -> list[str]:
    """
    Clean, tokenize, remove stopwords, and stem text.

    Args:
        text: Input document or query text.

    Returns:
        A list of preprocessed tokens.
    """
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    tokens = cleaned_text.split()

    processed_tokens = [
        _STEMMER.stem(token)
        for token in tokens
        if token not in _STOPWORDS
    ]

    return processed_tokens


def preprocess_to_string(text: str) -> str:
    """
    Return preprocessed tokens as a space-separated string.

    This representation can be useful for vectorizers that expect
    a string rather than a list of tokens.
    """
    return " ".join(preprocess_text(text))