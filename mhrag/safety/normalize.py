"""Reproduces the preprocessing applied to `mental_health.csv` so the classifier sees the same distribution at
inference time (CHANGELOG B43).

Reverse-engineered from the data: text is lower-cased and whitespace-split, apostrophes are dropped (the data
contains "dont", "im", "ive"), tokens that exactly match the NLTK English stop-word list are dropped *before*
other punctuation is stripped (so "myself." survives as "myself" and "life.I" becomes "lifei"), then every
non-letter character is removed. This deletes "not", "no" and "nor", so the classifier sees only part of the
negation signal ("dont" survives, "not" does not); the lexicon, which runs on raw text, compensates.
"""

from __future__ import annotations

import re

# NLTK english stop-word list (nltk.corpus.stopwords.words("english"), 179 words), inlined to avoid the
# nltk dependency and its data download.
NLTK_STOPWORDS = frozenset(
    ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"]
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_NONLETTER = re.compile(r"[^a-z ]+")


def normalize_like_training(text: str) -> str:
    text = _URL.sub(" ", text.lower().replace("’", "'")).replace("'", "")
    toks = [t for t in text.split() if t not in NLTK_STOPWORDS]
    out = _NONLETTER.sub("", " ".join(toks))
    return re.sub(r"\s+", " ", out).strip()
