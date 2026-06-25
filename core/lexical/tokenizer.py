"""Lexical tokenization."""
from __future__ import annotations

import re
from typing import Callable, List

from .stemmer import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z]+|\d[\d,\.]*\d|\d")
_YEAR_RE = re.compile(r"(1\d{3}|20\d{2})")

STEM = True
ADD_BIGRAMS = True

Tokenize = Callable[[str], List[str]]


class Tokenizer:
    """Creates lexical tokens."""

    def __init__(
        self,
        *,
        stemmer: PorterStemmer | None = None,
        stem: bool = STEM,
        add_bigrams: bool = ADD_BIGRAMS,
    ) -> None:
        self._stemmer = stemmer or PorterStemmer()
        self._stem = stem
        self._add_bigrams = add_bigrams

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text.

        Args:
            text: Text to tokenize.

        Returns:
            Lexical tokens.
        """
        raw = _TOKEN_RE.findall(text.lower())
        tokens: List[str] = []
        words: List[str] = []
        for token in raw:
            if token.isalpha():
                stemmed = self._stemmer.stem(token) if self._stem else token
                tokens.append(stemmed)
                words.append(stemmed)
            else:
                tokens.append(token)
        extra = [f"{t[:3]}x" for t in raw if _YEAR_RE.fullmatch(t)]
        if self._add_bigrams:
            extra += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return tokens + extra


_DEFAULT_TOKENIZER = Tokenizer()


def tokenize(text: str) -> List[str]:
    """Tokenize text.

    Args:
        text: Text to tokenize.

    Returns:
        Lexical tokens.
    """
    return _DEFAULT_TOKENIZER.tokenize(text)
