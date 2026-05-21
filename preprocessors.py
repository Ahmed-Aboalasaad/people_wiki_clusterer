"""
Text preprocessors.

Preprocessing is responsible ONLY for:
  - lowercasing
  - punctuation removal
  - stopword removal
  - lemmatisation
  - token truncation
  - token filtering

It does NOT produce embeddings or perform any statistical fitting on the corpus.
"""

from __future__ import annotations

import logging
import re
import string
from typing import List

from interfaces import BasePreprocessor
from config import PreprocessingConfig, SentenceTransformerEmbeddingConfig

logger = logging.getLogger(__name__)


class StandardPreprocessor(BasePreprocessor):
    """Full NLP preprocessor for bag-of-words / TF-IDF pipelines.

    Applies lowercasing, punctuation removal, stopword filtering, and
    lemmatisation as configured.  Token truncation limits the number of
    tokens fed to a downstream embedder; -1 means keep all tokens.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config
        self._nlp = None          # spaCy model, loaded lazily
        self._stopwords: set[str] = set()

    # ------------------------------------------------------------------
    # BasePreprocessor interface
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "StandardPreprocessor":
        """No corpus-level state to fit; loads NLP resources."""
        self._load_resources()
        return self

    def transform(self, texts: List[str]) -> List[str]:
        """Apply the full preprocessing pipeline to each text."""
        if self._nlp is None:
            self._load_resources()
        results = [self._process_one(t) for t in texts]
        logger.info("Preprocessed %d texts", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_resources(self) -> None:
        """Lazily load spaCy and NLTK stopwords."""
        if self._nlp is not None:
            return  # already loaded

        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("Loaded spaCy model 'en_core_web_sm'")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self._nlp = None

        if self._config.remove_stopwords:
            self._stopwords = self._load_stopwords()

    @staticmethod
    def _load_stopwords() -> set[str]:
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                return set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                return set(stopwords.words("english"))
        except ImportError:
            logger.warning("NLTK not available; stopword removal disabled.")
            return set()

    def _process_one(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        if self._config.lowercase:
            text = text.lower()

        if self._config.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub(r"\s+", " ", text).strip()

        # Token-level operations via spaCy (or simple split fallback)
        if self._nlp is not None:
            tokens = self._spacy_tokens(text)
        else:
            tokens = text.split()

        if self._config.remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        # Token truncation
        if self._config.token_limit != -1:
            tokens = tokens[: self._config.token_limit]

        return " ".join(tokens)

    def _spacy_tokens(self, text: str) -> List[str]:
        doc = self._nlp(text)
        tokens = []
        for token in doc:
            if token.is_space:
                continue
            lemma = token.lemma_ if self._config.lemmatize else token.text
            tokens.append(lemma)
        return tokens


class LightPreprocessor(BasePreprocessor):
    """Minimal preprocessor designed for Sentence Transformer pipelines.

    Preserves sentence structure; avoids lemmatisation and stopword removal
    so the transformer model receives natural language input.
    Only applies: optional lowercasing, optional punctuation stripping,
    and token truncation.
    """

    def __init__(self, config: SentenceTransformerEmbeddingConfig) -> None:
        self._config = config

    def fit(self, texts: List[str]) -> "LightPreprocessor":
        return self  # stateless

    def transform(self, texts: List[str]) -> List[str]:
        results = [self._process_one(t) for t in texts]
        logger.info("Light-preprocessed %d texts", len(results))
        return results

    def _process_one(self, text: str) -> str:
        if not text:
            return ""
        if self._config.lowercase:
            text = text.lower()
        if self._config.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub(r"\s+", " ", text).strip()
        if self._config.token_limit != -1:
            words = text.split()
            text = " ".join(words[: self._config.token_limit])
        return text
