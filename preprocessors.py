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

    def transform(self, texts: List[str], batch_size: int = 500) -> List[str]:
        """Apply the full preprocessing pipeline to each text.

        Uses ``nlp.pipe`` for batched spaCy processing, which is significantly
        faster than calling ``nlp(text)`` one document at a time.
        """
        if self._nlp is None:
            self._load_resources()

        # Pre-clean text before handing off to spaCy (faster per-doc work)
        pre_cleaned = [self._pre_clean(t) for t in texts]

        if self._nlp is not None:
            results = self._batch_spacy(pre_cleaned, batch_size)
        else:
            results = [self._fallback_tokens(t) for t in pre_cleaned]

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

    def _pre_clean(self, text: str) -> str:
        """Fast string-level cleaning applied before spaCy."""
        if not text or not text.strip():
            return ""
        if self._config.lowercase:
            text = text.lower()
        if self._config.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _batch_spacy(self, texts: List[str], batch_size: int) -> List[str]:
        """Process all texts in batches using ``nlp.pipe`` (much faster)."""
        from tqdm import tqdm

        results = []
        pipe = self._nlp.pipe(texts, batch_size=batch_size)
        for doc in tqdm(pipe, total=len(texts), desc="Preprocessing", unit="doc", dynamic_ncols=True):
            tokens = []
            for token in doc:
                if token.is_space:
                    continue
                word = token.lemma_ if self._config.lemmatize else token.text
                tokens.append(word)

            if self._config.remove_stopwords and self._stopwords:
                tokens = [t for t in tokens if t not in self._stopwords]

            if self._config.token_limit != -1:
                tokens = tokens[: self._config.token_limit]

            results.append(" ".join(tokens))

        return results

    def _fallback_tokens(self, text: str) -> str:
        """Simple whitespace tokeniser used when spaCy is unavailable."""
        tokens = text.split()
        if self._config.remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self._config.token_limit != -1:
            tokens = tokens[: self._config.token_limit]
        return " ".join(tokens)


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
        from tqdm import tqdm
        results = [
            self._process_one(t)
            for t in tqdm(texts, desc="Preprocessing", unit="doc", dynamic_ncols=True)
        ]
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