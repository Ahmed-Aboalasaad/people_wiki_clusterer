"""
Embedding modules.

Embedding is responsible ONLY for converting processed text into vectors.
It does NOT perform preprocessing, dimensionality reduction, or clustering.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy.sparse import issparse

from interfaces import BaseEmbedder
from config import TFIDFEmbeddingConfig, SentenceTransformerEmbeddingConfig

logger = logging.getLogger(__name__)


class TFIDFEmbedder(BaseEmbedder):
    """TF-IDF vectoriser that returns sparse matrices.

    Preferred downstream reducer: TruncatedSVD (LSA), because it works
    natively on sparse input without densifying the matrix.
    """

    def __init__(self, config: TFIDFEmbeddingConfig) -> None:
        self._config = config
        self._vectoriser = None

    def embed(self, texts: List[str]):
        """Fit and transform texts into a sparse TF-IDF matrix."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectoriser = TfidfVectorizer(
            ngram_range=self._config.ngram_range,
            max_features=self._config.max_features,
            min_df=self._config.min_df,
            max_df=self._config.max_df,
            lowercase=self._config.lowercase,
        )
        matrix = self._vectoriser.fit_transform(texts)
        logger.info(
            "TF-IDF matrix shape: %s | non-zero: %d",
            matrix.shape,
            matrix.nnz,
        )
        return matrix

    @property
    def feature_names(self) -> List[str]:
        """Return vocabulary feature names (available after embed())."""
        if self._vectoriser is None:
            raise RuntimeError("Call embed() before accessing feature_names.")
        return self._vectoriser.get_feature_names_out().tolist()


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embeddings via the sentence-transformers library.

    Preferred downstream reducers: PCA or UMAP.
    """

    def __init__(self, config: SentenceTransformerEmbeddingConfig) -> None:
        self._config = config
        self._model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts into dense embedding vectors."""
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            logger.info("Loading Sentence Transformer: %s", self._config.model_name)
            self._model = SentenceTransformer(self._config.model_name)

        embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info("Sentence Transformer embeddings shape: %s", embeddings.shape)
        return embeddings
