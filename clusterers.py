"""
Clustering algorithms.

Clustering is responsible ONLY for assigning cluster labels to embeddings.
It does NOT perform preprocessing, embedding, or evaluation.

IMPORTANT USAGE NOTE:
  - Avoid applying GMM directly on sparse TF-IDF vectors.  Densify first
    via TruncatedSVD (or another reducer) before using GMMClusterer.
"""

from __future__ import annotations

import logging

import numpy as np

from interfaces import BaseClusterer
from config import ClusteringConfig

logger = logging.getLogger(__name__)


def build_clusterer(config: ClusteringConfig) -> BaseClusterer:
    """Factory: return the correct BaseClusterer for *config.algorithm*."""
    algorithm = config.algorithm.lower()
    dispatch = {
        "kmeans": KMeansClusterer,
        "agglomerative": AgglomerativeClusterer,
        "gmm": GMMClusterer,
    }
    if algorithm not in dispatch:
        raise ValueError(
            f"Unknown clustering algorithm '{algorithm}'. "
            f"Choose from: {list(dispatch)}"
        )
    return dispatch[algorithm](config)


# ---------------------------------------------------------------------------
# Concrete clusterers
# ---------------------------------------------------------------------------

class KMeansClusterer(BaseClusterer):
    """K-Means clustering."""

    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config
        self._model = None

    def fit(self, embeddings: np.ndarray) -> "KMeansClusterer":
        from sklearn.cluster import KMeans

        self._model = KMeans(
            n_clusters=self._config.n_clusters,
            random_state=self._config.seed,
            n_init=self._config.extra_params.get("n_init", 10),
            max_iter=self._config.extra_params.get("max_iter", 300),
        )
        self._model.fit(embeddings)
        logger.info(
            "KMeans fitted | n_clusters=%d | inertia=%.4f",
            self._config.n_clusters,
            self._model.inertia_,
        )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict(embeddings)

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._model.cluster_centers_


class AgglomerativeClusterer(BaseClusterer):
    """Agglomerative (hierarchical) clustering."""

    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config
        self._labels: np.ndarray | None = None
        self._model = None

    def fit(self, embeddings: np.ndarray) -> "AgglomerativeClusterer":
        from sklearn.cluster import AgglomerativeClustering

        self._model = AgglomerativeClustering(
            n_clusters=self._config.n_clusters,
            linkage=self._config.extra_params.get("linkage", "ward"),
            metric=self._config.extra_params.get("metric", "euclidean"),
        )
        self._labels = self._model.fit_predict(embeddings)
        logger.info(
            "Agglomerative clustering fitted | n_clusters=%d",
            self._config.n_clusters,
        )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        # Agglomerative clustering is transductive; return stored labels.
        if self._labels is None:
            raise RuntimeError("Call fit() before predict().")
        return self._labels


class GMMClusterer(BaseClusterer):
    """Gaussian Mixture Model clustering.

    WARNING: Do NOT apply GMM on raw sparse TF-IDF vectors.
    Always reduce dimensionality first (e.g., with TruncatedSVD → PCA).
    """

    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config
        self._model = None

    def fit(self, embeddings: np.ndarray) -> "GMMClusterer":
        from sklearn.mixture import GaussianMixture

        self._model = GaussianMixture(
            n_components=self._config.n_clusters,
            random_state=self._config.seed,
            covariance_type=self._config.extra_params.get("covariance_type", "full"),
            max_iter=self._config.extra_params.get("max_iter", 100),
        )
        self._model.fit(embeddings)
        bic = self._model.bic(embeddings)
        logger.info(
            "GMM fitted | n_components=%d | BIC=%.4f",
            self._config.n_clusters,
            bic,
        )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return soft cluster membership probabilities."""
        return self._model.predict_proba(embeddings)
