"""
Dimensionality reduction modules.

Reduction is responsible ONLY for reducing embedding dimensionality.
It does NOT produce embeddings or perform clustering.

IMPORTANT USAGE NOTES:
  - t-SNE is for 2-D visualisation only.  Do NOT use it before clustering
    because it does not preserve global structure.
  - TruncatedSVD is the preferred reducer for sparse TF-IDF vectors.
  - PCA and UMAP are the preferred reducers for dense Sentence Transformer
    embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.sparse import issparse

from interfaces import BaseReducer
from config import ReductionConfig

logger = logging.getLogger(__name__)


def build_reducer(config: ReductionConfig) -> BaseReducer:
    """Factory: return the correct BaseReducer for *config.method*."""
    method = config.method.lower()
    dispatch = {
        "pca": PCAReducer,
        "truncated_svd": TruncatedSVDReducer,
        "tsne": TSNEReducer,
        "umap": UMAPReducer,
    }
    if method not in dispatch:
        raise ValueError(
            f"Unknown reduction method '{method}'. "
            f"Choose from: {list(dispatch)}"
        )
    return dispatch[method](config)


# ---------------------------------------------------------------------------
# Concrete reducers
# ---------------------------------------------------------------------------

class PCAReducer(BaseReducer):
    """Principal Component Analysis (dense input required)."""

    def __init__(self, config: ReductionConfig) -> None:
        self._config = config
        self._model = None

    def fit(self, embeddings: Any) -> "PCAReducer":
        from sklearn.decomposition import PCA

        X = _to_dense(embeddings)
        self._model = PCA(
            n_components=self._config.n_components,
            random_state=self._config.extra_params.get("random_state", 42),
            **{k: v for k, v in self._config.extra_params.items() if k != "random_state"},
        )
        self._model.fit(X)
        explained = self._model.explained_variance_ratio_.sum()
        logger.info(
            "PCA fitted | n_components=%d | explained_variance=%.3f",
            self._config.n_components,
            explained,
        )
        return self

    def transform(self, embeddings: Any) -> np.ndarray:
        return self._model.transform(_to_dense(embeddings))


class TruncatedSVDReducer(BaseReducer):
    """Truncated SVD (LSA) — works natively on sparse matrices.

    This is the preferred reducer for TF-IDF embeddings.
    """

    def __init__(self, config: ReductionConfig) -> None:
        self._config = config
        self._model = None

    def fit(self, embeddings: Any) -> "TruncatedSVDReducer":
        from sklearn.decomposition import TruncatedSVD

        self._model = TruncatedSVD(
            n_components=self._config.n_components,
            random_state=self._config.extra_params.get("random_state", 42),
        )
        self._model.fit(embeddings)
        explained = self._model.explained_variance_ratio_.sum()
        logger.info(
            "TruncatedSVD fitted | n_components=%d | explained_variance=%.3f",
            self._config.n_components,
            explained,
        )
        return self

    def transform(self, embeddings: Any) -> np.ndarray:
        return self._model.transform(embeddings)


class TSNEReducer(BaseReducer):
    """t-SNE — for 2-D / 3-D visualisation ONLY.

    WARNING: Do NOT use t-SNE before clustering.  t-SNE optimises for
    local structure and distorts global distances, making any downstream
    cluster assignments unreliable.
    """

    def __init__(self, config: ReductionConfig) -> None:
        self._config = config
        self._result: np.ndarray | None = None

    def fit(self, embeddings: Any) -> "TSNEReducer":
        """t-SNE has no separate fit/transform; this is a no-op."""
        return self

    def transform(self, embeddings: Any) -> np.ndarray:
        from sklearn.manifold import TSNE

        X = _to_dense(embeddings)
        n_components = min(self._config.n_components, 3)  # t-SNE ≤ 3 in sklearn
        model = TSNE(
            n_components=n_components,
            random_state=self._config.extra_params.get("random_state", 42),
            perplexity=self._config.extra_params.get("perplexity", 30),
            n_iter=self._config.extra_params.get("n_iter", 1000),
        )
        result = model.fit_transform(X)
        logger.info("t-SNE projected to %dD | input shape: %s", n_components, X.shape)
        return result

    def fit_transform(self, embeddings: Any) -> np.ndarray:
        return self.transform(embeddings)


class UMAPReducer(BaseReducer):
    """UMAP — fast non-linear dimensionality reduction.

    Preferred for dense Sentence Transformer embeddings alongside PCA.
    """

    def __init__(self, config: ReductionConfig) -> None:
        self._config = config
        self._model = None

    def fit(self, embeddings: Any) -> "UMAPReducer":
        import umap

        X = _to_dense(embeddings)
        self._model = umap.UMAP(
            n_components=self._config.n_components,
            random_state=self._config.extra_params.get("random_state", 42),
            n_neighbors=self._config.extra_params.get("n_neighbors", 15),
            min_dist=self._config.extra_params.get("min_dist", 0.1),
        )
        self._model.fit(X)
        logger.info(
            "UMAP fitted | n_components=%d | input shape: %s",
            self._config.n_components,
            X.shape,
        )
        return self

    def transform(self, embeddings: Any) -> np.ndarray:
        return self._model.transform(_to_dense(embeddings))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_dense(X: Any) -> np.ndarray:
    """Convert sparse matrix to dense ndarray if needed."""
    if issparse(X):
        return X.toarray()
    return np.asarray(X)
