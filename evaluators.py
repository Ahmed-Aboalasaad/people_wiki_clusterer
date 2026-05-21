"""
Evaluation module.

Evaluation is responsible ONLY for scoring and diagnostics.
It returns metric dictionaries and saves them to disk; it does NOT produce plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from interfaces import BaseEvaluator
from config import EvaluationConfig
from helpers import save_json

logger = logging.getLogger(__name__)


class ClusteringEvaluator(BaseEvaluator):
    """Computes quantitative and diagnostic metrics for a clustering result."""

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> dict:
        """Compute all configured metrics.

        Args:
            embeddings:  Dense array of shape (n_samples, n_features).
            labels:      Cluster label for each sample (shape: n_samples).
            output_path: If given, write metrics JSON to this path.

        Returns:
            Dictionary mapping metric name to value.
        """
        metrics: dict = {}

        metrics["n_clusters_found"] = int(len(set(labels)) - (1 if -1 in labels else 0))
        metrics["n_samples"] = int(len(labels))

        if self._config.compute_silhouette:
            metrics["silhouette_score"] = self._silhouette(embeddings, labels)

        if self._config.compute_davies_bouldin:
            metrics["davies_bouldin_index"] = self._davies_bouldin(embeddings, labels)

        metrics["cluster_sizes"] = self._cluster_sizes(labels)

        logger.info("Evaluation metrics: %s", {k: v for k, v in metrics.items() if k != "cluster_sizes"})

        if output_path is not None:
            save_json(metrics, output_path)
            logger.info("Metrics saved → %s", output_path)

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _silhouette(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        from sklearn.metrics import silhouette_score

        n = len(labels)
        sample_size = self._config.silhouette_sample_size
        if sample_size is not None and n > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=sample_size, replace=False)
            embeddings = embeddings[idx]
            labels = labels[idx]

        score = silhouette_score(embeddings, labels)
        logger.info("Silhouette score: %.4f", score)
        return float(score)

    @staticmethod
    def _davies_bouldin(embeddings: np.ndarray, labels: np.ndarray) -> float:
        from sklearn.metrics import davies_bouldin_score

        score = davies_bouldin_score(embeddings, labels)
        logger.info("Davies-Bouldin index: %.4f", score)
        return float(score)

    @staticmethod
    def _cluster_sizes(labels: np.ndarray) -> dict:
        unique, counts = np.unique(labels, return_counts=True)
        sizes = {int(k): int(v) for k, v in zip(unique, counts)}
        return sizes
