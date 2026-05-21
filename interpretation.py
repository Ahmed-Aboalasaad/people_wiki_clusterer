"""
Interpretation module.

Interpretation is responsible ONLY for helping humans understand clusters.

For each cluster it can produce:
  - Closest sample to the centroid
  - Top TF-IDF keywords
  - Cluster size
  - Word cloud image
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from helpers import save_json

logger = logging.getLogger(__name__)


class ClusterInterpreter:
    """Generate human-readable artefacts for each discovered cluster."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        tfidf_matrix=None,          # sparse or dense; optional
        tfidf_feature_names: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
        top_k_keywords: int = 15,
    ) -> Dict[int, dict]:
        """Build an interpretation summary for every cluster.

        Args:
            embeddings:          Dense embedding matrix (n_samples, n_features).
            labels:              Cluster label per sample.
            texts:               Raw (or preprocessed) text per sample.
            tfidf_matrix:        Optional TF-IDF matrix for keyword extraction.
            tfidf_feature_names: Feature names matching *tfidf_matrix* columns.
            names:               Optional human-readable name for each sample.
            top_k_keywords:      How many top keywords to extract per cluster.

        Returns:
            Dict mapping cluster_id → {size, closest_sample, top_keywords}.
        """
        cluster_ids = sorted(set(labels))
        summaries: Dict[int, dict] = {}

        cluster_centroids = self._compute_centroids(embeddings, labels)

        for cid in cluster_ids:
            mask = labels == cid
            cluster_embeddings = embeddings[mask]
            cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
            cluster_names = (
                [names[i] for i, m in enumerate(mask) if m] if names else None
            )
            original_indices = np.where(mask)[0]

            size = int(mask.sum())
            centroid = cluster_centroids[cid]

            closest_idx_local = self._closest_to_centroid(cluster_embeddings, centroid)
            closest_idx_global = int(original_indices[closest_idx_local])

            summary: dict = {
                "cluster_id": int(cid),
                "size": size,
                "closest_sample_index": closest_idx_global,
                "closest_sample_text": cluster_texts[closest_idx_local][:500],
            }

            if cluster_names:
                summary["closest_sample_name"] = cluster_names[closest_idx_local]

            if tfidf_matrix is not None and tfidf_feature_names:
                summary["top_keywords"] = self._top_keywords(
                    tfidf_matrix, mask, tfidf_feature_names, top_k_keywords
                )

            summaries[int(cid)] = summary

        # Save textual summaries
        output_path = self._output_dir / "cluster_interpretations.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(summaries, output_path)
        logger.info("Cluster interpretations saved → %s", output_path)

        # Word clouds (optional — gracefully skipped if wordcloud not installed)
        self._save_word_clouds(labels, texts, cluster_ids)

        return summaries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_centroids(
        embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        centroids = {}
        for cid in set(labels):
            mask = labels == cid
            centroids[int(cid)] = embeddings[mask].mean(axis=0)
        return centroids

    @staticmethod
    def _closest_to_centroid(
        cluster_embeddings: np.ndarray, centroid: np.ndarray
    ) -> int:
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        return int(np.argmin(dists))

    @staticmethod
    def _top_keywords(
        tfidf_matrix,
        mask: np.ndarray,
        feature_names: List[str],
        top_k: int,
    ) -> List[str]:
        from scipy.sparse import issparse

        cluster_matrix = tfidf_matrix[mask]
        if issparse(cluster_matrix):
            mean_vec = np.asarray(cluster_matrix.mean(axis=0)).flatten()
        else:
            mean_vec = cluster_matrix.mean(axis=0)

        top_indices = mean_vec.argsort()[::-1][:top_k]
        return [feature_names[i] for i in top_indices]

    def _save_word_clouds(
        self,
        labels: np.ndarray,
        texts: List[str],
        cluster_ids: List[int],
    ) -> None:
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.info("wordcloud not installed; skipping word cloud generation.")
            return

        wc_dir = self._output_dir / "wordclouds"
        wc_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for cid in cluster_ids:
            mask = labels == cid
            cluster_text = " ".join(texts[i] for i, m in enumerate(mask) if m)
            if not cluster_text.strip():
                continue

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                collocations=False,
            ).generate(cluster_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Cluster {cid}", fontsize=14)
            path = wc_dir / f"cluster_{cid}_wordcloud.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("Word cloud saved → %s", path)
