"""
Visualisation module.

Visualisation is responsible ONLY for plots and visual outputs.
All plots are saved to disk; nothing is shown interactively.

Supported plots:
  - PCA scatter
  - t-SNE scatter
  - UMAP scatter
  - Dendrogram (with sampling or pre-reduced embeddings)
  - Cluster size histogram
  - Cluster size pie chart
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from interfaces import BaseVisualizer
from config import VisualizationConfig

logger = logging.getLogger(__name__)

# Use non-interactive backend so plots can be saved without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Individual visualisers
# ---------------------------------------------------------------------------

class ScatterVisualizer(BaseVisualizer):
    """2-D scatter plot of cluster assignments after dimensionality reduction."""

    def __init__(self, config: VisualizationConfig, method: str) -> None:
        self._config = config
        self._method = method.lower()

    def plot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_path: Path,
        title: Optional[str] = None,
    ) -> None:
        """Project *embeddings* to 2-D and colour by *labels*."""
        coords_2d = self._reduce_to_2d(embeddings)
        title = title or f"{self._method.upper()} Scatter Plot"

        fig, ax = plt.subplots(figsize=self._config.figure_size)
        scatter = ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=labels,
            cmap="tab20",
            alpha=0.6,
            s=8,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        _save_figure(fig, output_path, self._config.dpi)

    def _reduce_to_2d(self, embeddings: np.ndarray) -> np.ndarray:
        method = self._method
        if method == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=42).fit_transform(embeddings)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embeddings)
        elif method == "umap":
            import umap
            return umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown scatter method: {method!r}")


class DendrogramVisualizer(BaseVisualizer):
    """Hierarchical clustering dendrogram on a sampled or pre-reduced matrix.

    Large datasets are either subsampled or should be fed pre-reduced
    embeddings to keep memory and compute manageable.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self._config = config

    def plot(
        self,
        embeddings: np.ndarray,
        output_path: Path,
        title: str = "Dendrogram",
    ) -> None:
        from scipy.cluster.hierarchy import linkage, dendrogram

        n = embeddings.shape[0]
        max_samples = self._config.dendrogram_sample_size

        if n > max_samples:
            logger.info(
                "Dendrogram: subsampling %d → %d samples", n, max_samples
            )
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_samples, replace=False)
            embeddings = embeddings[idx]

        Z = linkage(embeddings, method="ward")
        fig, ax = plt.subplots(figsize=self._config.figure_size)
        dendrogram(Z, ax=ax, no_labels=True, truncate_mode="lastp", p=30)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Distance")
        _save_figure(fig, output_path, self._config.dpi)


class ClusterSizeVisualizer(BaseVisualizer):
    """Bar/histogram and pie chart of cluster size distribution."""

    def __init__(self, config: VisualizationConfig) -> None:
        self._config = config

    def plot(
        self,
        cluster_sizes: dict,
        output_dir: Path,
        prefix: str = "cluster_sizes",
    ) -> None:
        """Save both a histogram and a pie chart for *cluster_sizes*.

        Args:
            cluster_sizes: {cluster_id: count} mapping.
            output_dir:    Directory in which to save the figures.
            prefix:        File-name prefix for both saved figures.
        """
        labels = [str(k) for k in sorted(cluster_sizes)]
        counts = [cluster_sizes[int(k)] for k in sorted(cluster_sizes)]

        # Histogram / bar chart
        fig, ax = plt.subplots(figsize=self._config.figure_size)
        ax.bar(labels, counts, color="steelblue", edgecolor="white")
        ax.set_title("Cluster Size Distribution", fontsize=14)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Number of samples")
        plt.xticks(rotation=45, ha="right")
        _save_figure(fig, output_dir / f"{prefix}_histogram.png", self._config.dpi)

        # Pie chart
        fig, ax = plt.subplots(figsize=self._config.figure_size)
        ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.set_title("Cluster Size Distribution", fontsize=14)
        _save_figure(fig, output_dir / f"{prefix}_pie.png", self._config.dpi)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

class VisualizationRunner:
    """Orchestrates all visualisation tasks for one experiment."""

    def __init__(self, config: VisualizationConfig, output_dir: Path) -> None:
        self._config = config
        self._output_dir = output_dir

    def run_scatter_plots(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        experiment_name: str,
    ) -> None:
        for method in self._config.scatter_methods:
            viz = ScatterVisualizer(self._config, method)
            path = self._output_dir / f"scatter_{method}.png"
            viz.plot(
                embeddings,
                labels,
                output_path=path,
                title=f"{experiment_name} — {method.upper()}",
            )

    def run_dendrogram(self, embeddings: np.ndarray, experiment_name: str) -> None:
        viz = DendrogramVisualizer(self._config)
        path = self._output_dir / "dendrogram.png"
        viz.plot(embeddings, output_path=path, title=f"{experiment_name} — Dendrogram")

    def run_cluster_size_plots(self, cluster_sizes: dict) -> None:
        viz = ClusterSizeVisualizer(self._config)
        viz.plot(cluster_sizes, output_dir=self._output_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)
