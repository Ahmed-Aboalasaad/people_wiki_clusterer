"""
Immutable configuration dataclasses for every pipeline stage.

Experiments are fully defined by a Config object that is frozen at creation.
No pipeline component should mutate its config after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frozen(cls):
    """Decorator: make a dataclass frozen (immutable)."""
    return dataclass(cls, frozen=True)


# ---------------------------------------------------------------------------
# Stage-level configs
# ---------------------------------------------------------------------------

@_frozen
class PreprocessingConfig:
    """Configuration for the text preprocessor."""

    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    # Maximum number of tokens to keep per document; -1 means keep all.
    token_limit: int = -1


@_frozen
class TFIDFEmbeddingConfig:
    """Configuration for TF-IDF embeddings."""

    ngram_range: Tuple[int, int] = (1, 1)
    max_features: Optional[int] = 50_000
    min_df: int = 2
    max_df: float = 0.95
    lowercase: bool = True


@_frozen
class SentenceTransformerEmbeddingConfig:
    """Configuration for Sentence Transformer embeddings."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    # Light preprocessing: preserve sentence structure
    lowercase: bool = False
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    lemmatize: bool = False
    token_limit: int = -1


@_frozen
class ReductionConfig:
    """Configuration for dimensionality reduction."""

    method: str = "pca"          # "pca" | "truncated_svd" | "tsne" | "umap"
    n_components: int = 50
    # Extra keyword arguments forwarded to the reducer constructor
    extra_params: Dict[str, Any] = field(default_factory=dict)


@_frozen
class ClusteringConfig:
    """Configuration for clustering algorithms."""

    algorithm: str = "kmeans"    # "kmeans" | "agglomerative" | "gmm"
    n_clusters: int = 10
    seed: int = 42
    # Extra keyword arguments forwarded to the clusterer constructor
    extra_params: Dict[str, Any] = field(default_factory=dict)


@_frozen
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    compute_silhouette: bool = True
    compute_davies_bouldin: bool = True
    silhouette_sample_size: Optional[int] = 5_000   # subsample for speed


@_frozen
class VisualizationConfig:
    """Configuration for visualizations."""

    # Which scatter-plot reducers to run for 2-D plots
    scatter_methods: List[str] = field(default_factory=lambda: ["pca", "tsne"])
    # For dendrograms: number of samples to draw from the dataset
    dendrogram_sample_size: int = 300
    dpi: int = 150
    figure_size: Tuple[int, int] = (12, 8)


@_frozen
class CacheConfig:
    """Configuration for the caching layer."""

    enabled: bool = True
    cache_dir: str = "outputs/cache"


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

@_frozen
class ExperimentConfig:
    """Top-level configuration object that fully defines one experiment."""

    name: str
    data_path: str
    seed: int = 42

    embedding_method: str = "tfidf"   # "tfidf" | "sentence_transformer"

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tfidf_embedding: TFIDFEmbeddingConfig = field(default_factory=TFIDFEmbeddingConfig)
    sentence_transformer_embedding: SentenceTransformerEmbeddingConfig = field(
        default_factory=SentenceTransformerEmbeddingConfig
    )
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    output_dir: str = "outputs"
