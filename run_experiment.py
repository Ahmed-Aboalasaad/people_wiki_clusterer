"""
Experiment runner.

Usage:
    python run_experiment.py --config configs/tfidf_kmeans.yaml

Each experiment is fully defined by its YAML config file.
All outputs are saved under outputs/<experiment_name>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---- project imports ----
from config import (
    ExperimentConfig,
    PreprocessingConfig,
    TFIDFEmbeddingConfig,
    SentenceTransformerEmbeddingConfig,
    ReductionConfig,
    ClusteringConfig,
    EvaluationConfig,
    VisualizationConfig,
    CacheConfig,
)
from logging_utils import setup_logging, get_logger
from helpers import (
    load_yaml,
    save_json,
    save_pickle,
    ensure_dir,
    set_global_seed,
    config_hash,
    cache_path,
    try_load_cache,
    save_cache,
)
from preprocessors import StandardPreprocessor, LightPreprocessor
from embedders import TFIDFEmbedder, SentenceTransformerEmbedder
from reducers import build_reducer
from clusterers import build_clusterer
from evaluators import ClusteringEvaluator
from visualizers import VisualizationRunner
from interpretation import ClusterInterpreter

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_experiment_config(yaml_path: str) -> ExperimentConfig:
    """Parse a YAML file into a frozen ExperimentConfig."""
    raw = load_yaml(yaml_path)

    def _get(section: str, defaults: dict) -> dict:
        return {**defaults, **(raw.get(section) or {})}

    preprocessing_raw = _get("preprocessing", {})
    tfidf_raw = _get("tfidf_embedding", {})
    st_raw = _get("sentence_transformer_embedding", {})
    reduction_raw = _get("reduction", {})
    clustering_raw = _get("clustering", {})
    evaluation_raw = _get("evaluation", {})
    visualization_raw = _get("visualization", {})
    cache_raw = _get("cache", {})

    # scatter_methods needs to be a tuple for frozen dataclass
    if "scatter_methods" in visualization_raw:
        visualization_raw["scatter_methods"] = list(visualization_raw["scatter_methods"])

    # ngram_range needs to be a tuple
    if "ngram_range" in tfidf_raw:
        tfidf_raw["ngram_range"] = tuple(tfidf_raw["ngram_range"])

    # figure_size needs to be a tuple
    if "figure_size" in visualization_raw:
        visualization_raw["figure_size"] = tuple(visualization_raw["figure_size"])

    # extra_params defaults
    reduction_raw.setdefault("extra_params", {})
    clustering_raw.setdefault("extra_params", {})

    return ExperimentConfig(
        name=raw["name"],
        data_path=raw["data_path"],
        seed=raw.get("seed", 42),
        embedding_method=raw.get("embedding_method", "tfidf"),
        preprocessing=PreprocessingConfig(**preprocessing_raw),
        tfidf_embedding=TFIDFEmbeddingConfig(**tfidf_raw),
        sentence_transformer_embedding=SentenceTransformerEmbeddingConfig(**st_raw),
        reduction=ReductionConfig(**reduction_raw),
        clustering=ClusteringConfig(**clustering_raw),
        evaluation=EvaluationConfig(**evaluation_raw),
        visualization=VisualizationConfig(**visualization_raw),
        cache=CacheConfig(**cache_raw),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the People Wikipedia dataset CSV."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    logger.info("Loaded dataset: %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_preprocessing(
    texts: list[str],
    config: ExperimentConfig,
) -> list[str]:
    if config.embedding_method == "tfidf":
        preprocessor = StandardPreprocessor(config.preprocessing)
    else:
        preprocessor = LightPreprocessor(config.sentence_transformer_embedding)
    return preprocessor.fit_transform(texts)


def run_embedding(
    processed_texts: list[str],
    config: ExperimentConfig,
    cache_dir: Path,
    cache_enabled: bool,
):
    embed_key = config_hash(
        {"method": config.embedding_method, "tfidf": asdict(config.tfidf_embedding),
         "st": asdict(config.sentence_transformer_embedding)}
    )
    cp = cache_path(cache_dir, f"embeddings_{embed_key}")

    if cache_enabled:
        cached = try_load_cache(cp)
        if cached is not None:
            return cached

    if config.embedding_method == "tfidf":
        embedder = TFIDFEmbedder(config.tfidf_embedding)
    else:
        embedder = SentenceTransformerEmbedder(config.sentence_transformer_embedding)

    embeddings = embedder.embed(processed_texts)

    if cache_enabled:
        save_cache((embeddings, embedder), cp)

    return embeddings, embedder


def run_reduction(
    embeddings,
    config: ExperimentConfig,
    cache_dir: Path,
    cache_enabled: bool,
) -> np.ndarray:
    reduce_key = config_hash(asdict(config.reduction))
    cp = cache_path(cache_dir, f"reduced_{reduce_key}")

    if cache_enabled:
        cached = try_load_cache(cp)
        if cached is not None:
            return cached

    reducer = build_reducer(config.reduction)
    reduced = reducer.fit_transform(embeddings)

    if cache_enabled:
        save_cache(reduced, cp)

    return reduced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    config = load_experiment_config(config_path)

    # Set up logging with per-experiment log file
    exp_dir = ensure_dir(Path(config.output_dir) / config.name)
    setup_logging(log_file=str(exp_dir / "experiment.log"))
    logger.info("=== Experiment: %s ===", config.name)
    logger.info("Config loaded from: %s", config_path)

    set_global_seed(config.seed)

    # Directories
    plots_dir = ensure_dir(exp_dir / "plots")
    metrics_dir = ensure_dir(exp_dir / "metrics")
    interp_dir = ensure_dir(exp_dir / "interpretations")
    models_dir = ensure_dir(exp_dir / "models")
    cache_dir = ensure_dir(config.cache.cache_dir)

    # --- Load data ---
    df = load_dataset(config.data_path)
    texts = df["text"].fillna("").tolist()
    names = df["name"].tolist() if "name" in df.columns else None

    # --- Preprocessing ---
    logger.info("Stage: Preprocessing")
    processed_texts = run_preprocessing(texts, config)

    # --- Embedding ---
    logger.info("Stage: Embedding (%s)", config.embedding_method)
    embedding_result = run_embedding(
        processed_texts, config, cache_dir, config.cache.enabled
    )
    # run_embedding returns (embeddings, embedder) or cached tuple
    if isinstance(embedding_result, tuple) and len(embedding_result) == 2:
        embeddings, embedder = embedding_result
    else:
        embeddings, embedder = embedding_result, None

    tfidf_matrix = embeddings if config.embedding_method == "tfidf" else None
    tfidf_feature_names = (
        embedder.feature_names
        if config.embedding_method == "tfidf" and embedder is not None
        else None
    )

    # --- Dimensionality Reduction ---
    logger.info("Stage: Dimensionality Reduction (%s)", config.reduction.method)
    reduced_embeddings = run_reduction(
        embeddings, config, cache_dir, config.cache.enabled
    )

    # --- Clustering ---
    logger.info("Stage: Clustering (%s)", config.clustering.algorithm)
    clusterer = build_clusterer(config.clustering)
    labels = clusterer.fit_predict(reduced_embeddings)

    # Save labels
    labels_path = exp_dir / "cluster_labels.npy"
    np.save(labels_path, labels)
    logger.info("Cluster labels saved → %s", labels_path)

    # Save model
    save_pickle(clusterer, models_dir / "clusterer.pkl")

    # --- Evaluation ---
    logger.info("Stage: Evaluation")
    evaluator = ClusteringEvaluator(config.evaluation)
    metrics = evaluator.evaluate(
        reduced_embeddings,
        labels,
        output_path=metrics_dir / "metrics.json",
    )

    # --- Visualisation ---
    logger.info("Stage: Visualisation")
    viz_runner = VisualizationRunner(config.visualization, plots_dir)
    viz_runner.run_scatter_plots(reduced_embeddings, labels, config.name)
    viz_runner.run_dendrogram(reduced_embeddings, config.name)
    viz_runner.run_cluster_size_plots(metrics["cluster_sizes"])

    # --- Interpretation ---
    logger.info("Stage: Interpretation")
    interpreter = ClusterInterpreter(interp_dir)
    interpreter.interpret(
        embeddings=reduced_embeddings,
        labels=labels,
        texts=processed_texts,
        tfidf_matrix=tfidf_matrix,
        tfidf_feature_names=tfidf_feature_names,
        names=names,
    )

    logger.info("=== Experiment '%s' complete. Outputs in: %s ===", config.name, exp_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an NLP clustering experiment from a YAML config file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
