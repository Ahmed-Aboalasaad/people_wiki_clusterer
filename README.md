# People Wikipedia NLP Clustering Framework

A clean, modular, experiment-oriented NLP clustering framework for clustering
people biographies by profession and life-topic.

---

## Project Structure

```
nlp_clustering/
│
├── data/                          # Place people_wikipedia.csv here
├── configs/                       # Experiment YAML files
│   ├── tfidf_kmeans.yaml
│   ├── sbert_agglomerative.yaml
│   └── tfidf_gmm.yaml
│
├── outputs/                       # All experiment outputs (auto-created)
│   └── <experiment_name>/
│       ├── plots/
│       ├── metrics/
│       ├── interpretations/
│       ├── models/
│       ├── cluster_labels.npy
│       └── experiment.log
│
├── interfaces.py                  # Abstract base classes
├── config.py                      # Immutable config dataclasses
├── preprocessors.py               # Text cleaning & normalisation
├── embedders.py                   # TF-IDF and Sentence Transformer
├── reducers.py                    # PCA, TruncatedSVD, t-SNE, UMAP
├── clusterers.py                  # KMeans, Agglomerative, GMM
├── evaluators.py                  # Silhouette, Davies-Bouldin, cluster sizes
├── visualizers.py                 # Scatter plots, dendrograms, size charts
├── interpretation.py              # Keywords, centroid samples, word clouds
├── logging_utils.py               # Logging setup
├── helpers.py                     # File I/O, caching, seeds
├── run_experiment.py              # Entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Prepare data

Place your CSV file at `data/people_wikipedia.csv`.  
Required columns: `text`  
Optional columns: `name`, `uri`

### 3. Run an experiment

```bash
python run_experiment.py --config configs/tfidf_kmeans.yaml
```

All outputs are written to `outputs/<experiment_name>/`.

---

## Embedding Methods

### TF-IDF
- Produces **sparse matrices**.
- **Preferred reducer: TruncatedSVD** (LSA). TruncatedSVD works natively on
  sparse input without densifying the matrix, making it efficient for large
  vocabularies.
- Use `reduction.method: truncated_svd` in your config.

### Sentence Transformers (`all-MiniLM-L6-v2`, etc.)
- Produces **dense vectors** (~384 dims for MiniLM).
- **Preferred reducers: PCA or UMAP**.
  - PCA is fast and deterministic.
  - UMAP preserves non-linear structure and is excellent for visualisation.
- Use `reduction.method: pca` or `reduction.method: umap`.
- Use **light preprocessing** (`LightPreprocessor`) to preserve sentence
  structure — avoid lemmatisation and stopword removal.

---

## Dimensionality Reduction

| Method | Notes |
|---|---|
| `pca` | Fast, deterministic; preferred for dense embeddings |
| `truncated_svd` | Sparse-compatible; preferred for TF-IDF |
| `umap` | Non-linear; great for visualisation and clustering |
| `tsne` | **Visualisation only** — do NOT use before clustering |

> **Important:** t-SNE optimises local structure and distorts global distances.
> Never apply t-SNE before clustering — it will produce misleading results.
> Use t-SNE only in `visualization.scatter_methods`.

---

## Clustering Algorithms

| Algorithm | Notes |
|---|---|
| `kmeans` | Fast, scalable, works on dense reduced embeddings |
| `agglomerative` | Hierarchical; transductive (no predict on new data) |
| `gmm` | Soft assignments; requires dense input |

> **Important:** Do NOT apply GMM directly on raw sparse TF-IDF vectors.
> Always reduce to a dense representation first (e.g., `truncated_svd` with
> `n_components: 50` or more).

---

## Experiment Outputs

Each experiment produces under `outputs/<experiment_name>/`:

| File/Dir | Contents |
|---|---|
| `cluster_labels.npy` | NumPy array of per-sample cluster IDs |
| `metrics/metrics.json` | Silhouette score, Davies-Bouldin index, cluster sizes |
| `plots/scatter_pca.png` | PCA 2-D scatter coloured by cluster |
| `plots/scatter_tsne.png` | t-SNE 2-D scatter (if configured) |
| `plots/scatter_umap.png` | UMAP 2-D scatter (if configured) |
| `plots/dendrogram.png` | Hierarchical dendrogram (subsampled) |
| `plots/cluster_sizes_histogram.png` | Bar chart of cluster sizes |
| `plots/cluster_sizes_pie.png` | Pie chart of cluster sizes |
| `interpretations/cluster_interpretations.json` | Top keywords + closest sample per cluster |
| `interpretations/wordclouds/` | Word cloud image per cluster |
| `models/clusterer.pkl` | Serialised clusterer object |
| `experiment.log` | Full experiment log |

---

## Configuration Reference

```yaml
name: my_experiment          # Experiment identifier
data_path: data/people_wikipedia.csv
seed: 42
embedding_method: tfidf      # "tfidf" | "sentence_transformer"

preprocessing:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  lemmatize: true
  token_limit: -1            # -1 = use all tokens

tfidf_embedding:
  ngram_range: [1, 2]
  max_features: 50000
  min_df: 2
  max_df: 0.95
  lowercase: true

sentence_transformer_embedding:
  model_name: all-MiniLM-L6-v2
  batch_size: 64
  token_limit: 256

reduction:
  method: truncated_svd
  n_components: 100
  extra_params: {}

clustering:
  algorithm: kmeans
  n_clusters: 15
  seed: 42
  extra_params: {}

evaluation:
  compute_silhouette: true
  compute_davies_bouldin: true
  silhouette_sample_size: 5000

visualization:
  scatter_methods: [pca, tsne]
  dendrogram_sample_size: 300
  dpi: 150
  figure_size: [12, 8]

cache:
  enabled: true
  cache_dir: outputs/cache
```

---

## Reproducibility

Every experiment config has a top-level `seed` parameter.
All random operations (numpy, sklearn, torch if available) are seeded at the
start of `run_experiment.py`.  Embedding and reduction outputs are cached
by a hash of their configuration so identical configs reuse cached artefacts.

---

## Design Principles

- **Separation of concerns**: Each module handles exactly one responsibility.
- **Immutable configs**: All config objects are frozen dataclasses.
- **Configuration-driven**: Experiments are fully specified by YAML files.
- **Interchangeable components**: All components implement abstract interfaces.
- **Reproducibility**: Seeded experiments + config-aware caching.
- **No notebooks**: Clean Python modules with type hints and docstrings.
