"""
Abstract base interfaces for all pipeline components.

Every concrete component must implement the interface defined here.
This ensures all components are interchangeable within the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class BasePreprocessor(ABC):
    """Interface for all text preprocessors."""

    @abstractmethod
    def fit(self, texts: List[str]) -> "BasePreprocessor":
        """Fit the preprocessor on the given texts (if stateful)."""

    @abstractmethod
    def transform(self, texts: List[str]) -> List[str]:
        """Transform texts using the fitted preprocessor."""

    def fit_transform(self, texts: List[str]) -> List[str]:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)


class BaseEmbedder(ABC):
    """Interface for all embedding methods."""

    @abstractmethod
    def embed(self, texts: List[str]) -> Any:
        """Convert texts into a matrix of embeddings.

        Returns either a dense np.ndarray or a sparse matrix.
        """


class BaseReducer(ABC):
    """Interface for all dimensionality reduction methods."""

    @abstractmethod
    def fit(self, embeddings: Any) -> "BaseReducer":
        """Fit the reducer on the given embeddings."""

    @abstractmethod
    def transform(self, embeddings: Any) -> np.ndarray:
        """Reduce dimensionality of embeddings."""

    def fit_transform(self, embeddings: Any) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(embeddings).transform(embeddings)


class BaseClusterer(ABC):
    """Interface for all clustering algorithms."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> "BaseClusterer":
        """Fit the clusterer on the given embeddings."""

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for the given embeddings."""

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(embeddings).predict(embeddings)


class BaseEvaluator(ABC):
    """Interface for all evaluation strategies."""

    @abstractmethod
    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Evaluate clustering quality. Returns a dict of metric name -> value."""


class BaseVisualizer(ABC):
    """Interface for all visualizers."""

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        """Produce and save a plot."""
