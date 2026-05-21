"""
General-purpose helpers used across multiple modules.

Nothing in this file should import from any other project module
(except logging_utils) to avoid circular dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise *data* as JSON and write to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)
    logger.debug("Saved JSON → %s", path)


def load_json(path: str | Path) -> Any:
    """Load JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Pickle *obj* to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug("Saved pickle → %s", path)


def load_pickle(path: str | Path) -> Any:
    """Unpickle object from *path*."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for non-standard types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def config_hash(config_dict: dict) -> str:
    """Return a short SHA-256 hex digest of a config dict."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def cache_path(cache_dir: str | Path, key: str, suffix: str = ".pkl") -> Path:
    """Return the full path for a cached artefact identified by *key*."""
    return Path(cache_dir) / f"{key}{suffix}"


def try_load_cache(path: Path) -> Any | None:
    """Return the cached object at *path*, or ``None`` if not found."""
    if path.exists():
        logger.info("Cache hit → %s", path)
        return load_pickle(path)
    return None


def save_cache(obj: Any, path: Path) -> None:
    """Save *obj* to the cache at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(obj, path)
    logger.info("Cache saved → %s", path)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.debug("Global seed set to %d", seed)
