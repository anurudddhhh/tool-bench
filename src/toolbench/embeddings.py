"""Shared embeddings helper and on-disk vector cache (F11h).

Provides CPU-only embeddings using sentence-transformers, atomic on-disk vector
caching in `.cache/toolbench/embeddings/{model}/`, and vectorized cosine similarity
computations shared across F3 (near-duplicate detection), F5 (retrieval), and D1.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def sanitize_model_dir_name(model_name: str) -> str:
    """Sanitize model identifier into a safe directory path component.

    Replaces slashes, colons, and filesystem-sensitive characters with underscores.
    Example: 'sentence-transformers/all-MiniLM-L6-v2' -> 'sentence-transformers_all-MiniLM-L6-v2'
    """
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", model_name)


class EmbeddingHelper:
    """CPU-only sentence-transformers embedding helper with persistent vector caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the embedding helper.

        Args:
            model_name: Name of the sentence-transformers model (default 'all-MiniLM-L6-v2').
            cache_dir: Optional custom cache directory. Defaults to
                '.cache/toolbench/embeddings/{sanitized_model_name}'.
            device: Compute device ('cpu' strictly enforced by architectural invariant).
        """
        self.model_name = model_name
        self.device = device
        sanitized_name = sanitize_model_dir_name(model_name)
        if cache_dir is None:
            self.cache_dir = Path(".cache") / "toolbench" / "embeddings" / sanitized_name
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    def _get_model(self) -> SentenceTransformer:
        """Lazily import and initialize the SentenceTransformer model on CPU."""
        if self._model is None:
            logger.info(
                "Loading SentenceTransformer model '%s' on %s...",
                self.model_name,
                self.device,
            )
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Cache the embedding dimensionality
            sample_emb = self._model.encode(["probe"], convert_to_numpy=True)
            self._dim = int(sample_emb.shape[1])
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the vector dimensionality of the loaded model."""
        if self._dim is None:
            self._get_model()
        return self._dim or 384

    def _compute_cache_path(self, text: str) -> Path:
        """Compute the SHA-256 .npy cache file path for normalized text."""
        # Normalize whitespace
        normalized = " ".join(text.strip().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def embed_texts(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Compute or retrieve embeddings for a list of texts.

        Checks the disk cache for each text. Missing embeddings are encoded in
        batch using sentence-transformers and persisted atomically to disk.

        Args:
            texts: List of strings to encode.
            show_progress_bar: Whether to display a tqdm progress bar during encoding.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with dtype float32.
        """
        if not texts:
            dim = self.embedding_dim if self._dim is not None else 384
            return np.empty((0, dim), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        # 1. Check disk cache
        for idx, text in enumerate(texts):
            path = self._compute_cache_path(text)
            if path.exists():
                try:
                    loaded = np.load(path)
                    if loaded.ndim == 1:
                        results[idx] = loaded.astype(np.float32)
                        if self._dim is None:
                            self._dim = loaded.shape[0]
                        continue
                except Exception as err:
                    logger.warning("Corrupted vector cache at %s: %s", path, err)

            missing_indices.append(idx)
            missing_texts.append(text)

        # 2. Encode cache misses in batch
        if missing_texts:
            model = self._get_model()
            encoded_batch = model.encode(
                missing_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
                device=self.device,
            ).astype(np.float32)

            for local_idx, orig_idx in enumerate(missing_indices):
                vec = encoded_batch[local_idx]
                results[orig_idx] = vec

                # Save atomically
                target_path = self._compute_cache_path(missing_texts[local_idx])
                # np.save automatically appends .npy if extension is not .npy,
                # so use .tmp.npy to keep the suffix exact
                tmp_path = target_path.with_suffix(".tmp.npy")
                try:
                    np.save(tmp_path, vec)
                    tmp_path.replace(target_path)
                except Exception as err:
                    logger.warning("Failed to write vector cache to %s: %s", target_path, err)
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)

        return np.vstack(results).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors.

    Formula: (a . b) / (||a|| * ||b||)

    Args:
        a: 1D numpy array.
        b: 1D numpy array of matching dimension.

    Returns:
        Float similarity score in [-1.0, 1.0]. Returns 0.0 if either norm is zero.
    """
    a_flat = np.asarray(a, dtype=np.float32).ravel()
    b_flat = np.asarray(b, dtype=np.float32).ravel()

    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = float(np.dot(a_flat, b_flat))
    sim = dot / (norm_a * norm_b)
    # Clip numerical precision drift
    return float(np.clip(sim, -1.0, 1.0))


def batch_cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between a 1D query vector and a 2D corpus matrix.

    Args:
        query: 1D array of shape (D,) or (1, D).
        corpus: 2D array of shape (N, D).

    Returns:
        1D array of shape (N,) containing similarity scores in [-1.0, 1.0].
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    c = np.asarray(corpus, dtype=np.float32)

    if c.ndim != 2:
        raise ValueError(f"Corpus must be 2-dimensional (N, D), got shape {c.shape}")
    if c.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if c.shape[1] != q.shape[0]:
        raise ValueError(
            f"Dimension mismatch: query has length {q.shape[0]} but corpus has width {c.shape[1]}"
        )

    norm_q = float(np.linalg.norm(q))
    if norm_q == 0.0:
        return np.zeros((c.shape[0],), dtype=np.float32)

    norm_c = np.linalg.norm(c, axis=1)
    # Avoid zero division
    zero_mask = norm_c == 0.0
    norm_c[zero_mask] = 1.0

    dots = np.dot(c, q)
    scores = dots / (norm_c * norm_q)
    scores[zero_mask] = 0.0
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


# Global singleton instance for convenient sharing across modules
_GLOBAL_HELPER: EmbeddingHelper | None = None


def get_default_embedding_helper() -> EmbeddingHelper:
    """Return the global default EmbeddingHelper singleton."""
    global _GLOBAL_HELPER
    if _GLOBAL_HELPER is None:
        _GLOBAL_HELPER = EmbeddingHelper()
    return _GLOBAL_HELPER


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convenience function: embed texts using the global default helper."""
    return get_default_embedding_helper().embed_texts(texts)
