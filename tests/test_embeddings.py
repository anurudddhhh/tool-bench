"""Unit tests for shared embeddings helper and on-disk vector cache (F11h)."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from toolbench.embeddings import (
    EmbeddingHelper,
    batch_cosine_similarity,
    cosine_similarity,
    sanitize_model_dir_name,
)


def test_sanitize_model_dir_name() -> None:
    """Verify model directory name sanitization removes illegal filesystem characters."""
    assert (
        sanitize_model_dir_name("sentence-transformers/all-MiniLM-L6-v2")
        == "sentence-transformers_all-MiniLM-L6-v2"
    )
    assert sanitize_model_dir_name("model:v1.0") == "model_v1.0"
    assert sanitize_model_dir_name("model name with spaces") == "model_name_with_spaces"


def test_cosine_similarity_edge_cases() -> None:
    """Verify cosine similarity mathematical properties and edge case handling."""
    # Identical vectors
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(v1, v1), 1e-6) == 1.0

    # Orthogonal vectors
    v_x = np.array([1.0, 0.0], dtype=np.float32)
    v_y = np.array([0.0, 1.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(v_x, v_y), 1e-6) == 0.0

    # Opposite vectors
    v_neg = np.array([-1.0, 0.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(v_x, v_neg), 1e-6) == -1.0

    # Zero vector handling
    v_zero = np.array([0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(v_x, v_zero) == 0.0
    assert cosine_similarity(v_zero, v_zero) == 0.0


def test_batch_cosine_similarity() -> None:
    """Verify vectorized batch cosine similarity calculation."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = np.array(
        [
            [1.0, 0.0],   # Identical -> 1.0
            [0.0, 1.0],   # Orthogonal -> 0.0
            [-1.0, 0.0],  # Opposite -> -1.0
            [0.0, 0.0],   # Zero -> 0.0
        ],
        dtype=np.float32,
    )

    scores = batch_cosine_similarity(query, corpus)
    assert scores.shape == (4,)
    assert pytest.approx(scores[0], 1e-6) == 1.0
    assert pytest.approx(scores[1], 1e-6) == 0.0
    assert pytest.approx(scores[2], 1e-6) == -1.0
    assert pytest.approx(scores[3], 1e-6) == 0.0

    # Empty corpus
    empty_corpus = np.empty((0, 2), dtype=np.float32)
    empty_scores = batch_cosine_similarity(query, empty_corpus)
    assert empty_scores.shape == (0,)

    # Dimension mismatch
    bad_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        batch_cosine_similarity(bad_query, corpus)
    assert "Dimension mismatch" in str(exc_info.value)


def test_embedding_helper_caching_and_persistence(tmp_path: Path) -> None:
    """Verify on-disk vector caching avoids recomputing existing text embeddings."""
    cache_dir = tmp_path / "cache"
    helper = EmbeddingHelper(cache_dir=cache_dir)

    # Inject mock model to count encode calls
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    helper._model = mock_model
    helper._dim = 3

    # 1. First embed call: cache miss, invokes model.encode
    vecs1 = helper.embed_texts(["fetch webpage content"])
    assert vecs1.shape == (1, 3)
    assert mock_model.encode.call_count == 1

    # Check cache file was created on disk
    npy_files = list(cache_dir.glob("*.npy"))
    assert len(npy_files) == 1

    # 2. Second embed call with same text: cache hit, does NOT call model.encode
    vecs2 = helper.embed_texts(["fetch webpage content"])
    assert vecs2.shape == (1, 3)
    assert np.allclose(vecs1, vecs2)
    # Call count remains 1!
    assert mock_model.encode.call_count == 1

    # 3. Third call with mixed texts (one cached, one new)
    mock_model.encode.return_value = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)
    vecs3 = helper.embed_texts(["fetch webpage content", "write to disk"])
    assert vecs3.shape == (2, 3)
    # Only "write to disk" was passed to encode
    assert mock_model.encode.call_count == 2
    args, kwargs = mock_model.encode.call_args
    assert args[0] == ["write to disk"]
    assert len(list(cache_dir.glob("*.npy"))) == 2


def test_embedding_helper_empty_list(tmp_path: Path) -> None:
    """Verify empty list input returns shape (0, dim) without errors."""
    helper = EmbeddingHelper(cache_dir=tmp_path / "cache")
    helper._dim = 384
    result = helper.embed_texts([])
    assert result.shape == (0, 384)
    assert result.dtype == np.float32


def test_embedding_helper_corrupted_cache_recovery(tmp_path: Path) -> None:
    """Verify corrupted .npy file is treated as a miss and recomputed."""
    cache_dir = tmp_path / "cache"
    helper = EmbeddingHelper(cache_dir=cache_dir)

    # Write corrupt non-npy content to target path
    corrupt_path = helper._compute_cache_path("corrupted entry")
    corrupt_path.write_text("corrupted junk", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.7, 0.8]], dtype=np.float32)
    helper._model = mock_model
    helper._dim = 2

    res = helper.embed_texts(["corrupted entry"])
    assert res.shape == (1, 2)
    assert mock_model.encode.call_count == 1
