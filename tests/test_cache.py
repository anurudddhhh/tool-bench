"""Unit tests for disk-based caching of LLM responses (F11a)."""

from pathlib import Path
from typing import Any

from toolbench.cache import DiskCache


def test_compute_key_deterministic() -> None:
    """Verify that identical payloads produce identical SHA-256 keys."""
    prompt = "Create a directory named test"
    tools = [
        {"name": "create_directory", "description": "Creates a directory", "parameters": {}},
    ]
    model = "gemini-2.5-flash"
    temp = 0.0

    key1 = DiskCache.compute_key(prompt, tools, model, temp)
    key2 = DiskCache.compute_key(prompt, tools, model, temp)

    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex length


def test_compute_key_tool_order_invariant() -> None:
    """Verify that permuting tool order produces the identical cache key."""
    tool_a: dict[str, Any] = {"name": "read_file", "description": "Reads a file"}
    tool_b: dict[str, Any] = {"name": "write_file", "description": "Writes a file"}

    key_ab = DiskCache.compute_key("test prompt", [tool_a, tool_b], "gemini-2.5-flash", 0.0)
    key_ba = DiskCache.compute_key("test prompt", [tool_b, tool_a], "gemini-2.5-flash", 0.0)

    assert key_ab == key_ba


def test_compute_key_sensitivity() -> None:
    """Verify that any modification to payload fields alters the key."""
    base_tools = [{"name": "read_file", "description": "Reads a file"}]
    base_key = DiskCache.compute_key("Prompt A", base_tools, "gemini-2.5-flash", 0.0)

    # Different prompt
    assert DiskCache.compute_key("Prompt B", base_tools, "gemini-2.5-flash", 0.0) != base_key
    # Different model
    assert DiskCache.compute_key("Prompt A", base_tools, "gemini-1.5-pro", 0.0) != base_key
    # Different temperature
    assert DiskCache.compute_key("Prompt A", base_tools, "gemini-2.5-flash", 0.7) != base_key
    # Different tool schema/description
    alt_tools = [{"name": "read_file", "description": "Updated description"}]
    assert DiskCache.compute_key("Prompt A", alt_tools, "gemini-2.5-flash", 0.0) != base_key


def test_set_and_get(tmp_path: Path) -> None:
    """Verify storing and retrieving response payloads."""
    cache = DiskCache(cache_dir=tmp_path / "cache")
    key = "a" * 64
    payload = {
        "chosen_tool": "read_file",
        "chosen_args": {"path": "/tmp/test.txt"},
        "prompt_tokens": 120,
        "completion_tokens": 15,
    }

    assert cache.get(key) is None
    assert not cache.has(key)
    assert cache.size() == 0

    cache.set(key, payload)

    assert cache.has(key)
    assert cache.size() == 1
    retrieved = cache.get(key)
    assert retrieved == payload


def test_get_corrupted_json_treated_as_miss(tmp_path: Path) -> None:
    """Verify that corrupt JSON files are safely handled as cache misses."""
    cache = DiskCache(cache_dir=tmp_path / "cache")
    key = "b" * 64
    bad_file = cache.cache_dir / f"{key}.json"
    bad_file.write_text("{corrupt json without closing brace", encoding="utf-8")

    assert cache.has(key)
    # Should not raise exception, must treat as miss
    assert cache.get(key) is None


def test_get_non_dict_json_treated_as_miss(tmp_path: Path) -> None:
    """Verify that non-dict JSON files are treated as cache misses."""
    cache = DiskCache(cache_dir=tmp_path / "cache")
    key = "c" * 64
    list_file = cache.cache_dir / f"{key}.json"
    list_file.write_text('["item1", "item2"]', encoding="utf-8")

    assert cache.get(key) is None


def test_atomic_overwrite(tmp_path: Path) -> None:
    """Verify overwriting an existing key safely updates content."""
    cache = DiskCache(cache_dir=tmp_path / "cache")
    key = "d" * 64

    cache.set(key, {"version": 1})
    assert cache.get(key) == {"version": 1}

    cache.set(key, {"version": 2})
    assert cache.get(key) == {"version": 2}
    assert cache.size() == 1


def test_clear(tmp_path: Path) -> None:
    """Verify clear removes all entries and temp files."""
    cache = DiskCache(cache_dir=tmp_path / "cache")
    cache.set("1" * 64, {"test": 1})
    cache.set("2" * 64, {"test": 2})

    # Simulate an orphaned temp file
    (cache.cache_dir / "temp.tmp_12345").write_text("orphan", encoding="utf-8")

    assert cache.size() == 2
    deleted = cache.clear()
    assert deleted == 3
    assert cache.size() == 0
