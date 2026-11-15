"""Disk-based caching of LLM responses (F11a).

Stores responses as JSON files keyed by a deterministic SHA-256 hash of the
prompt, exposed tools, model identifier, and temperature. Prevents redundant API
calls, preserves free-tier quotas, and ensures evaluation reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DiskCache:
    """Atomic, file-based cache for LLM responses keyed by canonical SHA-256."""

    def __init__(self, cache_dir: str | Path = ".cache/toolbench/llm") -> None:
        """Initialize cache in the specified directory.

        Args:
            cache_dir: Filesystem directory path for storing cache JSON files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(
        prompt: str | list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float,
        system_prompt: str | None = None,
        sort_tools: bool = False,
    ) -> str:
        """Compute a deterministic SHA-256 hash for an LLM invocation payload.

        Preserves the exact presentation order of tools (or sorts if sort_tools=True),
        and hashes system_prompt, messages, tools, model, and temperature.

        Args:
            prompt: User prompt text or structured list of message dictionaries.
            tools: List of exposed tool definition dictionaries in exact presentation order.
            model: Target model identifier string.
            temperature: Sampling temperature float.
            system_prompt: Optional system instruction string.
            sort_tools: If True, sorts tools by name. If False, preserves presentation order.

        Returns:
            64-character hexadecimal SHA-256 digest string.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        if sort_tools:
            tools_payload = sorted(tools, key=lambda t: str(t.get("name", "")))
        else:
            tools_payload = tools

        canonical_payload = {
            "system_prompt": system_prompt or "",
            "messages": messages,
            "tools": tools_payload,
            "model": model,
            "temperature": round(float(temperature), 4),
        }

        canonical_bytes = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        return hashlib.sha256(canonical_bytes).hexdigest()

    def _get_path(self, key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        return self.cache_dir / f"{key}.json"

    def has(self, key: str) -> bool:
        """Check whether a valid cache entry exists for the given key.

        Args:
            key: SHA-256 cache key.

        Returns:
            True if the cache file exists on disk, False otherwise.
        """
        return self._get_path(key).is_file()

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached response payload by key.

        Args:
            key: SHA-256 cache key.

        Returns:
            Cached response dictionary if present and valid JSON, else None.
        """
        target_path = self._get_path(key)
        if not target_path.is_file():
            return None

        try:
            content = target_path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, dict):
                return data
            logger.warning("Cache entry %s is not a JSON object, treating as miss", key)
            return None
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("Failed to read cache file %s: %s; treating as miss", target_path, err)
            return None

    def set(self, key: str, data: dict[str, Any]) -> None:
        """Write response payload to disk atomically.

        Uses write-to-temp-and-rename semantics to ensure no partially written
        files exist if the process is terminated mid-write.

        Args:
            key: SHA-256 cache key.
            data: Response dictionary to persist.
        """
        target_path = self._get_path(key)
        temp_path = self.cache_dir / f"{key}.tmp_{uuid.uuid4().hex}"

        serialized = json.dumps(
            data,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

        try:
            temp_path.write_text(serialized, encoding="utf-8")
            # os.replace is atomic on POSIX and modern Windows NTFS
            os.replace(temp_path, target_path)
        except OSError as err:
            logger.error("Failed to commit cache file %s: %s", target_path, err)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

    def clear(self) -> int:
        """Remove all cache entries and temporary files in the cache directory.

        Returns:
            Number of files successfully deleted.
        """
        deleted_count = 0
        if not self.cache_dir.exists():
            return 0

        for file_path in self.cache_dir.iterdir():
            if file_path.is_file() and (
                file_path.name.endswith(".json") or ".tmp_" in file_path.name
            ):
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError as err:
                    logger.warning("Failed to delete cache file %s: %s", file_path, err)

        return deleted_count

    def size(self) -> int:
        """Return the count of valid cached entries (.json files)."""
        if not self.cache_dir.exists():
            return 0
        return sum(1 for p in self.cache_dir.iterdir() if p.is_file() and p.name.endswith(".json"))


# Alias matching specification naming
DiskResponseCache = DiskCache
