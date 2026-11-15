"""Unit tests for dataset curation, complete-linkage clustering, splitting, and freezing (F3)."""

import json
from pathlib import Path
from typing import Any

import pytest

from toolbench.config import SplitConfig
from toolbench.dataset import (
    VerifiedPromptRecord,
    cluster_near_duplicates,
    largest_remainder_allocation,
    review_prompt_batch,
    save_dataset_split,
    split_dataset,
)


def test_verified_prompt_record_validation() -> None:
    """Verify VerifiedPromptRecord auto-populates acceptable_args from expected_args."""
    # When acceptable_args is empty, auto-insert expected_args
    rec1 = VerifiedPromptRecord(
        prompt_id=1,
        prompt_text="Fetch data",
        expected_tool="fetch",
        expected_args={"url": "https://example.com"},
    )
    assert len(rec1.acceptable_args) == 1
    assert rec1.acceptable_args[0] == {"url": "https://example.com"}

    # When explicit acceptable_args provided, retains all entries
    rec2 = VerifiedPromptRecord(
        prompt_id=2,
        prompt_text="Fetch data",
        expected_tool="fetch",
        expected_args={"url": "https://example.com"},
        acceptable_args=[
            {"url": "https://example.com"},
            {"url": "http://example.com"},
        ],
    )
    assert len(rec2.acceptable_args) == 2


def test_cluster_near_duplicates_complete_linkage_anti_chaining() -> None:
    """Verify complete-linkage prevents chaining artifacts across prompt pairs."""
    # Prompts where A~B, B~C, C~D (> 0.8) but A≁C, A≁D (< 0.8)
    prompts = [
        {"prompt_id": 1, "prompt_text": "show data alpha beta gamma"},  # A
        {"prompt_id": 2, "prompt_text": "show data delta beta gamma"},  # B
        {"prompt_id": 3, "prompt_text": "show info delta zeta gamma"},  # C
        {"prompt_id": 4, "prompt_text": "list info delta zeta gamma"},  # D
    ]

    clusters = cluster_near_duplicates(prompts, similarity_threshold=0.8, max_cluster_size=5)

    # Must NOT produce a single 4-prompt cluster!
    assert len(clusters) > 1
    for c in clusters:
        assert len(c) < 4


def test_cluster_near_duplicates_size_cap() -> None:
    """Verify clusters respect the max_cluster_size hard cap."""
    # 7 identical prompts
    prompts = [{"prompt_id": i, "prompt_text": "Identical query"} for i in range(7)]

    clusters = cluster_near_duplicates(prompts, similarity_threshold=0.8, max_cluster_size=5)

    # Split into 2 clusters (5 and 2)
    assert len(clusters) == 2
    assert len(clusters[0]) == 5
    assert len(clusters[1]) == 2


def test_largest_remainder_allocation() -> None:
    """Verify largest-remainder method allocates quotas summing exactly to target."""
    tool_counts = {
        "tool_a": 10,
        "tool_b": 10,
        "tool_c": 10,
    }
    # Allocate 10 slots among 3 equal tools:
    # 10 * 10 / 30 = 3.333... -> floor 3, remainder 0.333...
    # Slots left = 10 - 9 = 1. Awarded to tool_a (alphabetical tie-breaker)
    quotas = largest_remainder_allocation(tool_counts, target_total=10)

    assert sum(quotas.values()) == 10
    assert quotas["tool_a"] == 4
    assert quotas["tool_b"] == 3
    assert quotas["tool_c"] == 3


def test_split_dataset_volume_validation() -> None:
    """Verify split_dataset enforces V >= heldout_size + min_dev_size."""
    few_prompts = [
        {"prompt_id": i, "prompt_text": f"P{i}", "expected_tool": "t1"} for i in range(50)
    ]
    cfg = SplitConfig(heldout_size=50, min_dev_size=40)

    with pytest.raises(ValueError) as exc_info:
        split_dataset(few_prompts, config=cfg)
    assert "Insufficient verified prompts for splitting" in str(exc_info.value)


def test_split_dataset_distinct_tools_and_singletons() -> None:
    """Verify split_dataset isolates singletons to dev and enforces >= 10 distinct heldout tools."""
    prompts: list[dict[str, Any]] = []

    # 12 tools with 7 prompts each = 84 prompts
    for t_idx in range(12):
        tool_name = f"tool_{t_idx:02d}"
        for p_idx in range(7):
            prompts.append(
                {
                    "prompt_id": len(prompts) + 1,
                    "prompt_text": f"Query {p_idx} for {tool_name}",
                    "expected_tool": tool_name,
                    "expected_args": {"arg": p_idx},
                }
            )

    # 20 singleton tools (1 prompt each) = 20 prompts
    # Total = 104 prompts >= 90
    for s_idx in range(20):
        s_tool = f"singleton_{s_idx:02d}"
        prompts.append(
            {
                "prompt_id": len(prompts) + 1,
                "prompt_text": f"Unique query for {s_tool}",
                "expected_tool": s_tool,
                "expected_args": {"arg": 0},
            }
        )

    cfg = SplitConfig(heldout_size=50, min_dev_size=40, seed=42)
    dev_set, heldout_set, hash1 = split_dataset(prompts, config=cfg)

    # Held-out constraints
    assert len(heldout_set) >= 50
    distinct_heldout = {p["expected_tool"] for p in heldout_set}
    assert len(distinct_heldout) >= 10

    # Singletons must strictly reside in dev_set!
    for p in heldout_set:
        assert not p["expected_tool"].startswith("singleton_")

    # Dev set constraints
    assert len(dev_set) >= 40
    # All prompts accounted for
    assert len(dev_set) + len(heldout_set) == len(prompts)

    # Reproducibility: same input produces exact same SHA-256 hash
    _, _, hash2 = split_dataset(prompts, config=cfg)
    assert hash1 == hash2


def test_save_dataset_split(tmp_path: Path) -> None:
    """Verify persisting split datasets and hash to disk."""
    dev_data = [{"prompt_id": 1, "prompt_text": "dev"}]
    heldout_data = [{"prompt_id": 2, "prompt_text": "heldout"}]
    hash_str = "abc123sha256hash"

    dev_f, heldout_f, hash_f = save_dataset_split(
        dev_prompts=dev_data,
        heldout_prompts=heldout_data,
        heldout_hash=hash_str,
        data_dir=tmp_path / "data",
    )

    assert dev_f.exists()
    assert heldout_f.exists()
    assert hash_f.exists()
    assert hash_f.read_text(encoding="utf-8") == hash_str


def test_review_prompt_batch(tmp_path: Path) -> None:
    """Verify review_prompt_batch handles approval, edits, and rejections."""
    out_file = tmp_path / "verified.json"

    raw_prompts = [
        {"prompt_id": 1, "prompt_text": "Q1", "expected_tool": "t1", "expected_args": {"a": 1}},
        {"prompt_id": 2, "prompt_text": "Q2", "expected_tool": "t2", "expected_args": {"b": 2}},
        {"prompt_id": 3, "prompt_text": "Q3", "expected_tool": "t3", "expected_args": {"c": 3}},
    ]

    def mock_decision(
        item: dict[str, Any], idx: int, total: int
    ) -> tuple[str, dict[str, Any] | None]:
        if idx == 1:
            return "approve", None
        if idx == 2:
            return "edit", {
                "prompt_text": "Q2 edited",
                "expected_tool": "t2",
                "expected_args": {"b": 99},
            }
        return "reject", None

    verified = review_prompt_batch(
        raw_prompts=raw_prompts,
        decision_callback=mock_decision,
        output_path=out_file,
    )

    # Item 3 was rejected; items 1 and 2 accepted
    assert len(verified) == 2
    assert verified[0].prompt_id == 1
    assert verified[0].acceptable_args == [{"a": 1}]

    assert verified[1].prompt_id == 2
    assert verified[1].prompt_text == "Q2 edited"
    assert verified[1].acceptable_args == [{"b": 99}]

    # Verify JSON saved
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(saved) == 2
