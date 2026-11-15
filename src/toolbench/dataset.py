"""Prompt review, complete-linkage clustering, stratified splitting, and freezing (F3).

Provides human review workflow to curate verified prompt datasets, complete-linkage
near-duplicate clustering, largest-remainder stratified dataset splitting, and cryptographic
freezing with SHA-256 digest validation.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Self

from pydantic import BaseModel, Field, model_validator

from toolbench.config import SplitConfig
from toolbench.output import console, print_prompt_panel, print_success, print_warning

logger = logging.getLogger(__name__)


class VerifiedPromptRecord(BaseModel):
    """Curated, human-verified benchmark prompt record."""

    prompt_id: int
    prompt_text: str
    expected_tool: str
    expected_args: dict[str, Any] = Field(default_factory=dict)
    acceptable_args: list[dict[str, Any]] = Field(default_factory=list)
    source_tool_description: str = Field(default="")
    high_similarity: bool = Field(default=False)

    @model_validator(mode="after")
    def ensure_acceptable_args(self) -> Self:
        """Enforce invariant: acceptable_args must contain at least 1 entry."""
        if not self.acceptable_args:
            if self.expected_args:
                self.acceptable_args = [dict(self.expected_args)]
            else:
                self.acceptable_args = [{}]
        return self


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute normalized SequenceMatcher similarity ratio between two strings."""
    t1 = " ".join(text1.lower().strip().split())
    t2 = " ".join(text2.lower().strip().split())
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def cluster_near_duplicates(
    prompts: list[dict[str, Any]],
    similarity_threshold: float = 0.8,
    max_cluster_size: int = 5,
) -> list[list[int]]:
    """Group prompts into near-duplicate clusters using complete-linkage clustering.

    Rules:
    1. Complete-linkage: A prompt can only join a cluster if its similarity
       ratio exceeds similarity_threshold against EVERY existing member of the cluster.
    2. Size cap: A cluster may never exceed max_cluster_size prompts.
    3. Chains (A~B, B~C, C~D where A!~C) do NOT merge into a single cluster.

    Args:
        prompts: List of prompt dictionaries.
        similarity_threshold: SequenceMatcher ratio cutoff (default 0.8).
        max_cluster_size: Maximum prompts per cluster (default 5).

    Returns:
        List of clusters, where each cluster is a list of prompt integer indices.
    """
    clusters: list[list[int]] = []

    for i, p in enumerate(prompts):
        text = str(p.get("prompt_text", ""))
        placed = False

        for cluster in clusters:
            if len(cluster) >= max_cluster_size:
                continue

            # Complete-linkage check: must exceed threshold against ALL cluster members
            can_join = True
            for member_idx in cluster:
                member_text = str(prompts[member_idx].get("prompt_text", ""))
                sim = compute_text_similarity(text, member_text)
                if sim <= similarity_threshold:
                    can_join = False
                    break

            if can_join:
                cluster.append(i)
                placed = True
                break

        if not placed:
            clusters.append([i])

    return clusters


def largest_remainder_allocation(
    tool_counts: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    """Calculate per-tool quotas using the largest-remainder method (Hare-Niemeyer).

    Args:
        tool_counts: Dictionary mapping tool names to counts of available prompts.
        target_total: Desired total allocation.

    Returns:
        Dictionary mapping tool names to integer quotas summing exactly to target_total.
    """
    total_available = sum(tool_counts.values())
    if total_available == 0 or target_total == 0:
        return {t: 0 for t in tool_counts}

    if target_total >= total_available:
        return dict(tool_counts)

    # 1. Compute exact ideal allocations and integer floors
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []

    for tool, count in tool_counts.items():
        ideal = (target_total * count) / total_available
        fl = int(ideal)
        floors[tool] = fl
        remainders.append((ideal - fl, tool))

    # 2. Distribute remaining slots to tools with highest remainders
    # Break ties alphabetically by tool name
    sum_floors = sum(floors.values())
    slots_left = target_total - sum_floors

    remainders.sort(key=lambda item: (-item[0], item[1]))

    quotas = dict(floors)
    for i in range(slots_left):
        tool_to_increment = remainders[i][1]
        quotas[tool_to_increment] += 1

    return quotas


def split_dataset(
    verified_prompts: list[dict[str, Any]],
    config: SplitConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Split verified prompts into dev and held-out datasets with hash freezing (F3).

    Args:
        verified_prompts: Curated list of prompt records.
        config: SplitConfig detailing heldout_size, min_dev_size, seed, and cluster caps.

    Returns:
        Tuple of (dev_prompts, heldout_prompts, heldout_hash_sha256).
    """
    cfg = config or SplitConfig()
    v_count = len(verified_prompts)
    target_heldout = cfg.heldout_size
    min_dev = cfg.min_dev_size

    # Validate volume requirements
    min_required = target_heldout + min_dev
    if v_count < min_required:
        raise ValueError(
            f"Insufficient verified prompts for splitting: have {v_count}, "
            f"need at least {min_required} (held-out target {target_heldout} + min dev {min_dev})."
        )

    # Initialize seeded random number generator
    rng = random.Random(cfg.seed)

    # 1. Complete-linkage clustering
    clusters = cluster_near_duplicates(
        verified_prompts,
        similarity_threshold=cfg.similarity_threshold,
        max_cluster_size=cfg.max_cluster_size,
    )

    # Map prompt index to cluster id
    prompt_to_cluster: dict[int, int] = {}
    cluster_map: dict[int, list[int]] = {}
    for c_id, members in enumerate(clusters):
        cluster_map[c_id] = members
        for idx in members:
            prompt_to_cluster[idx] = c_id

    # 2. Identify singletons and group prompts by tool
    prompts_by_tool: dict[str, list[int]] = defaultdict(list)
    for idx, p in enumerate(verified_prompts):
        tool = str(p.get("expected_tool", ""))
        prompts_by_tool[tool].append(idx)

    singletons: set[int] = set()
    multi_tool_counts: dict[str, int] = {}

    for tool, indices in prompts_by_tool.items():
        if len(indices) == 1:
            singletons.add(indices[0])
        else:
            multi_tool_counts[tool] = len(indices)

    if len(multi_tool_counts) < 10:
        raise ValueError(
            f"Cannot split: only {len(multi_tool_counts)} tools have >= 2 prompts. "
            "Need at least 10 distinct tools with multiple prompts to satisfy "
            "held-out requirements."
        )

    # 3. Calculate per-tool quotas using largest-remainder method
    quotas = largest_remainder_allocation(multi_tool_counts, target_heldout)

    # 4. Selection of held-out prompts
    heldout_indices: set[int] = set()

    for tool, quota in quotas.items():
        if quota <= 0:
            continue
        candidates = list(prompts_by_tool[tool])
        rng.shuffle(candidates)

        selected_for_tool = 0
        for cand_idx in candidates:
            if selected_for_tool >= quota:
                break
            if cand_idx in heldout_indices:
                continue

            # Move entire cluster together to prevent near-duplicate leakage
            c_id = prompt_to_cluster[cand_idx]
            cluster_members = cluster_map[c_id]
            unadded = [
                m for m in cluster_members
                if m not in heldout_indices and m not in singletons
            ]
            # Protect dev set minimum size from cluster expansion
            if (v_count - (len(heldout_indices) + len(unadded))) < min_dev:
                continue

            for m in unadded:
                heldout_indices.add(m)
                if m in candidates:
                    selected_for_tool += 1

    # Dev set gets all remaining prompts (including singletons and all surplus)
    dev_indices = set(range(v_count)) - heldout_indices

    dev_prompts = [verified_prompts[i] for i in sorted(dev_indices)]
    heldout_prompts = [verified_prompts[i] for i in sorted(heldout_indices)]

    # Validate held-out constraints
    distinct_heldout_tools = {str(p.get("expected_tool", "")) for p in heldout_prompts}
    if len(distinct_heldout_tools) < 10:
        raise ValueError(
            f"Held-out dataset contains only {len(distinct_heldout_tools)} distinct tools. "
            "Minimum 10 distinct tools required."
        )

    if len(dev_prompts) < min_dev:
        raise ValueError(
            f"Dev dataset size ({len(dev_prompts)}) is below required minimum ({min_dev})."
        )

    # 5. Compute deterministic SHA-256 freeze hash
    heldout_json = json.dumps(heldout_prompts, indent=2, sort_keys=True)
    heldout_hash = hashlib.sha256(heldout_json.encode("utf-8")).hexdigest()

    logger.info(
        "Dataset split complete: dev=%d, heldout=%d (tools=%d), hash=%s",
        len(dev_prompts),
        len(heldout_prompts),
        len(distinct_heldout_tools),
        heldout_hash[:12],
    )
    return dev_prompts, heldout_prompts, heldout_hash


def save_dataset_split(
    dev_prompts: list[dict[str, Any]],
    heldout_prompts: list[dict[str, Any]],
    heldout_hash: str,
    data_dir: Path | str = "data",
) -> tuple[Path, Path, Path]:
    """Persist dev set, held-out set, and frozen SHA-256 hash to disk."""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_file = out_dir / "dev_prompts.json"
    heldout_file = out_dir / "heldout_prompts.json"
    hash_file = out_dir / "heldout_hash.sha256"

    dev_file.write_text(json.dumps(dev_prompts, indent=2), encoding="utf-8")
    heldout_file.write_text(json.dumps(heldout_prompts, indent=2, sort_keys=True), encoding="utf-8")
    hash_file.write_text(heldout_hash, encoding="utf-8")

    return dev_file, heldout_file, hash_file


ReviewDecisionCallback = Callable[
    [dict[str, Any], int, int], tuple[str, dict[str, Any] | None]
]


def review_prompt_batch(
    raw_prompts: list[dict[str, Any]],
    decision_callback: ReviewDecisionCallback | None = None,
    output_path: Path | str = "data/verified_prompts.json",
) -> list[VerifiedPromptRecord]:
    """Execute review loop over raw candidate prompts.

    Decisions:
    - 'a' / 'approve': Accepts prompt, populating acceptable_args = [expected_args].
    - 'e' / 'edit': Accepts edited prompt dict.
    - 'r' / 'reject': Drops prompt.

    Args:
        raw_prompts: List of raw candidate prompt dictionaries.
        decision_callback: Optional callable for automated/testing decisions.
        output_path: File path to save verified prompts JSON.

    Returns:
        List of approved VerifiedPromptRecord instances.
    """
    verified: list[VerifiedPromptRecord] = []
    total = len(raw_prompts)

    for i, p in enumerate(raw_prompts, start=1):
        prompt_id = int(p.get("prompt_id", i))
        text = str(p.get("prompt_text", ""))
        tool = str(p.get("expected_tool", ""))
        args = dict(p.get("expected_args") or {})
        desc = str(p.get("source_tool_description", ""))
        high_sim = bool(p.get("high_similarity", False))

        if decision_callback is not None:
            action, edited = decision_callback(p, i, total)
        else:
            print_prompt_panel(
                prompt_id=prompt_id,
                prompt_text=text,
                expected_tool=tool,
                expected_args=args,
                description=desc,
            )
            if high_sim:
                print_warning("High semantic similarity flag (> 0.85) detected on this prompt.")

            console.print(
                "[bold yellow]Options:[/] [bold green][A]pprove[/] | "
                "[bold cyan][E]dit[/] | [bold red][R]eject[/]"
            )
            choice = input("Enter decision (a/e/r): ").strip().lower()
            action = choice
            edited = None

        if action in ("a", "approve"):
            acceptable = p.get("acceptable_args") or [args]
            rec = VerifiedPromptRecord(
                prompt_id=prompt_id,
                prompt_text=text,
                expected_tool=tool,
                expected_args=args,
                acceptable_args=acceptable,
                source_tool_description=desc,
                high_similarity=high_sim,
            )
            verified.append(rec)
        elif action in ("e", "edit") and edited:
            e_text = str(edited.get("prompt_text", text))
            e_tool = str(edited.get("expected_tool", tool))
            e_args = dict(edited.get("expected_args", args))
            e_acc = edited.get("acceptable_args") or [e_args]
            rec = VerifiedPromptRecord(
                prompt_id=prompt_id,
                prompt_text=e_text,
                expected_tool=e_tool,
                expected_args=e_args,
                acceptable_args=e_acc,
                source_tool_description=desc,
                high_similarity=high_sim,
            )
            verified.append(rec)
        elif action in ("r", "reject"):
            logger.debug("Prompt %d rejected by reviewer.", prompt_id)

    # Persist verified records
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps([v.model_dump() for v in verified], indent=2),
        encoding="utf-8",
    )
    print_success(f"Review complete. {len(verified)} verified prompts saved to '{out_file}'.")
    return verified
