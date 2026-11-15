"""Embedding-based tool retrieval and dev-set k calibration (F5).

Provides local semantic tool retrieval using SentenceTransformers, deterministic
top-k ranking with alphabetical tie-breaking, and dev-set k-sweep calibration
to reach target recall@k thresholds.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.table import Table

from toolbench.embeddings import (
    EmbeddingHelper,
    batch_cosine_similarity,
    get_default_embedding_helper,
)
from toolbench.mcp_client import ToolInfo
from toolbench.output import console, print_info, print_success

logger = logging.getLogger(__name__)


class ToolRetriever:
    """Selects top-k most relevant tools for a user query using embedding similarity."""

    def __init__(
        self,
        tools: list[ToolInfo],
        embedding_helper: EmbeddingHelper | None = None,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        """Initialize the retriever with tools and embedding model.

        Args:
            tools: Full list of ToolInfo instances to index.
            embedding_helper: EmbeddingHelper instance (defaults to global CPU helper).
            descriptions: Optional custom tool description mapping {tool_name: description}.
                          Used in Setup D to index rewritten descriptions instead of originals.
        """
        self.tools = list(tools)
        self.helper = embedding_helper or get_default_embedding_helper()
        self.descriptions = descriptions or {}

        # Build corpus texts to embed
        tool_texts: list[str] = []
        for t in self.tools:
            # Setup D override: check custom descriptions mapping first
            custom_desc = self.descriptions.get(t.name)
            desc_to_use = custom_desc if custom_desc is not None else t.description
            # Fallback to name if description is empty
            text = desc_to_use if desc_to_use.strip() else t.name
            tool_texts.append(text)

        if tool_texts:
            logger.info("Computing or loading embeddings for %d tools...", len(tool_texts))
            self._tool_embeddings = self.helper.embed_texts(tool_texts)
        else:
            self._tool_embeddings = self.helper.embed_texts([])

    def retrieve_top_k(self, query: str, k: int) -> list[ToolInfo]:
        """Retrieve top-k tools most similar to the query.

        Rules:
        1. If available tools <= k, returns all tools.
        2. Exact tie-breaking: Sorts by (-similarity, tool_name) so boundary
           ties are broken deterministically by alphabetical name.
        3. Returns exactly min(k, len(tools)) tools.

        Args:
            query: User query string.
            k: Number of candidate tools to retrieve.

        Returns:
            List of ToolInfo instances.
        """
        if not self.tools or k <= 0:
            return []

        if len(self.tools) <= k:
            return list(self.tools)

        # 1. Embed query
        query_vec = self.helper.embed_texts([query])[0]

        # 2. Compute similarity scores
        scores = batch_cosine_similarity(query_vec, self._tool_embeddings)

        # 3. Deterministic ranking: (-score, tool_name)
        ranked_indices = sorted(
            range(len(self.tools)),
            key=lambda idx: (-scores[idx], self.tools[idx].name),
        )

        selected_indices = ranked_indices[:k]
        return [self.tools[i] for i in selected_indices]

    def compute_recall_at_k(self, prompts: list[dict[str, Any]], k: int) -> float:
        """Compute recall@k: fraction of prompts where expected tool is in top-k.

        Args:
            prompts: List of prompt dictionaries containing 'prompt_text' and 'expected_tool'.
            k: Candidate top-k size.

        Returns:
            Float recall in range [0.0, 1.0].
        """
        if not prompts:
            return 0.0

        hits = 0
        for p in prompts:
            query = str(p.get("prompt_text", ""))
            expected = str(p.get("expected_tool", ""))

            top_tools = self.retrieve_top_k(query, k=k)
            top_names = {t.name for t in top_tools}

            if expected in top_names:
                hits += 1

        return hits / len(prompts)


def render_calibration_table(
    recall_curve: dict[int, float],
    selected_k: int,
    target_recall: float,
) -> Table:
    """Format the k-sweep calibration curve into a Rich Table."""
    table = Table(title="Retrieval k-Sweep Calibration Curve (Dev Set)", show_header=True)
    table.add_column("Candidate k", justify="center", style="bold cyan")
    table.add_column("Recall@k", justify="right", style="white")
    table.add_column("Status", justify="center")

    for k, recall in sorted(recall_curve.items()):
        rec_pct = f"{recall * 100.0:.1f}%"
        if k == selected_k:
            status = "[bold green]★ Selected[/]"
            k_str = f"[bold green]{k}[/]"
            rec_str = f"[bold green]{rec_pct}[/]"
        elif recall >= target_recall:
            status = "[green]Meets Target[/]"
            k_str = str(k)
            rec_str = rec_pct
        else:
            status = "[yellow]Below Target[/]"
            k_str = str(k)
            rec_str = rec_pct

        table.add_row(k_str, rec_str, status)

    return table


def calibrate_k(
    prompts: list[dict[str, Any]],
    tools: list[ToolInfo],
    candidate_k_values: list[int] | None = None,
    target_recall: float = 0.95,
    embedding_helper: EmbeddingHelper | None = None,
    show_table: bool = True,
) -> tuple[int, dict[int, float]]:
    """Sweep k values on the dev set to select optimal k for retrieval setups C and D.

    Evaluates recall@k over candidate k values using original tool descriptions.
    Selects the smallest k achieving recall@k >= target_recall (or best recall if none
    reach target).

    Args:
        prompts: Curated dev-split prompts.
        tools: Full list of discovered tools.
        candidate_k_values: List of integer k values to sweep (default [5, 10, 15, 20, 25, 30]).
        target_recall: Threshold recall fraction (default 0.95).
        embedding_helper: Optional custom EmbeddingHelper.
        show_table: Whether to render the calibration table to the terminal.

    Returns:
        Tuple of (selected_k: int, recall_curve: dict[int, float]).
    """
    candidates = candidate_k_values or [5, 10, 15, 20, 25, 30]
    # Filter out k values larger than available tool count
    valid_candidates = sorted({k for k in candidates if k > 0})
    if not valid_candidates:
        valid_candidates = [min(5, len(tools))]

    print_info(
        f"Calibrating retrieval k across {len(prompts)} dev prompts with {len(tools)} tools..."
    )

    retriever = ToolRetriever(tools=tools, embedding_helper=embedding_helper)
    curve: dict[int, float] = {}

    for k in valid_candidates:
        recall = retriever.compute_recall_at_k(prompts, k=k)
        curve[k] = recall
        logger.debug("Sweep k=%d -> recall@%d = %.4f", k, k, recall)

    # Pick smallest k where recall >= target_recall
    selected_k = None
    for k in valid_candidates:
        if curve[k] >= target_recall:
            selected_k = k
            break

    # If none reach target_recall, pick k with highest recall (breaking ties by smallest k)
    if selected_k is None:
        best_recall = max(curve.values())
        for k in valid_candidates:
            if curve[k] == best_recall:
                selected_k = k
                break

    assert selected_k is not None

    if show_table:
        table = render_calibration_table(
            recall_curve=curve,
            selected_k=selected_k,
            target_recall=target_recall,
        )
        console.print(table)
        print_success(
            f"Selected k = {selected_k} (Recall@{selected_k} = {curve[selected_k] * 100.0:.1f}%)"
        )

    return selected_k, curve
