"""Embedding and multi-method tool retrieval and calibration (F5 & D1).

Provides local semantic tool retrieval using SentenceTransformers, BM25 lexical
matching, Reciprocal Rank Fusion (RRF) hybrid retrieval, and cross-encoder
reranking, along with dev-set calibration and method comparison.
"""

from __future__ import annotations

import logging
import re
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


# Alias for explicit naming in multi-method benchmarks
EmbeddingToolRetriever = ToolRetriever


class BM25ToolRetriever:
    """Selects top-k tools using BM25 lexical matching."""

    def __init__(
        self,
        tools: list[ToolInfo],
        descriptions: dict[str, str] | None = None,
    ) -> None:
        """Initialize BM25 retriever with tokenized tool descriptions.

        Args:
            tools: Full list of ToolInfo instances to index.
            descriptions: Optional custom tool description mapping {tool_name: description}.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("rank-bm25 is required for BM25ToolRetriever.") from exc

        self.tools = list(tools)
        self.descriptions = descriptions or {}

        corpus_texts: list[str] = []
        for t in self.tools:
            custom_desc = self.descriptions.get(t.name)
            desc_to_use = custom_desc if custom_desc is not None else t.description
            text = f"{t.name} {desc_to_use}".strip()
            corpus_texts.append(text)

        self._tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lowercase alphanumeric word tokenizer."""
        return re.findall(r"\w+", text.lower())

    def retrieve_top_k(self, query: str, k: int) -> list[ToolInfo]:
        """Retrieve top-k tools scored by BM25 with deterministic alphabetical tie-breaking."""
        if not self.tools or k <= 0:
            return []

        if len(self.tools) <= k:
            return list(self.tools)

        if not self._bm25:
            return list(self.tools[:k])

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(self.tools)),
            key=lambda idx: (-float(scores[idx]), self.tools[idx].name),
        )

        return [self.tools[i] for i in ranked_indices[:k]]

    def compute_recall_at_k(self, prompts: list[dict[str, Any]], k: int) -> float:
        """Compute recall@k: fraction of prompts where expected tool is in top-k."""
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


class HybridToolRetriever:
    """Combines Dense Embedding and BM25 Lexical retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        tools: list[ToolInfo],
        embedding_helper: EmbeddingHelper | None = None,
        descriptions: dict[str, str] | None = None,
        rrf_constant: int = 60,
    ) -> None:
        """Initialize hybrid retriever combining dense and BM25 retrievers.

        Args:
            tools: Full list of ToolInfo instances to index.
            embedding_helper: EmbeddingHelper instance for dense ranking.
            descriptions: Optional custom tool description mapping {tool_name: description}.
            rrf_constant: Smoothing constant for RRF (standard default: 60).
        """
        self.tools = list(tools)
        self.rrf_constant = rrf_constant
        self.dense_retriever = ToolRetriever(
            tools,
            embedding_helper=embedding_helper,
            descriptions=descriptions,
        )
        self.bm25_retriever = BM25ToolRetriever(tools, descriptions=descriptions)

    def retrieve_top_k(self, query: str, k: int) -> list[ToolInfo]:
        """Retrieve top-k tools ranked by Reciprocal Rank Fusion (RRF)."""
        if not self.tools or k <= 0:
            return []

        if len(self.tools) <= k:
            return list(self.tools)

        dense_ranked = self.dense_retriever.retrieve_top_k(query, k=len(self.tools))
        bm25_ranked = self.bm25_retriever.retrieve_top_k(query, k=len(self.tools))

        dense_rank_map = {t.name: rank for rank, t in enumerate(dense_ranked, start=1)}
        bm25_rank_map = {t.name: rank for rank, t in enumerate(bm25_ranked, start=1)}

        rrf_scores: dict[str, float] = {}
        for t in self.tools:
            r_dense = dense_rank_map.get(t.name, len(self.tools) + 1)
            r_bm25 = bm25_rank_map.get(t.name, len(self.tools) + 1)
            score = (1.0 / (self.rrf_constant + r_dense)) + (1.0 / (self.rrf_constant + r_bm25))
            rrf_scores[t.name] = score

        ranked_tools = sorted(
            self.tools,
            key=lambda t: (-rrf_scores[t.name], t.name),
        )

        return ranked_tools[:k]

    def compute_recall_at_k(self, prompts: list[dict[str, Any]], k: int) -> float:
        """Compute recall@k: fraction of prompts where expected tool is in top-k."""
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


class CrossEncoderToolRetriever:
    """Reranks tools using a Cross-Encoder model scoring query-tool pairs."""

    def __init__(
        self,
        tools: list[ToolInfo],
        model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        descriptions: dict[str, str] | None = None,
        scorer: Any | None = None,
    ) -> None:
        """Initialize cross-encoder reranker.

        Args:
            tools: Full list of ToolInfo instances to index.
            model_name: HuggingFace cross-encoder model identifier.
            descriptions: Optional custom tool description mapping.
            scorer: Optional pre-loaded model or callable mock accepting query-text pairs.
        """
        self.tools = list(tools)
        self.model_name = model_name
        self.descriptions = descriptions or {}
        self._scorer = scorer
        self._model: Any | None = None

    def _get_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def retrieve_top_k(self, query: str, k: int) -> list[ToolInfo]:
        """Retrieve top-k tools scored by cross-encoder with deterministic tie-breaking."""
        if not self.tools or k <= 0:
            return []

        if len(self.tools) <= k:
            return list(self.tools)

        scorer = self._get_scorer()
        pairs: list[list[str]] = []
        for t in self.tools:
            custom_desc = self.descriptions.get(t.name)
            desc_to_use = custom_desc if custom_desc is not None else t.description
            text = f"{t.name}: {desc_to_use}".strip()
            pairs.append([query, text])

        scores = scorer.predict(pairs)

        ranked_indices = sorted(
            range(len(self.tools)),
            key=lambda idx: (-float(scores[idx]), self.tools[idx].name),
        )

        return [self.tools[i] for i in ranked_indices[:k]]

    def compute_recall_at_k(self, prompts: list[dict[str, Any]], k: int) -> float:
        """Compute recall@k: fraction of prompts where expected tool is in top-k."""
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


def render_method_comparison_table(
    method_curves: dict[str, dict[int, float]],
    candidate_k_values: list[int],
) -> Table:
    """Format comparative recall@k curves across retrieval methods into a Rich Table."""
    table = Table(title="Retrieval Method Comparison (Dev Set Recall@k)", show_header=True)
    table.add_column("Retrieval Method", style="bold cyan")

    sorted_k = sorted(candidate_k_values)
    for k in sorted_k:
        table.add_column(f"Recall@{k}", justify="right")

    for method, curve in sorted(method_curves.items()):
        row = [method.upper()]
        for k in sorted_k:
            rec = curve.get(k, 0.0)
            rec_pct = f"{rec * 100.0:.1f}%"
            if rec >= 0.95:
                row.append(f"[green]{rec_pct}[/]")
            elif rec >= 0.80:
                row.append(f"[yellow]{rec_pct}[/]")
            else:
                row.append(f"[red]{rec_pct}[/]")
        table.add_row(*row)

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


def compare_retrieval_methods(
    prompts: list[dict[str, Any]],
    tools: list[ToolInfo],
    candidate_k_values: list[int] | None = None,
    methods: list[str] | None = None,
    embedding_helper: EmbeddingHelper | None = None,
    cross_encoder_scorer: Any | None = None,
    show_table: bool = True,
) -> dict[str, Any]:
    """Compare Recall@k across multiple retrieval methods (Dense, BM25, Hybrid, Cross-Encoder).

    Args:
        prompts: Curated dev-split prompts.
        tools: Full list of discovered tools.
        candidate_k_values: List of candidate k values (default [5, 10, 15, 20]).
        methods: List of retrieval methods to compare (default ["dense", "bm25", "hybrid"]).
        embedding_helper: Optional custom EmbeddingHelper for dense & hybrid retrieval.
        cross_encoder_scorer: Optional scorer or mock model for cross-encoder reranking.
        show_table: Whether to render the comparison table to the console.

    Returns:
        Dictionary mapping method names to {k: recall} dicts and list of evaluated k values.
    """
    candidates = candidate_k_values or [5, 10, 15, 20]
    valid_candidates = sorted({k for k in candidates if k > 0})
    if not valid_candidates:
        valid_candidates = [min(5, len(tools))]

    active_methods = methods or ["dense", "bm25", "hybrid"]
    retrievers: dict[str, Any] = {}

    for method in active_methods:
        method_lower = method.lower()
        if method_lower in ("dense", "embedding"):
            retrievers["dense"] = ToolRetriever(tools=tools, embedding_helper=embedding_helper)
        elif method_lower == "bm25":
            retrievers["bm25"] = BM25ToolRetriever(tools=tools)
        elif method_lower == "hybrid":
            retrievers["hybrid"] = HybridToolRetriever(
                tools=tools,
                embedding_helper=embedding_helper,
            )
        elif method_lower in ("cross_encoder", "cross-encoder"):
            retrievers["cross_encoder"] = CrossEncoderToolRetriever(
                tools=tools,
                scorer=cross_encoder_scorer,
            )
        else:
            logger.warning("Unrecognized retrieval method: %s (skipping)", method)

    method_curves: dict[str, dict[int, float]] = {}

    for name, retriever in retrievers.items():
        curve: dict[int, float] = {}
        for k in valid_candidates:
            recall = retriever.compute_recall_at_k(prompts, k=k)
            curve[k] = recall
        method_curves[name] = curve

    if show_table:
        table = render_method_comparison_table(
            method_curves=method_curves,
            candidate_k_values=valid_candidates,
        )
        console.print(table)

    return {
        "method_curves": method_curves,
        "k_values": valid_candidates,
    }
