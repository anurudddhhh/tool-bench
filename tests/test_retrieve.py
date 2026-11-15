"""Unit tests for embedding-based tool retrieval and k calibration (F5)."""

from unittest.mock import MagicMock, patch

import numpy as np

from toolbench.mcp_client import ToolInfo
from toolbench.retrieve import (
    ToolRetriever,
    calibrate_k,
    render_calibration_table,
)


def _make_tool(name: str, desc: str = "") -> ToolInfo:
    return ToolInfo(
        name=name,
        description=desc,
        input_schema={"type": "object", "properties": {}},
    )


def test_tool_retriever_ranking_and_tie_breaking() -> None:
    """Verify tool retriever ranks by similarity and breaks boundary ties alphabetically."""
    # 4 tools with defined similarity scores relative to query
    t_c = _make_tool("tool_c", "High similarity tool")
    t_a = _make_tool("tool_a", "Medium similarity tool A")
    t_b = _make_tool("tool_b", "Medium similarity tool B")
    t_d = _make_tool("tool_d", "Low similarity tool")
    tools = [t_c, t_a, t_b, t_d]

    mock_helper = MagicMock()
    # Tool vectors with distinct angles relative to query [1.0, 0.0, 0.0]
    tool_vecs = np.array(
        [
            [1.0, 0.0, 0.0],  # tool_c: score 1.0
            [0.5, 0.5, 0.0],  # tool_a: score ~0.707 (tied with b)
            [0.5, 0.5, 0.0],  # tool_b: score ~0.707 (tied with a)
            [0.0, 1.0, 0.0],  # tool_d: score 0.0
        ],
        dtype=np.float32,
    )
    query_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    # First call embeds tools, second call embeds query
    mock_helper.embed_texts.side_effect = [tool_vecs, query_vec]

    retriever = ToolRetriever(tools=tools, embedding_helper=mock_helper)

    # 1. Top-2 retrieval: tool_c (0.9), and tie between tool_a and tool_b (0.5)
    # tool_a comes before tool_b alphabetically
    top2 = retriever.retrieve_top_k("test query", k=2)
    assert len(top2) == 2
    assert [t.name for t in top2] == ["tool_c", "tool_a"]

    # 2. Top-3 retrieval: tool_c, tool_a, tool_b
    mock_helper.embed_texts.side_effect = [query_vec]
    top3 = retriever.retrieve_top_k("test query", k=3)
    assert len(top3) == 3
    assert [t.name for t in top3] == ["tool_c", "tool_a", "tool_b"]

    # 3. Top-k where k >= total tools returns all tools
    mock_helper.embed_texts.side_effect = [query_vec]
    all_res = retriever.retrieve_top_k("test query", k=10)
    assert len(all_res) == 4


def test_compute_recall_at_k() -> None:
    """Verify recall@k calculation across a prompt dataset."""
    t1 = _make_tool("tool_1")
    t2 = _make_tool("tool_2")
    t3 = _make_tool("tool_3")
    tools = [t1, t2, t3]

    mock_helper = MagicMock()
    mock_helper.embed_texts.return_value = np.zeros((3, 4), dtype=np.float32)

    retriever = ToolRetriever(tools=tools, embedding_helper=mock_helper)
    # Mock retrieve_top_k return values for 4 prompts:
    # Prompt 1: top-2 has tool_1 (Hit)
    # Prompt 2: top-2 has tool_2 (Hit)
    # Prompt 3: top-2 has tool_1, tool_2 (Miss for tool_3)
    # Prompt 4: top-2 has tool_1, tool_3 (Hit for tool_3)
    retriever.retrieve_top_k = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            [t1, t2],
            [t2, t3],
            [t1, t2],
            [t1, t3],
        ]
    )

    prompts = [
        {"prompt_text": "Q1", "expected_tool": "tool_1"},  # Hit
        {"prompt_text": "Q2", "expected_tool": "tool_2"},  # Hit
        {"prompt_text": "Q3", "expected_tool": "tool_3"},  # Miss
        {"prompt_text": "Q4", "expected_tool": "tool_3"},  # Hit
    ]

    recall = retriever.compute_recall_at_k(prompts, k=2)
    # 3 hits out of 4 = 0.75
    assert recall == 0.75


def test_calibrate_k_target_met() -> None:
    """Verify calibrate_k sweeps candidate k values and selects smallest k meeting target recall."""
    tools = [_make_tool(f"t_{i}") for i in range(10)]
    prompts = [{"prompt_text": f"Q{i}", "expected_tool": f"t_{i}"} for i in range(10)]

    mock_helper = MagicMock()
    mock_helper.embed_texts.return_value = np.zeros((10, 4), dtype=np.float32)

    # Simulated recall curve across k=[2, 4, 6]:
    # k=2 -> 0.80 (< 0.95)
    # k=4 -> 0.96 (>= 0.95 -> TARGET MET!)
    # k=6 -> 1.00 (>= 0.95)
    with patch.object(
        ToolRetriever,
        "compute_recall_at_k",
        side_effect=[0.80, 0.96, 1.00],
    ):
        selected_k, curve = calibrate_k(
            prompts=prompts,
            tools=tools,
            candidate_k_values=[2, 4, 6],
            target_recall=0.95,
            embedding_helper=mock_helper,
            show_table=False,
        )

    # Smallest k satisfying recall >= 0.95 is 4
    assert selected_k == 4
    assert curve[2] == 0.80
    assert curve[4] == 0.96
    assert curve[6] == 1.00


def test_calibrate_k_fallback_when_target_unmet() -> None:
    """Verify calibrate_k selects k with highest recall when target is not reached."""
    tools = [_make_tool(f"t_{i}") for i in range(10)]
    prompts = [{"prompt_text": f"Q{i}", "expected_tool": f"t_{i}"} for i in range(10)]

    mock_helper = MagicMock()
    mock_helper.embed_texts.return_value = np.zeros((10, 4), dtype=np.float32)

    # All candidate k values fall below 0.95 target
    # k=2 -> 0.60
    # k=4 -> 0.75
    # k=6 -> 0.85 (Best available recall)
    with patch.object(
        ToolRetriever,
        "compute_recall_at_k",
        side_effect=[0.60, 0.75, 0.85],
    ):
        selected_k, curve = calibrate_k(
            prompts=prompts,
            tools=tools,
            candidate_k_values=[2, 4, 6],
            target_recall=0.95,
            embedding_helper=mock_helper,
            show_table=False,
        )

    assert selected_k == 6
    assert curve[6] == 0.85


def test_retriever_setup_d_rewritten_descriptions_invariant() -> None:
    """Setup D Invariant: Verify rewritten descriptions are embedded instead of originals."""
    t1 = _make_tool("tool_git_commit", desc="Original vague git description")
    t2 = _make_tool("tool_fs_read", desc="Original fs description")
    tools = [t1, t2]

    rewritten_mapping = {
        "tool_git_commit": "Rewritten: Record staged repository changes with clear commit message.",
        "tool_fs_read": "Rewritten: Open and read file text from local path into string buffer.",
    }

    mock_helper = MagicMock()
    mock_helper.embed_texts.return_value = np.zeros((2, 4), dtype=np.float32)

    # Initialize retriever with custom rewritten descriptions (Setup D)
    ToolRetriever(
        tools=tools,
        embedding_helper=mock_helper,
        descriptions=rewritten_mapping,
    )

    assert mock_helper.embed_texts.call_count == 1
    embedded_texts: list[str] = mock_helper.embed_texts.call_args[0][0]

    # Must contain the REWRITTEN descriptions, NOT the originals!
    assert embedded_texts[0] == rewritten_mapping["tool_git_commit"]
    assert embedded_texts[1] == rewritten_mapping["tool_fs_read"]
    assert "Original vague git description" not in embedded_texts


def test_render_calibration_table() -> None:
    """Verify rendering of calibration curve table."""
    curve = {5: 0.82, 10: 0.96, 15: 1.0}
    table = render_calibration_table(curve, selected_k=10, target_recall=0.95)
    assert table.title == "Retrieval k-Sweep Calibration Curve (Dev Set)"
    assert len(table.rows) == 3
