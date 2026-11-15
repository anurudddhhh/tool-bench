"""Unit tests for retrieval method comparison (D1): BM25, Dense, Hybrid, Cross-Encoder."""

from unittest.mock import MagicMock

import numpy as np

from toolbench.mcp_client import ToolInfo
from toolbench.retrieve import (
    BM25ToolRetriever,
    CrossEncoderToolRetriever,
    HybridToolRetriever,
    compare_retrieval_methods,
    render_method_comparison_table,
)


def _make_tool(name: str, desc: str = "") -> ToolInfo:
    return ToolInfo(
        name=name,
        description=desc,
        input_schema={"type": "object", "properties": {}},
    )


def test_bm25_retriever_ranking_and_tie_breaking() -> None:
    """Verify BM25 ranks tools matching query keywords and breaks ties alphabetically."""
    t_git = _make_tool("git_commit", "Commit staged files to git repository")
    t_file = _make_tool("write_file", "Write text contents to a destination file on disk")
    t_read = _make_tool("read_file", "Read text contents from a source file on disk")
    t_diff = _make_tool("git_diff", "Show git changes between commits or staging")
    tools = [t_git, t_file, t_read, t_diff]

    bm25 = BM25ToolRetriever(tools=tools)

    # 1. Query for "git" should return git tools first
    top_git = bm25.retrieve_top_k("git commit changes", k=2)
    assert len(top_git) == 2
    top_names = [t.name for t in top_git]
    assert "git_commit" in top_names
    assert "git_diff" in top_names

    # 2. Query for "file disk" should rank file tools first
    top_file = bm25.retrieve_top_k("write data to file", k=2)
    assert len(top_file) == 2
    assert top_file[0].name == "write_file"

    # 3. If query matches nothing, scores are all 0.0 -> tie-break alphabetically
    zero_match = bm25.retrieve_top_k("completely unrelated astronaut telemetry", k=3)
    assert len(zero_match) == 3
    # Sorted by name alphabetically: git_commit, git_diff, read_file, write_file
    assert [t.name for t in zero_match] == ["git_commit", "git_diff", "read_file"]

    # 4. k >= len(tools) returns all tools
    all_tools = bm25.retrieve_top_k("git", k=10)
    assert len(all_tools) == 4

    # 5. k <= 0 returns empty
    assert bm25.retrieve_top_k("git", k=0) == []


def test_bm25_compute_recall_at_k() -> None:
    """Verify BM25 recall@k calculation."""
    t1 = _make_tool("search_repo", "Search for code patterns in git repository")
    t2 = _make_tool("edit_file", "Edit and replace text blocks in a file")
    tools = [t1, t2]

    bm25 = BM25ToolRetriever(tools=tools)

    prompts = [
        {"prompt_text": "search code in repo", "expected_tool": "search_repo"},  # Hit
        {"prompt_text": "edit line 5 in file", "expected_tool": "edit_file"},  # Hit
        {"prompt_text": "inspect git repository", "expected_tool": "edit_file"},  # Miss
    ]

    recall_top1 = bm25.compute_recall_at_k(prompts, k=1)
    assert abs(recall_top1 - (2 / 3)) < 1e-4

    # With k=2, both tools are returned -> 100% recall
    recall_top2 = bm25.compute_recall_at_k(prompts, k=2)
    assert recall_top2 == 1.0


def test_hybrid_retriever_rrf_scoring() -> None:
    """Verify HybridToolRetriever combines Dense and BM25 ranks via Reciprocal Rank Fusion."""
    t_a = _make_tool("tool_a", "Description tool a")
    t_b = _make_tool("tool_b", "Description tool b")
    t_c = _make_tool("tool_c", "Description tool c")
    tools = [t_a, t_b, t_c]

    mock_helper = MagicMock()
    # Dummy embedding return
    mock_helper.embed_texts.return_value = np.zeros((3, 4), dtype=np.float32)

    hybrid = HybridToolRetriever(
        tools=tools,
        embedding_helper=mock_helper,
        rrf_constant=60,
    )

    # Dense ranking: [t_b, t_a, t_c] -> ranks: t_b=1, t_a=2, t_c=3
    hybrid.dense_retriever.retrieve_top_k = MagicMock(return_value=[t_b, t_a, t_c])  # type: ignore[method-assign]
    # BM25 ranking: [t_a, t_b, t_c] -> ranks: t_a=1, t_b=2, t_c=3
    hybrid.bm25_retriever.retrieve_top_k = MagicMock(return_value=[t_a, t_b, t_c])  # type: ignore[method-assign]

    # RRF scores:
    # t_a: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # t_b: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.032522 (TIED with t_a!)
    # t_c: 1/(60+3) + 1/(60+3) = 2/63 = 0.031746
    # Tie between t_a and t_b is broken alphabetically -> t_a before t_b
    top2 = hybrid.retrieve_top_k("query", k=2)
    assert len(top2) == 2
    assert [t.name for t in top2] == ["tool_a", "tool_b"]

    # Now make t_b unequivocally rank 1 on both
    hybrid.dense_retriever.retrieve_top_k = MagicMock(return_value=[t_b, t_a, t_c])  # type: ignore[method-assign]
    hybrid.bm25_retriever.retrieve_top_k = MagicMock(return_value=[t_b, t_c, t_a])  # type: ignore[method-assign]
    top1 = hybrid.retrieve_top_k("query", k=1)
    assert len(top1) == 1
    assert top1[0].name == "tool_b"


def test_cross_encoder_retriever() -> None:
    """Verify CrossEncoderToolRetriever scores query-tool pairs and ranks by score."""
    t1 = _make_tool("db_query", "Execute SQL query on postgres database")
    t2 = _make_tool("file_write", "Write data to disk file")
    t3 = _make_tool("db_schema", "Inspect tables and schema of postgres database")
    tools = [t1, t2, t3]

    # Inject mock scorer
    mock_scorer = MagicMock()
    # Scorer returns scores for pairs [query, tool_text]:
    # Suppose db_schema gets 0.95, db_query gets 0.85, file_write gets 0.10
    mock_scorer.predict.return_value = np.array([0.85, 0.10, 0.95], dtype=np.float32)

    cross_retriever = CrossEncoderToolRetriever(tools=tools, scorer=mock_scorer)

    top2 = cross_retriever.retrieve_top_k("show me the postgres tables", k=2)
    assert len(top2) == 2
    assert [t.name for t in top2] == ["db_schema", "db_query"]

    # Check pair formatting passed to scorer
    assert mock_scorer.predict.called
    call_pairs = mock_scorer.predict.call_args[0][0]
    assert len(call_pairs) == 3
    assert call_pairs[0][0] == "show me the postgres tables"
    assert "db_query:" in call_pairs[0][1]


def test_compare_retrieval_methods_flow() -> None:
    """Verify compare_retrieval_methods sweeps k across methods and renders comparison."""
    t1 = _make_tool("tool_x", "Perform calculations on numbers")
    t2 = _make_tool("tool_y", "Search files in folder")
    tools = [t1, t2]

    prompts = [
        {"prompt_text": "calculate 2 + 2", "expected_tool": "tool_x"},
        {"prompt_text": "search files", "expected_tool": "tool_y"},
    ]

    mock_helper = MagicMock()
    # Dummy embedding
    mock_helper.embed_texts.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    mock_scorer = MagicMock()
    mock_scorer.predict.return_value = np.array([1.0, 0.0], dtype=np.float32)

    result = compare_retrieval_methods(
        prompts=prompts,
        tools=tools,
        candidate_k_values=[1, 2],
        methods=["dense", "bm25", "hybrid", "cross_encoder"],
        embedding_helper=mock_helper,
        cross_encoder_scorer=mock_scorer,
        show_table=True,
    )

    assert "method_curves" in result
    assert "k_values" in result
    assert result["k_values"] == [1, 2]

    curves = result["method_curves"]
    assert "dense" in curves
    assert "bm25" in curves
    assert "hybrid" in curves
    assert "cross_encoder" in curves

    for method, curve in curves.items():
        assert 1 in curve
        assert 2 in curve
        assert 0.0 <= curve[1] <= 1.0
        assert 0.0 <= curve[2] <= 1.0


def test_render_method_comparison_table() -> None:
    """Verify render_method_comparison_table generates Rich table correctly."""
    curves = {
        "dense": {5: 0.96, 10: 1.00},
        "bm25": {5: 0.88, 10: 0.94},
        "hybrid": {5: 0.98, 10: 1.00},
    }
    table = render_method_comparison_table(method_curves=curves, candidate_k_values=[5, 10])
    assert table.title == "Retrieval Method Comparison (Dev Set Recall@k)"
    assert len(table.columns) == 3  # Method, Recall@5, Recall@10
    assert len(table.rows) == 3
