"""Unit tests for synthetic prompt generation and anti-leakage filtering (F2)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from toolbench.config import GenerationConfig
from toolbench.generate import (
    check_description_overlap_leak,
    check_literal_identifier_leak,
    flag_semantic_similarity,
    generate_prompts,
    is_multipart_identifier,
    parse_generation_response,
)
from toolbench.mcp_client import ToolInfo
from toolbench.providers.mock import MockProvider


def test_is_multipart_identifier() -> None:
    """Verify classification of multi-part vs single-word identifiers."""
    # Multi-part identifiers
    assert is_multipart_identifier("read_file") is True
    assert is_multipart_identifier("git-status") is True
    assert is_multipart_identifier("server__read_file") is True
    assert is_multipart_identifier("getUserData") is True

    # Single-word identifiers
    assert is_multipart_identifier("read") is False
    assert is_multipart_identifier("ping") is False
    assert is_multipart_identifier("SEARCH") is False
    assert is_multipart_identifier("commit") is False


def test_check_literal_identifier_leak_rule1() -> None:
    """Anti-leakage Rule 1: Multi-part tool identifiers must be rejected."""
    # Multi-part identifier leaked in prompt
    assert (
        check_literal_identifier_leak("Please call read_file on doc.txt", "filesystem__read_file")
        is True
    )
    assert check_literal_identifier_leak("Can you run git-status for me?", "git-status") is True
    assert check_literal_identifier_leak("Use getUserData with id 123", "getUserData") is True

    # Natural language with single words from identifier is NOT rejected
    assert (
        check_literal_identifier_leak("Please read the file doc.txt", "filesystem__read_file")
        is False
    )
    assert (
        check_literal_identifier_leak("Show me the current repository status", "git__status")
        is False
    )

    # Single-word tool names are exempt from Rule 1
    assert check_literal_identifier_leak("Ping the server for latency", "ping") is False
    assert check_literal_identifier_leak("Search for python docs", "search") is False


def test_check_description_overlap_leak_rule2() -> None:
    """Anti-leakage Rule 2: Verbatim ngrams from tool description must be rejected."""
    desc = "Reads the entire contents of a specified file from disk into memory."

    # 1. Prompt copying 6+ consecutive verbatim words -> Hard reject
    leaked_6_words = "Can you please reads the entire contents of a specified file right now?"
    assert check_description_overlap_leak(leaked_6_words, desc) is True

    # 2. Prompt copying 4+ consecutive words including >= 2 non-stopwords -> Hard reject
    # Non-stopwords in slice: "entire", "contents", "specified"
    leaked_4_words = "I want to inspect the entire contents of a specified file"
    assert check_description_overlap_leak(leaked_4_words, desc) is True

    # 3. Prompt with natural phrasing and no verbatim sequences -> Allowed
    natural_prompt = "Open up report.txt and show me what was written inside."
    assert check_description_overlap_leak(natural_prompt, desc) is False

    # 4. Short descriptions or empty descriptions do not crash
    assert check_description_overlap_leak("Any prompt", "") is False
    assert check_description_overlap_leak("Any prompt", "Short desc") is False


def test_flag_semantic_similarity_rule3() -> None:
    """Anti-leakage Rule 3: Cosine similarity > 0.85 sets high_similarity flag."""
    mock_helper = MagicMock()

    # Case A: High similarity (> 0.85)
    # Return two nearly identical vectors (cosine sim ~ 1.0)
    mock_helper.embed_texts.return_value = np.array(
        [[1.0, 0.0], [0.99, 0.01]],
        dtype=np.float32,
    )
    assert flag_semantic_similarity("prompt", "desc", embedding_helper=mock_helper) is True

    # Case B: Low similarity (< 0.85)
    # Return orthogonal vectors (cosine sim = 0.0)
    mock_helper.embed_texts.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    assert flag_semantic_similarity("prompt", "desc", embedding_helper=mock_helper) is False


def test_parse_generation_response() -> None:
    """Verify parsing JSON arrays with raw text and markdown fences."""
    # Plain JSON
    raw_json = '[{"prompt": "Show status", "args": {"verbose": true}}]'
    parsed1 = parse_generation_response(raw_json)
    assert len(parsed1) == 1
    assert parsed1[0]["prompt"] == "Show status"
    assert parsed1[0]["args"] == {"verbose": True}

    # Markdown fenced JSON
    fenced_json = """```json
[
  {"prompt": "Read notes.txt", "args": {"path": "notes.txt"}},
  {"prompt": "List files in src", "args": {"dir": "src"}}
]
```"""
    parsed2 = parse_generation_response(fenced_json)
    assert len(parsed2) == 2
    assert parsed2[1]["prompt"] == "List files in src"

    # Malformed text
    assert parse_generation_response("Not JSON at all") == []


def test_generate_prompts_end_to_end(tmp_path: Path) -> None:
    """Verify generate_prompts filters leaked prompts, assigns IDs, and writes JSON."""
    out_file = tmp_path / "raw_prompts.json"

    tool = ToolInfo(
        name="filesystem__read_file",
        description="Reads the entire contents of a specified file from disk into memory.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        server_name="filesystem",
        original_name="read_file",
    )

    # Simulated LLM output containing 1 clean prompt, 1 Rule 1 leak, 1 Rule 2 leak
    llm_output = json.dumps(
        [
            # Clean prompt
            {"prompt": "Can you display the contents of data.csv?", "args": {"path": "data.csv"}},
            # Rule 1 leak: contains "read_file"
            {"prompt": "Call read_file on data.csv please", "args": {"path": "data.csv"}},
            # Rule 2 leak: copies 6 consecutive words from description
            {
                "prompt": "Please reads the entire contents of a specified file immediately",
                "args": {"path": "data.csv"},
            },
            # Second clean prompt
            {"prompt": "Open up README.md and show it to me", "args": {"path": "README.md"}},
        ]
    )

    provider = MockProvider(simulate_text_only=llm_output)

    mock_helper = MagicMock()
    mock_helper.embed_texts.return_value = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)

    cfg = GenerationConfig(count=40, prompts_per_tool=4)
    records = generate_prompts(
        tools=[tool],
        provider=provider,
        config=cfg,
        output_path=out_file,
        embedding_helper=mock_helper,
    )

    # Exactly 40 clean prompts collected
    assert len(records) == 40
    assert records[0].prompt_id == 1
    assert records[0].prompt_text == "Can you display the contents of data.csv?"
    assert records[0].expected_tool == "filesystem__read_file"
    assert records[0].expected_args == {"path": "data.csv"}

    assert records[1].prompt_id == 2
    assert records[1].prompt_text == "Open up README.md and show it to me"
    assert records[39].prompt_id == 40

    # Verify JSON file written to disk
    assert out_file.exists()
    saved_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(saved_data) == 40
    assert saved_data[0]["prompt_id"] == 1
    assert saved_data[39]["prompt_id"] == 40
