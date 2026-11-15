"""Unit tests for tool description rewriting and length truncation (F6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolbench.config import GenerationConfig
from toolbench.mcp_client import ToolInfo
from toolbench.providers.mock import MockProvider
from toolbench.rewrite import (
    clean_rewritten_text,
    count_words,
    load_rewritten_descriptions,
    rewrite_descriptions,
    truncate_to_word_limit,
)


def test_count_words():
    assert count_words("hello world") == 2
    assert count_words("   leading and   trailing   ") == 3
    assert count_words("") == 0


def test_truncate_to_word_limit_short_text():
    text = "This is a short sentence."
    assert truncate_to_word_limit(text, max_words=10) == text


def test_truncate_to_word_limit_at_sentence_boundary():
    # 15 words across 2 sentences
    text = "First sentence has five words. Second sentence also has some more words in it."
    # Truncate to 8 words: "First sentence has five words. Second sentence also"
    # Sentence ending "." is at word 5, so candidate should cut cleanly at the first sentence
    truncated = truncate_to_word_limit(text, max_words=8)
    assert truncated == "First sentence has five words."
    assert count_words(truncated) <= 8


def test_truncate_to_word_limit_no_punct():
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    truncated = truncate_to_word_limit(text, max_words=5)
    assert truncated == "word1 word2 word3 word4 word5..."
    assert count_words(truncated) <= 5


def test_clean_rewritten_text():
    assert clean_rewritten_text('"A tool to read files."') == "A tool to read files."
    assert clean_rewritten_text("'A tool to read files.'") == "A tool to read files."
    assert (
        clean_rewritten_text("Rewritten Description: Cleans up formatting.")
        == "Cleans up formatting."
    )
    assert clean_rewritten_text("Description: Execute a command.") == "Execute a command."


def test_rewrite_descriptions_and_load(tmp_path: Path):
    tools = [
        ToolInfo(
            name="read_file",
            description="Reads file contents.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        ToolInfo(
            name="write_file",
            description="Writes file contents.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    ]

    mock_provider = MockProvider(
        simulate_text_only="Enhanced tool description specifying precise trigger conditions."
    )

    out_file = tmp_path / "rewritten.json"
    result_map = rewrite_descriptions(
        tools=tools,
        provider=mock_provider,
        config=GenerationConfig(temperature=0.7),
        output_path=out_file,
        max_words=200,
    )

    assert "read_file" in result_map
    assert "write_file" in result_map
    assert (
        result_map["read_file"]
        == "Enhanced tool description specifying precise trigger conditions."
    )

    # Verify JSON file was created and is readable
    assert out_file.exists()
    saved_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(saved_data) == 2
    assert saved_data[0]["tool_name"] == "read_file"
    assert saved_data[0]["word_count"] > 0

    # Test load_rewritten_descriptions
    loaded_map = load_rewritten_descriptions(out_file)
    assert loaded_map == result_map


def test_load_rewritten_descriptions_not_found():
    with pytest.raises(FileNotFoundError):
        load_rewritten_descriptions("non_existent_file.json")


def test_load_rewritten_descriptions_invalid_format(tmp_path: Path):
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('{"tool": "desc"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON array"):
        load_rewritten_descriptions(invalid_file)
