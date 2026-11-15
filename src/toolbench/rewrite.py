"""LLM-assisted tool description rewriting and length truncation (F6).

Rewrites MCP tool descriptions to make them clearer, distinct, and informative
about trigger conditions vs similar tools, enforcing a strict <= 200 words length
limit with sentence-boundary truncation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from toolbench.config import GenerationConfig
from toolbench.mcp_client import ToolInfo
from toolbench.output import print_info, print_success
from toolbench.providers.base import ProviderBase

logger = logging.getLogger(__name__)


class RewrittenDescriptionRecord(BaseModel):
    """Container for original and rewritten tool descriptions."""

    tool_name: str
    original_description: str
    rewritten_description: str
    word_count: int = Field(default=0, description="Word count of rewritten description")


def count_words(text: str) -> int:
    """Count words in a string by whitespace splitting."""
    return len(text.strip().split())


def truncate_to_word_limit(text: str, max_words: int = 200) -> str:
    """Truncate text to at most max_words, cutting at the last sentence boundary.

    If cutting at a sentence boundary leaves no complete sentences, truncates at
    max_words and appends an ellipsis.

    Args:
        text: Input string.
        max_words: Maximum allowed words (default 200).

    Returns:
        String containing at most max_words.
    """
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()

    candidate = " ".join(words[:max_words])
    # Search for last sentence-ending punctuation (. ! ?)
    last_punct = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
    if last_punct > len(candidate) // 2:
        return candidate[: last_punct + 1].strip()

    return f"{candidate.rstrip('.,;:!')}..."


def build_rewrite_system_prompt() -> str:
    """System prompt instructing LLM how to rewrite tool descriptions."""
    return (
        "You are an expert technical documentation assistant specializing in LLM tool selection.\n"
        "Your task is to rewrite MCP tool descriptions to make them clear and distinctive.\n\n"
        "Instructions:\n"
        "1. Clarify what the tool does and specifically WHEN to use it vs similar tools.\n"
        "2. Explain what key arguments are required without repeating raw schema types.\n"
        "3. Keep the rewritten description concise and strictly under 180 words.\n"
        "4. Output ONLY the rewritten description text without quotation marks, markdown headings, "
        "or conversational preamble."
    )


def build_rewrite_user_query(tool: ToolInfo) -> str:
    """Construct prompt detailing the tool name, current description, and schema."""
    desc = tool.description if tool.description.strip() else "(No existing description provided)"
    schema_str = json.dumps(tool.input_schema, indent=2)

    return (
        f"Tool Name: {tool.name}\n"
        f"Current Description: {desc}\n"
        f"Input Parameter Schema:\n{schema_str}\n\n"
        "Please provide an improved, distinctive tool description:"
    )


def clean_rewritten_text(raw_text: str) -> str:
    """Strip unnecessary markdown prefixes or wrapping quotes from LLM output."""
    cleaned = raw_text.strip()
    # Strip wrapping quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    # Strip prefixes like "Rewritten Description:"
    cleaned = re.sub(
        r"^(?:rewritten(?:\s+tool)?\s+description:?|description:?)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def rewrite_descriptions(
    tools: list[ToolInfo],
    provider: ProviderBase,
    config: GenerationConfig | None = None,
    output_path: Path | str = "data/rewritten_descriptions.json",
    max_words: int = 200,
) -> dict[str, str]:
    """Rewrite descriptions for all discovered tools and freeze results to disk (F6).

    Args:
        tools: Full list of ToolInfo instances.
        provider: ProviderBase instance (GeminiProvider or MockProvider).
        config: GenerationConfig containing generation temperature (default 0.7).
        output_path: File path to save frozen rewritten descriptions JSON.
        max_words: Maximum words permitted per description (default 200).

    Returns:
        Dictionary mapping {tool_name: rewritten_description}.
    """
    cfg = config or GenerationConfig()
    system_prompt = build_rewrite_system_prompt()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print_info(f"Rewriting descriptions for {len(tools)} tools at temperature {cfg.temperature}...")

    records: list[RewrittenDescriptionRecord] = []
    rewritten_map: dict[str, str] = {}

    for tool in tools:
        user_query = build_rewrite_user_query(tool)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            resp = provider.generate(
                messages=messages,
                tools=[],  # Pure text generation prompt
                temperature=cfg.temperature,
            )
            raw_text = resp.text_response or tool.description or tool.name
        except Exception as err:
            logger.warning("Failed to rewrite description for tool '%s': %s", tool.name, err)
            raw_text = tool.description or tool.name

        cleaned = clean_rewritten_text(raw_text)
        truncated = truncate_to_word_limit(cleaned, max_words=max_words)
        words = count_words(truncated)

        record = RewrittenDescriptionRecord(
            tool_name=tool.name,
            original_description=tool.description,
            rewritten_description=truncated,
            word_count=words,
        )
        records.append(record)
        rewritten_map[tool.name] = truncated

    # Save frozen JSON to disk
    data_to_save = [r.model_dump() for r in records]
    out_file.write_text(json.dumps(data_to_save, indent=2), encoding="utf-8")
    print_success(f"Successfully saved {len(records)} rewritten descriptions to '{out_file}'.")

    return rewritten_map


def load_rewritten_descriptions(
    path: Path | str = "data/rewritten_descriptions.json",
) -> dict[str, str]:
    """Load frozen rewritten descriptions from disk.

    Args:
        path: Path to rewritten descriptions JSON file.

    Returns:
        Dictionary mapping {tool_name: rewritten_description}.
    """
    f_path = Path(path)
    if not f_path.exists():
        raise FileNotFoundError(
            f"Rewritten descriptions file not found at '{f_path}'. "
            "Run 'toolbench rewrite-descriptions' first."
        )

    raw_data = json.loads(f_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Invalid format in '{f_path}': expected JSON array.")

    mapping: dict[str, str] = {}
    for entry in raw_data:
        if isinstance(entry, dict) and "tool_name" in entry:
            mapping[entry["tool_name"]] = entry.get("rewritten_description", "")

    return mapping
