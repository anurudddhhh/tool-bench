"""Synthetic prompt and argument generation with anti-leakage filtering (F2).

Generates realistic user prompts and ground-truth arguments for MCP tools using
an LLM provider, enforces strict anti-leakage rules to prevent benchmark contamination,
and persists valid prompts to `data/raw_prompts.json`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from toolbench.config import GenerationConfig
from toolbench.embeddings import EmbeddingHelper, cosine_similarity, get_default_embedding_helper
from toolbench.mcp_client import ToolInfo
from toolbench.providers.base import ProviderBase

logger = logging.getLogger(__name__)

# Standard English stopword set for anti-leakage ngram filtering
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


class PromptRecord(BaseModel):
    """Ground-truth synthetic benchmark prompt record."""

    prompt_id: int = Field(..., description="Unique sequential 1-based prompt identifier")
    prompt_text: str = Field(..., description="Realistic user query requesting an action")
    expected_tool: str = Field(..., description="Namespaced tool name required to solve prompt")
    expected_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Ground-truth parameters for the expected tool",
    )
    source_tool_description: str = Field(
        default="",
        description="Original tool description used to generate this prompt",
    )
    high_similarity: bool = Field(
        default=False,
        description="Flag indicating semantic similarity > 0.85 between prompt and description",
    )


def is_multipart_identifier(identifier: str) -> bool:
    """Return True if an identifier consists of multiple subwords (snake, kebab, camelCase)."""
    # Contains underscore or hyphen
    if "_" in identifier or "-" in identifier:
        return True
    # Contains camelCase pattern: lowercase followed immediately by uppercase
    if re.search(r"[a-z][A-Z]", identifier):
        return True
    return False


def check_literal_identifier_leak(prompt_text: str, tool_name: str) -> bool:
    """Anti-leakage Rule 1: Check if prompt contains a literal multi-part tool identifier.

    Single-word identifiers (e.g. 'read', 'ping') are exempt from this rule to avoid
    rejecting natural English requests like 'read the file'.

    Args:
        prompt_text: The candidate synthetic prompt string.
        tool_name: The tool identifier (e.g. 'filesystem__read_file' or 'read_file').

    Returns:
        True if the prompt leaks the identifier (should be rejected), False otherwise.
    """
    prompt_lower = prompt_text.lower()

    # Consider both full namespaced identifier and un-namespaced base tool name
    candidates = [tool_name]
    if "__" in tool_name:
        candidates.append(tool_name.split("__", 1)[1])

    for cand in candidates:
        if is_multipart_identifier(cand):
            cand_lower = cand.lower()
            # Check literal substring match with word-boundary awareness
            # Handles separators like underscores, hyphens, spaces
            pattern = re.escape(cand_lower)
            if re.search(r"\b" + pattern + r"\b", prompt_lower):
                return True
            # Also check if raw string appears directly as token
            if cand_lower in prompt_lower:
                return True

    return False


def tokenize_words(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def check_description_overlap_leak(prompt_text: str, description: str) -> bool:
    """Anti-leakage Rule 2: Check if prompt copies description text verbatim.

    Hard rejects if:
    1. Prompt contains 6+ consecutive verbatim words from the tool description, OR
    2. Prompt contains 4+ consecutive verbatim words including >= 2 non-stopwords.

    Args:
        prompt_text: Candidate prompt.
        description: Source tool description.

    Returns:
        True if the prompt copies description text (should be rejected), False otherwise.
    """
    if not description.strip():
        return False

    prompt_words = tokenize_words(prompt_text)
    desc_words = tokenize_words(description)

    if len(desc_words) < 4 or len(prompt_words) < 4:
        return False

    # Check length-6 consecutive ngrams
    if len(desc_words) >= 6 and len(prompt_words) >= 6:
        desc_6grams = {tuple(desc_words[i : i + 6]) for i in range(len(desc_words) - 5)}
        for j in range(len(prompt_words) - 5):
            ngram = tuple(prompt_words[j : j + 6])
            if ngram in desc_6grams:
                return True

    # Check length-4 and length-5 consecutive ngrams with >= 2 non-stopwords
    for n in (4, 5):
        if len(desc_words) >= n and len(prompt_words) >= n:
            desc_ngrams = {tuple(desc_words[i : i + n]) for i in range(len(desc_words) - n + 1)}
            for j in range(len(prompt_words) - n + 1):
                ngram = tuple(prompt_words[j : j + n])
                if ngram in desc_ngrams:
                    non_stopword_count = sum(1 for w in ngram if w not in STOPWORDS)
                    if non_stopword_count >= 2:
                        return True

    return False


def flag_semantic_similarity(
    prompt_text: str,
    description: str,
    embedding_helper: EmbeddingHelper | None = None,
) -> bool:
    """Anti-leakage Rule 3: Flag-only check for high semantic similarity (> 0.85).

    Flags the prompt for review but NEVER auto-discards.

    Args:
        prompt_text: Synthetic prompt string.
        description: Source tool description.
        embedding_helper: EmbeddingHelper instance (defaults to global helper).

    Returns:
        True if cosine similarity > 0.85, False otherwise.
    """
    if not description.strip() or not prompt_text.strip():
        return False

    helper = embedding_helper or get_default_embedding_helper()
    try:
        vecs = helper.embed_texts([prompt_text, description])
        sim = cosine_similarity(vecs[0], vecs[1])
        return sim > 0.85
    except Exception as err:
        logger.debug("Semantic similarity calculation failed: %s", err)
        return False


def build_generation_system_prompt() -> str:
    """System instruction guiding the LLM to generate realistic diverse prompts."""
    return (
        "You are an expert AI benchmarking assistant. Your task is to generate realistic, "
        "natural human prompts that require calling a specific tool, along with the expected "
        "arguments matching the tool's JSON schema.\n\n"
        "Guidelines:\n"
        "1. Write realistic user queries: conversational, clear, and goal-oriented.\n"
        "2. DO NOT mention the tool name, function name, or code identifiers in the prompt.\n"
        "3. DO NOT copy phrases or definitions directly from the tool description.\n"
        "4. Include plausible, concrete argument values matching the parameter types.\n"
        "5. Output ONLY valid JSON matching this schema:\n"
        "   [\n"
        '     {"prompt": "user query string", "args": {"param1": "val1"}},\n'
        "     ...\n"
        "   ]"
    )


def build_tool_generation_query(tool: ToolInfo, count: int) -> str:
    """Construct the prompt sent to the LLM detailing the target tool."""
    desc = tool.description if tool.description.strip() else "No description available."
    schema_json = json.dumps(tool.input_schema, indent=2)

    return (
        f"Generate {count} diverse user prompts that would require using the following tool.\n\n"
        f"Tool Name: {tool.name}\n"
        f"Description: {desc}\n"
        f"Input Schema:\n{schema_json}\n\n"
        f"Output a JSON array of {count} objects with 'prompt' and 'args' keys."
    )


def parse_generation_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSON array of generated prompt candidates from LLM text."""
    clean_text = response_text.strip()
    # Strip markdown code blocks if present
    match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", clean_text)
    if match:
        clean_text = match.group(1).strip()
    elif clean_text.startswith("```") and clean_text.endswith("```"):
        clean_text = clean_text.strip("`").strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, list):
            valid_items: list[dict[str, Any]] = []
            for item in parsed:
                if isinstance(item, dict) and "prompt" in item:
                    valid_items.append(
                        {
                            "prompt": str(item.get("prompt", "")).strip(),
                            "args": dict(item.get("args") or {}),
                        }
                    )
            return valid_items
    except Exception as err:
        logger.warning(
            "Failed to parse LLM generation JSON: %s. Response: %s",
            err,
            clean_text[:200],
        )

    return []


def generate_prompts(
    tools: list[ToolInfo],
    provider: ProviderBase,
    config: GenerationConfig | None = None,
    output_path: Path | str = "data/raw_prompts.json",
    embedding_helper: EmbeddingHelper | None = None,
) -> list[PromptRecord]:
    """Generate and filter synthetic prompts across a list of MCP tools (F2).

    Args:
        tools: Discovered tools from F1/F1.5.
        provider: Initialized LLM provider (GeminiProvider or MockProvider).
        config: GenerationConfig detailing target count and temperature.
        output_path: Destination JSON file path (default 'data/raw_prompts.json').
        embedding_helper: Optional custom EmbeddingHelper instance.

    Returns:
        List of generated and validated PromptRecord instances.
    """
    cfg = config or GenerationConfig()
    target_count = cfg.count
    prompts_per_tool = cfg.prompts_per_tool

    if not tools:
        raise ValueError("Cannot generate prompts without tools. Tool list is empty.")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[PromptRecord] = []
    current_prompt_id = 1
    helper = embedding_helper or get_default_embedding_helper()

    system_prompt = build_generation_system_prompt()

    # Loop through tools generating candidates until target_count is reached
    tool_cycle = 0
    # Allow enough cycles to reach target even with small tool sets or strict filtering
    max_cycles = max(50, (target_count // max(1, len(tools))) * 3)

    while len(records) < target_count and tool_cycle < max_cycles:
        tool_cycle += 1
        logger.info(
            "Starting prompt generation cycle %d (current count: %d/%d)...",
            tool_cycle,
            len(records),
            target_count,
        )

        for tool in tools:
            if len(records) >= target_count:
                break

            user_query = build_tool_generation_query(tool, count=prompts_per_tool)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]
            try:
                response = provider.generate(
                    messages=messages,
                    tools=[],  # Pure text generation prompt
                    temperature=cfg.temperature,
                )
            except Exception as err:
                logger.warning("Generation call failed for tool '%s': %s", tool.name, err)
                continue

            candidates = parse_generation_response(response.text_response or "")

            for cand in candidates:
                if len(records) >= target_count:
                    break

                text = cand["prompt"]
                args = cand["args"]

                if not text:
                    continue

                # Apply Anti-Leakage Rule 1: Literal multi-part identifier
                if check_literal_identifier_leak(text, tool.name):
                    logger.debug("Prompt rejected by Rule 1 (identifier leak): '%s'", text)
                    continue

                # Apply Anti-Leakage Rule 2: Verbatim description overlap
                if check_description_overlap_leak(text, tool.description):
                    logger.debug("Prompt rejected by Rule 2 (description leak): '%s'", text)
                    continue

                # Apply Anti-Leakage Rule 3: Semantic similarity flag (never rejects)
                is_high_sim = flag_semantic_similarity(
                    text,
                    tool.description,
                    embedding_helper=helper,
                )

                record = PromptRecord(
                    prompt_id=current_prompt_id,
                    prompt_text=text,
                    expected_tool=tool.name,
                    expected_args=args,
                    source_tool_description=tool.description,
                    high_similarity=is_high_sim,
                )
                records.append(record)
                current_prompt_id += 1

    # Save to JSON file
    data_to_save = [r.model_dump() for r in records]
    out_file.write_text(json.dumps(data_to_save, indent=2), encoding="utf-8")
    logger.info(
        "Saved %d validated synthetic prompts to '%s'.",
        len(records),
        out_file,
    )
    return records
