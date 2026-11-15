"""Unit tests for schema validation and argument matching metrics (F7)."""

from typing import Any

import pytest

from toolbench.validate import (
    compute_args_match_score,
    compute_args_strict_match,
    normalize_value,
    validate_schema,
    values_are_equal,
)


def test_normalize_value() -> None:
    """Verify normalization rules for strings, numbers, booleans, and nested structures."""
    # Strings: Whitespace stripped, case strictly preserved
    assert normalize_value("  hello world  ") == "hello world"
    assert normalize_value("CasePreserved ") == "CasePreserved"

    # Numbers: Integer and float equality
    assert normalize_value(5) == 5.0
    assert normalize_value(5.0) == 5.0
    assert normalize_value(-0.0) == 0.0

    # Booleans: Preserved as boolean types, not converted to floats
    assert normalize_value(True) is True
    assert normalize_value(False) is False
    assert not isinstance(normalize_value(True), float)

    # Nested structures
    raw_dict = {" b ": 2, "a ": " test "}
    assert normalize_value(raw_dict) == {"a": "test", "b": 2.0}

    raw_list = ["  item1 ", 10]
    assert normalize_value(raw_list) == ["item1", 10.0]


def test_values_are_equal() -> None:
    """Verify semantic equivalence checks."""
    assert values_are_equal("text ", "  text") is True
    assert values_are_equal(42, 42.0) is True
    assert values_are_equal(True, 1) is False
    assert values_are_equal(False, 0) is False
    assert values_are_equal({"k": " v "}, {"k": "v"}) is True


def test_validate_schema_valid() -> None:
    """Verify schema validation passes for compliant arguments."""
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["path"],
    }

    # Valid args
    is_valid, err = validate_schema(schema, {"path": "test.txt", "count": 10})
    assert is_valid is True
    assert err is None

    # Empty schema and empty args
    assert validate_schema({}, {}) == (True, None)


def test_validate_schema_invalid() -> None:
    """Verify schema validation returns formatted error messages on failures."""
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["path"],
    }

    # Missing required field
    is_valid1, err1 = validate_schema(schema, {"count": 5})
    assert is_valid1 is False
    assert "required property" in str(err1)

    # Type mismatch
    is_valid2, err2 = validate_schema(schema, {"path": 12345})
    assert is_valid2 is False
    assert "Field 'path':" in str(err2)
    assert "is not of type 'string'" in str(err2)


def test_compute_args_strict_match() -> None:
    """Verify primary args_strict_match metric and extra keys detection."""
    acceptable: list[dict[str, Any]] = [
        {"path": "/tmp/data.csv", "format": "csv"},
    ]

    # Exact match
    match, has_extra = compute_args_strict_match(
        acceptable,
        {"path": " /tmp/data.csv ", "format": "csv"},
    )
    assert match is True
    assert has_extra is False

    # Match with extra optional keys (allowed for match, flagged in has_extra)
    match_extra, has_extra2 = compute_args_strict_match(
        acceptable,
        {"path": "/tmp/data.csv", "format": "csv", "optional_flag": True},
    )
    assert match_extra is True
    assert has_extra2 is True

    # Missing key -> match fails
    match_missing, _ = compute_args_strict_match(
        acceptable,
        {"path": "/tmp/data.csv"},
    )
    assert match_missing is False

    # Value mismatch -> match fails
    match_bad_val, _ = compute_args_strict_match(
        acceptable,
        {"path": "/tmp/data.csv", "format": "json"},
    )
    assert match_bad_val is False

    # Multiple acceptable candidates (e.g. alternate date format or casing)
    alternatives: list[dict[str, Any]] = [
        {"date": "2026-11-15"},
        {"date": "15-11-2026"},
    ]
    match_alt1, _ = compute_args_strict_match(alternatives, {"date": "15-11-2026"})
    assert match_alt1 is True

    match_alt2, _ = compute_args_strict_match(alternatives, {"date": "2026-11-15"})
    assert match_alt2 is True


def test_compute_args_match_score() -> None:
    """Verify fractional args_match_score calculation."""
    acceptable: list[dict[str, Any]] = [
        {"a": 1, "b": 2, "c": 3},
    ]

    # Suppressed when tool is incorrect
    assert compute_args_match_score(acceptable, {"a": 1, "b": 2, "c": 3}, tool_correct=False) == 0.0

    # 100% match
    score_full = compute_args_match_score(
        acceptable,
        {"a": 1, "b": 2, "c": 3},
        tool_correct=True,
    )
    assert score_full == 1.0

    # Partial match (2 out of 3 keys)
    score_partial = compute_args_match_score(
        acceptable,
        {"a": 1, "b": 2, "c": 999},
        tool_correct=True,
    )
    assert pytest.approx(score_partial, 1e-4) == 2.0 / 3.0

    # No matching keys
    score_zero = compute_args_match_score(
        acceptable,
        {"x": 10},
        tool_correct=True,
    )
    assert score_zero == 0.0

    # Empty acceptable candidates
    assert compute_args_match_score([], {"a": 1}, tool_correct=True) == 0.0
