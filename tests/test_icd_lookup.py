"""Unit tests for src/icd_lookup.py.

Run with: pytest -v
Pure Python -- no LLM, no network needed.
"""

from src.icd_lookup import lookup_icd_description


def test_known_code_returns_verified_description():
    """A code in the reference set returns its real description."""
    result = lookup_icd_description("Z47.1")
    assert result == "Aftercare following joint replacement surgery"


def test_unknown_code_returns_none():
    """A code not in the reference set returns None, not a guess."""
    assert lookup_icd_description("Z99.99") is None


def test_none_code_returns_none():
    """A missing icd_code (None) is handled without raising."""
    assert lookup_icd_description(None) is None


def test_empty_string_returns_none():
    """An empty string is treated the same as no code at all."""
    assert lookup_icd_description("") is None


def test_lookup_is_case_and_whitespace_insensitive():
    """'  z47.1  ' still matches 'Z47.1' in the reference table."""
    assert lookup_icd_description("  z47.1  ") == (
        "Aftercare following joint replacement surgery"
    )
