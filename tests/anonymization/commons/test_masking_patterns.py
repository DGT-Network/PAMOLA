"""
File: test_masking_patterns.py
Test Target: commons/masking_patterns.py
Version: 1.0
Coverage Status: In Progress
Last Updated: 2025-07-25
"""

import pytest
import re
import string
import random
from pamola_core.anonymization.commons import masking_patterns as mp

# Coverage Points: All public methods, edge cases, error handling
# Process Requirements: ≥90% line coverage, import hygiene, test isolation
# Import Hygiene: All dependencies must match codebase imports

def test_get_pattern_and_validate_pattern_type():
    pattern = mp.MaskingPatterns.get_pattern("email")
    assert pattern is not None
    assert pattern.regex.startswith("^([^@]{1,2}")
    assert mp.MaskingPatterns.validate_pattern_type("email")
    assert not mp.MaskingPatterns.validate_pattern_type("not_a_pattern")

def test_get_pattern_names_and_default_patterns():
    names = mp.MaskingPatterns.get_pattern_names()
    assert "email" in names
    defaults = mp.MaskingPatterns.get_default_patterns()
    assert isinstance(defaults, dict)
    assert "email" in defaults
    # Ensure copy is not the same object
    assert defaults is not mp.MaskingPatterns.PATTERNS

def test_detect_pattern_type():
    assert mp.MaskingPatterns.detect_pattern_type("john.doe@example.com") == "email"
    assert mp.MaskingPatterns.detect_pattern_type("123-45-6789") in ("ssn", "ssn_middle")
    assert mp.MaskingPatterns.detect_pattern_type("") is None
    assert mp.MaskingPatterns.detect_pattern_type(None) is None
    assert mp.MaskingPatterns.detect_pattern_type("not_a_match") == "patient_id"

def test_apply_pattern_mask_email():
    pattern = mp.MaskingPatterns.get_pattern("email")
    masked = mp.apply_pattern_mask("john.doe@example.com", pattern)
    assert masked.startswith("jo")
    assert masked.endswith("@example.com")
    assert "*" in masked

def test_apply_pattern_mask_invalid():
    pattern = mp.MaskingPatterns.get_pattern("email")
    # Too short
    assert mp.apply_pattern_mask("a@b.c", pattern) == "a@b.c"
    # No match
    assert mp.apply_pattern_mask("notanemail", pattern) == "notanemail"
    # No regex
    pattern2 = mp.PatternConfig(regex=None, mask_groups=[1], preserve_groups=[2], description="desc")
    assert mp.apply_pattern_mask("foo", pattern2) == "foo"
    # Not a string
    assert mp.apply_pattern_mask(12345, pattern) == 12345

def test_reconstruct_from_groups():
    # Should reconstruct with masked group
    regex = r"^(\d{3})-(\d{2})-(\d{4})$"
    original = "123-45-6789"
    masked_groups = ["***", "**", "6789"]
    result = mp._reconstruct_from_groups(original, regex, masked_groups)
    assert result.endswith("6789")
    # If no match, returns joined masked_groups
    assert mp._reconstruct_from_groups("foo", regex, ["a", "b", "c"]) == "abc"

def test_create_random_mask():
    mask = mp.create_random_mask(10)
    assert len(mask) == 10
    assert all(c in mp.MASK_CHAR_POOLS["alphanumeric"] for c in mask)
    # Custom pool
    mask2 = mp.create_random_mask(5, char_pool="XYZ")
    assert set(mask2) <= set("XYZ")

def test_validate_mask_character():
    assert mp.validate_mask_character("*")
    assert not mp.validate_mask_character("A")
    assert not mp.validate_mask_character("1")
    assert not mp.validate_mask_character(" ")
    assert not mp.validate_mask_character("")
    assert not mp.validate_mask_character("**")

def test_analyze_pattern_security():
    pattern = mp.MaskingPatterns.get_pattern("email")
    test_values = ["john.doe@example.com", "a@b.com"]
    result = mp.analyze_pattern_security(pattern, test_values)
    assert "pattern_type" in result
    assert isinstance(result["visibility_scores"], list)
    assert "warnings" in result

def test_get_format_preserving_mask():
    masked = mp.get_format_preserving_mask("123-45-6789")
    assert masked.count("-") == 2
    assert set(masked.replace("-", "")) == {"*"}
    # Empty
    assert mp.get_format_preserving_mask("") == ""

def test_generate_mask():
    mask = mp.generate_mask("#", False, "XYZ", 5)
    assert mask == "#####"
    mask2 = mp.generate_mask("#", True, "XYZ", 5)
    assert len(mask2) == 5 and set(mask2) <= set("XYZ")

def test_generate_mask_char():
    c = mp.generate_mask_char("#", False, "XYZ")
    assert c == "#"
    c2 = mp.generate_mask_char("#", True, "XYZ")
    assert c2 in "XYZ"

def test_is_separator():
    for sep in mp.DEFAULT_SEPARATORS:
        assert mp.is_separator(sep)
    assert not mp.is_separator("A")
    assert not mp.is_separator("1")

def test_preserve_pattern_mask():
    # Only digits preserved, rest masked
    masked = mp.preserve_pattern_mask("abc-123-xyz", "*", False, "*", r"\d+", True)
    assert masked.count("*") > 0
    assert "123" in masked
    # Separators preserved
    assert "-" in masked
    # No separators preserved
    masked2 = mp.preserve_pattern_mask("abc-123-xyz", "#", False, "#", r"\d+", False)
    assert "-" not in masked2 or masked2.count("-") < masked.count("-")

def test_get_set_clear_mask_char_pool():
    orig = mp.get_mask_char_pool("symbols")
    mp.set_mask_char_pool("testpool", "XYZ")
    assert mp.get_mask_char_pool("testpool") == "XYZ"
    mp.clear_mask_char_pools()
    # After clear, testpool should not exist, symbols should be default
    assert mp.get_mask_char_pool("symbols") == "*#@$%^&!+-=~"

def test_pattern_config_dataclass():
    pc = mp.PatternConfig(regex="foo", mask_groups=[1], preserve_groups=[2], description="desc", preserve_separators=False, min_length=3, validation_regex="bar")
    assert pc.regex == "foo"
    assert pc.mask_groups == [1]
    assert pc.preserve_groups == [2]
    assert pc.description == "desc"
    assert pc.preserve_separators is False
    assert pc.min_length == 3
    assert pc.validation_regex == "bar"
