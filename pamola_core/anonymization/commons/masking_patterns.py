"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Masking Patterns Library
Package:       pamola_core.anonymization.commons.masking_patterns
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
    Provides a library of common masking patterns and utilities for data anonymization.
    Supports configurable regex-based masking for emails, phone numbers, SSNs, credit cards,
    IP addresses, account numbers, and other sensitive identifiers.

Key Features:
    - Centralized pattern configuration for common identifier types
    - Regex-based masking with flexible group selection
    - Support for preserving separators and minimum length constraints
    - Pattern detection utility for automatic type inference
    - Easy extension for new masking patterns

Design Principles:
    - Configurable, extensible, and testable masking logic
    - Separation of pattern definition and masking application
    - Minimal dependencies, focused on privacy and robustness

Dependencies:
    - re       - Regular expressions for pattern matching
    - dataclasses - Structured pattern configuration
    - typing   - Type hints for clarity and safety
    - random, string - Utilities for advanced masking
"""

import re
import random
import string
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pamola_core.utils import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternConfig:
    """
    Configuration for a masking pattern.
    """

    regex: str
    mask_groups: List[int]
    preserve_groups: List[int]
    description: str
    preserve_separators: bool = True
    min_length: int = 0
    validation_regex: Optional[str] = None


# Common regex patterns for validation
EMAIL_REGEX = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
PHONE_REGEX = r"^\d{3}-?\d{3}-?\d{4}$"
SSN_REGEX = r"^\d{3}-?\d{2}-?\d{4}$"
CREDIT_CARD_REGEX = r"^\d{4}-?\d{4}-?\d{4}-?\d{4}$"
IP_REGEX = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
URL_REGEX = r"^https?://[^\s/$.?#].[^\s]*$"
DATE_MDY_REGEX = r"^(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](\d{4})$"
DATE_DMY_REGEX = r"^(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](\d{4})$"
DATE_YMD_REGEX = r"^(\d{4})[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])$"
ISO_DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"
DEFAULT_SEPARATORS = {"-", "_", ".", "@", "/", "\\", ":", ";", " ", "(", ")", "[", "]"}


class MaskingPatterns:
    """
    Static container for commonly used masking patterns.
    """

    PATTERNS: Dict[str, PatternConfig] = {
        # === Email patterns ===
        "email": PatternConfig(
            regex=r"^([^@]{1,2})([^@]+)(@.+)$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep first 1–2 chars and domain",
            preserve_separators=True,
            validation_regex=EMAIL_REGEX,
            min_length=5,
        ),
        "email_domain": PatternConfig(
            regex=r"^([^@]+)(@[^.]+)(\..+)$",
            mask_groups=[1],
            preserve_groups=[2, 3],
            description="Keep only TLD",
            preserve_separators=True,
            validation_regex=EMAIL_REGEX,
            min_length=5,
        ),
        # === Phone patterns ===
        "phone": PatternConfig(
            regex=r"^(\d{3})-?(\d{3})-?(\d{4})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep area code and last 4",
            preserve_separators=True,
            validation_regex=PHONE_REGEX,
            min_length=10,
        ),
        "phone_international": PatternConfig(
            regex=r"^(\+\d{1,3})-?(\d+)-?(\d{4})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep country code and last 4",
            preserve_separators=True,
            validation_regex=r"^\+\d{1,3}-?\d+-?\d{4}$",
            min_length=8,
        ),
        "phone_us_formatted": PatternConfig(
            regex=r"^\((\d{3})\) (\d{3})-(\d{4})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep area code and last 4 (US format)",
            preserve_separators=True,
            validation_regex=r"^\(\d{3}\) \d{3}-\d{4}$",
            min_length=14,
        ),
        "phone_us_compact": PatternConfig(
            regex=r"^\((\d{3})\)(\d{3})-(\d{4})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep area code and last 4 (US compact format)",
            preserve_separators=True,
            validation_regex=r"^\(\d{3}\)\d{3}-\d{4}$",
            min_length=13,
        ),
        # === SSN ===
        "ssn": PatternConfig(
            regex=r"^(\d{3})-?(\d{2})-?(\d{4})$",
            mask_groups=[1, 2],
            preserve_groups=[3],
            description="Keep last 4 digits only",
            preserve_separators=True,
            validation_regex=SSN_REGEX,
            min_length=9,
        ),
        "ssn_middle": PatternConfig(
            regex=r"^(\d{3})-?(\d{2})-?(\d{4})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep first 3 and last 4 digits",
            preserve_separators=True,
            validation_regex=SSN_REGEX,
            min_length=9,
        ),
        # === Credit card ===
        "credit_card": PatternConfig(
            regex=r"^(\d{4})-?(\d{4})-?(\d{4})-?(\d{4})$",
            mask_groups=[2, 3],
            preserve_groups=[1, 4],
            description="Keep first 4 and last 4 (PCI compliant)",
            preserve_separators=True,
            validation_regex=CREDIT_CARD_REGEX,
            min_length=13,
        ),
        "credit_card_strict": PatternConfig(
            regex=r"^(\d{4})-?(\d{4})-?(\d{4})-?(\d{4})$",
            mask_groups=[1, 2, 3],
            preserve_groups=[4],
            description="Keep last 4 only (strict)",
            preserve_separators=True,
            validation_regex=CREDIT_CARD_REGEX,
            min_length=13,
        ),
        # === IP address ===
        "ip_address": PatternConfig(
            regex=r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$",
            mask_groups=[3, 4],
            preserve_groups=[1, 2],
            description="Keep first two octets",
            preserve_separators=True,
            validation_regex=IP_REGEX,
            min_length=7,
        ),
        "ip_address_last_only": PatternConfig(
            regex=r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$",
            mask_groups=[2, 3, 4],
            preserve_groups=[1],
            description="Keep first octet only",
            preserve_separators=True,
            validation_regex=IP_REGEX,
            min_length=7,
        ),
        # === Date patterns ===
        "date_mdy": PatternConfig(
            regex=r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep month and year (MM/DD/YYYY)",
            preserve_separators=True,
            validation_regex=DATE_MDY_REGEX,
            min_length=8,
        ),
        "date_dmy": PatternConfig(
            regex=r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$",
            mask_groups=[1],
            preserve_groups=[2, 3],
            description="Keep month and year (DD/MM/YYYY)",
            preserve_separators=True,
            validation_regex=DATE_DMY_REGEX,
            min_length=8,
        ),
        "date_ymd": PatternConfig(
            regex=r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$",
            mask_groups=[3],
            preserve_groups=[1, 2],
            description="Keep year and month (YYYY/MM/DD)",
            preserve_separators=True,
            validation_regex=DATE_YMD_REGEX,
            min_length=8,
        ),
        "date_iso": PatternConfig(
            regex=r"^(\d{4})-(\d{2})-(\d{2})$",
            mask_groups=[3],
            preserve_groups=[1, 2],
            description="Keep year and month (ISO format)",
            preserve_separators=True,
            validation_regex=ISO_DATE_REGEX,
            min_length=10,
        ),
        "date_year_only": PatternConfig(
            regex=r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$",
            mask_groups=[2, 3],
            preserve_groups=[1],
            description="Keep year only",
            preserve_separators=True,
            validation_regex=DATE_YMD_REGEX,
            min_length=8,
        ),
        "birthdate": PatternConfig(
            regex=r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$",
            mask_groups=[1, 2],
            preserve_groups=[3],
            description="Keep birth year only (MM/DD/YYYY)",
            preserve_separators=True,
            validation_regex=DATE_MDY_REGEX,
            min_length=8,
        ),
        "birthdate_dmy": PatternConfig(
            regex=r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$",
            mask_groups=[1, 2],
            preserve_groups=[3],
            description="Keep birth year only (DD/MM/YYYY)",
            preserve_separators=True,
            validation_regex=DATE_DMY_REGEX,
            min_length=8,
        ),
        "date_month_year": PatternConfig(
            regex=r"^(\d{1,2})[/-](\d{4})$",
            mask_groups=[1],
            preserve_groups=[2],
            description="Keep year only (MM/YYYY)",
            preserve_separators=True,
            validation_regex=r"^(0?[1-9]|1[0-2])[/-]\d{4}$",
            min_length=6,
        ),
        "date_dotted": PatternConfig(
            regex=r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$",
            mask_groups=[1],
            preserve_groups=[2, 3],
            description="Keep month and year (DD.MM.YYYY)",
            preserve_separators=True,
            validation_regex=r"^(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.\d{4}$",
            min_length=8,
        ),
        # === Financial / ID ===
        "account_number": PatternConfig(
            regex=r"^(.{2})(.+)(.{4})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep first 2 and last 4 chars",
            preserve_separators=True,
            min_length=7,
        ),
        "account_number_last_only": PatternConfig(
            regex=r"^(.+)(.{4})$",
            mask_groups=[1],
            preserve_groups=[2],
            description="Keep last 4 chars only",
            preserve_separators=True,
            min_length=5,
        ),
        # === Government IDs ===
        "license_plate": PatternConfig(
            regex=r"^([A-Z]{2,3})(\d{2,4})([A-Z]{0,2})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep letter portions",
            preserve_separators=True,
            validation_regex=r"^[A-Z]{2,3}\d{2,4}[A-Z]{0,2}$",
            min_length=4,
        ),
        "driver_license": PatternConfig(
            regex=r"^([A-Z]{1,2})(\d+)$",
            mask_groups=[2],
            preserve_groups=[1],
            description="Keep state prefix",
            preserve_separators=False,
            validation_regex=r"^[A-Z]{1,2}\d+$",
            min_length=3,
        ),
        "passport": PatternConfig(
            regex=r"^([A-Z]{2})(\d{7})$",
            mask_groups=[2],
            preserve_groups=[1],
            description="Keep country code only",
            preserve_separators=False,
            validation_regex=r"^[A-Z]{2}\d{7}$",
            min_length=9,
        ),
        "iban": PatternConfig(
            regex=r"^([A-Z]{2}\d{2})(\d{4})(\d{4})(\d{4})(\d{4})(\d{0,4})$",
            mask_groups=[2, 3, 4, 5],
            preserve_groups=[1, 6],
            description="Keep country code and last portion",
            preserve_separators=True,
            validation_regex=r"^[A-Z]{2}\d{2}[\d]{4,20}$",
            min_length=15,
        ),
        # === Web identifiers ===
        "url": PatternConfig(
            regex=r"^(https?://)([^/]+)(.*)$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep protocol and path",
            preserve_separators=True,
            validation_regex=URL_REGEX,
            min_length=10,
        ),
        "username": PatternConfig(
            regex=r"^(.{2})(.+)(.{2})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep first 2 and last 2 characters",
            preserve_separators=True,
            min_length=5,
        ),
        # === Healthcare ===
        "medical_record": PatternConfig(
            regex=r"^([A-Z]{2,3})(\d+)$",
            mask_groups=[2],
            preserve_groups=[1],
            description="Keep facility code",
            preserve_separators=False,
            validation_regex=r"^[A-Z]{2,3}\d+$",
            min_length=4,
        ),
        "health_insurance_number": PatternConfig(
            regex=r"^([A-Z]{1,3})(\d{5,9})$",
            mask_groups=[2],
            preserve_groups=[1],
            description="Keep insurer prefix, mask insurance number",
            preserve_separators=False,
            validation_regex=r"^[A-Z]{1,3}\d{5,9}$",
            min_length=6,
        ),
        "icd10_code": PatternConfig(
            regex=r"^([A-Z])(\d{2})(\.\d+)?$",
            mask_groups=[2, 3],
            preserve_groups=[1],
            description="Keep category letter, mask detailed diagnosis",
            preserve_separators=True,
            validation_regex=r"^[A-Z]\d{2}(\.\d+)?$",
            min_length=3,
        ),
        "patient_id": PatternConfig(
            regex=r"^(.{2})(.+)(.{2})$",
            mask_groups=[2],
            preserve_groups=[1, 3],
            description="Keep first 2 and last 2 characters of patient ID",
            preserve_separators=False,
            validation_regex=r"^.{6,}$",
            min_length=6,
        ),
    }

    @classmethod
    def get_pattern(cls, pattern_type: str) -> Optional[PatternConfig]:
        """
        Get pattern configuration by type.

        Args:
            pattern_type (str): Name of pattern.

        Returns:
            Optional[PatternConfig]: Matching pattern config, or None.
        """
        return cls.PATTERNS.get(pattern_type)

    @staticmethod
    def get_default_patterns() -> Dict[str, PatternConfig]:
        """
        Return a copy of the default masking patterns.
        This avoids accidental modification of the static dictionary.
        """
        return MaskingPatterns.PATTERNS.copy()

    @classmethod
    def get_pattern_names(cls) -> List[str]:
        """
        List all available pattern types.

        Returns:
            List[str]: List of pattern names.
        """
        return list(cls.PATTERNS)

    @classmethod
    def validate_pattern_type(cls, pattern_type: str) -> bool:
        """
        Check if a pattern type exists.

        Args:
            pattern_type (str): Name to test.

        Returns:
            bool: True if pattern_type is registered.
        """
        return pattern_type in cls.PATTERNS

    @classmethod
    def detect_pattern_type(cls, value: str) -> Optional[str]:
        """
        Attempt to guess the pattern type of a value.

        Args:
            value (str): Value to analyze.

        Returns:
            Optional[str]: Most probable pattern type, or None.
        """
        if not isinstance(value, str) or not value.strip():
            return None

        candidates = []
        for name, config in cls.PATTERNS.items():
            pattern = config.validation_regex or config.regex
            if not pattern:
                continue

            match = re.match(pattern, value)
            if match:
                # Use higher score if validation_regex was used
                score = 1.0 if config.validation_regex else len(match.groups()) / 5.0
                candidates.append((name, score))

        if not candidates:
            return None

        # Return pattern type with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


def apply_pattern_mask(
    value: str, pattern_config: PatternConfig, mask_char: str = "*"
) -> str:
    """
    Apply pattern-based masking to a string value based on regex groups.

    Args:
        value (str): The input value to mask.
        pattern_config (PatternConfig): The pattern configuration, including:
            - regex (str): Regex pattern with groups.
            - min_length (int): Minimum required length to apply masking.
            - mask_groups (List[int]): List of group indices to mask.
        mask_char (str): Character used for masking. Default is "*".

    Returns:
        str: Masked value, or original value if pattern doesn't match.
    """
    if not isinstance(value, str) or not value:
        return value

    if len(value) < pattern_config.min_length:
        logger.warning(
            f"Value too short for pattern masking: {len(value)} < {pattern_config.min_length}"
        )
        return value

    if not pattern_config.regex:
        logger.warning("No regex pattern provided in pattern config.")
        return value

    match = re.match(pattern_config.regex, value)
    if not match:
        logger.warning(
            f"Value does not match the regex pattern: {pattern_config.regex}"
        )
        return value

    # Mask specified groups
    result_parts = []
    for i, group in enumerate(match.groups(), 1):
        if i in pattern_config.mask_groups:
            result_parts.append(mask_char * len(group))
        else:
            result_parts.append(group)

    return _reconstruct_from_groups(value, pattern_config.regex, result_parts)


def _reconstruct_from_groups(
    original: str, regex: str, masked_groups: List[str]
) -> str:
    """
    Reconstruct a masked string from masked groups while preserving non-captured text.

    Args:
        original (str): The original input string.
        regex (str): The regex used to extract groups.
        masked_groups (List[str]): The list of masked/unmasked group values.

    Returns:
        str: The reconstructed string with separators preserved.
    """
    try:
        pattern = re.compile(regex)
        match = pattern.match(original)
        if not match:
            return "".join(masked_groups)

        start, end = match.span()
        prefix = original[:start]
        suffix = original[end:]
        group_starts = [match.start(i + 1) - start for i in range(len(masked_groups))]
        group_ends = [match.end(i + 1) - start for i in range(len(masked_groups))]

        reconstructed = ""
        last_idx = 0
        for gs, ge, mg in zip(group_starts, group_ends, masked_groups):
            if gs > last_idx:
                reconstructed += original[start + last_idx : start + gs]
            reconstructed += mg
            last_idx = ge

        # Append any remaining part in the matched span
        if last_idx < end - start:
            reconstructed += original[start + last_idx : end]

        return prefix + reconstructed + suffix

    except Exception as e:
        logger.error(f"Error reconstructing from regex groups: {e}")
        return "".join(masked_groups)


def create_random_mask(length: int, char_pool: Optional[str] = None) -> str:
    """
    Create a random string of a specified length using characters from a pool.

    Args:
        length (int): Desired length of the mask.
        char_pool (Optional[str]): Optional character pool. If None, defaults to symbols.

    Returns:
        str: Random mask string.
    """
    if char_pool is None:
        char_pool = MASK_CHAR_POOLS["alphanumeric"]
    return "".join(random.choice(char_pool) for _ in range(length))


def validate_mask_character(mask_char: str) -> bool:
    """
    Validate that the mask character is safe and doesn't reveal information.

    Parameters
    ----------
    mask_char : str
        The character intended for masking. Should be a single, non-alphanumeric, non-whitespace symbol.

    Returns
    -------
    bool
        True if character is valid and safe for masking; otherwise False.
    """
    if not mask_char or len(mask_char) != 1:
        return False

    unsafe_chars = set(string.ascii_letters + string.digits)

    # Reject alphanumeric and whitespace characters
    if mask_char.isspace() or mask_char in unsafe_chars:
        return False

    return True


def analyze_pattern_security(
    pattern_config: PatternConfig, test_values: List[str]
) -> Dict[str, Any]:
    """
    Analyze the masking pattern's ability to conceal data and identify risks.

    Parameters
    ----------
    pattern_config : PatternConfig
        The configuration that defines regex pattern, min length, and mask/preserve group indices.

    test_values : List[str]
        Sample input values to evaluate the pattern's visibility exposure.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
            - pattern_type (str)
            - min_length (int)
            - mask_groups (List[int])
            - preserve_groups (List[int])
            - visibility_scores (List[float])
            - avg_visibility (float)
            - max_visibility (float)
            - warnings (List[str])
    """
    results = {
        "pattern_type": pattern_config.description,
        "min_length": pattern_config.min_length,
        "mask_groups": pattern_config.mask_groups,
        "preserve_groups": pattern_config.preserve_groups,
        "visibility_scores": [],
        "avg_visibility": 0.0,
        "max_visibility": 0.0,
        "warnings": [],
    }

    for test_value in test_values:
        if not test_value:
            continue
        try:
            masked = apply_pattern_mask(test_value, pattern_config)
            visible_chars = sum(
                1 for i, c in enumerate(masked) if i < len(test_value) and c != "*"
            )
            visibility = visible_chars / len(test_value)
            results["visibility_scores"].append(visibility)

            if visibility > 0.7:
                results["warnings"].append(
                    f"High visibility ({visibility:.1%}) for value: {test_value[:10]}..."
                )
        except Exception as e:
            results["warnings"].append(f"Error processing value: {e}")

    if results["visibility_scores"]:
        results["avg_visibility"] = sum(results["visibility_scores"]) / len(
            results["visibility_scores"]
        )
        results["max_visibility"] = max(results["visibility_scores"])

        if results["avg_visibility"] > 0.5:
            results["warnings"].append(
                f"Pattern exposes {results['avg_visibility']:.1%} of data on average"
            )

    preserve_count = len(results["preserve_groups"])
    total_groups = len(results["mask_groups"]) + preserve_count

    if total_groups > 0 and preserve_count / total_groups > 0.6:
        results["warnings"].append("Pattern preserves majority of groups")

    return results


def get_format_preserving_mask(value: str, mask_char: str = "*") -> str:
    """
    Mask only alphanumeric characters while preserving the original format (e.g., dashes, dots).

    Parameters
    ----------
    value : str
        The string to mask while maintaining its separators and structure.

    mask_char : str, optional
        The character to replace sensitive content with (default is '*').

    Returns
    -------
    str
        Format-preserving masked string.
    """
    if not value:
        return value

    return "".join(mask_char if c.isalnum() else c for c in value)


def generate_mask(
    mask_char: str, random_mask: bool, mask_char_pool: str, length: int
) -> str:
    """
    Generate a mask string of a specified length.

    If `random_mask` is True, use a random pool of characters.
    Otherwise, repeat a fixed mask character.
    """
    if random_mask:
        return create_random_mask(length, char_pool=mask_char_pool)
    return mask_char * length


def generate_mask_char(mask_char: str, random_mask: bool, mask_char_pool: str) -> str:
    """
    Generate a single mask character.

    Used when masking individual characters in position-based strategies.
    """
    if random_mask:
        return create_random_mask(1, char_pool=mask_char_pool)
    return mask_char


def is_separator(char: str) -> bool:
    """
    Check whether a character is considered a separator.

    Separator characters are preserved in some masking strategies (e.g., position-based).
    """
    return char in DEFAULT_SEPARATORS


def preserve_pattern_mask(
    value: str,
    mask_char: str,
    random_mask: bool,
    mask_char_pool: str,
    preserve_pattern: str,
    preserve_separators: bool,
) -> str:
    """
    Mask all characters in the input string except those that match the given pattern.

    Supports optional preservation of common separators (e.g., '-', '_', '.', etc.)
    if `preserve_separators` is True.

    Parameters
    ----------
    value : str
        Input string to be masked.

    Returns
    -------
    str
        Masked string where only pattern matches and optionally separators are preserved.
    """
    matches = list(re.finditer(preserve_pattern, value))
    result = [generate_mask_char(mask_char, random_mask, mask_char_pool) for _ in value]

    # Preserve matched sections
    for match in matches:
        start, end = match.start(), match.end()
        result[start:end] = value[start:end]

    # Optionally preserve separators
    if preserve_separators:
        for i, char in enumerate(value):
            if is_separator(char):
                result[i] = char

    return "".join(result)


# Predefined character pools for various masking strategies
MASK_CHAR_POOLS = {
    "symbols": "*#@$%^&!+-=~",
    "safe_symbols": "*#@$%",
    "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "numbers": "0123456789",
    "alphanumeric": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "extended": "*#@$%^&!+-=~ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
}


def get_mask_char_pool(pool_name: str) -> str:
    """
    Retrieve a predefined character pool used for masking.

    Args:
        pool_name (str): The name of the character pool to retrieve.
            Options include:
            - "symbols"
            - "safe_symbols"
            - "letters"
            - "numbers"
            - "alphanumeric"
            - "extended"

    Returns:
        str: A string of characters representing the mask pool.
             Defaults to 'symbols' pool if not found.
    """
    return MASK_CHAR_POOLS.get(pool_name, MASK_CHAR_POOLS["symbols"])


def set_mask_char_pool(pool_name: str, characters: str) -> None:
    """
    Define or override a custom character pool for future masking.

    Args:
        pool_name (str): The name of the new or existing character pool.
        characters (str): A string of characters to assign to the pool.
    """
    MASK_CHAR_POOLS[pool_name] = characters


def clear_mask_char_pools() -> None:
    """
    Reset all character pools to the default predefined values.
    This is useful when custom pools have been added and need to be discarded.
    """
    MASK_CHAR_POOLS.clear()
    MASK_CHAR_POOLS.update(
        {
            "symbols": "*#@$%^&!+-=~",
            "safe_symbols": "*#@$%",
            "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "numbers": "0123456789",
            "alphanumeric": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "extended": "*#@$%^&!+-=~ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        }
    )
