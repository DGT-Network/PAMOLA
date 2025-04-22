"""
Utility functions for name processing and analysis.

This module provides helper functions for parsing, validating, and
manipulating personal names for use in fake data generation.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd

from pamola_core.utils import logging
from pamola_core.fake_data.commons import dict_helpers

# Configure logger
logger = logging.get_logger("pamola_core.fake_data.commons.name_utils")


def parse_full_name(full_name: str, language: str = "ru") -> Dict[str, str]:
    """
    Parses a full name into components based on language conventions.

    Parameters:
    -----------
    full_name : str
        Full name to parse
    language : str
        Language code (default: "ru")

    Returns:
    --------
    Dict[str, str]
        Dictionary with name parts (first_name, last_name, middle_name)
    """
    if not full_name or not isinstance(full_name, str):
        return {"first_name": "", "last_name": "", "middle_name": ""}

    # Clean and normalize the name
    name = full_name.strip()

    # Default result with all parts empty
    result = {"first_name": "", "last_name": "", "middle_name": ""}

    # Split name into parts
    parts = [part for part in name.split() if part]

    if not parts:
        return result

    # Handle different languages and conventions
    if language == "ru":
        # Russian convention: Last First Middle
        if len(parts) >= 1:
            result["last_name"] = parts[0]
        if len(parts) >= 2:
            result["first_name"] = parts[1]
        if len(parts) >= 3:
            result["middle_name"] = parts[2]
    elif language in ["en", "us", "gb"]:
        # English convention: First Middle Last
        if len(parts) >= 1:
            result["first_name"] = parts[0]
        if len(parts) >= 2:
            if len(parts) == 2:
                result["last_name"] = parts[1]
            else:
                result["middle_name"] = parts[1]
        if len(parts) >= 3:
            result["last_name"] = " ".join(parts[2:])
    elif language == "vn":
        # Vietnamese convention: Last Middle First
        if len(parts) >= 1:
            result["last_name"] = parts[0]
        if len(parts) >= 2:
            if len(parts) == 2:
                result["first_name"] = parts[1]
            else:
                result["middle_name"] = parts[1]
        if len(parts) >= 3:
            result["first_name"] = " ".join(parts[2:])
    else:
        # Default: assume First Last
        if len(parts) >= 1:
            result["first_name"] = parts[0]
        if len(parts) >= 2:
            result["last_name"] = " ".join(parts[1:])

    return result


def format_name(name_parts: Dict[str, str], format_type: str = "full", language: str = "ru") -> str:
    """
    Formats name parts into a full name based on format type and language.

    Parameters:
    -----------
    name_parts : Dict[str, str]
        Dictionary with name parts (first_name, last_name, middle_name)
    format_type : str
        Format type: "full", "last_first", "first_last", "initials", "last_initials"
    language : str
        Language code (default: "ru")

    Returns:
    --------
    str
        Formatted name
    """
    first = name_parts.get("first_name", "")
    last = name_parts.get("last_name", "")
    middle = name_parts.get("middle_name", "")

    # Handle empty inputs
    if not first and not last and not middle:
        return ""

    # Format based on type and language
    if format_type == "full":
        if language == "ru":
            # Russian: Last First Middle
            parts = [last, first, middle]
            return " ".join(p for p in parts if p)
        elif language in ["en", "us", "gb"]:
            # English: First Middle Last
            parts = [first, middle, last]
            return " ".join(p for p in parts if p)
        elif language == "vn":
            # Vietnamese: Last Middle First
            parts = [last, middle, first]
            return " ".join(p for p in parts if p)
        else:
            # Default: First Last
            parts = [first, last]
            return " ".join(p for p in parts if p)

    elif format_type == "last_first":
        if middle and language == "ru":
            return f"{last} {first} {middle}"
        else:
            return f"{last} {first}" if last and first else (last or first)

    elif format_type == "first_last":
        return f"{first} {last}" if first and last else (first or last)

    elif format_type == "initials":
        first_init = first[0] + "." if first else ""
        middle_init = middle[0] + "." if middle else ""

        if language == "ru":
            return f"{last} {first_init} {middle_init}".strip()
        else:
            return f"{first_init} {middle_init} {last}".strip()

    elif format_type == "last_initials":
        first_init = first[0] + "." if first else ""
        middle_init = middle[0] + "." if middle else ""

        if language == "ru":
            return f"{last} {first_init}{middle_init}".strip()
        else:
            return f"{first_init}{middle_init} {last}".strip()

    # Default: return whatever we have
    return " ".join(p for p in [first, last] if p)


def detect_name_format(name: str, language: str = "ru") -> str:
    """
    Attempts to detect the format of a given name.

    Parameters:
    -----------
    name : str
        Name to analyze
    language : str
        Language code for cultural context

    Returns:
    --------
    str
        Detected format: "full", "first_only", "last_only", "initials"
    """
    if not name or not isinstance(name, str):
        return "unknown"

    # Clean and split
    parts = [p for p in name.strip().split() if p]

    if not parts:
        return "unknown"

    # Check for initials
    has_initials = any(len(p) == 2 and p.endswith('.') for p in parts)

    if has_initials:
        return "initials"

    # Count parts
    if len(parts) >= 3:
        return "full"
    elif len(parts) == 2:
        # Could be First Last or Last First - hard to be sure
        return "first_last" if language in ["en", "us", "gb"] else "last_first"
    elif len(parts) == 1:
        # Single name - check if it looks like a first or last name
        # This is a heuristic and might not always be correct
        if language == "ru":
            if parts[0].endswith(('ов', 'ев', 'ин', 'ын', 'ский', 'цкий', 'ая', 'яя')):
                return "last_only"
            else:
                return "first_only"
        else:
            # For other languages, hard to tell - just guess
            return "unknown"

    return "unknown"


def generate_patronymic(father_name: str, gender: str = "M") -> str:
    """
    Generates a Russian patronymic from a father's name.

    Parameters:
    -----------
    father_name : str
        Father's first name
    gender : str
        Gender of the person ("M" or "F")

    Returns:
    --------
    str
        Generated patronymic
    """
    if not father_name:
        return ""

    # Handle only Russian names for now
    # Normalize to lowercase and remove trailing spaces
    father = father_name.lower().strip()

    # Basic rules for Russian patronymics
    if gender == "M":
        if father.endswith("й"):
            return father[:-1] + "евич"
        elif father.endswith(("а", "я")):
            return father[:-1] + "ич"
        elif father.endswith(("ь", "ъ")):
            return father[:-1] + "евич"
        elif father.endswith(("е", "ё", "и", "о", "у", "ы", "э", "ю")):
            return father + "евич"
        else:
            return father + "ович"
    else:  # Female
        if father.endswith("й"):
            return father[:-1] + "евна"
        elif father.endswith(("а", "я")):
            return father[:-1] + "ична"
        elif father.endswith(("ь", "ъ")):
            return father[:-1] + "евна"
        elif father.endswith(("е", "ё", "и", "о", "у", "ы", "э", "ю")):
            return father + "евна"
        else:
            return father + "овна"


def is_compound_name(name: str) -> bool:
    """
    Checks if a name is compound (e.g., Jean-Pierre, O'Sullivan).

    Parameters:
    -----------
    name : str
        Name to check

    Returns:
    --------
    bool
        True if the name is compound
    """
    if not name or not isinstance(name, str):
        return False

    # Check for common compound name separators
    return any(sep in name for sep in ['-', "'", ' '])


def normalize_name(name: str, keep_case: bool = True) -> str:
    """
    Normalizes a name by handling special characters and formatting.

    Parameters:
    -----------
    name : str
        Name to normalize
    keep_case : bool
        Whether to preserve the original case

    Returns:
    --------
    str
        Normalized name
    """
    if not name or not isinstance(name, str):
        return ""

    # Remove extra spaces
    normalized = " ".join(name.split())

    # Normalize unicode characters (canonical decomposition)
    normalized = unicodedata.normalize('NFD', normalized)

    # Handling special characters
    normalized = re.sub(r'[^\w\s\'\-]', '', normalized)

    # Fix capitalization if needed
    if not keep_case and normalized:
        # Capitalize each part correctly
        parts = []
        for part in normalized.split():
            # Handle hyphenated names
            if '-' in part:
                hyph_parts = part.split('-')
                part = '-'.join(p.capitalize() for p in hyph_parts)
            # Handle names with apostrophes (O'Brien)
            elif "'" in part:
                apos_index = part.find("'")
                if apos_index < len(part) - 1:  # Ensure there's a char after the apostrophe
                    part = part[:apos_index].capitalize() + "'" + part[apos_index + 1:].capitalize()
                else:
                    part = part.capitalize()
            else:
                part = part.capitalize()
            parts.append(part)

        normalized = ' '.join(parts)

    return normalized


def extract_potential_names(text: str) -> List[str]:
    """
    Extracts potential names from a text based on capitalization patterns.

    Parameters:
    -----------
    text : str
        Text to analyze

    Returns:
    --------
    List[str]
        List of potential names found in the text
    """
    if not text or not isinstance(text, str):
        return []

    # Simple pattern matching for capitalized words that might be names
    # More sophisticated analysis would require NLP
    potential_names = []

    # Find sequences of capitalized words (potential names)
    name_pattern = r'\b[A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+){0,2}\b'
    matches = re.findall(name_pattern, text)

    potential_names.extend(matches)

    return potential_names


def analyze_name_statistics(names: List[str]) -> Dict[str, Any]:
    """
    Analyzes statistics for a list of names.

    Parameters:
    -----------
    names : List[str]
        List of names to analyze

    Returns:
    --------
    Dict[str, Any]
        Dictionary with name statistics
    """
    if not names:
        return {
            "count": 0,
            "unique_count": 0,
            "uniqueness_ratio": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "top_names": [],
            "length_distribution": {}
        }

    # Clean and filter names
    clean_names = [n.strip() for n in names if n and isinstance(n, str)]

    if not clean_names:
        return {
            "count": 0,
            "unique_count": 0,
            "uniqueness_ratio": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "top_names": [],
            "length_distribution": {}
        }

    # Count and measure names
    total_count = len(clean_names)
    name_series = pd.Series(clean_names)
    value_counts = name_series.value_counts()
    unique_count = len(value_counts)

    lengths = [len(n) for n in clean_names]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    # Get distribution of lengths
    length_distribution = pd.Series(lengths).value_counts().sort_index().to_dict()

    # Get top names
    top_names = value_counts.head(10).to_dict()

    return {
        "count": total_count,
        "unique_count": unique_count,
        "uniqueness_ratio": unique_count / total_count if total_count > 0 else 0,
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "top_names": top_names,
        "length_distribution": length_distribution
    }