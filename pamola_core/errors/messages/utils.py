"""
Utility functions for error message formatting and validation.

Pure string utilities — zero dependencies on ErrorMessages or any other
errors package module. This ensures no circular imports.

Dependency graph:
    registry.py  →  utils.py   (one-way)
    utils.py     →  [stdlib only]
"""

import re
from typing import Dict, List, Tuple


def format_message(template_str: str, **kwargs) -> str:
    """
    Format a template string with given parameters.

    Accepts a raw template string (not a template name). The caller
    (ErrorMessages.format) is responsible for resolving the name → string
    lookup before calling this function.

    Args:
        template_str: Raw template string, e.g.
                      "Field '{field_name}' not found. Available: {available_fields}"
        **kwargs: Template parameters to substitute

    Returns:
        Formatted message string, or descriptive error if formatting fails

    Examples:
        >>> format_message("Field '{field_name}' not found.", field_name="age")
        "Field 'age' not found."

        >>> format_message("Field '{field_name}' not found.")
        "Field '{field_name}' not found. [ERROR: Missing required parameter 'field_name']"
    """
    try:
        return template_str.format(**kwargs)
    except KeyError as exc:
        missing_key = str(exc).strip("'\"")
        return f"{template_str} [ERROR: Missing required parameter '{missing_key}']"
    except ValueError as exc:
        return f"{template_str} [ERROR: Invalid format string - {exc}]"
    except Exception as exc:
        return (
            f"{template_str} [ERROR: Formatting failed - {type(exc).__name__}: {exc}]"
        )


def validate_template_params_str(template_str: str, **kwargs) -> Tuple[bool, List[str]]:
    """
    Validate that all required placeholders in a template string are provided.

    Accepts a raw template string (not a template name). The caller
    (ErrorMessages.validate_template_params) is responsible for the
    name → string lookup before calling this function.

    Args:
        template_str: Raw template string to inspect for {placeholder} tokens
        **kwargs: Parameters to validate against the template

    Returns:
        Tuple of (is_valid, list_of_missing_param_names)
        - is_valid: True if all placeholders have corresponding kwargs
        - list_of_missing_param_names: Empty list if is_valid is True

    Examples:
        >>> validate_template_params_str(
        ...     "Field '{field_name}' not found. Available: {available_fields}",
        ...     field_name="age"
        ... )
        (False, ['available_fields'])

        >>> validate_template_params_str(
        ...     "Field '{field_name}' not found.",
        ...     field_name="age"
        ... )
        (True, [])
    """
    placeholders = set(re.findall(r"\{(\w+)\}", template_str))
    missing = placeholders - set(kwargs.keys())
    return len(missing) == 0, sorted(missing)


def extract_template_placeholders(template_str: str) -> List[str]:
    """
    Extract all placeholder names from a template string.

    Args:
        template_str: Raw template string to inspect

    Returns:
        Sorted list of unique placeholder names

    Examples:
        >>> extract_template_placeholders(
        ...     "Field '{field_name}' not found. Available: {available_fields}"
        ... )
        ['available_fields', 'field_name']
    """
    return sorted(set(re.findall(r"\{(\w+)\}", template_str)))


def build_templates_index(cls_vars: Dict[str, object]) -> Dict[str, str]:
    """
    Build a {name: template_str} index from a class's __dict__.

    Filters to only uppercase string attributes (i.e. template constants).
    Intended to be called with vars(ErrorMessages) by registry.get_all_templates().

    Args:
        cls_vars: Result of vars(SomeClass) — typically vars(ErrorMessages)

    Returns:
        Dictionary of {template_name: template_string}

    Examples:
        >>> build_templates_index(vars(ErrorMessages))
        {'DATA_LOAD_FAILED': "Failed to load data ...", ...}
    """
    return {
        name: value
        for name, value in cls_vars.items()
        if not name.startswith("_") and isinstance(value, str) and name.isupper()
    }
