"""
PAMOLA.CORE - Utility Functions
----------------------------------------------------
Module: Helpers
Description: Utility functions for various operations used in modules.
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Filtering keyword arguments based on function signatures
- General utility operations for data preparation and processing
"""
from typing import Dict, List, Any, Optional, Tuple
import inspect


def filter_used_kwargs(kwargs: dict, func) -> dict:
    """
    Remove keys from kwargs that conflict with the named parameters of the given function.

    :param kwargs: A dictionary of keyword arguments to filter.
    :param func: The target function or method to check against.
    :return: A filtered kwargs dictionary excluding keys that match the function's parameters.
    """
    used_keys = set(inspect.signature(func).parameters)
    return {k: v for k, v in kwargs.items() if k not in used_keys}