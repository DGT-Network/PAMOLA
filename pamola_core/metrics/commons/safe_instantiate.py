"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Safe Metric Instantiation
Package:       pamola_core.metrics.commons.safe_instantiate
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides a utility for safely instantiating metric classes by filtering out
  invalid constructor parameters. Ensures robust and error-free metric object
  creation from dynamic or user-supplied configurations.

Key Features:
  - Inspects metric class constructors to determine valid parameters
  - Filters user or config-supplied parameter dictionaries
  - Prevents runtime errors from unexpected or invalid arguments
  - Supports flexible and dynamic metric instantiation

Design Principles:
  - Fail-safe instantiation for metric operations
  - Type-safe, testable, and integration-ready
  - Minimal dependencies, focused on robustness

Dependencies:
  - inspect - Signature inspection for Python callables
  - typing  - Type hints for clarity and safety
"""

import inspect
from typing import Type, Any, Dict


def safe_instantiate(metric_class: Type, params: Dict[str, Any]) -> Any:
    """
    Safely instantiate a metric class by filtering out invalid constructor parameters.

    Parameters:
    -----------
    metric_class : Type
        The class to be instantiated.

    params : Dict[str, Any]
        The raw parameters passed (e.g., from user config or kwargs).

    Returns:
    --------
    Any
        An instance of `metric_class` with only valid constructor arguments.
    """
    sig = inspect.signature(metric_class.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in params.items() if k in valid_keys}
    return metric_class(**filtered_params)
