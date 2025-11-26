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
from typing import Any, Optional
import inspect
import logging
from pamola_core.utils.ops.op_data_processing import force_garbage_collection

# Configure logger
logger = logging.getLogger(__name__)


def filter_used_kwargs(kwargs: dict, func) -> dict:
    """
    Remove keys from kwargs that conflict with the named parameters of the given function.

    :param kwargs: A dictionary of keyword arguments to filter.
    :param func: The target function or method to check against.
    :return: A filtered kwargs dictionary excluding keys that match the function's parameters.
    """
    used_keys = set(inspect.signature(func).parameters)
    return {k: v for k, v in kwargs.items() if k not in used_keys}

def cleanup_memory(
        instance: Optional[Any] = None,
        force_gc: bool = True
    ) -> None:
    """
        Cleans up memory by clearing specific attributes of the provided instance.
        This function performs the following actions:
        - Clears the `operation_cache` attribute if it exists.
        - Resets the `process_kwargs` attribute to an empty dictionary if it exists.
        - Clears the `filter_mask` attribute if it exists.
        - Additionally, it removes any attributes that start with `_temp_` from the instance.
        - If `force_gc` is set to True, it triggers garbage collection.
        Args:
            instance (Optional[Any]): The instance from which to clear attributes. 
                                       If None, only class-level attributes will be cleared.
            force_gc (bool): A flag indicating whether to force garbage collection.
                             Defaults to True.
        Returns:
            None: This function does not return any value.
    """
    try:
        if instance is None:
            return
        
        # Clear operation cache
        if hasattr(instance, "operation_cache"):
            instance.operation_cache = None

        # Clear process kwargs
        if hasattr(instance, "process_kwargs"):
            instance.process_kwargs = {}

        # Clear filter mask
        if hasattr(instance, "filter_mask"):
            instance.filter_mask = None

        # Clear original dataframe cache
        if hasattr(instance, "_original_df"):
            instance._original_df = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(instance).keys()):
            if attr_name.startswith("_temp_"):
                delattr(instance, attr_name)
        
        # Force GC if explicitly requested
        if force_gc:
            force_garbage_collection()
    except Exception as e:
        logger.warning(f"Error cleanup_memory: {e}")