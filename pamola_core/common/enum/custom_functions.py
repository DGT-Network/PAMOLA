"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Custom Functions
Package:       pamola_core.common.enum
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines custom function names used in form schemas for dynamic field updates.
"""


class CustomFunctions:
    """Custom function names for x-custom-function attributes in schemas."""

    UPDATE_FIELD_OPTIONS = "update_field_options"
    UPDATE_CONDITION_OPERATOR = "update_condition_operator"
    UPDATE_CONDITION_VALUES = "update_condition_values"
    UPDATE_QUASI_FIELD_OPTIONS = "update_quasi_field_options"
    UPDATE_EXCLUSIVE_FIELD_OPTIONS = "update_exclusive_field_options"
    UPDATE_INT64_FIELD_OPTIONS = "update_int64_field_options"
