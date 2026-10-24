"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Clean Invalid Values Core Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of clean invalid values configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines data cleaning parameters for constraint validation and null replacement
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field constraint configuration for validation rules
- Whitelist and blacklist path management
- Null replacement strategy specification
- Flexible data cleaning strategies

Changelog:
1.0.0 - 2025-01-15 - Initial creation of clean invalid values core schema
1.1.0 - 2025-11-11 - Updated with enhanced constraint descriptions
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class CleanInvalidValuesOperationConfig(OperationConfig):
    """
    Core configuration schema for CleanInvalidValuesOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Clean Invalid Values Operation Core Configuration",
        "description": "Core schema for Clean Invalid Values operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_constraints": {
                        "type": ["object", "null"],
                        "title": "Field Constraints",
                        "description": (
                            "Dictionary defining validation rules for each field to identify invalid values.\n"
                            "Each field names one 'constraint_type' (a missing or unknown one is rejected). A constraint\n"
                            "that cannot apply to the field's kind is rejected too, never skipped silently.\n"
                            "Numeric fields (numeric dtype or \"data_type\": \"numeric\"): min_value, max_value (with the\n"
                            "same-named key), valid_range (min_value + max_value), custom_function.\n"
                            "Categorical fields (pandas category dtype or \"data_type\": \"categorical\"): allowed_values,\n"
                            "disallowed_values (value lists), whitelist_file, blacklist_file, pattern.\n"
                            "Date fields (datetime dtype or \"data_type\": \"date\"): min_date, max_date, date_range, valid_format.\n"
                            "Text fields (string / object dtype or \"data_type\": \"text\"): min_length, max_length,\n"
                            "valid_pattern, valid_format.\n"
                            "Invalid values become null; replace them with null_replacement.\n\n"
                            "Example: {'age': {'constraint_type': 'valid_range', 'min_value': 0, 'max_value': 150},\n"
                            "'status': {'data_type': 'categorical', 'constraint_type': 'allowed_values',\n"
                            "'allowed_values': ['active', 'inactive']}}"
                        ),
                        "default": None,
                    },
                    "whitelist_path": {
                        "type": ["object", "null"],
                        "title": "Whitelist Path",
                        "description": (
                            "Dictionary mapping field names to file paths containing allowed values.\n"
                            "Values not in the whitelist will be nullified.\n\n"
                            "Example: {'country': 'valid_countries.txt', 'department': 'valid_depts.txt'}\n"
                            "File format: One valid value per line"
                        ),
                        "default": None,
                    },
                    "blacklist_path": {
                        "type": ["object", "null"],
                        "title": "Blacklist Path",
                        "description": (
                            "Dictionary mapping field names to file paths containing prohibited values.\n"
                            "Values found in the blacklist will be nullified.\n\n"
                            "Example: {'username': 'blocked_names.txt', 'comment': 'profanity_list.txt'}\n"
                            "File format: One invalid value per line"
                        ),
                        "default": None,
                    },
                    "null_replacement": {
                        "type": ["string", "object", "null"],
                        "title": "Null Replacement",
                        "description": (
                            "Strategy or value to use when replacing invalid entries.\n\n"
                            "Options:\n"
                            "- String value: Use the same replacement for all fields (e.g., 'INVALID', 'N/A')\n"
                            "- Dictionary: Specify different replacements per field (e.g., {'age': -1, 'status': 'unknown'})\n"
                            "- Statistical methods: 'mean', 'median', 'mode' (for numeric fields)\n"
                            "- null/None: Replace with null (default)\n\n"
                            "Default: null"
                        ),
                        "default": None,
                    },
                },
            },
        ],
    }
