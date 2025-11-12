"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Clean Invalid Values Config Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating clean invalid values operations in PAMOLA.CORE.
- Supports constraint-based validation for identifying invalid values
- Handles whitelist/blacklist filtering from external files
- Provides flexible null replacement strategies (static values, statistical methods)
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of clean invalid values config file
1.1.0 - 2025-11-11 - Updated with x-component, x-group, and enhanced descriptions
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class CleanInvalidValuesOperationConfig(OperationConfig):
    """
    Configuration schema for CleanInvalidValuesOperation.

    Extends BaseOperationConfig with data cleaning parameters for constraint-based validation,
    whitelist/blacklist filtering, and null value replacement strategies.
    """

    schema = {
        "type": "object",
        "title": "Clean Invalid Values Operation Configuration",
        "description": (
            "Configuration options for cleaning or nullifying invalid values in data fields. "
            "Supports constraint validation, whitelist/blacklist filtering, and null replacement."
        ),
        "allOf": [
            BaseOperationConfig.schema,  # merge base schema
            {
                "type": "object",
                "properties": {
                    # === Validation constraints ===
                    "field_constraints": {
                        "type": ["object", "null"],
                        "title": "Field Constraints",
                        "x-component": "Input",
                        "description": (
                            "Dictionary defining validation rules for each field to identify invalid values.\n"
                            "Constraint types:\n"
                            "- min/max: Numeric range boundaries\n"
                            "- pattern: Regular expression pattern for text validation\n"
                            "- allowed_values: List of valid values\n"
                            "- data_type: Expected data type (int, float, str, etc.)\n"
                            "- not_null: Whether null values are allowed\n\n"
                            "Example: {'age': {'min': 0, 'max': 150}, 'email': {'pattern': r'^[\\w.-]+@'}}"
                        ),
                        "x-group": GroupName.FIELD_CONSTRAINTS_CONFIGURATION,
                        "default": None,
                    },
                    # === Whitelist/Blacklist filtering ===
                    "whitelist_path": {
                        "type": ["object", "null"],
                        "title": "Whitelist Path",
                        "x-component": "Input",
                        "description": (
                            "Dictionary mapping field names to file paths containing allowed values.\n"
                            "Values not in the whitelist will be nullified.\n\n"
                            "Example: {'country': 'valid_countries.txt', 'department': 'valid_depts.txt'}\n"
                            "File format: One valid value per line"
                        ),
                        "x-group": GroupName.WHITELIST_CONFIGURATION,
                        "default": None,
                    },
                    "blacklist_path": {
                        "type": ["object", "null"],
                        "title": "Blacklist Path",
                        "x-component": "Input",
                        "description": (
                            "Dictionary mapping field names to file paths containing prohibited values.\n"
                            "Values found in the blacklist will be nullified.\n\n"
                            "Example: {'username': 'blocked_names.txt', 'comment': 'profanity_list.txt'}\n"
                            "File format: One invalid value per line"
                        ),
                        "x-group": GroupName.BLACKLIST_CONFIGURATION,
                        "default": None,
                    },
                    # === Null replacement strategy ===
                    "null_replacement": {
                        "type": ["string", "object", "null"],
                        "title": "Null Replacement",
                        "x-component": "Input",
                        "description": (
                            "Strategy or value to use when replacing invalid entries.\n\n"
                            "Options:\n"
                            "- String value: Use the same replacement for all fields (e.g., 'INVALID', 'N/A')\n"
                            "- Dictionary: Specify different replacements per field (e.g., {'age': -1, 'status': 'unknown'})\n"
                            "- Statistical methods: 'mean', 'median', 'mode' (for numeric fields)\n"
                            "- null/None: Replace with null (default)\n\n"
                            "Default: null"
                        ),
                        "x-group": GroupName.NULL_REPLACEMENT_CONFIGURATION,
                        "default": None,
                    },
                },
            },
        ],
    }
