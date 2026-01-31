"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Phone Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of phone profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines phone analysis parameters, frequency thresholds, and country code filters
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name specification for phone analysis
- Minimum frequency controls for filtering results
- Country code filtering for targeted analysis
- Pattern CSV path for custom validation rules

Changelog:
1.0.0 - 2025-01-15 - Initial creation of phone profiler core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class PhoneOperationConfig(OperationConfig):
    """
    Core configuration schema for PhoneOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Phone Operation Core Configuration",
        "description": "Core schema for phone profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the phone number field (column) to analyze. This should be a column in the DataFrame containing phone numbers.",
                    },
                    "min_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 1,
                        "title": "Minimum Frequency",
                        "description": "Minimum number of occurrences for a phone number or component to be included in the results. Values appearing fewer times will be excluded from the output.",
                    },
                    "country_codes": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Country Codes",
                        "description": "List of country codes to restrict the analysis to specific countries. If null, all detected country codes will be included.",
                    },
                    "patterns_csv": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Patterns CSV",
                        "description": "Path to a CSV file containing phone number patterns for validation and parsing. If null, default patterns will be used.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
