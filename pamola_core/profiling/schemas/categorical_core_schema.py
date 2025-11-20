"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Categorical Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of categorical profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines categorical profiling parameters, top N values, frequency thresholds, and profile types
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name specification for categorical analysis
- Top N values and minimum frequency controls
- Profile type configuration (categorical, string, text)
- Anomaly analysis enablement

Changelog:
1.0.0 - 2025-01-15 - Initial creation of categorical profiler core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class CategoricalOperationConfig(OperationConfig):
    """
    Core configuration schema for CategoricalOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Categorical Operation Configuration",
        "description": "Core schema for categorical profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the categorical field to analyze.",
                    },
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 15,
                        "title": "Top N Values",
                        "description": "Number of most frequent values to include in the results and visualizations.",
                    },
                    "min_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                        "title": "Minimum Frequency",
                        "description": "Minimum frequency a value must have to be included in the value dictionary.",
                    },
                    "profile_type": {
                        "type": "string",
                        "default": "categorical",
                        "title": "Profile Type",
                        "oneOf": [
                            {"const": "categorical", "description": "Categorical"},
                            {"const": "string", "description": "String"},
                            {"const": "text", "description": "Text"},
                        ],
                        "description": "Type of profiling for organizing artifacts and analysis: 'categorical', 'string', or 'text'.",
                    },
                    "analyze_anomalies": {
                        "type": "boolean",
                        "default": True,
                        "title": "Analyze Anomalies",
                        "description": "If true, analyze the field for anomalies such as typos, rare values, and unusual patterns.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
