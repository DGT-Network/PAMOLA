"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Email Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of email profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines email analysis parameters, domain statistics, and privacy risk assessment
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name specification for email analysis
- Top N domains and minimum frequency controls
- Profile type configuration
- Privacy risk analysis enablement

Changelog:
1.0.0 - 2025-01-15 - Initial creation of email profiler core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class EmailOperationConfig(OperationConfig):
    """
    Core configuration schema for EmailOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Email Operation Core Configuration",
        "description": "Core schema for email profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the email field (column) to analyze. Must exist in the input DataFrame.",
                    },
                    "top_n": {
                        "type": "integer",
                        "title": "Top N Domains",
                        "description": "Number of top email domains to include in the results and visualizations. Must be at least 1.",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    },
                    "min_frequency": {
                        "type": "integer",
                        "title": "Minimum Domain Frequency",
                        "description": "Minimum frequency for a domain to be included in the domain dictionary. Must be at least 1.",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 1,
                    },
                    "profile_type": {
                        "type": "string",
                        "title": "Profile Type",
                        "description": "Type of profiling for organizing artifacts. Default is 'email'.",
                        "default": "email",
                    },
                    "analyze_privacy_risk": {
                        "type": "boolean",
                        "title": "Analyze Privacy Risk",
                        "description": "Whether to analyze potential privacy risks from email patterns and uniqueness.",
                        "default": True,
                    },
                },
                "required": ["field_name", "top_n", "min_frequency"],
            },
        ],
    }
