"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Identity Analysis Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of identity analysis configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines identity analysis parameters, reference fields, and cross-matching logic
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- UID and reference field configuration
- Cross-matching and fuzzy matching controls
- Similarity threshold validation
- Top N entries specification

Changelog:
1.0.0 - 2025-01-15 - Initial creation of identity analysis core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class IdentityAnalysisOperationConfig(OperationConfig):
    """
    Core configuration schema for IdentityAnalysisOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Identity Analysis Operation Core Configuration",
        "description": "Core schema for identity analysis profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "reference_fields": {
                        "type": "array",
                        "title": "Reference Fields",
                        "description": "List of fields used to identify an entity (e.g., ['first_name', 'last_name']). Must contain at least one field.",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "uid_field": {
                        "type": "string",
                        "title": "UID Field",
                        "description": "Primary identifier field to analyze (e.g., 'UID'). Must exist in the input DataFrame.",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Entity ID Field",
                        "description": "Optional entity-level identifier field for grouping or additional analysis.",
                        "default": None,
                    },
                    "top_n": {
                        "type": "integer",
                        "title": "Top N Entries",
                        "description": "Number of top entries to include in the results (e.g., for reporting most frequent identifiers). Must be at least 1.",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 15,
                    },
                    "min_similarity": {
                        "type": "number",
                        "title": "Minimum Similarity Threshold",
                        "description": "Minimum similarity threshold (between 0 and 1) for fuzzy matching when checking cross-matches.",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                    },
                    "check_cross_matches": {
                        "type": "boolean",
                        "title": "Check Cross-Matches",
                        "description": "Whether to check for cross-matching between identifiers and reference fields.",
                        "default": True,
                    },
                    "fuzzy_matching": {
                        "type": "boolean",
                        "title": "Fuzzy Matching",
                        "description": "Whether to use fuzzy matching for cross-matching identifiers and reference fields.",
                        "default": False,
                    },
                },
                "required": [
                    "uid_field",
                    "reference_fields",
                    "top_n",
                    "min_similarity",
                ],
            },
        ],
    }
