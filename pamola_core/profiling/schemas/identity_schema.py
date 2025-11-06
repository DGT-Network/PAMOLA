"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Identity Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating identity analysis operations in PAMOLA.CORE.
Supports parameters for uid fields, reference fields, and id fields.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of identity config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.common.enum.form_groups import GroupName


class IdentityAnalysisOperationConfig(OperationConfig):
    """Configuration for IdentityAnalysisOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Identity Analysis Operation Configuration",
        "description": "Configuration schema for identity analysis operations. Defines parameters for analyzing identifier fields, reference fields, and cross-matching logic.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields from BaseOperationConfig
            {
                "type": "object",
                "properties": {
                    # --- Operation-specific fields ---
                    "reference_fields": {
                        "type": "array",
                        "title": "Reference Fields",
                        "description": "List of fields used to identify an entity (e.g., ['first_name', 'last_name']). Must contain at least one field.",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "x-custom-function": ["update_condition_field"]
                    },
                    "uid_field": {
                        "type": "string",
                        "title": "UID Field",
                        "description": "Primary identifier field to analyze (e.g., 'UID'). Must exist in the input DataFrame.",
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "x-custom-function": ["update_condition_field"]
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Entity ID Field",
                        "description": "Optional entity-level identifier field for grouping or additional analysis.",
                        "default": None,
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "x-custom-function": ["update_condition_field"]
                    },
                    "top_n": {
                        "type": "integer",
                        "title": "Top N Entries",
                        "description": "Number of top entries to include in the results (e.g., for reporting most frequent identifiers). Must be at least 1.",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 15,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "check_cross_matches": {
                        "type": "boolean",
                        "title": "Check Cross-Matches",
                        "description": "Whether to check for cross-matching between identifiers and reference fields.",
                        "default": True,
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "min_similarity": {
                        "type": "number",
                        "title": "Minimum Similarity Threshold",
                        "description": "Minimum similarity threshold (between 0 and 1) for fuzzy matching when checking cross-matches.",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "x-component": "FloatPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "fuzzy_matching": {
                        "type": "boolean",
                        "title": "Fuzzy Matching",
                        "description": "Whether to use fuzzy matching for cross-matching identifiers and reference fields.",
                        "default": False,
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                },
                "required": ["uid_field", "reference_fields", "top_n", "min_similarity"],
            },
        ],
    }
