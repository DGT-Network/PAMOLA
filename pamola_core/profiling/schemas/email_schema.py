"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Email Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating email profiling operations in PAMOLA.CORE.
Supports parameters for field names, top N values, frequency, and profile types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of email config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.common.enum.form_groups import GroupName


class EmailOperationConfig(OperationConfig):
    """Configuration for EmailOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Email Operation Configuration",
        "description": "Configuration schema for email profiling operations. Defines parameters for analyzing an email field, including domain statistics, frequency thresholds, and privacy risk assessment.",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common operation fields
            {
                "type": "object",
                "properties": {
                    # --- Email-specific parameters ---
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
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "min_frequency": {
                        "type": "integer",
                        "title": "Minimum Domain Frequency",
                        "description": "Minimum frequency for a domain to be included in the domain dictionary. Must be at least 1.",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 1,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "profile_type": {
                        "type": "string",
                        "title": "Profile Type",
                        "description": "Type of profiling for organizing artifacts. Default is 'email'.",
                        "enum": ["email"],
                        "default": "email",
                    },
                    "analyze_privacy_risk": {
                        "type": "boolean",
                        "title": "Analyze Privacy Risk",
                        "description": "Whether to analyze potential privacy risks from email patterns and uniqueness.",
                        "default": True,
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                },
                "required": ["field_name", "top_n", "min_frequency"],
            },
        ],
    }
