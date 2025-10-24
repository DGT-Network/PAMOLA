"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating numeric profiling operations in PAMOLA.CORE.
Supports parameters for field names, binning, outlier/normality detection, and thresholds.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of numeric config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class NumericOperationConfig(OperationConfig):
    """Configuration for NumericOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- Numeric-specific fields ---
                    "field_name": {"type": "string"},
                    "bins": {"type": "integer", "minimum": 1, "default": 10},
                    "detect_outliers": {"type": "boolean", "default": True},
                    "test_normality": {"type": "boolean", "default": True},
                    "near_zero_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "default": 1e-10,
                    },
                    "profile_type": {"type": "string", "default": "numeric"},
                },
                "required": ["field_name"],
            },
        ],
    }
