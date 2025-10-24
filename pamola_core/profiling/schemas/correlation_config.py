"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating correlation profiling operations in PAMOLA.CORE.
Supports parameters for field pairs, correlation methods, and profiling options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of correlation config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class CorrelationOperationConfig(OperationConfig):
    """Configuration for CorrelationOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common BaseOperation fields
            {
                "type": "object",
                "properties": {
                    "field1": {"type": "string"},
                    "field2": {"type": "string"},
                    "method": {
                        "type": ["string", "null"],
                        "enum": [
                            "pearson",
                            "spearman",
                            "cramers_v",
                            "correlation_ratio",
                            "point_biserial",
                            None,
                        ],
                        "default": None,
                    },
                    "null_handling": {
                        "type": ["string", "null"],
                        "enum": ["drop", "fill_zero", "fill_mean", None],
                        "default": "drop",
                    },
                    "mvf_parser": {"type": ["string", "null"], "default": None},
                },
                "required": ["field1", "field2"],
            },
        ],
    }

class CorrelationMatrixOperationConfig(OperationConfig):
    """Configuration for CorrelationMatrixOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "default": [],
                    },
                    "methods": {
                        "type": ["object", "null"],
                        "additionalProperties": {
                            "type": "string",
                            "enum": [
                                "pearson",
                                "spearman",
                                "cramers_v",
                                "correlation_ratio",
                                "point_biserial",
                            ],
                        },
                        "default": None,
                    },
                    "min_threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.3,
                    },
                    "null_handling": {
                        "type": "string",
                        "enum": ["drop", "fill_zero", "fill_mean"],
                        "default": "drop",
                    },
                },
                "required": ["fields"],
            },
        ],
    }