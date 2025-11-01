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
        "title": "Numeric Operation Configuration",
        "description": "Configuration schema for numeric profiling operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- Numeric-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the numeric field (column) to analyze. This should be a column in the DataFrame containing numeric values."
                    },
                    "bins": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                        "title": "Histogram Bins",
                        "description": "Number of bins to use for histogram analysis of the numeric field. Controls the granularity of the distribution visualization."
                    },
                    "detect_outliers": {
                        "type": "boolean",
                        "default": True,
                        "title": "Detect Outliers",
                        "description": "Whether to perform outlier detection on the numeric field. If true, the analysis will identify and report outlier values."
                    },
                    "test_normality": {
                        "type": "boolean",
                        "default": True,
                        "title": "Test Normality",
                        "description": "Whether to perform normality testing on the numeric field. If true, the analysis will include statistical tests to assess if the data is normally distributed."
                    },
                    "near_zero_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "default": 1e-10,
                        "title": "Near Zero Threshold",
                        "description": "Threshold below which values are considered 'near zero'. Used to identify and report values that are effectively zero for the purposes of analysis."
                    },
                    "profile_type": {
                        "type": "string",
                        "default": "numeric",
                        "title": "Profile Type",
                        "description": "Type of profiling operation. For numeric analysis, this should be set to 'numeric'."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
