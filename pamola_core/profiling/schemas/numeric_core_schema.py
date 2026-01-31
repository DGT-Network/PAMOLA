"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of numeric profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines numeric analysis parameters, binning, outlier detection, and normality testing
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name specification for numeric analysis
- Histogram binning controls
- Near-zero threshold configuration
- Outlier detection and normality testing enablement
- Profile type specification

Changelog:
1.0.0 - 2025-01-15 - Initial creation of numeric profiler core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class NumericOperationConfig(OperationConfig):
    """
    Core configuration schema for NumericOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Numeric Operation Core Configuration",
        "description": "Core schema for numeric profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the numeric field (column) to analyze. This should be a column in the DataFrame containing numeric values.",
                    },
                    "bins": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "title": "Histogram Bins",
                        "description": "Number of bins to use for histogram analysis of the numeric field. Controls the granularity of the distribution visualization.",
                    },
                    "near_zero_threshold": {
                        "type": "number",
                        "minimum": 1e-10,
                        "maximum": 1.0,
                        "default": 1e-10,
                        "title": "Near Zero Threshold",
                        "description": "Threshold below which values are considered 'near zero'. Used to identify and report values that are effectively zero for the purposes of analysis.",
                    },
                    "detect_outliers": {
                        "type": "boolean",
                        "default": True,
                        "title": "Detect Outliers",
                        "description": "Whether to perform outlier detection on the numeric field. If true, the analysis will identify and report outlier values.",
                    },
                    "test_normality": {
                        "type": "boolean",
                        "default": True,
                        "title": "Test Normality",
                        "description": "Whether to perform normality testing on the numeric field. If true, the analysis will include statistical tests to assess if the data is normally distributed.",
                    },
                    "profile_type": {
                        "type": "string",
                        "default": "numeric",
                        "title": "Profile Type",
                        "description": "Type of profiling operation. For numeric analysis, this should be set to 'numeric'.",
                    },
                },
                "required": ["field_name", "bins", "near_zero_threshold"],
            },
        ],
    }