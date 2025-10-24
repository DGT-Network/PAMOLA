"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Temporal Noise Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating uniform temporal noise operations in PAMOLA.CORE.
Supports parameters for field names, temporal noise ranges, and noise types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of uniform temporal noise config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class UniformTemporalNoiseConfig(OperationConfig):
    """Configuration for UniformTemporalNoiseOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields from base config
            {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    # Temporal noise parameters
                    "noise_range_days": {"type": ["number", "null"]},
                    "noise_range_hours": {"type": ["number", "null"]},
                    "noise_range_minutes": {"type": ["number", "null"]},
                    "noise_range_seconds": {"type": ["number", "null"]},
                    "noise_range": {
                        "type": ["object", "null"],
                        "properties": {
                            "noise_range_days": {"type": ["number", "null"]},
                            "noise_range_hours": {"type": ["number", "null"]},
                            "noise_range_minutes": {"type": ["number", "null"]},
                            "noise_range_seconds": {"type": ["number", "null"]},
                        },
                    },
                    # Direction control
                    "direction": {
                        "type": "string",
                        "enum": ["both", "forward", "backward"],
                        "default": "both",
                    },
                    # Boundary constraints
                    "min_datetime": {"type": ["string", "null"]},
                    "max_datetime": {"type": ["string", "null"]},
                    # Special date handling
                    "preserve_special_dates": {"type": "boolean", "default": False},
                    "special_dates": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "preserve_weekends": {"type": "boolean", "default": False},
                    "preserve_time_of_day": {"type": "boolean", "default": False},
                    # Granularity
                    "output_granularity": {
                        "type": ["string", "null"],
                        "enum": ["day", "hour", "minute", "second", None],
                    },
                    # Reproducibility
                    "random_seed": {"type": ["integer", "null"]},
                    "use_secure_random": {"type": "boolean", "default": True},
                    # Multi-field conditions
                    "multi_conditions": {"type": ["array", "null"]},
                    "condition_logic": {"type": "string"},
                    # Conditional processing parameters
                    "condition_field": {"type": ["string", "null"]},
                    "condition_values": {"type": ["array", "null"]},
                    "condition_operator": {"type": "string"},
                    # K-anonymity integration
                    "ka_risk_field": {"type": ["string", "null"]},
                    "risk_threshold": {"type": "number"},
                    "vulnerable_record_strategy": {"type": "string"},
                    # Output field name configuration
                    "output_field_name": {"type": ["string", "null"]},
                },
                "required": ["field_name"],
            },
        ],
    }