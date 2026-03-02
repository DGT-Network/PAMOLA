"""
PAMOLA System Configuration - L-Diversity Module
-----------------------------------------------
This module centralizes configuration settings for L-Diversity
anonymization model within the PAMOLA ecosystem.

Settings are dynamically loaded and can be overridden via environment
variables or user-defined configuration files.

(C) 2024 Realm Inveo Inc. & DGT Network Inc.
"""

import os
from typing import Dict, Any
from pamola_core.errors.exceptions import InvalidParameterError

# L-Diversity Configuration with Environment Variable Support
L_DIVERSITY_DEFAULTS: Dict[str, Any] = {
    # Pamola Core l-diversity parameters
    "l": int(os.getenv("PAMOLA_L_DIVERSITY_L", 3)),  # Minimum diversity level
    "diversity_type": os.getenv("PAMOLA_L_DIVERSITY_TYPE", "distinct"),
    "c_value": float(os.getenv("PAMOLA_L_DIVERSITY_C_VALUE", 1.0)),
    # K-anonymity baseline
    "k": int(os.getenv("PAMOLA_L_DIVERSITY_K", 2)),
    # Anonymization strategies
    "use_dask": os.getenv("PAMOLA_L_DIVERSITY_USE_DASK", "False").lower() == "true",
    "mask_value": os.getenv("PAMOLA_L_DIVERSITY_MASK_VALUE", "MASKED"),
    "suppression": os.getenv("PAMOLA_L_DIVERSITY_SUPPRESSION", "True").lower()
    == "true",
    # Performance and scalability
    "npartitions": int(os.getenv("PAMOLA_L_DIVERSITY_NPARTITIONS", 4)),
    "optimize_memory": os.getenv("PAMOLA_L_DIVERSITY_OPTIMIZE_MEMORY", "True").lower()
    == "true",
    # Logging
    "log_level": os.getenv("PAMOLA_L_DIVERSITY_LOG_LEVEL", "INFO"),
    # Visualization settings
    "visualization": {
        "hist_bins": int(os.getenv("PAMOLA_L_DIVERSITY_HIST_BINS", 20)),
        "save_format": os.getenv("PAMOLA_L_DIVERSITY_SAVE_FORMAT", "png"),
    },
    # Compliance settings
    "compliance": {
        "risk_threshold": float(os.getenv("PAMOLA_L_DIVERSITY_RISK_THRESHOLD", 0.5)),
        "supported_regulations": ["GDPR", "HIPAA", "CCPA"],
    },
}


def validate_l_diversity_config(config: Dict[str, Any]) -> bool:
    """
    Validate l-diversity configuration parameters

    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary to validate

    Returns:
    --------
    bool
        True if configuration is valid, False otherwise
    """
    try:
        # Validate l-value
        l_value = config.get("l", 3)
        if not isinstance(l_value, int) or l_value < 1:
            raise InvalidParameterError(
                param_name="l",
                param_value=l_value,
                reason=f"l must be a positive integer, got {l_value}",
            )

        # Validate diversity type
        diversity_type = config.get("diversity_type", "distinct")
        valid_types = ["distinct", "entropy", "recursive"]
        if diversity_type not in valid_types:
            raise InvalidParameterError(
                param_name="diversity_type",
                param_value=diversity_type,
                reason=f"l must be a positive integer, got {valid_types}",
            )

        # Validate c-value for recursive diversity
        if diversity_type == "recursive":
            c_value = config.get("c_value", 1.0)
            if not isinstance(c_value, (int, float)) or c_value <= 0:
                raise InvalidParameterError(
                    param_name="c_value",
                    param_value=c_value,
                    reason=f"c_value must be a positive number, got {c_value}",
                )

        # Validate k-value
        k_value = config.get("k", 2)
        if not isinstance(k_value, int) or k_value < 1:
            raise InvalidParameterError(
                param_name="k_value",
                param_value=k_value,
                reason=f"k must be a positive integer, got {k_value}",
            )

        return True

    except Exception as e:
        print(f"L-Diversity configuration validation error: {e}")
        return False


def get_l_diversity_config(config_override: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Retrieve l-diversity configuration with optional overrides

    Parameters:
    -----------
    config_override : Dict[str, Any], optional
        Configuration parameters to override defaults

    Returns:
    --------
    Dict[str, Any]
        Validated and merged configuration
    """
    # Create a deep copy of the default configuration
    from copy import deepcopy

    config = deepcopy(L_DIVERSITY_DEFAULTS)

    # Apply configuration overrides if provided
    if config_override:
        # Deep update to preserve nested structure
        def deep_update(original, update):
            for key, value in update.items():
                if isinstance(value, dict):
                    original[key] = deep_update(original.get(key, {}), value)
                else:
                    original[key] = value
            return original

        config = deep_update(config, config_override)

    # Validate the final configuration
    if not validate_l_diversity_config(config):
        print(
            "Warning: L-Diversity configuration validation failed. Using default settings."
        )
        return L_DIVERSITY_DEFAULTS

    return config
