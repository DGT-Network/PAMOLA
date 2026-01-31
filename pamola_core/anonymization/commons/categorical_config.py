"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------

Module:        categorical_generalization_config.py
Package:       pamola_core.anonymization.generalization
Version:       1.1.0
Status:        Stable
Author:        PAMOLA Core Team
Created:       2025-01-23
Updated:       2025-10-14
License:       BSD 3-Clause

Description
-----------
Configuration management for categorical data generalization operations in PAMOLA.CORE.
Provides JSON Schema-based validation, strategy-specific parameter handling, and
comprehensive configuration management for categorical anonymization techniques.

This module serves as the central configuration layer for categorical generalization,
ensuring type safety, parameter validation, and compatibility with PAMOLA's operation
configuration framework.

Features
--------
- **Strategy Support**: Multiple generalization strategies (hierarchy, frequency-based, merge)
- **NULL Handling**: Configurable strategies for NULL value processing
- **Privacy Controls**: Built-in privacy threshold validation and k-anonymity checks
- **Text Processing**: Configurable normalization and fuzzy matching
- **Performance**: Optimized for large datasets with Dask support
- **Extensibility**: Easy addition of new strategies and parameters

Supported Strategies
--------------------
1. **HIERARCHY**: Dictionary-based hierarchical generalization
2. **MERGE_LOW_FREQ**: Merge low-frequency categories into groups
3. **FREQUENCY_BASED**: Frequency-threshold based category preservation

# Get strategy-specific parameters
params = get_strategy_params(config.params)
```

Configuration Schema
--------------------
Required fields:
    - field_name: Target field name
    - strategy: Generalization strategy (hierarchy|merge_low_freq|frequency_based)

Optional fields vary by strategy - see schema definition for details.

Dependencies
------------
- enum: Strategy and option enumerations
- typing: Type annotations and hints

See Also
--------
- Generalization operations documentation

Notes
-----
- All numeric thresholds are validated at configuration time
- Enum values are pre-computed for performance optimization
- Schema validation occurs during initialization
- Configuration is serializable for export/import

Changelog
---------
1.1.0 (2025-10-14):
    - Added privacy check parameters
    - Enhanced text normalization options
    - Improved schema validation

1.0.0 (2025-01-23):
    - Initial stable release
"""

from enum import Enum
from typing import Any, Dict


# =============================================================================
# CONSTANTS - Data Processing
# =============================================================================

DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CACHE_SIZE = 10000
DEFAULT_DASK_CHUNK_SIZE = 5000
DEFAULT_MAX_ROWS_IN_MEMORY = 1000000

# =============================================================================
# CONSTANTS - Hierarchy & Categories
# =============================================================================

MAX_HIERARCHY_LEVELS = 5
MAX_CATEGORIES = 1000000
DEFAULT_MIN_GROUP_SIZE = 10
DEFAULT_MAX_CATEGORIES_FOR_DIVERSITY = 1000
DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS = 20
DEFAULT_MAX_HIERARCHY_CHILDREN_DISPLAY = 10

# =============================================================================
# CONSTANTS - Thresholds
# =============================================================================

DEFAULT_FREQ_THRESHOLD = 0.01
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_MAX_SUPPRESSION_RATE = 0.2
DEFAULT_MIN_COVERAGE = 0.95
FLOAT_COMPARISON_EPSILON = 1e-10

# =============================================================================
# CONSTANTS - Sampling & Analysis
# =============================================================================

DEFAULT_SAMPLE_SIZE = 10000
OPERATION_NAME = "categorical_generalization"

# =============================================================================
# CONSTANTS - Supported Formats
# =============================================================================

SUPPORTED_DICT_FORMATS = ["auto", "json", "csv"]


# =============================================================================
# ENUMERATIONS
# =============================================================================


class GeneralizationStrategy(str, Enum):
    """
    Categorical generalization strategies.

    Attributes
    ----------
    HIERARCHY : str
        Dictionary-based hierarchical generalization
    MERGE_LOW_FREQ : str
        Merge low-frequency categories
    FREQUENCY_BASED : str
        Frequency threshold-based preservation
    """

    HIERARCHY = "hierarchy"
    MERGE_LOW_FREQ = "merge_low_freq"
    FREQUENCY_BASED = "frequency_based"


class NullStrategy(str, Enum):
    """
    NULL value handling strategies.

    Attributes
    ----------
    PRESERVE : str
        Keep NULL values unchanged
    EXCLUDE : str
        Remove rows with NULL values
    ANONYMIZE : str
        Replace NULL with unknown_value
    ERROR : str
        Raise error when NULL encountered
    """

    PRESERVE = "PRESERVE"
    EXCLUDE = "EXCLUDE"
    ANONYMIZE = "ANONYMIZE"
    ERROR = "ERROR"


class OperationMode(str, Enum):
    """
    Field update operation modes.

    Attributes
    ----------
    REPLACE : str
        Replace original field with generalized values
    ENRICH : str
        Add new field with generalized values
    """

    REPLACE = "REPLACE"
    ENRICH = "ENRICH"


class GroupRareAs(str, Enum):
    """
    Rare category grouping strategies.

    Attributes
    ----------
    OTHER : str
        Group all rare values as single "OTHER" category
    CATEGORY_N : str
        Create numbered categories (CATEGORY_1, CATEGORY_2, ...)
    RARE_N : str
        Create numbered rare groups (RARE_1, RARE_2, ...)
    """

    OTHER = "OTHER"
    CATEGORY_N = "CATEGORY_N"
    RARE_N = "RARE_N"


class TextNormalization(str, Enum):
    """
    Text normalization levels.

    Attributes
    ----------
    NONE : str
        No normalization applied
    BASIC : str
        Basic normalization (whitespace, case)
    ADVANCED : str
        Advanced normalization (punctuation, special chars)
    AGGRESSIVE : str
        Aggressive normalization (stemming, lemmatization)
    """

    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


# =============================================================================
# PRE-COMPUTED ENUM VALUES (Performance Optimization)
# =============================================================================

STRATEGY_VALUES = [s.value for s in GeneralizationStrategy]
NULL_STRATEGY_VALUES = [n.value for n in NullStrategy]
MODE_VALUES = [m.value for m in OperationMode]
GROUP_RARE_VALUES = [g.value for g in GroupRareAs]
TEXT_NORM_VALUES = [t.value for t in TextNormalization]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_strategy_params(config_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract strategy-specific parameters from configuration.

    Parameters
    ----------
    config_params : Dict[str, Any]
        Configuration parameters as a dictionary.

    Returns
    -------
    Dict[str, Any]
        Strategy-specific parameters including base parameters.
    """
    # Common parameters for all strategies
    base_params = {
        "allow_unknown": config_params.get("allow_unknown"),
        "unknown_value": config_params.get("unknown_value"),
        "text_normalization": config_params.get("text_normalization"),
        "case_sensitive": config_params.get("case_sensitive"),
        "null_strategy": config_params.get("null_strategy"),
    }

    strategy = config_params.get("strategy")

    # Hierarchy-based generalization
    if strategy == GeneralizationStrategy.HIERARCHY.value:
        return {
            **base_params,
            "external_dictionary_path": config_params.get("external_dictionary_path"),
            "dictionary_format": config_params.get("dictionary_format"),
            "hierarchy_level": config_params.get("hierarchy_level"),
            "fuzzy_matching": config_params.get("fuzzy_matching"),
            "similarity_threshold": config_params.get("similarity_threshold"),
        }

    # Low-frequency merging
    elif strategy == GeneralizationStrategy.MERGE_LOW_FREQ.value:
        return {
            **base_params,
            "min_group_size": config_params.get("min_group_size"),
            "freq_threshold": config_params.get("freq_threshold"),
            "group_rare_as": config_params.get("group_rare_as"),
            "rare_value_template": config_params.get("rare_value_template"),
            "max_categories": config_params.get("max_categories"),
        }

    # Frequency-based preservation
    elif strategy == GeneralizationStrategy.FREQUENCY_BASED.value:
        return {
            **base_params,
            "max_categories": config_params.get("max_categories"),
            "min_group_size": config_params.get("min_group_size"),
            "group_rare_as": config_params.get("group_rare_as"),
            "rare_value_template": config_params.get("rare_value_template"),
        }

    # Default: return base parameters only
    return base_params


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Helper functions
    "get_strategy_params",
    # Enumerations
    "GeneralizationStrategy",
    "NullStrategy",
    "OperationMode",
    "GroupRareAs",
    "TextNormalization",
    # Pre-computed enum values
    "STRATEGY_VALUES",
    "NULL_STRATEGY_VALUES",
    "MODE_VALUES",
    "GROUP_RARE_VALUES",
    "TEXT_NORM_VALUES",
    # Supported formats
    "SUPPORTED_DICT_FORMATS",
]
