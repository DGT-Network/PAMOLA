"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Categorical Generalization Strategy Implementations
Package:       pamola_core.anonymization.generalization
Version:       1.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-23
Updated:       2025-01-23
License:       BSD 3-Clause

Description:
   This module implements the core generalization strategies for categorical
   data anonymization. It provides pure functions for applying various
   generalization techniques including hierarchical mapping, frequency-based
   grouping, and rare category merging.

Purpose:
   Separates strategy implementation logic from the main operation class,
   enabling better testability, reusability, and maintainability. Each
   strategy function is stateless and thread-safe.

Key Features:
   - Hierarchical generalization using external dictionaries
   - Frequency-based category grouping
   - Rare category merging with configurable thresholds
   - NULL and unknown value handling
   - Conditional mapping support
   - Vectorized operations for performance
   - Thread-safe random number generation
   - Dask support for large datasets (future)

Design Principles:
   - Stateless: All functions are pure with no side effects
   - Thread-safe: No shared mutable state, local RNG instances
   - Performance: Vectorized operations and caching where possible
   - Extensibility: Easy to add new strategies
   - Testability: Each strategy can be tested independently

Dependencies:
   - pandas: For Series operations
   - numpy: For numerical operations
   - logging: For operational diagnostics
   - pamola_core.anonymization.commons: Utility functions

Usage:
   ```python
   # Apply hierarchy strategy
   result = apply_hierarchy(
       series=data['city'],
       config=strategy_params,
       context={'hierarchy': hierarchy_dict}
   )

   # Apply NULL handling
   result = apply_null_and_unknown_strategy(
       series=result,
       null_strategy=NullStrategy.PRESERVE,
       unknown_value="OTHER"
   )
   ```

Changelog:
   1.2.0 (2025-01-23):
     - Added ENRICH mode warnings to all strategies
     - Removed unused imports
     - Fixed template pattern passing to use None for defaults
     - Fixed cache size calculation reuse
     - Enhanced Enum handling for better type safety
     - Fixed coverage calculation with case sensitivity
     - Reduced duplicate grouping logic
     - Changed Dask warnings to DEBUG level
   1.1.0 (2025-01-23):
     - Fixed global random seed issue - now uses local RNG
     - Optimized hierarchy lookups with vectorized normalization
     - Added caching for fuzzy matching value lists
     - Pre-compiled regex patterns for better performance
     - Fixed prefix pattern defaults for template formatting
     - Improved coverage metric calculation timing
     - Added warnings for ENRICH mode field conflicts
   1.0.0 - Initial implementation extracted from categorical.py
"""

import logging
import re
import warnings
from enum import Enum
from typing import Any, Dict, Optional, Union
import dask.dataframe as dd
import numpy as np
import pandas as pd

# Import from commons
from pamola_core.anonymization.commons.hierarchy_dictionary import HierarchyDictionary
from pamola_core.anonymization.commons.category_mapping import (
    CategoryMappingEngine
)
from pamola_core.anonymization.commons.category_utils import (
    identify_rare_categories,
    group_rare_categories
)
from pamola_core.anonymization.commons.text_processing_utils import (
    normalize_text,
    find_closest_category
)

# Import configuration enums
from .categorical_config import (
    NullStrategy,
    GroupRareAs,
    OperationMode
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_UNKNOWN_VALUE = "OTHER"
DEFAULT_RARE_PREFIX = "RARE_"
BATCH_LOG_THRESHOLD = 100000  # Log progress for large batches
DEFAULT_PREFIX_PATTERN = r"^(OTHER|RARE|CATEGORY)_(\d+)"

# Pre-compiled regex patterns for performance
NUMERIC_SUFFIX_PATTERN = re.compile(r'(\d+)$')
PREFIX_NUMBER_PATTERN = re.compile(DEFAULT_PREFIX_PATTERN)


def _check_enrich_mode_safety(context: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Check for ENRICH mode field conflicts and issue warnings.

    Parameters:
    -----------
    context : Dict[str, Any]
        Processing context
    logger : Optional[logging.Logger]
        Logger instance
    """
    mode = context.get('mode', OperationMode.REPLACE.value)
    # Handle both string and Enum
    if isinstance(mode, Enum):
        mode_value = mode.value
    else:
        mode_value = mode

    if mode_value == OperationMode.ENRICH.value:
        output_field = context.get('output_field_name')
        if not output_field:
            warnings.warn(
                "ENRICH mode without output_field_name may overwrite source field",
                RuntimeWarning
            )


def apply_hierarchy(
        series: pd.Series,
        config: Dict[str, Any],
        context: Dict[str, Any],
        logger: Optional[logging.Logger] = None
) -> pd.Series:
    """
    Apply hierarchical generalization using external dictionary.

    This strategy maps values to higher-level categories based on a
    predefined hierarchy (e.g., city → state → country).

    Parameters:
    -----------
    series : pd.Series
        The series to generalize
    config : Dict[str, Any]
        Strategy configuration containing:
        - hierarchy_level: int - Level of generalization (1-based)
        - text_normalization: str - Normalization level
        - case_sensitive: bool - Case sensitivity
        - fuzzy_matching: bool - Enable fuzzy matching
        - similarity_threshold: float - Threshold for fuzzy matching
        - unknown_value: str - Value for unmapped entries
        - allow_unknown: bool - Whether to allow unknowns
        - random_seed: Optional[int] - For reproducibility
    context : Dict[str, Any]
        Processing context containing:
        - hierarchy: HierarchyDictionary - Loaded hierarchy
        - batch_id: Optional[str] - Batch identifier
        - batch_df: Optional[pd.DataFrame] - Full batch for conditional rules
        - mode: Optional[str/Enum] - Operation mode (REPLACE/ENRICH)
        - output_field_name: Optional[str] - For ENRICH mode
    logger : Optional[logging.Logger]
        Logger instance for diagnostics

    Returns:
    --------
    pd.Series
        Generalized series

    Raises:
    -------
    ValueError
        If hierarchy not provided or invalid configuration
    """
    if logger:
        logger.debug(f"[{context.get('batch_id', 'unknown')}] Applying hierarchy strategy")

    # Check for ENRICH mode field conflicts
    _check_enrich_mode_safety(context, logger)

    # Validate inputs
    hierarchy = context.get('hierarchy')
    if not hierarchy or not isinstance(hierarchy, HierarchyDictionary):
        raise ValueError("Hierarchy dictionary not provided in context")

    # Extract configuration
    hierarchy_level = config.get('hierarchy_level', 1)
    text_normalization = config.get('text_normalization', 'basic')
    case_sensitive = config.get('case_sensitive', False)
    fuzzy_matching = config.get('fuzzy_matching', False)
    similarity_threshold = config.get('similarity_threshold', 0.85)
    unknown_value = config.get('unknown_value', DEFAULT_UNKNOWN_VALUE)
    allow_unknown = config.get('allow_unknown', True)
    random_seed = config.get('random_seed')

    # Create local RNG for thread safety
    rng = np.random.default_rng(random_seed)

    # Vectorized preprocessing for performance
    unique_series = pd.Series(series.dropna().unique())
    unique_count = len(unique_series)

    # Initialize mapping engine with cached size
    engine = CategoryMappingEngine(
        unknown_value=unknown_value,
        cache_size=min(10000, unique_count * 2)
    )

    # Track statistics
    fuzzy_matches = 0
    unknown_values = set()

    # Vectorized normalization
    if not case_sensitive or text_normalization != 'none':
        normalized_series = unique_series.astype(str)
        if text_normalization != 'none':
            # Apply normalization (vectorized where possible)
            normalized_series = normalized_series.apply(
                lambda x: normalize_text(x, text_normalization, case_sensitive)
            )
        if not case_sensitive:
            normalized_series = normalized_series.str.lower()
    else:
        normalized_series = unique_series.astype(str)

    # Calculate coverage on normalized data for accurate metric
    if case_sensitive:
        coverage_values = unique_series.tolist()
    else:
        coverage_values = normalized_series.tolist()
    original_coverage = hierarchy.get_coverage(coverage_values)

    # Cache fuzzy matching values if needed
    all_hierarchy_values = None
    if fuzzy_matching:
        all_hierarchy_values = list(hierarchy.get_all_values_at_level(0))
        if logger:
            logger.debug(f"Cached {len(all_hierarchy_values)} hierarchy values for fuzzy matching")

    # Log progress for large datasets
    if unique_count > BATCH_LOG_THRESHOLD and logger:
        logger.info(f"Processing {unique_count} unique values for hierarchy mapping")

    # Build mappings for all unique values
    for i, (orig_value, norm_value) in enumerate(zip(unique_series, normalized_series)):
        str_value = str(orig_value)

        # Try direct hierarchy lookup
        category = hierarchy.get_hierarchy(
            norm_value,
            hierarchy_level,
            normalize=False  # Already normalized
        )

        if category:
            engine.add_mapping(str_value, category)
        elif fuzzy_matching and all_hierarchy_values:
            # Try fuzzy matching with cached values
            closest = find_closest_category(
                norm_value,
                all_hierarchy_values,
                similarity_threshold
            )

            if closest:
                category = hierarchy.get_hierarchy(closest, hierarchy_level)
                if category:
                    engine.add_mapping(str_value, category)
                    fuzzy_matches += 1
                    if logger:
                        logger.debug(f"Fuzzy matched '{str_value}' to '{closest}' → '{category}'")
                else:
                    unknown_values.add(str_value)
            else:
                unknown_values.add(str_value)
        else:
            # Handle unknown value
            if not allow_unknown:
                raise ValueError(f"Unknown value not allowed: '{str_value}'")
            unknown_values.add(str_value)

        # Log progress for very large sets
        if (i + 1) % 10000 == 0 and logger:
            logger.debug(f"Processed {i + 1}/{unique_count} unique values")

    # Apply mapping to series
    result = engine.apply_to_series(series)

    # Update context with statistics
    context['fuzzy_matches'] = context.get('fuzzy_matches', 0) + fuzzy_matches
    context['unknown_values'] = context.get('unknown_values', set()).union(unknown_values)
    context['category_mapping'] = engine.get_mapping_dict()
    context['hierarchy_info'] = {
        'level': hierarchy_level,
        'coverage': original_coverage
    }

    # Log summary
    if logger:
        stats = engine.get_statistics()
        logger.debug(
            f"Hierarchy mapping complete: "
            f"{unique_count} unique → {result.nunique()} categories, "
            f"fuzzy matches: {fuzzy_matches}, unknowns: {len(unknown_values)}"
        )

    return result


def apply_merge_low_freq(
        series: pd.Series,
        config: Dict[str, Any],
        context: Dict[str, Any],
        logger: Optional[logging.Logger] = None
) -> pd.Series:
    """
    Apply merging of low frequency categories.

    This strategy identifies and groups rare categories based on
    frequency thresholds to improve k-anonymity.

    Parameters:
    -----------
    series : pd.Series
        The series to process
    config : Dict[str, Any]
        Strategy configuration containing:
        - min_group_size: int - Minimum group size
        - freq_threshold: float - Frequency threshold
        - group_rare_as: str - Grouping strategy
        - rare_value_template: str - Template for rare groups
        - max_categories: int - Maximum categories to preserve
        - text_normalization: str - Text normalization level
        - random_seed: Optional[int] - For reproducibility
    context : Dict[str, Any]
        Processing context
    logger : Optional[logging.Logger]
        Logger instance

    Returns:
    --------
    pd.Series
        Series with merged categories
    """
    if logger:
        logger.debug(f"[{context.get('batch_id', 'unknown')}] Applying merge low frequency strategy")

    # Check for ENRICH mode field conflicts
    _check_enrich_mode_safety(context, logger)

    # Extract configuration
    min_group_size = config.get('min_group_size', 10)
    freq_threshold = config.get('freq_threshold', 0.01)
    group_rare_as = config.get('group_rare_as', GroupRareAs.OTHER.value)
    rare_value_template = config.get('rare_value_template', 'OTHER_{n}')
    max_categories = config.get('max_categories', 1000)
    random_seed = config.get('random_seed')

    # Create local RNG for thread safety
    rng = np.random.default_rng(random_seed)

    # Get value counts (cache if available)
    value_counts = context.get('value_counts')
    if value_counts is None:
        value_counts = series.value_counts()
        context['value_counts'] = value_counts

    # Identify rare categories
    rare_categories, rare_info = identify_rare_categories(
        series,
        count_threshold=min_group_size,
        percent_threshold=freq_threshold,
        combined_criteria=True,
        value_counts=value_counts
    )

    # Determine grouping strategy
    if group_rare_as == GroupRareAs.CATEGORY_N.value:
        grouping_strategy = "numbered"
        group_prefix = "CATEGORY_"
    elif group_rare_as == GroupRareAs.RARE_N.value:
        grouping_strategy = "numbered"
        group_prefix = "RARE_"
    else:  # OTHER
        grouping_strategy = "single_other"
        group_prefix = DEFAULT_RARE_PREFIX

    # Apply grouping
    grouped_series, grouping_info = group_rare_categories(
        series,
        grouping_strategy=grouping_strategy,
        threshold=min_group_size,
        max_groups=100,  # Reasonable limit
        group_prefix=group_prefix,
        preserve_top_n=max_categories,
        other_label=config.get('unknown_value', DEFAULT_UNKNOWN_VALUE),
        value_counts=value_counts
    )

    # Update context
    context['rare_categories'] = rare_categories
    context['rare_info'] = rare_info
    context['category_mapping'] = grouping_info.get('group_mapping', {})
    context['grouping_info'] = grouping_info

    # Handle template formatting if needed
    if group_rare_as in [GroupRareAs.CATEGORY_N.value, GroupRareAs.RARE_N.value]:
        # Apply template formatting with None to use default pattern
        grouped_series = _apply_rare_value_template(
            grouped_series,
            rare_value_template,
            None  # Use default pattern
        )

    # Log summary
    if logger:
        logger.debug(
            f"Merge low frequency complete: "
            f"{len(rare_categories)} rare categories → "
            f"{grouping_info['groups_created']} groups, "
            f"reduction: {grouping_info['reduction_ratio']:.1%}"
        )

    return grouped_series


def apply_frequency_based(
        series: pd.Series,
        config: Dict[str, Any],
        context: Dict[str, Any],
        logger: Optional[logging.Logger] = None
) -> pd.Series:
    """
    Apply frequency-based generalization.

    This strategy preserves the top K most frequent categories and
    groups the rest based on the configured strategy.

    Parameters:
    -----------
    series : pd.Series
        The series to process
    config : Dict[str, Any]
        Strategy configuration containing:
        - max_categories: int - Maximum categories to preserve
        - min_group_size: int - Minimum size for groups
        - group_rare_as: str - How to group rare categories
        - rare_value_template: str - Template for rare groups
        - unknown_value: str - Default unknown value
        - random_seed: Optional[int] - For reproducibility
    context : Dict[str, Any]
        Processing context
    logger : Optional[logging.Logger]
        Logger instance

    Returns:
    --------
    pd.Series
        Series with frequency-based generalization
    """
    if logger:
        logger.debug(f"[{context.get('batch_id', 'unknown')}] Applying frequency-based strategy")

    # Check for ENRICH mode field conflicts
    _check_enrich_mode_safety(context, logger)

    # Extract configuration
    max_categories = config.get('max_categories', 100)
    min_group_size = config.get('min_group_size', 10)
    group_rare_as = config.get('group_rare_as', GroupRareAs.OTHER.value)
    rare_value_template = config.get('rare_value_template', 'OTHER_{n}')
    unknown_value = config.get('unknown_value', DEFAULT_UNKNOWN_VALUE)
    random_seed = config.get('random_seed')

    # Create local RNG for thread safety
    rng = np.random.default_rng(random_seed)

    # Get value counts
    value_counts = context.get('value_counts')
    if value_counts is None:
        value_counts = series.value_counts()
        context['value_counts'] = value_counts

    # Get top categories
    top_categories = set(value_counts.head(max_categories).index)

    # Create mapping engine
    engine = CategoryMappingEngine(
        unknown_value=unknown_value
    )

    # Add mappings for top categories (map to themselves)
    for category in top_categories:
        engine.add_mapping(str(category), str(category))

    # Handle rare categories
    rare_mask = ~series.isin(top_categories)
    rare_values = series[rare_mask].dropna().unique()

    if len(rare_values) > 0:
        if group_rare_as == GroupRareAs.OTHER.value:
            # Map all rare to single value
            for value in rare_values:
                engine.add_mapping(str(value), unknown_value)

        elif group_rare_as in [GroupRareAs.CATEGORY_N.value, GroupRareAs.RARE_N.value]:
            # Use group_rare_categories for consistent grouping
            rare_series = series[rare_mask]

            # Determine prefix
            if group_rare_as == GroupRareAs.CATEGORY_N.value:
                group_prefix = "CATEGORY_"
            else:
                group_prefix = "RARE_"

            # Apply grouping utility
            grouped_rare, grouping_info = group_rare_categories(
                rare_series,
                grouping_strategy="numbered",
                threshold=min_group_size,
                group_prefix=group_prefix,
                value_counts=value_counts[value_counts.index.isin(rare_values)]
            )

            # Create mapping from grouping info
            for original, grouped in grouping_info['group_mapping'].items():
                engine.add_mapping(str(original), grouped)

            # Apply template if needed
            if "{n}" in rare_value_template:
                # Get all unique grouped values and apply template
                result_series = engine.apply_to_series(series)
                result_series = _apply_rare_value_template(
                    result_series,
                    rare_value_template,
                    None  # Use default pattern
                )
                # Update context with final mapping
                context['category_mapping'] = engine.get_mapping_dict()
                context['top_categories'] = list(top_categories)
                context['rare_count'] = len(rare_values)

                # Log summary
                if logger:
                    logger.debug(
                        f"Frequency-based complete: "
                        f"preserved top {len(top_categories)} categories, "
                        f"grouped {len(rare_values)} rare values"
                    )

                return result_series

    # Apply mapping
    result = engine.apply_to_series(series)

    # Update context
    context['category_mapping'] = engine.get_mapping_dict()
    context['top_categories'] = list(top_categories)
    context['rare_count'] = len(rare_values)

    # Log summary
    if logger:
        logger.debug(
            f"Frequency-based complete: "
            f"preserved top {len(top_categories)} categories, "
            f"grouped {len(rare_values)} rare values"
        )

    return result


def apply_null_and_unknown_strategy(
        series: pd.Series,
        null_strategy: Union[str, NullStrategy],
        unknown_value: str = DEFAULT_UNKNOWN_VALUE,
        rare_value_template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
) -> pd.Series:
    """
    Apply NULL and unknown value handling strategy.

    This function handles NULL values according to the specified strategy
    and can apply template-based formatting for unknown values.

    Parameters:
    -----------
    series : pd.Series
        The series to process
    null_strategy : Union[str, NullStrategy]
        NULL handling strategy (can be string or Enum):
        - "PRESERVE"/NullStrategy.PRESERVE: Keep NULL as is
        - "EXCLUDE"/NullStrategy.EXCLUDE: Will be handled by filtering (returns as-is)
        - "ANONYMIZE"/NullStrategy.ANONYMIZE: Replace NULL with unknown_value
        - "ERROR"/NullStrategy.ERROR: Raise error if NULL found
    unknown_value : str
        Value to use for unknowns (default: "OTHER")
    rare_value_template : Optional[str]
        Template for formatting rare values (must contain {n})
    context : Optional[Dict[str, Any]]
        Optional context for tracking statistics
    logger : Optional[logging.Logger]
        Logger instance

    Returns:
    --------
    pd.Series
        Series with NULL/unknown handling applied

    Raises:
    -------
    ValueError
        If null_strategy is "ERROR" and NULLs are found
    """
    if logger:
        logger.debug(f"Applying NULL strategy: {null_strategy}")

    # Handle both string and Enum
    if isinstance(null_strategy, Enum):
        strategy_value = null_strategy.value
    else:
        strategy_value = null_strategy

    # Handle NULL strategy
    null_count = series.isna().sum()

    if strategy_value == NullStrategy.ERROR.value and null_count > 0:
        raise ValueError(f"Found {null_count} NULL values but null_strategy is ERROR")

    elif strategy_value == NullStrategy.ANONYMIZE.value:
        # Replace NULLs with unknown_value
        series = series.fillna(unknown_value)
        if logger:
            logger.debug(f"Replaced {null_count} NULL values with '{unknown_value}'")

    elif strategy_value == NullStrategy.EXCLUDE.value:
        # EXCLUDE is typically handled by filtering before this function
        # But we can mark in context for later filtering
        if context is not None:
            context['null_indices'] = series[series.isna()].index.tolist()
        if logger:
            logger.debug(f"Marked {null_count} NULL values for exclusion")

    # PRESERVE: do nothing (default behavior)

    # Apply rare value template if provided
    if rare_value_template and "{n}" in rare_value_template:
        series = _apply_rare_value_template(
            series,
            rare_value_template,
            None  # Use default pattern
        )

    return series


def format_rare_value(template: str, index: int) -> str:
    """
    Format a rare value using the provided template.

    Parameters:
    -----------
    template : str
        Template string containing {n} placeholder
    index : int
        Index to insert

    Returns:
    --------
    str
        Formatted value
    """
    return template.format(n=index)


# Helper functions

def _apply_rare_value_template(
        series: pd.Series,
        template: str,
        prefix_pattern: Union[str, None] = None
) -> pd.Series:
    """
    Apply template formatting to rare/grouped values.

    Parameters:
    -----------
    series : pd.Series
        Series with values to format
    template : str
        Template with {n} placeholder
    prefix_pattern : Union[str, None]
        Pattern to match values to format (None = use default)

    Returns:
    --------
    pd.Series
        Series with formatted values
    """
    if "{n}" not in template:
        return series

    # Use default pattern if none provided
    if prefix_pattern is None:
        pattern = PREFIX_NUMBER_PATTERN
    else:
        pattern = re.compile(prefix_pattern)

    result = series.copy()

    def format_value(val):
        if pd.isna(val):
            return val

        match = pattern.match(str(val))
        if match and len(match.groups()) >= 2:
            try:
                index = int(match.group(2))
                return template.format(n=index)
            except (ValueError, IndexError):
                pass

        return val

    result = result.map(format_value)

    return result


# Dask implementations (placeholders for future)

def apply_hierarchy_dask(
        series: "dd.Series",
        config: Dict[str, Any],
        context: Dict[str, Any],
        logger: Optional[logging.Logger] = None
) -> "dd.Series":
    """
    Apply hierarchy strategy using Dask for large datasets.

    TODO: Implement proper Dask-based processing to avoid memory issues.
    Note: This is a placeholder for future implementation.
    Currently falls back to pandas implementation.
    """
    if logger:
        logger.debug("Dask implementation not yet available, using pandas fallback")
    return apply_hierarchy(_to_pandas(series), config, context, logger)


def apply_merge_low_freq_dask(
        series: "dd.Series",
        config: Dict[str, Any],
        context: Dict[str, Any],
        logger: Optional[logging.Logger] = None
) -> "dd.Series":
    """
    Apply merge low frequency strategy using Dask.

    TODO: Implement distributed frequency counting and grouping.
    Note: This is a placeholder for future implementation.
    """
    if logger:
        logger.debug("Dask implementation not yet available, using pandas fallback")
    return apply_merge_low_freq(_to_pandas(series), config, context, logger)


def apply_frequency_based_dask(
        series: "dd.Series",
        config: Dict[str, Any],
        context: Dict[str, Any],
        logger: Optional[logging.Logger] = None
) -> "dd.Series":
    """
    Apply frequency-based strategy using Dask.

    TODO: Implement distributed top-k selection and grouping.
    Note: This is a placeholder for future implementation.
    """
    if logger:
        logger.debug("Dask implementation not yet available, using pandas fallback")
    return apply_frequency_based(_to_pandas(series), config, context, logger)


# Module metadata
__version__ = "1.2.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export strategy functions
__all__ = [
    # Main strategies
    'apply_hierarchy',
    'apply_merge_low_freq',
    'apply_frequency_based',
    'apply_null_and_unknown_strategy',

    # Helper functions
    'format_rare_value',

    # Dask versions (future)
    'apply_hierarchy_dask',
    'apply_merge_low_freq_dask',
    'apply_frequency_based_dask'
]
def _to_pandas(series):
    """Return pandas Series whether input is pandas or Dask."""
    if hasattr(series, "compute"):
        return series.compute()
    return series