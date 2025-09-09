"""
AMOLA.CORE - Privacy-Preserving AI Data Processors
------------------------------------------------------------
Module:        Category Analysis and Manipulation Utilities
Package:       pamola_core.anonymization.commons
Version:       2.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01
Updated:       2025-01-23
License:       BSD 3-Clause

Description:
  This module provides statistical analysis, distribution metrics,
  and grouping strategies for categorical data in the context of
  privacy-preserving anonymization operations.

Purpose:
  Supports categorical generalization operations by providing:
  - Distribution analysis for understanding data characteristics
  - Rare category identification for privacy risk assessment
  - Basic grouping strategies for generalization
  - Validation utilities for category mappings

Key Features:
  - Fast distribution analysis with privacy-relevant metrics
  - Configurable rare category detection
  - Simple but effective grouping strategies
  - Integration with statistical and diversity metrics
  - Robust error handling for production use

Dependencies:
  - pandas: DataFrame and Series operations
  - numpy: Numerical computations
  - typing: Type hints
  - collections.Counter: Frequency counting
  - pamola_core.utils.statistical_metrics: Gini, concentration, entropy
  - pamola_core.utils.nlp.diversity_metrics: Semantic diversity
  - pamola_core.anonymization.commons.text_processing_utils: Text normalization

Usage:
  Used primarily by categorical generalization operations to:
  1. Analyze category distributions before anonymization
  2. Identify rare categories that pose privacy risks
  3. Group rare categories to achieve k-anonymity
  4. Validate the effectiveness of generalizations

Examples:
  >>> from pamola_core.anonymization.commons.category_utils import analyze_category_distribution
  >>> distribution = analyze_category_distribution(df['job_title'], top_n=20) # type:ignore
  >>> print(f"Entropy: {distribution['entropy']:.2f}")
  >>> print(f"Top 10 coverage: {distribution['concentration_metrics']['cr_10']:.1f}%")

Changelog:
  2.1.0 (2025-01-23):
    - Added TypedDict definitions for better type safety
    - Improved performance by caching value_counts
    - Added configurable coverage threshold for validation
    - Enhanced normalization level parameter for semantic diversity
    - Adjusted logging levels (info -> debug for routine operations)
    - Improved percent_threshold handling with automatic detection
    - Clarified group number calculation in numbered strategy
  2.0.0 (2025-01-23):
    - REMOVED: collect_categorical_metrics() - functionality moved to metric_utils
    - Updated for categorical generalization refactoring v3.0
    - Improved thread-safety considerations
    - Enhanced documentation
  1.0.0 (2025-01):
    - Initial implementation
"""

import logging
from typing import Callable, Dict, List, Set, Tuple, Optional, Any, Union, TypedDict

import dask
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# Import text processing utilities
from pamola_core.anonymization.commons.text_processing_utils import normalize_text

# Import diversity metrics for semantic analysis
from pamola_core.utils.io_helpers.dask_utils import convert_to_dask
from pamola_core.utils.nlp.diversity_metrics import calculate_semantic_diversity

# Import statistical metrics
from pamola_core.utils.ops.op_data_processing import get_dataframe_chunks
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.statistical_metrics import (
    calculate_gini_coefficient,
    calculate_concentration_metrics,
    calculate_shannon_entropy,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Constants for anonymization-specific thresholds
DEFAULT_TOP_N = 20
DEFAULT_RARE_THRESHOLD = 10
DEFAULT_RARE_PERCENT = 0.01
DEFAULT_GROUP_PREFIX = "GROUP_"
DEFAULT_OTHER_LABEL = "OTHER"
MAX_GROUPS = 100  # Privacy limit on number of groups
EPSILON = 1e-10  # For numerical stability
DEFAULT_COVERAGE_WARNING_THRESHOLD = 0.95  # Configurable coverage threshold


# Type definitions for better type safety
class RareCategoryInfo(TypedDict):
    """Type definition for rare category detailed information."""

    count: int
    percentage: float
    rank: int
    below_count_threshold: bool
    below_percent_threshold: bool


class GroupingInfo(TypedDict):
    """Type definition for grouping operation results."""

    groups_created: int
    group_mapping: Dict[str, str]
    group_sizes: Dict[str, int]
    reduction_ratio: float
    categories_grouped: int
    original_categories: int
    final_categories: int
    threshold_used: int
    strategy: str


class ValidationResult(TypedDict):
    """Type definition for validation results."""

    is_valid: bool
    length_match: bool
    null_preservation: bool
    unmapped_categories: List[str]
    inconsistent_mappings: List[Tuple[str, str, str]]
    coverage: float
    reduction_ratio: float
    original_unique: int
    mapped_unique: int
    warnings: List[str]
    errors: List[str]


def analyze_category_distribution(
    series: pd.Series,
    top_n: int = DEFAULT_TOP_N,
    min_frequency: int = 1,
    calculate_entropy: bool = True,
    calculate_gini: bool = True,
    calculate_concentration: bool = True,
    value_counts: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of category distribution for anonymization planning.

    Provides metrics essential for understanding privacy risks and planning
    appropriate generalization strategies.

    Parameters:
    -----------
    series : pd.Series
        Categorical data to analyze
    top_n : int, optional
        Number of top categories to detail (default: 20)
    min_frequency : int, optional
        Minimum frequency to include in detailed analysis (default: 1)
    calculate_entropy : bool, optional
        Whether to calculate Shannon entropy (default: True)
    calculate_gini : bool, optional
        Whether to calculate Gini coefficient (default: True)
    calculate_concentration : bool, optional
        Whether to calculate concentration metrics (default: True)
    value_counts : Optional[pd.Series], optional
        Pre-computed value counts to avoid recalculation (default: None)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - total_categories: int - Total unique categories
        - total_records: int - Total number of records
        - frequency_counts: Dict[str, int] - Category frequencies
        - percentage_distribution: Dict[str, float] - Category percentages
        - top_n_categories: List[Tuple[str, int, float]] - Top categories
        - rare_categories: List[str] - Categories below min_frequency
        - null_count: int - Number of null values
        - entropy: float - Shannon entropy (if calculated)
        - normalized_entropy: float - Normalized entropy [0, 1]
        - gini_coefficient: float - Inequality measure (if calculated)
        - concentration_metrics: Dict[str, float] - CR-5, CR-10, etc.
        - coverage_90_percentile: int - Categories needed for 90% coverage

    Examples:
    ---------
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', None, 'D'] * 100)
    >>> dist = analyze_category_distribution(data)
    >>> print(f"Unique categories: {dist['total_categories']}")
    >>> print(f"Gini coefficient: {dist['gini_coefficient']:.3f}")
    """
    try:
        # Handle empty series
        if len(series) == 0:
            logger.warning("Empty series provided for distribution analysis")
            return _empty_distribution_result()

        # Basic statistics
        total_records = len(series)
        null_count = series.isna().sum()
        non_null_series = series.dropna()

        # Use provided value_counts or calculate
        if value_counts is None:
            value_counts = non_null_series.value_counts()
        else:
            # Ensure it matches the series
            value_counts = value_counts[
                value_counts.index.isin(non_null_series.unique())
            ]

        total_categories = len(value_counts)

        # Handle edge case of no categories
        if total_categories == 0:
            logger.warning("No non-null categories found")
            return _empty_distribution_result(
                total_records=total_records, null_count=null_count
            )

        # Calculate frequencies and percentages
        frequency_counts = value_counts.to_dict()
        total_non_null = len(non_null_series)
        percentage_distribution = {
            cat: (count / total_non_null * 100) if total_non_null > 0 else 0.0
            for cat, count in frequency_counts.items()
        }

        # Top N categories with details
        top_n_categories = [
            (cat, int(count), float(percentage_distribution[cat]))
            for cat, count in value_counts.head(top_n).items()
        ]

        # Identify rare categories
        rare_categories = [
            cat for cat, count in frequency_counts.items() if count < min_frequency
        ]

        # Build result dictionary
        result = {
            "total_categories": int(total_categories),
            "total_records": int(total_records),
            "null_count": int(null_count),
            "null_percentage": (
                float(null_count / total_records * 100) if total_records > 0 else 0.0
            ),
            "frequency_counts": frequency_counts,
            "percentage_distribution": percentage_distribution,
            "top_n_categories": top_n_categories,
            "rare_categories": rare_categories,
            "rare_category_count": len(rare_categories),
        }

        # Calculate entropy if requested
        if calculate_entropy and total_categories > 0:
            entropy = calculate_shannon_entropy(
                non_null_series, base=2.0, normalize=False
            )
            # Normalized entropy (0 to 1)
            max_entropy = np.log2(total_categories) if total_categories > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            result["entropy"] = float(entropy)
            result["normalized_entropy"] = float(normalized_entropy)
            result["max_entropy"] = float(max_entropy)

        # Calculate Gini coefficient if requested
        if calculate_gini and total_categories > 0:
            gini = calculate_gini_coefficient(
                list(frequency_counts.values()), is_frequency=True
            )
            result["gini_coefficient"] = float(gini)

        # Calculate concentration metrics if requested
        if calculate_concentration and total_categories > 0:
            # Standard concentration ratios for privacy analysis
            concentration = calculate_concentration_metrics(
                frequency_counts, top_k=[1, 5, 10, 20], as_percentage=True
            )
            result["concentration_metrics"] = concentration

            # Additional privacy-relevant metric: categories for 90% coverage
            cumsum = np.cumsum(value_counts.values)
            threshold_90 = total_non_null * 0.9
            coverage_90 = int(np.searchsorted(cumsum, threshold_90, side="right") + 1)
            result["coverage_90_percentile"] = min(coverage_90, total_categories)

        return result

    except Exception as e:
        logger.error(f"Error in category distribution analysis: {str(e)}")
        return _empty_distribution_result(error=str(e))


def identify_rare_categories(
    series: pd.Series,
    count_threshold: int = DEFAULT_RARE_THRESHOLD,
    percent_threshold: float = DEFAULT_RARE_PERCENT,
    combined_criteria: bool = True,
    value_counts: Optional[pd.Series] = None,
) -> Tuple[Set[str], Dict[str, RareCategoryInfo]]:
    """
    Identify rare categories based on privacy risk criteria.

    Rare categories pose higher re-identification risks and are primary
    candidates for generalization or suppression.

    Parameters:
    -----------
    series : pd.Series
        Categorical data to analyze
    count_threshold : int, optional
        Minimum count threshold (default: 10)
    percent_threshold : float, optional
        Minimum percentage threshold. If < 1.0, treated as fraction (e.g., 0.01 = 1%).
        If >= 1.0, treated as percentage (e.g., 1.0 = 1%).
        Default: 0.01 (1%)
    combined_criteria : bool, optional
        If True, both criteria must be met (AND)
        If False, either criterion triggers (OR)
        Default: True (more conservative)
    value_counts : Optional[pd.Series], optional
        Pre-computed value counts to avoid recalculation (default: None)

    Returns:
    --------
    Tuple[Set[str], Dict[str, RareCategoryInfo]]
        - rare_categories: Set of category names identified as rare
        - detailed_info: Dictionary with details for each rare category:
            - count: int - Frequency count
            - percentage: float - Percentage of total
            - rank: int - Rank by frequency (1 = most common)
            - below_count_threshold: bool
            - below_percent_threshold: bool

    Examples:
    ---------
    >>> data = pd.Series(['Common'] * 100 + ['Rare1'] * 5 + ['Rare2'] * 3)
    >>> rare_cats, info = identify_rare_categories(data, count_threshold=10)
    >>> print(f"Rare categories: {rare_cats}")
    >>> for cat, details in info.items():
    ...     print(f"{cat}: count={details['count']}, rank={details['rank']}") # type:ignore
    """
    try:
        # Handle empty series
        if len(series) == 0:
            logger.warning("Empty series provided for rare category identification")
            return set(), {}

        # Get value counts
        if value_counts is None:
            value_counts = series.value_counts()

        total_count = len(series.dropna())

        if total_count == 0:
            return set(), {}

        # Handle percent_threshold format
        # If >= 1.0, treat as percentage; if < 1.0, treat as fraction
        if percent_threshold >= 1.0:
            effective_percent_threshold = percent_threshold / 100.0
            logger.debug(
                f"Treating percent_threshold {percent_threshold} as percentage, using {effective_percent_threshold:.4f}"
            )
        else:
            effective_percent_threshold = percent_threshold
            logger.debug(f"Treating percent_threshold {percent_threshold} as fraction")

        # Calculate percentages and ranks
        percentages = (value_counts / total_count * 100).to_dict()
        ranks = {cat: rank + 1 for rank, cat in enumerate(value_counts.index)}

        # Identify rare categories based on criteria
        rare_categories = set()
        detailed_info: Dict[str, RareCategoryInfo] = {}

        for category, count in value_counts.items():
            percentage = percentages[category]

            # Check criteria
            below_count = count < count_threshold
            below_percent = percentage < (effective_percent_threshold * 100)

            is_rare = (
                (below_count and below_percent)
                if combined_criteria
                else (below_count or below_percent)
            )

            if is_rare:
                rare_categories.add(category)
                key = str(category)
                detailed_info[key] = {
                    "count": int(count),
                    "percentage": float(percentage),
                    "rank": int(ranks[category]),
                    "below_count_threshold": below_count,
                    "below_percent_threshold": below_percent,
                }

        # Log summary (use debug level for routine operations)
        logger.debug(
            f"Identified {len(rare_categories)} rare categories out of {len(value_counts)}"
        )

        return rare_categories, detailed_info

    except Exception as e:
        logger.error(f"Error identifying rare categories: {str(e)}")
        return set(), {}


def group_rare_categories(
    series: pd.Series,
    grouping_strategy: str = "single_other",
    threshold: Union[int, float] = DEFAULT_RARE_THRESHOLD,
    max_groups: int = MAX_GROUPS,
    group_prefix: str = DEFAULT_GROUP_PREFIX,
    preserve_top_n: Optional[int] = None,
    other_label: str = DEFAULT_OTHER_LABEL,
    value_counts: Optional[pd.Series] = None,
) -> Tuple[pd.Series, GroupingInfo]:
    """
    Group rare categories using privacy-preserving strategies.

    Essential for achieving k-anonymity by ensuring no category has
    too few members.

    Parameters:
    -----------
    series : pd.Series
        Categorical data to process
    grouping_strategy : str, optional
        Strategy for grouping (default: "single_other"):
        - "single_other": All rare categories → single "OTHER" group
        - "numbered": Create numbered groups (GROUP_001, GROUP_002, etc.)
          Number of groups = min(max_groups, max(1, len(rare_categories) // 10))
          This ensures reasonable group sizes while respecting the max_groups limit.
        - "frequency_bands": Group by frequency ranges (MVP: equal bands)
    threshold : Union[int, float], optional
        Count threshold (int) or percentage (float < 1.0)
        Default: 10
    max_groups : int, optional
        Maximum number of groups to create (default: 100)
    group_prefix : str, optional
        Prefix for numbered groups (default: "GROUP_")
    preserve_top_n : Optional[int], optional
        Number of top categories to always preserve
    other_label : str, optional
        Label for single "other" group (default: "OTHER")
    value_counts : Optional[pd.Series], optional
        Pre-computed value counts to avoid recalculation (default: None)

    Returns:
    --------
    Tuple[pd.Series, GroupingInfo]
        - grouped_series: Series with rare categories grouped
        - grouping_info: Dictionary containing:
            - groups_created: int - Number of groups created
            - group_mapping: Dict[str, str] - Original → group mapping
            - group_sizes: Dict[str, int] - Size of each group
            - reduction_ratio: float - Reduction in unique values
            - categories_grouped: int - Number of categories grouped

    Examples:
    ---------
    >>> data = pd.Series(['A'] * 100 + ['B'] * 50 + ['C'] * 5 + ['D'] * 3)
    >>> grouped, info = group_rare_categories(data, threshold=10)
    >>> print(f"Groups created: {info['groups_created']}")
    >>> print(f"Reduction: {info['reduction_ratio']:.1%}")
    """
    try:
        # Validate inputs
        if len(series) == 0:
            logger.warning("Empty series provided for grouping")
            return series.copy(), _empty_grouping_result()

        if grouping_strategy not in ["single_other", "numbered", "frequency_bands"]:
            logger.warning(
                f"Unknown strategy '{grouping_strategy}', using 'single_other'"
            )
            grouping_strategy = "single_other"

        # Prepare data
        result_series = series.copy()

        # Use provided value_counts or calculate once
        if value_counts is None:
            value_counts = series.value_counts()

        total_count = len(series.dropna())

        # Determine threshold type and identify rare categories
        if isinstance(threshold, float) and threshold < 1.0:
            # Percentage threshold
            count_threshold = int(total_count * threshold)
        else:
            # Absolute count threshold
            count_threshold = int(threshold)

        # Preserve top N categories if specified
        if preserve_top_n and preserve_top_n > 0:
            top_categories = set(value_counts.head(preserve_top_n).index)
        else:
            top_categories = set()

        # Identify categories to group
        categories_to_group = []
        for category, count in value_counts.items():
            if category not in top_categories and count < count_threshold:
                categories_to_group.append(category)

        # Apply grouping strategy
        group_mapping = {}
        groups_created = 0

        if grouping_strategy == "single_other":
            # Simple strategy: all rare → OTHER
            for category in categories_to_group:
                group_mapping[category] = other_label
            groups_created = 1 if categories_to_group else 0

        elif grouping_strategy == "numbered":
            # Create numbered groups of approximately equal size
            if categories_to_group:
                # Sort by frequency to group similar-sized categories
                sorted_categories = sorted(
                    categories_to_group, key=lambda x: value_counts[x], reverse=True
                )

                # Calculate number of groups needed
                # Formula: min(max_groups, max(1, len(categories) // 10))
                # This creates ~10 categories per group, but respects max_groups limit
                n_groups = min(max_groups, max(1, len(categories_to_group) // 10))
                groups_created = n_groups

                logger.debug(
                    f"Numbered strategy: {len(categories_to_group)} categories "
                    f"→ {n_groups} groups (max_groups={max_groups})"
                )

                # Assign to groups
                for i, category in enumerate(sorted_categories):
                    group_num = i % n_groups
                    group_mapping[category] = f"{group_prefix}{group_num:03d}"
            else:
                groups_created = 0

        elif grouping_strategy == "frequency_bands":
            # Group by frequency bands (MVP: simple equal bands)
            if categories_to_group:
                # Get counts for rare categories
                rare_counts = [value_counts[cat] for cat in categories_to_group]
                min_count = min(rare_counts)
                max_count = max(rare_counts)

                if min_count < max_count:
                    # Create frequency bands
                    n_bands = min(5, max_groups)  # Limit bands for simplicity
                    band_size = (max_count - min_count) / n_bands

                    for category in categories_to_group:
                        count = value_counts[category]
                        band = min(int((count - min_count) / band_size), n_bands - 1)
                        group_mapping[category] = f"{group_prefix}BAND_{band}"

                    groups_created = n_bands
                else:
                    # All same frequency, use single group
                    for category in categories_to_group:
                        group_mapping[category] = f"{group_prefix}BAND_0"
                    groups_created = 1
            else:
                groups_created = 0

        # Apply mapping to series
        if group_mapping:
            mask = result_series.isin(group_mapping.keys())
            result_series.loc[mask] = result_series.loc[mask].map(group_mapping)

        # Calculate group sizes
        group_sizes = {}
        for original_cat, group in group_mapping.items():
            if group not in group_sizes:
                group_sizes[group] = 0
            group_sizes[group] += int(value_counts[original_cat])

        # Calculate metrics
        original_unique = len(value_counts)
        final_unique = result_series.nunique()
        reduction_ratio = (
            1 - (final_unique / original_unique) if original_unique > 0 else 0.0
        )

        # Build info dictionary
        grouping_info: GroupingInfo = {
            "groups_created": int(groups_created),
            "group_mapping": group_mapping,
            "group_sizes": group_sizes,
            "reduction_ratio": float(reduction_ratio),
            "categories_grouped": len(categories_to_group),
            "original_categories": int(original_unique),
            "final_categories": int(final_unique),
            "threshold_used": int(count_threshold),
            "strategy": grouping_strategy,
        }

        # Log summary (use debug level for routine operations)
        logger.debug(
            f"Grouped {len(categories_to_group)} categories into {groups_created} groups "
            f"using '{grouping_strategy}' strategy"
        )

        return result_series, grouping_info

    except Exception as e:
        logger.error(f"Error in category grouping: {str(e)}")
        return series.copy(), _empty_grouping_result(error=str(e))


def calculate_category_entropy(
    series: pd.Series, base: float = 2.0, normalize: bool = True
) -> float:
    """
    Calculate Shannon entropy of categorical distribution.

    Entropy measures the randomness/unpredictability of categories,
    useful for assessing anonymization effectiveness.

    Parameters:
    -----------
    series : pd.Series
        Categorical data
    base : float, optional
        Logarithm base (default: 2 for bits)
    normalize : bool, optional
        Whether to normalize by maximum entropy (default: True)

    Returns:
    --------
    float
        Entropy value (normalized to [0, 1] if normalize=True)

    Examples:
    ---------
    >>> data = pd.Series(['A', 'B', 'C', 'D'])  # Maximum entropy
    >>> entropy = calculate_category_entropy(data)
    >>> print(f"Entropy: {entropy:.3f}")  # Should be ~1.0 (normalized)
    """
    try:
        # Remove nulls
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return 0.0

        # Calculate entropy using statistical metrics
        entropy = calculate_shannon_entropy(
            clean_series, base=base, normalize=normalize
        )

        return float(entropy)

    except Exception as e:
        logger.error(f"Error calculating entropy: {str(e)}")
        return 0.0


def validate_category_mapping(
    original: pd.Series,
    mapped: pd.Series,
    mapping: Optional[Dict[str, str]] = None,
    coverage_threshold: float = DEFAULT_COVERAGE_WARNING_THRESHOLD,
) -> ValidationResult:
    """
    Validate category mapping for anonymization correctness.

    Ensures that generalizations are applied consistently and no
    data is lost or corrupted during the mapping process.

    Parameters:
    -----------
    original : pd.Series
        Original categorical data
    mapped : pd.Series
        Data after category mapping
    mapping : Optional[Dict[str, str]]
        Expected mapping dictionary (original → generalized)
    coverage_threshold : float, optional
        Threshold for coverage warning (default: 0.95 = 95%)

    Returns:
    --------
    ValidationResult
        Validation results:
        - is_valid: bool - Overall validation status
        - length_match: bool - Same number of records
        - null_preservation: bool - Nulls handled correctly
        - unmapped_categories: List[str] - Categories without mapping
        - inconsistent_mappings: List[Tuple[str, str, str]] - Errors
        - coverage: float - Percentage of values successfully mapped
        - reduction_ratio: float - Reduction in unique values
        - warnings: List[str] - Non-critical issues

    Examples:
    ---------
    >>> original = pd.Series(['Engineer', 'Manager', 'Director'])
    >>> mapped = pd.Series(['Technical', 'Business', 'Business'])
    >>> mapping = {'Engineer': 'Technical', 'Manager': 'Business'}
    >>> result = validate_category_mapping(original, mapped, mapping)
    >>> print(f"Valid: {result['is_valid']}")
    """
    try:
        warnings = []
        errors = []

        # Basic length check
        length_match = len(original) == len(mapped)
        if not length_match:
            errors.append(f"Length mismatch: {len(original)} vs {len(mapped)}")

        # Null preservation check
        orig_nulls = original.isna().sum()
        mapped_nulls = mapped.isna().sum()
        null_preservation = orig_nulls == mapped_nulls
        if not null_preservation:
            warnings.append(f"Null count changed: {orig_nulls} → {mapped_nulls}")

        # Prepare for detailed validation
        orig_values = set(original.dropna().unique())
        mapped_values = set(mapped.dropna().unique())

        unmapped_categories = []
        inconsistent_mappings = []

        # If mapping provided, validate it
        if mapping:
            # Check all original values are in mapping
            for orig_val in orig_values:
                if orig_val not in mapping:
                    unmapped_categories.append(str(orig_val))

            # Check mapping consistency
            for idx in range(len(original)):
                if pd.notna(original.iloc[idx]) and pd.notna(mapped.iloc[idx]):
                    orig_val = original.iloc[idx]
                    mapped_val = mapped.iloc[idx]

                    if orig_val in mapping:
                        expected = mapping[orig_val]
                        if mapped_val != expected:
                            inconsistent_mappings.append(
                                (str(orig_val), str(expected), str(mapped_val))
                            )

        # Calculate coverage
        non_null_original = original.dropna()
        if len(non_null_original) > 0:
            successfully_mapped = sum(
                pd.notna(mapped.iloc[i])
                for i in range(len(original))
                if pd.notna(original.iloc[i])
            )
            coverage = successfully_mapped / len(non_null_original)
        else:
            coverage = 1.0

        # Calculate reduction ratio
        orig_unique = original.nunique()
        mapped_unique = mapped.nunique()
        if orig_unique > 0:
            reduction_ratio = 1 - (mapped_unique / orig_unique)
        else:
            reduction_ratio = 0.0

        # Determine overall validity
        is_valid = length_match and len(errors) == 0 and len(inconsistent_mappings) == 0

        # Additional checks
        if mapped_unique > orig_unique:
            warnings.append(
                f"Mapped categories ({mapped_unique}) exceed "
                f"original ({orig_unique})"
            )

        # Use configurable coverage threshold
        if coverage < coverage_threshold:
            warnings.append(
                f"Low coverage: {coverage:.1%} (threshold: {coverage_threshold:.1%})"
            )

        result: ValidationResult = {
            "is_valid": is_valid,
            "length_match": length_match,
            "null_preservation": null_preservation,
            "unmapped_categories": unmapped_categories,
            "inconsistent_mappings": inconsistent_mappings,
            "coverage": float(coverage),
            "reduction_ratio": float(reduction_ratio),
            "original_unique": int(orig_unique),
            "mapped_unique": int(mapped_unique),
            "warnings": warnings,
            "errors": errors,
        }

        return result

    except Exception as e:
        logger.error(f"Error validating category mapping: {str(e)}")
        return {
            "is_valid": False,
            "length_match": False,
            "null_preservation": False,
            "unmapped_categories": [],
            "inconsistent_mappings": [],
            "coverage": 0.0,
            "reduction_ratio": 0.0,
            "original_unique": 0,
            "mapped_unique": 0,
            "errors": [str(e)],
            "warnings": [],
        }


def calculate_semantic_diversity_safe(
    categories: List[str],
    method: str = "token_overlap",
    normalization_level: str = "basic",
) -> float:
    """
    Calculate semantic diversity with anonymization-safe defaults.

    Wrapper around diversity metrics with privacy-preserving settings.

    Parameters:
    -----------
    categories : List[str]
        List of category names
    method : str, optional
        Diversity calculation method (default: "token_overlap")
    normalization_level : str, optional
        Text normalization level: "basic", "advanced", "none" (default: "basic")
        Use "advanced" for multilingual datasets

    Returns:
    --------
    float
        Diversity score [0, 1]
    """
    try:
        if not categories:
            return 0.0

        # Use specified normalization level
        normalized_categories = [
            normalize_text(cat, level=normalization_level) for cat in categories
        ]

        diversity = calculate_semantic_diversity(
            normalized_categories,
            method=method,
            normalize=True,
            min_token_length=2,  # Ignore very short tokens
            case_sensitive=False,
        )

        return float(diversity)

    except Exception as e:
        logger.error(f"Error calculating semantic diversity: {str(e)}")
        return 0.0


# Note: collect_categorical_metrics has been REMOVED in v2.0.0
# This functionality is now provided by metric_utils.collect_operation_metrics()
# DO NOT add load_hierarchy_dictionary() here - it belongs in hierarchy_dictionary.py


# Helper functions for consistent empty results
def _empty_distribution_result(
    total_records: int = 0, null_count: int = 0, error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a consistent empty result dictionary for category distribution analysis.

    Parameters
    ----------
    total_records : int
        Number of records in the input data (default: 0)
    null_count : int
        Number of null/missing entries (default: 0)
    error : str or None
        Optional error message to include in the result

    Returns
    -------
    result : dict
        A dictionary containing default/empty values for all expected metrics and,
        if specified, an 'error' field for diagnostics.
    """
    result: Dict[str, Any] = {
        "total_categories": 0,
        "total_records": total_records,
        "null_count": null_count,
        "null_percentage": (
            100.0 if total_records == 0 else (null_count / total_records * 100.0)
        ),
        "frequency_counts": {},
        "percentage_distribution": {},
        "top_n_categories": [],
        "rare_categories": [],
        "rare_category_count": 0,
    }

    # Optionally add error message (will always be str if present)
    if error is not None:
        result["error"] = str(error)

    return result


def _empty_grouping_result(error: Optional[str] = None) -> GroupingInfo:
    """Create empty grouping result with consistent structure."""
    result: GroupingInfo = {
        "groups_created": 0,
        "group_mapping": {},
        "group_sizes": {},
        "reduction_ratio": 0.0,
        "categories_grouped": 0,
        "original_categories": 0,
        "final_categories": 0,
        "threshold_used": 0,
        "strategy": "none",
    }

    # TypedDict doesn't allow extra keys, so we return as-is without error field
    # If error handling is needed, it should be done at the caller level
    return result


# Helper function for processing DataFrames
def process_dataframe_using_dask(
    df: pd.DataFrame,
    process_function: Callable,
    is_use_batch_dask: bool = False,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, bool, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Process DataFrame using Dask with metadata collection.

    Returns:
        Tuple of:
        - processed DataFrame
        - success flag
        - combined category_mapping
        - combined hierarchy_info
        - combined hierarchy_cache
    """
    task_logger = task_logger or logging.getLogger(__name__)
    total_rows = len(df)
    npartitions = kwargs.get("npartitions", 2)
    dask_partition_size = kwargs.get("dask_partition_size", "100MB")

    all_category_mapping: Dict[str, Any] = {}
    all_hierarchy_info: Dict[str, Any] = {}
    all_hierarchy_cache: Dict[str, Any] = {}
    all_fuzzy_matches: int = 0
    all_unknown_values: set[str] = set()

    result_df = pd.DataFrame()  # fallback in case of error

    try:
        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Dask processing setup",
                    "total_parts": npartitions,
                    "dask_partition_size": dask_partition_size,
                    "total_rows": total_rows,
                },
            )

        # Convert to Dask DataFrame
        ddf, npartitions = convert_to_dask(
            df,
            dask_npartitions=npartitions,
            dask_partition_size=dask_partition_size or "100MB",
            logger=task_logger,
        )

        if progress_tracker:
            progress_tracker.update(
                2,
                {
                    "step": "Dask processing started",
                    "total_parts": npartitions,
                    "dask_partition_size": dask_partition_size,
                    "total_rows": total_rows,
                },
            )

        task_logger.info(f"Processing {total_rows} rows in {npartitions} chunks")
        task_logger.info(
            f"Batch Dask mode: {'enabled' if is_use_batch_dask else 'disabled'}"
        )

        if is_use_batch_dask:
            # Batch mode: process whole ddf
            processed_partitions = process_function(ddf, **kwargs)
        else:
            # Define a function for processing that can be applied to Dask partitions
            def process_partition(partition):
                processed_partition = process_function(
                    partition.copy(deep=True), **kwargs
                )
                return processed_partition

            # Apply to Dask DataFrame
            processed_partitions = ddf.map_partitions(process_partition)

        # Compute results
        results = processed_partitions.compute()

        # Collect results
        result_dfs = []
        for df_chunk, cat_map, hier_info, hier_cache, fuz_mat, unkn_val in results:
            result_dfs.append(df_chunk)
            all_category_mapping.update(cat_map or {})
            all_hierarchy_info.update(hier_info or {})
            all_hierarchy_cache.update(hier_cache or {})
            all_fuzzy_matches += fuz_mat
            all_unknown_values.update(unkn_val or {})

        # Concatenate all DataFrames
        result_df = pd.concat(result_dfs, ignore_index=True)

        if progress_tracker:
            progress_tracker.update(
                3,
                {
                    "step": "Dask finalization",
                    "total_parts": npartitions,
                    "dask_partition_size": dask_partition_size,
                    "total_rows": total_rows,
                },
            )

        task_logger.info("Processing completed.")
        return (
            result_df,
            True,
            all_category_mapping,
            all_hierarchy_info,
            all_hierarchy_cache,
            all_fuzzy_matches,
            all_unknown_values,
        )

    except Exception as e:
        task_logger.exception("Dask processing failed")
        return result_df, False, {}, {}, {}, 0, set()


def process_dataframe_using_joblib(
    df: pd.DataFrame,
    process_function: Callable,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, bool, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Process DataFrame using Joblib and return result with shared states.

    Returns:
        - Concatenated result DataFrame
        - Success flag
        - Combined category_mapping
        - Combined hierarchy_info
        - Combined hierarchy_cache
    """
    task_logger = task_logger or logging.getLogger(__name__)
    n_jobs = kwargs.get("parallel_processes", -1)
    chunk_size = kwargs.get("chunk_size", 10000)

    if n_jobs <= 0 and n_jobs != -1:
        return df, False, {}, {}, {}

    try:
        chunks = list(get_dataframe_chunks(df, chunk_size))
        total_chunks = len(chunks)
        total_rows = len(df)

        task_logger.info(
            f"Processing {total_rows} rows in {total_chunks} chunks with Joblib"
        )

        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Parallel processing setup",
                    "total_chunks": total_chunks,
                    "total_rows": total_rows,
                },
            )

        # Function to process each chunk with error handling
        def process_with_progress(chunk):
            try:
                processed_chunk = process_function(chunk, **kwargs)
                return processed_chunk
            except Exception as e:
                return None

        # Update progress tracker for each chunk
        if progress_tracker:
            progress_tracker.update(
                2,
                {
                    "step": "Parallel processing",
                    "total_chunks": total_chunks,
                    "total_rows": total_rows,
                },
            )

        # Directly use the generator to iterate through chunks
        processed_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_with_progress)(chunk) for chunk in chunks
        )

        task_logger.info(
            f"Processing completed for {total_rows} rows in {total_chunks} chunks"
        )

        # Compute final result
        if progress_tracker:
            progress_tracker.update(
                3,
                {
                    "step": "Parallel finalization",
                    "total_chunks": total_chunks,
                },
            )

        # Check for any failed chunks
        if any(r is None for r in processed_chunks):
            task_logger.warning("Some chunks failed during processing.")
            return df, False, {}, {}, {}, 0, set()

        # Combine all results
        result_dfs = []
        all_category_mapping: Dict[str, Any] = {}
        all_hierarchy_info: Dict[str, Any] = {}
        all_hierarchy_cache: Dict[str, Any] = {}
        all_fuzzy_matches: int = 0
        all_unknown_values: set[str] = set()

        for batch_df, cat_map, hier_info, hier_cache, fuz_mat, unkn_val in processed_chunks:
            result_dfs.append(batch_df)
            all_category_mapping.update(cat_map or {})
            all_hierarchy_info.update(hier_info or {})
            all_hierarchy_cache.update(hier_cache or {})
            all_fuzzy_matches += fuz_mat
            all_unknown_values.update(unkn_val or {})

        return (
            pd.concat(result_dfs, ignore_index=True),
            True,
            all_category_mapping,
            all_hierarchy_info,
            all_hierarchy_cache,
            all_fuzzy_matches,
            all_unknown_values,
        )

    except Exception as e:
        task_logger.exception("Error during Joblib processing.")
        return df, False, {}, {}, {}, 0, set()


def process_dataframe_using_chunk(
    df: pd.DataFrame,
    process_function: Callable,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, bool, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Process DataFrame using chunked loop with metadata tracking.

    Returns:
        - Concatenated result DataFrame
        - Success flag
        - Combined category_mapping
        - Combined hierarchy_info
        - Combined hierarchy_cache
    """
    task_logger = task_logger or logging.getLogger(__name__)
    chunk_size = kwargs.get("chunk_size", 10000)

    if chunk_size <= 1:
        task_logger.warning("Chunk size invalid. Returning as is.")
        return df, False, {}, {}, {}

    task_logger.info(f"Processing {len(df)} rows using chunk size {chunk_size}")

    processed_chunks = []
    all_category_mapping: Dict[str, Any] = {}
    all_hierarchy_info: Dict[str, Any] = {}
    all_hierarchy_cache: Dict[str, Any] = {}
    all_fuzzy_matches: int = 0
    all_unknown_values: set[str] = set()

    try:
        chunks = list(get_dataframe_chunks(df, chunk_size))
        total_chunks = len(chunks)

        if progress_tracker:
            progress_tracker.total = total_chunks
            progress_tracker.update(
                1, {"step": "Chunk processing setup", "total_chunks": total_chunks}
            )

        for i, chunk in enumerate(chunks):
            try:
                if progress_tracker:
                    progress_tracker.update(
                        i + 1, {"step": f"Processing chunk {i + 1}/{total_chunks}"}
                    )

                result = process_function(chunk, **kwargs)

                if isinstance(result, tuple) and len(result) == 6:
                    batch_df, cat_map, hier_info, hier_cache, fuz_mat, unkn_val = result
                else:
                    raise ValueError(
                        "Expected tuple of 6 elements from process_function"
                    )

                processed_chunks.append(batch_df)
                all_category_mapping.update(cat_map or {})
                all_hierarchy_info.update(hier_info or {})
                all_hierarchy_cache.update(hier_cache or {})
                all_fuzzy_matches += fuz_mat
                all_unknown_values.update(unkn_val or {})

            except Exception as e:
                task_logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                processed_chunks.append(None)
                continue

        task_logger.info(
            f"Completed processing {len(df)} rows in {total_chunks} chunks"
        )

        if progress_tracker:
            progress_tracker.update(
                3, {"step": "Chunk finalization", "total_chunks": total_chunks}
            )

        if any(chunk is None for chunk in processed_chunks):
            task_logger.warning("Some chunks failed to process.")
            return df, False, {}, {}, {}, 0, set()

        return (
            pd.concat(processed_chunks, ignore_index=True),
            True,
            all_category_mapping,
            all_hierarchy_info,
            all_hierarchy_cache,
            all_fuzzy_matches,
            all_unknown_values
        )

    except Exception as e:
        task_logger.exception("Error during chunked processing.")
        return df, False, {}, {}, {}, 0, set()


# Module metadata
__version__ = "2.1.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main functions
__all__ = [
    # Core analysis functions
    "analyze_category_distribution",
    "identify_rare_categories",
    "group_rare_categories",
    # Metrics functions
    "calculate_category_entropy",
    "calculate_semantic_diversity_safe",
    # Validation functions
    "validate_category_mapping",
    # Processing functions
    "process_dataframe_using_dask",
    "process_dataframe_using_joblib",
    "process_dataframe_using_chunk",
    # Type definitions
    "RareCategoryInfo",
    "GroupingInfo",
    "ValidationResult",
    # Constants
    "DEFAULT_TOP_N",
    "DEFAULT_RARE_THRESHOLD",
    "DEFAULT_RARE_PERCENT",
    "DEFAULT_GROUP_PREFIX",
    "DEFAULT_OTHER_LABEL",
    "MAX_GROUPS",
    "DEFAULT_COVERAGE_WARNING_THRESHOLD",
]
