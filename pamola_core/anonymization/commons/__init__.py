"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Anonymization Common Utilities
Package:       pamola_core.anonymization.commons
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2024
Revised:       2025
License:       BSD 3-Clause

Description:
    This package provides common utilities used across all anonymization operations
    in the PAMOLA.CORE framework. It includes validation utilities, metric collection,
    visualization helpers, privacy-specific data processing, generalization algorithms,
    specialized data type processors, and configuration helpers.

Package Structure:
    - validation_utils.py: Parameter validation and error handling
    - metric_utils.py: Lightweight process metrics for anonymization
    - privacy_metric_utils.py: Privacy-specific metrics and monitoring
    - visualization_utils.py: Visualization helpers for anonymization metrics
    - data_utils.py: Privacy-specific data processing utilities
    - generalization_algorithms.py: Core generalization algorithms
    - specialized_processors.py: Processors for specialized data types
    - anonymization_helpers.py: Configuration and profiling integration helpers

Usage:
    The utilities in this package are designed to be used by anonymization
    operations to ensure consistent behavior, validation, metrics collection,
    and integration with the PAMOLA.CORE framework.

    Example:
        from pamola_core.anonymization.commons import (
            validate_field_exists,
            calculate_anonymization_effectiveness,
            process_nulls,
            numeric_generalization_binning
        )
"""

# =============================================================================
# Version Information
# =============================================================================

__version__ = "2.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# =============================================================================
# Imports from validation_utils.py
# =============================================================================

from pamola_core.anonymization.commons.validation_utils import (
    # Core field validation
    validate_field_exists,
    validate_multiple_fields_exist,
    validate_numeric_field,
    validate_categorical_field,
    validate_datetime_field,

    # Strategy and parameter validation
    validate_generalization_strategy,
    validate_bin_count,
    validate_precision,
    validate_range_limits,
    validate_percentiles,

    # Mode and strategy validation
    validate_null_strategy,

    # Specialized data type validation
    validate_geographic_data,
    validate_temporal_sequence,
    validate_network_identifiers,
    validate_financial_data,

    # File and path validation
    validate_file_path,
    validate_directory_path,

    # Utility functions
    get_validation_error_result,
    get_validation_success_result,

    # Custom exception classes
    ValidationError,
    ConditionalValidationError,

    # Profiling integration validation
    validate_specialized_type,

)

# =============================================================================
# Imports from metric_utils.py
# =============================================================================

from pamola_core.anonymization.commons.metric_utils import (
    # Basic metrics
    calculate_anonymization_effectiveness,
    calculate_generalization_metrics,
    calculate_masking_metrics,
    calculate_suppression_metrics,

    # Performance metrics
    calculate_process_performance,

    # Distribution summaries
    get_value_distribution_summary,

    # Collection and persistence
    collect_operation_metrics,
    save_process_metrics,
    get_process_summary_message,
)

# =============================================================================
# Imports from privacy_metric_utils.py
# =============================================================================

from pamola_core.anonymization.commons.privacy_metric_utils import (
    # Coverage metrics
    calculate_anonymization_coverage,
    calculate_suppression_rate,

    # Group metrics
    get_group_size_distribution,
    calculate_min_group_size,
    calculate_vulnerable_records_ratio,

    # Generalization metrics
    calculate_generalization_level,
    calculate_value_reduction_ratio,

    # Risk indicators
    calculate_uniqueness_score,
    calculate_simple_disclosure_risk,

    # Process control
    check_anonymization_thresholds,
    get_process_summary,
    calculate_batch_metrics,

    # Constants
    DEFAULT_K_THRESHOLD,
    DEFAULT_SUPPRESSION_WARNING,
    DEFAULT_COVERAGE_TARGET,
    EPSILON,
)

# =============================================================================
# Imports from visualization_utils.py
# =============================================================================

from pamola_core.anonymization.commons.visualization_utils import (
    # Filename and path generation
    generate_visualization_filename,
    register_visualization_artifact,

    # Data preparation
    sample_large_dataset,
    prepare_comparison_data,
    calculate_optimal_bins,

    # Visualization creation
    create_metric_visualization,
    create_comparison_visualization,
    create_distribution_visualization,

    # Constants
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MAX_CATEGORIES,
    DEFAULT_HISTOGRAM_BINS,
)

# =============================================================================
# Imports from data_utils.py
# =============================================================================

from pamola_core.anonymization.commons.data_utils import (
    # Privacy-aware null processing
    process_nulls,

    # Risk-based filtering
    filter_records_conditionally,
    handle_vulnerable_records,

    # Factory functions
    create_risk_based_processor,
    create_privacy_level_processor,

    # Adaptive anonymization
    apply_adaptive_anonymization,

    # Utility functions
    get_risk_statistics,
    get_privacy_recommendations,

    # Constants and configurations
    DEFAULT_K_THRESHOLD as DATA_K_THRESHOLD,
    DEFAULT_SUPPRESSION_WARNING as DATA_SUPPRESSION_WARNING,
    DEFAULT_COVERAGE_TARGET as DATA_COVERAGE_TARGET,
    RISK_LEVELS,
    NULL_STRATEGIES as DATA_NULL_STRATEGIES,
    VULNERABLE_STRATEGIES as DATA_VULNERABLE_STRATEGIES,
    PRIVACY_LEVELS,
)

# =============================================================================
# Imports from visualization_utils.py
# =============================================================================

from pamola_core.anonymization.commons.masking_patterns import (
    # Masking pattern utilities
    apply_pattern_mask,
    create_random_mask,
    validate_mask_character,
    analyze_pattern_security,
    get_format_preserving_mask,
    generate_mask,
    generate_mask_char,
    is_separator,
    preserve_pattern_mask,
    get_mask_char_pool,
    set_mask_char_pool,
    clear_mask_char_pools,
    # Constants
    MASK_CHAR_POOLS,
)

# =============================================================================
# Define __all__ for explicit exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",

    # Validation utilities
    "validate_field_exists",
    "validate_multiple_fields_exist",
    "validate_numeric_field",
    "validate_categorical_field",
    "validate_datetime_field",
    "validate_generalization_strategy",
    "validate_bin_count",
    "validate_precision",
    "validate_range_limits",
    "validate_percentiles",
    "validate_null_strategy",
    "validate_conditional_parameters",
    "validate_geographic_data",
    "validate_temporal_sequence",
    "validate_network_identifiers",
    "validate_financial_data",
    "validate_file_path",
    "validate_directory_path",
    "get_validation_error_result",
    "get_validation_success_result",
    "ValidationError",
    "ConditionalValidationError",
    "validate_specialized_type",

    # Metric utilities
    "calculate_anonymization_effectiveness",
    "calculate_generalization_metrics",
    "calculate_masking_metrics",
    "calculate_suppression_metrics",
    "calculate_process_performance",
    "get_value_distribution_summary",
    "collect_operation_metrics",
    "save_process_metrics",
    "get_process_summary_message",

    # Privacy metric utilities
    "calculate_anonymization_coverage",
    "calculate_suppression_rate",
    "get_group_size_distribution",
    "calculate_min_group_size",
    "calculate_vulnerable_records_ratio",
    "calculate_generalization_level",
    "calculate_value_reduction_ratio",
    "calculate_uniqueness_score",
    "calculate_simple_disclosure_risk",
    "check_anonymization_thresholds",
    "get_process_summary",
    "calculate_batch_metrics",

    # Visualization utilities
    "generate_visualization_filename",
    "register_visualization_artifact",
    "sample_large_dataset",
    "prepare_comparison_data",
    "calculate_optimal_bins",
    "create_metric_visualization",
    "create_comparison_visualization",
    "create_distribution_visualization",

    # Data processing utilities
    "process_nulls",
    "filter_records_conditionally",
    "handle_vulnerable_records",
    "create_risk_based_processor",
    "create_privacy_level_processor",
    "apply_adaptive_anonymization",
    "get_risk_statistics",
    "get_privacy_recommendations",

    # Masking patterns
    "apply_pattern_mask",
    "create_random_mask",
    "validate_mask_character",
    "analyze_pattern_security",
    "get_format_preserving_mask",
    "generate_mask",
    "generate_mask_char",
    "is_separator",
    "preserve_pattern_mask",
    "get_mask_char_pool",
    "set_mask_char_pool",
    "clear_mask_char_pools",

    # Constants
    "DEFAULT_K_THRESHOLD",
    "DEFAULT_SUPPRESSION_WARNING",
    "DEFAULT_COVERAGE_TARGET",
    "EPSILON",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_MAX_CATEGORIES",
    "DEFAULT_HISTOGRAM_BINS",
    "RISK_LEVELS",
    "PRIVACY_LEVELS",
    "MASK_CHAR_POOLS",
]

# =============================================================================
# Module initialization logging (optional)
# =============================================================================

import logging

logger = logging.getLogger(__name__)
logger.debug(f"Initialized anonymization.commons package v{__version__}")