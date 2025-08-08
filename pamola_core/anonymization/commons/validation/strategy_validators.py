"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Strategy Parameter Validators
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Validators for anonymization strategy parameters and operation modes.
  Provides validation functions for strategy selection, mode validation,
  parameter bounds checking, and strategy-specific parameter validation.

Key Features:
  - Strategy name validation against allowed lists
  - Operation mode validation (REPLACE/ENRICH)
  - Strategy-specific parameter validation
  - Null handling strategy validation
  - Parameter range and bounds checking
  - Strategy compatibility checking

Design Principles:
  - Return ValidationResult for consistency
  - Use decorators for common functionality
  - Support both strict and lenient validation modes
  - Clear error messages with suggestions

Usage:
  Used by anonymization operations to validate strategy selection
  and associated parameters before processing begins.

Dependencies:
  - typing - Type hints
  - pandas - DataFrame operations
  - logging - Error logging
  - validation.base - Base classes and utilities
  - validation.decorators - Validation decorators
  - validation.exceptions - Custom exceptions

Changelog:
  1.0.0 - Initial implementation extracted from validation_utils
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

from .base import ValidationResult
from .decorators import standard_validator, validate_types
import re
from .exceptions import (
    InvalidParameterError,
    InvalidStrategyError
)

# Configure module logger
logger = logging.getLogger(__name__)

# Standard strategy constants
GENERALIZATION_STRATEGIES = {
   "numeric": ["binning", "rounding", "range", "custom"],
   "categorical": ["merge_low_freq", "hierarchy", "frequency_based", "custom"],
   "datetime": ["truncate", "shift", "generalize", "custom"]
}

NOISE_STRATEGIES = ["gaussian", "laplace", "uniform", "custom"]
SUPPRESSION_STRATEGIES = ["cell", "record", "attribute", "conditional"]
MASKING_STRATEGIES = ["full", "partial", "pattern", "format_preserving"]
PSEUDONYMIZATION_STRATEGIES = ["hash", "mapping", "sequential", "random"]

# Standard modes
OPERATION_MODES = ["REPLACE", "ENRICH"]
NULL_STRATEGIES = ["PRESERVE", "EXCLUDE", "ERROR", "ANONYMIZE"]


# =============================================================================
# Strategy Validation Functions
# =============================================================================

@standard_validator(field_name="strategy")
@validate_types(strategy=str, valid_strategies=list)
def validate_strategy(strategy: str,
                     valid_strategies: List[str],
                     operation_type: Optional[str] = None) -> ValidationResult:
   """
   Validate that a strategy is in the list of valid strategies.

   Args:
       strategy: Strategy name to validate
       valid_strategies: List of allowed strategy names
       operation_type: Optional operation type for context

   Returns:
       ValidationResult with validation outcome

   Raises:
       InvalidStrategyError: If strategy is not valid
   """
   if strategy not in valid_strategies:
       raise InvalidStrategyError(
           strategy=strategy,
           valid_strategies=valid_strategies,
           operation_type=operation_type
       )

   return ValidationResult(
       is_valid=True,
       field_name="strategy",
       details={
           "strategy": strategy,
           "operation_type": operation_type
       }
   )


@standard_validator(field_name="strategy")
def validate_generalization_strategy(strategy: str,
                                    data_type: str) -> ValidationResult:
   """
   Validate generalization strategy for specific data type.

   Args:
       strategy: Generalization strategy name
       data_type: Data type (numeric, categorical, datetime)

   Returns:
       ValidationResult with validation outcome
   """
   if data_type not in GENERALIZATION_STRATEGIES:
       return ValidationResult(
           is_valid=False,
           field_name="strategy",
           errors=[f"Unknown data type '{data_type}' for generalization"],
           details={"data_type": data_type, "valid_types": list(GENERALIZATION_STRATEGIES.keys())}
       )

   valid_strategies = GENERALIZATION_STRATEGIES[data_type]
   return validate_strategy(strategy, valid_strategies, f"generalization_{data_type}")


@standard_validator(field_name="strategy")
def validate_noise_strategy(strategy: str) -> ValidationResult:
   """
   Validate noise addition strategy.

   Args:
       strategy: Noise strategy name

   Returns:
       ValidationResult with validation outcome
   """
   return validate_strategy(strategy, NOISE_STRATEGIES, "noise")


@standard_validator(field_name="strategy")
def validate_suppression_strategy(strategy: str) -> ValidationResult:
   """
   Validate suppression strategy.

   Args:
       strategy: Suppression strategy name

   Returns:
       ValidationResult with validation outcome
   """
   return validate_strategy(strategy, SUPPRESSION_STRATEGIES, "suppression")


@standard_validator(field_name="strategy")
def validate_masking_strategy(strategy: str) -> ValidationResult:
   """
   Validate masking strategy.

   Args:
       strategy: Masking strategy name

   Returns:
       ValidationResult with validation outcome
   """
   return validate_strategy(strategy, MASKING_STRATEGIES, "masking")


@standard_validator(field_name="strategy")
def validate_pseudonymization_strategy(strategy: str) -> ValidationResult:
   """
   Validate pseudonymization strategy.

   Args:
       strategy: Pseudonymization strategy name

   Returns:
       ValidationResult with validation outcome
   """
   return validate_strategy(strategy, PSEUDONYMIZATION_STRATEGIES, "pseudonymization")


# =============================================================================
# Mode Validation Functions
# =============================================================================

@standard_validator(field_name="mode")
@validate_types(mode=str)
def validate_operation_mode(mode: str,
                           valid_modes: Optional[List[str]] = None) -> ValidationResult:
   """
   Validate operation mode (REPLACE/ENRICH).

   Args:
       mode: Operation mode to validate
       valid_modes: Optional list of valid modes (defaults to OPERATION_MODES)

   Returns:
       ValidationResult with validation outcome
   """
   if valid_modes is None:
       valid_modes = OPERATION_MODES

   if mode not in valid_modes:
       raise InvalidParameterError(
           param_name="mode",
           param_value=mode,
           reason=f"Must be one of {valid_modes}",
           valid_range=str(valid_modes)
       )

   return ValidationResult(
       is_valid=True,
       field_name="mode",
       details={"mode": mode}
   )

@standard_validator(field_name="null_strategy")
@validate_types(strategy=str)
def validate_null_strategy(strategy: str,
                          valid_strategies: Optional[List[str]] = None) -> ValidationResult:
   """
   Validate null handling strategy.

   Args:
       strategy: Null handling strategy
       valid_strategies: Optional list of valid strategies

   Returns:
       ValidationResult with validation outcome
   """
   if valid_strategies is None:
       valid_strategies = NULL_STRATEGIES

   if strategy not in valid_strategies:
       raise InvalidStrategyError(
           strategy=strategy,
           valid_strategies=valid_strategies,
           operation_type="null_handling"
       )

   return ValidationResult(
       is_valid=True,
       field_name="null_strategy",
       details={"strategy": strategy}
   )


# =============================================================================
# Parameter Range Validation Functions
# =============================================================================

@standard_validator()
@validate_types(bin_count=int)
def validate_bin_count(bin_count: int,
                      min_bins: int = 2,
                      max_bins: int = 1000) -> ValidationResult:
   """
   Validate bin count for binning strategies.

   Args:
       bin_count: Number of bins
       min_bins: Minimum allowed bins
       max_bins: Maximum allowed bins

   Returns:
       ValidationResult with validation outcome
   """
   if bin_count < min_bins or bin_count > max_bins:
       raise InvalidParameterError(
           param_name="bin_count",
           param_value=bin_count,
           reason=f"Must be between {min_bins} and {max_bins}",
           valid_range=f"[{min_bins}, {max_bins}]"
       )

   return ValidationResult(
       is_valid=True,
       field_name="bin_count",
       details={
           "bin_count": bin_count,
           "min_bins": min_bins,
           "max_bins": max_bins
       }
   )


@standard_validator()
@validate_types(precision=int)
def validate_precision(precision: int,
                      min_precision: int = -10,
                      max_precision: int = 10) -> ValidationResult:
   """
   Validate precision for rounding strategies.

   Args:
       precision: Decimal precision
       min_precision: Minimum precision
       max_precision: Maximum precision

   Returns:
       ValidationResult with validation outcome
   """
   if precision < min_precision or precision > max_precision:
       raise InvalidParameterError(
           param_name="precision",
           param_value=precision,
           reason=f"Must be between {min_precision} and {max_precision}",
           valid_range=f"[{min_precision}, {max_precision}]"
       )

   return ValidationResult(
       is_valid=True,
       field_name="precision",
       details={
           "precision": precision,
           "min_precision": min_precision,
           "max_precision": max_precision
       }
   )


@standard_validator()
def validate_range_limits(range_limits: Tuple[float, float]) -> ValidationResult:
   """
   Validate range limits for range-based strategies.

   Args:
       range_limits: Tuple of (min, max) values

   Returns:
       ValidationResult with validation outcome
   """
   if not isinstance(range_limits, (tuple, list)) or len(range_limits) != 2:
       raise InvalidParameterError(
           param_name="range_limits",
           param_value=range_limits,
           reason="Must be a tuple of two numeric values (min, max)"
       )

   try:
       min_val = float(range_limits[0])
       max_val = float(range_limits[1])
   except (ValueError, TypeError) as e:
       raise InvalidParameterError(
           param_name="range_limits",
           param_value=range_limits,
           reason=f"Values must be numeric: {e}"
       )

   if min_val >= max_val:
       raise InvalidParameterError(
           param_name="range_limits",
           param_value=range_limits,
           reason=f"Minimum ({min_val}) must be less than maximum ({max_val})"
       )

   return ValidationResult(
       is_valid=True,
       field_name="range_limits",
       details={
           "min_value": min_val,
           "max_value": max_val,
           "range": max_val - min_val
       }
   )


@standard_validator()
def validate_percentiles(percentiles: List[float]) -> ValidationResult:
   """
   Validate percentile values for quantile-based strategies.

   Args:
       percentiles: List of percentile values (0-100)

   Returns:
       ValidationResult with validation outcome
   """
   if not isinstance(percentiles, (list, tuple)) or len(percentiles) == 0:
       raise InvalidParameterError(
           param_name="percentiles",
           param_value=percentiles,
           reason="Must be a non-empty list of numeric values"
       )

   # Validate each percentile
   for i, p in enumerate(percentiles):
       try:
           p_float = float(p)
       except (ValueError, TypeError):
           raise InvalidParameterError(
               param_name=f"percentiles[{i}]",
               param_value=p,
               reason="Must be numeric"
           )

       if not 0 <= p_float <= 100:
           raise InvalidParameterError(
               param_name=f"percentiles[{i}]",
               param_value=p_float,
               reason="Must be between 0 and 100",
               valid_range="[0, 100]"
           )

   # Check for duplicates
   if len(set(percentiles)) != len(percentiles):
       return ValidationResult(
           is_valid=False,
           field_name="percentiles",
           errors=["Percentiles list contains duplicates"],
           details={"percentiles": percentiles}
       )

   return ValidationResult(
       is_valid=True,
       field_name="percentiles",
       details={
           "percentiles": sorted(percentiles),
           "count": len(percentiles)
       }
   )


# =============================================================================
# Strategy-Specific Parameter Validation
# =============================================================================

@standard_validator()
def validate_noise_parameters(noise_level: float,
                             strategy: str,
                             bounds: Optional[Tuple[float, float]] = None) -> ValidationResult:
   """
   Validate noise addition parameters.

   Args:
       noise_level: Noise level/scale parameter
       strategy: Noise strategy type
       bounds: Optional bounds for clipping

   Returns:
       ValidationResult with validation outcome
   """
   result = ValidationResult(is_valid=True)

   # Validate noise level
   if noise_level <= 0:
       raise InvalidParameterError(
           param_name="noise_level",
           param_value=noise_level,
           reason="Must be positive",
           valid_range="(0, ∞)"
       )

   # Strategy-specific validation
   if strategy == "gaussian":
       # For Gaussian, noise_level is standard deviation
       result.details["interpretation"] = "standard deviation"
   elif strategy == "laplace":
       # For Laplace, noise_level is scale parameter
       result.details["interpretation"] = "scale parameter"
   elif strategy == "uniform":
       # For uniform, noise_level is range
       result.details["interpretation"] = "range (-noise_level, +noise_level)"

   # Validate bounds if provided
   if bounds is not None:
       bounds_result = validate_range_limits(bounds)
       if not bounds_result.is_valid:
           return bounds_result
       result.details["bounds"] = bounds

   result.details.update({
       "noise_level": noise_level,
       "strategy": strategy
   })

   return result


@standard_validator()
def validate_masking_parameters(mask_char: str,
                               preserve_length: bool = True,
                               pattern: Optional[str] = None) -> ValidationResult:
   """
   Validate masking parameters.

   Args:
       mask_char: Character used for masking
       preserve_length: Whether to preserve original length
       pattern: Optional regex pattern for partial masking

   Returns:
       ValidationResult with validation outcome
   """
   result = ValidationResult(is_valid=True)

   # Validate mask character
   if not isinstance(mask_char, str) or len(mask_char) != 1:
       raise InvalidParameterError(
           param_name="mask_char",
           param_value=mask_char,
           reason="Must be a single character"
       )

   # Validate pattern if provided
   if pattern is not None:
       try:

           re.compile(pattern)
       except re.error as e:
           raise InvalidParameterError(
               param_name="pattern",
               param_value=pattern,
               reason=f"Invalid regex pattern: {e}"
           )

   result.details = {
       "mask_char": mask_char,
       "preserve_length": preserve_length,
       "pattern": pattern
   }

   return result


@standard_validator()
def validate_hierarchy_parameters(hierarchy_depth: int,
                                 merge_threshold: Optional[int] = None) -> ValidationResult:
   """
   Validate categorical hierarchy parameters.

   Args:
       hierarchy_depth: Depth of hierarchy to use
       merge_threshold: Optional threshold for merging categories

   Returns:
       ValidationResult with validation outcome
   """
   result = ValidationResult(is_valid=True)

   # Validate depth
   if hierarchy_depth < 1 or hierarchy_depth > 10:
       raise InvalidParameterError(
           param_name="hierarchy_depth",
           param_value=hierarchy_depth,
           reason="Must be between 1 and 10",
           valid_range="[1, 10]"
       )

   # Validate merge threshold
   if merge_threshold is not None:
       if merge_threshold < 1:
           raise InvalidParameterError(
               param_name="merge_threshold",
               param_value=merge_threshold,
               reason="Must be at least 1"
           )
       result.details["merge_threshold"] = merge_threshold

   result.details["hierarchy_depth"] = hierarchy_depth

   return result


# =============================================================================
# Composite Validation Functions
# =============================================================================

@standard_validator()
def validate_strategy_compatibility(strategy: str,
                                   field_type: str,
                                   operation_type: str) -> ValidationResult:
   """
   Validate that strategy is compatible with field and operation type.

   Args:
       strategy: Strategy name
       field_type: Type of field (numeric, categorical, etc.)
       operation_type: Type of operation (generalization, noise, etc.)

   Returns:
       ValidationResult with validation outcome
   """
   result = ValidationResult(is_valid=True)

   # Define compatibility matrix
   compatibility = {
       "generalization": {
           "numeric": ["binning", "rounding", "range", "custom"],
           "categorical": ["merge_low_freq", "hierarchy", "frequency_based", "custom"],
           "datetime": ["truncate", "shift", "generalize", "custom"]
       },
       "noise": {
           "numeric": ["gaussian", "laplace", "uniform", "custom"],
           "categorical": [],  # Noise not typically applied to categorical
           "datetime": ["shift"]  # Only shift makes sense for dates
       },
       "masking": {
           "numeric": ["full", "partial"],
           "categorical": ["full"],
           "datetime": ["partial", "format_preserving"]
       }
   }

   # Check compatibility
   if operation_type in compatibility:
       if field_type in compatibility[operation_type]:
           valid_strategies = compatibility[operation_type][field_type]
           if strategy not in valid_strategies:
               result.is_valid = False
               result.errors.append(
                   f"Strategy '{strategy}' not compatible with {field_type} fields "
                   f"for {operation_type} operations"
               )
               result.details["valid_strategies"] = valid_strategies
       else:
           result.is_valid = False
           result.errors.append(
               f"Operation type '{operation_type}' not supported for {field_type} fields"
           )
   else:
       result.warnings.append(f"Unknown operation type '{operation_type}'")

   result.details.update({
       "strategy": strategy,
       "field_type": field_type,
       "operation_type": operation_type
   })

   return result


@standard_validator()
def validate_output_field_configuration(df: pd.DataFrame,
                                       mode: str,
                                       output_field_name: Optional[str] = None,
                                       original_field_name: Optional[str] = None) -> ValidationResult:
   """
   Validate output field configuration based on mode.

   Args:
       df: DataFrame to check
       mode: Operation mode (REPLACE/ENRICH)
       output_field_name: Proposed output field name
       original_field_name: Original field name

   Returns:
       ValidationResult with validation outcome
   """
   result = ValidationResult(is_valid=True)

   # Validate mode first
   mode_result = validate_operation_mode(mode)
   if not mode_result.is_valid:
       return mode_result

   if mode == "ENRICH":
       # ENRICH mode requires output field name
       if not output_field_name:
           result.is_valid = False
           result.errors.append("Output field name required for ENRICH mode")
           return result

       # Check if output field already exists
       if output_field_name in df.columns:
           result.warnings.append(
               f"Output field '{output_field_name}' already exists and will be overwritten"
           )

       # Check if output field is different from original
       if output_field_name == original_field_name:
           result.warnings.append(
               "Output field same as original field in ENRICH mode"
           )

   elif mode == "REPLACE":
       # REPLACE mode shouldn't have different output field
       if output_field_name and output_field_name != original_field_name:
           result.warnings.append(
               f"Output field name '{output_field_name}' ignored in REPLACE mode"
           )

   result.details = {
       "mode": mode,
       "output_field_name": output_field_name,
       "original_field_name": original_field_name
   }

   return result


# Module exports
__all__ = [
   # Strategy validators
   'validate_strategy',
   'validate_generalization_strategy',
   'validate_noise_strategy',
   'validate_suppression_strategy',
   'validate_masking_strategy',
   'validate_pseudonymization_strategy',

   # Mode validators
   'validate_operation_mode',
   'validate_null_strategy',

   # Parameter validators
   'validate_bin_count',
   'validate_precision',
   'validate_range_limits',
   'validate_percentiles',

   # Strategy-specific validators
   'validate_noise_parameters',
   'validate_masking_parameters',
   'validate_hierarchy_parameters',

   # Composite validators
   'validate_strategy_compatibility',
   'validate_output_field_configuration',

   # Constants
   'GENERALIZATION_STRATEGIES',
   'NOISE_STRATEGIES',
   'SUPPRESSION_STRATEGIES',
   'MASKING_STRATEGIES',
   'PSEUDONYMIZATION_STRATEGIES',
   'OPERATION_MODES',
   'NULL_STRATEGIES'
]