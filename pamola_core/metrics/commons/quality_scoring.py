"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Data Quality Scoring Utilities
Package:       pamola_core.metrics.commons.quality_scoring
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides comprehensive data quality calculation with validation rules integration.
  Applies validation rules to calculate completeness, validity, and diversity metrics
  with support for both system defaults and user-defined custom rules.

Key Features:
  - Integration with validation rules framework
  - Configurable quality weights (q1, q2, q3)
  - Per-column and dataset-level metrics
  - UI-friendly output with factual vs effective scores
  - Warning messages for low quality data
  - Support for required vs non-required field handling

Dependencies:
  - pandas - DataFrame operations
  - numpy - statistical calculations
  - typing - type hints
  - dataclasses - data structures
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from .validation_rules import (
    ValidationResult,
    rule_registry,
    create_rule_from_code,
)
from .schema_manager import SchemaManager, FieldDefinition

logger = logging.getLogger(__name__)

DATE_PATTERN = re.compile(
    r"""(
        \d{4}[-/.]\d{1,2}[-/.]\d{1,2}   |   # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        \d{1,2}[-/.]\d{1,2}[-/.]\d{4}   |   # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        \d{8}                           |   # YYYYMMDD
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4} |  # Sep 16, 2025
        \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}     # 16 Sep 2025
    )""",
    re.VERBOSE | re.IGNORECASE,
)


@dataclass
class QualityWeights:
    """Configuration for quality scoring weights."""

    completeness: float = 0.5  # q1
    validity: float = 0.3  # q2
    diversity: float = 0.2  # q3

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.completeness + self.validity + self.diversity
        if not np.isclose(total, 1.0, rtol=1e-9):
            raise ValueError(f"Quality weights must sum to 1.0, got {total:.4f}")


@dataclass
class ColumnQualityMetrics:
    """Quality metrics for a single column."""

    completeness: float
    validity: float
    diversity: float
    consistency: float
    quality_score: float
    factual_completeness: float  # Actual ratio regardless of required setting
    factual_diversity: float  # Actual ratio regardless of unique setting
    non_missing_count: int
    missing_count: int
    unique_count: int
    error_count: int
    total_count: int
    validation_results: Dict[str, ValidationResult]
    timeliness: Dict[str, Any]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class DatasetQualityMetrics:
    """Quality metrics for the entire dataset."""

    overall_quality: float
    required_fields_quality: float
    required_fields_completeness: float
    unique_fields_quality: float
    unique_fields_diversity: float
    effective_dataset_completeness: float
    effective_dataset_diversity: float
    factual_dataset_completeness: float
    factual_dataset_validity: float
    factual_dataset_diversity: float
    factual_dataset_consistency: float
    required_fields_count: int
    unique_fields_count: int
    required_fields_non_missing_count: int
    factual_dataset_non_missing_count: int
    required_fields_missing_count: int
    factual_dataset_missing_count: int
    field_unique_values_count: int
    factual_dataset_unique_values_count: int
    factual_dataset_diversity_range: str
    unique_fields_diversity_range: str
    total_fields_count: int
    quality_issues: List[str]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class DataQualityCalculator:
    """Main calculator for data quality metrics with validation rules."""

    def __init__(self, weights: Optional[QualityWeights] = None):
        """
        Initialize data quality calculator.

        Parameters
        ----------
        weights : QualityWeights, optional
            Custom weights for quality components. Defaults to standard weights.
        """
        self.weights = weights or QualityWeights()
        self.rule_registry = rule_registry

    def calculate_quality(
        self,
        df: pd.DataFrame,
        schema: SchemaManager,
        analyze_scope: str = "dataset",
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset
        schema : SchemaManager
            Schema with field definitions and validation rules

        Returns
        -------
        Dict[str, Any]
            Comprehensive quality metrics including UI-friendly outputs
        """
        # Basic input validation
        if df is None or df.empty:
            return self._empty_dataset_result()

        # Validate scope and columns parameters
        if analyze_scope not in ("dataset", "column"):
            logger.warning(
                f"Invalid analyze_scope '{analyze_scope}', defaulting to 'dataset'"
            )
            analyze_scope = "dataset"
        if analyze_scope == "column" and (not columns or not isinstance(columns, list)):
            logger.warning(
                "analyze_scope='column' requires a non-empty list of columns; falling back to full dataset analysis"
            )
            analyze_scope = "dataset"

        # Choose columns to analyze
        if analyze_scope == "column" and columns:
            target_columns = [c for c in columns if c in df.columns]
        else:
            target_columns = list(df.columns)

        # Calculate per-column metrics
        column_metrics = {}
        for column in target_columns:
            field_def = schema.get_field(column)
            if field_def is None:
                # Create default field definition if not in schema
                field_def = FieldDefinition(name=column, data_type="unknown")
                schema.add_field(field_def)

            column_metrics[column] = self._calculate_column_quality(
                df[column], field_def
            )

        # Calculate dataset-level metrics only for full-dataset runs
        dataset_metrics = None
        if analyze_scope != "column":
            dataset_metrics = self._calculate_dataset_quality(
                df, schema, column_metrics
            )

        # Prepare UI-friendly output
        return self._prepare_ui_output(
            df,
            schema,
            column_metrics,
            dataset_metrics,
            analyze_scope=analyze_scope,
            analyzed_columns=target_columns,
        )

    def _calculate_column_quality(
        self, series: pd.Series, field_def: FieldDefinition
    ) -> ColumnQualityMetrics:
        """
        Calculate quality metrics for a single column.

        Parameters
        ----------
        series : pd.Series
            Data series to analyze
        field_def : FieldDefinition
            Field definition with validation rules

        Returns
        -------
        ColumnQualityMetrics
            Quality metrics for the column
        """
        total_count = len(series)
        if total_count == 0:
            return ColumnQualityMetrics(
                completeness=0.0,
                validity=0.0,
                diversity=0.0,
                consistency=0.0,
                quality_score=0.0,
                factual_completeness=0.0,
                factual_diversity=0.0,
                non_missing_count=0,
                missing_count=0,
                unique_count=0,
                error_count=0,
                total_count=0,
                validation_results={},
                timeliness={},
            )

        # Apply validation rules
        validation_results = self._apply_validation_rules(series, field_def)

        # Calculate completeness
        completeness, factual_completeness, non_missing_count, missing_count = (
            self._calculate_completeness(series, field_def.is_required)
        )

        # Calculate validity
        validity = self._calculate_validity(series, validation_results)

        # Calculate diversity
        diversity, factual_diversity, unique_count = self._calculate_diversity(
            series, field_def.is_unique
        )

        # Calculate consistency (format/values stability)
        consistency = self._calculate_consistency(series)

        # Timeliness (for datetime-like)
        timeliness = self._calculate_timeliness(series)

        # Calculate overall quality score
        quality_score = (
            self.weights.completeness * completeness
            + self.weights.validity * validity
            + self.weights.diversity * diversity
        ) * 100

        # Count total errors
        error_count = sum(result.error_count for result in validation_results.values())

        # Generate warnings
        warnings = self._generate_column_warnings(
            series,
            field_def,
            validity,
            consistency,
            factual_completeness,
            factual_diversity,
            timeliness,
        )

        return ColumnQualityMetrics(
            completeness=completeness,
            validity=validity,
            diversity=diversity,
            consistency=consistency,
            quality_score=quality_score,
            factual_completeness=factual_completeness,
            factual_diversity=factual_diversity,
            non_missing_count=non_missing_count,
            missing_count=missing_count,
            unique_count=unique_count,
            error_count=error_count,
            total_count=total_count,
            validation_results=validation_results,
            timeliness=timeliness,
            warnings=warnings,
        )

    def _apply_validation_rules(
        self, series: pd.Series, field_def: FieldDefinition
    ) -> Dict[str, ValidationResult]:
        """
        Apply all validation rules to a series.

        Parameters
        ----------
        series : pd.Series
            Data series to validate
        field_def : FieldDefinition
            Field definition with rules

        Returns
        -------
        Dict[str, ValidationResult]
            Results from each validation rule
        """
        results = {}

        for rule_name in field_def.validation_rules:
            try:
                rule = self._resolve_rule(rule_name, field_def)

                if rule and getattr(rule, "enabled", True):
                    results[rule_name] = rule.validate(series)
                else:
                    logger.debug(f"Rule '{rule_name}' skipped (not found or disabled).")

            except Exception as e:
                logger.exception(
                    f"Error applying rule '{rule_name}' for field '{field_def.name}'"
                )
                results[rule_name] = ValidationResult(
                    is_valid=False,
                    error_count=None,
                    error_indices=[],
                    error_messages=[f"Rule error: {str(e)}"],
                    rule_name=rule_name,
                )

        return results

    def _resolve_rule(self, rule_name: str, field_def: FieldDefinition):
        """
        Resolve a validation rule by name/code/alias.

        Parameters
        ----------
        rule_name : str
            The name or code of the rule.
        field_def : FieldDefinition
            Field definition with metadata for parameterized rules.

        Returns
        -------
        BaseValidationRule | None
            The resolved rule instance or None if not found.
        """
        metadata = field_def.metadata or {}
        alias = rule_name.strip().lower()

        # --- 1. Try enum-based resolution ---
        try:
            rule = create_rule_from_code(rule_name.strip(), metadata)
            if rule:
                return rule
        except Exception as e:
            logger.debug(f"Enum resolution failed for '{rule_name}': {e}")

        # --- 2. Try registry lookup ---
        rule = self.rule_registry.get_rule_by_code(
            rule_name
        ) or self.rule_registry.get_rule(rule_name)
        if rule:
            return rule

        # --- 3. Alias-based fallbacks ---
        try:
            if alias in ("minmax", "min/max range"):
                from .validation_rules import MinMaxRule

                return MinMaxRule(
                    min_value=metadata.get("min"),
                    max_value=metadata.get("max"),
                    enabled=True,
                )

            if alias in ("validvalues", "valid values"):
                from .validation_rules import ValidValuesRule

                values = metadata.get("values") or metadata.get("valid_values") or []
                return ValidValuesRule(valid_values=list(values), enabled=True)

            if alias in ("regex", "regex pattern"):
                from .validation_rules import RegexRule

                pattern = metadata.get("pattern")
                if pattern:
                    return RegexRule(pattern=pattern, enabled=True)

            if alias in ("format (email)", "format (phone)"):
                from .validation_rules import FormatRule

                fmt = "email" if "email" in alias else "phone"
                return FormatRule(format_type=fmt, enabled=True)

        except Exception:
            logger.exception(
                f"Failed to construct rule '{rule_name}' for field '{field_def.name}'"
            )

        # --- Not found ---
        logger.debug(
            f"No matching rule found for '{rule_name}' (field '{field_def.name}')"
        )
        return None

    def _calculate_completeness(
        self, series: pd.Series, is_required: bool
    ) -> Tuple[float, float, int, int]:
        """
        Calculate completeness metrics.

        Parameters
        ----------
        series : pd.Series
            Data series to analyze
        is_required : bool
            Whether field is required

        Returns
        -------
        Tuple[float, float, int, int]
            (effective_completeness, factual_completeness, non_missing_count, missing_count)
        """
        total_count = len(series)
        if total_count == 0:
            return 0.0, 0.0, 0, 0

        # Count non-missing values (treat empty strings as missing for string/object types)
        non_missing_mask = series.notna()
        complete_mask = non_missing_mask
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            try:
                non_empty_mask = pd.Series(True, index=series.index)
                non_empty_mask.loc[series[non_missing_mask].index] = series[non_missing_mask].ne("")
                complete_mask = non_missing_mask & non_empty_mask
            except Exception as e:
                logger.debug(f"String completeness handling failed: {e}")
                # Fallback already set: complete_mask = non_missing_mask
                complete_mask = non_missing_mask

        factual_completeness = complete_mask.mean()
        non_missing_count = int(complete_mask.sum())
        missing_count = total_count - non_missing_count

        # Effective completeness: 1.0 if not required, actual ratio if required
        if is_required:
            effective_completeness = factual_completeness
        else:
            effective_completeness = 1.0

        return (
            effective_completeness,
            factual_completeness,
            non_missing_count,
            missing_count,
        )

    def _calculate_validity(
        self, series: pd.Series, validation_results: Dict[str, ValidationResult]
    ) -> float:
        """
        Calculate validity based on validation rule results.

        Parameters
        ----------
        series : pd.Series
            Data series to analyze
        validation_results : Dict[str, ValidationResult]
            Results from validation rules

        Returns
        -------
        float
            Validity score (0.0 to 1.0)
        """
        total_count = len(series)
        if total_count == 0:
            return 0.0

        if not validation_results:
            # No validation rules, assume all values are valid
            return 1.0

        # Collect all error indices
        all_error_indices = set()
        for result in validation_results.values():
            all_error_indices.update(result.error_indices)

        error_count = len(all_error_indices)
        return (total_count - error_count) / total_count

    def _calculate_diversity(
        self, series: pd.Series, is_unique: bool
    ) -> Tuple[float, float, int]:
        """
        Calculate diversity metrics.

        Parameters
        ----------
        series : pd.Series
            Data series to analyze
        is_unique : bool
            Whether field should be unique

        Returns
        -------
        Tuple[float, float]
            (effective_diversity, factual_diversity, unique_count)
        """
        clean_series = series.dropna()
        if clean_series.empty:
            return 0.0, 0.0, 0

        unique_count = clean_series.nunique()
        total_count = len(clean_series)

        factual_diversity = unique_count / total_count

        # Effective diversity: 1.0 if not unique, actual ratio if unique
        if is_unique:
            effective_diversity = factual_diversity
        else:
            effective_diversity = 1.0

        return effective_diversity, factual_diversity, unique_count

    def _calculate_dataset_quality(
        self,
        df: pd.DataFrame,
        schema: SchemaManager,
        column_metrics: Dict[str, ColumnQualityMetrics],
    ) -> DatasetQualityMetrics:
        """
        Calculate dataset-level quality metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset
        schema : SchemaManager
            Schema with field definitions
        column_metrics : Dict[str, ColumnQualityMetrics]
            Per-column quality metrics

        Returns
        -------
        DatasetQualityMetrics
            Dataset-level quality metrics
        """
        required_fields = schema.get_required_fields()
        unique_fields = schema.get_unique_fields()
        total_fields = len(df.columns)

        # Guard: no columns or no computed metrics
        if total_fields == 0 or not column_metrics:
            return DatasetQualityMetrics(
                overall_quality=0.0,
                required_fields_quality=0.0,
                required_fields_completeness=0.0,
                unique_fields_quality=0.0,
                unique_fields_diversity=0.0,
                effective_dataset_completeness=0.0,
                effective_dataset_diversity=0.0,
                factual_dataset_completeness=0.0,
                factual_dataset_validity=0.0,
                factual_dataset_diversity=0.0,
                factual_dataset_consistency=0.0,
                required_fields_count=0,
                unique_fields_count=0,
                required_fields_non_missing_count=0,
                factual_dataset_non_missing_count=0,
                required_fields_missing_count=0,
                factual_dataset_missing_count=0,
                field_unique_values_count=0,
                factual_dataset_unique_values_count=0,
                factual_dataset_diversity_range="0-0%",
                unique_fields_diversity_range="0-0%",
                total_fields_count=0,
                quality_issues=["Dataset has no columns or metrics unavailable"],
                warnings=["Dataset is empty or metrics could not be computed"],
            )

        # --- Weighted average for required fields ---
        # Dataset-level completeness aggregates
        required_fields_completenesses = []
        weighted_sum, total_weight = 0.0, 0.0
        required_fields_non_missing_count = 0
        required_fields_missing_count = 0
        for field_name in required_fields:
            if field_name in column_metrics:
                quality_col = column_metrics[field_name].quality_score
                series = df[field_name]
                weight_col = self._calculate_column_weight(series)
                weighted_sum += quality_col * weight_col
                total_weight += weight_col
                required_fields_completenesses.append(
                    column_metrics[field_name].completeness
                )
                required_fields_non_missing_count += int(
                    column_metrics[field_name].non_missing_count
                )
                required_fields_missing_count += int(
                    column_metrics[field_name].missing_count
                )

        required_fields_quality = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )

        required_fields_completeness = (
            float(np.mean(required_fields_completenesses))
            if required_fields_completenesses
            else 0.0
        )

        # --- Weighted average for unique fields ---
        # Dataset-level diversity aggregates
        unique_fields_diversities = []
        weighted_sum, total_weight = 0.0, 0.0
        field_unique_values_count = 0
        for field_name in unique_fields:
            if field_name in column_metrics:
                quality_col = column_metrics[field_name].quality_score
                series = df[field_name]
                weight_col = self._calculate_column_weight(series)
                weighted_sum += quality_col * weight_col
                total_weight += weight_col
                unique_fields_diversities.append(column_metrics[field_name].diversity)
                field_unique_values_count += int(
                    column_metrics[field_name].unique_count
                )

        unique_fields_quality = weighted_sum / total_weight if total_weight > 0 else 0.0

        unique_fields_diversity = (
            float(np.mean(unique_fields_diversities))
            if unique_fields_diversities
            else 0.0
        )

        # Dataset-level completeness, consistency, validity, diversity aggregates
        effective_dataset_completenesses = []
        effective_dataset_diversities = []
        factual_dataset_completenesses = []
        factual_dataset_consistencies = []
        factual_dataset_validities = []
        factual_dataset_diversities = []

        # --- Weighted average for all fields (overall quality) ---
        weighted_sum, total_weight = 0.0, 0.0
        factual_dataset_non_missing_count = 0
        factual_dataset_missing_count = 0
        factual_dataset_unique_values_count = 0
        for field_name, metrics in column_metrics.items():
            quality_col = metrics.quality_score
            series = df[field_name]
            weight_col = self._calculate_column_weight(series)
            weighted_sum += quality_col * weight_col
            total_weight += weight_col
            # Accumulate for dataset-level metrics
            effective_dataset_completenesses.append(column_metrics[field_name].completeness)
            effective_dataset_diversities.append(column_metrics[field_name].diversity)
            factual_dataset_completenesses.append(column_metrics[field_name].factual_completeness)
            factual_dataset_consistencies.append(column_metrics[field_name].consistency)
            factual_dataset_validities.append(column_metrics[field_name].validity)
            factual_dataset_diversities.append(column_metrics[field_name].factual_diversity)
            factual_dataset_non_missing_count += int(
                column_metrics[field_name].non_missing_count
            )
            factual_dataset_missing_count += int(
                column_metrics[field_name].missing_count
            )
            factual_dataset_unique_values_count += int(
                column_metrics[field_name].unique_count
            )

        overall_quality = weighted_sum / total_weight if total_weight > 0 else 0.0

        effective_dataset_completeness = (
            float(np.mean(effective_dataset_completenesses)) if effective_dataset_completenesses else 0.0
        )

        effective_dataset_diversity = (
            float(np.mean(effective_dataset_diversities)) if effective_dataset_diversities else 0.0
        )

        factual_dataset_completeness = (
            float(np.mean(factual_dataset_completenesses)) if factual_dataset_completenesses else 0.0
        )
        factual_dataset_consistency = (
            float(np.mean(factual_dataset_consistencies)) if factual_dataset_consistencies else 0.0
        )
        factual_dataset_validity = (
            float(np.mean(factual_dataset_validities)) if factual_dataset_validities else 0.0
        )
        factual_dataset_diversity = (
            float(np.mean(factual_dataset_diversities)) if factual_dataset_diversities else 0.0
        )

        # Create range factual dataset diversity string with safe min/max calculation
        if factual_dataset_diversity:
            factual_dataset_diversity_range = self.format_percentage_range(
                factual_dataset_diversities
            )
        else:
            unique_fields_diversity_range = "0-0%"

        # Create range unique fields diversity string with safe min/max calculation
        if unique_fields_diversity:
            unique_fields_diversity_range = self.format_percentage_range(
                unique_fields_diversities
            )
        else:
            unique_fields_diversity_range = "0-0%"

        # Generate quality issues and warnings
        quality_issues = self._generate_dataset_issues(df, column_metrics)
        warnings = self._generate_dataset_warnings(
            required_fields_quality, unique_fields_quality, overall_quality
        )

        return DatasetQualityMetrics(
            overall_quality=overall_quality,
            required_fields_quality=required_fields_quality,
            required_fields_completeness=required_fields_completeness,
            unique_fields_quality=unique_fields_quality,
            unique_fields_diversity=unique_fields_diversity,
            effective_dataset_completeness=effective_dataset_completeness,
            effective_dataset_diversity=effective_dataset_diversity,
            factual_dataset_completeness=factual_dataset_completeness,
            factual_dataset_validity=factual_dataset_validity,
            factual_dataset_diversity=factual_dataset_diversity,
            factual_dataset_consistency=factual_dataset_consistency,
            required_fields_count=len(required_fields),
            unique_fields_count=len(unique_fields),
            required_fields_non_missing_count=required_fields_non_missing_count,
            factual_dataset_non_missing_count=factual_dataset_non_missing_count,
            required_fields_missing_count=required_fields_missing_count,
            factual_dataset_missing_count=factual_dataset_missing_count,
            field_unique_values_count=field_unique_values_count,
            factual_dataset_unique_values_count=factual_dataset_unique_values_count,
            factual_dataset_diversity_range=factual_dataset_diversity_range,
            unique_fields_diversity_range=unique_fields_diversity_range,
            total_fields_count=total_fields,
            quality_issues=quality_issues,
            warnings=warnings,
        )

    def _calculate_column_weight(self, series: pd.Series) -> int:
        """Weight = count of non-null values."""
        return int(series.notna().sum())

    def _generate_column_warnings(
        self,
        series: pd.Series,
        field_def: FieldDefinition,
        validity: float,
        consistency: float,
        factual_completeness: float,
        factual_diversity: float,
        timeliness: Dict[str, Any],
    ) -> List[str]:
        """Generate warnings for column quality issues."""
        warnings = []

        # Low factual completeness warning
        if factual_completeness < 0.5:
            warnings.append(f"High missing data rate ({factual_completeness:.1%})")

        # Low validity warning
        if validity < 0.7:
            warnings.append(f"Low data validity ({validity:.1%})")

        # Low factual diversity warning (for non-unique fields)
        if not field_def.is_unique and factual_diversity < 0.1:
            warnings.append(
                f"Very low diversity ({factual_diversity:.1%}) - possible constant values"
            )

        # High diversity warning (possible ID field)
        if factual_diversity > 0.95 and len(series) > 100:
            warnings.append("Very high diversity - possible ID column")

        # Low consistency warning
        if consistency < 0.6:
            warnings.append(f"Low data consistency ({consistency:.1%})")

        # Timeliness warnings
        freshness = (
            timeliness.get("freshness_score") if isinstance(timeliness, dict) else None
        )
        if freshness is not None and 0.0 < freshness < 0.3:
            days = timeliness.get("days_since_last_update", "unknown")
            warnings.append(f"Outdated data (last updated {days} days ago)")

        return warnings

    def _calculate_consistency(self, series: pd.Series) -> float:
        """Estimate consistency based on type-specific heuristics (0..1)."""
        clean = series.dropna()
        if clean.empty:
            return 0.0

        # Check integer or float dtype
        if pd.api.types.is_integer_dtype(clean) or pd.api.types.is_float_dtype(clean):
            numeric = pd.to_numeric(clean, errors="coerce").dropna()
            if numeric.empty:
                return 0.0
            # For very small samples, avoid unstable statistics
            if len(numeric) < 3:
                return 1.0

            # Decimal places consistency
            def count_decimals(x: float) -> int:
                s = f"{x:.16f}".rstrip("0")
                return len(s.split(".")[1]) if "." in s else 0

            decimals = numeric.map(count_decimals)
            dec_var = decimals.nunique()
            dec_consistency = (
                1.0
                if dec_var <= 1
                else max(0.0, 1.0 - (dec_var - 1) / max(1, len(decimals)))
            )
            # IQR outlier share
            q1, q3 = numeric.quantile([0.25, 0.75])
            if pd.isna(q1) or pd.isna(q3):
                return float(dec_consistency)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                out_consistency = 1.0
            else:
                out_mask = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
                out_consistency = 1.0 - float(out_mask.mean())
            return float((dec_consistency + out_consistency) / 2)

        if pd.api.types.is_datetime64_any_dtype(clean) or pd.api.types.is_bool_dtype(
            clean
        ):
            return 1.0

        # Strings/objects
        s = clean.astype(str)
        lengths = s.str.len()
        mean_len = lengths.mean()
        std_len = lengths.std()
        length_consistency = 1.0 - min(
            (std_len / mean_len) if mean_len > 0 else 1.0, 1.0
        )
        same_case = (
            s.str.isupper().all() or s.str.islower().all() or s.str.istitle().all()
        )
        case_consistency = 1.0 if same_case else 0.5
        return float((length_consistency + case_consistency) / 2)

    def _calculate_timeliness(self, series: pd.Series) -> Dict[str, Any]:
        """Timeliness for datetime columns: last update, age, freshness score."""
        result = {
            "last_updated": None,
            "days_since_last_update": None,
            "freshness_score": None,
            "temporal_coverage": None,
        }

        # --- 1. Ensure datetime dtype using safe auto-cast ---
        if not pd.api.types.is_datetime64_any_dtype(series):
            converted = self.auto_cast_series_for_datetime(series)
            if converted is None or converted.notna().sum() == 0:
                return result
            if converted.isna().sum() > 0:
                logger.warning(
                    f"Timeliness calculation: {converted.isna().sum()} invalid datetime values coerced to NaT"
                )
            series = converted

        # --- 2. Drop NaT and calculate ---
        clean = series.dropna()
        if clean.empty:
            return result

        latest = pd.to_datetime(clean.max(), utc=True)
        earliest = pd.to_datetime(clean.min(), utc=True)
        now = pd.Timestamp.now(tz="UTC")

        if pd.notna(latest):
            result["last_updated"] = latest.isoformat()
            days = (now - latest).days
            result["days_since_last_update"] = days

            # Freshness scoring
            if days <= 1:
                freshness = 1.0
            elif days <= 7:
                freshness = 0.8
            elif days <= 30:
                freshness = 0.6
            elif days <= 90:
                freshness = 0.4
            elif days <= 365:
                freshness = 0.2
            else:
                freshness = 0.1
            result["freshness_score"] = freshness

        if pd.notna(earliest) and pd.notna(latest):
            result["temporal_coverage"] = f"{(latest - earliest).days} days"

        return result

    def auto_cast_series_for_datetime(self, series: pd.Series) -> Optional[pd.Series]:
        """
        Try to cast a Series to datetime if it looks like datetime.
        Avoids converting numeric-like series (e.g., prices).
        Supports both 4-digit and 2-digit years.
        Returns converted Series or None if not suitable.
        """
        # --- Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        # --- Only consider string/object dtypes
        if not (
            pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        ):
            return None

        # --- Heuristic 1: If mostly numeric, skip
        numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
        if numeric_ratio > 0.8:
            return None

        # --- Heuristic 2: Quick regex check for date-like strings
        sample_values = series.dropna().astype(str).head(50)
        if not any(DATE_PATTERN.search(v) for v in sample_values):
            return None

        # --- Try conversion: First strict ISO (YYYY-MM-DD)
        converted = pd.to_datetime(series, errors="coerce")

        if converted.notna().sum() > 0:
            return converted

        # --- Fallback: try with 2-digit year handling
        try:
            converted = pd.to_datetime(series, errors="coerce", format="%d-%m-%y")
            if converted.notna().sum() > 0:
                return converted
        except Exception:
            pass

        return None

    def format_percentage_range(self, values, decimals: int = 1) -> str:
        """Format min-max range from list of ratio values ​​(0–1) into percentage form."""
        if not values:
            return "0-0%"

        min_val = round(min(values) * 100, decimals)
        max_val = round(max(values) * 100, decimals)

        if min_val == max_val:
            return f"{min_val:.{decimals}f}%"
        return f"{min_val:.{decimals}f}-{max_val:.{decimals}f}%"

    def _generate_dataset_issues(
        self,
        df: pd.DataFrame,
        column_metrics: Dict[str, ColumnQualityMetrics],
    ) -> List[str]:
        """Generate dataset-level quality issues."""
        issues = []

        # Check for high missing data
        total_missing = df.isnull().sum().sum()
        missing_rate = total_missing / df.size
        if missing_rate > 0.3:
            issues.append(f"High overall missing data rate ({missing_rate:.1%})")

        # Check for small dataset
        if len(df) < 10:
            issues.append("Dataset is very small - quality metrics may be unreliable")

        # Check for many columns with issues
        problematic_columns = sum(
            1
            for metrics in column_metrics.values()
            if metrics.validity < 0.7 or metrics.completeness < 0.5
        )
        if problematic_columns > len(df.columns) * 0.5:
            issues.append(
                f"Many columns have quality issues ({problematic_columns}/{len(df.columns)})"
            )

        return issues

    def _generate_dataset_warnings(
        self,
        required_fields_quality: float,
        unique_fields_quality: float,
        overall_quality: float,
    ) -> List[str]:
        """Generate dataset-level warnings."""
        warnings = []

        # Warning about quality difference
        if abs(required_fields_quality - overall_quality) > 0.1:
            warnings.append(
                f"Quality based on required fields ({required_fields_quality:.1f}%) differs from "
                f"overall quality across all columns ({overall_quality:.1f}%)"
            )

        # Warning about quality difference
        if abs(unique_fields_quality - overall_quality) > 0.1:
            warnings.append(
                f"Quality based on unique fields ({unique_fields_quality:.1f}%) differs from "
                f"overall quality across all columns ({overall_quality:.1f}%)"
            )

        return warnings

    def _prepare_ui_output(
        self,
        df: pd.DataFrame,
        schema: SchemaManager,
        column_metrics: Dict[str, ColumnQualityMetrics],
        dataset_metrics: Optional[DatasetQualityMetrics],
        analyze_scope: str = "dataset",
        analyzed_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
            Prepare UI-friendly output with structured data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset
            schema : SchemaManager
                Schema with field definitions
            column_metrics : Dict[str, ColumnQualityMetrics]
                Per-column quality metrics
            dataset_metrics : DatasetQualityMetrics
                Dataset-level quality metrics

        Returns
        -------
            Dict[str, Any]
                UI-friendly structured output
        """
        # Prepare schema view (per-column details)
        schema_view = {}
        iter_columns = (
            analyzed_columns if analyzed_columns is not None else list(df.columns)
        )
        for column in iter_columns:
            field_def = schema.get_field(column)
            metrics = column_metrics[column]

            schema_view[column] = {
                "data_type": field_def.data_type if field_def else "unknown",
                "is_required": field_def.is_required if field_def else False,
                "is_unique": field_def.is_unique if field_def else False,
                "validation_rules": field_def.validation_rules if field_def else [],
                "completeness": round(metrics.completeness, 4),
                "validity": round(metrics.validity, 4),
                "diversity": round(metrics.diversity, 4),
                "consistency": round(metrics.consistency, 4),
                "quality_score": round(metrics.quality_score, 2),
                "factual_completeness": round(metrics.factual_completeness, 4),
                "factual_diversity": round(metrics.factual_diversity, 4),
                "non_missing_count": metrics.non_missing_count,
                "missing_count": metrics.missing_count,
                "unique_count": metrics.unique_count,
                "error_count": metrics.error_count,
                "total_count": metrics.total_count,
                "warnings": metrics.warnings,
                "timeliness": metrics.timeliness,
                "validation_results": {
                    rule_name: {
                        "is_valid": result.is_valid,
                        "error_count": result.error_count,
                        "error_messages": result.error_messages[:5],  # Limit for UI
                    }
                    for rule_name, result in metrics.validation_results.items()
                },
            }

        # Prepare dataset card (overall metrics)
        if dataset_metrics is not None:
            dataset_card = {
                "overall_quality": float(round(dataset_metrics.overall_quality, 2)),
                "required_fields_quality": float(
                    round(dataset_metrics.required_fields_quality, 2)
                ),
                "required_fields_completeness": float(
                    round(dataset_metrics.required_fields_completeness, 4)
                ),
                "unique_fields_quality": float(
                    round(dataset_metrics.unique_fields_quality, 2)
                ),
                "unique_fields_diversity": float(
                    round(dataset_metrics.unique_fields_diversity, 4)
                ),
                "effective_dataset_completeness": float(
                    round(dataset_metrics.effective_dataset_completeness, 4)
                ),
                "effective_dataset_diversity": float(
                    round(dataset_metrics.effective_dataset_diversity, 4)
                ),
                "factual_dataset_completeness": float(
                    round(dataset_metrics.factual_dataset_completeness, 4)
                ),
                "factual_dataset_validity": float(
                    round(dataset_metrics.factual_dataset_validity, 4)
                ),
                "factual_dataset_diversity": float(
                    round(dataset_metrics.factual_dataset_diversity, 4)
                ),
                "factual_dataset_consistency": float(
                    round(dataset_metrics.factual_dataset_consistency, 4)
                ),
                "required_fields_count": int(dataset_metrics.required_fields_count),
                "unique_fields_count": int(dataset_metrics.unique_fields_count),
                "required_fields_non_missing_count": int(
                    dataset_metrics.required_fields_non_missing_count
                ),
                "factual_dataset_non_missing_count": int(
                    dataset_metrics.factual_dataset_non_missing_count
                ),
                "required_fields_missing_count": int(
                    dataset_metrics.required_fields_missing_count
                ),
                "factual_dataset_missing_count": int(
                    dataset_metrics.factual_dataset_missing_count
                ),
                "unique_fields_diversity_range": dataset_metrics.unique_fields_diversity_range,
                "factual_dataset_diversity_range": dataset_metrics.factual_dataset_diversity_range,
                "field_unique_values_count": int(
                    dataset_metrics.field_unique_values_count
                ),
                "factual_dataset_unique_values_count": int(
                    dataset_metrics.factual_dataset_unique_values_count
                ),
                "total_fields_count": int(dataset_metrics.total_fields_count),
                "required_fields_quality_summary": f"Required fields quality: {dataset_metrics.required_fields_quality:.1f}% (across {dataset_metrics.required_fields_count} required fields)",
                "unique_fields_quality_summary": f"Unique fields quality: {dataset_metrics.unique_fields_quality:.1f}% (across {dataset_metrics.unique_fields_count} unique fields)",
                "overall_quality_summary": f"Overall quality across all columns: {dataset_metrics.overall_quality:.1f}%",
                "warnings": dataset_metrics.warnings,
                "partial": False,
            }
        else:
            # Partial result (per-column run). Provide a minimal dataset_card.
            dataset_card = {
                "overall_quality": None,
                "required_fields_quality": None,
                "required_fields_completeness": None,
                "unique_fields_quality": None,
                "unique_fields_diversity": None,
                "effective_dataset_completeness": None,
                "effective_dataset_diversity": None,
                "factual_dataset_completeness": None,
                "factual_dataset_validity": None,
                "factual_dataset_diversity": None,
                "factual_dataset_consistency": None,
                "required_fields_count": None,
                "unique_fields_count": None,
                "required_fields_non_missing_count": None,
                "factual_dataset_non_missing_count": None,
                "required_fields_missing_count": None,
                "factual_dataset_missing_count": None,
                "unique_fields_diversity_range": None,
                "factual_dataset_diversity_range": None,
                "field_unique_values_count": None,
                "factual_dataset_unique_values_count": None,
                "total_fields_count": len(df.columns),
                "required_fields_quality_summary": "Partial required fields analysis",
                "unique_fields_quality_summary": "Partial unique fields analysis",
                "overall_quality_summary": "Run full analysis to compute dataset-level metrics",
                "warnings": [],
                "partial": True,
            }

        return {
            "schema_view": schema_view,
            "dataset_card": dataset_card,
            "quality_issues": (
                dataset_metrics.quality_issues if dataset_metrics is not None else []
            ),
            "metadata": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "analyze_scope": analyze_scope,
                "analyzed_columns": analyzed_columns,
                "weights": {
                    "completeness": self.weights.completeness,
                    "validity": self.weights.validity,
                    "diversity": self.weights.diversity,
                },
            },
        }

    def _empty_dataset_result(self) -> Dict[str, Any]:
        """Return result for empty dataset."""
        return {
            "schema_view": {},
            "dataset_card": {
                "overall_quality": 0.0,
                "required_fields_quality": 0.0,
                "required_fields_completeness": 0.0,
                "unique_fields_quality": 0.0,
                "unique_fields_diversity": 0.0,
                "effective_dataset_completeness": 0.0,
                "effective_dataset_diversity": 0.0,
                "factual_dataset_completeness": 0.0,
                "factual_dataset_validity": 0.0,
                "factual_dataset_diversity": 0.0,
                "factual_dataset_consistency": 0.0,
                "required_fields_count": 0,
                "unique_fields_count": 0,
                "required_fields_non_missing_count": 0,
                "factual_dataset_non_missing_count": 0,
                "required_fields_missing_count": 0,
                "factual_dataset_missing_count": 0,
                "unique_fields_diversity_range": "0-0%",
                "factual_dataset_diversity_range": "0-0%",
                "field_unique_values_count": 0,
                "factual_dataset_unique_values_count": 0,
                "total_fields_count": 0,
                "required_fields_quality_summary": "0.0% (required fields empty dataset)",
                "overall_quality_summary": "Dataset is empty",
                "warnings": ["Dataset is empty"],
            },
            "quality_issues": ["Dataset is empty"],
            "metadata": {
                "total_rows": 0,
                "total_columns": 0,
                "weights": {
                    "completeness": self.weights.completeness,
                    "validity": self.weights.validity,
                    "diversity": self.weights.diversity,
                },
            },
        }
