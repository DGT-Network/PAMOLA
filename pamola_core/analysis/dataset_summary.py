"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Dataset Summary Analyzer
Package:       pamola_core.analysis
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides dataset summarization utilities including type detection, missing value
  analysis and outlier detection using IQR. Designed for robust error handling and
  numeric coercion for object-like columns.

Key Features:
  - Automatic numeric-like detection with configurable threshold
  - Missing values summary and per-field counts
  - Outlier detection using IQR
  - Backward compatible function wrapper
  - Detailed logging and safe fallbacks

Dependencies:
  - pandas - DataFrame operations
  - typing - type hints
  - dataclasses - structured results
  - pamola_core.profiling.commons.statistical_analysis - outlier detection
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
from dataclasses import dataclass

# Use standardized outlier detection (IQR)
from pamola_core.profiling.commons.statistical_analysis import detect_outliers_iqr


@dataclass
class DatasetSummary:
    """Data class for structured dataset summary results."""

    rows: int
    columns: int
    missing_values: Dict
    numeric_fields: Dict
    categorical_fields: Dict
    outliers: Dict


class DatasetAnalyzer:
    """Enhanced dataset analyzer with improved error handling and performance."""

    def __init__(
        self, numeric_threshold: float = 0.75, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the analyzer.

        Args:
            numeric_threshold: Minimum ratio of convertible values to consider column numeric-like
            logger: Optional logger instance
        """
        self.numeric_threshold = max(
            0.0, min(1.0, numeric_threshold)
        )  # Clamp between 0-1
        self.logger = logger or logging.getLogger(__name__)

    def _analyze_field_types(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, pd.Series]]:
        """
        Analyze and categorize field types with improved numeric detection.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (numeric_cols, categorical_cols, coerced_numeric_series)
        """
        try:
            # Get native numeric columns
            native_numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            # Get object/category columns for potential numeric coercion
            obj_cat_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            numeric_like_cols = []
            coerced_numeric_series = {}

            # Analyze object/category columns for numeric-like content
            for col in obj_cat_cols:
                if col not in df.columns:  # Safety check
                    continue

                try:
                    # Convert to numeric with error handling
                    converted = pd.to_numeric(df[col], errors="coerce")

                    # Check if we have valid conversions
                    if converted.notna().sum() == 0:
                        continue

                    # Calculate conversion ratio only for non-null original values
                    original_non_null = df[col].notna().sum()
                    if original_non_null == 0:
                        continue

                    conversion_ratio = converted.notna().sum() / original_non_null

                    if conversion_ratio >= self.numeric_threshold:
                        numeric_like_cols.append(col)
                        coerced_numeric_series[col] = converted

                except Exception as e:
                    self.logger.warning(
                        f"Error analyzing column '{col}' for numeric conversion: {e}"
                    )
                    continue

            numeric_cols = native_numeric_cols + numeric_like_cols
            categorical_cols = [c for c in obj_cat_cols if c not in numeric_like_cols]

            return numeric_cols, categorical_cols, coerced_numeric_series

        except Exception as e:
            self.logger.error(f"Error in field type analysis: {e}")
            # Fallback to basic type detection
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            return numeric_cols, categorical_cols, {}

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        coerced_numeric_series: Dict[str, pd.Series],
    ) -> Tuple[int, List[str]]:
        """
        Detect outliers in numeric columns with robust error handling.

        Args:
            df: Input DataFrame
            numeric_cols: List of numeric column names
            coerced_numeric_series: Dictionary of coerced numeric series

        Returns:
            Tuple of (outlier_count, outlier_fields)
        """
        outlier_count = 0
        outlier_fields = []

        for col in numeric_cols:
            try:
                # Use coerced series if available, otherwise original
                base_series = coerced_numeric_series.get(col, df[col])
                series = base_series.dropna()

                if len(series) < 3:  # Need at least 3 points for IQR
                    continue

                result = detect_outliers_iqr(series)
                count = result.get("count", 0)

                if isinstance(count, (int, float)) and count > 0:
                    outlier_count += int(count)
                    outlier_fields.append(col)

            except Exception as e:
                self.logger.warning(f"Error detecting outliers in column '{col}': {e}")
                continue

        return outlier_count, outlier_fields

    def analyze_dataset_summary(
        self,
        df: pd.DataFrame,
    ) -> Dict:
        """
        Analyze dataset and return comprehensive summary with improved error handling.

        Args:
            df: Input DataFrame to analyze

        Returns:
            Dictionary containing comprehensive dataset analysis

        Raises:
            ValueError: If DataFrame is invalid
        """
        try:
            rows, cols = df.shape
            total_cells = rows * cols

            # Field type analysis
            numeric_cols, categorical_cols, coerced_numeric_series = (
                self._analyze_field_types(df)
            )

            # Missing values analysis (with safe calculation)
            try:
                missing_vals = int(df.isna().sum().sum())
                fields_missing = int((df.isna().sum() > 0).sum())
            except Exception as e:
                self.logger.warning(f"Error calculating missing values: {e}")
                missing_vals = 0
                fields_missing = 0

            # Outlier detection
            outlier_count, outlier_fields = self._detect_outliers(
                df, numeric_cols, coerced_numeric_series
            )

            return {
                "rows": rows,
                "columns": cols,
                "total_cells": total_cells,
                "missing_values": {
                    "value": missing_vals,
                    "fields_with_missing": fields_missing,
                },
                "numeric_fields": {
                    "count": len(numeric_cols),
                    "percentage": len(numeric_cols) / cols if cols > 0 else 0.0,
                },
                "categorical_fields": {
                    "count": len(categorical_cols),
                    "percentage": len(categorical_cols) / cols if cols > 0 else 0.0,
                },
                "outliers": {
                    "count": outlier_count,
                    "affected_fields": outlier_fields,
                },
            }

        except Exception as e:
            self.logger.error(f"Critical error in dataset analysis: {e}")
            raise


# Convenience function to maintain backward compatibility
def analyze_dataset_summary(
    df: pd.DataFrame,
) -> Dict:
    """
    Backward compatible function wrapper.

    Args:
        df: Input DataFrame to analyze

    Returns:
        Dictionary containing dataset analysis results
    """
    analyzer = DatasetAnalyzer()
    return analyzer.analyze_dataset_summary(df)
