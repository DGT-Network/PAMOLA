"""
PAMOLA.CORE - Information Loss Metrics
--------------------------------------
This module provides metrics for quantifying information loss in anonymized
or synthetic datasets. Information loss metrics measure how much utility
has been sacrificed to achieve privacy protection.

Key features:
- Comprehensive measures of information loss across different data types
- Support for numerical, categorical, and mixed-type data
- Column-level and dataset-level loss assessments
- Configurable weighting for different aspects of information loss

These metrics help data custodians balance privacy protection with
data utility when applying anonymization techniques.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

from core.metrics.base import UtilityMetric, round_metric_values

# Configure logging
logger = logging.getLogger(__name__)


class InformationLossMetric(UtilityMetric):
    """
    Calculates information loss metrics for anonymized data.

    This class measures various aspects of information loss:
    - Record loss due to suppression
    - Generalization loss for numerical attributes
    - Diversity loss for categorical attributes

    The loss values are expressed as percentages, with higher values
    indicating more information loss (lower utility).
    """

    def __init__(self,
                 record_weight: float = 0.4,
                 numerical_weight: float = 0.3,
                 categorical_weight: float = 0.3):
        """
        Initialize the information loss metric.

        Parameters:
        -----------
        record_weight : float, optional
            Weight for record loss in overall calculation (default: 0.4).
        numerical_weight : float, optional
            Weight for numerical attribute loss in overall calculation (default: 0.3).
        categorical_weight : float, optional
            Weight for categorical attribute loss in overall calculation (default: 0.3).
        """
        super().__init__(
            name="Information Loss",
            description="Measures the loss of information after anonymization"
        )
        self.record_weight = record_weight
        self.numerical_weight = numerical_weight
        self.categorical_weight = categorical_weight

        # Ensure weights sum to 1
        total_weight = record_weight + numerical_weight + categorical_weight
        if abs(total_weight - 1.0) > 1e-10:
            self.record_weight /= total_weight
            self.numerical_weight /= total_weight
            self.categorical_weight /= total_weight

    def calculate(self, original_data: pd.DataFrame,
                  anonymized_data: pd.DataFrame,
                  numerical_columns: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate information loss metrics.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset.
        anonymized_data : pd.DataFrame
            The anonymized dataset.
        numerical_columns : list[str], optional
            List of numerical columns to use for information loss calculation.
            If None, all numeric columns will be used.
        **kwargs : dict
            Additional parameters for calculation.

        Returns:
        --------
        dict
            Dictionary with information loss metrics:
            - "total_records_loss": Percentage of records lost
            - "avg_generalization_loss": Average precision loss for numeric attributes
            - "categorical_diversity_loss": Reduction in categorical diversity
            - "overall_information_loss": Weighted overall information loss
            - "column_level_loss": Column-by-column information loss
        """
        logger.info("Calculating information loss metrics")

        if original_data is None or anonymized_data is None:
            raise ValueError("Both original and anonymized datasets are required")

        try:
            # Calculate record loss
            record_loss = 100 * (1 - len(anonymized_data) / len(original_data)) if len(original_data) > 0 else 0

            # Get numeric columns if not specified
            if numerical_columns is None:
                numerical_columns = original_data.select_dtypes(include=['number']).columns.tolist()

            # Calculate column-level loss
            column_level_loss = {}

            # Calculate generalization loss for numeric columns
            generalization_loss = 0
            numeric_loss_values = []

            if numerical_columns:
                # Ensure we only use columns present in both datasets
                valid_numeric_columns = [col for col in numerical_columns
                                         if col in anonymized_data.columns and col in original_data.columns]

                if valid_numeric_columns:
                    # Calculate variance for each column
                    orig_var = original_data[valid_numeric_columns].var()
                    anon_var = anonymized_data[valid_numeric_columns].var()

                    # Compute loss percentage, handling zeros properly
                    for col in valid_numeric_columns:
                        if orig_var[col] > 0:
                            col_loss = 100 * (1 - anon_var[col] / orig_var[col])
                        else:
                            col_loss = 0

                        numeric_loss_values.append(col_loss)
                        column_level_loss[col] = {
                            "type": "numeric",
                            "loss_percentage": col_loss
                        }

                    # Average across columns
                    generalization_loss = np.mean(numeric_loss_values) if numeric_loss_values else 0

            # Calculate diversity loss for categorical columns
            categorical_columns = [col for col in original_data.columns if col not in numerical_columns]
            categorical_loss = 0
            categorical_loss_values = []

            if categorical_columns:
                # Ensure we only use columns present in both datasets
                valid_cat_columns = [col for col in categorical_columns
                                     if col in anonymized_data.columns and col in original_data.columns]

                if valid_cat_columns:
                    # Calculate unique value counts for each column
                    for col in valid_cat_columns:
                        orig_unique = original_data[col].nunique()
                        anon_unique = anonymized_data[col].nunique()

                        if orig_unique > 0:
                            col_loss = 100 * (1 - anon_unique / orig_unique)
                        else:
                            col_loss = 0

                        categorical_loss_values.append(col_loss)
                        column_level_loss[col] = {
                            "type": "categorical",
                            "loss_percentage": col_loss
                        }

                    # Average across columns
                    categorical_loss = np.mean(categorical_loss_values) if categorical_loss_values else 0

            # Calculate overall information loss (weighted average)
            overall_loss = (self.record_weight * record_loss +
                            self.numerical_weight * generalization_loss +
                            self.categorical_weight * categorical_loss)

            # Prepare result
            result = {
                "total_records_loss": record_loss,
                "avg_generalization_loss": generalization_loss,
                "categorical_diversity_loss": categorical_loss,
                "overall_information_loss": overall_loss,
                "column_level_loss": column_level_loss,
                "weights": {
                    "record_weight": self.record_weight,
                    "numerical_weight": self.numerical_weight,
                    "categorical_weight": self.categorical_weight
                }
            }

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"Information loss analysis: Overall loss = {overall_loss:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Error during information loss calculation: {e}")
            raise

    def interpret(self, value: float) -> str:
        """
        Interpret an information loss value.

        For information loss, lower values indicate better utility
        (less information was lost).

        Parameters:
        -----------
        value : float
            The information loss value (percentage).

        Returns:
        --------
        str
            Human-readable interpretation of the information loss.
        """
        if value < 10:
            return f"Information Loss: {value:.2f}% - Minimal loss, excellent utility preserved"
        elif value < 20:
            return f"Information Loss: {value:.2f}% - Low loss, good utility preserved"
        elif value < 40:
            return f"Information Loss: {value:.2f}% - Moderate loss, acceptable utility for many use cases"
        elif value < 60:
            return f"Information Loss: {value:.2f}% - Significant loss, limited utility remains"
        else:
            return f"Information Loss: {value:.2f}% - Severe loss, very limited utility remains"


class GeneralizationLossMetric(UtilityMetric):
    """
    Specialized metric for measuring information loss due to generalization.

    This metric focuses on how generalization techniques (like binning numeric
    values or generalizing categories) affect the precision and granularity
    of the data.
    """

    def __init__(self):
        """
        Initialize the generalization loss metric.
        """
        super().__init__(
            name="Generalization Loss",
            description="Measures information loss due to generalization of attribute values"
        )

    def calculate(self, original_data: pd.DataFrame,
                  anonymized_data: pd.DataFrame,
                  columns: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate generalization loss metrics.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset.
        anonymized_data : pd.DataFrame
            The anonymized dataset with generalized values.
        columns : list[str], optional
            List of columns to evaluate for generalization loss.
            If None, all common columns will be used.
        **kwargs : dict
            Additional parameters for calculation.

        Returns:
        --------
        dict
            Dictionary with generalization loss metrics:
            - "overall_generalization_loss": Average loss across all columns
            - "column_level_loss": Loss for each individual column
            - "worst_column": Column with the highest generalization loss
        """
        logger.info("Calculating generalization loss metrics")

        try:
            # Determine columns to analyze
            if columns is None:
                columns = [col for col in original_data.columns
                           if col in anonymized_data.columns]
            else:
                columns = [col for col in columns
                           if col in original_data.columns and col in anonymized_data.columns]

            if not columns:
                raise ValueError("No valid columns for generalization loss calculation")

            # Calculate column-level generalization loss
            column_loss = {}
            loss_values = []

            for col in columns:
                # Different calculation methods based on column type
                if np.issubdtype(original_data[col].dtype, np.number):
                    # Numeric column: compare variance
                    orig_var = original_data[col].var()
                    anon_var = anonymized_data[col].var()

                    if orig_var > 0:
                        col_loss = 100 * (1 - anon_var / orig_var)
                    else:
                        col_loss = 0

                    # Additional metric: range preservation
                    orig_range = original_data[col].max() - original_data[col].min()
                    anon_range = anonymized_data[col].max() - anonymized_data[col].min()

                    range_preservation = 100
                    if orig_range > 0:
                        range_preservation = 100 * (anon_range / orig_range)

                    column_loss[col] = {
                        "type": "numeric",
                        "loss_percentage": col_loss,
                        "range_preservation": min(100, range_preservation)
                    }
                else:
                    # Categorical column: compare unique values
                    orig_unique = original_data[col].nunique()
                    anon_unique = anonymized_data[col].nunique()

                    if orig_unique > 0:
                        col_loss = 100 * (1 - anon_unique / orig_unique)
                    else:
                        col_loss = 0

                    # Additional metric: value distribution
                    orig_entropy = calculate_entropy(original_data[col])
                    anon_entropy = calculate_entropy(anonymized_data[col])

                    entropy_ratio = 100
                    if orig_entropy > 0:
                        entropy_ratio = 100 * (anon_entropy / orig_entropy)

                    column_loss[col] = {
                        "type": "categorical",
                        "loss_percentage": col_loss,
                        "entropy_preservation": min(100, entropy_ratio)
                    }

                loss_values.append(col_loss)

            # Calculate overall generalization loss
            overall_loss = np.mean(loss_values) if loss_values else 0

            # Find worst column (highest loss)
            worst_column = None
            if column_loss:
                worst_column = max(column_loss.items(), key=lambda x: x[1]["loss_percentage"])[0]

            # Prepare result
            result = {
                "overall_generalization_loss": overall_loss,
                "column_level_loss": column_loss,
                "worst_column": worst_column
            }

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"Generalization loss analysis: Overall loss = {overall_loss:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Error during generalization loss calculation: {e}")
            raise


class SuppressionLossMetric(UtilityMetric):
    """
    Specialized metric for measuring information loss due to suppression.

    This metric focuses on the impact of removing records or values
    as part of the anonymization process.
    """

    def __init__(self):
        """
        Initialize the suppression loss metric.
        """
        super().__init__(
            name="Suppression Loss",
            description="Measures information loss due to suppression of records or values"
        )

    def calculate(self, original_data: pd.DataFrame,
                  anonymized_data: pd.DataFrame,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate suppression loss metrics.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset.
        anonymized_data : pd.DataFrame
            The anonymized dataset after suppression.
        **kwargs : dict
            Additional parameters for calculation.

        Returns:
        --------
        dict
            Dictionary with suppression loss metrics:
            - "records_suppressed": Number of records suppressed
            - "record_suppression_rate": Percentage of records suppressed
            - "values_suppressed": Number of individual values suppressed
            - "value_suppression_rate": Percentage of individual values suppressed
        """
        logger.info("Calculating suppression loss metrics")

        try:
            # Calculate record suppression
            records_suppressed = len(original_data) - len(anonymized_data)
            record_suppression_rate = 100 * records_suppressed / len(original_data) if len(original_data) > 0 else 0

            # Calculate value suppression (for cell-level suppression)
            values_suppressed = 0
            total_values = 0

            # Count missing values in both datasets
            common_columns = [col for col in original_data.columns if col in anonymized_data.columns]

            for col in common_columns:
                # Count original non-null values
                orig_non_null = original_data[col].notna().sum()
                total_values += orig_non_null

                # Count anonymized non-null values
                anon_non_null = anonymized_data[col].notna().sum()

                # Calculate difference (suppressed values)
                values_suppressed += max(0, orig_non_null - anon_non_null)

            # Calculate suppression rates
            value_suppression_rate = 100 * values_suppressed / total_values if total_values > 0 else 0

            # Prepare result
            result = {
                "records_suppressed": int(records_suppressed),
                "record_suppression_rate": record_suppression_rate,
                "values_suppressed": int(values_suppressed),
                "value_suppression_rate": value_suppression_rate
            }

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"Suppression loss analysis: Record suppression rate = {record_suppression_rate:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Error during suppression loss calculation: {e}")
            raise


# Helper function for calculating entropy of a series
def calculate_entropy(series: pd.Series) -> float:
    """
    Calculate the Shannon entropy of a series.

    Parameters:
    -----------
    series : pd.Series
        The series to calculate entropy for.

    Returns:
    --------
    float
        The entropy value.
    """
    # Remove NaN values
    series = series.dropna()

    if len(series) == 0:
        return 0

    # Calculate value frequencies
    value_counts = series.value_counts(normalize=True)

    # Calculate entropy
    entropy = -np.sum(value_counts * np.log2(value_counts))

    return entropy


# Convenience function for calculating all information loss metrics
def calculate_information_loss_metrics(original_data: pd.DataFrame,
                                       anonymized_data: pd.DataFrame,
                                       **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Calculate multiple information loss metrics for anonymized data.

    Parameters:
    -----------
    original_data : pd.DataFrame
        The original dataset.
    anonymized_data : pd.DataFrame
        The anonymized dataset.
    **kwargs : dict
        Additional parameters for calculation.

    Returns:
    --------
    dict
        Dictionary with results from all information loss metrics.
    """
    results = {}

    # Calculate general information loss
    info_loss = InformationLossMetric(
        record_weight=kwargs.get('record_weight', 0.4),
        numerical_weight=kwargs.get('numerical_weight', 0.3),
        categorical_weight=kwargs.get('categorical_weight', 0.3)
    )
    results["information_loss"] = info_loss.calculate(
        original_data, anonymized_data, kwargs.get('numerical_columns')
    )

    # Calculate generalization loss
    gen_loss = GeneralizationLossMetric()
    results["generalization_loss"] = gen_loss.calculate(
        original_data, anonymized_data, kwargs.get('columns')
    )

    # Calculate suppression loss
    supp_loss = SuppressionLossMetric()
    results["suppression_loss"] = supp_loss.calculate(
        original_data, anonymized_data
    )

    return results