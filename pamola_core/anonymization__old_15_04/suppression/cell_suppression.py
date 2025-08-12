"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Cell Suppression Processor
-----------------------------------
This module provides an implementation of **cell-level suppression**
for anonymization-preserving data transformations. It extends the
BaseSuppressionProcessor and enables targeted removal or masking
of specific cell values based on predefined rules.

Cell suppression ensures that sensitive or rare values are hidden
without completely removing entire attributes or records.

Common use cases include:
- Suppressing values in low-count categories to prevent inference attacks.
- Redacting PII (Personally Identifiable Information) at the cell level.
- Ensuring compliance with statistical disclosure control policies.

Features:
- **Threshold-based suppression**: Suppresses rare values below a frequency threshold.
- **Custom rule suppression**: Allows user-defined rules for targeted cell suppression.
- **Masking instead of deletion**: Replaces suppressed values with `"MASKED"` or `NaN`.

NOTE: This module requires `pandas` as a dependency.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
from abc import ABC
from pamola_core.anonymization__old_15_04.suppression.base import BaseSuppressionProcessor

class CellSuppressionProcessor(BaseSuppressionProcessor, ABC):
    """
    Cell Suppression Processor for targeted value suppression in datasets.

    This class extends BaseSuppressionProcessor and provides techniques
    for suppressing individual cell values based on predefined conditions.

    Methods:
    --------
    - suppress(): Suppresses values in specified columns based on thresholds or rules.
    """

    def __init__(self, suppression_threshold: int = None, mask_value: str = "MASKED"):
        """
        Initializes the cell suppression processor.

        Parameters:
        -----------
        suppression_threshold : int, optional
            The minimum frequency required for a value to remain in the dataset
            (default: None, meaning no threshold-based suppression).
        mask_value : str, optional
            The value to replace suppressed cells with (default: "MASKED").
        """
        self.suppression_threshold = suppression_threshold
        self.mask_value = mask_value

    def suppress(self, data: pd.DataFrame, columns: list[str] = None, condition: callable = None, rule_based: dict = None, **kwargs) -> pd.DataFrame:
        """
        Apply cell suppression to specified columns based on thresholds, conditions, or custom rules.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing values to be suppressed.
        columns : list[str], optional
            List of column names where suppression should be applied.
        condition : callable, optional
            A function that determines which records to suppress.
        rule_based : dict, optional
            A dictionary defining custom suppression rules per column, e.g.,
            `{"age": lambda x: x < 18, "income": lambda x: x > 1000000}`.
        kwargs : dict
            Additional parameters for suppression.

        Returns:
        --------
        pd.DataFrame
            The dataset with suppressed values.
        """
        if columns is None:
            columns = data.columns.tolist()

        for column in columns:
            if self.suppression_threshold is not None:
                data = self._apply_threshold_suppression(data, column)

            if condition:
                data[column] = data[column].apply(lambda x: self.mask_value if condition(x) else x)

            if rule_based and column in rule_based:
                data[column] = data[column].apply(lambda x: self.mask_value if rule_based[column](x) else x)

        return data

    def _apply_threshold_suppression(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply suppression to values that appear less frequently than the defined threshold.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing values to be suppressed.
        column : str
            The column name where threshold-based suppression should be applied.

        Returns:
        --------
        pd.DataFrame
            The dataset with suppressed values.
        """
        value_counts = data[column].value_counts()
        rare_values = value_counts[value_counts < self.suppression_threshold].index

        data[column] = data[column].replace(rare_values, self.mask_value)
        return data
