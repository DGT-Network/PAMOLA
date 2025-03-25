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

Module: Record Suppression Processor
-------------------------------------
This module provides an implementation of **record-level suppression**
for anonymization-preserving data transformations. It extends the
BaseSuppressionProcessor and enables the removal of entire rows
(records) based on predefined conditions.

Record suppression ensures that high-risk or non-compliant data
is entirely removed from datasets rather than masked or generalized.

Common use cases include:
- Removing records with missing or incorrect values.
- Suppressing records where a combination of attributes creates high re-identification risk.
- Enforcing compliance with statistical disclosure control policies.

Features:
- **Threshold-based suppression**: Removes records where a column’s value appears below a frequency threshold.
- **Rule-based suppression**: Supports user-defined conditions for removing records.
- **Configurable strategies**: Provides flexibility for different anonymization requirements.

NOTE: This module requires `pandas` as a dependency.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
from abc import ABC
from pamola.pamola_core.anonymization.suppression.base import BaseSuppressionProcessor

class RecordSuppressionProcessor(BaseSuppressionProcessor, ABC):
    """
    Record Suppression Processor for removing entire rows (records)
    from a dataset based on predefined conditions.

    This class extends BaseSuppressionProcessor and provides techniques
    for record-level suppression for anonymization compliance.

    Methods:
    --------
    - suppress(): Removes records based on frequency thresholds or rule-based conditions.
    """

    def __init__(self, suppression_threshold: int = None):
        """
        Initializes the record suppression processor.

        Parameters:
        -----------
        suppression_threshold : int, optional
            The minimum frequency required for a value to remain in the dataset.
            Records containing values below this threshold will be removed
            (default: None, meaning no threshold-based suppression).
        """
        self.suppression_threshold = suppression_threshold

    def suppress(self, data: pd.DataFrame, columns: list[str] = None, condition: callable = None, **kwargs) -> pd.DataFrame:
        """
        Apply record suppression based on thresholds or custom conditions.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing records to be suppressed.
        columns : list[str], optional
            List of column names used for frequency-based suppression (default: None).
        condition : callable, optional
            A function that determines which records should be suppressed.
            Example: `lambda row: row["age"] < 18` (removes underage users).
        kwargs : dict
            Additional parameters for suppression logic.

        Returns:
        --------
        pd.DataFrame
            The dataset with suppressed records removed.
        """
        if condition:
            data = data[~data.apply(condition, axis=1)]

        if self.suppression_threshold is not None and columns:
            data = self._apply_threshold_suppression(data, columns)

        return data

    def _apply_threshold_suppression(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Remove records where specified column values appear less frequently than the defined threshold.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing records to be suppressed.
        columns : list[str]
            List of columns used to determine suppression criteria.

        Returns:
        --------
        pd.DataFrame
            The dataset with suppressed records removed.
        """
        for column in columns:
            value_counts = data[column].value_counts()
            rare_values = value_counts[value_counts < self.suppression_threshold].index
            data = data[~data[column].isin(rare_values)]

        return data
