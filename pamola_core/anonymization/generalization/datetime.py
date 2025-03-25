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

Module: Datetime Generalization Processor
-------------------------------------------
This module provides methods for generalizing datetime attributes
to enhance anonymization. It extends the BaseGeneralizationProcessor and
implements techniques such as range generalization, period grouping,
and date formatting.

Datetime generalization reduces the specificity of timestamps
while preserving essential temporal trends.

Common approaches include:
- **Range Generalization**: Replacing precise timestamps with predefined
  date intervals (e.g., "January 2023", "Q1 2022").
- **Period Grouping**: Converting timestamps into higher-level categories
  (e.g., weekday vs. weekend, work hours vs. non-work hours).
- **Date Formatting**: Transforming timestamps into standardized formats
  while stripping unnecessary granularity.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
from abc import ABC
from pamola.pamola_core.anonymization.generalization.base import BaseGeneralizationProcessor


class DatetimeGeneralizationProcessor(BaseGeneralizationProcessor, ABC):
    """
    Datetime Generalization Processor for anonymizing datetime attributes.
    This class extends BaseGeneralizationProcessor and provides techniques
    for generalizing timestamps to enhance anonymization.

    Methods:
    --------
    - generalize_by_range(): Groups datetime values into predefined intervals.
    - group_by_period(): Converts timestamps into period-based categories.
    - format_dates(): Standardizes datetime format and strips granularity.
    """

    def __init__(self, date_format: str = "%Y-%m-%d", bins: list = None, period: str = None):
        """
        Initializes the datetime generalization processor.

        Parameters:
        -----------
        date_format : str, optional
            Desired format for datetime representation (default: "%Y-%m-%d").
        bins : list, optional
            Predefined date ranges for generalization (default: None).
        period : str, optional
            The time period for grouping, such as 'month', 'quarter', or 'weekday' (default: None).
        """
        self.date_format = date_format
        self.bins = bins
        self.period = period

    def generalize(self, data: pd.DataFrame, column: str, method: str = "range") -> pd.DataFrame:
        """
        Apply datetime generalization to a specified column.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data to be generalized.
        column : str
            The column name to be processed.
        method : str, optional
            The generalization method to apply ("range", "period", "format").

        Returns:
        --------
        pd.DataFrame
            The dataset with generalized datetime values.
        """
        if method == "range" and self.bins:
            return self.generalize_by_range(data, column)
        elif method == "period" and self.period:
            return self.group_by_period(data, column)
        elif method == "format":
            return self.format_dates(data, column)
        else:
            raise ValueError(f"Invalid generalization method: {method}")

    def generalize_by_range(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Generalize datetime values by mapping them to predefined date ranges.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with datetime values mapped to range categories.
        """
        if not self.bins:
            raise ValueError("Bins must be defined for range generalization.")

        data[column] = pd.to_datetime(data[column])
        data[column] = pd.cut(data[column], bins=self.bins,
                              labels=[f"Range {i + 1}" for i in range(len(self.bins) - 1)])
        return data

    def group_by_period(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Convert datetime values into categorical periods (e.g., day of the week, month).

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with datetime values transformed into period categories.
        """
        data[column] = pd.to_datetime(data[column])

        if self.period == "month":
            data[column] = data[column].dt.strftime("%B")  # Converts to full month name
        elif self.period == "quarter":
            data[column] = data[column].dt.to_period("Q").astype(str)  # Converts to Quarter (Q1, Q2, etc.)
        elif self.period == "weekday":
            data[column] = data[column].dt.day_name()  # Converts to "Monday", "Tuesday", etc.
        elif self.period == "hour":
            data[column] = data[column].dt.hour  # Extracts only the hour component
        else:
            raise ValueError(f"Invalid period type: {self.period}")

        return data

    def format_dates(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Standardize datetime values by formatting them according to a specified pattern.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with formatted datetime values.
        """
        data[column] = pd.to_datetime(data[column])
        data[column] = data[column].dt.strftime(self.date_format)
        return data
