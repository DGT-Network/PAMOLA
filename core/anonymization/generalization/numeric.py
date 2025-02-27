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

Module: Numeric Generalization Processor
-----------------------------------------
This module provides methods for generalizing numerical attributes
to enhance anonymization. It extends the BaseGeneralizationProcessor and
implements techniques such as binning, rounding, and normalization.

Numeric generalization reduces the granularity of numerical values
while preserving statistical distribution and analytical utility.

Common approaches include:
- **Range Binning**: Replacing continuous values with predefined intervals
  (e.g., converting ages into groups like "18-25", "26-35").
- **Precision Reduction**: Rounding numerical values to lower precision
  (e.g., truncating currency values to nearest hundred).
- **Scaling and Normalization**: Transforming values to a common scale
  while preserving proportional relationships.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
from abc import ABC
from core.anonymization.generalization.base import BaseGeneralizationProcessor


class NumericGeneralizationProcessor(BaseGeneralizationProcessor, ABC):
    """
    Numeric Generalization Processor for anonymizing numerical attributes.
    This class extends BaseGeneralizationProcessor and provides techniques
    for generalizing numerical data to enhance anonymization.

    Methods:
    --------
    - bin_values(): Groups numeric values into predefined bins.
    - round_values(): Reduces numerical precision by rounding.
    - normalize_values(): Normalizes numerical values to a standard scale.
    """

    def __init__(self, bins: list = None, round_precision: int = None, scale: str = None):
        """
        Initializes the numeric generalization processor.

        Parameters:
        -----------
        bins : list, optional
            Predefined bins for range generalization (default: None).
        round_precision : int, optional
            Number of decimal places to retain for rounding (default: None).
        scale : str, optional
            Scaling method ('min-max', 'z-score') for normalization (default: None).
        """
        self.bins = bins
        self.round_precision = round_precision
        self.scale = scale

    def generalize(self, data: pd.DataFrame, column: str, method: str = "binning") -> pd.DataFrame:
        """
        Apply numeric generalization to a specified column.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data to be generalized.
        column : str
            The column name to be processed.
        method : str, optional
            The generalization method to apply ("binning", "rounding", "scaling").

        Returns:
        --------
        pd.DataFrame
            The dataset with generalized numeric values.
        """
        if method == "binning" and self.bins:
            return self.bin_values(data, column)
        elif method == "rounding" and self.round_precision is not None:
            return self.round_values(data, column)
        elif method == "scaling" and self.scale:
            return self.normalize_values(data, column)
        else:
            raise ValueError(f"Invalid generalization method: {method}")

    def bin_values(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Generalize numeric values by mapping them to predefined bins.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with numeric values mapped to bins.
        """
        if not self.bins:
            raise ValueError("Bins must be defined for range binning.")

        data[column] = pd.cut(data[column], bins=self.bins, labels=[f"Bin {i + 1}" for i in range(len(self.bins) - 1)])
        return data

    def round_values(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Reduce precision of numeric values by rounding.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with rounded numeric values.
        """
        if self.round_precision is None:
            raise ValueError("Rounding precision must be specified.")

        data[column] = data[column].round(self.round_precision)
        return data

    def normalize_values(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Normalize numeric values using a specified scaling method.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with normalized numeric values.
        """
        if self.scale not in ["min-max", "z-score"]:
            raise ValueError("Invalid scaling method. Use 'min-max' or 'z-score'.")

        if self.scale == "min-max":
            min_val, max_val = data[column].min(), data[column].max()
            data[column] = (data[column] - min_val) / (max_val - min_val)

        elif self.scale == "z-score":
            mean, std = data[column].mean(), data[column].std()
            data[column] = (data[column] - mean) / std

        return data
