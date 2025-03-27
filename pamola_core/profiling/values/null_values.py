"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor`  
for identifying and analyzing null, missing, or zero values in datasets.  

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Null Values Analysis Processor  
--------------------------------   
It includes the following capabilities:  
- Total count and percentage of null values  
- Distribution by types of null values (NULL, empty strings, special markers)  
- Visualization of the ratio between null and non-null values  
- Detailed statistics for each type of null value  

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import ABC
from typing import Any, Dict, List, Optional
import pandas as pd

from pamola_core.profiling.base import BaseProfilingProcessor

class NullValuesProfilingProcessor(BaseProfilingProcessor, ABC):
    """
    Processor for analyzing null values in datasets.
    Identifies and quantifies missing data occurrences.
    """
    
    def __init__(self, count_zeros_as_null: bool = False, custom_null_markers: Optional[List[Any]] = None):
        """
        Initializes the NullValuesProcessor with configurable options.

        Parameters:
        -----------
        count_zeros_as_null : bool, optional
            Whether to count zero values (0) as null (default: False).
        custom_null_markers : list, optional
            Additional values to be considered as null (default: None).
        """
        super().__init__()
        self.count_zeros_as_null = count_zeros_as_null
        self.custom_null_markers = custom_null_markers or []

    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform null value analysis on specific columns or the entire dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        columns : List[str], optional
            A list of specific columns to analyze. If None, analyze all columns.
        **kwargs : dict
            Dynamic parameter overrides:
            
            - `count_zeros_as_null` (bool): If True, treat zeros as null values.
            - `custom_null_markers` (list): Additional values to be considered as null.

        Returns:
        --------
        Dict[str, Any]
            Analysis results including:
            - Total nulls
            - Null percentage
            - Null distribution
            - Non-null ratio
            - Detailed statistics
        """
        count_zeros_as_null = kwargs.get("count_zeros_as_null", self.count_zeros_as_null)
        custom_null_markers = kwargs.get("custom_null_markers", self.custom_null_markers)

        results = {}
        columns_to_check = columns or df.columns

        for col in columns_to_check:
            if col not in df.columns:
                continue  # Skip missing columns

            col_data = df[col]
            total_nulls, null_distribution = self._analyze_nulls(col_data, count_zeros_as_null, custom_null_markers)
            total_values = len(col_data)
            
            null_percentage = (total_nulls / total_values * 100) if total_values > 0 else 0
            non_null_ratio = (1 - (total_nulls / total_values)) if total_values > 0 else 0

            results[col] = {
                "total_nulls": total_nulls,
                "null_percentage": round(null_percentage, 2),
                "null_distribution": null_distribution,
                "non_null_ratio": round(non_null_ratio, 2),
                "detailed_statistics": self._generate_statistics(col_data, count_zeros_as_null, custom_null_markers)
            }

        return results

    def _analyze_nulls(self, series: pd.Series, count_zeros_as_null: bool, custom_null_markers: List[Any]) -> tuple:
        """
        Analyze null values in a given series.

        Parameters:
        -----------
        series : pd.Series
            The column to analyze.
        count_zeros_as_null : bool
            If True, treat zero values as null.
        custom_null_markers : list
            Additional values to be treated as null.

        Returns:
        --------
        tuple
            - Total null count
            - Distribution of null types (NULL, empty, custom markers)
        """
        standard_nulls = series.isna().sum()
        empty_strings = (series == "").sum()
        custom_nulls = series.isin(custom_null_markers).sum()
        zero_nulls = (series == 0).sum() if count_zeros_as_null else 0
        
        total_nulls = standard_nulls + empty_strings + custom_nulls + zero_nulls
        distribution = {
            "standard_nulls": standard_nulls,
            "empty_strings": empty_strings,
            "custom_nulls": custom_nulls,
            "zero_values": zero_nulls
        }
        
        return total_nulls, distribution

    def _generate_statistics(self, series: pd.Series, count_zeros_as_null: bool, custom_null_markers: List[Any]) -> Dict[str, Any]:
        """
        Generate statistics for null value patterns.

        Parameters:
        -----------
        series : pd.Series
            The column to analyze.
        count_zeros_as_null : bool
            If True, treat zero values as null.
        custom_null_markers : list
            Additional values to be treated as null.

        Returns:
        --------
        Dict[str, Any]
            Detailed statistics about the null values.
        """
        total_values = len(series)
        unique_values = series.nunique()
        _, null_counts = self._analyze_nulls(series, count_zeros_as_null, custom_null_markers)
        
        mode_value = series.mode().tolist() if not series.isna().all() else None
        most_frequent_null = max(null_counts, key=null_counts.get) if null_counts else None
        
        return {
            "total_values": total_values,
            "unique_values": unique_values,
            "mode": mode_value,
            "most_frequent_null": most_frequent_null
        }