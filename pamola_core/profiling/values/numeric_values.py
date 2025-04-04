"""
PAMOLA.CORE - Data Field Analysis Processor
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor`  
for analyzing numeric fields in datasets.  

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Numeric Fields Analysis Processor  
--------------------------------   
It identifies key characteristics such as:  
- Basic statistics (min, max, mean, median, standard deviation)  
- Percentiles (5%, 25%, 50%, 75%, 95%)  
- Outlier detection using IQR, z-score, modified z-score  
- Distribution skewness and kurtosis  
- Normality test  
- Data distribution visualizations  

NOTE: Requires `pandas` and `numpy`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import ABC
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import scipy.stats as stats
from pamola_core.profiling.values.base import BaseDataFieldProfilingProcessor

class NumericFieldsProfilingProcessor(BaseDataFieldProfilingProcessor, ABC):
    """
    Processor for analyzing numeric fields in datasets.
    """
    def __init__(self, exclude_zeros=False, exclude_nulls=True, outlier_method="iqr", 
             outlier_threshold=1.5, advanced_metrics=True, ignore_non_numeric=True):
        """
        Initialize the NumericFieldAnalysisProcessor.

        Parameters:
        -----------
        exclude_zeros : bool, optional (default=False)
            Whether to exclude zero values from analysis.
        
        exclude_nulls : bool, optional (default=True)
            Whether to exclude null values from analysis.

        outlier_method : str, optional (default="iqr")
            The method for detecting outliers. Supported methods:
            - "iqr" (Interquartile Range)
            - "zscore" (Standard Z-score)
            - "modified_zscore" (Modified Z-score)

        outlier_threshold : float, optional (default=1.5)
            The threshold for outlier detection, depending on the chosen method.

        advanced_metrics : bool, optional (default=True)
            Whether to compute advanced metrics like skewness and kurtosis.

        ignore_non_numeric : bool, optional (default=True)
            Whether to ignore non-numeric values in the dataset.
        """
        super().__init__() 
        
        # Set class attributes
        self.exclude_zeros = exclude_zeros
        self.exclude_nulls = exclude_nulls
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.advanced_metrics = advanced_metrics
        self.ignore_non_numeric = ignore_non_numeric

    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform numeric field analysis on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing numeric columns to analyze.
        
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all numeric columns will be automatically selected.

        **kwargs : dict
            Optional parameters to override class-level settings dynamically:

            - `ignore_non_numeric` (bool, default=self.ignore_non_numeric): 
                Whether to ignore non-numeric values (convert them to NaN).

            - `exclude_nulls` (bool, default=self.exclude_nulls): 
                Whether to exclude NaN values from the analysis.

            - `exclude_zeros` (bool, default=self.exclude_zeros): 
                Whether to exclude zero values from the analysis.

            - `outlier_method` (str, default=self.outlier_method): 
                Method for outlier detection ("iqr", "zscore", or "modified_zscore").

            - `outlier_threshold` (float, default=self.outlier_threshold): 
                Threshold value for outlier detection.

            - `advanced_metrics` (bool, default=self.advanced_metrics): 
                Whether to compute advanced metrics (skewness, kurtosis, normality test).

        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their numeric analysis results.
        """
        # Override class attributes dynamically using kwargs
        ignore_non_numeric = kwargs.get("ignore_non_numeric", self.ignore_non_numeric)
        exclude_nulls = kwargs.get("exclude_nulls", self.exclude_nulls)
        exclude_zeros = kwargs.get("exclude_zeros", self.exclude_zeros)
        outlier_method = kwargs.get("outlier_method", self.outlier_method)
        outlier_threshold = kwargs.get("outlier_threshold", self.outlier_threshold)
        advanced_metrics = kwargs.get("advanced_metrics", self.advanced_metrics)

        if columns is None:
            # Auto-detect numeric columns
            columns = df.select_dtypes(include=["number"]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns found for analysis.")

        results = {}
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

            series = df[col]
            if ignore_non_numeric:
                series = pd.to_numeric(series, errors="coerce")
            if exclude_nulls:
                series = series.dropna()
            if exclude_zeros:
                series = series[series != 0]

            if series.empty:
                results[col] = {"message": "No valid numeric data available."}
            else:
                results[col] = self._generate_statistics(series, advanced_metrics, outlier_method, outlier_threshold)

        return results
    
    def _generate_statistics(
        self,
        series: pd.Series,
        advanced_metrics: bool = False,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Generate basic and advanced statistics for a numeric field.

        Parameters:
        -----------
        series : pd.Series
            The numeric column to analyze.
        advanced_metrics : bool, optional
            Whether to compute advanced metrics like skewness, kurtosis, and normality test.
        outlier_method : str, optional
            Method for outlier detection ("iqr", "zscore", "modified_zscore").
        outlier_threshold : float, optional
            Threshold value for the chosen outlier detection method.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing:
            - `count`: Total number of values.
            - `unique`: Number of unique values.
            - `min`: Minimum value.
            - `max`: Maximum value.
            - `mean`: Mean (average) value.
            - `median`: Median value.
            - `std`: Standard deviation.
            - `percentiles`: Percentile values (5%, 25%, 50%, 75%, 95%).
            - `skewness`: Skewness of the distribution (if `advanced_metrics` is True).
            - `kurtosis`: Kurtosis value (if `advanced_metrics` is True).
            - `normality_p_value`: Result of normality test (if `advanced_metrics` is True).
            - `outliers`: Outlier statistics.
        """
        series = series.dropna()  # Remove NaNs to avoid computation errors

        if series.empty:
            return {"message": "No valid numeric data available."}

        stats_summary = {
            "count": series.size,
            "unique": series.nunique(),
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
            "std_dev": series.std(),
            "percentiles": {
                "5%": np.percentile(series, 5),
                "25%": np.percentile(series, 25),
                "50%": np.percentile(series, 50),
                "75%": np.percentile(series, 75),
                "95%": np.percentile(series, 95)
            }
        }

        if advanced_metrics:
            stats_summary.update({
                "skewness": stats.skew(series, nan_policy="omit"),
                "kurtosis": stats.kurtosis(series, nan_policy="omit"),
                "normality_p_value": stats.shapiro(series)[1] if len(series) < 5000 else None  # Shapiro test for normality
            })

        stats_summary["outliers"] = self._detect_outliers(series, outlier_method, outlier_threshold)

        return stats_summary
    
    def _detect_outliers(
        self,
        series: pd.Series,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Identify outliers in a numeric field using the selected method.

        Parameters:
        -----------
        series : pd.Series
            The numeric column to analyze for outliers.
        outlier_method : str, optional
            Method for outlier detection ("iqr", "zscore", "modified_zscore").
        outlier_threshold : float, optional
            Threshold value for the chosen outlier detection method.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing:
            - `outlier_count`: Number of detected outliers.
            - `outlier_indices`: Indices of detected outliers.
            - `outlier_values`: List of outlier values.
            - `method`: Outlier detection method used.
        """
        series = series.dropna()
        outliers = {"method": outlier_method, "outlier_count": 0, "outlier_indices": [], "outlier_values": []}

        if series.empty:
            return outliers

        if outlier_method == "iqr":
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - outlier_threshold * iqr, q3 + outlier_threshold * iqr
            outlier_mask = (series < lower) | (series > upper)

        elif outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            outlier_mask = z_scores > outlier_threshold

        elif outlier_method == "modified_zscore":
            median, mad = np.median(series), np.median(np.abs(series - np.median(series)))
            mod_z_scores = 0.6745 * (series - median) / (mad if mad else 1)
            outlier_mask = np.abs(mod_z_scores) > outlier_threshold

        else:
            outliers["message"] = "Invalid outlier detection method."
            return outliers

        outliers.update(
            {
                "outlier_count": outlier_mask.sum(),
                "outlier_indices": series.index[outlier_mask].tolist(),
                "outlier_values": series[outlier_mask].tolist(),
            }
        )
        return outliers