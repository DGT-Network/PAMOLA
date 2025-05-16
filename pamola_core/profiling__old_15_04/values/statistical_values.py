"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor`  
that automatically determines the field type (numeric, date, or categorical)
and applies the appropriate analysis to datasets.  

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Universal Statistical Analysis Processor  
--------------------------------   
It includes the following capabilities:  
- Automatic data type determination
- For numeric fields: complete statistics (as in numeric_values)
- For date fields: time series analysis (as in date_values)
- For categorical fields: category distribution analysis, entropy, most frequent values
- Customized visualizations depending on data type


NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import ABC
from typing import Any, Dict, List, Optional
import pandas as pd
from scipy.stats import entropy

from pamola_core.profiling.values.base import BaseDataFieldProfilingProcessor
from pamola_core.profiling.values.date_values import DateValuesProfilingProcessor
from pamola_core.profiling.values.numeric_values import NumericFieldsProfilingProcessor

class StatisticalValuesProfilingProcessor(BaseDataFieldProfilingProcessor, ABC):
    """
    Processor that automatically determines the field type (numeric, date, or categorical)
    and applies the appropriate analysis.
    """
    
    def __init__(
        self,
        exclude_zeros: bool = False,
        exclude_nulls: bool = True,
        bins: int = 20,
        format_dates: bool = True,
        outlier_method: str = "iqr",
        sample_size: Optional[int] = None,
        advanced_metrics: bool = True,
        time_series_analysis: bool = True,
        save_distribution: bool = False,
    ):
        """
        Initializes the Statistical Values Profiling Processor with configurable options.

        Parameters:
        -----------
        exclude_zeros : bool, optional
            Whether to exclude zero values from analysis.
        exclude_nulls : bool, optional
            Whether to exclude null values from analysis.
        bins : int, optional
            Number of bins for histogram.
        format_dates : bool, optional
            Whether to format dates in a human-readable format.
        outlier_method : str, optional
            Method for outlier detection ("iqr", "zscore", "modified_zscore").
        sample_size : Optional[int], optional
            Maximum number of samples to analyze.
        advanced_metrics : bool, optional
            Whether to calculate advanced metrics.
        time_series_analysis : bool, optional
            Whether to perform time series analysis for dates.
        save_distribution : bool, optional
            Whether to save the full data distribution.
        """
        super().__init__()  # Ensure proper inheritance from BaseProfilingProcessor
        self.exclude_zeros = exclude_zeros
        self.exclude_nulls = exclude_nulls
        self.bins = bins
        self.format_dates = format_dates
        self.outlier_method = outlier_method
        self.sample_size = sample_size
        self.advanced_metrics = advanced_metrics
        self.time_series_analysis = time_series_analysis
        self.save_distribution = save_distribution

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

            - `exclude_zeros`: (bool, default=False)
                Whether to exclude zero values from the analysis.
            - `exclude_nulls`: (bool, default=True)
                Whether to exclude NaN values from the analysis.
            - `bins`: (int, default=20)
                Number of bins for histogram.
            - `format_dates`: (bool, default=True)
                Whether to format dates in a human-readable format.
            - `outlier_method`: (str, default="iqr")
                Method for outlier detection. Supported values: "iqr", "zscore", "modified_zscore".
            - `sample_size`: (int, default=None)
                Maximum number of samples to analyze. Example values: 1000, 5000, 10000.
            - `advanced_metrics`: (bool, default=True)
                Whether to compute advanced metrics such as skewness, kurtosis, and normality tests.
            - `time_series_analysis`: (bool, default=True)
                Whether to perform time series analysis for dates.
            - `save_distribution`: (bool, default=False)
                Whether to save the full data distribution.

        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their statistical analysis results.
        """

        exclude_nulls = kwargs.get("exclude_nulls", self.exclude_nulls)
        exclude_zeros = kwargs.get("exclude_zeros", self.exclude_zeros)
        bins = kwargs.get("bins", self.bins)
        format_dates = kwargs.get("format_dates", self.format_dates)
        outlier_method = kwargs.get("outlier_method", self.outlier_method)
        sample_size = kwargs.get("sample_size", self.sample_size)
        advanced_metrics = kwargs.get("advanced_metrics", self.advanced_metrics)
        time_series_analysis = kwargs.get("time_series_analysis", self.time_series_analysis)
        save_distribution = kwargs.get("save_distribution", self.save_distribution)

        results = {}
        columns = columns or df.columns.tolist()

        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            
            series = df[col]

            if exclude_nulls:
                series = series.dropna()
            if exclude_zeros:
                series = series[series != 0]
            if sample_size:
                series = series.sample(min(len(series), sample_size))

            date_time_series = None
            if any(keyword in col.lower() for keyword in ['birth', 'day', 'date', 'time']):
                date_time_series = series.apply(pd.to_datetime, errors='coerce')
            
            if pd.api.types.is_numeric_dtype(series):
                processor = NumericFieldsProfilingProcessor(
                    exclude_zeros=exclude_zeros,
                    exclude_nulls=exclude_nulls,
                    outlier_method=outlier_method,
                    advanced_metrics=advanced_metrics
                )
                results[col] = processor.execute(df, [col])
            elif pd.api.types.is_datetime64_any_dtype(date_time_series) and time_series_analysis:
                processor = DateValuesProfilingProcessor(
                    exclude_nulls = exclude_nulls,
                    format_dates = format_dates,
                    time_series_analysis = time_series_analysis,
                )
                results[col] = processor.execute(df, [col])
            else:
                results[col] = self._generate_categorical_statistics(series, bins, sample_size, advanced_metrics)
            
        return results

    def _generate_categorical_statistics(
        self,
        series: pd.Series,
        bins: int = 20,
        sample_size: Optional[int] = None,
        advanced_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates statistical insights for categorical data.

        Parameters:
        -----------
        series : pd.Series
            The categorical data series to analyze.

        bins : int, optional (default=20)
            Not used in the current implementation but reserved for potential histogram-based analysis.

        sample_size : Optional[int], optional (default=None)
            If specified, limits the number of samples analyzed to improve performance on large datasets.

        advanced_metrics : bool, optional (default=True)
            Whether to compute additional metrics like entropy, mode frequency, and rarity threshold.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing:
            - "unique_values": Number of unique values in the series.
            - "most_common": The most frequently occurring value.
            - "least_common": The least frequently occurring value.
            - "frequency_distribution": A dictionary mapping each unique value to its occurrence count.
            - (If advanced_metrics is enabled)
                - "entropy": A measure of distribution randomness (Shannon entropy).
                - "mode_frequency": The relative frequency of the most common value.
                - "rarity_threshold": Number of categories appearing in less than 1% of the dataset.
        """
        
        # Apply sampling if sample_size is set
        if sample_size and len(series) > sample_size:
            series = series.sample(n=sample_size, random_state=42)

        # Calculate value counts
        value_counts = series.value_counts()
        total_count = value_counts.sum()

        # Determine most/least common categories
        most_common = value_counts.idxmax() if not value_counts.empty else None
        least_common = value_counts.idxmin() if not value_counts.empty else None

        # Limit to `bins` categories and group the rest into "Other"
        if len(value_counts) > bins:
            top_values = value_counts.nlargest(bins)
            other_values = value_counts.iloc[bins:].sum()
            value_counts = pd.concat([top_values, pd.Series({"Other": other_values})])  # Group rare categories

        # Convert frequency distribution to dictionary
        frequency_distribution = value_counts.to_dict()

        # Initialize statistics dictionary
        statistics = {
            "unique_values": series.nunique(),
            "most_common": most_common,
            "least_common": least_common,
            "frequency_distribution": frequency_distribution
        }

        # Compute advanced metrics if enabled
        if advanced_metrics and total_count > 0:
            probabilities = value_counts / total_count
            statistics.update({
                "entropy": entropy(probabilities) if total_count > 0 else 0,
                "mode_frequency": value_counts.max() / total_count,
                "rarity_threshold": (value_counts.astype(float) < total_count * 0.01).sum()

            })

        return statistics
