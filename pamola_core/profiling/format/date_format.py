"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of BaseProfilingProcessor or analyzing and validating
date formats, including format compliance, acceptable date ranges, and pattern detection.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Format Analysis Profiling Operations 
--------------------------------   
It includes the following capabilities:  
- Format compliance statistics
- Breakdown of non-compliant date values
- Year distribution analysis
- Out-of-range dates (before min_year or after max_year)
- Detected date formats if multiple are present
- Null and empty values statistics
- Most common date patterns
- Error categories (format errors, range errors, null errors

NOTE: Requires pandas.

Author: Realm Inveo Inc. & DGT Network Inc.
"""


from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from pamola_core.utils.date_format_utils import detect_date_format
from pamola_core.profiling.format.base import BaseDataFormatProfilingProcessor

class DateFormatProfilingProcessor(BaseDataFormatProfilingProcessor):
    """
    Processor for analyzing and validating date formats, including format compliance, acceptable date
    ranges, and pattern detection.
    """
    
    def __init__(
        self,
        expected_format=None, 
        min_year=1900,
        max_year=None,
        allow_nulls=True, 
        treat_empty_as_null=True, 
        detect_formats=True,
        save_invalid_dates=False,
        max_invalid_examples=100,
    ):
        """
        Initializes the Date Format Profiling Processor.

        Parameters:
        -----------
        expected_format : str, optional
            Expected date format pattern (default=None).
        min_year : int, optional
            Minimum acceptable year (default=1900).
        max_year : int, optional
            Maximum acceptable year (default=None).
        allow_nulls : bool, optional
            Whether null/empty values are allowed (default=True).
        treat_empty_as_null : bool, optional
            Whether to treat empty strings as nulls (default=True).
        detect_formats: bool, optional
            Whether to detect multiple formats in data (default=True).
        save_invalid_dates: bool, optional
            Whether to save invalid dates to a file (default=False).
        max_invalid_examples: int, optional
            Maximum invalid examples to include in report (default=100).

        """
        super().__init__()
        self.expected_format = expected_format
        self.min_year = min_year
        self.max_year = max_year
        self.allow_nulls = allow_nulls
        self.treat_empty_as_null = treat_empty_as_null
        self.detect_formats = detect_formats
        self.save_invalid_dates = save_invalid_dates
        self.max_invalid_examples = max_invalid_examples
    
    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform date format and validity analysis on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to analyze.
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all categorical columns will be selected.
        **kwargs : dict
            Dynamic parameter overrides:
            
            - expected_format (str, default=self.expected_format):
                Expected date format pattern.
            - min_year (int, default=self.min_year):
                Minimum acceptable year.
            - max_year (int, default=self.max_year):
                Maximum acceptable year.
            - allow_nulls (bool, default=self.allow_nulls):
                Whether null/empty values are allowed.
            - treat_empty_as_null (bool, default=self.treat_empty_as_null):
                Whether to treat empty strings as nulls.
            - detect_formats (bool, default=self.detect_formats):
                Whether to detect multiple formats in data.
            - save_invalid_dates (bool, default=self.save_invalid_dates):
                Whether to save invalid dates to a file.
            - max_invalid_examples (int, default=self.max_invalid_examples):
                Maximum invalid examples to include in report.
        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their pattern analysis results.
        """

        expected_format = kwargs.get("expected_format", self.expected_format)
        min_year = kwargs.get("min_year", self.min_year)
        max_year = kwargs.get("max_year", self.max_year)
        allow_nulls = kwargs.get("allow_nulls", self.allow_nulls)
        treat_empty_as_null = kwargs.get("treat_empty_as_null", self.treat_empty_as_null)
        detect_formats = kwargs.get("detect_formats", self.detect_formats)
        save_invalid_dates = kwargs.get("save_invalid_dates", self.save_invalid_dates)
        max_invalid_examples = kwargs.get("max_invalid_examples", self.max_invalid_examples)

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        results = {}

        for col in columns:
            series = df[col]

            if not allow_nulls:
                series = series.dropna()

            if treat_empty_as_null:
                series = series.replace("", pd.NA)

            # Identify null or empty series
            null_mask = series.isna()

            # Parse dates, coercing errors to NaT (invalid dates)
            parsed_dates = pd.to_datetime(series, errors="coerce", dayfirst=True)
            valid_dates_mask = ~parsed_dates.isna() & ~null_mask
            valid_dates = parsed_dates[valid_dates_mask]
            valid_count = valid_dates.count()

            # Identify format errors (entries that failed to parse)
            format_errors = parsed_dates.isna() & ~null_mask

            # Collect sample invalid examples if enabled
            invalid_examples = series.loc[format_errors].head(max_invalid_examples).tolist()

            invalid_count = format_errors.sum()

            results[col] = {
                "total_values": len(series),
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "null_or_empty_values": np.sum(null_mask),
                "sample_invalid_dates": invalid_examples,
            }

            error_categories = {
                "format_errors": np.sum(format_errors),
            }

            if valid_count > 0:
                # Identify out-of-range dates
                if max_year is not None:
                    out_of_range_mask = (valid_dates.dt.year < min_year) | (valid_dates.dt.year > max_year)
                else:
                    out_of_range_mask = valid_dates.dt.year < min_year

                year_distribution = valid_dates.dt.year.dropna().value_counts().to_dict()
                out_of_range_dates = valid_dates[out_of_range_mask].astype(str).tolist()

                error_categories.update({
                    "range_errors": np.sum(out_of_range_mask),
                    "null_errors": np.sum(null_mask)
                })

                results[col].update({
                    "year_distribution": year_distribution,
                    "out_of_range_dates": out_of_range_dates,
                })

            unique_formats = None
            # Detect unique date formats if enabled
            if detect_formats:
                unique_formats = sorted(set(detect_date_format(str(x), expected_format) 
                     for x in series[~format_errors & ~null_mask].dropna().unique()))

            results[col].update(error_categories)
            results[col]["detected_formats"] = unique_formats

        return results
