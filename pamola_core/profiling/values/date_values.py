"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor`  
for analyzing date and timestamp fields in datasets.  

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE 

Module: Date Values Analysis Processor  
--------------------------------
It identifies key characteristics such as:  
- Minimum and maximum dates  
- Date range analysis (days, months, years)  
- Temporal distribution  
- Time series trend detection  

NOTE: Requires `pandas` and `numpy`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import ABC
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import linregress
import dask.dataframe as dd

from pamola_core.profiling.base import BaseProfilingProcessor

class DateValuesProfilingProcessor(BaseProfilingProcessor, ABC):
    """
    Processor for analyzing date and timestamp fields.
    Identifies date distributions, trends, and seasonality.
    """

    def __init__(
        self,
        exclude_nulls: bool = True,
        format_dates: bool = True,
        time_series_analysis: bool = True,
        min_data_points: int = 30,
        ignore_non_date: bool = True,
    ):
        """
        Initializes the DateValuesProcessor with configurable options.

        Parameters:
        -----------
        exclude_nulls : bool
            Whether to exclude null values from analysis.
        format_dates : bool
            Whether to format dates in human-readable format.
        time_series_analysis : bool
            Whether to perform time series analysis.
        min_data_points : int
            Minimum number of data points required for time series analysis.
        ignore_non_date : bool
            Whether to ignore non-date values (convert to dates when possible).
        """
        super().__init__()
        self.exclude_nulls = exclude_nulls
        self.format_dates = format_dates
        self.time_series_analysis = time_series_analysis
        self.min_data_points = min_data_points
        self.ignore_non_date = ignore_non_date
    
    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform advanced analysis on specific date columns or all detected date fields.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        columns : List[str], optional
            The specific columns to analyze. If None, analyze all detected date columns.
        **kwargs : dict
            Additional parameters that can override instance attributes:
            
            - exclude_nulls (bool): Whether to exclude null values before analysis.
            - format_dates (bool): Whether to return date values as formatted strings.
            - time_series_analysis (bool): Whether to perform trend and seasonality analysis.
            - min_data_points (int): Minimum number of data points required for time series analysis.
            - ignore_non_date (bool): Whether to convert non-date values to NaT and drop them.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing analysis results, including:
            
            - "min_date": Minimum date value in the column.
            - "max_date": Maximum date value in the column.
            - "date_range": Dictionary with counts of days, months, and years covered.
            - "distribution": Yearly, monthly, and weekday distributions.
            - "business_weekend_ratio": Ratio of business days to weekends.
            - "trend_analysis": Detected trends in daily, weekly, and monthly frequency.
            - "seasonality_analysis": Patterns in weekly and monthly distributions.
            - "gap_analysis": Gaps in time series data.
        """

        # Override instance attributes with kwargs
        exclude_nulls = kwargs.get("exclude_nulls", self.exclude_nulls)
        format_dates = kwargs.get("format_dates", self.format_dates)
        time_series_analysis = kwargs.get("time_series_analysis", self.time_series_analysis)
        min_data_points = kwargs.get("min_data_points", self.min_data_points)
        ignore_non_date = kwargs.get("ignore_non_date", self.ignore_non_date)

        results = {}
        
        # Auto-detect date columns if `columns` is None
        date_cols = columns if columns else self._identify_date_columns(df, ignore_non_date)

        for col in date_cols:
            if col not in df.columns:
                continue  # Skip if column is missing

            date_series = df[col]

            # Handle missing values and convert if necessary
            if exclude_nulls:
                date_series = date_series.dropna()

            if ignore_non_date:
                date_series = pd.to_datetime(date_series, errors="coerce").dropna()

            if date_series.empty:
                continue  # Skip empty columns after processing

            # Compute date statistics
            min_date = date_series.min()
            max_date = date_series.max()
            date_range_days = (max_date - min_date).days
            date_range_months = date_series.dt.to_period("M").nunique()
            date_range_years = date_series.dt.to_period("Y").nunique()

            # Compute distributions
            distribution_yearly = date_series.dt.year.value_counts().to_dict()
            distribution_monthly = date_series.dt.month.value_counts().to_dict()
            distribution_weekday = date_series.dt.weekday.value_counts().to_dict()

            # Compute business vs weekend ratio
            business_days = (date_series.dt.weekday < 5).sum()  # Weekdays (0-4)
            weekends = (date_series.dt.weekday >= 5).sum()  # Weekends (5-6)
            business_weekend_ratio = business_days / max(1, weekends)  # Prevent division by zero

            # Perform analysis if enabled
            trend_analysis = self._analyze_trends(date_series, min_data_points) if time_series_analysis else None
            seasonality_analysis = self._analyze_seasonality(date_series, min_data_points) if time_series_analysis else None
            gap_analysis = self._analyze_gaps(date_series, time_series_analysis, min_data_points)

            # Store results
            results[col] = {
                "min_date": min_date.strftime('%Y-%m-%d') if format_dates else min_date,
                "max_date": max_date.strftime('%Y-%m-%d') if format_dates else max_date,
                "date_range": {
                    "days": date_range_days,
                    "months": date_range_months,
                    "years": date_range_years,
                },
                "distribution": {
                    "yearly": distribution_yearly,
                    "monthly": distribution_monthly,
                    "weekday": distribution_weekday,
                },
                "business_weekend_ratio": business_weekend_ratio,
                "trend_analysis": trend_analysis,
                "seasonality_analysis": seasonality_analysis,
                "gap_analysis": gap_analysis,
            }

        return results

    def _identify_date_columns(self, df: pd.DataFrame, ignore_non_date: bool = False) -> List[str]:
        """
        Identify columns that contain date or timestamp values.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        ignore_non_date : bool
            Whether to attempt conversion of non-date columns (e.g., object type) to datetime.

        Returns:
        --------
        List[str]
            A list of column names containing date or timestamp data.
        """
        if df.empty:
            return []

        # Directly select datetime columns
        date_columns = list(df.select_dtypes(include=["datetime"]).columns)

        # Convert object-type columns to datetime if enabled
        if ignore_non_date:
            obj_cols = df.select_dtypes(include=["object"]).columns
            for col in obj_cols:
                non_null_values = df[col].dropna()  # Remove NaN values before conversion
                converted = pd.to_datetime(non_null_values, errors="coerce")
                if converted.notna().mean() > 0.9:  # Ensure at least 90% valid dates
                    date_columns.append(col)

        return date_columns

    def _analyze_trends(self, date_series: pd.Series, min_data_points: int) -> Dict[str, Any]:
        """
        Analyze trends in the date series.

        Parameters:
        -----------
        date_series : pd.Series
            The time series data to analyze.
        min_data_points : int
            Minimum number of data points required for trend detection.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing detected trends in daily, weekly, and monthly frequency.
        """
        # Ensure date_series is datetime
        date_series = pd.to_datetime(date_series, errors="coerce").dropna()

        if date_series.empty or date_series.isnull().all():
            return {"error": "No valid dates available for trend analysis"}

        # Aggregate counts per day (handling duplicate timestamps)
        df = pd.DataFrame({"date": date_series})
        df["count"] = 1
        df = df.groupby(df["date"]).size().reset_index(name="count")

        if len(df) < min_data_points:
            return {"error": "Insufficient valid data points for trend analysis"}

        # Resample to daily frequency, filling missing days with zero
        df.set_index("date", inplace=True)
        df = df.resample("D").sum().fillna(0)

        # Compute trends at different time frequencies
        trends = {
            "daily": df["count"],
            "weekly": df["count"].resample("W").sum(),
            "monthly": df["count"].resample("ME").sum(),
        }

        def compute_trend(series: pd.Series) -> Dict[str, Any]:
            """Computes trend using linear regression."""
            series = series.dropna()  # Remove any NaN values
            if len(series) < min_data_points:
                return None  # Skip if not enough data points

            x = np.arange(len(series))
            y = series.values
            slope, _, r_value, p_value, _ = linregress(x, y)

            return {
                "slope": slope,
                "p_value": p_value,
                "r_squared": r_value ** 2,
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "significant": p_value < 0.05,
            }

        # Generate final trend analysis dictionary
        trend_analysis = {
            f"{freq}_trend": {
                "counts": trends[freq].to_dict(),
                "analysis": compute_trend(trends[freq])
            }
            for freq in trends
        }

        return trend_analysis

    def _analyze_seasonality(self, date_series: pd.Series, min_data_points: int) -> Dict[str, Any]:
        """
        Analyze seasonality patterns in time series data.

        Parameters:
        -----------
        date_series : pd.Series
            The series of date values.
        min_data_points : int
            Minimum number of data points required for analysis.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing detected seasonal patterns.
        """
        # Ensure date_series is in datetime format
        date_series = pd.to_datetime(date_series, errors="coerce").dropna()

        if len(date_series) < min_data_points:
            return {"error": "Not enough valid data points for seasonality analysis"}

        # Aggregate counts per unique date
        df = pd.DataFrame({"date": date_series})
        df["count"] = 1
        df = df.groupby(df["date"]).size().reset_index(name="count")

        # Extract seasonal components
        df["weekday"] = df["date"].dt.weekday  # 0=Monday, 6=Sunday
        df["month"] = df["date"].dt.month  # 1-12
        df["day_of_month"] = df["date"].dt.day  # 1-31
        df["quarter"] = df["date"].dt.quarter  # Q1-Q4

        # Compute seasonality patterns (normalized frequencies)
        weekly_seasonality = df["weekday"].value_counts(normalize=True).sort_index().to_dict()
        monthly_seasonality = df["month"].value_counts(normalize=True).sort_index().to_dict()
        day_of_month_seasonality = df["day_of_month"].value_counts(normalize=True).sort_index().to_dict()
        quarterly_seasonality = df["quarter"].value_counts(normalize=True).sort_index().to_dict()

        # If timestamps exist, analyze hourly seasonality
        if df["date"].dt.hour.nunique() > 1:  # Ensure there are different hours in the data
            df["hour"] = df["date"].dt.hour  # 0-23
            hourly_seasonality = df["hour"].value_counts(normalize=True).sort_index().to_dict()
        else:
            hourly_seasonality = None  # No meaningful hourly seasonality

        return {
            "weekly_seasonality": weekly_seasonality,
            "monthly_seasonality": monthly_seasonality,
            "day_of_month_seasonality": day_of_month_seasonality,
            "quarterly_seasonality": quarterly_seasonality,
            "hourly_seasonality": hourly_seasonality,
        }

    def _analyze_gaps(self, date_series: pd.Series, time_series_analysis: bool, min_data_points: int) -> Dict[str, Any]:
        """
        Detect gaps in a time series.

        Parameters:
        -----------
        date_series : pd.Series
            The time series data to analyze (must be a datetime series).
        time_series_analysis : bool
            Whether to perform time series analysis.
        min_data_points : int
            Minimum number of data points required for analysis.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing gap detection results.
        """
        if not time_series_analysis:
            return {"error": "Time series analysis is disabled"}

        # Ensure date_series is in datetime format
        date_series = pd.to_datetime(date_series, errors="coerce").dropna().drop_duplicates().sort_values()

        if len(date_series) < min_data_points:
            return {"error": "Insufficient valid data points for gap analysis"}

        # Compute gaps between consecutive dates
        gaps = date_series.diff().dropna()

        if gaps.empty:
            return {"message": "No gaps detected"}

        # Convert timedelta to days
        gap_days = gaps.dt.days

        return {
            "max_gap_days": int(gap_days.max()),
            "min_gap_days": int(gap_days.min()),
            "average_gap_days": round(gap_days.mean(), 2),
            "median_gap_days": int(gap_days.median()),
            "gap_distribution": gap_days.value_counts().to_dict(),
            "percentiles": {
                "25%": int(gap_days.quantile(0.25)),
                "50% (median)": int(gap_days.median()),
                "75%": int(gap_days.quantile(0.75))
            },
            "total_gaps_detected": len(gap_days)
        }