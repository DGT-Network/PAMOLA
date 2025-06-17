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
from typing import Dict, List, Optional
import pandas as pd
from abc import ABC
from pamola_core.anonymization__old_15_04.generalization.base import BaseGeneralizationProcessor
from pamola_core.common.constants import Constants
from pamola_core.common.enum.datetime_generalization import DatePeriod, DatetimeMethod
from pamola_core.common.logging.privacy_logging import log_privacy_transformations, save_privacy_logging
from pamola_core.common.validation.check_column import check_columns_exist


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

    def __init__(
        self,
        target_fields: List[str] = [],
        method: Optional[Dict[str, str]] = None,
        date_format: Optional[Dict[str, str]] = None,
        period: Optional[Dict[str, str]] = None,
        ranges: Optional[Dict[str, List[Dict[str, str]]]] = None,
        auto_ranges: Optional[Dict[str, bool]] = None,
        range_interval: Optional[Dict[str, str]] = None,
        num_ranges: Optional[Dict[str, int]] = None,
        special_handling: Optional[Dict] = None,
        preserve_fields: Optional[List[str]] = None,
        track_changes: bool = True,
        generalization_log: Optional[str] = None,
    ):
        """
        Initializes the datetime generalization processor with configurable parameters.

        Parameters:
        -----------
        target_fields : List[str], default=[]
            List of fields to apply generalization.
        method : Dict[str, str], optional
            Dictionary mapping fields to their generalization methods 
            (e.g., {"birth_date": "format", "last_visit": "range"}).
        date_format : Dict[str, str], optional
            Dictionary mapping fields to specific formats (e.g., {"birth_date": "%Y"}).
        period : Dict[str, str], optional
            Time period for grouping per column (e.g., {"registration_date": "quarter"}).
        ranges : Dict[str, List[Dict[str, str]]], optional
            Custom date ranges for binning (e.g., {"birth_date":[{"start": "2020-01-01", "end": "2020-06-30", "label": "H1 2020"}]}).
        auto_ranges : Dict[str, bool], optional
            Whether to generate ranges automatically (e.g., {"last_visit": True}).
        range_interval : Dict[str, str], optional
            Interval for auto-generated ranges per column (e.g., {"last_visit": "month"}).
        num_ranges : Dict[str, int], optional
            Number of auto-generated ranges per column (e.g., {"last_visit": 6}).
        special_handling : Dict, optional
            Special cases handling (e.g., {"weekends": "Weekend"}).
        preserve_fields : List[str], optional
            Fields to exclude from generalization.
        track_changes : bool
            Whether to track transformations.
        generalization_log : str, optional
            Path to save generalization log.
        """
        # Initialize parameters with safe default values
        self.target_fields = target_fields or []
        self.method = method or {field: DatetimeMethod.RANGE.value for field in self.target_fields}
        self.date_format = date_format or {}
        self.period = period or {}
        self.ranges = ranges or {}
        self.auto_ranges = auto_ranges or {}
        self.range_interval = range_interval or {}
        self.num_ranges = num_ranges or {}
        self.special_handling = special_handling or {}
        self.preserve_fields = preserve_fields or []
        self.track_changes = track_changes
        self.generalization_log = generalization_log
        self.change_log = {}


    def generalize(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply datetime generalization to specified target fields.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        **kwargs : dict, optional
            target_fields : list, optional
                Fields to generalize. Defaults to self.target_fields.
            method : Dict[str, str], optional
                Dictionary mapping fields to their generalization methods.
            preserve_fields : list, optional
                Fields to exclude from generalization.
            track_changes : bool, optional
                Whether to track changes.
            generalization_log : str, optional
                Path to save generalization log.

        Returns:
        --------
        pd.DataFrame
            The dataset with generalized datetime values.
        """

        # Retrieve parameters from kwargs or fallback to instance attributes
        target_fields = kwargs.get("target_fields", self.target_fields or [])
        preserve_fields = kwargs.get("preserve_fields", self.preserve_fields or [])
        track_changes = kwargs.get("track_changes", self.track_changes)
        generalization_log = kwargs.get("generalization_log", self.generalization_log)
        special_handling = kwargs.get("special_handling", self.special_handling)

        check_columns_exist(data, target_fields)

        for column in target_fields:
            if column in preserve_fields:
                continue

            # Optional special handling
            if column in special_handling:
                data = self.apply_special_handling(data, column, **kwargs)

            col_method = (kwargs.get("method") or self.method).get(column, "format")

            # Apply transformation
            if col_method == DatetimeMethod.RANGE.value:
                data = self.generalize_by_range(data, column, **kwargs)
            elif col_method == DatetimeMethod.PERIOD.value:
                data = self.group_by_period(data, column, **kwargs)
            elif col_method == DatetimeMethod.FORMAT.value:
                data = self.format_dates(data, column, **kwargs)
            else:
                raise ValueError(f"Invalid generalization method '{col_method}' for column '{column}'")

        # Save log if needed
        if track_changes and generalization_log:
            save_privacy_logging(
                change_log=self.change_log,
                log_str=generalization_log,
                track_changes=track_changes
            )

        return data

    def generalize_by_range(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Generalize datetime values by mapping them to predefined or auto-generated date ranges.
        Only values that fall within the bin ranges are transformed; others remain unchanged.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        column : str
            The column name to be processed.
        **kwargs : dict, optional
            ranges : dict, optional
                Dictionary mapping columns to custom bin ranges.
            auto_ranges : dict, optional
                Dictionary specifying whether to generate ranges automatically.
            range_interval : dict, optional
                Dictionary specifying interval for auto-generated ranges.
            num_ranges : dict, optional
                Dictionary specifying number of bins to create if auto_ranges=True.
            track_changes : bool, optional
                Whether to track transformation changes. Defaults to self.track_changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with datetime values mapped to range categories (if in bins).
        """
        # Retrieve parameters from kwargs or fallback to instance attributes
        col_ranges = (kwargs.get("ranges") or self.ranges).get(column, None)
        col_auto_ranges = (kwargs.get("auto_ranges") or self.auto_ranges).get(column, False)
        col_range_interval = (kwargs.get("range_interval") or self.range_interval).get(column, None)
        col_num_ranges = (kwargs.get("num_ranges") or self.num_ranges).get(column, 12)
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Track original values if needed
        original_values = data[column].copy() if track_changes else None

        # Convert column to datetime in-place
        if not pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], errors="coerce")
  
        valid_data = data[column].dropna()
        if valid_data.empty:
            raise ValueError(f"Column '{column}' contains only NaN values after conversion.")

        bins = []
        bin_labels = []

        # Custom predefined ranges
        if isinstance(col_ranges, list) and col_ranges and isinstance(col_ranges[0], dict):
            try:
                sorted_ranges = sorted(col_ranges, key=lambda x: pd.to_datetime(x["start"]))
                bins = [pd.to_datetime(r["start"]) for r in sorted_ranges]
                bins.append(pd.to_datetime(sorted_ranges[-1]["end"]))

                # Assign labels by custom or automatically by start - end
                bin_labels = [
                    r["label"] if "label" in r else f"{pd.to_datetime(r['start']).date()} - {pd.to_datetime(r['end']).date()}"
                    for r in sorted_ranges
                ]
            except KeyError:
                raise ValueError("Each range dictionary must contain 'start' and 'end' keys.")

        # Auto-generated ranges
        if col_auto_ranges and not bins:
            min_date, max_date = valid_data.min(), valid_data.max()

            if col_range_interval:
                freq_code = Constants.FREQ_MAP.get(col_range_interval.lower())
                if not freq_code:
                    raise ValueError(f"Invalid frequency: {col_range_interval}")

                bins = pd.date_range(start=min_date, end=max_date, freq=freq_code).tolist()
                if bins[0] > min_date:
                    bins.insert(0, min_date)
                if bins[-1] < max_date:
                    bins.append(max_date)
            else:
                bins = pd.date_range(start=min_date, end=max_date, periods=col_num_ranges).tolist()

            # Default bin labels using date intervals
            bin_labels = [f"{bins[i].date()} - {bins[i+1].date()}" for i in range(len(bins) - 1)]

        if len(bins) < 2 or len(set(bins)) != len(bins):
            raise ValueError(f"No valid or unique ranges provided for column '{column}'.")

        # Apply binning
        cut_series = pd.cut(valid_data, bins=bins, labels=bin_labels, include_lowest=True)

        # Apply transformation only to binned values
        result_series = valid_data.copy()
        result_series.loc[result_series.index[cut_series.notna()]] = cut_series.loc[cut_series.notna()]

        data[column] = result_series

        # Track changes if enabled
        if track_changes:
            mask = (original_values != data[column]) & ~(original_values.isna() & data[column].isna())
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.generalize_by_range.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[column],
                fields=[column]
            )

        return data

    def group_by_period(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Convert datetime values into categorical periods.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        column : str
            The column name to be processed.
        **kwargs : dict, optional
            period : str, optional
                Period type for grouping. Supported options:
                    - "year": Extracts the year (e.g., 2025).
                    - "quarter": Converts to quarter periods (e.g., "2025Q1").
                    - "month": Converts to month names (e.g., "January").
                    - "weekday": Converts to day names (e.g., "Monday").
                    - "hour": Extracts the hour (e.g., 15 for 3 PM).
                Defaults to self.period.
            track_changes : bool, optional
                Whether to track changes and log original values. Defaults to self.track_changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with datetime values transformed into period categories.
        """
        # Retrieve parameters
        col_period = (kwargs.get("period") or self.period).get(column, "month")
        track_changes = kwargs.get("track_changes", self.track_changes)

        if not col_period:
            raise ValueError("A valid period type must be provided.")

        # Track original values if needed
        original_values = data[column].copy() if track_changes else None

        # Convert column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], errors="coerce")

        # Handle NaN values
        valid_data = data[column].dropna()
        if valid_data.empty:
            raise ValueError(f"Column '{column}' contains only NaN values after conversion.")

        # Apply period transformation
        if col_period == DatePeriod.YEAR.value:
            transformed_values = valid_data.dt.year
        elif col_period == DatePeriod.QUARTER.value:
            transformed_values = valid_data.dt.to_period("Q").astype(str)
        elif col_period == DatePeriod.MONTH.value:
            transformed_values = valid_data.dt.strftime("%B")
        elif col_period == DatePeriod.WEEKDAY.value:
            transformed_values = valid_data.dt.day_name()
        elif col_period == DatePeriod.HOUR.value:
            transformed_values = valid_data.dt.hour
        else:
            raise ValueError(f"Invalid period type: {col_period}")

        # Update only non-null values
        data[column] = data[column].astype(object)
        data.loc[valid_data.index, column] = transformed_values

        # Log changes efficiently
        if track_changes:
            mask = (original_values != data[column]) & ~(original_values.isna() & data[column].isna())
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.group_by_period.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[column],
                fields=[column]
            )
        
        return data

    def format_dates(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Format datetime values according to a specified pattern.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        column : str
            The column name to be processed.
        **kwargs : dict, optional
            date_format : str, optional
                Format pattern for timestamps. Defaults to self.date_format.
            track_changes : bool, optional
                Whether to track transformation changes. Defaults to self.track_changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with formatted datetime values as strings.
        """
        # Retrieve parameters
        col_date_format = (kwargs.get("date_format") or self.date_format).get(column, "%Y-%m")
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Track original values if needed
        original_values = data[column].copy() if track_changes else None

        # Ensure column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], errors="coerce")

        # Dropna to avoid formatting NaT
        valid_data = data[column].dropna()
        if valid_data.empty:
            raise ValueError(f"Column '{column}' contains only NaN values after conversion.")

        # Format datetime → string (NOT back to datetime!)
        formatted_strings = valid_data.dt.strftime(col_date_format)

        # Avoid re-casting strings back to datetime
        data[column] = data[column].astype(object)
        data.loc[formatted_strings.index, column] = formatted_strings

        # Track changes
        if track_changes:
            mask = (original_values != data[column]) & ~(original_values.isna() & data[column].isna())
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.format_dates.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[column],
                fields=[column]
            )

        return data

    def apply_special_handling(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Applies special handling rules to a datetime column in a DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing datetime data.
        column : str
            The column name to be processed.
        **kwargs : dict, optional
            special_handling : dict
                Dictionary defining special cases like weekends, holidays, etc.
            track_changes : bool, optional
                Whether to track transformation changes. Defaults to False.

        Returns:
        --------
        pd.DataFrame
            The dataset with special handling applied.
        """
        # Retrieve parameters
        col_handling = (kwargs.get("special_handling") or self.special_handling).get(column, {})
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Track original values if needed
        original_values = data[column].copy() if track_changes else None

        # Convert column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], errors="coerce")

        # Cast to object so we can assign strings later
        data[column] = data[column].astype(object)

        # Apply holiday handling first
        if "holidays" in col_handling:
            holidays = pd.to_datetime(col_handling["holidays"], errors="coerce")
            holiday_label = "Holiday"
            data[column] = data[column].apply(
                lambda x: holiday_label if isinstance(x, pd.Timestamp) and x in holidays else x
            )

        # Apply weekend handling (Saturday=5, Sunday=6)
        if "weekends" in col_handling:
            weekend_label = col_handling["weekends"]
            data[column] = data[column].apply(
                lambda x: weekend_label if isinstance(x, pd.Timestamp) and x.weekday() >= 5 else x
            )

        # Apply future_dates handling
        if "future_dates" in col_handling:
            future_label = col_handling["future_dates"]
            now = pd.Timestamp.now()
            data[column] = data[column].apply(
                lambda x: future_label if isinstance(x, pd.Timestamp) and x > now else x
            )

        # Track changes
        if track_changes:
            mask = (original_values != data[column]) & ~(original_values.isna() & data[column].isna())
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.apply_special_handling.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[column],
                fields=[column]
            )

        return data