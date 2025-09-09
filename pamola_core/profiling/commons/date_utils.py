"""
Date analysis utilities for the project.

This module provides utility functions for analyzing date fields, including
validation, distribution analysis, anomaly detection, and group analysis.
"""

import logging
from datetime import datetime
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from dask.delayed import delayed
from dask.base import compute
import dask.dataframe as dd

from joblib import Parallel, delayed as joblib_delayed

from pamola_core.common.constants import Constants
from pamola_core.common.helpers.data_helper import DataHelper
from pamola_core.common.regex.patterns import CommonPatterns
from pamola_core.utils.io_helpers.dask_utils import get_computed_df
from pamola_core.utils.progress import HierarchicalProgressTracker

# Configure logger
logger = logging.getLogger(__name__)


def convert_to_datetime_flexible(
    series,
    date_formats: List[str],
    is_dask: bool = False,
    sample_size: int = 1000,
    success_threshold: float = 0.95,
) -> pd.Series:
    """
    Convert a Series to datetime, trying Pandas inference first,
    then falling back to guessed formats if needed.

    Parameters
    ----------
    series : pd.Series or dask.Series
        Input datetime-like string series.
    date_formats : List[str]
        List of datetime formats to try if Pandas inference is insufficient.
        Example: ['%Y-%m-%d', '%d/%m/%Y']
    is_dask : bool
        Whether this is a Dask Series.
    sample_size : int
        Number of samples to use for format guessing.
    success_threshold : float
        Minimum success rate to consider parsing good enough (skip guessing).

    Returns
    -------
    pd.Series or dask.Series
        Datetime-parsed series.
    """

    def try_parse(s, fmt=None, meta=None):
        """Attempt datetime parsing with optional format."""
        return (
            s.map_partitions(pd.to_datetime, format=fmt, errors="coerce", meta=meta)
            if is_dask
            else pd.to_datetime(s, format=fmt, errors="coerce")
        )

    def get_success_rate(s):
        """Calculate fraction of non-null parsed values."""
        nulls = s.isna().sum().compute() if is_dask else s.isna().sum()
        total = s.size.compute() if is_dask else len(s)
        return 1.0 - (nulls / total) if total else 0.0

    # --- Step 1: Sample data for guessing ---
    try:
        if is_dask:
            total_size = series.size.compute()
            sample_frac = min(1.0, sample_size / total_size)
            sample_vals = (
                series.dropna().sample(frac=sample_frac, random_state=42).compute()
            )
        else:
            sample_vals = series.dropna().sample(
                min(sample_size, len(series)), random_state=42
            )

        sample_strs = [
            str(v).strip()
            for v in sample_vals
            if isinstance(v, str) or isinstance(v, (np.str_, np.object_))
        ]
        guessed_formats = DataHelper.guess_date_formats(
            sample_strs, date_formats, success_threshold
        )
    except Exception as e:
        logger.warning(f"Sampling for date format guessing failed: {e}")
        guessed_formats = date_formats  # Fallback

    # --- Step 2: If Pandas parsing was good enough, skip guessing ---
    if guessed_formats is None:
        try:
            return try_parse(series, meta=pd.Series(dtype="datetime64[ns]"))
        except Exception as e:
            logger.warning(
                f"Pandas datetime inference failed despite good sample rate: {e}"
            )

    # --- Step 3: Try guessed formats in order ---
    best_result = None
    best_success_rate = 0.0

    for fmt in guessed_formats:
        try:
            result = try_parse(
                series,
                (
                    fmt if fmt != "ISO8601" else None
                ),  # Allow Pandas to auto-detect for ISO
                meta=pd.Series(dtype="datetime64[ns]"),
            )
            success_rate = get_success_rate(result)

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_result = result

            if success_rate >= success_threshold:
                break  # Early exit on good enough result
        except Exception:
            continue

    # --- Step 4: Return best result found ---
    if best_result is not None and best_success_rate > 0:
        return best_result

    # --- Step 5: Final fallback to loose Pandas parser ---
    return try_parse(series, meta=pd.Series(dtype="datetime64[ns]"))


def prepare_date_data(
    df: Union[pd.DataFrame, dd.DataFrame],
    field_name: str,
    date_formats: Optional[List[str]] = None,
) -> Tuple[Union[pd.Series, dd.Series], int, int]:
    """
    Prepare date data for analysis with enhanced format support.

    Parameters:
    -----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    date_formats : Optional[List[str]]
        List of date formats to try. If None, will use default common formats.

    Returns:
    --------
    Tuple[Union[pd.Series, dd.Series], int, int]
        Tuple containing the prepared date series, null count, and non-null count
    """
    # Check if field exists
    if field_name not in df.columns:
        raise ValueError(f"Field {field_name} not found in DataFrame")

    # Default common date formats to try
    date_formats = date_formats or Constants.COMMON_DATE_FORMATS

    if isinstance(df, dd.DataFrame):
        # Dask DataFrame path
        field_data = df[field_name]
        null_count_future = field_data.isna().sum()
        total_count_future = field_data.size

        # Convert to datetime with flexible parsing
        try:
            dates = convert_to_datetime_flexible(field_data, date_formats, is_dask=True)
        except Exception as e:
            raise ValueError(
                f"Field '{field_name}' cannot be converted to datetime with any attempted format: {e}"
            )

        null_count, total_count = dd.compute(null_count_future, total_count_future)
        non_null_count = total_count - null_count
        return dates, int(null_count), int(non_null_count)

    else:
        # Pandas DataFrame path
        field_data = df[field_name]
        null_count = field_data.isna().sum()
        non_null_count = len(df) - null_count

        try:
            dates = convert_to_datetime_flexible(
                field_data, date_formats, is_dask=False
            )
        except Exception as e:
            raise ValueError(
                f"Field '{field_name}' cannot be converted to datetime with any attempted format: {e}"
            )

        return dates, int(null_count), int(non_null_count)


def calculate_date_stats(dates: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic date statistics, with support for object, datetime, and categorical columns.

    Parameters:
    -----------
    dates : pd.Series
        Series of date values (can be string, datetime, or categorical)

    Returns:
    --------
    Dict[str, Any]
        Dictionary with basic date stats
    """
    original_count = len(dates)

    # Step 1: If Categorical, ensure it's ordered or convert to datetime
    if pd.api.types.is_categorical_dtype(dates):
        if not dates.cat.ordered:
            dates = dates.cat.as_ordered()
        # Try converting categories to datetime
        try:
            dates = pd.to_datetime(dates.astype(str), errors="coerce")
        except Exception:
            dates = pd.Series([pd.NaT] * original_count)

    # Step 2: If not datetime, try to convert
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates, errors="coerce")

    # Step 3: Split valid and invalid
    valid_dates = dates.dropna()
    invalid_count = original_count - len(valid_dates)

    # Step 4: Extract stats
    if valid_dates.empty:
        return {
            "valid_count": 0,
            "invalid_count": invalid_count,
            "min_date": None,
            "max_date": None,
        }

    # Calculate date range
    min_date = valid_dates.min()
    max_date = valid_dates.max()

    return {
        "valid_count": len(valid_dates),
        "invalid_count": invalid_count,
        "min_date": min_date.strftime("%Y-%m-%d"),
        "max_date": max_date.strftime("%Y-%m-%d"),
    }


def calculate_distributions(dates: pd.Series) -> Dict[str, Dict[str, int]]:
    """
    Calculate various date distributions (year, decade, month, day of week).

    Parameters:
    -----------
    dates : pd.Series
        Series of dates to analyze

    Returns:
    --------
    Dict[str, Dict[str, int]]
        Dictionary with various date distributions
    """
    # Ensure datetime type
    if pd.api.types.is_categorical_dtype(dates):
        dates = pd.to_datetime(dates.astype(str), errors="coerce")
    elif not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates, errors="coerce")

    # Drop NaT values
    valid_dates = dates.dropna()

    if valid_dates.empty:
        return {}

    result = {}

    # Year distribution
    year_distribution = valid_dates.dt.year.value_counts().sort_index().to_dict()
    result["year_distribution"] = {
        str(year): int(count) for year, count in year_distribution.items()
    }

    # Decade distribution
    decade_distribution = (
        (valid_dates.dt.year // 10 * 10).value_counts().sort_index().to_dict()
    )
    result["decade_distribution"] = {
        f"{decade}s": int(count) for decade, count in decade_distribution.items()
    }

    # Month distribution
    month_distribution = valid_dates.dt.month.value_counts().sort_index().to_dict()
    result["month_distribution"] = {
        str(month): int(count) for month, count in month_distribution.items()
    }

    # Day of week distribution
    dow_distribution = valid_dates.dt.dayofweek.value_counts().sort_index().to_dict()
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    result["day_of_week_distribution"] = {
        day_names[dow]: int(count) for dow, count in dow_distribution.items()
    }

    return result


def validate_date_format(date_str: str, format_str: Optional[str] = None) -> bool:
    """
    Validate a date string using a given format or detect format automatically via regex.

    Parameters:
    -----------
    date_str : str
        The date string to validate.
    format_str : Optional[str]
        Specific format to validate against. If None, auto-detect using DATE_REGEX_FORMATS.

    Returns:
    --------
    bool
        True if date_str is valid in the given or detected format.
    """
    if not isinstance(date_str, str):
        return False

    # Case 1: Try the provided format directly
    if format_str:
        try:
            datetime.strptime(date_str, format_str)
            return True
        except Exception:
            return False

    # Case 2: Auto-detect format using regex
    for pattern, fmt in CommonPatterns.DATE_REGEX_FORMATS.items():
        if re.fullmatch(pattern, date_str):
            try:
                if fmt == "ISO8601":
                    datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                else:
                    datetime.strptime(date_str, fmt)
                return True
            except Exception:
                return False

    return False


def detect_date_anomalies(
    dates: Union[pd.Series, dd.Series], min_year: int = 1940, max_year: int = 2005
) -> Dict[str, List[Any]]:
    """
    Detect anomalies in date fields with support for Dask and Pandas.

    Parameters:
    ----------
    dates : Union[pd.Series, dd.Series]
        Series of date strings to analyze.
    min_year : int
        Minimum acceptable year.
    max_year : int
        Maximum acceptable year.
    sample_limit : int
        Maximum number of sample values to return per anomaly type.

    Returns:
    -------
    Dict[str, List[Any]]
        Dictionary with anomaly categories and examples.
    """

    # Choose execution method
    if isinstance(dates, dd.Series):
        try:
            # Use delayed processing for better control over the results
            partitions = dates.to_delayed()

            # Process each partition using delayed
            delayed_results = []
            for partition in partitions:
                delayed_result = delayed(detect_date_anomalies_partition)(
                    partition, min_year=min_year, max_year=max_year
                )
                delayed_results.append(delayed_result)

            # Compute all results
            all_anomalies = compute(*delayed_results)

            # Merge all anomaly categories and sample
            merged = {
                "invalid_format": [],  # Invalid format
                "too_old": [],  # Too old (before min_year)
                "future_dates": [],  # Future dates
                "too_young": [],  # Too young (after max_year)
                "negative_years": [],  # Negative years
            }

            # Handle the case where all_anomalies might be a single dict or list of dicts
            if isinstance(all_anomalies, dict):
                # If it's a single dictionary, treat it as one partition result
                for k in merged:
                    merged[k].extend(all_anomalies.get(k, []))
            else:
                # If it's a list/tuple of dictionaries (normal case)
                for part in all_anomalies:
                    if isinstance(part, dict):
                        for k in merged:
                            merged[k].extend(part.get(k, []))

            return merged

        except Exception as e:
            # Fallback to pandas processing if Dask fails
            logger.warning(
                f"Dask anomaly detection failed, falling back to pandas: {e}"
            )
            return detect_date_anomalies_partition(
                dates.compute(), min_year=min_year, max_year=max_year
            )
    else:
        return detect_date_anomalies_partition(
            dates, min_year=min_year, max_year=max_year
        )


def detect_date_anomalies_partition(
    partition: pd.Series, min_year: int = 1940, max_year: int = 2005
) -> Dict[str, List[Any]]:
    """
    Analyze a partition of date strings for anomalies.

    Parameters:
    ----------
    partition : pd.Series
        Series of date strings.
    min_year : int
        Minimum acceptable year.
    max_year : int
        Maximum acceptable year.

    Returns:
    -------
    Dict[str, List[Any]]
        Dictionary with anomaly categories and example values.
    """
    results = {
        "invalid_format": [],
        "too_old": [],
        "future_dates": [],
        "too_young": [],
        "negative_years": [],
    }

    now_year = datetime.now().year
    parsed_dates = pd.to_datetime(partition, errors="coerce")

    for i, (original, parsed) in enumerate(zip(partition, parsed_dates)):
        if pd.isna(original):
            continue

        original_str = str(original)

        # Check valid format first using COMMON_DATE_FORMATS
        if not validate_date_format(original_str):
            results["invalid_format"].append((i, original))
            continue

        # Try catch date parsing and year-based anomaly classification
        try:
            if pd.isna(parsed):
                if original_str.startswith("-"):
                    results["negative_years"].append((i, original_str))
                else:
                    results["invalid_format"].append((i, original_str))
                continue

            year = parsed.year

            if year < min_year:
                results["too_old"].append((i, original_str, year))
            elif year > now_year:
                results["future_dates"].append((i, original_str, year))
            elif year > max_year:
                results["too_young"].append((i, original_str, year))

        except Exception:
            results["invalid_format"].append((i, original_str))

    return results


def detect_date_changes_within_group(
    df: Union[pd.DataFrame, dd.DataFrame],
    group_column: str,
    date_column: str,
    example_limit: int = 10,
) -> Dict[str, Any]:
    """
    Detect date changes within groups, supports both Pandas and Dask DataFrames.

    Parameters:
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The input DataFrame (can be pandas or dask)
    group_column : str
        The column to group by
    date_column : str
        The date column to analyze
    example_limit : int
        Max number of examples to return

    Returns:
    -------
    Dict[str, Any]
    """
    if group_column not in df.columns:
        return {"error": f"Group column {group_column} not found in DataFrame"}

    if date_column not in df.columns:
        return {"error": f"Date column {date_column} not found in DataFrame"}

    # Dispatch to the appropriate function based on DataFrame type
    if isinstance(df, pd.DataFrame):
        # Use pandas-based implementation
        return _detect_date_changes_pandas(df, group_column, date_column, example_limit)
    elif isinstance(df, dd.DataFrame):
        # Use dask-based implementation for large or distributed data
        return _detect_date_changes_dask(df, group_column, date_column, example_limit)
    else:
        # Handle unsupported DataFrame types
        return {
            "error": "Unsupported DataFrame type. Only pandas and dask are supported."
        }


def _detect_date_changes_pandas(
    df: pd.DataFrame, group_column: str, date_column: str, example_limit: int = 10
) -> Dict[str, Any]:
    """
    Internal function to detect changes in date values within groups using Pandas.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to process.
    group_column : str
        The column to group by (e.g., 'resume_id').
    date_column : str
        The date column to inspect for changes.
    example_limit : int, optional
        The number of example groups to return that show date variation (default is 10).

    Returns:
    --------
    Dict[str, Any]
        A dictionary with:
        - 'groups_with_changes': total number of groups where date values vary.
        - 'examples': a list of sample groups and their unique date values.
    """

    # Group the DataFrame by the specified column
    grouped = df.groupby(group_column)

    # Initialize result dictionary
    results = {
        "groups_with_changes": 0,  # Count of groups with different date values
        "examples": [],  # Example group entries to be returned
    }

    # Iterate through each group
    for group_id, group_df in grouped:
        # Drop rows with missing date values in the group
        dates = group_df[date_column].dropna()

        # Skip group if it has one or no valid date entries
        if len(dates) <= 1:
            continue

        # Check if the group has more than one unique date
        if dates.nunique() > 1:
            # Count this group as one with date variation
            results["groups_with_changes"] += 1

            # Save up to 'example_limit' examples for inspection
            if len(results["examples"]) < example_limit:
                results["examples"].append(
                    {
                        "group_id": group_id,
                        "date_values": dates.unique().tolist(),  # Show all unique dates
                    }
                )

    return results


def _detect_date_changes_dask(
    ddf: dd.DataFrame, group_column: str, date_column: str, example_limit: int = 10
) -> Dict[str, Any]:
    """
    Internal function to detect changes in date values within groups using Dask and delayed processing.

    Parameters:
    -----------
    ddf : dd.DataFrame
        The Dask DataFrame to process.
    group_column : str
        The column to group by (e.g., an ID field like 'resume_id').
    date_column : str
        The date column to check for changes.
    example_limit : int, optional
        The number of example groups with changes to return (default is 10).

    Returns:
    --------
    Dict[str, Any]
        A dictionary with:
        - 'groups_with_changes': the number of groups with more than one unique date.
        - 'examples': a list of example group IDs and their unique date values (up to example_limit).
    """

    # Drop rows where group or date column is missing
    ddf_filtered = ddf.dropna(
        subset=[group_column, date_column]
    )  # Convert Dask DataFrame into delayed partitions (for parallel processing)
    partitions = ddf_filtered.to_delayed()

    @delayed
    def process_partition(part: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a single Pandas partition to detect groups with multiple unique dates.

        Parameters:
        -----------
        part : pd.DataFrame
            A Pandas DataFrame (a partition from the Dask DataFrame).

        Returns:
        --------
        List[Dict[str, Any]]
            List of groups where the date values differ, including the group_id and unique sorted date values.
        """
        results = []
        for group_id, group_df in part.groupby(group_column):
            dates = group_df[date_column].dropna()
            # Only keep groups with more than one unique date value
            if dates.nunique() > 1:
                results.append(
                    {
                        "group_id": group_id,
                        "date_values": sorted(dates.unique()),  # Sort for readability
                    }
                )
        return results

    # Apply delayed processing to all partitions
    delayed_results = [process_partition(part) for part in partitions]

    # Compute the delayed results and flatten the list of lists
    try:
        all_nested = compute(*delayed_results)  # List of lists of dicts
        all_results = []

        # Safely flatten the results
        for sublist in all_nested:
            if isinstance(sublist, list):
                all_results.extend(sublist)
            elif isinstance(sublist, dict):
                all_results.append(sublist)

        # Collect a limited number of example groups with changes
        seen_group_ids = set()
        examples = []
        for entry in all_results:
            if isinstance(entry, dict) and "group_id" in entry:
                gid = entry["group_id"]
                if gid not in seen_group_ids:
                    seen_group_ids.add(gid)
                    if len(examples) < example_limit:
                        examples.append(entry)
    except Exception as e:
        # Fallback: process without Dask if there's an issue
        logger.warning(f"Dask processing failed, falling back to pandas: {e}")
        return _detect_date_changes_pandas(
            ddf.compute(), group_column, date_column, example_limit
        )

    return {"groups_with_changes": len(seen_group_ids), "examples": examples}


def detect_date_inconsistencies_by_uid(
    df: Union[pd.DataFrame, dd.DataFrame],
    uid_column: str,
    date_column: str,
    example_limit: int = 10,
) -> Dict[str, Any]:
    """
    Detect date inconsistencies by UID (person identifier) for both Pandas and Dask DataFrames.

    Parameters:
    -----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The DataFrame to analyze
    uid_column : str
        The UID column
    date_column : str
        The date column to analyze
    example_limit : int
        Number of example UIDs to return

    Returns:
    --------
    Dict[str, Any]
        Dictionary with count of inconsistencies and example UIDs
    """
    if uid_column not in df.columns:
        return {"error": f"UID column {uid_column} not found in DataFrame"}
    if date_column not in df.columns:
        return {"error": f"Date column {date_column} not found in DataFrame"}

    df = get_computed_df(df)

    # Pandas version
    grouped = df.groupby(uid_column)
    results = {"uids_with_inconsistencies": 0, "examples": []}

    for uid, group_df in grouped:
        dates = group_df[date_column].dropna()
        if len(dates) <= 1:
            continue
        if dates.nunique() > 1:
            results["uids_with_inconsistencies"] += 1
            if len(results["examples"]) < example_limit:
                results["examples"].append(
                    {"uid": uid, "date_values": dates.unique().tolist()}
                )

    return results


def partition_date_stats(partition):
    """
    Calculate date statistics (min, max, valid count) for a partition of date data.

    Parameters:
    -----------
    partition : pd.Series or array-like
        Partition of date values to analyze.

    Returns:
    --------
    pd.Series
        Series containing:
            - min_date: Minimum valid date in the partition (pd.Timestamp or pd.NaT)
            - max_date: Maximum valid date in the partition (pd.Timestamp or pd.NaT)
            - valid_count: Number of valid (non-null) dates in the partition (int)
    """
    # Convert to datetime for this partition
    partition_dates = pd.to_datetime(partition, errors="coerce")
    valid_mask = ~partition_dates.isna()
    valid_dates = partition_dates[valid_mask]

    if len(valid_dates) == 0:
        return pd.Series({"min_date": pd.NaT, "max_date": pd.NaT, "valid_count": 0})

    return pd.Series(
        {
            "min_date": valid_dates.min(),
            "max_date": valid_dates.max(),
            "valid_count": len(valid_dates),
        }
    )


def partition_distributions(partition):
    """
    Calculate date distributions (year, month, day of week, decade) for a partition of date data.

    Parameters:
    -----------
    partition : pd.Series or array-like
        Partition of date values to analyze.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
            - type: Distribution type ('year', 'month', 'dow', 'decade')
            - key: Distribution key (e.g., year, month number, day index, decade label)
            - count: Count of occurrences for each key
        If no valid dates are present, returns an empty DataFrame with these columns.
    """
    partition_dates = pd.to_datetime(partition, errors="coerce")
    valid_mask = ~partition_dates.isna()
    valid_dates = partition_dates[valid_mask]

    if len(valid_dates) == 0:
        return pd.DataFrame({"type": [], "key": [], "count": []})

    # Return distributions for this partition
    year_counts = valid_dates.dt.year.value_counts()
    month_counts = valid_dates.dt.month.value_counts()
    dow_counts = valid_dates.dt.dayofweek.value_counts()
    decade_counts = (valid_dates.dt.year // 10 * 10).value_counts()

    # Combine all distributions into a single DataFrame
    result_data = []

    for year, count in year_counts.items():
        result_data.append({"type": "year", "key": str(year), "count": count})

    for month, count in month_counts.items():
        result_data.append({"type": "month", "key": str(month), "count": count})

    for dow, count in dow_counts.items():
        result_data.append({"type": "dow", "key": str(dow), "count": count})

    for decade, count in decade_counts.items():
        result_data.append({"type": "decade", "key": f"{decade}s", "count": count})

    return pd.DataFrame(result_data)


def aggregate_distributions_data(
    distributions_data: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """
    Aggregate distribution data from Dask partitions into the expected format.

    Parameters:
    -----------
    distributions_data : pd.DataFrame
        DataFrame with columns ['type', 'key', 'count'] containing distribution data

    Returns:
    --------
    Dict[str, Dict[str, int]]
        Dictionary with aggregated distributions in the expected format
    """
    if len(distributions_data) == 0:
        return {}

    # Aggregate by grouping and summing counts
    aggregated = (
        distributions_data.groupby(["type", "key"])["count"].sum().reset_index()
    )

    # Convert back to the expected format
    distributions = {}
    # Year distribution
    year_data = aggregated[aggregated["type"] == "year"]
    if len(year_data) > 0:
        # Sort by year in ascending order
        year_data_sorted = year_data.sort_values("key", key=lambda x: x.astype(int))
        distributions["year_distribution"] = dict(
            zip(year_data_sorted["key"], year_data_sorted["count"])
        )

    # Month distribution
    month_data = aggregated[aggregated["type"] == "month"]
    if len(month_data) > 0:
        # Sort by month in ascending order (1-12)
        month_data_sorted = month_data.sort_values("key", key=lambda x: x.astype(int))
        distributions["month_distribution"] = dict(
            zip(month_data_sorted["key"], month_data_sorted["count"])
        )

    # Day of week distribution
    dow_data = aggregated[aggregated["type"] == "dow"]
    if len(dow_data) > 0:
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        # Sort by day of week index (0=Monday to 6=Sunday) to maintain weekday order
        dow_data_sorted = dow_data.sort_values("key")
        distributions["day_of_week_distribution"] = {
            day_names[int(k)]: int(v)
            for k, v in zip(dow_data_sorted["key"], dow_data_sorted["count"])
        }

    # Decade distribution
    decade_data = aggregated[aggregated["type"] == "decade"]
    if len(decade_data) > 0:
        distributions["decade_distribution"] = dict(
            zip(decade_data["key"], decade_data["count"])
        )

    return distributions


def process_with_dask(
    ddf: dd.DataFrame,
    field_name: str,
    total_records: int,
    logger: logging.Logger,
    min_year: int = 1940,
    max_year: int = 2005,
    is_birth_date: bool = False,
    id_column: Optional[str] = None,
    uid_column: Optional[str] = None,
    npartitions: Optional[int] = 2,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
) -> Dict[str, Any]:
    """
    Process date field analysis using Dask distributed computing framework.

    This function leverages Dask's distributed computing capabilities to analyze
    large datasets efficiently by processing data across multiple partitions in parallel.
    It uses Dask's lazy evaluation and built-in aggregation functions for optimal performance.

    Parameters:
    -----------
    ddf : dd.DataFrame
        The DataFrame containing the data to analyze (preferably Dask for large datasets)
    field_name : str
        The name of the date field to analyze
    total_records : int
        Total number of records in the DataFrame
    npartitions : int
        Number of partitions in the Dask DataFrame (used for worker count logging)
    logger : logging.Logger
        Logger instance for tracking progress and debugging
    progress_tracker : Optional[HierarchicalProgressTracker]
        Optional progress tracker for reporting processing status

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results including:
        - Basic statistics: total_records, null_count, valid_count, fill_rate, etc.
        - Date range: min_date, max_date
        - Distributions: year, month, day of week, and decade distributions

    Raises:
    -------
    ImportError
        If Dask is not available, falls back gracefully with warning
    """
    try:
        # Log Dask processing configuration
        logger.info("Parallel Enabled")
        logger.info("Parallel Engine: Dask")
        logger.info(f"Parallel Workers: {npartitions}")

        # Prepare date data for analysis using lazy evaluation
        # This step converts the field to datetime and counts nulls without computing
        dates, null_count, non_null_count = prepare_date_data(ddf, field_name)

        # Count valid dates using Dask's distributed counting
        # .count() is computed immediately for statistics calculation
        valid_count = (
            dates.count().compute() if isinstance(dates, dd.Series) else dates.count()
        )
        invalid_count = non_null_count - valid_count

        # Calculate data quality metrics
        fill_rate = (
            round((non_null_count / total_records) * 100, 2) if total_records > 0 else 0
        )
        valid_rate = (
            round((valid_count / non_null_count) * 100, 2) if non_null_count > 0 else 0
        )

        # Initialize results dictionary with basic statistics
        result = {
            "total_records": total_records,
            "null_count": int(null_count),
            "non_null_count": int(non_null_count),
            "valid_count": int(valid_count),
            "invalid_count": int(invalid_count),
            "fill_rate": fill_rate,
            "valid_rate": valid_rate,
        }

        # Process date statistics and distributions only if we have valid dates
        if valid_count > 0:
            # Filter out null dates for further processing
            dates_filtered = dates.dropna()

            # Use Dask's lazy evaluation to prepare min/max computations
            # These create computation graphs without executing
            min_date_future = dates_filtered.min()
            max_date_future = dates_filtered.max()

            # Execute both aggregations together for efficiency
            # This minimizes data passes and network communication
            min_date, max_date = dd.compute(min_date_future, max_date_future)

            # Add date range to results if valid dates exist
            if not pd.isna(min_date) and not pd.isna(max_date):
                result.update(
                    {
                        "min_date": min_date.strftime("%Y-%m-%d"),
                        "max_date": max_date.strftime("%Y-%m-%d"),
                    }
                )

            # Calculate distributions using map_partitions for parallel processing
            # This applies partition_distributions to each partition independently
            dist_ddf = dates.map_partitions(
                partition_distributions,
                meta=pd.DataFrame(
                    {
                        "type": pd.Series(dtype="object"),
                        "key": pd.Series(dtype="object"),
                        "count": pd.Series(dtype="int64"),
                    }
                ),
            ).persist()  # Cache intermediate results in memory/disk

            # Compute the distribution data and aggregate results
            distributions_data = dist_ddf.compute()
            if not distributions_data.empty:
                # Aggregate and format distributions into expected output format
                result.update(aggregate_distributions_data(distributions_data))

            # Analyze anomalies
            anomalies = detect_date_anomalies(
                dates, min_year=min_year, max_year=max_year
            )
            result["anomalies"] = {k: len(v) for k, v in anomalies.items()}

            # Include examples of anomalies
            for anomaly_type, examples in anomalies.items():
                if examples:
                    result[f"{anomaly_type}_examples"] = examples[
                        :10
                    ]  # First 10 examples

        # Group analysis if id_column is specified (applies to all processing methods)
        if id_column and id_column in ddf.columns:
            group_changes = detect_date_changes_within_group(ddf, id_column, field_name)
            result["date_changes_within_group"] = group_changes

        # UID analysis if uid_column is specified (applies to all processing methods)
        if uid_column and uid_column in ddf.columns:
            uid_inconsistencies = detect_date_inconsistencies_by_uid(
                ddf, uid_column, field_name
            )
            result["date_inconsistencies_by_uid"] = uid_inconsistencies

        # For birth dates, calculate age distribution
        if is_birth_date:
            result.update(calculate_age_distribution(ddf, field_name))

    except ImportError:
        # Handle case where Dask is not available
        logger.warning(
            "Dask requested but not available. Falling back to chunked processing."
        )
        if progress_tracker:
            progress_tracker.update(
                0,
                {
                    "step": "Dask fallback",
                    "warning": "Dask not available, using chunks",
                },
            )
    return result


def process_with_vectorization(
    df: Union[pd.DataFrame, dd.DataFrame],
    field_name: str,
    total_records: int,
    chunk_size: int,
    parallel_processes: int,
    min_year: int,
    max_year: int,
    logger: logging.Logger,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
) -> Dict[str, Any]:
    """
    Process date field analysis using parallel vectorized processing with joblib.

    This function divides the DataFrame into chunks and processes them in parallel
    using joblib's Parallel execution engine to improve performance on large datasets.

    Parameters:
    -----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The DataFrame containing the data to analyze
    field_name : str
        The name of the date field to analyze
    total_records : int
        Total number of records in the DataFrame
    chunk_size : int
        Number of rows per chunk for parallel processing
    parallel_processes : int
        Number of parallel processes to use for processing
    min_year : int
        Minimum valid year for anomaly detection
    max_year : int
        Maximum valid year for anomaly detection
    logger : logging.Logger
        Logger instance for tracking progress and debugging
    progress_tracker : Optional[HierarchicalProgressTracker]
        Optional progress tracker for reporting processing status

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing aggregated analysis results including:
        - valid_count: Number of valid dates
        - min_date, max_date: Date range
        - distributions: Various date distributions (year, month, etc.)
        - anomalies: Anomaly counts and examples
    """
    # Log parallel processing configuration
    logger.info("Parallel Enabled")
    logger.info("Parallel Engine: Joblib")
    logger.info(f"Parallel Workers: {parallel_processes}")

    # Prepare date data for analysis (convert to datetime, count nulls)
    dates, null_count, non_null_count = prepare_date_data(df, field_name)

    # Update progress tracker with setup information
    if progress_tracker:
        progress_tracker.update(
            0,
            {
                "step": "Parallel processing setup",
                "processes": parallel_processes,
                "chunk_size": chunk_size,
            },
        )

    # Initialize result dictionary
    result = {}

    # Split DataFrame into chunks for parallel processing
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, total_records, chunk_size)]
    logger.info(
        f"Vectorized parallel processing with {parallel_processes} workers and {len(chunks)} chunks"
    )

    # Update progress tracker with chunk information
    if progress_tracker:
        progress_tracker.update(
            0, {"step": "Processing chunks", "total_chunks": len(chunks)}
        )

    # Execute parallel processing using joblib
    # Each chunk is processed independently by process_date_chunk function
    processed = Parallel(n_jobs=parallel_processes)(
        joblib_delayed(process_date_chunk)(i, chunk, field_name, min_year, max_year)
        for i, chunk in enumerate(chunks)
    )

    # Update progress tracker for aggregation phase
    if progress_tracker:
        progress_tracker.update(0, {"step": "Aggregating parallel results"})

    # Aggregate results from all processed chunks
    agg = aggregate_chunk_results(processed)
    if "error" not in agg:
        result.update(agg)
        logger.info("Parallel processing done")
    else:
        logger.warning(f"Parallel error: {agg['error']}")

    return result


def process_with_chunks(
    df,
    field_name,
    total_records,
    chunk_size,
    valid_count,
    logger,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
):
    logger.info(
        f"Processing large dataset with {total_records} rows using chunked processing"
    )

    if progress_tracker:
        progress_tracker.update(
            0,
            {
                "step": "Chunked processing setup",
                "total_rows": total_records,
                "chunk_size": chunk_size,
            },
        )

    result = {}
    total_chunks = (total_records + chunk_size - 1) // chunk_size
    logger.info(f"Processing {total_chunks} chunks sequentially")

    # Process chunks sequentially
    stats, distributions = [], []
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_records)
        chunk = df.iloc[start_idx:end_idx]

        if progress_tracker:
            progress_tracker.update(
                0,
                {
                    "step": f"Processing chunk {i + 1}/{total_chunks}",
                    "chunk_start": start_idx,
                    "chunk_end": end_idx,
                },
            )

        logger.debug(
            f"Processing chunk {i + 1}/{total_chunks} (rows {start_idx}-{end_idx})"
        )

        chunk_dates = pd.to_datetime(chunk[field_name], errors="coerce")
        chunk_dates = pd.Series(chunk_dates.values, index=chunk_dates.index)
        valid_chunk_mask = ~chunk_dates.isna()
        valid_chunk_dates = chunk_dates[valid_chunk_mask]

        if len(valid_chunk_dates) > 0:
            stats.append(
                {
                    "min_date": valid_chunk_dates.min(),
                    "max_date": valid_chunk_dates.max(),
                    "valid_count": len(valid_chunk_dates),
                }
            )
            distributions.append(calculate_distributions(chunk_dates))

    if stats and valid_count > 0:
        min_dt = min(s["min_date"] for s in stats)
        max_dt = max(s["max_date"] for s in stats)
        result.update(
            {
                "valid_count": int(valid_count),
                "min_date": min_dt.strftime("%Y-%m-%d"),
                "max_date": max_dt.strftime("%Y-%m-%d"),
            }
        )

        aggregated_distributions = {}
        for dist in distributions:
            for dtype, ddata in dist.items():
                if dtype not in aggregated_distributions:
                    aggregated_distributions[dtype] = {}
                for key, count in ddata.items():
                    aggregated_distributions[dtype][key] = (
                        aggregated_distributions[dtype].get(key, 0) + count
                    )

        result.update(aggregated_distributions)

    if progress_tracker:
        progress_tracker.update(
            0, {"step": "Chunked processing completed", "total_chunks": total_chunks}
        )

    logger.info(f"Chunked processing completed successfully")

    return result


def analyze_date_field(
    df: Union[pd.DataFrame, dd.DataFrame],
    field_name: str,
    min_year: int = 1940,
    max_year: int = 2005,
    is_birth_date: bool = False,
    id_column: Optional[str] = None,
    uid_column: Optional[str] = None,
    chunk_size: int = 10000,
    use_dask: bool = False,
    use_vectorization: bool = False,
    npartitions: Optional[int] = 2,
    parallel_processes: Optional[int] = 1,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a date field in a DataFrame, including statistics, distributions, anomaly detection, and optional group/UID analysis.

    Parameters:
    -----------
     df : Union[pd.DataFrame, dd.DataFrame]
        The DataFrame containing the data to analyze.
    field_name : str
        The name of the date field to analyze.
    min_year : int, optional
        Minimum valid year for anomaly detection (default: 1940).
    max_year : int, optional
        Maximum valid year for anomaly detection (default: 2005).
    id_column : str, optional
        The column to use for group analysis (e.g., to detect date changes within groups).
    uid_column : str, optional
        The column to use for UID analysis (e.g., to detect inconsistencies by unique identifier).
    chunk_size : int, optional
        The number of rows per chunk for chunked or parallel processing (default: 10000).
    use_dask : bool, optional
        Whether to use Dask for large DataFrame processing (default: False).
    use_vectorization : bool, optional
        Whether to use vectorized parallel processing (default: False).
    parallel_processes : int, optional
        Number of parallel processes to use if vectorization is enabled (default: 1).
    progress_tracker : HierarchicalProgressTracker, optional
        Optional progress tracker for reporting progress.
    task_logger : Optional[logging.Logger]
        Logger for tracking task progress and debugging.
    **kwargs : dict
        Additional keyword arguments for advanced configuration (e.g., npartitions for Dask).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results, including:
            - total_records, null_count, non_null_count, valid_count, invalid_count
            - fill_rate, valid_rate
            - min_date, max_date
            - year_distribution, month_distribution, day_of_week_distribution, decade_distribution
            - anomalies (counts and examples)
            - date_changes_within_group (if id_column specified)
            - date_inconsistencies_by_uid (if uid_column specified)
    """
    global logger
    if task_logger:
        logger = task_logger

    logger.info(f"Analyzing date field: {field_name}")

    # Basic validation
    if field_name not in df.columns:
        return {"error": f"Field {field_name} not found in DataFrame"}

    total_records = (
        int(df.map_partitions(len).sum().compute())
        if isinstance(df, dd.DataFrame)
        else len(df)
    )
    is_large_df = total_records > chunk_size
    results = {}

    if is_large_df is False:
        logger.warning("Small DataFrame! Process as usual")
        # Calculate basic date statistics
        basic_stats = calculate_basic_date_stats(
            get_computed_df(df), field_name, total_records
        )
        dates = basic_stats.pop("dates")  # Extract dates for further processing
        results = basic_stats

        if results["valid_count"] > 0:
            # Calculate date range and distributions if we have valid dates
            # Get date range
            # Ensure dates is a pandas Series before passing to calculate_date_stats
            if isinstance(dates, dd.Series):
                date_stats = calculate_date_stats(dates.compute())
            else:
                date_stats = calculate_date_stats(dates)
            results.update(date_stats)  # Calculate distributions
            # Ensure dates is a pandas Series before passing to calculate_distributions
            distributions = calculate_distributions(dates)

            results.update(distributions)

    if use_dask and is_large_df:
        # Get npartitions from kwargs or use a default based on DataFrame partitions
        return process_with_dask(
            ddf=df,
            field_name=field_name,
            total_records=total_records,
            logger=logger,
            min_year=min_year,
            max_year=max_year,
            is_birth_date=is_birth_date,
            id_column=id_column,
            uid_column=uid_column,
            npartitions=npartitions,
            progress_tracker=progress_tracker,
        )

    elif use_vectorization and parallel_processes and parallel_processes > 0:
        basic_stats = calculate_basic_date_stats(
            get_computed_df(df), field_name, total_records
        )
        dates = basic_stats.pop("dates")  # Extract dates for further processing
        results = basic_stats

        results.update(
            process_with_vectorization(
                df,
                field_name,
                total_records,
                chunk_size,
                parallel_processes,
                min_year,
                max_year,
                logger,
                progress_tracker,
            )
        )
    elif is_large_df:
        basic_stats = calculate_basic_date_stats(
            get_computed_df(df), field_name, total_records
        )
        dates = basic_stats.pop("dates")  # Extract dates for further processing
        results = basic_stats

        results.update(
            process_with_chunks(
                df,
                field_name,
                total_records,
                chunk_size,
                results["valid_count"],
                logger,
                progress_tracker,
            )
        )

    if results["valid_count"] > 0:
        # Analyze anomalies
        anomalies = detect_date_anomalies(dates, min_year=min_year, max_year=max_year)
        results["anomalies"] = {k: len(v) for k, v in anomalies.items()}

        # Include examples of anomalies
        for anomaly_type, examples in anomalies.items():
            if examples:
                results[f"{anomaly_type}_examples"] = examples[:10]  # First 10 examples

    # Group analysis if id_column is specified (applies to all processing methods)
    if id_column and id_column in df.columns:
        group_changes = detect_date_changes_within_group(
            get_computed_df(df), id_column, field_name
        )
        results["date_changes_within_group"] = group_changes

    # UID analysis if uid_column is specified (applies to all processing methods)
    if uid_column and uid_column in df.columns:
        uid_inconsistencies = detect_date_inconsistencies_by_uid(
            get_computed_df(df), uid_column, field_name
        )
        results["date_inconsistencies_by_uid"] = uid_inconsistencies

    # For birth dates, calculate age distribution
    if is_birth_date:
        results.update(calculate_age_distribution(get_computed_df(df), field_name))

    return results


def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """
    Estimate resources needed for date field analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field to analyze

    Returns:
    --------
    Dict[str, Any]
        Estimated resource requirements
    """
    # Basic resource estimation based on DataFrame size
    row_count = len(df)

    # Memory estimation (rough approximation)
    if field_name in df.columns:
        # Estimate based on field type and non-null values
        non_null_count = df[field_name].notna().sum()
        bytes_per_value = 8  # 8 bytes for datetime64

        # Base memory for analysis
        base_memory_mb = 30

        # Memory for field data
        field_memory_mb = (non_null_count * bytes_per_value) / (1024 * 1024)

        # Memory for intermediate calculations
        calc_memory_mb = (
            field_memory_mb * 2
        )  # Multiplication factor for intermediate calculations

        # Total estimated memory
        estimated_memory_mb = base_memory_mb + field_memory_mb + calc_memory_mb

        # Estimated time (very rough approximation)
        if row_count < 10000:
            estimated_time_seconds = 1
        elif row_count < 100000:
            estimated_time_seconds = 3
        elif row_count < 1000000:
            estimated_time_seconds = 15
        else:
            estimated_time_seconds = 60

        return {
            "estimated_memory_mb": estimated_memory_mb,
            "estimated_time_seconds": estimated_time_seconds,
            "recommended_chunk_size": min(100000, max(10000, row_count // 10)),
            "use_chunks_recommended": row_count > 100000,
        }
    else:
        # Field not found, return minimal estimates
        return {
            "estimated_memory_mb": 10,
            "estimated_time_seconds": 1,
            "error": f"Field {field_name} not found in DataFrame",
        }


def process_date_chunk(
    chunk_index: int,
    chunk_data: pd.DataFrame,
    field_name: str,
    min_year: int = 1940,
    max_year: int = 2005,
) -> Dict[str, Any]:
    """
    Process a single chunk of data for date analysis using vectorization.

    Parameters:
    -----------
    chunk_index : int
        Index of the chunk being processed
    chunk_data : pd.DataFrame
        The chunk of data to process
    field_name : str
        The name of the date field to analyze
    min_year : int
        Minimum valid year for anomaly detection
    max_year : int
        Maximum valid year for anomaly detection

    Returns:
    --------
    Dict[str, Any]
        Results from processing this chunk
    """
    if field_name not in chunk_data.columns:
        return {
            "chunk_index": chunk_index,
            "error": f"Field {field_name} not found in chunk",
            "valid_count": 0,
            "distributions": {},
            "anomalies": {},
        }

    # Convert to datetime for this chunk
    chunk_dates = pd.to_datetime(chunk_data[field_name], errors="coerce")
    chunk_dates = pd.Series(chunk_dates.values, index=chunk_dates.index)
    valid_mask = ~chunk_dates.isna()
    valid_dates = chunk_dates[valid_mask]

    chunk_result = {
        "chunk_index": chunk_index,
        "valid_count": len(valid_dates),
        "distributions": {},
        "anomalies": {},
        "date_stats": {},
    }

    if len(valid_dates) == 0:
        return chunk_result

    # Calculate date stats for this chunk
    min_date = valid_dates.min()
    max_date = valid_dates.max()
    chunk_result["date_stats"] = {
        "min_date": min_date,
        "max_date": max_date,
        "valid_count": len(valid_dates),
    }

    # Calculate distributions for this chunk using vectorized operations
    distributions = {}

    # Year distribution
    year_counts = valid_dates.dt.year.value_counts()
    distributions["year"] = year_counts.to_dict()

    # Month distribution
    month_counts = valid_dates.dt.month.value_counts()
    distributions["month"] = month_counts.to_dict()

    # Day of week distribution
    dow_counts = valid_dates.dt.dayofweek.value_counts()
    distributions["dow"] = dow_counts.to_dict()

    # Decade distribution
    decade_counts = (valid_dates.dt.year // 10 * 10).value_counts()
    distributions["decade"] = decade_counts.to_dict()

    chunk_result["distributions"] = distributions

    # Anomaly detection using vectorized operations
    anomalies = {
        "invalid_format": 0,
        "too_old": 0,
        "future_dates": 0,
        "too_young": 0,
        "negative_years": 0,
    }

    # Check for anomalies using vectorized operations
    current_year = datetime.now().year
    years = valid_dates.dt.year

    # Count anomalies
    anomalies["too_old"] = (years < min_year).sum()
    anomalies["future_dates"] = (years > current_year).sum()
    anomalies["too_young"] = ((years > max_year) & (years <= current_year)).sum()

    # Check for invalid formats (non-null values that couldn't be converted)
    non_null_mask = chunk_data[field_name].notna()
    invalid_mask = non_null_mask & chunk_dates.isna()
    anomalies["invalid_format"] = invalid_mask.sum()

    # Check for negative years (simple string check for performance)
    if invalid_mask.any():
        negative_strings = (
            chunk_data.loc[invalid_mask, field_name].astype(str).str.startswith("-")
        )
        anomalies["negative_years"] = negative_strings.sum()
        anomalies["invalid_format"] -= anomalies["negative_years"]  # Adjust count

    chunk_result["anomalies"] = anomalies

    return chunk_result


def aggregate_chunk_results(chunk_results) -> Dict[str, Any]:
    """
    Aggregate results from parallel chunk processing.

    Parameters:
    -----------
    chunk_results : List[Dict[str, Any]]
        List of results from each processed chunk

    Returns:
    --------
    Dict[str, Any]
        Aggregated results in the expected format
    """
    if not chunk_results:
        return {}

    # Filter out error chunks
    valid_chunks = [chunk for chunk in chunk_results if "error" not in chunk]

    if not valid_chunks:
        return {"error": "No valid chunks processed"}

    # Aggregate date stats
    min_dates = []
    max_dates = []
    total_valid_count = 0

    for chunk in valid_chunks:
        if chunk["valid_count"] > 0 and "date_stats" in chunk:
            min_dates.append(chunk["date_stats"]["min_date"])
            max_dates.append(chunk["date_stats"]["max_date"])
            total_valid_count += chunk["valid_count"]

    aggregated = {}

    if min_dates and max_dates:
        overall_min = min(min_dates)
        overall_max = max(max_dates)
        aggregated.update(
            {
                "valid_count": total_valid_count,
                "min_date": (
                    overall_min.strftime("%Y-%m-%d")
                    if not pd.isna(overall_min)
                    else None
                ),
                "max_date": (
                    overall_max.strftime("%Y-%m-%d")
                    if not pd.isna(overall_max)
                    else None
                ),
            }
        )

    # Aggregate distributions
    combined_distributions = {"year": {}, "month": {}, "dow": {}, "decade": {}}

    for chunk in valid_chunks:
        distributions = chunk.get("distributions", {})
        for dist_type in combined_distributions:
            if dist_type in distributions:
                for key, count in distributions[dist_type].items():
                    combined_distributions[dist_type][key] = (
                        combined_distributions[dist_type].get(key, 0) + count
                    )

    #  Convert to expected format (sorted)
    if combined_distributions["year"]:
        aggregated["year_distribution"] = {
            str(k): int(v)
            for k, v in sorted(
                combined_distributions["year"].items(), key=lambda x: int(x[0])
            )
        }

    if combined_distributions["month"]:
        aggregated["month_distribution"] = {
            str(k): int(v)
            for k, v in sorted(
                combined_distributions["month"].items(), key=lambda x: int(x[0])
            )
        }

    if combined_distributions["dow"]:
        # Sort by day index (0=Monday, 6=Sunday)
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        sorted_dow = sorted(
            combined_distributions["dow"].items(), key=lambda x: int(x[0])
        )
        aggregated["day_of_week_distribution"] = {
            day_names[int(k)]: int(v) for k, v in sorted_dow
        }

    if combined_distributions["decade"]:
        aggregated["decade_distribution"] = {
            f"{k}s": int(v)
            for k, v in sorted(
                combined_distributions["decade"].items(), key=lambda x: int(x[0])
            )
        }

    # Aggregate anomalies
    combined_anomalies = {
        "invalid_format": 0,
        "too_old": 0,
        "future_dates": 0,
        "too_young": 0,
        "negative_years": 0,
    }

    for chunk in valid_chunks:
        anomalies = chunk.get("anomalies", {})
        for anomaly_type in combined_anomalies:
            if anomaly_type in anomalies:
                combined_anomalies[anomaly_type] += anomalies[anomaly_type]

    aggregated["anomalies"] = combined_anomalies

    return aggregated


def _calculate_age_distribution_pandas(
    df: pd.DataFrame, field_name: str, today
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate age distribution using Pandas processing.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame
    field_name : str
        The name of the birth date field
    today : date
        Today's date for age calculation

    Returns:
    -------
    Tuple[pd.Series, pd.Series]
        Tuple containing (ages, age_distribution)
    """
    valid_dates = pd.to_datetime(df[field_name], errors="coerce")
    valid_mask = ~valid_dates.isna()

    if valid_mask.sum() == 0:
        return pd.Series(dtype="int64"), pd.Series(dtype="int64")

    birth_dates = valid_dates[valid_mask].dt.date
    ages = birth_dates.apply(
        lambda d: today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    )
    ages = ages[ages >= 0]

    if ages.empty:
        return pd.Series(dtype="int64"), pd.Series(dtype="int64")

    age_groups = ages.apply(lambda a: f"{5 * (a // 5)}-{5 * (a // 5) + 4}")
    age_distribution = age_groups.value_counts().sort_index(
        key=lambda x: x.map(lambda g: int(g.split("-")[0]))
    )

    return ages, age_distribution


def _calculate_age_distribution_dask(
    df: Union[pd.DataFrame, dd.DataFrame], field_name: str, today
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate age distribution using Dask distributed processing.

    Parameters:
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The input DataFrame
    field_name : str
        The name of the birth date field
    today : date
        Today's date for age calculation

    Returns:
    -------
    Tuple[pd.Series, pd.Series]
        Tuple containing (ages, age_distribution)
    """
    if isinstance(df, pd.DataFrame):
        df = dd.from_pandas(df, npartitions=4)

    # Convert to datetime
    df[field_name] = dd.to_datetime(df[field_name], errors="coerce")

    # Drop nulls and compute age
    df = df.dropna(subset=[field_name])
    df["age"] = df[field_name].map_partitions(
        lambda part: part.dt.date.apply(
            lambda d: today.year
            - d.year
            - ((today.month, today.day) < (d.month, d.day))
        ),
        meta=("age", "int"),
    )

    # Filter non-negative ages
    df = df[df["age"] >= 0]

    # Age group column
    df["age_group"] = df["age"].map(
        lambda a: f"{5 * (a // 5)}-{5 * (a // 5) + 4}", meta=("age_group", "str")
    )

    # Group by age_group and count
    age_distribution = (
        df.groupby("age_group")
        .size()
        .compute()
        .sort_index(key=lambda s: s.map(lambda x: int(x.split("-")[0])))
    )

    # Stats
    ages = df["age"].compute()

    return ages, age_distribution


def calculate_age_distribution(
    df: Union[pd.DataFrame, dd.DataFrame], field_name: str, use_dask: bool = False
) -> Dict[str, Any]:
    """
    Calculate age distribution from birth dates using Pandas or Dask.

    Parameters:
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The input DataFrame
    field_name : str
        The name of the birth date field
    use_dask : bool
        Whether to use Dask for distributed computation

    Returns:
    -------
    Dict[str, Any]
        Dictionary with age distribution and statistics
    """
    today = datetime.now().date()

    # Call appropriate processing function based on use_dask flag
    if use_dask:
        ages, age_distribution = _calculate_age_distribution_dask(df, field_name, today)
    else:
        # Convert to pandas DataFrame if it's a Dask DataFrame for pandas processing
        if isinstance(df, dd.DataFrame):
            df_pandas = df.compute()
        else:
            df_pandas = df
        ages, age_distribution = _calculate_age_distribution_pandas(
            df_pandas, field_name, today
        )

    # Check if we have valid results
    if ages.empty:
        return {
            "age_distribution": {},
            "age_statistics": {
                "min_age": None,
                "max_age": None,
                "mean_age": None,
                "median_age": None,
            },
        }

    # Compute statistics
    sorted_ages = sorted(ages)
    n = len(sorted_ages)
    median_age = (
        sorted_ages[n // 2]
        if n % 2 != 0
        else (sorted_ages[n // 2 - 1] + sorted_ages[n // 2]) / 2
    )

    return {
        "age_distribution": dict(age_distribution),
        "age_statistics": {
            "min_age": min(ages),
            "max_age": max(ages),
            "mean_age": sum(ages) / n,
            "median_age": median_age,
        },
    }


def calculate_basic_date_stats(
    df: pd.DataFrame, field_name: str, total_records: int
) -> Dict[str, Any]:
    """
    Calculate basic date field statistics including counts and rates.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the date field to analyze
    total_records : int
        Total number of records in the DataFrame

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing basic statistics:
        - total_records, null_count, non_null_count, valid_count, invalid_count
        - fill_rate, valid_rate
        - dates: the prepared date series
    """
    # Get prepared data
    dates, null_count, non_null_count = prepare_date_data(df, field_name)

    valid_mask = ~dates.isna()
    valid_count = valid_mask.sum()
    invalid_count = non_null_count - valid_count

    # Calculate fill and validity rates
    fill_rate = (
        round((non_null_count / total_records) * 100, 2) if total_records > 0 else 0
    )
    valid_rate = (
        round((valid_count / non_null_count) * 100, 2) if non_null_count > 0 else 0
    )

    # Return results with basic stats
    return {
        "total_records": total_records,
        "null_count": int(null_count),
        "non_null_count": int(non_null_count),
        "valid_count": int(valid_count),
        "invalid_count": int(invalid_count),
        "fill_rate": fill_rate,
        "valid_rate": valid_rate,
        "dates": dates,  # Include the prepared dates for further processing
    }
