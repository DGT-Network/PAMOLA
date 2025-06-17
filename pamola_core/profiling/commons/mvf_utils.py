"""
Utility functions for analyzing multi-valued fields (MVF) in the project.

This module provides pure analytical functions for MVF analysis,
separate from operation logic, focusing on parsing, metrics calculation,
pattern extraction, and data preparation.

MVF fields contain multiple values per record, typically stored as:
- String representations of arrays: "['Value1', 'Value2']"
- JSON arrays: ["Value1", "Value2"]
- Comma-separated values: "Value1, Value2"
"""

import ast
from itertools import chain
import json
import logging
from collections import Counter
from pathlib import Path
import pickle
import time
from typing import Dict, List, Any, Tuple, Optional, Union

from joblib import Parallel, delayed
import pandas as pd

from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import plot_value_distribution

# Configure logger
logger = logging.getLogger(__name__)


def parse_mvf(
    value: Any,
    format_type: Optional[str] = None,
    separator: str = ",",
    quote_char: str = '"',
    array_markers: Tuple[str, str] = ("[", "]"),
    handle_json: bool = True,
) -> List[str]:
    """
    Parse a multi-valued field value into a list of individual values.

    Parameters:
    -----------
    value : Any
        The MVF value to parse
    format_type : str, optional
        Format type hint: 'list', 'json', 'array_string', 'csv', or None (auto-detect)
    separator : str
        Character used to separate values
    quote_char : str
        Character used for quoting values
    array_markers : Tuple[str, str]
        Start and end markers for array representation
    handle_json : bool
        Whether to attempt parsing as JSON

    Returns:
    --------
    List[str]
        List of individual values
    """

    # 1. Handle null / empty values
    if pd.isna(value) or value in ("", "[]", "None", "nan"):
        return []

    # 2. If the value is already a list and format_type is 'list'
    if format_type == "list":
        if isinstance(value, list):
            return [str(item).strip() for item in value]
        else:
            logger.warning(
                f"Expected list for format_type='list' but got {type(value)}: {value}"
            )
            return [str(value)]

    # 3. If not string, return as single value list
    if not isinstance(value, str):
        return [str(value)]

    value = value.strip()

    # 4. Use format_type hint if provided
    if format_type == "json":
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
            elif isinstance(parsed, dict):
                return [str(k).strip() for k in parsed.keys()]
            else:
                return [str(parsed)]
        except Exception:
            logger.warning(f"Failed to parse JSON for value: {value}")
            return [value]

    elif format_type == "array_string":
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except Exception:
            logger.warning(f"Failed to parse array string for value: {value}")
            return manual_parse_array_string(value, separator, quote_char)

    elif format_type == "csv":
        return [item.strip() for item in value.split(separator) if item.strip()]

    # 5. Auto-detect parsing if no format_type or failed above
    if handle_json and (
        (value.startswith("{") and value.endswith("}"))
        or (value.startswith("[") and value.endswith("]"))
    ):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
            elif isinstance(parsed, dict):
                return [str(k).strip() for k in parsed.keys()]
            else:
                return [str(parsed)]
        except Exception:
            pass  # fallback

    if value.startswith(array_markers[0]) and value.endswith(array_markers[1]):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except Exception:
            return manual_parse_array_string(value, separator, quote_char)

    # 6. Fallback: separator-based splitting
    return [item.strip() for item in value.split(separator) if item.strip()]


def manual_parse_array_string(value: str, separator: str, quote_char: str) -> List[str]:
    """
    Manually parse a value that looks like an array string, e.g., "['a', 'b']".

    Parameters:
    -----------
    value : str
        The value to parse
    separator : str
        Separator used to split values
    quote_char : str
        Quote character used for wrapping values

    Returns:
    --------
    List[str]
    """
    inner = value[1:-1].strip()
    if not inner:
        return []

    result = []
    in_quotes = False
    current = ""

    for char in inner:
        if char == quote_char:
            in_quotes = not in_quotes
        elif char == separator and not in_quotes:
            result.append(current.strip().strip(quote_char))
            current = ""
        else:
            current += char

    if current:
        result.append(current.strip().strip(quote_char))

    return result


def detect_mvf_format(values: List[Any]) -> str:
    """
    Detect the most likely format of MVF values in a sample.

    Parameters:
    -----------
    values : List[Any]
        A list of sample multi-valued field (MVF) values to analyze.
        Each item can be a string, list, or other serializable format.

    Returns:
    --------
    str
        Detected format of the MVF values. One of:
        - 'json': JSON-style list string, e.g., '["a", "b"]'
        - 'array_string': Python-style list string, e.g., "['a', 'b']"
        - 'csv': Comma-separated values, e.g., "a,b,c"
        - 'list': Python list object, e.g., ['a', 'b']
        - 'unknown': Could not determine the format reliably
    """
    format_counts = Counter()
    sample_values = [v for v in values if not pd.isna(v)][:100]

    for value in sample_values:
        fmt = detect_single_format(value)
        format_counts[fmt] += 1

    if format_counts:
        most_common, count = format_counts.most_common(1)[0]
        if count > len(sample_values) * 0.5:
            return most_common

    return "unknown"


def detect_single_format(value: Any) -> str:
    """
    Detect format of a single MVF value.

    Parameters:
    -----------
    value : Any
        A single MVF value to check. Can be string, list, or other.

    Returns:
    --------
    str
        One of: 'json', 'array_string', 'csv', 'list', or 'unknown'
    """
    # Direct Python list object
    if isinstance(value, list):
        return "list"

    if not isinstance(value, str):
        return "unknown"

    value = value.strip()

    # JSON-style list: ["a", "b"]
    if value.startswith("[") and value.endswith("]") and '"' in value and "," in value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return "json"
        except Exception:
            pass

    # Python-style list string: ['a', 'b']
    if value.startswith("[") and value.endswith("]") and "'" in value and "," in value:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return "array_string"
        except Exception:
            pass

    # CSV-style: a,b,c
    if "," in value and not (value.startswith("[") and value.endswith("]")):
        return "csv"

    return "unknown"


def standardize_mvf_format(
    value: Any, target_format: str = "list"
) -> Union[List[str], str]:
    """
    Standardize an MVF value to a specified format.

    Parameters:
    -----------
    value : Any
        The MVF value to standardize
    target_format : str
        Target format: 'list', 'json', 'csv', or 'array_string'

    Returns:
    --------
    Union[List[str], str]
        Standardized MVF value
    """
    # Parse the MVF value to get a list of values
    values = parse_mvf(value)

    # Return in the target format
    if target_format == "list":
        return values
    elif target_format == "json":
        return json.dumps(values)
    elif target_format == "csv":
        return ", ".join(values)
    elif target_format == "array_string":
        formatted_values = []
        for v in values:
            # Properly escape single quotes in the values
            escaped_v = v.replace("'", "\\'")
            formatted_values.append(f"'{escaped_v}'")
        return f"[{', '.join(formatted_values)}]"
    else:
        logger.warning(f"Unknown target format: {target_format}. Returning list.")
        return values


def _analyze_chunk(
    chunk: pd.Series, parse_args: Dict[str, Any]
) -> Tuple[List[str], List[int], List[Tuple[str, ...]], int, int]:
    """
    Analyze a chunk of MVF (multi-valued field) data.

    Parameters:
    -----------
    chunk : pd.Series
        The chunk of data to analyze
    parse_args : dict
        Additional arguments for the MVF parser

    Returns:
    --------
    Tuple[
        parsed_values: List[str],
        value_counts: List[int],
        combinations: List[Tuple[str]],
        empty_arrays_count: int,
        error_count: int
    ]
    """
    parsed_values = []
    value_counts = []
    combinations = []
    empty_arrays_count = 0
    error_count = 0

    for value in chunk:
        try:
            values = parse_mvf(value, **parse_args)
            parsed_values.extend(values)
            value_counts.append(len(values))
            if not values:
                empty_arrays_count += 1
            combinations.append(tuple(sorted(values)))
        except Exception as e:
            error_count += 1
            if error_count <= 10:
                logger.warning(f"Error parsing MVF value '{value}': {str(e)}")
            elif error_count == 11:
                logger.warning(
                    "Too many parsing errors. Further errors will not be logged."
                )
            value_counts.append(0)
            combinations.append(tuple())

    return parsed_values, value_counts, combinations, empty_arrays_count, error_count


def analyze_mvf_in_chunks(
    df: pd.DataFrame,
    field_name: str,
    top_n: int,
    parse_args: Dict[str, Any] = {},
    chunk_size: int = 10000,
    progress_tracker: Optional["HierarchicalProgressTracker"] = None,
    task_logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Analyze a multi-valued field (MVF) in chunks for large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    top_n : int
        Number of top items to include in the analysis
    format_type : Optional[str]
        Format type passed to the parser (e.g., 'json', 'string', etc.)
    parse_args : dict
        Additional parameters for parsing
    chunk_size : int
        Size of each chunk to process
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring
    task_logger : Optional[logging.Logger]
        Logger to track errors and progress

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    logger = task_logger or logging.getLogger(__name__)
    logger.info(f"Chunked analysis started for MVF field: {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return None

    if progress_tracker:
        progress_tracker.update(1, {"step": "Setting up chunked processing"})

    try:
        start_time = time.time()

        total_records = len(df)
        null_count = df[field_name].isna().sum()
        non_null_count = total_records - null_count
        null_percentage = (
            round((null_count / total_records) * 100, 2) if total_records else 0
        )

        parsed_values = []
        value_counts = []
        combinations = []
        empty_arrays_count = 0
        error_count = 0

        non_null_series = df[field_name].dropna()
        total_chunks = (len(non_null_series) + chunk_size - 1) // chunk_size

        if progress_tracker:
            progress_tracker.update(
                2, {"step": "Processing chunks", "total_chunks": total_chunks}
            )
        logger.info(
            f"Processing {len(non_null_series)} non-null records in {total_chunks} chunks of size {chunk_size}"
        )

        for i in range(total_chunks):
            if progress_tracker:
                progress_tracker.update(
                    2, {"step": f"Processing chunk {i+1}/{total_chunks}"}
                )

            chunk = non_null_series.iloc[i * chunk_size : (i + 1) * chunk_size]
            try:
                parsed, counts, combs, empty_count, errors = _analyze_chunk(
                    chunk, parse_args
                )

                parsed_values.extend(parsed)
                value_counts.extend(counts)
                combinations.extend(combs)
                empty_arrays_count += empty_count
                error_count += errors

                if error_count > 1000:
                    logger.error(
                        f"Too many parsing errors (>1000) in field {field_name}"
                    )
                    return None

            except Exception as e:
                logger.error(
                    f"Error analyzing MVF field {field_name} in chunk {i+1}: {str(e)}",
                    exc_info=True,
                )
                return None

        unique_values_count = len(set(parsed_values))
        avg_values_per_record = (
            round(sum(value_counts) / len(value_counts), 2) if value_counts else 0
        )
        max_values_per_record = max(value_counts) if value_counts else 0

        value_counter = Counter(parsed_values)
        combination_counter = Counter(combinations)
        value_counts_distribution = Counter(value_counts)

        stats = {
            "field_name": field_name,
            "total_records": total_records,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "non_null_count": non_null_count,
            "empty_arrays_count": empty_arrays_count,
            "empty_arrays_percentage": (
                round((empty_arrays_count / non_null_count) * 100, 2)
                if non_null_count
                else 0
            ),
            "unique_values": unique_values_count,
            "unique_combinations": len(combination_counter),
            "avg_values_per_record": avg_values_per_record,
            "max_values_per_record": max_values_per_record,
            "values_analysis": dict(value_counter.most_common(top_n)),
            "combinations_analysis": {
                ", ".join(combo) if combo else "Empty": count
                for combo, count in combination_counter.most_common(top_n)
            },
            "value_counts_distribution": dict(
                sorted(value_counts_distribution.items())
            ),
        }

        if error_count > 0:
            stats["error_count"] = error_count
            stats["error_percentage"] = round((error_count / total_records) * 100, 2)

        if progress_tracker:
            progress_tracker.update(
                3, {"step": "Chunked MVF analysis complete", "field": field_name}
            )

        elapsed_time = time.time() - start_time
        logger.info(
            f"Analysis performed in chunks completed in {elapsed_time:.2f} seconds"
        )
        stats["note"] = "Analysis performed in chunks for large dataset."

        return stats
    except Exception as e:
        logger.error(
            f"Error analyzing MVF field {field_name} in chunks: {str(e)}",
            exc_info=True,
        )
        return None


def analyze_mvf_field_with_parallel(
    df: pd.DataFrame,
    field_name: str,
    top_n: int,
    parse_args: Dict[str, Any] = {},
    chunk_size: int = 10000,
    n_jobs: int = -1,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Analyze a multi-valued field in parallel using joblib for chunked processing.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The field to analyze
    top_n : int
        Number of top items to include in the result
    format_type : str, optional
        Format type for the analysis (default: None)
    parse_args : dict
        Parsing configuration arguments passed to `parse_mvf`
    chunk_size : int
        Size of each chunk to process (default: 10000)
    n_jobs : int
        Number of parallel jobs (default -1 uses all CPUs)
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the analysis progress
    task_logger : Optional[logging.Logger]
        Logger for task-specific logging

    Returns:
    --------
    Dict[str, Any]
        Analysis results
    """
    if task_logger:
        logger = task_logger

    logger.info(f"Analyzing MVF field (parallel): {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return None

    try:
        # Initialize start time for performance tracking
        start_time = time.time()

        # Estimate total records
        total_records = len(df)
        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Parallel processing setup",
                    "n_jobs": n_jobs,
                    "total_records": total_records,
                },
            )

        null_count = df[field_name].isna().sum()
        non_null_count = total_records - null_count
        null_percentage = (
            round((null_count / total_records) * 100, 2) if total_records > 0 else 0
        )

        non_null_series = df[field_name].dropna()
        
        # Split into chunks
        chunks = [
            non_null_series.iloc[i : i + chunk_size].copy(deep=True)
            for i in range(0, len(non_null_series), chunk_size)
        ]

        logger.info(f"Processing {total_records} rows in Parallel with Joblib")

        # Update progress for Joblib processing
        if progress_tracker:
            progress_tracker.update(
                2,
                {
                    "step": "Joblib MVF processing",
                    "n_jobs": n_jobs,
                    "total_records": total_records,
                },
            )

        # Process chunks in parallel
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_analyze_chunk)(chunk, parse_args) for chunk in chunks
        )

        # Aggregate results
        all_parsed_values = list(chain.from_iterable(r[0] for r in results))
        all_value_counts = list(chain.from_iterable(r[1] for r in results))
        all_combinations = list(chain.from_iterable(r[2] for r in results))
        total_empty_arrays = sum(r[3] for r in results)
        total_errors = sum(r[4] for r in results)

        if total_errors > 1000:
            logger.error(f"Too many parsing errors (>1000) in field {field_name}")
            return None

        # Stats
        unique_values_count = len(set(all_parsed_values)) if all_parsed_values else 0
        avg_values_per_record = (
            sum(all_value_counts) / len(all_value_counts) if all_value_counts else 0
        )

        value_counter = Counter(all_parsed_values)
        values_analysis = {str(k): int(v) for k, v in value_counter.most_common(top_n)}

        combination_counter = Counter(all_combinations)
        combinations_analysis = {
            ", ".join(combo) if combo else "Empty": count
            for combo, count in combination_counter.most_common(top_n)
        }

        value_counts_counter = Counter(all_value_counts)
        value_counts_distribution = {
            str(k): int(v) for k, v in sorted(value_counts_counter.items())
        }

        stats = {
            "field_name": field_name,
            "total_records": total_records,
            "null_count": int(null_count),
            "null_percentage": null_percentage,
            "non_null_count": int(non_null_count),
            "empty_arrays_count": int(total_empty_arrays),
            "empty_arrays_percentage": (
                round((total_empty_arrays / non_null_count) * 100, 2)
                if non_null_count > 0
                else 0
            ),
            "unique_values": unique_values_count,
            "unique_combinations": len(combination_counter),
            "avg_values_per_record": round(avg_values_per_record, 2),
            "max_values_per_record": max(all_value_counts) if all_value_counts else 0,
            "values_analysis": values_analysis,
            "combinations_analysis": combinations_analysis,
            "value_counts_distribution": value_counts_distribution,
        }

        if total_errors > 0:
            stats["error_count"] = total_errors
            stats["error_percentage"] = round((total_errors / total_records) * 100, 2)

        # Compute elapsed time
        elapsed_time = time.time() - start_time

        logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")

        # Combine results
        if stats is not None:
            # Compute final result
            if progress_tracker:
                progress_tracker.update(
                    3,
                    {
                        "step": "Parallel MVF finalization",
                        "n_jobs": n_jobs,
                        "total_records": total_records,
                    },
                )
        stats["note"] = "Analysis performed using Joblib for parallel processing."
        return stats

    except Exception as e:
        logger.error(
            f"Error analyzing MVF field {field_name} with Parallel: {str(e)}",
            exc_info=True,
        )
        return None


def analyze_mvf_field_with_dask(
    df: pd.DataFrame,
    field_name: str,
    top_n: int = 20,
    parse_args: Optional[Dict[str, Any]] = None,
    chunk_size: int = 10000,
    npartitions: Optional[int] = None,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Analyze a multi-valued field (MVF) using Dask for scalable parallel processing.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data to analyze.
    field_name : str
        The name of the multi-valued field to analyze.
    top_n : int
        The number of top items to include in the analysis (default: 20).
    parse_args : Optional[Dict[str, Any]] = None
        Additional parsing arguments (default: empty dict).
    chunk_size : int
        The size of chunks to process (default: 10000).
    npartitions : Optional[int]
        The number of Dask partitions to use (default: None).
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the analysis progress.
    task_logger : Optional[logging.Logger]
        Logger to track errors and progress.

    Returns:
    --------
    Dict[str, Any]
        A dictionary with statistical summaries of the multi-valued field, or an error message.
    """
    if task_logger:
        logger = task_logger

    logger.info(f"Analyzing MVF field (Dask): {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return None

    try:
        try:
            import dask.dataframe as dd
        except ImportError:
            raise ImportError(
                "Dask is required for distributed processing but not installed. "
                "Install with: pip install dask[dataframe]"
            )

        # Initialize start time for performance tracking
        start_time = time.time()

        # Estimate Dask resources
        total_records = len(df)
        if npartitions is None or npartitions < 1:
            nparts = (total_records + chunk_size - 1) // chunk_size
        else:
            nparts = npartitions

        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.total = nparts
            progress_tracker.update(
                1, {"step": "Dask processing setup", "total_parts": nparts}
            )

        # Create Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=nparts)
        null_count = ddf[field_name].isna().sum().compute()

        logger.info(f"Processing {total_records} rows in {nparts} partitions with Dask")

        # Update progress for Dask processing
        if progress_tracker:
            progress_tracker.update(
                2,
                {
                    "step": "Dask MVF processing",
                    "total_parts": nparts,
                },
            )

        # Map partitions for processing
        parsed_df = ddf.map_partitions(
            process_mvf_partition,
            field_name=field_name,
            parse_args=parse_args,
        ).compute()

        # Aggregate results
        stats = aggregate_mvf_analysis(
            parsed_df, total_records, null_count, field_name, top_n=top_n
        )

        # Compute elapsed time
        elapsed_time = time.time() - start_time

        logger.info(f"Dask processing completed in {elapsed_time:.2f} seconds")

        # Combine results
        if stats is not None:
            # Compute final result
            if progress_tracker:
                progress_tracker.update(
                    3,
                    {
                        "step": "Dask MVF finalization",
                        "total_parts": nparts,
                    },
                )

        stats["note"] = "Analysis performed using Dask for large dataset."
        return stats

    except Exception as e:
        logger.error(
            f"Error analyzing MVF field {field_name} with Dask: {str(e)}",
            exc_info=True,
        )
        return None


def process_mvf_partition(
    partition: pd.DataFrame,
    field_name: str,
    parse_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Parse and extract information from a multi-valued field in a DataFrame partition.

    Parameters:
    -----------
    partition : pd.DataFrame
        A partition of the full DataFrame to process (used with Dask).
    field_name : str
        The name of the multi-valued field (MVF) to parse.
    format_type : Optional[str]
        The format type of the MVF values (e.g., 'json', 'array_string', 'csv').
    parse_args : Optional[Dict[str, Any]]
        Additional keyword arguments to pass to the parse_mvf function.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing parsed values, value counts, combinations, and error flags.
    """
    results = []
    for value in partition[field_name].dropna():
        try:
            values = parse_mvf(value, **parse_args)
            combination = tuple(sorted(values))
            results.append(
                {
                    "values": values,
                    "value_count": len(values),
                    "combination": combination,
                    "is_empty": len(values) == 0,
                    "error": False,
                }
            )
        except Exception:
            results.append(
                {
                    "values": [],
                    "value_count": 0,
                    "combination": tuple(),
                    "is_empty": False,
                    "error": True,
                }
            )
    return pd.DataFrame(results)


def aggregate_mvf_analysis(
    parsed_df: pd.DataFrame,
    total_records: int,
    null_count: int,
    field_name: str,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Aggregate and analyze parsed MVF results to compute statistics.

    Parameters:
    -----------
    parsed_df : pd.DataFrame
        The DataFrame resulting from process_mvf_partition, containing parsed details.
    total_records : int
        Total number of records in the original dataset.
    null_count : int
        Number of null values in the field.
    field_name : str
        The name of the field being analyzed.
    top_n : int, optional
        Number of top values/combinations to include in the result (default is 20).

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing statistics on the field, such as null rate, unique values,
        top values/combinations, and value count distribution.
    """
    parsed_values = [v for sublist in parsed_df["values"] for v in sublist]
    value_counts = parsed_df["value_count"].tolist()
    combinations = parsed_df["combination"].tolist()
    empty_arrays_count = parsed_df["is_empty"].sum()
    error_count = parsed_df["error"].sum()
    non_null_count = total_records - null_count

    unique_values_count = len(set(parsed_values)) if parsed_values else 0
    avg_values_per_record = sum(value_counts) / len(value_counts) if value_counts else 0

    value_counter = Counter(parsed_values)
    values_analysis = {str(k): int(v) for k, v in value_counter.most_common(top_n)}

    combination_counter = Counter(combinations)
    combinations_analysis = {
        ", ".join(combo) if combo else "Empty": count
        for combo, count in combination_counter.most_common(top_n)
    }

    value_counts_counter = Counter(value_counts)
    value_counts_distribution = {
        str(k): int(v) for k, v in sorted(value_counts_counter.items())
    }

    stats = {
        "field_name": field_name,
        "total_records": total_records,
        "null_count": int(null_count),
        "null_percentage": (
            round((null_count / total_records) * 100, 2) if total_records > 0 else 0
        ),
        "non_null_count": int(non_null_count),
        "empty_arrays_count": int(empty_arrays_count),
        "empty_arrays_percentage": (
            round((empty_arrays_count / non_null_count) * 100, 2)
            if non_null_count > 0
            else 0
        ),
        "unique_values": unique_values_count,
        "unique_combinations": len(combination_counter),
        "avg_values_per_record": round(avg_values_per_record, 2),
        "max_values_per_record": max(value_counts) if value_counts else 0,
        "values_analysis": values_analysis,
        "combinations_analysis": combinations_analysis,
        "value_counts_distribution": value_counts_distribution,
    }

    if error_count > 0:
        stats["error_count"] = int(error_count)
        stats["error_percentage"] = round((error_count / total_records) * 100, 2)

    return stats


def create_value_dictionary(
    df: pd.DataFrame,
    field_name: str,
    min_frequency: int = 1,
    parse_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Create a dictionary of values with frequencies for an MVF field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    min_frequency : int
        Minimum frequency for inclusion in the dictionary
    parse_args : Optional[Dict[str, Any]]
        Arguments to pass to parse_mvf

    Returns:
    --------
    pd.DataFrame
        DataFrame with values and frequencies
    """
    logger.info(f"Creating value dictionary for MVF field: {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return pd.DataFrame(columns=["value", "frequency", "percentage"])

    custom_parse_args = parse_args or {}

    try:
        # Parse MVF values
        all_values = []
        for value in df[field_name].dropna():
            try:
                values = parse_mvf(value, **custom_parse_args)
                all_values.extend(values)
            except Exception as e:
                logger.warning(f"Error parsing MVF value '{value}': {str(e)}")

        # Count frequencies
        value_counter = Counter(all_values)

        # Filter by minimum frequency
        filtered_counter = {
            k: v for k, v in value_counter.items() if v >= min_frequency
        }

        if not filtered_counter:
            return pd.DataFrame(columns=["value", "frequency", "percentage"])

        # Create DataFrame
        values_df = pd.DataFrame(
            {
                "value": list(filtered_counter.keys()),
                "frequency": list(filtered_counter.values()),
            }
        )

        # Calculate percentages
        total = values_df["frequency"].sum()
        values_df["percentage"] = (
            values_df["frequency"] / total * 100 if total > 0 else 0
        ).round(2)

        # Sort by frequency in descending order
        return values_df.sort_values("frequency", ascending=False).reset_index(
            drop=True
        )

    except Exception as e:
        logger.error(
            f"Error creating value dictionary for {field_name}: {str(e)}", exc_info=True
        )
        return pd.DataFrame(columns=["value", "frequency", "percentage"])


def create_combinations_dictionary(
    df: pd.DataFrame,
    field_name: str,
    min_frequency: int = 1,
    parse_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Create a dictionary of value combinations with frequencies for an MVF field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    min_frequency : int
        Minimum frequency for inclusion in the dictionary
    parse_args : Optional[Dict[str, Any]]
        Arguments to pass to parse_mvf

    Returns:
    --------
    pd.DataFrame
        DataFrame with combinations and frequencies
    """
    logger.info(f"Creating combinations dictionary for MVF field: {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return pd.DataFrame(columns=["combination", "frequency", "percentage"])

    custom_parse_args = parse_args or {}

    try:
        # Parse MVF values and create combinations
        combinations = []
        for value in df[field_name].dropna():
            try:
                values = parse_mvf(value, **custom_parse_args)
                combinations.append(tuple(sorted(values)))
            except Exception as e:
                logger.warning(f"Error parsing MVF value '{value}': {str(e)}")

        # Count frequencies
        combination_counter = Counter(combinations)

        # Filter by minimum frequency
        filtered_counter = {
            k: v for k, v in combination_counter.items() if v >= min_frequency
        }

        if not filtered_counter:
            return pd.DataFrame(columns=["combination", "frequency", "percentage"])

        # Create DataFrame
        combinations_df = pd.DataFrame(
            {
                "combination": [
                    ", ".join(combo) if combo else "Empty"
                    for combo in filtered_counter.keys()
                ],
                "frequency": list(filtered_counter.values()),
            }
        )

        # Calculate percentages
        total = combinations_df["frequency"].sum()
        combinations_df["percentage"] = (
            combinations_df["frequency"] / total * 100 if total > 0 else 0
        )

        # Sort by frequency in descending order
        return combinations_df.sort_values("frequency", ascending=False)

    except Exception as e:
        logger.error(
            f"Error creating combinations dictionary for {field_name}: {str(e)}",
            exc_info=True,
        )
        return pd.DataFrame(columns=["combination", "frequency", "percentage"])


def analyze_value_count_distribution(
    df: pd.DataFrame, field_name: str, parse_args: Dict[str, Any] = None
) -> Dict[str, int]:
    """
    Analyze the distribution of value counts per record in an MVF field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    parse_args : Dict[str, Any], optional
        Arguments to pass to parse_mvf

    Returns:
    --------
    Dict[str, int]
        Distribution of value counts
    """
    logger.info(f"Analyzing value count distribution for MVF field: {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return {}

    custom_parse_args = parse_args or {}

    try:
        # Count values per record
        value_counts = []
        for value in df[field_name].dropna():
            try:
                values = parse_mvf(value, **custom_parse_args)
                value_counts.append(len(values))
            except Exception as e:
                logger.warning(f"Error parsing MVF value '{value}': {str(e)}")
                value_counts.append(0)  # Count as 0 for error cases

        # Count frequencies
        counts_counter = Counter(value_counts)

        # Sort by count number
        return {str(k): int(v) for k, v in sorted(counts_counter.items())}

    except Exception as e:
        logger.error(
            f"Error analyzing value count distribution for {field_name}: {str(e)}",
            exc_info=True,
        )
        return {}


def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """
    Estimate resources needed for analyzing an MVF field.

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
    if field_name not in df.columns:
        return {"error": f"Field {field_name} not found in DataFrame"}

    # Estimate basic metrics
    total_records = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_records - null_count

    # Sample a few values to estimate complexity
    sample_size = min(100, non_null_count)
    sample_values = (
        df[field_name].dropna().sample(sample_size) if sample_size > 0 else []
    )

    # Estimate average values per record
    total_values = 0
    complex_values = 0

    for value in sample_values:
        try:
            values = parse_mvf(value)
            total_values += len(values)
            if len(values) > 5:
                complex_values += 1
        except Exception:
            complex_values += 1

    avg_values_per_record = (
        total_values / len(sample_values) if sample_values.size > 0 else 0
    )
    complex_percentage = (
        (complex_values / len(sample_values) * 100) if sample_values.size > 0 else 0
    )

    # Detect format
    detected_format = detect_mvf_format(sample_values)

    # Calculate approximate memory requirement
    memory_per_value = 200  # bytes per unique value with overhead
    estimated_memory = (non_null_count * avg_values_per_record * memory_per_value) / (
        1024 * 1024
    )  # in MB

    # Estimate processing time based on row count and complexity
    base_time = 0.1  # seconds
    per_row_time = 0.0001  # seconds per row
    per_value_time = 0.00005  # seconds per value

    estimated_time = (
        base_time
        + (non_null_count * per_row_time)
        + (non_null_count * avg_values_per_record * per_value_time)
    )

    # Scale up for complex values
    if complex_percentage > 20:
        estimated_time *= 1.5
        estimated_memory *= 1.2

    return {
        "field_name": field_name,
        "total_records": total_records,
        "non_null_count": int(non_null_count),
        "detected_format": detected_format,
        "estimated_avg_values_per_record": round(avg_values_per_record, 2),
        "complex_values_percentage": round(complex_percentage, 2),
        "estimated_memory_mb": round(estimated_memory, 2),
        "estimated_time_seconds": round(estimated_time, 2),
        "large_dataset": total_records > 1000000,
        "dask_recommended": total_records > 1000000 and estimated_memory > 500,
    }


def generate_analysis_distribution_vis(
    analysis_results: Dict[str, Any],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    top_n: int = 15,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate bar chart for distribution analysis using create_bar_plot.

    Parameters
    ----------
    analysis_results : Dict[str, Any]
        Dictionary containing the analysis results data.
    field_label : str
        Label used in naming the output visualization file.
    operation_name : str
        Name of the operation for tracking or labeling purposes.
    task_dir : Path
        Directory path where the output pie chart will be saved.
    timestamp : str
        Timestamp string to help uniquely name the output file.
    top_n : int, optional
        Number of top items to include in the visualization (default is 15).
    theme : Optional[str], optional
        Visualization theme to use (if any).
    backend : Optional[str], optional
        Visualization backend to use (if any).
    strict : Optional[bool], optional
        Whether to enforce strict visualization rules.
    visualization_paths : Optional[Dict[str, Any]], optional
        Dictionary to store and return paths to visualization outputs. If None, a new one is created.
    **kwargs : Any
        Additional keyword arguments for the pie chart creation function.

    Returns
    -------
    Dict[str, Any]
        Dictionary with visualization paths including:
        - "consistency_analysis_bar_chart": Path to the generated bar chart file.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating field distribution bar chart for operation '%s' (field_label='%s')",
        operation_name,
        field_label,
    )

    if "values_analysis" in analysis_results and analysis_results["values_analysis"]:
        # Check if values_analysis exists
        values_analysis = analysis_results.get("values_analysis", {})
        if not values_analysis:
            logger.warning(
                "No distribution data found for field '%s'. Skipping visualization.",
                field_label,
            )
        else:
            # Prepare output path
            values_distribution_path = (
                task_dir
                / f"{field_label}_{operation_name}_values_distribution_{timestamp}.png"
            )

            # Normalize and sort the value counts
            sorted_counts_data = normalize_and_sort_value_counts(values_analysis)

            # Create bar chart using helper
            values_distribution_stats_result = plot_value_distribution(
                data=sorted_counts_data,
                output_path=str(values_distribution_path),
                title=f"Distribution of '{field_label}' Values",
                max_items=top_n,
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs,
            )

            logger.debug(
                "Values distribution bar chart saved to: %s",
                values_distribution_stats_result,
            )
            visualization_paths["values_distribution_stats_bar_chart"] = (
                values_distribution_stats_result
            )

    # Combinations distribution visualization
    if (
        "combinations_analysis" in analysis_results
        and analysis_results["combinations_analysis"]
    ):
        # Check if combinations_analysis exists
        combinations_analysis = analysis_results.get("combinations_analysis", {})
        if not combinations_analysis:
            logger.warning(
                "No combinations data found for field '%s'. Skipping visualization.",
                field_label,
            )
        else:
            # Prepare output path
            combinations_distribution_path = (
                task_dir
                / f"{field_label}_{operation_name}_combinations_distribution_{timestamp}.png"
            )

            # Normalize and sort the value counts
            sorted_counts_data = normalize_and_sort_value_counts(combinations_analysis)

            combinations_distribution_stats_result = plot_value_distribution(
                data=sorted_counts_data,
                output_path=str(combinations_distribution_path),
                title=f"Distribution of '{field_label}' Combinations",
                max_items=top_n,
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs,
            )

            logger.debug(
                "Combinations distribution bar chart saved to: %s",
                combinations_distribution_stats_result,
            )
            visualization_paths["combinations_distribution_stats_bar_chart"] = (
                combinations_distribution_stats_result
            )
    # Value counts distribution visualization
    if (
        "value_counts_distribution" in analysis_results
        and analysis_results["value_counts_distribution"]
    ):
        # Check if value_counts_distribution exists
        value_counts_distribution = analysis_results.get(
            "value_counts_distribution", {}
        )
        if not value_counts_distribution:
            logger.warning(
                "No value counts data found for field '%s'. Skipping visualization.",
                field_label,
            )
        else:
            # Prepare output path
            value_counts_distribution_path = (
                task_dir
                / f"{field_label}_{operation_name}_value_counts_distribution_{timestamp}.png"
            )

            # Normalize and sort the value counts
            sorted_counts_data = normalize_and_sort_value_counts(
                value_counts_distribution
            )

            value_counts_distribution_stats_result = plot_value_distribution(
                data=sorted_counts_data,
                output_path=str(value_counts_distribution_path),
                title=f"Distribution of '{field_label}' Value Counts",
                max_items=top_n,
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs,
            )

            logger.debug(
                "Value counts distribution bar chart saved to: %s",
                value_counts_distribution_stats_result,
            )
            visualization_paths["value_counts_distribution_stats_bar_chart"] = (
                value_counts_distribution_stats_result
            )
    return visualization_paths


def normalize_and_sort_value_counts(value_counts: dict) -> dict:
    """
    Normalize the keys of a value_counts dictionary and sort them numerically if possible.

    Args:
        value_counts (dict): A dictionary where keys are values (may be str or int) and values are their counts.

    Returns:
        dict: A new dictionary with normalized keys, sorted by numeric order when applicable.
    """
    # Normalize keys: convert numeric strings like '01' to '1', keep others as str
    counts_data = {
        (str(int(k)) if str(k).isdigit() else str(k)): v
        for k, v in value_counts.items()
    }

    # Sort by numeric key if possible, else float('inf') to send to end
    sorted_counts = dict(
        sorted(
            counts_data.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else float("inf"),
        )
    )
    return sorted_counts
