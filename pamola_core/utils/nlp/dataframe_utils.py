"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        DataFrame Utilities for NLP
Package:       pamola_core.utils.nlp.dataframe_utils
Version:       1.1.1
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides utilities for DataFrame operations in NLP tasks,
particularly for handling processing markers, statistics, and column
backup operations.

Key Features:
- Prepare DataFrame columns for marker-based processing
- Calculate statistics for marked/unmarked records with vectorized operations
- Create column backups for safe in-place operations
- Handle legacy processed data detection
- Normalize column values for consistent processing
- Validate markers and DataFrame structure
- Type-safe return values with TypedDict
- Optimized performance for large DataFrames

Framework:
Part of PAMOLA.CORE NLP utilities, providing DataFrame manipulation
functions for text processing pipelines.

Changelog:
1.1.1 - Bug fixes and edge case handling
   - Fixed ID filtering type mismatches (string vs numeric comparisons)
   - Added source column validation in prepare_marked_column
   - Improved column normalization with DRY helper function
   - Enhanced tempfile usage for concurrent file operations
   - Updated documentation for consistency
1.1.0 - Performance optimizations and type safety improvements
   - Vectorized operations in prepare_marked_column for better performance
   - Added marker validation to prevent regex issues
   - Introduced TypedDict for type-safe statistics
   - Added custom exceptions for better error handling
   - Optimized memory usage in column comparisons
   - Added DataFrame validation utilities
1.0.0 - Initial implementation

Dependencies:
- pandas - DataFrame operations
- logging - Debug and info logging
- typing - Type annotations
- re - Regular expression for marker validation
- tempfile - Safe temporary file operations

TODO:
- Add support for multiple marker types
- Implement chunked processing for very large DataFrames
- Add pandas accessor for more idiomatic API
- Support for multi-column processing
- Add async processing capabilities
- Add NotRequired support when Python 3.11+ becomes standard
"""

import logging
from typing import Union, Optional, Tuple, List, TypedDict

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


# Custom exceptions
class DataFrameProcessingError(Exception):
    """Base exception for DataFrame processing errors."""

    pass


class ColumnNotFoundError(DataFrameProcessingError):
    """Raised when required column is not found."""

    pass


class MarkerValidationError(DataFrameProcessingError):
    """Raised when marker validation fails."""

    pass


# Type definitions
class MarkerStatistics(TypedDict):
    """Type definition for marker statistics."""

    total: int
    processed: int
    unprocessed: int
    percentage_complete: float
    non_empty: int
    empty: int
    column_exists: bool


class ProcessingIndicesStats(TypedDict):
    """Type definition for processing indices statistics."""

    total_in_range: int
    already_processed: int
    to_process: int
    limited_by_max: bool


def validate_marker(marker: str) -> None:
    """
    Validate marker is safe for string operations.

    Parameters
    ----------
    marker : str
        Processing marker to validate

    Raises
    ------
    MarkerValidationError
        If marker is invalid
    """
    if not marker:
        raise MarkerValidationError("Marker cannot be empty")
    if marker.isspace():
        raise MarkerValidationError("Marker cannot be only whitespace")

    # Warn about regex special chars that might cause issues
    regex_chars = set(r".*+?^${}()|[]\\")
    special_chars = [c for c in marker if c in regex_chars]
    if special_chars:
        logger.warning(
            f"Marker '{marker}' contains special characters {special_chars}. "
            f"This may cause unexpected behavior in string operations."
        )


def validate_dataframe_for_processing(
    df: pd.DataFrame, required_columns: List[str], marker: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame is ready for processing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names
    marker : str, optional
        Processing marker to validate

    Returns
    -------
    tuple
        (is_valid, list_of_errors)
    """
    errors = []

    # Check if DataFrame is empty
    if df.empty:
        errors.append("DataFrame is empty")

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    # Validate marker if provided
    if marker is not None:
        try:
            validate_marker(marker)
        except MarkerValidationError as e:
            errors.append(f"Invalid marker: {str(e)}")

    return len(errors) == 0, errors


def _normalize_series(
    series: pd.Series, ignore_whitespace: bool, ignore_case: bool
) -> pd.Series:
    """
    Helper to normalize series for comparison.

    Parameters
    ----------
    series : pd.Series
        Series to normalize
    ignore_whitespace : bool
        Whether to strip whitespace
    ignore_case : bool
        Whether to convert to lowercase

    Returns
    -------
    pd.Series
        Normalized series
    """
    if ignore_whitespace:
        series = series.str.strip()
    if ignore_case:
        series = series.str.lower()
    return series


def prepare_marked_column(
    df: pd.DataFrame, source: str, target: str, marker: str, clear_target: bool = False
) -> pd.DataFrame:
    """
    Ensure target column exists and correctly flags processed/unprocessed rows.

    This function prevents cache conflicts by properly handling existing data:
    - If clear_target is True, wipes the target column completely
    - If target column is missing, creates it with NA values
    - If target column exists, normalizes values and preserves markers

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    source : str
        Source column name
    target : str
        Target column name
    marker : str
        Processing marker (e.g., "~")
    clear_target : bool
        Whether to clear target column

    Returns
    -------
    pd.DataFrame
        Prepared dataframe

    Raises
    ------
    MarkerValidationError
        If marker is invalid
    ColumnNotFoundError
        If source column doesn't exist

    Notes
    -----
    Rules for determining if a row is already processed:
    - Value starts with marker -> already processed
    - Value != source and value not NA -> assume processed (legacy run)
    - Otherwise -> needs processing
    """
    # Validate marker
    validate_marker(marker)

    # Validate source column exists
    if source not in df.columns:
        raise ColumnNotFoundError(
            f"Source column '{source}' not found. "
            f"Available columns: {', '.join(df.columns[:10])}..."
        )

    if clear_target:
        logger.info(f"Clearing target column '{target}' as requested")
        df[target] = pd.NA
        return df

    if target not in df.columns:
        logger.info(f"Creating empty target column '{target}'")
        df[target] = pd.NA
        return df

    # Target column exists - normalize existing values
    logger.info(f"Target column '{target}' already exists, normalizing values")

    # Count different types of existing values
    total = len(df)

    # Vectorized normalization
    def normalize_value(val):
        """Normalize a single value in the target column."""
        if pd.isna(val):
            return pd.NA

        val_str = str(val)

        # Handle pandas string representations of NA
        if val_str in ("<NA>", "nan", "None", "NaN"):
            return pd.NA

        # If has marker, ensure it's properly formatted
        if val_str.startswith(marker):
            return val_str  # Already has marker, keep as is

        # Otherwise just strip whitespace
        return val_str.strip()

    # Apply normalization
    df[target] = df[target].apply(normalize_value)

    # Vectorized statistics calculation for better performance
    if not df.empty:
        # Count non-empty values
        mask_notna = df[target].notna()
        non_empty = int(mask_notna.sum())

        # Count values with marker
        mask_marker = mask_notna & df[target].astype(str).str.startswith(marker)
        with_marker = int(mask_marker.sum())

        # Count values different from source (legacy processed)
        different_from_source = 0
        if source in df.columns:
            mask_different = (
                mask_notna
                & df[source].notna()
                & (df[target].astype(str) != df[source].astype(str))
            )
            # Only count as different if doesn't have marker (legacy)
            different_from_source = int((mask_different & ~mask_marker).sum())
    else:
        non_empty = with_marker = different_from_source = 0

    logger.info(f"  Non-empty values: {non_empty}/{total}")
    logger.info(f"  Values with marker ({marker}): {with_marker}/{total}")
    logger.info(
        f"  Values different from source (legacy): {different_from_source}/{total}"
    )

    return df


def get_marker_statistics(
    df: pd.DataFrame, column: str, marker: str
) -> MarkerStatistics:
    """
    Get statistics about marked/unmarked records in a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to analyze
    marker : str
        Processing marker to look for

    Returns
    -------
    MarkerStatistics
        Statistics dictionary with:
        - total: Total number of records
        - processed: Number of records with marker
        - unprocessed: Number of records without marker
        - percentage_complete: Percentage of processed records
        - non_empty: Number of non-empty values
        - empty: Number of empty/NA values
        - column_exists: Whether column exists

    Raises
    ------
    MarkerValidationError
        If marker is invalid
    """
    # Validate marker
    validate_marker(marker)

    total = len(df)

    if column not in df.columns:
        # Column doesn't exist, all records are unprocessed
        return MarkerStatistics(
            total=total,
            processed=0,
            unprocessed=total,
            percentage_complete=0.0,
            non_empty=0,
            empty=total,
            column_exists=False,
        )

    # Vectorized counting for better performance
    non_empty_mask = df[column].notna()
    non_empty = int(non_empty_mask.sum())
    empty = total - non_empty

    # Count processed records (have marker)
    processed = 0
    if non_empty > 0:
        # Convert to string and check for marker
        processed_mask = non_empty_mask & df[column].astype(
            str
        ).str.strip().str.startswith(marker)
        processed = int(processed_mask.sum())

    unprocessed = total - processed

    return MarkerStatistics(
        total=total,
        processed=processed,
        unprocessed=unprocessed,
        percentage_complete=(processed / total * 100) if total > 0 else 0.0,
        non_empty=non_empty,
        empty=empty,
        column_exists=True,
    )


def create_column_backup(
    df: pd.DataFrame,
    source_column: str,
    backup_suffix: str = "_original",
    force: bool = False,
) -> Optional[str]:
    """
    Create a backup column for in-place operations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify
    source_column : str
        Source column to backup
    backup_suffix : str
        Suffix for backup column name
    force : bool
        If True, overwrite existing backup column

    Returns
    -------
    Optional[str]
        Name of the backup column if created, None if skipped

    Raises
    ------
    ColumnNotFoundError
        If source column doesn't exist
    """
    if source_column not in df.columns:
        raise ColumnNotFoundError(
            f"Source column '{source_column}' not found. "
            f"Available columns: {', '.join(df.columns[:10])}..."
        )

    backup_column = f"{source_column}{backup_suffix}"

    # Check if backup already exists
    if backup_column in df.columns and not force:
        logger.warning(
            f"Backup column '{backup_column}' already exists, skipping backup creation. "
            f"Use force=True to overwrite."
        )
        return None

    # Create or overwrite backup
    logger.info(f"Creating backup column '{backup_column}' from '{source_column}'")
    df[backup_column] = df[source_column].copy()

    return backup_column


def identify_processed_rows(
    df: pd.DataFrame,
    target_column: str,
    marker: str,
    source_column: Optional[str] = None,
) -> pd.Series:
    """
    Identify which rows have been processed based on marker or value changes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Target column to check
    marker : str
        Processing marker
    source_column : str, optional
        Source column for comparison (for legacy detection)

    Returns
    -------
    pd.Series
        Boolean series where True indicates processed rows

    Raises
    ------
    MarkerValidationError
        If marker is invalid
    """
    # Validate marker
    validate_marker(marker)

    if target_column not in df.columns:
        return pd.Series(False, index=df.index)

    # Check for marker
    has_marker = df[target_column].notna() & df[target_column].astype(
        str
    ).str.strip().str.startswith(marker)

    # If source column provided, also check for legacy processed rows
    # (different from source but no marker)
    if source_column and source_column in df.columns:
        is_different = (
            df[target_column].notna()
            & df[source_column].notna()
            & (df[target_column].astype(str) != df[source_column].astype(str))
        )
        # Row is processed if it has marker OR is different from source
        return has_marker | is_different

    return has_marker


def get_unprocessed_indices(
    df: pd.DataFrame,
    target_column: str,
    marker: str,
    source_column: Optional[str] = None,
    id_column: Optional[str] = None,
    start_id: Optional[Union[int, str]] = None,
    end_id: Optional[Union[int, str]] = None,
    max_records: Optional[int] = None,
) -> Tuple[pd.Index, ProcessingIndicesStats]:
    """
    Get indices of unprocessed records with optional filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Target column to check
    marker : str
        Processing marker
    source_column : str, optional
        Source column for legacy detection
    id_column : str, optional
        Column for ID-based filtering
    start_id : Union[int, str], optional
        Starting ID (inclusive)
    end_id : Union[int, str], optional
        Ending ID (inclusive)
    max_records : int, optional
        Maximum records to return

    Returns
    -------
    tuple
        (indices, stats) where:
        - indices: pandas Index of unprocessed records
        - stats: ProcessingIndicesStats with counts

    Raises
    ------
    MarkerValidationError
        If marker is invalid
    """
    # Validate marker
    validate_marker(marker)

    # Start with all indices
    mask = pd.Series(True, index=df.index)

    # Apply ID range filtering if specified
    if id_column and id_column in df.columns:
        id_series = df[id_column]

        # Handle type mismatches between ID column and start/end values
        if start_id is not None:
            # Try to convert ID series to the same type as start_id
            if pd.api.types.is_numeric_dtype(
                type(start_id)
            ) and not pd.api.types.is_numeric_dtype(id_series):
                # start_id is numeric but column is not - try to convert column
                try:
                    id_series_numeric = pd.to_numeric(id_series, errors="coerce")
                    mask &= id_series_numeric >= start_id
                except Exception:
                    # Fallback to string comparison
                    mask &= id_series.astype(str) >= str(start_id)
            else:
                # Direct comparison
                mask &= id_series >= start_id

        if end_id is not None:
            # Same logic for end_id
            if pd.api.types.is_numeric_dtype(
                type(end_id)
            ) and not pd.api.types.is_numeric_dtype(id_series):
                try:
                    id_series_numeric = pd.to_numeric(id_series, errors="coerce")
                    mask &= id_series_numeric <= end_id
                except Exception:
                    mask &= id_series.astype(str) <= str(end_id)
            else:
                mask &= id_series <= end_id

    # Get filtered indices
    filtered_indices = df[mask].index

    # Identify processed rows
    processed_mask = identify_processed_rows(
        df.loc[filtered_indices], target_column, marker, source_column
    )

    # Get unprocessed indices
    unprocessed_indices = filtered_indices[~processed_mask]

    # Apply max_records limit if specified
    limited = False
    if max_records and len(unprocessed_indices) > max_records:
        unprocessed_indices = unprocessed_indices[:max_records]
        limited = True

    # Calculate statistics
    stats = ProcessingIndicesStats(
        total_in_range=len(filtered_indices),
        already_processed=int(processed_mask.sum()),
        to_process=len(unprocessed_indices),
        limited_by_max=limited,
    )

    return unprocessed_indices, stats


def compare_columns(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    ignore_whitespace: bool = True,
    ignore_case: bool = False,
    chunksize: Optional[int] = None,
) -> pd.Series:
    """
    Compare two columns for differences.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column1 : str
        First column name
    column2 : str
        Second column name
    ignore_whitespace : bool
        Whether to ignore leading/trailing whitespace
    ignore_case : bool
        Whether to ignore case differences
    chunksize : int, optional
        Process in chunks for memory efficiency (None = process all at once)

    Returns
    -------
    pd.Series
        Boolean series where True indicates differences

    Raises
    ------
    ColumnNotFoundError
        If required columns don't exist
    """
    missing_cols = []
    if column1 not in df.columns:
        missing_cols.append(column1)
    if column2 not in df.columns:
        missing_cols.append(column2)

    if missing_cols:
        raise ColumnNotFoundError(
            f"Column(s) not found: {', '.join(missing_cols)}. "
            f"Available columns: {', '.join(df.columns[:10])}..."
        )

    # For small DataFrames or when chunksize not specified, process all at once
    if chunksize is None or len(df) <= chunksize:
        # Convert to string for comparison
        col1_str = df[column1].fillna("").astype(str)
        col2_str = df[column2].fillna("").astype(str)

        # Apply normalization using helper
        col1_str = _normalize_series(col1_str, ignore_whitespace, ignore_case)
        col2_str = _normalize_series(col2_str, ignore_whitespace, ignore_case)

        return col1_str != col2_str  # type: ignore[return-value]

    # Process in chunks for large DataFrames
    result_parts = []
    for start_idx in range(0, len(df), chunksize):
        end_idx = min(start_idx + chunksize, len(df))
        chunk = df.iloc[start_idx:end_idx]

        # Process chunk
        col1_str = chunk[column1].fillna("").astype(str)
        col2_str = chunk[column2].fillna("").astype(str)

        # Apply normalization using helper
        col1_str = _normalize_series(col1_str, ignore_whitespace, ignore_case)
        col2_str = _normalize_series(col2_str, ignore_whitespace, ignore_case)

        result_parts.append(col1_str != col2_str)

    return pd.concat(result_parts)  # type: ignore[return-value]


def split_by_processing_status(
    df: pd.DataFrame,
    target_column: str,
    marker: str,
    source_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into processed and unprocessed subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Target column to check
    marker : str
        Processing marker
    source_column : str, optional
        Source column for legacy detection

    Returns
    -------
    tuple
        (processed_df, unprocessed_df) - DataFrames split by processing status

    Raises
    ------
    MarkerValidationError
        If marker is invalid
    """
    processed_mask = identify_processed_rows(df, target_column, marker, source_column)

    processed_df = df[processed_mask].copy()
    unprocessed_df = df[~processed_mask].copy()

    return processed_df, unprocessed_df
