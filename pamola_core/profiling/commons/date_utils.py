"""
Date analysis utilities for the project.

This module provides utility functions for analyzing date fields, including
validation, distribution analysis, anomaly detection, and group analysis.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from pamola_core.utils.progress import HierarchicalProgressTracker

# Configure logger
logger = logging.getLogger(__name__)


def prepare_date_data(df: pd.DataFrame, field_name: str) -> Tuple[pd.Series, int, int]:
    """
    Prepare date data for analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze

    Returns:
    --------
    Tuple[pd.Series, int, int]
        Tuple containing the prepared date series, null count, and non-null count
    """
    # Check if field exists
    if field_name not in df.columns:
        raise ValueError(f"Field {field_name} not found in DataFrame")

    # Get basic counts
    total_records = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_records - null_count

    # Convert to datetime for analysis
    dates = pd.to_datetime(df[field_name], errors='coerce')

    return dates, null_count, non_null_count


def calculate_date_stats(dates: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic date statistics.

    Parameters:
    -----------
    dates : pd.Series
        Series of dates to analyze

    Returns:
    --------
    Dict[str, Any]
        Dictionary with date statistics
    """
    # Skip if no valid dates
    valid_mask = ~dates.isna()
    valid_count = valid_mask.sum()

    if valid_count == 0:
        return {
            'valid_count': 0,
            'invalid_count': 0,
            'min_date': None,
            'max_date': None
        }

    # Calculate date range
    min_date = dates[valid_mask].min()
    max_date = dates[valid_mask].max()

    return {
        'valid_count': int(valid_count),
        'min_date': min_date.strftime('%Y-%m-%d') if not pd.isna(min_date) else None,
        'max_date': max_date.strftime('%Y-%m-%d') if not pd.isna(max_date) else None
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
    # Skip if no valid dates
    valid_mask = ~dates.isna()
    valid_count = valid_mask.sum()

    if valid_count == 0:
        return {}

    result = {}

    # Year distribution
    year_distribution = dates[valid_mask].dt.year.value_counts().sort_index().to_dict()
    result['year_distribution'] = {str(year): int(count) for year, count in year_distribution.items()}

    # Decade distribution
    decades = (dates[valid_mask].dt.year // 10 * 10).value_counts().sort_index().to_dict()
    result['decade_distribution'] = {f"{decade}s": int(count) for decade, count in decades.items()}

    # Month distribution
    month_distribution = dates[valid_mask].dt.month.value_counts().sort_index().to_dict()
    result['month_distribution'] = {str(month): int(count) for month, count in month_distribution.items()}

    # Day of week distribution
    dow_distribution = dates[valid_mask].dt.dayofweek.value_counts().sort_index().to_dict()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    result['day_of_week_distribution'] = {day_names[dow]: int(count) for dow, count in dow_distribution.items()}

    return result


def validate_date_format(date_str: str, format_str: str = '%Y-%m-%d') -> bool:
    """
    Check if a date string matches the specified format.

    Parameters:
    -----------
    date_str : str
        The date string to validate
    format_str : str
        The expected date format

    Returns:
    --------
    bool
        True if the date matches the format, False otherwise
    """
    try:
        datetime.strptime(date_str, format_str)
        return True
    except (ValueError, TypeError):
        return False


def detect_date_anomalies(dates: pd.Series, min_year: int = 1940, max_year: int = 2005) -> Dict[str, List[Any]]:
    """
    Detect anomalies in dates (too old, future dates, invalid formats).

    Parameters:
    -----------
    dates : pd.Series
        Series of dates to analyze
    min_year : int
        Minimum valid year
    max_year : int
        Maximum valid year

    Returns:
    --------
    Dict[str, List[Any]]
        Dictionary with anomaly categories and examples
    """
    # Convert dates to datetime with errors flagged
    date_objects = pd.to_datetime(dates, errors='coerce')

    # Initialize results
    anomalies = {
        'invalid_format': [],  # Invalid format
        'too_old': [],  # Too old (before min_year)
        'future_dates': [],  # Future dates
        'too_young': [],  # Too young (after max_year)
        'negative_years': []  # Negative years
    }

    # Check each date
    for i, date_str in enumerate(dates):
        if pd.isna(date_str):
            continue

        # Check format
        if not validate_date_format(str(date_str)):
            anomalies['invalid_format'].append((i, date_str))
            continue

        # If format is valid but date doesn't convert, look for the reason
        if pd.isna(date_objects[i]):
            if str(date_str).startswith('-'):
                anomalies['negative_years'].append((i, date_str))
            else:
                anomalies['invalid_format'].append((i, date_str))
        else:
            # For convertible dates, check the range
            year = date_objects[i].year

            if year < min_year:
                anomalies['too_old'].append((i, date_str, year))
            elif year > datetime.now().year:
                anomalies['future_dates'].append((i, date_str, year))
            elif year > max_year:
                anomalies['too_young'].append((i, date_str, year))

    return anomalies


def detect_date_changes_within_group(df: pd.DataFrame, group_column: str, date_column: str) -> Dict[str, Any]:
    """
    Detect date changes within groups (e.g., resume_id).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    group_column : str
        The column to group by
    date_column : str
        The date column to analyze

    Returns:
    --------
    Dict[str, Any]
        Results of the analysis
    """
    # Check if columns exist
    if group_column not in df.columns:
        return {'error': f"Group column {group_column} not found in DataFrame"}

    if date_column not in df.columns:
        return {'error': f"Date column {date_column} not found in DataFrame"}

    # Group by the specified column
    grouped = df.groupby(group_column)

    # Look for groups with varying dates
    results = {
        'groups_with_changes': 0,
        'examples': []
    }

    for group_id, group_df in grouped:
        # Skip groups with missing dates
        dates = group_df[date_column].dropna()
        if len(dates) <= 1:
            continue

        # If there's more than one unique value
        if dates.nunique() > 1:
            results['groups_with_changes'] += 1

            # Add example if we haven't reached the limit
            if len(results['examples']) < 10:
                results['examples'].append({
                    'group_id': group_id,
                    'date_values': dates.unique().tolist()
                })

    return results


def detect_date_inconsistencies_by_uid(df: pd.DataFrame, uid_column: str, date_column: str) -> Dict[str, Any]:
    """
    Detect date inconsistencies by UID (person identifier).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    uid_column : str
        The UID column
    date_column : str
        The date column to analyze

    Returns:
    --------
    Dict[str, Any]
        Results of the analysis
    """
    # Check if columns exist
    if uid_column not in df.columns:
        return {'error': f"UID column {uid_column} not found in DataFrame"}

    if date_column not in df.columns:
        return {'error': f"Date column {date_column} not found in DataFrame"}

    # Group by UID
    grouped = df.groupby(uid_column)

    # Look for UIDs with varying dates
    results = {
        'uids_with_inconsistencies': 0,
        'examples': []
    }

    for uid, group_df in grouped:
        # Skip groups with missing dates
        dates = group_df[date_column].dropna()
        if len(dates) <= 1:
            continue

        # If there's more than one unique value
        if dates.nunique() > 1:
            results['uids_with_inconsistencies'] += 1

            # Add example if we haven't reached the limit
            if len(results['examples']) < 10:
                results['examples'].append({
                    'uid': uid,
                    'date_values': dates.unique().tolist()
                })

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
    partition_dates = pd.to_datetime(partition, errors='coerce')
    valid_mask = ~partition_dates.isna()
    valid_dates = partition_dates[valid_mask]
    
    if len(valid_dates) == 0:
        return pd.Series({
            'min_date': pd.NaT,
            'max_date': pd.NaT,
            'valid_count': 0
        })
    
    return pd.Series({
        'min_date': valid_dates.min(),
        'max_date': valid_dates.max(),
        'valid_count': len(valid_dates)
    })

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
    partition_dates = pd.to_datetime(partition, errors='coerce')
    valid_mask = ~partition_dates.isna()
    valid_dates = partition_dates[valid_mask]
    
    if len(valid_dates) == 0:
        return pd.DataFrame({
            'type': [], 'key': [], 'count': []
        })
    
    # Return distributions for this partition
    year_counts = valid_dates.dt.year.value_counts()
    month_counts = valid_dates.dt.month.value_counts()
    dow_counts = valid_dates.dt.dayofweek.value_counts()
    decade_counts = (valid_dates.dt.year // 10 * 10).value_counts()
    
    # Combine all distributions into a single DataFrame
    result_data = []
    
    for year, count in year_counts.items():
        result_data.append({'type': 'year', 'key': str(year), 'count': count})
    
    for month, count in month_counts.items():
        result_data.append({'type': 'month', 'key': str(month), 'count': count})
    
    for dow, count in dow_counts.items():
        result_data.append({'type': 'dow', 'key': str(dow), 'count': count})
    
    for decade, count in decade_counts.items():
        result_data.append({'type': 'decade', 'key': f"{decade}s", 'count': count})
    
    return pd.DataFrame(result_data)

def aggregate_distributions_data(distributions_data: pd.DataFrame) -> Dict[str, Dict[str, int]]:
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
    aggregated = distributions_data.groupby(['type', 'key'])['count'].sum().reset_index()
    
    # Convert back to the expected format
    distributions = {}
    # Year distribution
    year_data = aggregated[aggregated['type'] == 'year']
    if len(year_data) > 0:
        # Sort by year in ascending order
        year_data_sorted = year_data.sort_values('key', key=lambda x: x.astype(int))
        distributions['year_distribution'] = dict(zip(year_data_sorted['key'], year_data_sorted['count']))
    
    # Month distribution
    month_data = aggregated[aggregated['type'] == 'month']
    if len(month_data) > 0:
        # Sort by month in ascending order (1-12)
        month_data_sorted = month_data.sort_values('key', key=lambda x: x.astype(int))
        distributions['month_distribution'] = dict(zip(month_data_sorted['key'], month_data_sorted['count']))
    
    # Day of week distribution
    dow_data = aggregated[aggregated['type'] == 'dow']
    if len(dow_data) > 0:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Sort by day of week index (0=Monday to 6=Sunday) to maintain weekday order
        dow_data_sorted = dow_data.sort_values('key')
        distributions['day_of_week_distribution'] = {
            day_names[int(k)]: int(v) for k, v in zip(dow_data_sorted['key'], dow_data_sorted['count'])
        }
    
    # Decade distribution
    decade_data = aggregated[aggregated['type'] == 'decade']
    if len(decade_data) > 0:
        distributions['decade_distribution'] = dict(zip(decade_data['key'], decade_data['count']))
    
    return distributions

def analyze_date_field(df: pd.DataFrame,
                       field_name: str,
                       min_year: int = 1940,
                       max_year: int = 2005,
                       id_column: Optional[str] = None,
                       uid_column: Optional[str] = None,
                       chunk_size: int = 10000,
                       use_dask: bool = False,
                       use_vectorization: bool = False,
                       parallel_processes: int = 1,
                       progress_tracker: Optional[HierarchicalProgressTracker] = None,
                       task_logger: Optional[logging.Logger] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a date field in a DataFrame, including statistics, distributions, anomaly detection, and optional group/UID analysis.

    Parameters:
    -----------
    df : pd.DataFrame
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
    if task_logger:
        logger = task_logger

    logger.info(f"Analyzing date field: {field_name}")

    # Basic validation
    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}
    

    total_records = len(df)
    is_large_df = total_records > chunk_size

    # Get prepared data
    dates, null_count, non_null_count = prepare_date_data(df, field_name)
    total_records = len(df)

    valid_mask = ~dates.isna()
    valid_count = valid_mask.sum()
    invalid_count = non_null_count - valid_count

    # Calculate fill and validity rates
    fill_rate = round((non_null_count / total_records) * 100, 2) if total_records > 0 else 0
    valid_rate = round((valid_count / non_null_count) * 100, 2) if non_null_count > 0 else 0

    # Initialize results with basic stats
    results = {
        'total_records': total_records,
        'null_count': int(null_count),
        'non_null_count': int(non_null_count),
        'valid_count': int(valid_count),
        'invalid_count': int(invalid_count),
        'fill_rate': fill_rate,
        'valid_rate': valid_rate    
        }
    if use_dask and is_large_df:
        try:
            npartitions = kwargs.get('npartitions', None)

            logger.info("Parallel Enabled")
            logger.info("Parallel Engine: Dask")
            logger.info(f"Parallel Workers: {npartitions}")

            import dask.dataframe as dd
            import dask

            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=npartitions)
            
            logger.info(f"Using Dask for large dataset with {total_records} rows")

            if valid_count > 0:
                # Define function to calculate date stats for each partition                
                # Calculate date stats using map_partitions
                partition_stats_list = ddf[field_name].map_partitions(
                    partition_date_stats,
                    meta=pd.Series({
                        'min_date': pd.Timestamp('2000-01-01'),
                        'max_date': pd.Timestamp('2000-01-01'), 
                        'valid_count': 0
                    })
                ).compute()

                # Convert list of Series to list for processing
                partition_stats_data = []
                for stats in partition_stats_list:
                    if isinstance(stats, pd.Series):
                        partition_stats_data.append(stats)
                    else:
                        # Handle case where stats might be a different type
                        partition_stats_data.append(pd.Series(stats))

                # Aggregate results from all partitions
                if len(partition_stats_data) > 0:
                    # Filter for partitions with valid data
                    valid_partitions = [stats for stats in partition_stats_data if stats.get('valid_count', 0) > 0]
                    
                    if len(valid_partitions) > 0:
                        # Get min and max dates from valid partitions
                        min_dates = [stats.get('min_date') for stats in valid_partitions if not pd.isna(stats.get('min_date'))]
                        max_dates = [stats.get('max_date') for stats in valid_partitions if not pd.isna(stats.get('max_date'))]
                        if min_dates and max_dates:
                            overall_min = min(min_dates)
                            overall_max = max(max_dates)
                            
                            date_stats = {
                                'valid_count': int(valid_count),
                                'min_date': overall_min.strftime('%Y-%m-%d') if not pd.isna(overall_min) else None,
                                'max_date': overall_max.strftime('%Y-%m-%d') if not pd.isna(overall_max) else None
                            }
                            results.update(date_stats)

                # Calculate distributions using map_partitions
                distributions_data = ddf[field_name].map_partitions(
                    partition_distributions,
                    meta=pd.DataFrame({
                        'type': pd.Series(dtype='object'),
                        'key': pd.Series(dtype='object'), 
                        'count': pd.Series(dtype='int64')
                    })
                ).compute()

                # Aggregate distributions
                if not distributions_data.empty:
                    aggregated = aggregate_distributions_data(distributions_data)
                    
                    results.update(aggregated)
                
        except ImportError:
            logger.warning("Dask requested but not available. Falling back to chunked processing.")
            if progress_tracker:
                progress_tracker.update(0, {
                    "step": "Dask fallback",
                    "warning": "Dask not available, using chunks"
                })
    elif use_vectorization and parallel_processes > 0:
        try:
            logger.info("Parallel Enabled")
            logger.info("Parallel Engine: Joblib")
            logger.info(f"Parallel Workers: {parallel_processes}")

            from joblib import Parallel, delayed

            logger.info(f"Using parallel processing with {parallel_processes} processes for {total_records} rows")
            
            if progress_tracker:
                progress_tracker.update(0, {
                    "step": "Parallel processing setup",
                    "processes": parallel_processes,
                    "chunk_size": chunk_size
                })
            
            # Split DataFrame into chunks
            chunks = []
            for i in range(0, total_records, chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                chunks.append(chunk)
            
            logger.info(f"Processing {len(chunks)} chunks in parallel")
            
            if progress_tracker:
                progress_tracker.update(0, {
                    "step": "Processing chunks",
                    "total_chunks": len(chunks)
                })
              # Process chunks in parallel
            processed_chunks = list(Parallel(n_jobs=parallel_processes)(
                delayed(process_date_chunk)(i, chunk, field_name, min_year, max_year) 
                for i, chunk in enumerate(chunks)
            ))
            
            if progress_tracker:
                progress_tracker.update(0, {
                    "step": "Aggregating results",
                    "chunks_processed": len(processed_chunks)
                })
            
            # Aggregate results from all chunks
            aggregated_results = aggregate_chunk_results(processed_chunks)
            
            if 'error' not in aggregated_results:
                results.update(aggregated_results)
                logger.info(f"Parallel processing completed successfully")
            else:
                logger.warning(f"Error in parallel processing: {aggregated_results['error']}")
            
            if progress_tracker:
                progress_tracker.update(0, {
                    "step": "Parallel processing completed"
                })
                
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            logger.info("Falling back to standard processing")
            
            if progress_tracker:
                progress_tracker.update(0, {
                    "step": "Parallel fallback",
                    "error": str(e)
                })
    elif is_large_df and not use_dask:
        # Process large DataFrame in chunks without Dask
        logger.info(f"Processing large dataset with {total_records} rows using chunked processing")
        
        if progress_tracker:
            progress_tracker.update(0, {
                "step": "Chunked processing setup",
                "total_rows": total_records,
                "chunk_size": chunk_size
            })
        
        total_chunks = (total_records + chunk_size - 1) // chunk_size
        logger.info(f"Processing {total_chunks} chunks sequentially")
        
        # Initialize aggregated results for chunked processing
        chunk_results = {
            'date_stats': [],
            'distributions': [],
            'anomalies': []
        }
        
        # Process chunks sequentially
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
                        "chunk_end": end_idx
                    }
                )
            
            logger.debug(f"Processing chunk {i + 1}/{total_chunks} (rows {start_idx}-{end_idx})")
            
            # Process this chunk
            chunk_dates = pd.to_datetime(chunk[field_name], errors='coerce')
            valid_chunk_mask = ~chunk_dates.isna()
            valid_chunk_dates = chunk_dates[valid_chunk_mask]
            
            if len(valid_chunk_dates) > 0:
                # Calculate date stats for this chunk
                chunk_min = valid_chunk_dates.min()
                chunk_max = valid_chunk_dates.max()
                chunk_results['date_stats'].append({
                    'min_date': chunk_min,
                    'max_date': chunk_max,
                    'valid_count': len(valid_chunk_dates)
                })
                
                # Calculate distributions for this chunk
                chunk_distributions = calculate_distributions(chunk_dates)
                chunk_results['distributions'].append(chunk_distributions)
                
        if progress_tracker:
            progress_tracker.update(0, {
                "step": "Aggregating chunk results",
                "chunks_processed": total_chunks
            })
        
        # Aggregate results from all chunks
        if chunk_results['date_stats'] and valid_count > 0:
            # Aggregate date stats
            valid_stats = [stats for stats in chunk_results['date_stats'] if stats['valid_count'] > 0]
            if valid_stats:
                overall_min = min(stats['min_date'] for stats in valid_stats)
                overall_max = max(stats['max_date'] for stats in valid_stats)
                
                results.update({
                    'valid_count': int(valid_count),
                    'min_date': overall_min.strftime('%Y-%m-%d') if not pd.isna(overall_min) else None,
                    'max_date': overall_max.strftime('%Y-%m-%d') if not pd.isna(overall_max) else None
                })
            
            # Aggregate distributions
            aggregated_distributions = {}
            for chunk_dist in chunk_results['distributions']:
                for dist_type, dist_data in chunk_dist.items():
                    if dist_type not in aggregated_distributions:
                        aggregated_distributions[dist_type] = {}
                    
                    for key, count in dist_data.items():
                        if key in aggregated_distributions[dist_type]:
                            aggregated_distributions[dist_type][key] += count
                        else:
                            aggregated_distributions[dist_type][key] = count
            
            results.update(aggregated_distributions)
        
        if progress_tracker:
            progress_tracker.update(0, {
                "step": "Chunked processing completed",
                "total_chunks": total_chunks
            })
        
        logger.info(f"Chunked processing completed successfully")
    else:
        if valid_count > 0:
            # Calculate date range and distributions if we have valid dates
            # Get date range
            date_stats = calculate_date_stats(dates)
            results.update(date_stats)

            # Calculate distributions
            distributions = calculate_distributions(dates)
            results.update(distributions)
    
    if valid_count > 0:
        # Analyze anomalies
        anomalies = detect_date_anomalies(df[field_name], min_year=min_year, max_year=max_year)
        results['anomalies'] = {k: len(v) for k, v in anomalies.items()}

        # Include examples of anomalies
        for anomaly_type, examples in anomalies.items():
            if examples:
                results[f'{anomaly_type}_examples'] = examples[:10]  # First 10 examples

    # Group analysis if id_column is specified (applies to all processing methods)
    if id_column and id_column in df.columns:
        group_changes = detect_date_changes_within_group(df, id_column, field_name)
        results['date_changes_within_group'] = group_changes

    # UID analysis if uid_column is specified (applies to all processing methods)
    if uid_column and uid_column in df.columns:
        uid_inconsistencies = detect_date_inconsistencies_by_uid(df, uid_column, field_name)
        results['date_inconsistencies_by_uid'] = uid_inconsistencies

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
        calc_memory_mb = field_memory_mb * 2  # Multiplication factor for intermediate calculations

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
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_time_seconds': estimated_time_seconds,
            'recommended_chunk_size': min(100000, max(10000, row_count // 10)),
            'use_chunks_recommended': row_count > 100000
        }
    else:
        # Field not found, return minimal estimates
        return {
            'estimated_memory_mb': 10,
            'estimated_time_seconds': 1,
            'error': f"Field {field_name} not found in DataFrame"
        }

def process_date_chunk(chunk_index: int, chunk_data: pd.DataFrame, field_name: str, 
                      min_year: int = 1940, max_year: int = 2005) -> Dict[str, Any]:
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
            'chunk_index': chunk_index,
            'error': f"Field {field_name} not found in chunk",
            'valid_count': 0,
            'distributions': {},
            'anomalies': {}
        }
    
    # Convert to datetime for this chunk
    chunk_dates = pd.to_datetime(chunk_data[field_name], errors='coerce')
    valid_mask = ~chunk_dates.isna()
    valid_dates = chunk_dates[valid_mask]
    
    chunk_result = {
        'chunk_index': chunk_index,
        'valid_count': len(valid_dates),
        'distributions': {},
        'anomalies': {},
        'date_stats': {}
    }
    
    if len(valid_dates) == 0:
        return chunk_result
    
    # Calculate date stats for this chunk
    min_date = valid_dates.min()
    max_date = valid_dates.max()
    chunk_result['date_stats'] = {
        'min_date': min_date,
        'max_date': max_date,
        'valid_count': len(valid_dates)
    }
    
    # Calculate distributions for this chunk using vectorized operations
    distributions = {}
    
    # Year distribution
    year_counts = valid_dates.dt.year.value_counts()
    distributions['year'] = year_counts.to_dict()
    
    # Month distribution
    month_counts = valid_dates.dt.month.value_counts()
    distributions['month'] = month_counts.to_dict()
    
    # Day of week distribution
    dow_counts = valid_dates.dt.dayofweek.value_counts()
    distributions['dow'] = dow_counts.to_dict()
    
    # Decade distribution
    decade_counts = (valid_dates.dt.year // 10 * 10).value_counts()
    distributions['decade'] = decade_counts.to_dict()
    
    chunk_result['distributions'] = distributions
    
    # Anomaly detection using vectorized operations
    anomalies = {
        'invalid_format': 0,
        'too_old': 0,
        'future_dates': 0,
        'too_young': 0,
        'negative_years': 0
    }
    
    # Check for anomalies using vectorized operations
    current_year = datetime.now().year
    years = valid_dates.dt.year
    
    # Count anomalies
    anomalies['too_old'] = (years < min_year).sum()
    anomalies['future_dates'] = (years > current_year).sum()
    anomalies['too_young'] = ((years > max_year) & (years <= current_year)).sum()
    
    # Check for invalid formats (non-null values that couldn't be converted)
    non_null_mask = chunk_data[field_name].notna()
    invalid_mask = non_null_mask & chunk_dates.isna()
    anomalies['invalid_format'] = invalid_mask.sum()
    
    # Check for negative years (simple string check for performance)
    if invalid_mask.any():
        negative_strings = chunk_data.loc[invalid_mask, field_name].astype(str).str.startswith('-')
        anomalies['negative_years'] = negative_strings.sum()
        anomalies['invalid_format'] -= anomalies['negative_years']  # Adjust count
    
    chunk_result['anomalies'] = anomalies
    
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
    valid_chunks = [chunk for chunk in chunk_results if 'error' not in chunk]
    
    if not valid_chunks:
        return {'error': 'No valid chunks processed'}
    
    # Aggregate date stats
    min_dates = []
    max_dates = []
    total_valid_count = 0
    
    for chunk in valid_chunks:
        if chunk['valid_count'] > 0 and 'date_stats' in chunk:
            min_dates.append(chunk['date_stats']['min_date'])
            max_dates.append(chunk['date_stats']['max_date'])
            total_valid_count += chunk['valid_count']
    
    aggregated = {}
    
    if min_dates and max_dates:
        overall_min = min(min_dates)
        overall_max = max(max_dates)
        aggregated.update({
            'valid_count': total_valid_count,
            'min_date': overall_min.strftime('%Y-%m-%d') if not pd.isna(overall_min) else None,
            'max_date': overall_max.strftime('%Y-%m-%d') if not pd.isna(overall_max) else None
        })
    
    # Aggregate distributions
    combined_distributions = {
        'year': {},
        'month': {},
        'dow': {},
        'decade': {}
    }
    
    for chunk in valid_chunks:
        distributions = chunk.get('distributions', {})
        for dist_type in combined_distributions:
            if dist_type in distributions:
                for key, count in distributions[dist_type].items():
                    combined_distributions[dist_type][key] = combined_distributions[dist_type].get(key, 0) + count
    
    # Convert to expected format
    if combined_distributions['year']:
        aggregated['year_distribution'] = {str(k): int(v) for k, v in combined_distributions['year'].items()}
    
    if combined_distributions['month']:
        aggregated['month_distribution'] = {str(k): int(v) for k, v in combined_distributions['month'].items()}
    
    if combined_distributions['dow']:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        aggregated['day_of_week_distribution'] = {
            day_names[int(k)]: int(v) for k, v in combined_distributions['dow'].items()
        }
    
    if combined_distributions['decade']:
        aggregated['decade_distribution'] = {f"{k}s": int(v) for k, v in combined_distributions['decade'].items()}
    
    # Aggregate anomalies
    combined_anomalies = {
        'invalid_format': 0,
        'too_old': 0,
        'future_dates': 0,
        'too_young': 0,
        'negative_years': 0
    }
    
    for chunk in valid_chunks:
        anomalies = chunk.get('anomalies', {})
        for anomaly_type in combined_anomalies:
            if anomaly_type in anomalies:
                combined_anomalies[anomaly_type] += anomalies[anomaly_type]
    
    aggregated['anomalies'] = combined_anomalies
    
    return aggregated