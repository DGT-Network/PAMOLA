"""
Email data analysis utilities for the anonymization project.

This module provides analytical functions for email data fields,
including email validation, domain extraction, and pattern detection.
Functions in this module focus purely on analytics, without dependencies
on operation infrastructure, IO, or visualization components.

Main functions:
- is_valid_email: Validate email format
- extract_email_domain: Extract domain from email address
- detect_personal_patterns: Identify name patterns in emails
- analyze_email_field: Perform complete email field analysis
- create_domain_dictionary: Create frequency dictionary for domains
- estimate_resources: Estimate resources for analysis
"""

from collections import Counter, defaultdict
import logging
import re
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import dask.dataframe as dd
from joblib import Parallel, delayed

from pamola_core.common.constants import Constants
from pamola_core.profiling.commons.analysis_utils import process_dataframe_with_config
from pamola_core.utils.io_helpers.dask_utils import get_computed_df
from pamola_core.utils.ops.op_data_processing import get_dataframe_chunks
from pamola_core.utils.progress import HierarchicalProgressTracker

# Configure logger
logger = logging.getLogger(__name__)


def is_valid_email(value) -> bool:
    """
    Validate if a value is a properly formatted email address.

    Parameters:
    -----------
    value : Any
        The value to validate

    Returns:
    --------
    bool
        True if the value is a valid email address, False otherwise
    """
    if pd.isna(value) or not isinstance(value, str):
        return False

    # Basic email regex pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, value))


def extract_email_domain(email: str) -> Optional[str]:
    """
    Extract the domain part from an email address.

    Parameters:
    -----------
    email : str
        The email address to analyze

    Returns:
    --------
    Optional[str]
        The domain part of the email, or None if invalid format
    """
    if not isinstance(email, str):
        return None

    # Simple split approach to extract domain
    try:
        parts = email.strip().split("@")
        if len(parts) == 2 and parts[1]:
            return parts[1].lower()
    except Exception:
        pass

    return None


def detect_personal_patterns(emails: pd.Series) -> Dict[str, Any]:
    """
    Detect personal patterns in email addresses (name.surname@domain, etc.).

    Parameters:
    -----------
    emails : pd.Series
        Series containing email addresses to analyze

    Returns:
    --------
    Dict[str, Any]
        Statistics about personal patterns
    """
    # Count patterns
    pattern_counts = {
        pattern_name: 0 for pattern_name in Constants.COMMON_EMAIL_PATTERNS
    }
    total_valid = 0

    for email in emails.dropna():
        if is_valid_email(email):
            total_valid += 1
            for pattern_name, pattern in Constants.COMMON_EMAIL_PATTERNS.items():
                if re.search(pattern, email):
                    pattern_counts[pattern_name] += 1

    # Calculate percentages
    pattern_percentages = {}
    if total_valid > 0:
        for pattern_name, count in pattern_counts.items():
            pattern_percentages[pattern_name] = round((count / total_valid) * 100, 2)

    return {
        "total_valid_emails": total_valid,
        "pattern_counts": pattern_counts,
        "pattern_percentages": pattern_percentages,
    }


def _process_with_dask(
    ddf: dd.DataFrame, field_name: str, top_n: int, current_logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze email field using Dask DataFrame for memory-efficient processing.

    Parameters:
    -----------
    df : dd.DataFrame
        The Dask DataFrame containing the data to analyze.
    field_name : str
        The name of the email field to analyze.
    top_n : int
        Number of top domains to include in the results.
    current_logger : logging.Logger
        Logger for tracking task progress and debugging.

    Returns:
    --------
    Dict[str, Any]
        Analysis results containing counts, domains, and patterns.
    """
    current_logger.info("Parallel Enabled")
    current_logger.info("Parallel Engine: Dask")

    current_logger.info("Processing with Dask DataFrame - memory optimized mode")

    total_rows = int(ddf.map_partitions(len).sum().compute())

    null_count = ddf[field_name].isna().sum().compute()

    non_null_count = total_rows - null_count 
    
    # Single map_partitions call for comprehensive email analysis
    def analyze_partition(partition):
        """Analyze emails in a partition - validate, extract domains, and sample for patterns"""
        result = _analyze_email_data(partition)
        return pd.DataFrame([result])

    # Apply the comprehensive analysis to all partitions
    partition_results = (
        ddf[field_name]
        .map_partitions(
            analyze_partition,
            meta={
                "valid_count": int,
                "domains": object,
                "pattern_counts": object,
            },
        )
        .compute()
    )

    # Aggregate results from all partitions
    total_valid_count = 0
    all_domains = []
    total_pattern_counts = {}

    for _, result in partition_results.iterrows():
        total_valid_count += result["valid_count"]
        all_domains.extend(result["domains"])

        for pattern_name, count in result["pattern_counts"].items():
            total_pattern_counts[pattern_name] = (
                total_pattern_counts.get(pattern_name, 0) + count
            )

    invalid_count = non_null_count - total_valid_count

    # Top domain counts
    domain_counts = Counter(all_domains)
    sorted_domains = dict(domain_counts.most_common())
    top_domains = dict(list(sorted_domains.items())[:top_n])

    # Pattern percentages
    pattern_percentages = {}
    if total_valid_count > 0:
        for pattern_name, count in total_pattern_counts.items():
            pattern_percentages[pattern_name] = round(
                (count / total_valid_count) * 100, 2
            )

    personal_patterns = {
        "total_valid_emails": total_valid_count,
        "pattern_counts": total_pattern_counts,
        "pattern_percentages": pattern_percentages,
    }

    return {
        "total_rows": total_rows,
        "null_count": null_count,
        "non_null_count": non_null_count,
        "valid_count": total_valid_count,
        "invalid_count": invalid_count,
        "domains": sorted_domains,
        "top_domains": top_domains,
        "personal_patterns": personal_patterns,
    }


def _get_basic_statistics(df: pd.DataFrame, field_name: str) -> Dict[str, int]:
    """
    Compute basic statistics for a DataFrame column: total rows, null count, and non-null count.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze.
    field_name : str
        The name of the field to analyze.

    Returns:
    --------
    Dict[str, int]
        Dictionary with total_rows, null_count, non_null_count.
    """
    total_rows = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_rows - null_count
    return {
        "total_rows": total_rows,
        "null_count": null_count,
        "non_null_count": non_null_count,
    }


def _process_small_dataset(
    df: pd.DataFrame, field_name: str, top_n: int, current_logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze email field using pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame containing the data to analyze.
    field_name : str
        The name of the email field to analyze.
    top_n : int
        Number of top domains to include in the results.    use_vectorization : bool
        Whether to use vectorized operations for processing.
    current_logger : logging.Logger
        Logger for tracking task progress and debugging.    Returns:
    --------
    Dict[str, Any]
        Analysis results containing counts, domains, and patterns.
    """
    current_logger.info("Processing with pandas DataFrame")

    stats = _get_basic_statistics(df, field_name)
    total_rows = stats["total_rows"]
    null_count = stats["null_count"]
    non_null_count = stats["non_null_count"]

    # Use shared email analysis function
    analysis_result = _analyze_email_data(df, field_name)
    
    valid_count = analysis_result["valid_count"]
    invalid_count = non_null_count - valid_count
    
    # Process domains
    domain_counts = Counter(analysis_result["domains"])
    domains = dict(domain_counts.most_common())
    
    # Top domains
    top_domains = (
        {k: domains[k] for k in list(domains.keys())[:top_n]}
        if len(domains) > top_n
        else domains
    )

    # Calculate pattern percentages
    pattern_percentages = {}
    total_valid = analysis_result["valid_count"]
    if total_valid > 0:
        for pattern_name, count in analysis_result["pattern_counts"].items():
            pattern_percentages[pattern_name] = round((count / total_valid) * 100, 2)

    personal_patterns = {
        "total_valid_emails": total_valid,
        "pattern_counts": analysis_result["pattern_counts"],
        "pattern_percentages": pattern_percentages,
    }

    return {
        "total_rows": total_rows,
        "null_count": null_count,
        "non_null_count": non_null_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "domains": domains,
        "top_domains": top_domains,
        "personal_patterns": personal_patterns,
    }


def _process_with_joblib(
    df: pd.DataFrame,
    field_name: str,
    top_n: int,
    current_logger: logging.Logger,
    chunk_size: Optional[int] = 10000,
    n_jobs: Optional[int] = 1,
) -> Dict[str, Any]:
    """
    Analyze email field using joblib for parallel processing.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame containing the data to analyze.
    field_name : str
        The name of the email field to analyze.
    top_n : int
        Number of top domains to include in the results.
    chunk_size : int
        Size of chunks for parallel processing.
    n_jobs : int
        Number of parallel jobs (-1 for all available cores).
    current_logger : logging.Logger
        Logger for tracking task progress and debugging.

    Returns:
    --------
    Dict[str, Any]
        Analysis results containing counts, domains, and patterns.
    """
    current_logger.info("Parallel Enabled")
    current_logger.info("Parallel Engine: Joblib")
    current_logger.info(f"Processing with joblib - n_jobs: {n_jobs}")

    stats = _get_basic_statistics(df, field_name)
    total_rows = stats["total_rows"]
    null_count = stats["null_count"]
    non_null_count = stats["non_null_count"]

    def analyze_chunk(chunk_data):
        """Analyze emails in a chunk - validate, extract domains, and count patterns"""
        return _analyze_email_data(chunk_data, field_name)

    # Split the DataFrame into chunks
    chunks = list(get_dataframe_chunks(df, chunk_size=chunk_size))

    current_logger.info(f"Split data into {len(chunks)} chunks for parallel processing")

    # Parallel processing with joblib
    results = Parallel(n_jobs=n_jobs)(delayed(analyze_chunk)(chunk) for chunk in chunks)

    # Aggregate results from all chunks
    total_valid_count = 0
    all_domains = []
    total_pattern_counts = {
        pattern_name: 0 for pattern_name in Constants.COMMON_EMAIL_PATTERNS
    }

    for result in results:
        if result:
            total_valid_count += result["valid_count"]
            all_domains.extend(result["domains"])
            for pattern_name, count in result["pattern_counts"].items():
                total_pattern_counts[pattern_name] += count

    invalid_count = non_null_count - total_valid_count

    # Top domain counts
    domain_counts = Counter(all_domains)
    sorted_domains = dict(domain_counts.most_common())
    top_domains = dict(list(sorted_domains.items())[:top_n])

    # Pattern percentages
    pattern_percentages = {}
    if total_valid_count > 0:
        for pattern_name, count in total_pattern_counts.items():
            pattern_percentages[pattern_name] = round(
                (count / total_valid_count) * 100, 2
            )

    personal_patterns = {
        "total_valid_emails": total_valid_count,
        "pattern_counts": total_pattern_counts,
        "pattern_percentages": pattern_percentages,
    }

    return {
        "total_rows": total_rows,
        "null_count": null_count,
        "non_null_count": non_null_count,
        "valid_count": total_valid_count,
        "invalid_count": invalid_count,
        "domains": sorted_domains,
        "top_domains": top_domains,
        "personal_patterns": personal_patterns,
    }


def _process_with_chunks(
    df: pd.DataFrame,
    field_name: str,
    top_n: int,
    chunk_size: int,
    current_logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Analyze email field using index-based DataFrame chunking for sequential processing.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame containing the data to analyze.
    field_name : str
        The name of the email field to analyze.
    top_n : int
        Number of top domains to include in the results.
    chunk_size : int
        Size of chunks for processing.
    current_logger : logging.Logger
        Logger for tracking task progress and debugging.

    Returns:
    --------
    Dict[str, Any]
        Analysis results containing counts, domains, and patterns.
    """
    current_logger.info("Sequential Processing")
    current_logger.info("Processing Engine: Index-based Chunks")
    current_logger.info(
        f"Processing with index-based chunks - chunk size: {chunk_size}"
    )

    stats = _get_basic_statistics(df, field_name)
    total_rows = stats["total_rows"]
    null_count = stats["null_count"]
    non_null_count = stats["non_null_count"]

    # Split DataFrame into chunks using index-based slicing
    total_records = len(df)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    current_logger.info(f"Split DataFrame into {total_chunks} chunks for processing")
    current_logger.info(f"Total records: {total_rows}, Chunk size: {chunk_size}")

    # Initialize aggregation variables
    total_valid_count = 0
    all_domains = []
    total_pattern_counts = defaultdict(int)    # Process each chunk sequentially
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_records)
        chunk = df.iloc[start_idx:end_idx]

        current_logger.info(
            f"Processing chunk {i + 1}/{total_chunks} (rows {start_idx} to {end_idx - 1})"
        )        # Use shared email analysis function
        chunk_result = _analyze_email_data(chunk, field_name)

        # Aggregate chunk results
        total_valid_count += chunk_result["valid_count"]
        all_domains.extend(chunk_result["domains"])
        for pattern_name, count in chunk_result["pattern_counts"].items():
            total_pattern_counts[pattern_name] += count

        current_logger.info(
            f"Chunk {i+1} completed: {chunk_result['valid_count']} valid emails, {len(chunk_result['domains'])} domains"
        )

    invalid_count = non_null_count - total_valid_count

    # Top domain counts
    domain_counts = Counter(all_domains)
    sorted_domains = dict(domain_counts.most_common())
    top_domains = dict(list(sorted_domains.items())[:top_n])

    # Pattern percentages
    pattern_percentages = {}
    if total_valid_count > 0:
        for pattern_name, count in total_pattern_counts.items():
            pattern_percentages[pattern_name] = round(
                (count / total_valid_count) * 100, 2
            )

    pattern_counts_full = {
        k: total_pattern_counts.get(k, 0) for k in Constants.COMMON_EMAIL_PATTERNS
    }
    pattern_percentages_full = {
        k: pattern_percentages.get(k, 0.0) for k in Constants.COMMON_EMAIL_PATTERNS
    }

    personal_patterns = {
        "total_valid_emails": total_valid_count,
        "pattern_counts": pattern_counts_full,
        "pattern_percentages": pattern_percentages_full,
    }

    current_logger.info(
        f"Processing completed: {total_valid_count} valid emails, {len(sorted_domains)} unique domains"
    )

    return {
        "total_rows": total_rows,
        "null_count": null_count,
        "non_null_count": non_null_count,
        "valid_count": total_valid_count,
        "invalid_count": invalid_count,
        "domains": sorted_domains,
        "top_domains": top_domains,
        "personal_patterns": personal_patterns,
    }


def _analyze_email_data(email_series: Union[pd.Series, pd.DataFrame], field_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Shared function to analyze emails in a data chunk/partition.
    
    Parameters:
    -----------
    email_series : Union[pd.Series, pd.DataFrame]
        Series containing email data to analyze, or DataFrame with email column. 
        Can be a partition (Dask) or chunk data (Joblib).
    field_name : Optional[str]
        Field name for chunk data access. If None, treats email_series as the email column directly.
        
    Returns:
    --------
    Dict[str, Any]
        Analysis results containing valid_count, domains, and pattern_counts.
    """
    valid_count = 0
    domains = []
    pattern_counts = {
        pattern_name: 0 for pattern_name in Constants.COMMON_EMAIL_PATTERNS
    }    # Handle different input types - if field_name is provided, extract the column
    if field_name is not None:
        email_data = email_series[field_name].dropna()
    else:
        email_data = email_series.dropna()

    for email in email_data:
        # Ensure email is a string for validation
        if isinstance(email, str) and is_valid_email(email):
            valid_count += 1

            # Extract domain
            domain = extract_email_domain(email)
            if domain:
                domains.append(domain)

            # Pattern analysis
            for pattern_name, pattern in Constants.COMMON_EMAIL_PATTERNS.items():
                if re.search(pattern, email):
                    pattern_counts[pattern_name] += 1

    return {
        "valid_count": valid_count,
        "domains": domains,
        "pattern_counts": pattern_counts,
    }


def analyze_email_field(
    df: Union[pd.DataFrame, dd.DataFrame],
    field_name: str,
    top_n: int = 20,
    use_dask: bool = False,
    use_vectorization: bool = False,
    chunk_size: int = 1000,
    parallel_processes: Optional[int] = 1,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Analyze an email field in the given DataFrame.

    Parameters:
    -----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The DataFrame containing the data to analyze.
    field_name : str
        The name of the email field to analyze.
    top_n : int, default 20
        Number of top domains to include in the results.
    use_dask : bool, default False
        Whether to use Dask for parallel processing.
    use_vectorization : bool, default False
        Whether to use vectorized operations for processing.
    chunk_size : int, default 1000
        Batch size for chunked processing.
    npartitions : int, default 1
        Number of partitions if using Dask.
    parallel_processes : int, default 1
        Number of parallel processes to use.
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker object for monitoring progress.
    task_logger : Optional[logging.Logger]
        Logger for tracking task progress and debugging.    

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    if task_logger:
        current_logger = task_logger
    else:
        current_logger = logger

    current_logger.info(f"Analyzing email field: {field_name}")

    if field_name not in df.columns:
        return {"error": f"Field {field_name} not found in DataFrame"}

    total_rows = (
        int(df.map_partitions(len).sum().compute())
        if isinstance(df, dd.DataFrame)
        else len(df)
    )

    is_large_df = total_rows > chunk_size

    if is_large_df is False:
        if task_logger:
            task_logger.warning("Small DataFrame! Process as usual")
        analysis_result = _process_small_dataset(
            get_computed_df(df), field_name, top_n, current_logger
        )
    else:        # Process based on DataFrame type and configuration
        if use_dask and isinstance(df, dd.DataFrame):
            analysis_result = _process_with_dask(df, field_name, top_n, current_logger)
        elif use_vectorization and parallel_processes is not None and parallel_processes > 1:
            # Use joblib for parallel processing
            analysis_result = _process_with_joblib(
                get_computed_df(df),
                field_name,
                top_n,
                current_logger,
                chunk_size,
                parallel_processes,
            )
        else:
            analysis_result = _process_with_chunks(
                get_computed_df(df), field_name, top_n, chunk_size, current_logger
            )

    # Extract values from analysis result
    total_rows = analysis_result["total_rows"]
    null_count = analysis_result["null_count"]
    non_null_count = analysis_result["non_null_count"]
    valid_count = analysis_result["valid_count"]
    invalid_count = analysis_result["invalid_count"]
    domains = analysis_result["domains"]
    top_domains = analysis_result["top_domains"]
    personal_patterns = analysis_result["personal_patterns"]

    # Create result stats
    stats = {
        "total_rows": int(total_rows),
        "null_count": int(null_count),
        "null_percentage": (
            round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0
        ),
        "non_null_count": int(non_null_count),
        "valid_count": int(valid_count),
        "valid_percentage": (
            round((valid_count / total_rows) * 100, 2) if total_rows > 0 else 0
        ),
        "invalid_count": int(invalid_count),
        "invalid_percentage": (
            round((invalid_count / total_rows) * 100, 2) if total_rows > 0 else 0
        ),
        "unique_domains": len(domains),
        "top_domains": top_domains,
        "personal_patterns": personal_patterns,
    }

    return stats


def create_domain_dictionary(
    df: Union[pd.DataFrame, dd.DataFrame], field_name: str, min_count: int = 1, **kwargs
) -> Dict[str, Any]:
    """
    Create a frequency dictionary for email domains.

    Parameters:
    -----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The DataFrame containing the data
    field_name : str
        The name of the email field
    min_count : int
        Minimum frequency for inclusion in the dictionary
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    Dict[str, Any]
        Dictionary with domain frequency data and metadata
    """
    logger.info(f"Creating domain dictionary for email field: {field_name}")

    df = get_computed_df(df)

    if field_name not in df.columns:
        return {"error": f"Field {field_name} not found in DataFrame"}

    try:
        # Extract domains
        domains = {}
        for email in df[field_name].dropna():
            domain = extract_email_domain(email)
            if domain:
                domains[domain] = domains.get(domain, 0) + 1

        # Sort by frequency in descending order
        domains = dict(sorted(domains.items(), key=lambda x: x[1], reverse=True))

        # Filter by minimum count
        filtered_domains = {
            domain: count for domain, count in domains.items() if count >= min_count
        }

        # Convert to list format
        domain_list = [
            {"domain": domain, "count": count}
            for domain, count in filtered_domains.items()
        ]

        # Add percentages
        total_emails = sum(filtered_domains.values())
        for item in domain_list:
            item["percentage"] = (
                round((item["count"] / total_emails) * 100, 2)
                if total_emails > 0
                else 0
            )

        return {
            "field_name": field_name,
            "total_domains": len(domain_list),
            "total_emails": total_emails,
            "domains": domain_list,
        }

    except Exception as e:
        logger.error(
            f"Error creating domain dictionary for {field_name}: {e}", exc_info=True
        )
        return {"error": str(e)}


def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """
    Estimate resources needed for analyzing the email field.

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

    # Get basic data about the field
    total_rows = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_rows - null_count

    # Sample a subset of values for quicker analysis
    sample_size = min(1000, non_null_count)
    if non_null_count > 0:
        sample = df[field_name].dropna().sample(n=sample_size, random_state=42)
    else:
        sample = pd.Series([])

    # Estimate number of unique domains from sample
    sample_domains = set()
    for email in sample:
        domain = extract_email_domain(email)
        if domain:
            sample_domains.add(domain)

    # Extrapolate to full dataset
    estimated_unique_domains = int(
        len(sample_domains) * (non_null_count / max(1, sample_size))
    )

    # Calculate memory estimation
    avg_email_length = sample.str.len().mean() if len(sample) > 0 else 0
    avg_domain_length = (
        np.mean([len(d) for d in sample_domains]) if sample_domains else 0
    )

    # Estimate memory requirements
    estimated_memory_mb = (
        (non_null_count * avg_email_length * 2)  # Original emails (2 bytes per char)
        + (estimated_unique_domains * avg_domain_length * 2)  # Domain dictionary
        + (
            non_null_count * 20
        )  # Additional data structures, approximately 20 bytes per record
    ) / (
        1024 * 1024
    )  # Convert to MB

    return {
        "total_rows": total_rows,
        "non_null_count": int(non_null_count),
        "estimated_valid_emails": int(
            non_null_count * 0.95
        ),  # Assuming 95% of non-null are valid
        "estimated_unique_domains": estimated_unique_domains,
        "estimated_memory_mb": round(estimated_memory_mb, 2),
        "estimated_processing_time_sec": round(
            non_null_count * 0.0001, 2
        ),  # Rough estimate
    }
