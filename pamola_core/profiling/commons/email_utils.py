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

import logging
import re
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from pamola_core.profiling.commons.analysis_utils import process_dataframe_with_config
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
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
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
        parts = email.strip().split('@')
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
    # Define patterns
    patterns = {
        'name_dot_surname': r'^[a-zA-Z]+\.[a-zA-Z]+@',
        'name_underscore_surname': r'^[a-zA-Z]+_[a-zA-Z]+@',
        'surname_dot_name': r'^[a-zA-Z]+\.[a-zA-Z]+@',
        'surname_underscore_name': r'^[a-zA-Z]+_[a-zA-Z]+@',
        'name_surname': r'^[a-zA-Z]+[a-zA-Z]+@',
        'surname_name': r'^[a-zA-Z]+[a-zA-Z]+@',
    }

    # Count patterns
    pattern_counts = {pattern_name: 0 for pattern_name in patterns}
    total_valid = 0

    for email in emails.dropna():
        if is_valid_email(email):
            total_valid += 1
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, email):
                    pattern_counts[pattern_name] += 1

    # Calculate percentages
    pattern_percentages = {}
    if total_valid > 0:
        for pattern_name, count in pattern_counts.items():
            pattern_percentages[pattern_name] = round((count / total_valid) * 100, 2)

    return {
        'total_valid_emails': total_valid,
        'pattern_counts': pattern_counts,
        'pattern_percentages': pattern_percentages
    }


def analyze_email_field(df: pd.DataFrame,
                        field_name: str,
                        top_n: int = 20,
                        use_dask: bool = False,
                        use_vectorization: bool = False,
                        chunk_size: int = 1000,
                        npartitions: int = 1,
                        parallel_processes: int = 1,
                        progress_tracker: Optional[HierarchicalProgressTracker] = None,
                        task_logger: Optional[logging.Logger] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Analyze an email field in the given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
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
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    if task_logger:
        logger = task_logger

    logger.info(f"Analyzing email field: {field_name}")

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Basic statistics
    total_rows = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_rows - null_count

    # Validate emails
    valid_emails = df[field_name].apply(lambda x: is_valid_email(x) if not pd.isna(x) else False)
    if valid_emails.dtype.name == 'category':
        valid_emails = valid_emails.astype(bool)
    valid_count = valid_emails.sum()
    invalid_count = non_null_count - valid_count

    # Extract domains
    domains = {}
    def process_batch(batch):
        batch[f"{field_name}_analyzer"] = batch[field_name].dropna().apply(extract_email_domain)

        return batch

    processed_df = process_dataframe_with_config(
        df=df,
        process_function=process_batch,
        chunk_size=chunk_size,
        use_dask=use_dask,
        npartitions=npartitions,
        meta=None,
        use_vectorization=use_vectorization,
        parallel_processes=parallel_processes,
        progress_tracker=progress_tracker,
        task_logger=logger
    )

    if isinstance(processed_df, pd.DataFrame):
        domains = processed_df[f"{field_name}_analyzer"].value_counts().to_dict()

    # Sort domains by frequency in descending order
    domains = dict(sorted(domains.items(), key=lambda x: x[1], reverse=True))

    # Top domains
    top_domains = {k: domains[k] for k in list(domains.keys())[:top_n]} if len(domains) > top_n else domains

    # Analyze personal patterns in emails
    personal_patterns = detect_personal_patterns(df[field_name])

    # Create result stats
    stats = {
        'total_rows': total_rows,
        'null_count': int(null_count),
        'null_percentage': round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0,
        'non_null_count': int(non_null_count),
        'valid_count': int(valid_count),
        'valid_percentage': round((valid_count / total_rows) * 100, 2) if total_rows > 0 else 0,
        'invalid_count': int(invalid_count),
        'invalid_percentage': round((invalid_count / total_rows) * 100, 2) if total_rows > 0 else 0,
        'unique_domains': len(domains),
        'top_domains': top_domains,
        'personal_patterns': personal_patterns
    }

    return stats


def create_domain_dictionary(
    df: pd.DataFrame, field_name: str, min_count: int = 1
) -> Dict[str, Any]:
    """
    Create a frequency dictionary for email domains.

    Parameters:
    -----------
    df : pd.DataFrame
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

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

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
        filtered_domains = {domain: count for domain, count in domains.items() if count >= min_count}

        # Convert to list format
        domain_list = [{'domain': domain, 'count': count} for domain, count in filtered_domains.items()]

        # Add percentages
        total_emails = sum(filtered_domains.values())
        for item in domain_list:
            item['percentage'] = round((item['count'] / total_emails) * 100, 2) if total_emails > 0 else 0

        return {
            'field_name': field_name,
            'total_domains': len(domain_list),
            'total_emails': total_emails,
            'domains': domain_list
        }

    except Exception as e:
        logger.error(f"Error creating domain dictionary for {field_name}: {e}", exc_info=True)
        return {'error': str(e)}


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
        return {'error': f"Field {field_name} not found in DataFrame"}

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
    estimated_unique_domains = int(len(sample_domains) * (non_null_count / max(1, sample_size)))

    # Calculate memory estimation
    avg_email_length = sample.str.len().mean() if len(sample) > 0 else 0
    avg_domain_length = np.mean([len(d) for d in sample_domains]) if sample_domains else 0

    # Estimate memory requirements
    estimated_memory_mb = (
                                  (non_null_count * avg_email_length * 2) +  # Original emails (2 bytes per char)
                                  (estimated_unique_domains * avg_domain_length * 2) +  # Domain dictionary
                                  (non_null_count * 20)  # Additional data structures, approximately 20 bytes per record
                          ) / (1024 * 1024)  # Convert to MB

    return {
        'total_rows': total_rows,
        'non_null_count': int(non_null_count),
        'estimated_valid_emails': int(non_null_count * 0.95),  # Assuming 95% of non-null are valid
        'estimated_unique_domains': estimated_unique_domains,
        'estimated_memory_mb': round(estimated_memory_mb, 2),
        'estimated_processing_time_sec': round(non_null_count * 0.0001, 2),  # Rough estimate
    }