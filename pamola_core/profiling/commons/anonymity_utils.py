"""
K-anonymity utilities for the HHR project.

This module provides utilities for analyzing k-anonymity in data, including
functions for generating KA indices, calculating k-anonymity metrics,
calculating Shannon entropy, and detecting vulnerable records.
"""

import hashlib
import logging
from typing import Dict, List, Set, Any, Optional

import numpy as np
import pandas as pd

from pamola_core.utils.io import write_json
from pamola_core.utils.progress import ProgressTracker

# Configure logger
logger = logging.getLogger(__name__)


def generate_ka_index(fields: List[str], prefix_length: int = 2, max_prefix_length: int = 4,
                      existing_indices: Optional[Set[str]] = None) -> str:
    """
    Generate a KA index name from field names.

    Parameters:
    -----------
    fields : List[str]
        List of field names
    prefix_length : int
        Initial number of characters to use from each field name
    max_prefix_length : int
        Maximum number of characters to use from each field name
    existing_indices : Set[str], optional
        Set of existing indices to avoid collisions

    Returns:
    --------
    str
        Generated KA index name
    """
    if existing_indices is None:
        existing_indices = set()

    # Generate field abbreviations
    field_abbrs = [field[:prefix_length].lower() for field in fields]
    ka_suffix = "_".join(field_abbrs)
    ka_index = f"KA_{ka_suffix}"

    # Check for collisions, increase prefix length if needed
    current_prefix_length = prefix_length
    while ka_index in existing_indices and current_prefix_length < max_prefix_length:
        current_prefix_length += 1
        field_abbrs = [field[:current_prefix_length].lower() for field in fields]
        ka_suffix = "_".join(field_abbrs)
        ka_index = f"KA_{ka_suffix}"

    # If still collision, add numeric suffix
    if ka_index in existing_indices:
        # Try numeric suffixes
        counter = 1
        base_index = ka_index
        while ka_index in existing_indices:
            ka_index = f"{base_index}_{counter}"
            counter += 1

    # If still collision, use hash as fallback
    if ka_index in existing_indices:
        # Use hash of concatenated fields as fallback
        fields_str = "_".join(fields)

        hash_suffix = hashlib.md5(fields_str.encode()).hexdigest()[:8]
        ka_index = f"KA_hash_{hash_suffix}"

    return ka_index


def get_field_combinations(fields: List[str], min_size: int = 2, max_size: int = 4,
                           excluded_combinations: Optional[List[List[str]]] = None) -> List[List[str]]:
    """
    Generate all combinations of fields within given size range.

    Parameters:
    -----------
    fields : List[str]
        List of fields to combine
    min_size : int
        Minimum size of combinations
    max_size : int
        Maximum size of combinations
    excluded_combinations : List[List[str]], optional
        List of specific combinations to exclude

    Returns:
    --------
    List[List[str]]
        List of field combinations
    """
    from itertools import combinations

    if excluded_combinations is None:
        excluded_combinations = []

    # Convert exclusions to tuple sets for easier comparison
    excluded_sets = [set(combo) for combo in excluded_combinations]

    # Generate all possible combinations
    all_combinations = []
    for size in range(min_size, min(max_size + 1, len(fields) + 1)):
        for combo in combinations(fields, size):
            combo_set = set(combo)
            # Check if this combination should be excluded
            if not any(combo_set == excl for excl in excluded_sets):
                all_combinations.append(list(combo))

    return all_combinations


def create_ka_index_map(field_combinations: List[List[str]]) -> Dict[str, List[str]]:
    """
    Create a mapping between KA indices and their corresponding field combinations.

    Parameters:
    -----------
    field_combinations : List[List[str]]
        List of field combinations

    Returns:
    --------
    Dict[str, List[str]]
        Mapping from KA indices to field lists
    """
    index_map = {}
    existing_indices = set()

    for fields in field_combinations:
        ka_index = generate_ka_index(fields, existing_indices=existing_indices)
        index_map[ka_index] = fields
        existing_indices.add(ka_index)

    return index_map


def calculate_k_anonymity(df: pd.DataFrame, fields: List[str],
                          progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
    """
    Calculate k-anonymity metrics for a set of fields.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    fields : List[str]
        List of fields (quasi-identifiers)
    progress_tracker : ProgressTracker, optional
        Progress tracker for operations

    Returns:
    --------
    Dict[str, Any]
        K-anonymity metrics
    """
    # Track progress
    if progress_tracker:
        progress_tracker.update(0, {"fields": fields, "step": "Calculating k-anonymity"})

    # Check if all fields exist in the DataFrame
    missing_fields = [f for f in fields if f not in df.columns]
    if missing_fields:
        logger.warning(f"Missing fields in DataFrame: {missing_fields}")
        # Adjust fields to only include existing columns
        fields = [f for f in fields if f in df.columns]
        if not fields:
            return {
                "error": "None of the specified fields exist in the DataFrame",
                "missing_fields": missing_fields
            }

    try:
        # Group by the specified fields and count occurrences
        # Handle NaN values by treating them as a separate category
        groupby_result = df.groupby(fields, dropna=False).size().reset_index(name='k')

        # Get k values
        k_values = groupby_result['k'].values

        # Calculate basic metrics
        total_records = len(df)
        unique_groups = len(k_values)
        min_k = k_values.min() if len(k_values) > 0 else 0
        max_k = k_values.max() if len(k_values) > 0 else 0
        mean_k = k_values.mean() if len(k_values) > 0 else 0
        median_k = np.median(k_values) if len(k_values) > 0 else 0

        # Calculate uniqueness metrics
        k1_count = np.sum(k_values == 1)
        k1_percentage = (k1_count / total_records) * 100 if total_records > 0 else 0

        # Update progress if provided
        if progress_tracker:
            progress_tracker.update(0, {"step": "Calculating k distributions"})

        # Calculate k distributions for visualization
        k_ranges = {
            "k=1": np.sum(k_values == 1),
            "k=2-4": np.sum((k_values >= 2) & (k_values <= 4)),
            "k=5-9": np.sum((k_values >= 5) & (k_values <= 9)),
            "k=10-19": np.sum((k_values >= 10) & (k_values <= 19)),
            "k=20-49": np.sum((k_values >= 20) & (k_values <= 49)),
            "k=50-99": np.sum((k_values >= 50) & (k_values <= 99)),
            "k=100+": np.sum(k_values >= 100)
        }

        # Convert counts to percentages
        k_range_distribution = {
            k_range: (count / total_records) * 100 if total_records > 0 else 0
            for k_range, count in k_ranges.items()
        }

        # Calculate threshold metrics
        threshold_metrics = {
            "k≥2": (np.sum(k_values >= 2) / total_records) * 100 if total_records > 0 else 0,
            "k≥5": (np.sum(k_values >= 5) / total_records) * 100 if total_records > 0 else 0,
            "k≥10": (np.sum(k_values >= 10) / total_records) * 100 if total_records > 0 else 0,
            "k≥20": (np.sum(k_values >= 20) / total_records) * 100 if total_records > 0 else 0
        }

        # Update progress if provided
        if progress_tracker:
            progress_tracker.update(0, {"step": "Calculating entropy"})

        # Calculate entropy
        entropy = calculate_shannon_entropy(df, fields)

        # Normalize entropy
        normalized_entropy = normalize_entropy(entropy, unique_groups)

        # Compile results
        results = {
            "min_k": int(min_k),
            "max_k": int(max_k),
            "mean_k": float(mean_k),
            "median_k": float(median_k),
            "unique_groups": int(unique_groups),
            "unique_percentage": float((unique_groups / total_records) * 100) if total_records > 0 else 0,
            "k=1_count": int(k1_count),
            "k=1_percentage": float(k1_percentage),
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "k_range_distribution": k_range_distribution,
            "threshold_metrics": threshold_metrics,
            "total_records": int(total_records)
        }

        # Update progress if provided
        if progress_tracker:
            progress_tracker.update(0, {"step": "Completed k-anonymity calculation"})

        return results

    except Exception as e:
        logger.error(f"Error calculating k-anonymity: {e}", exc_info=True)
        return {"error": str(e)}


def calculate_shannon_entropy(df: pd.DataFrame, fields: List[str]) -> float:
    """
    Calculate Shannon entropy for a specific combination of fields.

    H(X) = -∑(p(x) * log2(p(x)))
    where p(x) is the probability of a specific combination of values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    fields : List[str]
        List of fields to calculate entropy for

    Returns:
    --------
    float
        Entropy value in bits
    """
    try:
        # Group by the fields and count occurrences
        value_counts = df.groupby(fields, dropna=False).size()

        # Calculate probabilities
        total_records = len(df)
        probabilities = value_counts / total_records

        # Calculate entropy: -∑(p(x) * log2(p(x)))
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy
    except Exception as e:
        logger.error(f"Error calculating entropy: {e}")
        return 0.0


def normalize_entropy(entropy: float, unique_values_count: int) -> float:
    """
    Normalize entropy to [0,1] range.

    H_norm(X) = H(X) / log2(n)
    where n is the number of unique values.

    Parameters:
    -----------
    entropy : float
        Raw entropy value
    unique_values_count : int
        Number of unique values/groups

    Returns:
    --------
    float
        Normalized entropy value
    """
    if unique_values_count <= 1:
        return 0.0

    # Maximum possible entropy is log2(n) where n is the number of unique values
    max_possible_entropy = np.log2(unique_values_count)

    # Avoid division by zero
    if max_possible_entropy == 0:
        return 0.0

    # Normalize
    normalized_entropy = entropy / max_possible_entropy

    return normalized_entropy


def find_vulnerable_records(df: pd.DataFrame, fields: List[str], k_threshold: int = 5,
                            max_examples: int = 10, id_field: Optional[str] = None) -> Dict[str, Any]:
    """
    Find and return information about vulnerable records (with k < threshold).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    fields : List[str]
        List of fields (quasi-identifiers)
    k_threshold : int
        Threshold for vulnerability (k < threshold)
    max_examples : int
        Maximum number of example records to include
    id_field : str, optional
        Field containing record IDs

    Returns:
    --------
    Dict[str, Any]
        Information about vulnerable records
    """
    try:
        # Group by fields and get size of each group
        group_sizes = df.groupby(fields, dropna=False).size().reset_index(name='k')

        # Filter to vulnerable groups (k < threshold)
        vulnerable_groups = group_sizes[group_sizes['k'] < k_threshold]

        # Count vulnerable records
        vulnerable_count = vulnerable_groups['k'].sum()
        total_records = len(df)
        vulnerable_percentage = (vulnerable_count / total_records) * 100 if total_records > 0 else 0

        # Get example vulnerable records if id_field is provided
        top_vulnerable_ids = []
        if id_field and id_field in df.columns:
            # Create a merged DataFrame with groups and their k values
            merged_df = pd.merge(df, vulnerable_groups, on=fields, how='inner')

            # Get top vulnerable IDs
            if len(merged_df) > 0:
                # Sort by k ascending and take max_examples
                merged_df = merged_df.sort_values('k')
                top_vulnerable_ids = merged_df[id_field].head(max_examples).tolist()

        return {
            "vulnerable_count": int(vulnerable_count),
            "vulnerable_percentage": float(vulnerable_percentage),
            "top_vulnerable_ids": top_vulnerable_ids
        }
    except Exception as e:
        logger.error(f"Error finding vulnerable records: {e}", exc_info=True)
        return {
            "error": str(e),
            "vulnerable_count": 0,
            "vulnerable_percentage": 0,
            "top_vulnerable_ids": []
        }


def prepare_metrics_for_spider_chart(ka_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Prepare metrics data for spider chart visualization.

    Parameters:
    -----------
    ka_metrics : Dict[str, Dict[str, Any]]
        Dictionary mapping KA indices to their metrics

    Returns:
    --------
    Dict[str, Dict[str, float]]
        Data formatted for spider chart
    """
    spider_data = {}

    for ka_index, metrics in ka_metrics.items():
        spider_data[ka_index] = {
            "Unique Records (%)": metrics.get("unique_percentage", 0),
            "Vulnerable Records (k<5) (%)": 100 - metrics.get("threshold_metrics", {}).get("k≥5", 0),
            "Normalized Average K": metrics.get("mean_k", 0) / 100 if metrics.get("mean_k", 0) > 100 else metrics.get(
                "mean_k", 0) / 100,
            "Entropy": metrics.get("normalized_entropy", 0)
        }

    return spider_data


def prepare_field_uniqueness_data(df: pd.DataFrame, fields: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Prepare data about the uniqueness of individual fields.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    fields : List[str]
        List of fields to analyze

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Data about field uniqueness
    """
    results = {}
    total_records = len(df)

    for field in fields:
        if field in df.columns:
            # Count unique values
            unique_values = df[field].nunique(dropna=False)

            # Calculate uniqueness percentage
            uniqueness_percentage = (unique_values / total_records) * 100 if total_records > 0 else 0

            results[field] = {
                "unique_values": int(unique_values),
                "uniqueness_percentage": float(uniqueness_percentage)
            }
        else:
            results[field] = {
                "error": f"Field {field} not found in DataFrame",
                "unique_values": 0,
                "uniqueness_percentage": 0
            }

    return results


def save_ka_index_map(ka_index_map: Dict[str, List[str]], output_path: str) -> str:
    """
    Save KA index mapping to a CSV file.

    Parameters:
    -----------
    ka_index_map : Dict[str, List[str]]
        Mapping from KA indices to field lists
    output_path : str
        Path to save the CSV file

    Returns:
    --------
    str
        Path to the saved file
    """
    try:
        # Convert to DataFrame
        data = []
        for ka_index, fields in ka_index_map.items():
            data.append({
                "KA_INDEX": ka_index,
                "FIELDS": ", ".join(fields)
            })

        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(output_path, index=False)

        return output_path
    except Exception as e:
        logger.error(f"Error saving KA index map: {e}")
        return str(e)


def save_ka_metrics(ka_metrics: Dict[str, Dict[str, Any]], output_path: str, ka_index_map: Dict[str, List[str]]) -> str:
    """
    Save K-anonymity metrics to a CSV file.

    Parameters:
    -----------
    ka_metrics : Dict[str, Dict[str, Any]]
        Dictionary of KA metrics
    output_path : str
        Path to save the CSV file
    ka_index_map : Dict[str, List[str]]
        Mapping from KA indices to field lists

    Returns:
    --------
    str
        Path to the saved file
    """
    try:
        # Convert to DataFrame
        data = []
        for i, (ka_index, metrics) in enumerate(ka_metrics.items(), 1):
            fields = ka_index_map.get(ka_index, [])

            data.append({
                "#": i,
                "KA_INDEX": ka_index,
                "FIELDS": ", ".join(fields),
                "KA_MIN": metrics.get("min_k", 0),
                "KA_MAX": metrics.get("max_k", 0),
                "KA_MEAN": metrics.get("mean_k", 0),
                "KA_MEDIAN": metrics.get("median_k", 0),
                "UNIQUE_VALUES (%)": metrics.get("unique_percentage", 0),
                "VULNERABLE_RECORDS (%)": 100 - metrics.get("threshold_metrics", {}).get("k≥5", 0),
                "ENTROPY": metrics.get("entropy", 0)
            })

        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(output_path, index=False)

        return output_path
    except Exception as e:
        logger.error(f"Error saving KA metrics: {e}")
        return str(e)


def save_vulnerable_records(vulnerable_records: Dict[str, Dict[str, Any]], output_path: str) -> str:
    """
    Save information about vulnerable records to a JSON file.

    Parameters:
    -----------
    vulnerable_records : Dict[str, Dict[str, Any]]
        Dictionary mapping KA indices to vulnerable record information
    output_path : str
        Path to save the JSON file

    Returns:
    --------
    str
        Path to the saved file
    """
    try:
        # Prepare data for JSON
        data = []
        for ka_index, info in vulnerable_records.items():
            data.append({
                "ka_index": ka_index,
                "min_k": info.get("min_k", 0),
                "vulnerable_count": info.get("vulnerable_count", 0),
                "vulnerable_percent": info.get("vulnerable_percentage", 0),
                "top_10_vulnerable_ids": info.get("top_vulnerable_ids", [])
            })

        # Save to JSON
        write_json(data, output_path)

        return output_path
    except Exception as e:
        logger.error(f"Error saving vulnerable records: {e}")
        return str(e)