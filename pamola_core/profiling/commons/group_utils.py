"""
Analytical utilities for group analysis in the project.

This module provides pamola_core analytical functions for evaluating variation within groups,
analyzing cross-group relationships, and supporting anonymization algorithms.
Separates analytical logic from operational components.
"""

import hashlib
import logging
from typing import Dict, List, Any, Set, Tuple, Optional, Union

import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


def calculate_field_variation(group: pd.DataFrame, field: str, handle_nulls: str = 'as_value') -> float:
    """
    Calculate variation for a single field within a group.

    Parameters:
    -----------
    group : pd.DataFrame
        Group of records (typically with same group_id)
    field : str
        Field name to analyze
    handle_nulls : str
        How to handle nulls: 'as_value' treats nulls as a separate value,
        'exclude' excludes nulls from calculation

    Returns:
    --------
    float
        Variation value from 0 to 1, where 0 means all values are identical
        and 1 means all values are different
    """
    if len(group) <= 1:
        return 0.0

    # Extract the field values
    values = group[field]

    if handle_nulls == 'exclude':
        values = values.dropna()
        if len(values) <= 1:
            return 0.0

    # For MVF fields (stored as string representations of lists)
    if values.dtype == 'object' and values.iloc[0] is not None and isinstance(values.iloc[0], str) and values.iloc[
        0].startswith('['):
        try:
            # Try to interpret as list representation
            import ast
            processed_values = values.apply(lambda x: tuple(sorted(ast.literal_eval(x))) if pd.notna(x) else None)
        except (SyntaxError, ValueError):
            # If parsing fails, treat as regular strings
            processed_values = values
    else:
        processed_values = values

    # Count unique values
    unique_count = processed_values.nunique(dropna=False)
    total_count = len(processed_values)

    # Calculate normalized variation (0 to 1)
    if total_count > 1:
        variation = (unique_count - 1) / (total_count - 1)
    else:
        variation = 0.0

    return variation


def calculate_weighted_variation(group: pd.DataFrame, fields_weights: Dict[str, float],
                                 handle_nulls: str = 'as_value') -> float:
    """
    Calculate weighted variation across multiple fields within a group.

    Parameters:
    -----------
    group : pd.DataFrame
        Group of records (typically with same group_id)
    fields_weights : Dict[str, float]
        Dictionary with fields and their weights
    handle_nulls : str
        How to handle nulls: 'as_value' or 'exclude'

    Returns:
    --------
    float
        Weighted variation value from 0 to 1
    """
    if len(group) <= 1 or not fields_weights:
        return 0.0

    total_weight = sum(fields_weights.values())
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0

    # Calculate variation for each field and apply weight
    for field, weight in fields_weights.items():
        if field in group.columns:
            field_variation = calculate_field_variation(group, field, handle_nulls)
            weighted_sum += field_variation * weight

    # Normalize by total weight
    return weighted_sum / total_weight


def calculate_change_frequency(group: pd.DataFrame, fields: List[str],
                               handle_nulls: str = 'as_value') -> Dict[str, float]:
    """
    Calculate how often fields change within a group.

    Parameters:
    -----------
    group : pd.DataFrame
        Group of records (typically with same group_id)
    fields : List[str]
        List of fields to analyze
    handle_nulls : str
        How to handle nulls: 'as_value' or 'exclude'

    Returns:
    --------
    Dict[str, float]
        Dictionary mapping field names to change frequencies (0 to 1)
    """
    if len(group) <= 1:
        return {field: 0.0 for field in fields if field in group.columns}

    change_freqs = {}

    for field in fields:
        if field not in group.columns:
            continue

        values = group[field]

        if handle_nulls == 'exclude':
            values = values.dropna()
            if len(values) <= 1:
                change_freqs[field] = 0.0
                continue

        # Get total potential transitions
        transitions = len(values) - 1

        # Count actual changes
        changes = sum(values.iloc[i] != values.iloc[i + 1] for i in range(len(values) - 1))

        # Calculate change frequency
        change_freqs[field] = changes / transitions if transitions > 0 else 0.0

    return change_freqs


def create_identifier_hash(row: pd.Series, fields: List[str]) -> Optional[str]:
    """
    Create a hash-based identifier from specified fields.

    Parameters:
    -----------
    row : pd.Series
        Row with data
    fields : List[str]
        Fields to use for hash creation

    Returns:
    --------
    Optional[str]
        Hash-based identifier or None if required fields are missing
    """
    # Check if all required fields have values
    if any(pd.isna(row[field]) for field in fields if field in row):
        return None

    # Concatenate values and create hash
    values = ''.join(str(row[field]) for field in fields if field in row)
    return hashlib.md5(values.encode('utf-8')).hexdigest()


def analyze_cross_groups(df: pd.DataFrame, primary_group_field: str,
                         secondary_identifier_fields: List[str],
                         min_group_size: int = 2,
                         handle_nulls: str = 'exclude',
                         threshold: float = 0.8) -> Dict[str, Any]:
    """
    Analyze relationships between different group identifiers.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    primary_group_field : str
        Primary field to group by (e.g., 'resume_id')
    secondary_identifier_fields : List[str]
        Fields that form secondary identifiers (e.g., ['first_name', 'last_name', 'birth_day'])
    min_group_size : int
        Minimum size of groups to analyze
    handle_nulls : str
        How to handle nulls
    threshold : float
        Minimum confidence threshold for secondary identification

    Returns:
    --------
    Dict[str, Any]
        Analysis results containing relationships between identifiers
    """
    # Check for required fields
    if primary_group_field not in df.columns:
        return {'error': f"Primary group field {primary_group_field} not found in DataFrame"}

    for field in secondary_identifier_fields:
        if field not in df.columns:
            return {'error': f"Secondary identifier field {field} not found in DataFrame"}

    # Create secondary identifier hash
    df = df.copy()  # Avoid modifying the original DataFrame
    df['secondary_id_hash'] = df.apply(
        lambda row: create_identifier_hash(row, secondary_identifier_fields),
        axis=1
    )

    # Filter out rows with missing hash values if requested
    if handle_nulls == 'exclude':
        df_filtered = df.dropna(subset=['secondary_id_hash'])
    else:
        df_filtered = df.copy()

    # Group by primary field
    primary_groups = df_filtered.groupby(primary_group_field)

    # Filter by minimum group size
    primary_groups_filtered = [group for name, group in primary_groups if len(group) >= min_group_size]

    # Analyze cross-group relationships
    cross_group_data = []

    # Count unique secondary IDs
    secondary_id_count = df_filtered['secondary_id_hash'].nunique()

    # Group by secondary ID
    secondary_groups = df_filtered.groupby('secondary_id_hash')

    # Identify cross-group relationships
    for sec_id, sec_group in secondary_groups:
        if sec_id is None or len(sec_group) < min_group_size:
            continue

        primary_ids = sec_group[primary_group_field].unique()

        if len(primary_ids) > 1:
            # This secondary ID spans multiple primary groups
            cross_group_data.append({
                'secondary_id': sec_id,
                'count': len(sec_group),
                'primary_ids': primary_ids.tolist(),
                'primary_ids_count': len(primary_ids),
                'fields_used': secondary_identifier_fields,
                'confidence': 1.0 if not pd.isna(sec_id) else 0.0
            })

    # Prepare result
    result = {
        'primary_group_field': primary_group_field,
        'secondary_identifier_fields': secondary_identifier_fields,
        'total_records': len(df),
        'valid_secondary_ids': secondary_id_count,
        'cross_group_count': len(cross_group_data),
        'cross_group_details': cross_group_data,
        'cross_group_percentage': round((len(cross_group_data) / secondary_id_count) * 100,
                                        2) if secondary_id_count > 0 else 0
    }

    return result


def extract_group_metadata(group: pd.DataFrame, metadata_fields: List[str]) -> Dict[str, Any]:
    """
    Extract metadata about a group based on specified fields.

    Parameters:
    -----------
    group : pd.DataFrame
        Group of records (typically with same group_id)
    metadata_fields : List[str]
        Fields to extract metadata from

    Returns:
    --------
    Dict[str, Any]
        Dictionary with group metadata
    """
    metadata = {
        'group_size': len(group)
    }

    for field in metadata_fields:
        if field in group.columns:
            metadata[f"{field}_unique_count"] = group[field].nunique(dropna=False)
            metadata[f"{field}_most_common"] = group[field].value_counts(dropna=False).index[0] if len(
                group) > 0 else None
            metadata[f"{field}_null_count"] = group[field].isna().sum()

    return metadata


def analyze_collapsibility(variation_results: List[Dict[str, Any]],
                           threshold: float = 0.2) -> Dict[str, Any]:
    """
    Analyze potential for collapsing records within groups.

    Parameters:
    -----------
    variation_results : List[Dict[str, Any]]
        List of group variation results
    threshold : float
        Maximum variation threshold for collapsibility

    Returns:
    --------
    Dict[str, Any]
        Analysis of collapsibility potential
    """
    collapsible_groups = []
    collapsible_records = 0
    total_records = 0

    for result in variation_results:
        group_id = result.get('group_id')
        variation = result.get('variation', 1.0)
        size = result.get('size', 0)

        total_records += size

        if variation <= threshold:
            collapsible_groups.append({
                'group_id': group_id,
                'variation': variation,
                'size': size,
                'field_variations': result.get('field_variations', {})
            })
            collapsible_records += size

    return {
        'threshold': threshold,
        'collapsible_groups_count': len(collapsible_groups),
        'total_groups_count': len(variation_results),
        'collapsible_groups_percentage': round((len(collapsible_groups) / len(variation_results)) * 100,
                                               2) if variation_results else 0,
        'collapsible_records_count': collapsible_records,
        'total_records_count': total_records,
        'collapsible_records_percentage': round((collapsible_records / total_records) * 100,
                                                2) if total_records > 0 else 0,
        'collapsible_groups': collapsible_groups
    }


def identify_change_patterns(variation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identify common patterns in how fields change within groups.

    Parameters:
    -----------
    variation_results : List[Dict[str, Any]]
        List of group variation results

    Returns:
    --------
    Dict[str, Any]
        Analysis of change patterns
    """
    # Extract field variations from all groups
    all_field_variations = {}

    for result in variation_results:
        field_variations = result.get('field_variations', {})

        for field, variation in field_variations.items():
            if field not in all_field_variations:
                all_field_variations[field] = []

            all_field_variations[field].append(variation)

    # Calculate statistics for each field
    field_stats = {}

    for field, variations in all_field_variations.items():
        if not variations:
            continue

        variations_array = np.array(variations)

        field_stats[field] = {
            'min': float(np.min(variations_array)),
            'max': float(np.max(variations_array)),
            'mean': float(np.mean(variations_array)),
            'median': float(np.median(variations_array)),
            'std': float(np.std(variations_array)),
            'low_variation_percentage': float(np.mean(variations_array <= 0.2) * 100),
            'high_variation_percentage': float(np.mean(variations_array >= 0.8) * 100)
        }

    # Identify correlated field changes
    correlation_matrix = {}

    for field1 in all_field_variations:
        for field2 in all_field_variations:
            if field1 >= field2:  # Only calculate upper triangle
                continue

            if len(all_field_variations[field1]) != len(all_field_variations[field2]):
                continue

            # Calculate correlation
            corr = np.corrcoef(all_field_variations[field1], all_field_variations[field2])[0, 1]

            # Store significant correlations
            if abs(corr) > 0.5:
                key = f"{field1}_{field2}"
                correlation_matrix[key] = float(corr)

    return {
        'field_statistics': field_stats,
        'field_correlations': correlation_matrix
    }


def calculate_variation_distribution(variations: List[float], bins: int = 10) -> Dict[str, int]:
    """
    Calculate distribution of variation values.

    Parameters:
    -----------
    variations : List[float]
        List of variation values
    bins : int
        Number of bins for distribution

    Returns:
    --------
    Dict[str, int]
        Distribution of variation values
    """
    if not variations:
        return {}

    # Create bins
    bin_edges = np.linspace(0, 1, bins + 1)
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(bins)]

    # Calculate histogram
    hist, _ = np.histogram(variations, bins=bin_edges)

    # Convert to dictionary
    distribution = {labels[i]: int(hist[i]) for i in range(bins)}

    return distribution


def analyze_group_in_chunks(df: pd.DataFrame,
                            group_field: str,
                            fields_weights: Dict[str, float],
                            chunk_size: int = 50000,
                            min_group_size: int = 2,
                            handle_nulls: str = 'as_value') -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Analyze groups in chunks for large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    group_field : str
        Field to group by
    fields_weights : Dict[str, float]
        Dictionary of fields and their weights
    chunk_size : int
        Size of chunks to process
    min_group_size : int
        Minimum size of groups to analyze
    handle_nulls : str
        How to handle nulls

    Returns:
    --------
    Tuple[List[Dict[str, Any]], Dict[str, Any]]
        Tuple containing:
        1. List of group variation results
        2. Overall statistics
    """
    # Process in chunks
    total_rows = len(df)
    chunk_count = (total_rows + chunk_size - 1) // chunk_size

    logger.info(f"Processing {total_rows} rows in {chunk_count} chunks, chunk size: {chunk_size}")

    # Store results for all chunks
    all_results = []

    # Track groups across chunks
    group_sizes = {}
    group_data = {}

    # Process each chunk
    for i in range(chunk_count):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)

        chunk = df.iloc[start_idx:end_idx]

        # Group the chunk
        for name, group in chunk.groupby(group_field):
            if name not in group_sizes:
                group_sizes[name] = 0
                group_data[name] = []

            group_sizes[name] += len(group)
            group_data[name].append(group)

        # Save memory
        del chunk

    # Now process each group that meets the minimum size
    for group_id, size in group_sizes.items():
        if size < min_group_size:
            continue

        # Combine group data from chunks
        combined_group = pd.concat(group_data[group_id])

        # Calculate variation
        variation = calculate_weighted_variation(
            combined_group,
            fields_weights,
            handle_nulls
        )

        # Calculate field variations
        field_variations = {
            field: calculate_field_variation(combined_group, field, handle_nulls)
            for field in fields_weights if field in combined_group.columns
        }

        # Add result
        all_results.append({
            'group_id': group_id,
            'size': size,
            'variation': variation,
            'field_variations': field_variations
        })

        # Clean up to save memory
        del combined_group

    # Clean up group data to save memory
    del group_data

    # Calculate overall statistics
    total_groups = len(group_sizes)
    analyzed_groups = len(all_results)

    overall_stats = {
        'total_groups': total_groups,
        'analyzed_groups': analyzed_groups,
        'min_group_size': min_group_size
    }

    if analyzed_groups > 0:
        variations = [r['variation'] for r in all_results]
        overall_stats['overall_stats'] = {
            'min_variation': min(variations),
            'max_variation': max(variations),
            'mean_variation': sum(variations) / len(variations),
            'median_variation': sorted(variations)[len(variations) // 2]
        }

    return all_results, overall_stats


def estimate_resources(df: pd.DataFrame, group_field: str, fields_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Estimate resources needed for group analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    group_field : str
        Field to group by
    fields_weights : Dict[str, float]
        Dictionary of fields and their weights

    Returns:
    --------
    Dict[str, Any]
        Estimated resource requirements
    """
    # Basic resource estimation based on DataFrame size
    row_count = len(df)

    # Count groups and estimate average group size
    if group_field in df.columns:
        group_count = df[group_field].nunique()
        avg_group_size = row_count / group_count if group_count > 0 else 0

        # Memory estimation (rough approximation)
        base_memory_mb = 50

        # Memory for analysis scales with number of groups and fields
        field_count = sum(1 for field in fields_weights if field in df.columns)
        analysis_memory_mb = group_count * field_count * 0.001  # Very rough estimate

        # Memory for temporary storage during processing
        temp_memory_mb = row_count * field_count * 0.0001  # Very rough estimate

        # Total estimated memory
        estimated_memory_mb = base_memory_mb + analysis_memory_mb + temp_memory_mb

        # Estimated time (very rough approximation)
        if row_count < 10000:
            estimated_time_seconds = 5
        elif row_count < 100000:
            estimated_time_seconds = 30
        elif row_count < 1000000:
            estimated_time_seconds = 120
        else:
            estimated_time_seconds = 600

        # Adjust for group complexity
        if avg_group_size > 10:
            estimated_time_seconds *= 1.5

        # Recommend chunk size based on row count
        recommended_chunk_size = min(100000, max(10000, row_count // 10))

        return {
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_time_seconds': estimated_time_seconds,
            'group_count': group_count,
            'avg_group_size': avg_group_size,
            'field_count': field_count,
            'recommended_chunk_size': recommended_chunk_size,
            'use_chunks_recommended': row_count > 100000 or group_count > 10000
        }
    else:
        # Field not found
        return {
            'estimated_memory_mb': 10,
            'estimated_time_seconds': 1,
            'error': f"Group field {group_field} not found in DataFrame"
        }

def calculate_field_variance(field_series: pd.Series,
                            text_length_threshold: int,
                            hash_algorithm: str,
                            use_minhash: bool,
                            minhash_similarity_threshold: float,
                            minhash_cache: Dict[str, Any],
                            logger: logging.Logger) -> Tuple[float, float]:
    """
    Calculate the variance and duplication ratio for a single field.

    Parameters:
    -----------
    field_series : pd.Series
        Series containing values for a single field
    text_length_threshold : int
        Threshold for text length processing
    hash_algorithm : str
        Hash algorithm to use
    use_minhash : bool
        Whether to use MinHash algorithm
    minhash_similarity_threshold : float
        Similarity threshold for MinHash
    minhash_cache : Dict[str, Any]
        Cache for MinHash signatures
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    Tuple[float, float]
        (variance, duplication_ratio)
    """
    total_records = len(field_series)

    if total_records <= 1:
        return 0.0, 0.0

    # Get unique values
    unique_values, value_counts = get_unique_values_improved(
        field_series,
        text_length_threshold=text_length_threshold,
        hash_algorithm=hash_algorithm,
        use_minhash=use_minhash,
        minhash_similarity_threshold=minhash_similarity_threshold,
        minhash_cache=minhash_cache,
        logger=logger
    )
    unique_count = len(unique_values)

    # Calculate variance: (unique_count - 1) / (total_records - 1)
    # This gives 0 when all values are identical, 1 when all values are unique
    # Low variance means many duplicates (high similarity), high variance means few duplicates
    variance = (unique_count - 1) / (total_records - 1) if total_records > 1 else 0

    # Calculate duplication ratio: total_records / unique_count
    duplication_ratio = total_records / unique_count if unique_count > 0 else 0

    return float(variance), float(duplication_ratio)

def get_unique_values_improved(field_series: pd.Series,
                                text_length_threshold: int,
                                hash_algorithm: str,
                                use_minhash: bool,
                                minhash_similarity_threshold: float,
                                minhash_cache: Dict[str, Any],
                                logger: logging.Logger) -> Tuple[Set[Any], Dict[Any, int]]:
    """
    Get unique values from a field series, handling text and NULL values.
    Improved version with better categorical data handling.

    Parameters:
    -----------
    field_series : pd.Series
        Series containing values for a single field
    text_length_threshold : int
        Threshold for text length processing
    hash_algorithm : str
        Hash algorithm to use
    use_minhash : bool
        Whether to use MinHash algorithm
    minhash_similarity_threshold : float
        Similarity threshold for MinHash
    minhash_cache : Dict[str, Any]
        Cache for MinHash signatures
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    Tuple[Set[Any], Dict[Any, int]]
        Set of unique values and dictionary with value counts
    """
    unique_keys = set()
    counts = {}

    # Initialize MinHash-related variables
    _minhash_imported = False
    _compute_minhash = None

    # Process each value individually to avoid categorical data issues
    for raw_val in field_series:
        # Handle NULL values
        if pd.isna(raw_val):
            key = None  # Use None as the key for NaN values
        # Handle string values with potential hashing
        elif isinstance(raw_val, str):
            text = raw_val.strip()
            if len(text) > text_length_threshold:
                if use_minhash:
                    # Lazy import MinHash only if needed
                    if not _minhash_imported:
                        try:
                            # Dynamically import minhash module when needed
                            from pamola_core.utils.nlp.minhash import compute_minhash
                            _compute_minhash = compute_minhash
                            _minhash_imported = True
                        except ImportError:
                            logger.warning("MinHash library not available, falling back to MD5")
                            use_minhash = False
                            _minhash_imported = False
                            _compute_minhash = lambda _: []

                    if use_minhash and _minhash_imported:
                        # Use cached MinHash signatures when possible
                        if text not in minhash_cache:
                            minhash_cache[text] = _compute_minhash(text)

                        # Use the first 8 elements of the signature as the key
                        signature = minhash_cache[text]
                        signature_str = str(signature[:8])  # Convert to string for consistent keys
                        key = signature_str
                    else:
                        # Fallback to MD5 if MinHash is not available
                        key = hashlib.md5(text.encode('utf-8')).hexdigest()
                else:
                    # Use MD5 for exact matching of long texts
                    key = hashlib.md5(text.encode('utf-8')).hexdigest()
            else:
                # Use the text directly for short strings
                key = text
        else:
            # For other types (numbers, dates, etc.), use direct comparison
            key = raw_val

        # Count occurrences
        unique_keys.add(key)
        counts[key] = counts.get(key, 0) + 1

    # For MinHash, we could cluster similar signatures here if needed
    if use_minhash and _minhash_imported:
        minhash_keys = [k for k in counts.keys() if isinstance(k, str) and k.startswith('[')]

        if len(minhash_keys) > 1:
            # Group similar signatures
            clusters = cluster_minhash_signatures_from_keys(
                minhash_keys, minhash_similarity_threshold
            )

            # Update counts based on clusters
            for cluster in clusters:
                if len(cluster) > 1:
                    # Use the first signature as the representative
                    primary_key = cluster[0]
                    total_count = counts.get(primary_key, 0)

                    # Combine counts from all similar signatures
                    for other_key in cluster[1:]:
                        if other_key in counts:
                            total_count += counts.pop(other_key)
                            unique_keys.remove(other_key)

                    # Update count for the primary key
                    counts[primary_key] = total_count

    return unique_keys, counts

def cluster_minhash_signatures_from_keys(signature_keys: List[str], 
                                        minhash_similarity_threshold: float) -> List[List[str]]:
    """
    Cluster similar MinHash signature keys.

    Parameters:
    -----------
    signature_keys : List[str]
        List of MinHash signature keys to cluster
    minhash_similarity_threshold : float
        Similarity threshold for clustering

    Returns:
    --------
    List[List[str]]
        List of clusters, where each cluster is a list of similar signature keys
    """
    if len(signature_keys) <= 1:
        return [signature_keys]

    # Parse signatures from string representation
    parsed_signatures = []
    for key in signature_keys:
        try:
            # Extract numbers from string representation like '[1, 2, 3, 4]'
            nums = key.strip('[]').split(',')
            sig = [int(n.strip()) for n in nums if n.strip()]
            parsed_signatures.append((key, sig))
        except (ValueError, AttributeError):
            # Skip keys that can't be parsed as signatures
            continue

    # Create clusters
    clusters = []
    processed = set()

    # For each signature, find similar signatures and create a cluster
    for i, (key1, sig1) in enumerate(parsed_signatures):
        if key1 in processed:
            continue

        # Start a new cluster with this key
        cluster = [key1]
        processed.add(key1)

        # Compare with all other signatures
        for key2, sig2 in parsed_signatures[i + 1:]:
            if key2 in processed:
                continue

            # Calculate Jaccard similarity
            similarity = calculate_simple_jaccard(sig1, sig2)

            # If similar enough, add to the cluster
            if similarity >= minhash_similarity_threshold:
                cluster.append(key2)
                processed.add(key2)

        # Add the cluster to the results
        clusters.append(cluster)

    return clusters

def calculate_simple_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """
    Calculate a simple Jaccard similarity between two signatures.

    Parameters:
    -----------
    sig1 : List[int]
        First signature
    sig2 : List[int]
        Second signature

    Returns:
    --------
    float
        Jaccard similarity (0-1)
    """
    # Ensure both signatures have values
    if not sig1 or not sig2:
        return 0.0

    # Convert to sets for intersection and union
    set1 = set(sig1)
    set2 = set(sig2)

    # Calculate Jaccard similarity: |intersection| / |union|
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0

def calculate_weighted_variance(field_variances: Dict[str, float], 
                                fields_config: Dict[str, int]) -> float:
    """
    Calculate weighted variance across all fields.

    Parameters:
    -----------
    field_variances : Dict[str, float]
        Dictionary mapping field names to their variance values
    fields_config : Dict[str, int]
        Configuration mapping field names to their weights

    Returns:
    --------
    float
        Weighted variance across all fields
    """
    weighted_sum = 0
    total_weight = 0

    for field, variance in field_variances.items():
        weight = fields_config.get(field, 1)
        weighted_sum += variance * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0

def should_aggregate(weighted_variance: float, 
                    group_size: int,
                    variance_threshold: float,
                    large_group_threshold: int,
                    large_group_variance_threshold: float) -> bool:
    """
    Determine if a group should be aggregated based on variance and size.

    Parameters:
    -----------
    weighted_variance : float
        Weighted variance of the group
    group_size : int
        Number of records in the group
    variance_threshold : float
        Variance threshold for normal groups
    large_group_threshold : int
        Threshold for large group size
    large_group_variance_threshold : float
        Variance threshold for large groups

    Returns:
    --------
    bool
        True if the group should be aggregated, False otherwise
    """
    # Apply different thresholds based on group size
    if group_size > large_group_threshold:
        return weighted_variance <= large_group_variance_threshold
    else:
        return weighted_variance <= variance_threshold

def calculate_field_metrics(group_metrics: Dict[str, Dict[str, Any]], 
                            fields: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each field across all groups.

    Parameters:
    -----------
    group_metrics : Dict[str, Dict[str, Any]]
        Dictionary with metrics for each group
    fields : List[str]
        List of fields to analyze

    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary with metrics for each field
    """
    field_metrics = {}

    for field in fields:
        # Initialize metrics for this field
        field_metrics[field] = {
            "avg_variance": 0,
            "max_variance": 0,
            "avg_duplication_ratio": 0,
            "unique_values_total": 0
        }

        # Collect variances and duplication ratios for this field
        variances = []
        duplication_ratios = []
        unique_values_count = 0

        for group_data in group_metrics.values():
            if field in group_data["field_variances"]:
                variances.append(group_data["field_variances"][field])

            if field in group_data["duplication_ratios"]:
                duplication_ratios.append(group_data["duplication_ratios"][field])

                # Estimate unique values from duplication ratio: total_records / duplication_ratio
                if group_data["duplication_ratios"][field] > 0:
                    unique_values_count += group_data["total_records"] / group_data["duplication_ratios"][field]

        # Calculate average and max values
        field_metrics[field]["avg_variance"] = float(np.mean(variances)) if variances else 0
        field_metrics[field]["max_variance"] = float(np.max(variances)) if variances else 0
        field_metrics[field]["avg_duplication_ratio"] = float(
            np.mean(duplication_ratios)) if duplication_ratios else 0
        field_metrics[field]["unique_values_total"] = int(unique_values_count)

    return field_metrics

def analyze_group(group_df: pd.DataFrame,
                  fields: List[str],
                  fields_config: Dict[str, int],
                  text_length_threshold: int,
                  hash_algorithm: str,
                  use_minhash: bool,
                  minhash_similarity_threshold: float,
                  minhash_cache: Dict[str, Any],
                  large_group_threshold: int,
                  large_group_variance_threshold: float,
                  variance_threshold: float) -> Dict[str, Any]:
        """
        Analyze a single group of records.

        Parameters:
        -----------
        group_df : pd.DataFrame
            DataFrame containing records for a single group.
        fields : List[str]
            List of field names to analyze.
        fields_config : Dict[str, int]
            Dictionary mapping field names to their weights for weighted variance calculation.
        text_length_threshold : int
            Threshold for text length to determine when to use hashing or MinHash.
        hash_algorithm : str
            Hash algorithm to use for text fields (e.g., 'md5').
        use_minhash : bool
            Whether to use MinHash for long text fields.
        minhash_similarity_threshold : float
            Similarity threshold for clustering MinHash signatures.
        minhash_cache : Dict[str, Any]
            Cache for MinHash signatures to avoid recomputation.
        large_group_threshold : int
            Size above which a group is considered "large" for aggregation logic.
        large_group_variance_threshold : float
            Variance threshold for large groups.
        variance_threshold : float
            Variance threshold for normal groups.

        Returns:
        --------
        Dict[str, Any]
            Dictionary with group metrics
        """
        # Initialize metrics
        field_variances = {}
        duplication_ratios = {}

        # Calculate variance and duplication ratio for each field
        for field in fields:
            # Calculate variance for this field
            variance, duplications = calculate_field_variance(group_df[field],
                                                              text_length_threshold,
                                                              hash_algorithm,
                                                              use_minhash,
                                                              minhash_similarity_threshold,
                                                              minhash_cache,
                                                              logger)

            field_variances[field] = variance
            duplication_ratios[field] = duplications

        # Calculate weighted variance
        weighted_variance = calculate_weighted_variance(field_variances, fields_config)

        # Determine if this group should be aggregated
        should_aggregate = is_group_aggregatable(weighted_variance, len(group_df),
                                            large_group_threshold, large_group_variance_threshold,variance_threshold)

        # Get max field variance
        max_field_variance = max(field_variances.values()) if field_variances else 0

        return {
            "weighted_variance": weighted_variance,
            "max_field_variance": max_field_variance,
            "total_records": len(group_df),
            "field_variances": field_variances,
            "duplication_ratios": duplication_ratios,
            "should_aggregate": should_aggregate
        }

def is_group_aggregatable(weighted_variance: float,
                     group_size: int,
                     large_group_threshold: int,
                     large_group_variance_threshold: float,
                     variance_threshold: float) -> bool:
        """
        Determine if a group should be aggregated based on variance and size.

        Parameters:
        -----------
        weighted_variance : float
            Weighted variance of the group
        group_size : int
            Number of records in the group
        variance_threshold : float
            Variance threshold for normal groups
        large_group_threshold : int
            Threshold for large group size
        large_group_variance_threshold : float
            Variance threshold for large groups

        Returns:
        --------
        bool
            True if the group should be aggregated, False otherwise
        """
        # Apply different thresholds based on group size
        if group_size > large_group_threshold:
            return weighted_variance <= large_group_variance_threshold
        else:
            return weighted_variance <= variance_threshold
