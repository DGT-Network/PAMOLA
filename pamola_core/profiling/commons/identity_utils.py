"""
Identity analysis utilities for the HHR project.

This module provides utility functions for identity analysis, including
identifier distribution analysis, consistency checking, and cross-matching.
These functions are used by the identity analyzer module.
"""

import hashlib
import logging
from collections import Counter
from typing import Dict, List, Any, Optional

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


def calculate_hash(values: List[Any], algorithm: str = "md5") -> str:
    """
    Calculate a hash from a list of values.

    Parameters:
    -----------
    values : List[Any]
        Values to hash
    algorithm : str
        Hash algorithm to use

    Returns:
    --------
    str
        Calculated hash value
    """
    # Convert all values to strings and concatenate
    values_str = ''.join([str(v) if v is not None else '' for v in values])

    # Calculate hash using the specified algorithm
    if algorithm.lower() == "md5":
        return hashlib.md5(values_str.encode()).hexdigest().upper()
    elif algorithm.lower() == "sha1":
        return hashlib.sha1(values_str.encode()).hexdigest().upper()
    elif algorithm.lower() == "sha256":
        return hashlib.sha256(values_str.encode()).hexdigest().upper()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def compute_identifier_stats(df: pd.DataFrame,
                             id_field: str,
                             entity_field: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute basic statistics about an identifier field.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    id_field : str
        Identifier field to analyze
    entity_field : str, optional
        Entity identifier field for relation analysis

    Returns:
    --------
    Dict[str, Any]
        Basic statistics about the identifier
    """
    if id_field not in df.columns:
        return {
            'error': f"Field {id_field} not found in DataFrame",
            'total_records': len(df),
            'unique_identifiers': 0,
            'null_identifiers': 0,
            'coverage_percentage': 0
        }

    # Compute basic statistics
    total_records = len(df)
    null_identifiers = df[id_field].isnull().sum()
    unique_identifiers = df[id_field].nunique()
    coverage_percentage = 100 * (total_records - null_identifiers) / total_records if total_records > 0 else 0

    # Compute relationship statistics if entity_field is provided
    relationship_stats = {}
    if entity_field and entity_field in df.columns:
        # Count entities per identifier
        entities_per_id = df.groupby(id_field)[entity_field].nunique()

        # Count identifiers per entity
        ids_per_entity = df.groupby(entity_field)[id_field].nunique()

        # Calculate metrics for relationships
        if not entities_per_id.empty:
            # Count using DataFrame filtering rather than Series boolean operations
            # This approach avoids the boolean Series sum() issue
            one_to_one_count = len(entities_per_id[entities_per_id == 1])
            one_to_many_count = len(entities_per_id[entities_per_id > 1])

            relationship_stats = {
                'avg_entities_per_id': float(entities_per_id.mean()),
                'max_entities_per_id': int(entities_per_id.max()),
                'one_to_one_count': one_to_one_count,
                'one_to_many_count': one_to_many_count
            }
        else:
            relationship_stats = {
                'avg_entities_per_id': 0,
                'max_entities_per_id': 0,
                'one_to_one_count': 0,
                'one_to_many_count': 0
            }

        # Add metrics for the other direction of the relationship
        if not ids_per_entity.empty:
            relationship_stats.update({
                'avg_ids_per_entity': float(ids_per_entity.mean()),
                'max_ids_per_entity': int(ids_per_entity.max())
            })
        else:
            relationship_stats.update({
                'avg_ids_per_entity': 0,
                'max_ids_per_entity': 0
            })

    return {
        'total_records': total_records,
        'unique_identifiers': unique_identifiers,
        'null_identifiers': null_identifiers,
        'coverage_percentage': coverage_percentage,
        'uniqueness_ratio': unique_identifiers / total_records if total_records > 0 else 0,
        **relationship_stats
    }


def analyze_identifier_distribution(df: pd.DataFrame,
                                    id_field: str,
                                    entity_field: Optional[str] = None,
                                    top_n: int = 15) -> Dict[str, Any]:
    """
    Analyze the distribution of entities per identifier.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    id_field : str
        Identifier field to analyze (e.g., 'UID')
    entity_field : str, optional
        Entity identifier field (e.g., 'resume_id')
    top_n : int
        Number of top examples to include

    Returns:
    --------
    Dict[str, Any]
        Analysis results including distribution statistics
    """
    if id_field not in df.columns:
        return {
            'error': f"Field {id_field} not found in DataFrame",
            'total_records': len(df)
        }

    if entity_field and entity_field not in df.columns:
        entity_field = None
        logger.warning(f"Entity field {entity_field} not found in DataFrame. Using record counts instead.")

    # Analyze distribution based on entity field or just count occurrences
    if entity_field:
        # Count unique entities per ID
        counts = df.groupby(id_field)[entity_field].nunique()
    else:
        # Count occurrences of each ID
        counts = df[id_field].value_counts()

    # Calculate statistics
    total_ids = len(counts)
    total_records = counts.sum()
    max_count = counts.max() if not counts.empty else 0
    min_count = counts.min() if not counts.empty else 0
    avg_count = counts.mean() if not counts.empty else 0
    median_count = counts.median() if not counts.empty else 0

    # Create distribution of counts
    distribution = Counter(counts.values)
    distribution_data = {str(count): freq for count, freq in sorted(distribution.items())}

    # Get top examples
    top_examples = []

    # Simplified approach to get top values
    if not counts.empty:
        # Convert to dictionary, sort by value (count), and get top N
        counts_dict = {str(k): v for k, v in counts.items()}
        top_ids = sorted(counts_dict.keys(), key=lambda k: counts_dict[k], reverse=True)[:top_n]

        for id_val in top_ids:
            count_val = counts_dict[id_val]
            example_rows = df[df[id_field] == id_val]
            sample_row = example_rows.iloc[0].to_dict() if not example_rows.empty else {}

            # If entity_field exists, include list of entities
            entities = []
            if entity_field and entity_field in df.columns:
                entities = example_rows[entity_field].unique().tolist()

            # Create example record
            top_examples.append({
                'identifier': id_val,
                'count': int(count_val),
                'entities': entities,
                'sample': {k: v for k, v in sample_row.items() if k in [id_field, entity_field]
                           if k is not None}  # Filter out None keys
            })

    return {
        'total_identifiers': total_ids,
        'total_records': int(total_records),
        'max_count': int(max_count),
        'min_count': int(min_count),
        'avg_count': float(avg_count),
        'median_count': float(median_count),
        'distribution': distribution_data,
        'top_examples': top_examples
    }


def analyze_identifier_consistency(df: pd.DataFrame,
                                   id_field: str,
                                   reference_fields: List[str]) -> Dict[str, Any]:
    """
    Analyze consistency between an identifier and reference fields.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    id_field : str
        Identifier field to analyze
    reference_fields : List[str]
        Fields that define an entity's identity

    Returns:
    --------
    Dict[str, Any]
        Analysis results including consistency statistics
    """
    if id_field not in df.columns:
        return {
            'error': f"Field {id_field} not found in DataFrame",
            'total_records': len(df)
        }

    # Validate reference fields
    valid_reference_fields = [field for field in reference_fields if field in df.columns]
    if not valid_reference_fields:
        return {
            'error': f"None of the reference fields {reference_fields} found in DataFrame",
            'total_records': len(df)
        }

    # Create a copy of the DataFrame with only needed columns
    columns_to_use = [id_field] + valid_reference_fields
    analysis_df = df[columns_to_use].copy()

    # Drop rows where id_field is null
    analysis_df = analysis_df.dropna(subset=[id_field])

    # Group by reference fields and check for consistency in id_field
    total_combinations = 0
    inconsistent_combinations = 0
    inconsistent_groups = []

    # If there are valid reference fields, group by them
    if valid_reference_fields:
        # Group by reference fields
        grouped = analysis_df.groupby(valid_reference_fields)

        # Check each group for consistent id_field
        for group_values, group_df in grouped:
            total_combinations += 1
            unique_ids = group_df[id_field].unique()

            # If more than one unique ID, the combination is inconsistent
            if len(unique_ids) > 1:
                inconsistent_combinations += 1

                # Create a sample for reporting
                if isinstance(group_values, tuple):
                    group_dict = {field: value for field, value in zip(valid_reference_fields, group_values)}
                else:
                    group_dict = {valid_reference_fields[0]: group_values}

                # Add to inconsistent groups
                inconsistent_groups.append({
                    'reference_values': group_dict,
                    'id_values': unique_ids.tolist(),
                    'count': len(group_df)
                })

    # Calculate match percentage
    match_percentage = 100 * (
                total_combinations - inconsistent_combinations) / total_combinations if total_combinations > 0 else 0

    # Find top mismatch examples
    top_mismatches = sorted(inconsistent_groups, key=lambda x: x['count'], reverse=True)[:15]

    return {
        'total_records': len(analysis_df),
        'total_combinations': total_combinations,
        'consistent_combinations': total_combinations - inconsistent_combinations,
        'inconsistent_combinations': inconsistent_combinations,
        'match_percentage': match_percentage,
        'mismatch_count': inconsistent_combinations,
        'reference_fields_used': valid_reference_fields,
        'mismatch_examples': top_mismatches
    }


def find_cross_matches(df: pd.DataFrame,
                       id_field: str,
                       reference_fields: List[str],
                       min_similarity: float = 0.8,
                       fuzzy_matching: bool = False) -> Dict[str, Any]:
    """
    Find cases where reference fields match but identifiers differ.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    id_field : str
        Identifier field to analyze
    reference_fields : List[str]
        Fields that define an entity's identity
    min_similarity : float
        Minimum similarity for fuzzy matching
    fuzzy_matching : bool
        Whether to use fuzzy matching

    Returns:
    --------
    Dict[str, Any]
        Cross-matching analysis results
    """
    if id_field not in df.columns:
        return {
            'error': f"Field {id_field} not found in DataFrame",
            'total_records': len(df)
        }

    # Validate reference fields
    valid_reference_fields = [field for field in reference_fields if field in df.columns]
    if not valid_reference_fields:
        return {
            'error': f"None of the reference fields {reference_fields} found in DataFrame",
            'total_records': len(df)
        }

    # Create a copy of the DataFrame with only needed columns
    columns_to_use = [id_field] + valid_reference_fields
    analysis_df = df[columns_to_use].copy()

    # Drop rows where id_field or any reference field is null
    analysis_df = analysis_df.dropna(subset=[id_field] + valid_reference_fields)

    # Find cases where reference fields match but id_field differs
    cross_matches = []
    total_cross_matches = 0

    # Group by the combination of reference fields
    grouped = analysis_df.groupby(valid_reference_fields)

    # Check each group for different id_field values
    for group_values, group_df in grouped:
        unique_ids = group_df[id_field].unique()

        # If more than one unique ID, add to cross matches
        if len(unique_ids) > 1:
            total_cross_matches += 1

            # Create a sample for reporting
            if isinstance(group_values, tuple):
                group_dict = {field: value for field, value in zip(valid_reference_fields, group_values)}
            else:
                group_dict = {valid_reference_fields[0]: group_values}

            # Add to cross matches
            cross_matches.append({
                'reference_values': group_dict,
                'id_values': unique_ids.tolist(),
                'count': len(group_df)
            })

    # Get top cross matches by count
    top_cross_matches = sorted(cross_matches, key=lambda x: x['count'], reverse=True)[:15]

    return {
        'total_records': len(analysis_df),
        'total_cross_matches': total_cross_matches,
        'reference_fields_used': valid_reference_fields,
        'cross_match_examples': top_cross_matches
    }