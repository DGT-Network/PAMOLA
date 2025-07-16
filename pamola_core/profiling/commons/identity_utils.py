"""
Identity analysis utilities for the project.

This module provides utility functions for identity analysis, including
identifier distribution analysis, consistency checking, and cross-matching.
These functions are used by the identity analyzer module.
"""

from rapidfuzz import fuzz
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from pamola_core.utils.visualization import create_bar_plot, plot_value_distribution

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
    values_str = "".join([str(v) if v is not None else "" for v in values])

    # Calculate hash using the specified algorithm
    if algorithm.lower() == "md5":
        return hashlib.md5(values_str.encode()).hexdigest().upper()
    elif algorithm.lower() == "sha1":
        return hashlib.sha1(values_str.encode()).hexdigest().upper()
    elif algorithm.lower() == "sha256":
        return hashlib.sha256(values_str.encode()).hexdigest().upper()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def compute_identifier_stats(
    df: pd.DataFrame, id_field: str, entity_field: Optional[str] = None
) -> Dict[str, Any]:
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
            "error": f"Field {id_field} not found in DataFrame",
            "total_records": len(df),
            "unique_identifiers": 0,
            "null_identifiers": 0,
            "coverage_percentage": 0,
        }

    # Compute basic statistics
    total_records = len(df)
    null_identifiers = df[id_field].isnull().sum()
    unique_identifiers = df[id_field].nunique()
    coverage_percentage = (
        100 * (total_records - null_identifiers) / total_records
        if total_records > 0
        else 0
    )

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
                "avg_entities_per_id": float(entities_per_id.mean()),
                "max_entities_per_id": int(entities_per_id.max()),
                "one_to_one_count": one_to_one_count,
                "one_to_many_count": one_to_many_count,
            }
        else:
            relationship_stats = {
                "avg_entities_per_id": 0,
                "max_entities_per_id": 0,
                "one_to_one_count": 0,
                "one_to_many_count": 0,
            }

        # Add metrics for the other direction of the relationship
        if not ids_per_entity.empty:
            relationship_stats.update(
                {
                    "avg_ids_per_entity": float(ids_per_entity.mean()),
                    "max_ids_per_entity": int(ids_per_entity.max()),
                }
            )
        else:
            relationship_stats.update(
                {"avg_ids_per_entity": 0, "max_ids_per_entity": 0}
            )

    return {
        "total_records": total_records,
        "unique_identifiers": unique_identifiers,
        "null_identifiers": null_identifiers,
        "coverage_percentage": coverage_percentage,
        "uniqueness_ratio": (
            unique_identifiers / total_records if total_records > 0 else 0
        ),
        **relationship_stats,
    }


def analyze_identifier_distribution(
    df: pd.DataFrame, id_field: str, entity_field: Optional[str] = None, top_n: int = 15
) -> Dict[str, Any]:
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
            "error": f"Field {id_field} not found in DataFrame",
            "total_records": len(df),
        }

    if entity_field and entity_field not in df.columns:
        entity_field = None
        logger.warning(
            f"Entity field {entity_field} not found in DataFrame. Using record counts instead."
        )

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
    distribution_data = {
        str(count): freq for count, freq in sorted(distribution.items())
    }

    # Get top examples
    top_examples = []

    # Simplified approach to get top values
    if not counts.empty:
        # Convert to dictionary, sort by value (count), and get top N
        counts_dict = {str(k): v for k, v in counts.items()}
        top_ids = sorted(
            counts_dict.keys(), key=lambda k: counts_dict[k], reverse=True
        )[:top_n]

        for id_val in top_ids:
            count_val = counts_dict[id_val]
            example_rows = df[df[id_field] == id_val]
            sample_row = (
                example_rows.iloc[0].to_dict() if not example_rows.empty else {}
            )

            # If entity_field exists, include list of entities
            entities = []
            if entity_field and entity_field in df.columns:
                entities = example_rows[entity_field].unique().tolist()

            # Create example record
            top_examples.append(
                {
                    "identifier": id_val,
                    "count": int(count_val),
                    "entities": entities,
                    "sample": {
                        k: v
                        for k, v in sample_row.items()
                        if k in [id_field, entity_field]
                        if k is not None
                    },  # Filter out None keys
                }
            )

    return {
        "total_identifiers": total_ids,
        "total_records": int(total_records),
        "max_count": int(max_count),
        "min_count": int(min_count),
        "avg_count": float(avg_count),
        "median_count": float(median_count),
        "distribution": distribution_data,
        "top_examples": top_examples,
    }


def analyze_identifier_consistency(
    df: pd.DataFrame, id_field: str, reference_fields: List[str]
) -> Dict[str, Any]:
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
            "error": f"Field {id_field} not found in DataFrame",
            "total_records": len(df),
        }

    # Validate reference fields
    valid_reference_fields = [
        field for field in reference_fields if field in df.columns
    ]
    if not valid_reference_fields:
        return {
            "error": f"None of the reference fields {reference_fields} found in DataFrame",
            "total_records": len(df),
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
                    group_dict = {
                        field: value
                        for field, value in zip(valid_reference_fields, group_values)
                    }
                else:
                    group_dict = {valid_reference_fields[0]: group_values}

                # Add to inconsistent groups
                inconsistent_groups.append(
                    {
                        "reference_values": group_dict,
                        "id_values": unique_ids.tolist(),
                        "count": len(group_df),
                    }
                )

    # Calculate match percentage
    match_percentage = (
        100 * (total_combinations - inconsistent_combinations) / total_combinations
        if total_combinations > 0
        else 0
    )

    # Find top mismatch examples
    top_mismatches = sorted(
        inconsistent_groups, key=lambda x: x["count"], reverse=True
    )[:15]

    return {
        "total_records": len(analysis_df),
        "total_combinations": total_combinations,
        "consistent_combinations": total_combinations - inconsistent_combinations,
        "inconsistent_combinations": inconsistent_combinations,
        "match_percentage": match_percentage,
        "mismatch_count": inconsistent_combinations,
        "reference_fields_used": valid_reference_fields,
        "mismatch_examples": top_mismatches,
    }


def find_cross_matches(
    df: pd.DataFrame,
    id_field: str,
    reference_fields: List[str],
    min_similarity: float = 0.8,
    fuzzy_matching: bool = False,
) -> Dict[str, Any]:
    """
    Find cases where reference fields match but identifiers differ.

    If fuzzy_matching is True, use fuzzy matching based on min_similarity.
    Otherwise, use exact matching on reference fields.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data frame.
    id_field : str
        Identifier field name.
    reference_fields : List[str]
        List of fields defining an entity's identity.
    min_similarity : float, optional
        Minimum similarity threshold for fuzzy matching (default 0.8).
    fuzzy_matching : bool, optional
        Whether to use fuzzy matching (default False).

    Returns:
    --------
    Dict[str, Any]
        Cross-matching analysis result.
    """
    if fuzzy_matching:
        # Use fuzzy matching function
        return find_cross_matches_fuzzy(df, id_field, reference_fields, min_similarity)

    # Exact matching code
    if id_field not in df.columns:
        return {
            "error": f"Field {id_field} not found in DataFrame",
            "total_records": len(df),
        }

    valid_reference_fields = [
        field for field in reference_fields if field in df.columns
    ]
    if not valid_reference_fields:
        return {
            "error": f"None of the reference fields {reference_fields} found in DataFrame",
            "total_records": len(df),
        }

    columns_to_use = [id_field] + valid_reference_fields
    analysis_df = df[columns_to_use].copy()

    # Drop rows with nulls in relevant columns
    analysis_df = analysis_df.dropna(subset=[id_field] + valid_reference_fields)

    cross_matches = []
    total_cross_matches = 0

    # Group by the exact combination of reference fields
    grouped = analysis_df.groupby(valid_reference_fields)

    for group_values, group_df in grouped:
        unique_ids = group_df[id_field].unique()
        if len(unique_ids) > 1:
            total_cross_matches += 1

            if isinstance(group_values, tuple):
                group_dict = {
                    field: value
                    for field, value in zip(valid_reference_fields, group_values)
                }
            else:
                group_dict = {valid_reference_fields[0]: group_values}

            cross_matches.append(
                {
                    "reference_values": group_dict,
                    "id_values": unique_ids.tolist(),
                    "count": len(group_df),
                }
            )

    return {
        "total_records": len(analysis_df),
        "total_cross_matches": total_cross_matches,
        "reference_fields_used": valid_reference_fields,
        "cross_match_examples": sorted(
            cross_matches, key=lambda x: x["count"], reverse=True
        )[:15],
    }


def find_cross_matches_fuzzy(
    df: pd.DataFrame,
    id_field: str,
    reference_fields: List[str],
    min_similarity: float = 0.8,
) -> Dict[str, Any]:
    """
    Find groups of records where reference fields are similar (using fuzzy matching)
    but identifiers differ.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data frame.
    id_field : str
        Identifier field name.
    reference_fields : List[str]
        List of fields to compare for similarity.
    min_similarity : float
        Minimum similarity threshold between 0 and 1.

    Returns:
    --------
    Dict[str, Any]
        Cross-matching analysis result.
    """
    if id_field not in df.columns:
        return {"error": f"{id_field} not found", "total_records": len(df)}

    valid_fields = [f for f in reference_fields if f in df.columns]
    if not valid_fields:
        return {"error": f"No valid reference fields found", "total_records": len(df)}

    # Keep only relevant columns and drop nulls
    df_clean = df[[id_field] + valid_fields].dropna()

    matched_indices = set()
    groups = []
    rows = df_clean.to_dict(orient="records")

    for i in range(len(rows)):
        if i in matched_indices:
            continue

        base_row = rows[i]
        group = [i]
        matched_indices.add(i)

        for j in range(i + 1, len(rows)):
            if j in matched_indices:
                continue

            compare_row = rows[j]
            similarity = all(
                fuzz.ratio(str(base_row[f]), str(compare_row[f])) / 100
                >= min_similarity
                for f in valid_fields
            )

            if similarity:
                group.append(j)
                matched_indices.add(j)

        if len(set(df_clean.iloc[group][id_field])) > 1:
            groups.append(
                {
                    "reference_values": {
                        f: df_clean.iloc[group[0]][f] for f in valid_fields
                    },
                    "id_values": df_clean.iloc[group][id_field].unique().tolist(),
                    "count": len(group),
                }
            )

    return {
        "total_records": len(df_clean),
        "total_cross_matches": len(groups),
        "reference_fields_used": valid_fields,
        "cross_match_examples": groups[:15],
    }


def generate_identifier_statistics_vis(
    identifier_stats: Dict[str, Any],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate bar chart for identifier statistics using create_bar_plot.

    Parameters
    ----------
    identifier_stats : Dict[str, Any]
        The dictionary containing identifier statistics.
    field_label : str
        Label used in naming the output visualization file.
    operation_name : str
        Name of the operation for tracking or labeling purposes.
    task_dir : Path
        Directory path where the output pie chart will be saved.
    timestamp : str
        Timestamp string to help uniquely name the output file.
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
        - "identifier_stats_bar_chart": Path to the generated bar chart file.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating identifier statistics bar chart for operation '%s' (field_label='%s')",
        operation_name,
        field_label,
    )
    bar_data = [
        {"key": "Total Records", "value": identifier_stats["total_records"]},
        {"key": "Unique Identifiers", "value": identifier_stats["unique_identifiers"]},
        {"key": "Null Identifiers", "value": identifier_stats["null_identifiers"]},
    ]

    # Add 1-n relationship if any
    if (
        "one_to_one_count" in identifier_stats
        and "one_to_many_count" in identifier_stats
    ):
        bar_data.extend(
            [
                {
                    "key": "1-to-1 Relationships",
                    "value": identifier_stats["one_to_one_count"],
                },
                {
                    "key": "1-to-N Relationships",
                    "value": identifier_stats["one_to_many_count"],
                },
            ]
        )

    # Add avg and max ids per entity if any
    if (
        "avg_ids_per_entity" in identifier_stats
        and "max_ids_per_entity" in identifier_stats
    ):
        bar_data.extend(
            [
                {
                    "key": "Avg IDs per Entity",
                    "value": identifier_stats["avg_ids_per_entity"],
                },
                {
                    "key": "Max IDs per Entity",
                    "value": identifier_stats["max_ids_per_entity"],
                },
            ]
        )

    identifier_stats_path = (
        task_dir / f"{field_label}_{operation_name}_identifier_stats_{timestamp}.png"
    )

    bar_chart_result_path = create_bar_plot(
        data=bar_data,
        output_path=identifier_stats_path,
        title=f"Identifier Statistics for '{field_label}'",
        x_label="Metric",
        y_label="Value",
        orientation="v",
        sort_by="key",
        showlegend=False,
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs,
    )

    logger.debug("Bar chart saved to: %s", bar_chart_result_path)
    visualization_paths["identifier_stats_bar_chart"] = bar_chart_result_path
    return visualization_paths


def generate_consistency_analysis_vis(
    consistency_analysis: Dict[str, Any],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate bar chart for consistency analysis using create_bar_plot.

    Parameters
    ----------
    consistency_analysis : Dict[str, Any]
        The dictionary containing consistency analysis results.
    field_label : str
        Label used in naming the output visualization file.
    operation_name : str
        Name of the operation for tracking or labeling purposes.
    task_dir : Path
        Directory path where the output pie chart will be saved.
    timestamp : str
        Timestamp string to help uniquely name the output file.
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
        "Generating consistency analysis bar chart for operation '%s' (field_label='%s')",
        operation_name,
        field_label,
    )

    # Correct format for bar chart
    match_percentage = consistency_analysis.get("match_percentage", 0)
    consistency_data = [
        {"key": "Consistent", "value": match_percentage},
        {"key": "Inconsistent", "value": 100 - match_percentage},
    ]

    consistency_stats_path = (
        task_dir / f"{field_label}_{operation_name}_consistency_stats_{timestamp}.png"
    )

    consistency_stats_result = create_bar_plot(
        data=consistency_data,
        output_path=str(consistency_stats_path),
        title=f"{field_label} Consistency Analysis",
        x_label="Consistency",
        y_label="Percentage",
        orientation="h",  # Horizontal for proportion clarity
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs,
    )

    logger.debug("Bar chart saved to: %s", consistency_stats_result)
    visualization_paths["consistency_stats_bar_chart"] = consistency_stats_result
    return visualization_paths


def generate_field_distribution_vis(
    distribution_analysis: Dict[str, Any],
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
    distribution_analysis : Dict[str, Any]
        The dictionary containing distribution analysis results.
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

    # Check if distribution exists
    distribution = distribution_analysis.get("distribution", {})
    if not distribution:
        logger.warning(
            "No distribution data found for field '%s'. Skipping visualization.",
            field_label,
        )
        return visualization_paths

    # Prepare output path
    distribution_analysis_path = (
        task_dir
        / f"{field_label}_{operation_name}_distribution_analysis_{timestamp}.png"
    )

    # Create bar chart using helper
    distribution_stats_result = plot_value_distribution(
        data=distribution,
        output_path=str(distribution_analysis_path),
        title=f"Distribution of '{field_label}' Values",
        max_items=top_n,
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs,
    )

    logger.debug("Distribution bar chart saved to: %s", distribution_stats_result)
    visualization_paths["distribution_stats_bar_chart"] = distribution_stats_result
    return visualization_paths


def generate_cross_match_distribution_vis(
    cross_match_analysis: Dict[str, Any],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate bar chart for cross match analysis using create_bar_plot.

    Parameters
    ----------
    cross_match_analysis : Dict[str, Any]
        The dictionary containing cross match analysis results.
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
        "Generating cross match bar chart for operation '%s' (field_label='%s')",
        operation_name,
        field_label,
    )

    cross_match_examples = cross_match_analysis.get("cross_match_examples", [])

    if not cross_match_examples:
        logger.warning("No cross match examples found for visualization.")
        return visualization_paths

    # Tạo dict dữ liệu: {'label': count}
    cross_match_data = {}
    for example in cross_match_examples:
        ref_vals = example.get("reference_values", {})
        label = ", ".join(f"{k}={v}" for k, v in ref_vals.items())  # readable label
        cross_match_data[label] = example.get("count", 0)

    cross_match_path = (
        task_dir / f"{field_label}_{operation_name}_cross_match_bar_{timestamp}.png"
    )

    cross_match_chart_path = create_bar_plot(
        data=cross_match_data,
        output_path=str(cross_match_path),
        title=f"Top Cross Match Examples by '{field_label}'",
        orientation="v",
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs,
    )

    logger.debug("Cross match bar chart saved to: %s", cross_match_chart_path)
    visualization_paths["cross_match_bar_chart"] = cross_match_chart_path
    return visualization_paths
