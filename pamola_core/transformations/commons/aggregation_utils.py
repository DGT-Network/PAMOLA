"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Aggregate Records Visualization & Utility Functions

This module provides utility functions for:
    - Preparing and generating visualizations for aggregate/groupby operations
    - Creating bar charts and histograms for record counts, aggregation comparisons, and group size distributions
    - Building aggregation dictionaries and flattening MultiIndex columns
    - Supporting both standard and custom aggregation functions

All major steps and data preparation are logged at DEBUG level for traceability.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
from pamola_core.common.helpers.custom_aggregations_helper import (
    CUSTOM_AGG_FUNCTIONS,
    STANDARD_AGGREGATIONS,
)
from pamola_core.utils.visualization import create_bar_plot, create_histogram

logger = logging.getLogger(__name__)


def create_record_count_per_group_data(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Prepare data for bar chart showing record count per group.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated DataFrame (result of groupby + agg).
    group_by_fields : List[str]
        List of fields used for grouping.
    operation_name : str
        Name of the aggregate operation.
    output_path : Path
        Path where the visualization or related outputs may be saved.
    Returns
    -------
    Dict[str, Any]
        {
            "group_labels": List[str],
            "record_count_per_group": Dict[str, int],
            "chart_recommendation": str
        }
    """
    logger.debug(
        "Preparing record count per group data for operation: %s", operation_name
    )
    group_label = (
        agg_df[group_by_fields].astype(str).agg("_".join, axis=1)
        if len(group_by_fields) > 1
        else agg_df[group_by_fields[0]].astype(str)
    )
    record_count_per_group = (
        agg_df["count"] if "count" in agg_df.columns else pd.Series([1] * len(agg_df))
    )
    record_count_per_group.index = group_label

    logger.debug("Group labels: %s", group_label.tolist())
    logger.debug("Record count per group: %s", record_count_per_group.to_dict())

    return {
        "group_labels": group_label.tolist(),
        "record_count_per_group": record_count_per_group.to_dict(),
        "chart_recommendation": "Bar chart showing record count per group.",
    }


def generate_record_count_per_group_vis(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate bar chart visualization for record count per group.

    Parameters
    ----------
    agg_df : pd.DataFrame
        The DataFrame containing aggregated data.
    group_by_fields : List[str]
        The fields used for grouping.
    field_label : str
        Used in filenames and chart titles.
    operation_name : str
        Name of the aggregate operation.
    task_dir : Path
        Directory to save plots.
    timestamp : str
        For unique filenames.
    visualization_paths : Optional[Dict[str, Any]]
        Dict to collect plot paths.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating record count per group visualization for operation: %s",
        operation_name,
    )
    record_count_data = create_record_count_per_group_data(
        agg_df, group_by_fields, operation_name, task_dir
    )
    bar_path = (
        task_dir
        / f"{field_label}_{operation_name}_aggregate_record_count_per_group_{timestamp}.png"
    )
    vis_path = create_bar_plot(
        data=record_count_data["record_count_per_group"],
        output_path=bar_path,
        title="Record Count per Group",
        x_label="Group",
        y_label="Record Count",
        orientation="v",
        sort_by="value",
        max_items=30,
        **kwargs
    )
    logger.debug("Bar chart saved to: %s", vis_path)
    visualization_paths["record_count_per_group_bar_chart"] = vis_path
    return visualization_paths


def create_aggregation_comparison_data(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    agg_fields: List[str],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Prepare data for aggregation comparison bar charts across groups.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated DataFrame (result of groupby + agg).
    group_by_fields : List[str]
        List of fields used for grouping.
    agg_fields : List[str]
        List of fields that were aggregated.
    operation_name : str
        Name of the aggregate operation.
    output_path : Path
        Path where the visualization or related outputs may be saved.
    Returns
    -------
    Dict[str, Any]
        {
            "group_labels": List[str],
            "agg_comparison": Dict[str, List[float]],
            "chart_recommendation": str
        }
    """
    logger.debug(
        "Preparing aggregation comparison data for operation: %s", operation_name
    )
    group_label = (
        agg_df[group_by_fields].astype(str).agg("_".join, axis=1)
        if len(group_by_fields) > 1
        else agg_df[group_by_fields[0]].astype(str)
    )

    agg_comparison = {}
    for field in agg_fields:
        if field in agg_df.columns:
            agg_comparison[field] = agg_df[field].values.tolist()
        else:
            # Multi-agg: columns like ('field', 'mean'), ('field', 'sum')
            for col in agg_df.columns:
                if isinstance(col, tuple) and col[0] == field:
                    agg_comparison[f"{field}_{col[1]}"] = agg_df[col].values.tolist()

    logger.debug("Group labels: %s", group_label.tolist())
    logger.debug("Aggregation comparison data: %s", agg_comparison)

    return {
        "group_labels": group_label.tolist(),
        "agg_comparison": agg_comparison,
        "chart_recommendation": "Bar chart comparing aggregation values across groups (one per aggregation field).",
    }


def generate_aggregation_comparison_vis(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    agg_fields: List[str],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate bar chart visualizations for aggregation comparison across groups.

    Parameters
    ----------
    agg_df : pd.DataFrame
        The DataFrame containing aggregated data.
    group_by_fields : List[str]
        The fields used for grouping.
    agg_fields : List[str]
        The fields that were aggregated.
    field_label : str
        Used in filenames and chart titles.
    operation_name : str
        Name of the aggregate operation.
    task_dir : Path
        Directory to save plots.
    timestamp : str
        For unique filenames.
    visualization_paths : Optional[Dict[str, Any]]
        Dict to collect plot paths.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating aggregation comparison visualizations for operation: %s",
        operation_name,
    )
    agg_comparison_data = create_aggregation_comparison_data(
        agg_df, group_by_fields, agg_fields, operation_name, task_dir
    )

    for agg_field, values in agg_comparison_data["agg_comparison"].items():
        agg_path = (
            task_dir
            / f"{field_label}_{operation_name}_aggregate_{agg_field}_comparison_{timestamp}.png"
        )
        data = dict(zip(agg_comparison_data["group_labels"], values))
        vis_path = create_bar_plot(
            data=data,
            output_path=agg_path,
            title=f"{agg_field} Comparison Across Groups",
            x_label="Group",
            y_label=agg_field,
            orientation="v",
            sort_by="value",
            max_items=30,
            **kwargs
        )
        logger.debug(
            "Aggregation comparison bar chart for '%s' saved to: %s",
            agg_field,
            vis_path,
        )
        visualization_paths[f"agg_comparison_{agg_field}_bar_chart"] = vis_path

    return visualization_paths


def create_group_size_distribution_data(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Prepare data for histogram showing distribution of group sizes.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated DataFrame (result of groupby + agg).
    group_by_fields : List[str]
        List of fields used for grouping.
    operation_name : str
        Name of the aggregate operation.
    output_path : Path
        Path where the visualization or related outputs may be saved.
    Returns
    -------
    Dict[str, Any]
        {
            "group_size_distribution": Dict[int, int],
            "chart_recommendation": str
        }
    """
    logger.debug(
        "Preparing group size distribution data for operation: %s", operation_name
    )
    group_label = (
        agg_df[group_by_fields].astype(str).agg("_".join, axis=1)
        if len(group_by_fields) > 1
        else agg_df[group_by_fields[0]].astype(str)
    )
    record_count_per_group = (
        agg_df["count"] if "count" in agg_df.columns else pd.Series([1] * len(agg_df))
    )
    record_count_per_group.index = group_label
    group_sizes = record_count_per_group.value_counts().sort_index()

    logger.debug("Group size distribution: %s", group_sizes.to_dict())

    return {
        "group_size_distribution": group_sizes.to_dict(),
        "chart_recommendation": "Histogram showing distribution of group sizes.",
    }


def generate_group_size_distribution_vis(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate histogram visualization for distribution of group sizes.

    Parameters
    ----------
    agg_df : pd.DataFrame
        The DataFrame containing aggregated data.
    group_by_fields : List[str]
        The fields used for grouping.
    field_label : str
        Used in filenames and chart titles.
    operation_name : str
        Name of the aggregate operation.
    task_dir : Path
        Directory to save plots.
    timestamp : str
        For unique filenames.
    visualization_paths : Optional[Dict[str, Any]]
        Dict to collect plot paths.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating group size distribution histogram for operation: %s", operation_name
    )
    group_size_data = create_group_size_distribution_data(
        agg_df, group_by_fields, operation_name, task_dir
    )

    hist_path = (
        task_dir
        / f"{field_label}_{operation_name}_aggregate_group_size_distribution_{timestamp}.png"
    )
    vis_path = create_histogram(
        data=list(group_size_data["group_size_distribution"].keys()),
        output_path=hist_path,
        title="Distribution of Group Sizes",
        x_label="Group Size",
        y_label="Frequency",
        bins=20,
        kde=False,
        **kwargs
    )
    logger.debug("Group size distribution histogram saved to: %s", vis_path)
    visualization_paths["group_size_distribution_histogram"] = vis_path
    return visualization_paths


def build_aggregation_dict(
    aggregations: Optional[Dict[str, List[str]]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
) -> Dict[str, List[Callable]]:
    """
    Build aggregation dictionary by combining default and custom aggregation functions.
    - If a function exists in both aggregations and custom_aggregations for a field, prefer the one in aggregations.
    - If functions are different, include all from both for that field (no duplicates).

    Returns
    -------
    Dict[str, List[Callable]]
        A dictionary mapping fields to a list of callable aggregation functions.
    """
    agg_dict = {}

    # Add all standard aggregations first
    if aggregations:
        for field, funcs in aggregations.items():
            agg_dict[field] = [_get_aggregation_function(func) for func in funcs]

    # Add custom aggregations, but only if not already present for that field/function
    if custom_aggregations:
        for field, funcs in custom_aggregations.items():
            # Ensure funcs is always a list
            if callable(funcs):
                funcs = [funcs]
            # Convert to function objects
            custom_funcs = [_get_aggregation_function(func) for func in funcs]
            if field in agg_dict:
                # Only add custom functions that are not already in agg_dict[field]
                for cf in custom_funcs:
                    # Compare by function object identity or function name
                    if not any(
                        (cf == af)
                        or (
                            hasattr(cf, "__name__")
                            and hasattr(af, "__name__")
                            and cf.__name__ == af.__name__
                        )
                        for af in agg_dict[field]
                    ):
                        agg_dict[field].append(cf)
            else:
                agg_dict[field] = custom_funcs

    logger.debug(
        "Built aggregation dict (agg prioritized, merged if different): %s", agg_dict
    )
    return agg_dict


def flatten_multiindex_columns(columns) -> List[str]:
    """
    Flatten a MultiIndex column to a single-level column name for readability.

    Parameters
    ----------
    columns : Union[pd.Index, pd.MultiIndex]
        The DataFrame column index to flatten.

    Returns
    -------
    List[str]
        A list of flattened column names as strings.
    """
    flattened = []
    for col in columns:
        part1 = str(col[0]) if col[0] not in [None, ""] else "col"
        part2 = ""

        if len(col) > 1:
            if callable(col[1]):
                part2 = col[1].__name__
            elif col[1] not in [None, ""]:
                part2 = str(col[1])

        if part2:
            flattened.append(f"{part1}_{part2}")
        else:
            flattened.append(part1)

    logger.debug("Flattened columns: %s", flattened)
    return flattened


def _get_aggregation_function(agg_name: str) -> Callable:
    """Get the aggregation function by name.

    Args:
        agg_name: Name of the aggregation function
    Returns:
        Callable aggregation function
    """
    if agg_name in STANDARD_AGGREGATIONS:
        return STANDARD_AGGREGATIONS[agg_name]
    elif agg_name in CUSTOM_AGG_FUNCTIONS:
        return CUSTOM_AGG_FUNCTIONS[agg_name]
    else:
        raise ValueError(
            f"Aggregation function '{agg_name}' not found in allowed registries"
        )


def is_dask_compatible_function(func: Union[str, Callable]) -> bool:
    """
    Check if a function is compatible with Dask aggregation.

    Parameters
    ----------
    func : Union[str, Callable]
        Function to check

    Returns
    -------
    bool
        True if function is Dask-compatible
    """
    if isinstance(func, str):
        return func in STANDARD_AGGREGATIONS

    if hasattr(func, "__module__"):
        module = func.__module__
        return module and (
            module.startswith("numpy")
            or module.startswith("pandas")
            or module.startswith("builtins")
        )

    return False


def apply_custom_aggregations_post_dask(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    custom_agg_dict: Dict[str, List[Callable]],
    group_by_fields: List[str],
) -> pd.DataFrame:
    """
    Apply custom aggregation functions to the original DataFrame grouped by group_by_fields,
    then merge the results into the aggregated result DataFrame.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original DataFrame before aggregation (needed for custom aggregations).
    result_df : pd.DataFrame
        The DataFrame after Dask-compatible aggregations.
    custom_agg_dict : Dict[str, List[Callable]]
        Dictionary mapping column names to a list of custom aggregation functions.
    group_by_fields : List[str]
        List of fields used for grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for custom aggregations merged in.
    """
    logger.debug(
        "Applying custom aggregations post-Dask using original_df. Fields: %s, Functions: %s",
        list(custom_agg_dict.keys()),
        {
            k: [getattr(f, "__name__", str(f)) for f in v]
            for k, v in custom_agg_dict.items()
        },
    )

    custom_agg_results = pd.DataFrame()

    for col, funcs in custom_agg_dict.items():
        for func in funcs:
            func_name = getattr(func, "__name__", str(func))
            try:
                agg_series = (
                    original_df.groupby(group_by_fields)[col]
                    .agg(func)
                    .reset_index()
                    .rename(columns={col: f"{col}_{func_name}"})
                )
                if custom_agg_results.empty:
                    custom_agg_results = agg_series
                else:
                    custom_agg_results = pd.merge(
                        custom_agg_results,
                        agg_series,
                        on=group_by_fields,
                        how="outer",
                    )
            except Exception as e:
                logger.error(
                    f"Error applying custom aggregation '{func_name}' on column '{col}': {e}"
                )

    # Ensure result_df has flat columns
    if isinstance(result_df.columns, pd.MultiIndex):
        result_df.columns = flatten_multiindex_columns(result_df.columns)

    # Group_by_fields as columns (not index)
    if group_by_fields and not all(f in result_df.columns for f in group_by_fields):
        result_df = result_df.reset_index()

    if not custom_agg_results.empty:
        result_df = pd.merge(
            result_df,
            custom_agg_results,
            on=group_by_fields,
            how="left",
        )

    return result_df
