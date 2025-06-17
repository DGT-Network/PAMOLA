"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Merging Visualization & Utility Functions

This module provides utility functions for:
    - Preparing and generating visualizations for merging/join operations
    - Creating Venn diagrams for record and field overlaps
    - Creating bar charts and pie charts for dataset size and join result distributions
    - Summarizing overlap, size, and join-type statistics for reporting and visualization

All major steps and data preparation are logged at DEBUG level for traceability.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from pamola_core.utils.visualization import (
    create_bar_plot,
    create_pie_chart,
    create_venn_diagram,
)

logger = logging.getLogger(__name__)


def create_record_overlap_data(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Prepare data for record overlap (venn) visualization.

    Parameters
    ----------
    left_df : pd.DataFrame
        Left-side DataFrame (e.g., source or batch data).
    right_df : pd.DataFrame
        Right-side DataFrame (e.g., reference or lookup data).
    left_key : str
        Column name in the left DataFrame to match records.
    right_key : str
        Column name in the right DataFrame to match records.
    operation_name : str
        Name of the current operation (used for labeling).
    output_path : Path
        Path where the visualization or related outputs may be saved.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - operation_name: Operation label.
        - left_count: Number of unique keys in the left DataFrame.
        - right_count: Number of unique keys in the right DataFrame.
        - overlap_count: Number of keys common to both DataFrames.
        - only_left_count: Keys only present in the left DataFrame.
        - only_right_count: Keys only present in the right DataFrame.
        - chart_recommendation: Visualization suggestion.
        - left_keys: Set of unique keys from the left DataFrame.
        - right_keys: Set of unique keys from the right DataFrame.
    """
    logger.debug(
        "Creating record overlap data for operation '%s' using keys: left_key='%s', right_key='%s'",
        operation_name,
        left_key,
        right_key,
    )
    left_keys = set(left_df[left_key])
    right_keys = set(right_df[right_key])
    overlap = len(left_keys & right_keys)
    only_left = len(left_keys - right_keys)
    only_right = len(right_keys - left_keys)
    result = {
        "operation_name": operation_name,
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "overlap_count": overlap,
        "only_left_count": only_left,
        "only_right_count": only_right,
        "chart_recommendation": "Venn diagram showing record overlap by key.",
        "left_keys": left_keys,
        "right_keys": right_keys,
    }
    logger.debug(
        "Record overlap result: left_count=%d, right_count=%d, overlap_count=%d, only_left_count=%d, only_right_count=%d",
        result["left_count"],
        result["right_count"],
        result["overlap_count"],
        result["only_left_count"],
        result["only_right_count"],
    )
    return result


def generate_record_overlap_vis(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate and save a Venn diagram showing record key overlap between two DataFrames.

    Parameters
    ----------
    left_df : pd.DataFrame
        Left-side input DataFrame.
    right_df : pd.DataFrame
        Right-side input DataFrame.
    left_key : str
        Column name for keys in the left DataFrame.
    right_key : str
        Column name for keys in the right DataFrame.
    field_label : str
        Descriptive label for the field or transformation being visualized.
    operation_name : str
        Name of the transformation or operation.
    task_dir : Path
        Directory where the Venn diagram image will be saved.
    timestamp : str
        Timestamp string for uniquely naming output files.
    theme : Optional[str]
        Visualization theme to use.
    backend : Optional[str]
        Visualization backend to use.
    strict : Optional[bool]
        Whether to enforce strict visualization rules.
    visualization_paths : Optional[Dict[str, Any]]
        Dictionary to store paths to generated visualizations. If None, a new one is created.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths dictionary with a new key:
        - "record_overlap_venn": Path to the saved Venn diagram image.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating record overlap visualization for operation '%s' (field_label='%s')",
        operation_name,
        field_label,
    )
    overlap_data = create_record_overlap_data(
        left_df, right_df, left_key, right_key, operation_name, task_dir
    )
    venn_path = (
        task_dir / f"{field_label}_{operation_name}_record_overlap_venn_{timestamp}.png"
    )
    venn_diagram_result_path = create_venn_diagram(
        set1=overlap_data["left_keys"],
        set2=overlap_data["right_keys"],
        output_path=venn_path,
        set1_label="Left",
        set2_label="Right",
        title="Record Overlap (Key Fields)",
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs
    )
    logger.debug("Venn diagram saved to: %s", venn_diagram_result_path)
    visualization_paths["record_overlap_venn"] = venn_diagram_result_path
    return visualization_paths


def create_dataset_size_comparison(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Prepare data for dataset size comparison bar chart.

    Parameters
    ----------
    left_df : pd.DataFrame
        The original left-side DataFrame.
    right_df : pd.DataFrame
        The original right-side DataFrame.
    merged_df : pd.DataFrame
        The resulting DataFrame after merging left and right.
    operation_name : str
        Name of the transformation or merge operation.
    output_path : Path
        Directory path where the visualization is expected to be saved.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing dataset sizes and visualization suggestions:
        - operation_name: Label of the current operation.
        - left_size: Number of rows in the left DataFrame.
        - right_size: Number of rows in the right DataFrame.
        - merged_size: Number of rows in the merged DataFrame.
        - chart_recommendation: Suggested chart type.
    """
    logger.debug(
        "Creating dataset size comparison for operation '%s': left_size=%d, right_size=%d, merged_size=%d",
        operation_name,
        len(left_df),
        len(right_df),
        len(merged_df),
    )
    result = {
        "operation_name": operation_name,
        "left_size": len(left_df),
        "right_size": len(right_df),
        "merged_size": len(merged_df),
        "chart_recommendation": "Bar chart comparing dataset sizes before and after merge.",
    }
    return result


def generate_dataset_size_comparison_vis(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate and save a bar chart comparing dataset sizes before and after a merge operation.

    Parameters
    ----------
    left_df : pd.DataFrame
        Left-side DataFrame before the merge.
    right_df : pd.DataFrame
        Right-side DataFrame before the merge.
    merged_df : pd.DataFrame
        DataFrame after merging left and right.
    field_label : str
        Descriptive label for the field or step being visualized.
    operation_name : str
        Name of the current transformation or merge operation.
    task_dir : Path
        Directory path where the output image will be saved.
    timestamp : str
        Timestamp string used to uniquely name the output file.
    theme : Optional[str]
        Visualization theme to use.
    backend : Optional[str]
        Visualization backend to use.
    strict : Optional[bool]
        Whether to enforce strict visualization rules.
    visualization_paths : Optional[Dict[str, Any]]
        Dictionary to store paths to generated visualizations. If None, a new one is created.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated dictionary containing visualization paths including:
        - "dataset_size_comparison_bar_chart": Path to the saved bar chart image.
    """
    if visualization_paths is None:
        visualization_paths = {}

    size_data = create_dataset_size_comparison(
        left_df, right_df, merged_df, operation_name, task_dir
    )
    bar_data = {
        "Left": size_data["left_size"],
        "Right": size_data["right_size"],
        "Merged": size_data["merged_size"],
    }
    bar_path = (
        task_dir
        / f"{field_label}_{operation_name}_dataset_size_comparison_{timestamp}.png"
    )
    logger.debug(
        "Generating dataset size comparison bar chart at: %s with data: %s",
        bar_path,
        bar_data,
    )
    bar_chart_result_path = create_bar_plot(
        data=bar_data,
        output_path=bar_path,
        title="Dataset Sizes Before/After Merge",
        x_label="Dataset",
        y_label="Number of Records",
        orientation="v",
        sort_by="key",
        showlegend=False,
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs
    )
    logger.debug("Bar chart saved to: %s", bar_chart_result_path)
    visualization_paths["dataset_size_comparison_bar_chart"] = bar_chart_result_path
    return visualization_paths


def create_field_overlap_data(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Prepare data for field overlap Venn diagram.

    Parameters
    ----------
    left_df : pd.DataFrame
        The left-side DataFrame to compare fields (columns) from.
    right_df : pd.DataFrame
        The right-side DataFrame to compare fields (columns) from.
    operation_name : str
        Name of the operation being performed (used for labeling).
    output_path : Path
        Directory path where the output file or visualization will be saved.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing metadata and data for Venn diagram creation:
        - operation_name: Name of the current operation.
        - left_field_count: Number of fields in the left DataFrame.
        - right_field_count: Number of fields in the right DataFrame.
        - overlap_field_count: Number of common fields between both DataFrames.
        - only_left_field_count: Fields present only in the left DataFrame.
        - only_right_field_count: Fields present only in the right DataFrame.
        - chart_recommendation: Suggested visualization (Venn diagram).
        - left_fields: Set of field names from the left DataFrame.
        - right_fields: Set of field names from the right DataFrame.
    """
    logger.debug("Creating field overlap data for operation '%s'", operation_name)
    left_fields = set(left_df.columns)
    right_fields = set(right_df.columns)
    overlap = len(left_fields & right_fields)
    only_left = len(left_fields - right_fields)
    only_right = len(right_fields - left_fields)
    result = {
        "operation_name": operation_name,
        "left_field_count": len(left_fields),
        "right_field_count": len(right_fields),
        "overlap_field_count": overlap,
        "only_left_field_count": only_left,
        "only_right_field_count": only_right,
        "chart_recommendation": "Venn diagram showing field (column) overlap.",
        "left_fields": left_fields,
        "right_fields": right_fields,
    }
    logger.debug(
        "Field overlap result: left_field_count=%d, right_field_count=%d, overlap_field_count=%d, only_left_field_count=%d, only_right_field_count=%d",
        result["left_field_count"],
        result["right_field_count"],
        result["overlap_field_count"],
        result["only_left_field_count"],
        result["only_right_field_count"],
    )
    return result


def generate_field_overlap_vis(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate and save a Venn diagram that visualizes field (column) overlap between two DataFrames.

    Parameters
    ----------
    left_df : pd.DataFrame
        The left-side DataFrame containing the first set of columns.
    right_df : pd.DataFrame
        The right-side DataFrame containing the second set of columns.
    field_label : str
        Label used for identifying the output chart file.
    operation_name : str
        Name of the operation being visualized.
    task_dir : Path
        Directory where the generated image should be saved.
    timestamp : str
        Timestamp string for uniquely naming the visualization file.
    theme : Optional[str]
        Visualization theme to use.
    backend : Optional[str]
        Visualization backend to use.
    strict : Optional[bool]
        Whether to enforce strict visualization rules.
    visualization_paths : Optional[Dict[str, Any]]
        Dictionary to store paths to generated visualizations. If None, a new one is created.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated dictionary with:
        - "field_overlap_venn": Path to the saved Venn diagram image.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating field overlap visualization for operation '%s' (field_label='%s')",
        operation_name,
        field_label,
    )
    overlap_data = create_field_overlap_data(
        left_df, right_df, operation_name, task_dir
    )
    venn_path = (
        task_dir / f"{field_label}_{operation_name}_field_overlap_venn_{timestamp}.png"
    )
    venn_diagram_result_path = create_venn_diagram(
        set1=overlap_data["left_fields"],
        set2=overlap_data["right_fields"],
        output_path=venn_path,
        set1_label="Left Fields",
        set2_label="Right Fields",
        title="Field Overlap",
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs
    )
    logger.debug("Field overlap Venn diagram saved to: %s", venn_diagram_result_path)
    visualization_paths["field_overlap_venn"] = venn_diagram_result_path
    return visualization_paths


def create_join_type_distribution_data(
    merged_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    join_type: str,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Prepare data for join type distribution pie chart.

    Parameters
    ----------
    merged_df : pd.DataFrame
        The merged DataFrame after a join operation.
    left_key : str
        The key column name from the left DataFrame used in the join.
    right_key : str
        The key column name from the right DataFrame used in the join.
    join_type : str
        The type of join performed (e.g., 'inner', 'left', 'right', 'outer').
    operation_name : str
        The name of the operation, used for labeling or logging.
    output_path : Path
        Path to the directory where any output or visualizations will be saved.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - operation_name: Name of the current operation.
        - matched_count: Number of records where keys matched.
        - only_left_count: Records only from the left side of the join.
        - only_right_count: Records only from the right side of the join.
        - join_type: The type of join performed.
        - chart_recommendation: Suggested visualization (pie chart).
    """
    logger.debug(
        "Creating join type distribution data for operation '%s' (join_type='%s')",
        operation_name,
        join_type,
    )
    matched = merged_df[left_key].notnull() & merged_df[right_key].notnull()
    only_left = merged_df[left_key].notnull() & merged_df[right_key].isnull()
    only_right = merged_df[left_key].isnull() & merged_df[right_key].notnull()
    result = {
        "operation_name": operation_name,
        "matched_count": int(matched.sum()),
        "only_left_count": int(only_left.sum()),
        "only_right_count": int(only_right.sum()),
        "join_type": join_type,
        "chart_recommendation": "Pie chart showing join result distribution.",
    }
    logger.debug(
        "Join type distribution: matched=%d, only_left=%d, only_right=%d",
        result["matched_count"],
        result["only_left_count"],
        result["only_right_count"],
    )
    return result


def generate_join_type_distribution_vis(
    merged_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    join_type: str,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: Optional[bool] = None,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate pie chart for join type distribution using create_pie_chart.

    Parameters
    ----------
    merged_df : pd.DataFrame
        The DataFrame resulting from the join operation.
    left_key : str
        Key column name from the left DataFrame.
    right_key : str
        Key column name from the right DataFrame.
    join_type : str
        Type of join performed (e.g., 'inner', 'left', 'right', 'outer').
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
        - "join_type_distribution_pie_chart": Path to the generated pie chart file.
    """
    if visualization_paths is None:
        visualization_paths = {}

    logger.debug(
        "Generating join type distribution pie chart for operation '%s' (field_label='%s', join_type='%s')",
        operation_name,
        field_label,
        join_type,
    )
    dist_data = create_join_type_distribution_data(
        merged_df, left_key, right_key, join_type, operation_name, task_dir
    )
    join_path = (
        task_dir
        / f"{field_label}_{operation_name}_join_type_distribution_{timestamp}.png"
    )
    pie_data = {
        "Matched": dist_data["matched_count"],
        "Only Left": dist_data["only_left_count"],
        "Only Right": dist_data["only_right_count"],
    }
    pie_chart_result_path = create_pie_chart(
        data=pie_data,
        output_path=join_path,
        title=f"Join Result Distribution ({join_type} join)",
        show_percentages=True,
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs
    )
    logger.debug("Pie chart saved to: %s", pie_chart_result_path)
    visualization_paths["join_type_distribution_pie_chart"] = pie_chart_result_path
    return visualization_paths
