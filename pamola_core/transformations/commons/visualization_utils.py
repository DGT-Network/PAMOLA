"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Transformation Visualization Utilities

This module provides utility functions for visualizing and analyzing the impact of data transformation operations on datasets.
It includes tools for:
    - Comparing field (column) counts and record counts before and after transformation
    - Comparing data distributions for specific fields
    - Generating dataset overviews and profiling statistics
    - Creating visualizations such as bar charts, histograms, and pie charts for both original and transformed data

These utilities support both before/after comparisons and aggregate/grouped data visualizations.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

from pamola_core.utils.visualization import (
    create_bar_plot,
    create_histogram,
    create_pie_chart,
)

logger = logging.getLogger(__name__)


def generate_visualization_filename(
    operation_name: str,
    visualization_type: str,
    extension: str = "png",
    join_filename: Optional[str] = None,
    include_timestamp: Optional[bool] = None,
) -> str:
    """
    Generate a standardized filename for a visualization.

    Parameters:
    -----------
    operation_name : str
        Name of the operation creating the visualization
    visualization_type : str
        Type of visualization (e.g., "histogram", "distribution")
    extension : str
        File extension (default: "png")
    join_filename : str, optional
        String join to filename.
    include_timestamp : bool, optional
        Timestamp for file naming. If None, current timestamp is used.

    Returns:
    --------
    str
        Standardized filename
    """
    filename = f"{operation_name}_{visualization_type}"

    if join_filename:
        filename = f"{filename}_{join_filename}"

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"

    return f"{filename}.{extension}"


def create_field_count_comparison(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Generate a comparison of field (column) counts before and after a transformation.

    Parameters:
        original_df (pd.DataFrame): The original dataset before transformation.
        transformed_df (pd.DataFrame): The dataset after transformation.
        operation_name (str): Name of the transformation operation.
        output_path (Path): Path to save visualizations or reports (not used yet).

    Returns:
        Dict[str, Any]: Dictionary with count comparison, added/removed fields,
                        and chart recommendation.
    """
    logger.debug("Creating field count comparison for operation: %s", operation_name)

    original_fields = set(original_df.columns)
    transformed_fields = set(transformed_df.columns)

    original_count = len(original_fields)
    transformed_count = len(transformed_fields)
    field_diff = transformed_count - original_count
    percent_change = (field_diff / original_count * 100) if original_count else 0

    added_fields = list(transformed_fields - original_fields)
    removed_fields = list(original_fields - transformed_fields)

    result = {
        "operation_name": operation_name,
        "original_field_count": original_count,
        "transformed_field_count": transformed_count,
        "absolute_change": field_diff,
        "percent_change": percent_change,
        "added_fields": added_fields,
        "removed_fields": removed_fields,
        "chart_recommendation": (
            "Use a bar chart to compare original vs transformed field counts. "
            "Highlight added fields in green and removed fields in red."
        ),
    }

    logger.debug("Field count comparison result: %s", result)
    return result


def create_record_count_comparison(
    original_df: pd.DataFrame,
    transformed_dfs: Dict[str, pd.DataFrame],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Create a record count comparison summary between the original dataset
    and one or more transformed datasets.

    Args:
        original_df (pd.DataFrame): The original input DataFrame.
        transformed_dfs (Dict[str, pd.DataFrame]): A dictionary mapping output names to transformed DataFrames.
        operation_name (str): The name of the operation being analyzed.
        output_path (Path): Path to where visualizations or logs may be stored (currently unused).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - operation_name (str): Name of the transformation operation.
            - original_record_count (int): Number of records in the original dataset.
            - transformed_record_counts (Dict[str, int]): Record counts per transformed dataset.
            - total_transformed_records (int): Combined record count across all transformed datasets.
            - absolute_change (int): Absolute difference in total records after transformation.
            - percent_change (float): Percent change in record count.
            - chart_recommendation (str): Suggestion for visualizing the comparison.
            - additional_chart_recommendation (str, optional): Suggested pie chart if multiple outputs exist.
    """
    logger.debug("Creating record count comparison for operation: %s", operation_name)

    original_count: int = len(original_df)
    transformed_counts: Dict[str, int] = {
        name: len(df) for name, df in transformed_dfs.items()
    }
    total_transformed: int = sum(transformed_counts.values())

    record_diff: int = total_transformed - original_count
    percent_change: float = (
        (record_diff / original_count) * 100 if original_count else 0.0
    )

    result: Dict[str, Any] = {
        "operation_name": operation_name,
        "original_record_count": original_count,
        "transformed_record_counts": transformed_counts,
        "total_transformed_records": total_transformed,
        "absolute_change": record_diff,
        "percent_change": percent_change,
        "chart_recommendation": (
            "Bar chart comparing original record count to transformed record counts "
            "by output dataset."
        ),
    }

    if len(transformed_dfs) > 1:
        result["additional_chart_recommendation"] = (
            "Pie chart showing distribution of records across different output datasets."
        )

    logger.debug("Record count comparison result: %s", result)
    return result


def create_data_distribution_comparison(
    original_series: pd.Series,
    transformed_series: pd.Series,
    field_name: str,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Analyze and summarize data distribution before and after transformation,
    including ready-to-use data for plotting.

    Parameters
    ----------
    original_series : pd.Series
        Original data series.
    transformed_series : pd.Series
        Transformed (anonymized) data series.
    field_name : str
        Name of the field being analyzed.
    operation_name : str
        Name of the transformation operation.
    output_path : Path
        Directory path where outputs are saved (not used here, but kept for interface consistency).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing statistics, plot data, and chart recommendations.
    """
    logger.debug("Analyzing data distribution for field: %s", field_name)

    # Check if data is numeric
    is_numeric = pd.api.types.is_numeric_dtype(
        original_series
    ) and pd.api.types.is_numeric_dtype(transformed_series)

    def describe(series: pd.Series) -> Dict[str, Any]:
        """
        Get summary statistics or value counts for the given series.

        If the series is numeric, includes min, max, mean, median, and std.
        If non-numeric, includes the top 10 most frequent values.

        Parameters
        ----------
        series : pd.Series
            Data series to summarize.

        Returns
        -------
        Dict[str, Any]
            Summary statistics or value counts for the series.
        """
        desc = {
            "count": len(series),
            "null_count": series.isna().sum(),
            "null_percentage": (
                (series.isna().sum() / len(series)) * 100 if len(series) else 0.0
            ),
            "unique_values": series.nunique(),
        }
        if is_numeric and not series.empty:
            desc.update(
                {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                }
            )
        elif not is_numeric:
            desc["top_values"] = series.value_counts().head(10).to_dict()
        return desc

    # Gather statistics for original and transformed series
    stats = {
        "original": describe(original_series),
        "transformed": describe(transformed_series),
    }

    # Prepare plot data based on data type
    if is_numeric:
        # If numeric, prepare clean data lists for plotting
        original_clean = original_series.dropna().tolist()
        transformed_clean = transformed_series.dropna().tolist()
        plot_data = {"Original": original_clean, "Transformed": transformed_clean}
    else:
        # If categorical, prepare top value counts for plotting
        max_categories = 10
        orig_counts = original_series.value_counts().head(max_categories).to_dict()
        anon_counts = transformed_series.value_counts().head(max_categories).to_dict()
        plot_data = {"Original": orig_counts, "Transformed": anon_counts}

    # Compile results and recommendations
    result = {
        "field_name": field_name,
        "operation_name": operation_name,
        "is_numeric": is_numeric,
        "statistics": stats,
        "plot_data": plot_data,
        "chart_recommendation": (
            "Histogram or box plot comparing the distribution of values before and after transformation."
            if is_numeric
            else "Bar chart comparing top categories before and after transformation."
        ),
    }

    logger.debug(
        "Data distribution comparison result for field '%s': %s", field_name, result
    )
    return result


def create_dataset_overview(
    df: pd.DataFrame, title: str, output_path: Path
) -> Dict[str, Any]:
    """
    Generate a comprehensive overview of a dataset including statistics
    for different data types and profiling information.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataset to be profiled.
    title : str
        A descriptive title for the dataset (used in reports/logs).
    output_path : Path
        The path where any generated reports or visualizations may be saved.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing:
            - title: str - Provided title
            - record_count: int - Number of rows in the dataset
            - field_count: int - Number of columns in the dataset
            - memory_usage_mb: float - Memory usage in megabytes
            - data_types: Dict[str, int] - Count of columns by data type
            - columns_with_nulls: Dict[str, int] - Columns with non-zero null values
            - unique_value_counts: Dict[str, int] - Unique value counts per column
            - numeric_stats: Dict[str, Dict[str, float]] - Min, max, mean, median, std for numeric fields
            - categorical_stats: Dict[str, Dict] - Top 5 values and unique count for categorical fields
            - datetime_stats: Dict[str, Dict[str, str]] - Min and max dates for datetime fields
            - boolean_stats: Dict[str, Dict[str, int]] - Count of True/False values for boolean fields
            - chart_recommendations: List[str] - Suggested visualizations for the dataset
    """
    logger.debug("Creating dataset overview for: %s", title)

    record_count = len(df)
    field_count = len(df.columns)
    dtype_counts = df.dtypes.value_counts().to_dict()
    null_counts = df.isna().sum()
    columns_with_nulls = {
        col: int(count) for col, count in null_counts.items() if count > 0
    }
    unique_values = {col: df[col].nunique() for col in df.columns}
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Numeric stats
    numeric_stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
        }
        for col in df.select_dtypes(include=["number"]).columns
    }

    # Categorical stats
    categorical_stats = {
        col: {
            "top": df[col].value_counts().head(20).to_dict(),
            "unique": int(df[col].nunique()),
        }
        for col in df.select_dtypes(include=["object", "category"]).columns
    }

    # Datetime stats
    datetime_stats = {
        col: {
            "min": str(df[col].min()) if not df[col].isna().all() else None,
            "max": str(df[col].max()) if not df[col].isna().all() else None,
        }
        for col in df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns
    }

    # Boolean stats
    boolean_stats = {
        col: {
            "true": int((df[col] == True).sum()),
            "false": int((df[col] == False).sum()),
        }
        for col in df.select_dtypes(include=["bool"]).columns
    }

    result = {
        "title": title,
        "record_count": record_count,
        "field_count": field_count,
        "memory_usage_mb": memory_usage,
        "data_types": dtype_counts,
        "columns_with_nulls": columns_with_nulls,
        "unique_value_counts": unique_values,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "datetime_stats": datetime_stats,
        "boolean_stats": boolean_stats,
        "chart_recommendations": [
            "Bar chart showing count of columns by data type.",
            "Bar chart showing columns with highest percentage of null values.",
            "Bar chart showing columns with highest cardinality (unique values).",
            "Bar chart for numeric fields: min/max/mean/median/std.",
            "Bar chart for categorical fields: top values and unique counts.",
            "Bar chart for datetime fields: min/max timestamps.",
            "Bar chart for boolean fields: count of True/False values.",
        ],
    }

    logger.debug("Dataset overview result: %s", result)
    return result


def generate_dataset_overview_vis(
    df: pd.DataFrame,
    operation_name: str,
    dataset_label: str,
    field_label: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Generate visualization charts for a dataset overview.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    operation_name : str
        Operation name (e.g. "Imputation").
    dataset_label : str
        Either "original" or "transformed".
    field_label : str
        Used in filenames and chart titles.
    task_dir : Path
        Directory to store output charts.
    timestamp : str
        Timestamp string to ensure unique filenames.
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    visualization_paths : Optional[Dict[str, Any]]
        Dictionary to update with chart paths.
    kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths dictionary.
    """
    if visualization_paths is None:
        visualization_paths = {}

    overview = create_dataset_overview(
        df, f"{operation_name} - {dataset_label.title()} Dataset", task_dir
    )

    # 1. Data type count
    dtype_counts = overview.get("data_types", {})
    if dtype_counts:
        # Convert dtype keys to string for serialization/plotting
        dtype_counts_str = {str(k): int(v) for k, v in dtype_counts.items()}
        path = (
            task_dir
            / f"{field_label}_{operation_name}_{dataset_label}_dtype_counts_{timestamp}.png"
        )
        vis_path = create_bar_plot(
            data=dtype_counts_str,
            output_path=path,
            title=f"Count of Columns by Data Type - {field_label} ({dataset_label.title()})",
            x_label="Data Type",
            y_label="Number of Columns",
            orientation="v",
            sort_by="value",
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs
        )
        visualization_paths[f"{dataset_label}_dtype_counts_bar_chart"] = vis_path

    # 2. Null percentage
    columns_with_nulls = overview.get("columns_with_nulls", {})
    if columns_with_nulls:
        total_rows = overview.get("record_count", 1)
        null_percentages = {
            col: (count / total_rows) * 100 for col, count in columns_with_nulls.items()
        }
        top_nulls = dict(
            sorted(null_percentages.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        path = (
            task_dir
            / f"{field_label}_{operation_name}_{dataset_label}_columns_null_percentage_{timestamp}.png"
        )
        vis_path = create_bar_plot(
            data=top_nulls,
            output_path=path,
            title=f"Columns with Highest Percentage of Null Values - {field_label} ({dataset_label.title()})",
            x_label="Column",
            y_label="Null Percentage (%)",
            orientation="v",
            sort_by="value",
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs
        )
        visualization_paths[f"{dataset_label}_columns_null_percentage_bar_chart"] = (
            vis_path
        )

    # 3. Cardinality
    unique_value_counts = overview.get("unique_value_counts", {})
    if unique_value_counts:
        top_cardinalities = dict(
            sorted(unique_value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        path = (
            task_dir
            / f"{field_label}_{operation_name}_{dataset_label}_columns_cardinality_{timestamp}.png"
        )
        vis_path = create_bar_plot(
            data=top_cardinalities,
            output_path=path,
            title=f"Columns with Highest Cardinality (Unique Values) - {field_label} ({dataset_label.title()})",
            x_label="Column",
            y_label="Unique Value Count",
            orientation="v",
            sort_by="value",
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs
        )
        visualization_paths[f"{dataset_label}_columns_cardinality_bar_chart"] = vis_path

    # 4. Numeric field stats
    numeric_stats = overview.get("numeric_stats", {})
    if numeric_stats:
        for stat in ["min", "max", "mean", "median", "std"]:
            stat_values = {
                col: vals.get(stat)
                for col, vals in numeric_stats.items()
                if vals.get(stat) is not None
            }
            if stat_values:
                path = (
                    task_dir
                    / f"{field_label}_{operation_name}_{dataset_label}_numeric_{stat}_{timestamp}.png"
                )
                vis_path = create_bar_plot(
                    data=stat_values,
                    output_path=path,
                    title=f"{stat.capitalize()} for Numeric Fields - {field_label} ({dataset_label.title()})",
                    x_label="Column",
                    y_label=stat.capitalize(),
                    orientation="v",
                    sort_by="value",
                    theme=theme,
                    backend=backend,
                    strict=strict,
                    **kwargs
                )
                visualization_paths[f"{dataset_label}_numeric_{stat}_bar_chart"] = (
                    vis_path
                )

    # 5. Categorical field stats
    categorical_stats = overview.get("categorical_stats", {})
    if categorical_stats:
        top_counts = {
            col: vals.get("top_value_count", 0)
            for col, vals in categorical_stats.items()
            if "top_value_count" in vals
        }
        unique_counts = {
            col: vals.get("unique", 0)
            for col, vals in categorical_stats.items()
            if "unique" in vals
        }

        if top_counts:
            path = (
                task_dir
                / f"{field_label}_{operation_name}_{dataset_label}_categorical_top_value_count_{timestamp}.png"
            )
            vis_path = create_bar_plot(
                data=top_counts,
                output_path=path,
                title=f"Top Value Count for Categorical Fields - {field_label} ({dataset_label.title()})",
                x_label="Column",
                y_label="Top Value Count",
                orientation="v",
                sort_by="value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs
            )
            visualization_paths[
                f"{dataset_label}_categorical_top_value_count_bar_chart"
            ] = vis_path

        if unique_counts:
            path = (
                task_dir
                / f"{field_label}_{operation_name}_{dataset_label}_categorical_unique_count_{timestamp}.png"
            )
            vis_path = create_bar_plot(
                data=unique_counts,
                output_path=path,
                title=f"Unique Value Count for Categorical Fields - {field_label} ({dataset_label.title()})",
                x_label="Column",
                y_label="Unique Count",
                orientation="v",
                sort_by="value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs
            )
            visualization_paths[
                f"{dataset_label}_categorical_unique_count_bar_chart"
            ] = vis_path

    # 6. Datetime field stats
    datetime_stats = overview.get("datetime_stats", {})
    if datetime_stats:
        min_timestamps = {
            col: pd.to_datetime(vals.get("min")).timestamp()
            for col, vals in datetime_stats.items()
            if vals.get("min")
        }
        max_timestamps = {
            col: pd.to_datetime(vals.get("max")).timestamp()
            for col, vals in datetime_stats.items()
            if vals.get("max")
        }

        if min_timestamps:
            path = (
                task_dir
                / f"{field_label}_{operation_name}_{dataset_label}_datetime_min_{timestamp}.png"
            )
            vis_path = create_bar_plot(
                data=min_timestamps,
                output_path=path,
                title=f"Min Timestamps for Datetime Fields - {field_label} ({dataset_label.title()})",
                x_label="Column",
                y_label="Timestamp (Min)",
                orientation="v",
                sort_by="value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs
            )
            visualization_paths[f"{dataset_label}_datetime_min_bar_chart"] = vis_path

        if max_timestamps:
            path = (
                task_dir
                / f"{field_label}_{operation_name}_{dataset_label}_datetime_max_{timestamp}.png"
            )
            vis_path = create_bar_plot(
                data=max_timestamps,
                output_path=path,
                title=f"Max Timestamps for Datetime Fields - {field_label} ({dataset_label.title()})",
                x_label="Column",
                y_label="Timestamp (Max)",
                orientation="v",
                sort_by="value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs
            )
            visualization_paths[f"{dataset_label}_datetime_max_bar_chart"] = vis_path

    # 7. Boolean field stats
    boolean_stats = overview.get("boolean_stats", {})
    if boolean_stats:
        true_counts = {
            col: vals.get("true", 0)
            for col, vals in boolean_stats.items()
            if "true" in vals
        }
        false_counts = {
            col: vals.get("false", 0)
            for col, vals in boolean_stats.items()
            if "false" in vals
        }

        if true_counts:
            path = (
                task_dir
                / f"{field_label}_{operation_name}_{dataset_label}_boolean_true_counts_{timestamp}.png"
            )
            vis_path = create_bar_plot(
                data=true_counts,
                output_path=path,
                title=f"True Counts for Boolean Fields - {field_label} ({dataset_label.title()})",
                x_label="Column",
                y_label="Count of True",
                orientation="v",
                sort_by="value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs
            )
            visualization_paths[f"{dataset_label}_boolean_true_counts_bar_chart"] = (
                vis_path
            )

        if false_counts:
            path = (
                task_dir
                / f"{field_label}_{operation_name}_{dataset_label}_boolean_false_counts_{timestamp}.png"
            )
            vis_path = create_bar_plot(
                data=false_counts,
                output_path=path,
                title=f"False Counts for Boolean Fields - {field_label} ({dataset_label.title()})",
                x_label="Column",
                y_label="Count of False",
                orientation="v",
                sort_by="value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs
            )
            visualization_paths[f"{dataset_label}_boolean_false_counts_bar_chart"] = (
                vis_path
            )

    return visualization_paths


def generate_data_distribution_comparison_vis(
    original_data: pd.Series,
    transformed_data: pd.Series,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Generate distribution comparison visualization for a single field.

    Parameters
    ----------
    original_data : pd.Series
        Original column data.
    transformed_data : pd.Series
        Transformed column data.
    field_label : str
        Name of the column being visualized.
    operation_name : str
        Operation label (e.g. "Imputation").
    task_dir : Path
        Directory to save plots.
    timestamp : str
        Timestamp for unique filenames.
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    visualization_paths : Optional[Dict[str, Any]]
        Dictionary to store visualization paths.
    kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths.
    """
    if visualization_paths is None:
        visualization_paths = {}

    dist_summary = create_data_distribution_comparison(
        original_series=original_data,
        transformed_series=transformed_data,
        field_name=field_label,
        operation_name=operation_name,
        output_path=task_dir,
    )

    if dist_summary["is_numeric"]:
        # Histogram for numeric fields
        hist_file_name = f"{field_label}_{operation_name}_data_distribution_numeric_comparison_{timestamp}.png"
        hist_path = task_dir / hist_file_name

        vis_path = create_histogram(
            data=dist_summary["plot_data"],
            output_path=hist_path,
            title=f"Distribution Comparison for {field_label}",
            x_label=field_label,
            y_label="Frequency",
            bins=30,
            kde=True,
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs
        )
        visualization_paths["numeric_comparison_histogram"] = vis_path

    else:
        # Bar chart for categorical fields
        bar_file_name = f"{field_label}_{operation_name}_data_distribution_category_comparison_{timestamp}.png"
        bar_path = task_dir / bar_file_name

        vis_path = create_bar_plot(
            data=dist_summary["plot_data"],
            output_path=bar_path,
            title=f"Category Comparison for {field_label}",
            x_label="Category",
            y_label="Count",
            orientation="v",
            sort_by="value",
            max_items=10,
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs
        )
        visualization_paths["category_comparison_bar_chart"] = vis_path

    return visualization_paths


def generate_record_count_comparison_vis(
    original_df: pd.DataFrame,
    transformed_dfs: Dict[str, pd.DataFrame],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Generate record count comparison visualization between original and transformed dataset.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original dataset.
    transformed_dfs : Dict[str, pd.DataFrame]
        A dictionary mapping output names to transformed DataFrames.
    field_label : str
        Name of the field/column.
    operation_name : str
        Operation being performed (e.g. "Imputation").
    task_dir : Path
        Directory where plots are saved.
    timestamp : str
        Timestamp for filename uniqueness.
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    visualization_paths : Optional[Dict[str, Any]]
        Dict to collect plot paths.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths.
    """
    if visualization_paths is None:
        visualization_paths = {}

    record_count_summary = create_record_count_comparison(
        original_df=original_df,
        transformed_dfs=transformed_dfs,
        operation_name=operation_name,
        output_path=task_dir,
    )

    record_counts: Dict[str, int] = record_count_summary["transformed_record_counts"]

    if len(transformed_dfs) > 1:
        pie_chart_path = (
            task_dir
            / f"{field_label}_{operation_name}_record_count_distribution_{timestamp}.png"
        )

        pie_chart_result_path = create_pie_chart(
            data=record_counts,
            output_path=pie_chart_path,
            title=f"Record Distribution by Output Dataset for '{field_label}'",
            show_percentages=True,
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs
        )

        visualization_paths["record_count_distribution_pie_chart"] = (
            pie_chart_result_path
        )

    else:
        bar_data: Dict[str, int] = {
            "Original Records": record_count_summary["original_record_count"],
            "Transformed Records": record_count_summary["total_transformed_records"],
        }

        bar_chart_path = (
            task_dir
            / f"{field_label}_{operation_name}_record_count_comparison_{timestamp}.png"
        )

        bar_chart_result_path = create_bar_plot(
            data=bar_data,
            output_path=bar_chart_path,
            title=f"Record Count Comparison for '{field_label}'",
            x_label="Dataset",
            y_label="Record Count",
            orientation="v",
            showlegend=False,
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs
        )

        visualization_paths["record_count_comparison_bar_chart"] = bar_chart_result_path

    return visualization_paths


def generate_field_count_comparison_vis(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    visualization_paths: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate field count comparison visualization between original and transformed dataset.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original dataset.
    transformed_df : pd.DataFrame
        Transformed dataset for the field.
    field_label : str
        Name of the field/column.
    operation_name : str
        Operation being performed (e.g. "Imputation").
    task_dir : Path
        Directory where plots are saved.
    timestamp : str
        Timestamp for filename uniqueness.
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    visualization_paths : Optional[Dict[str, Any]]
        Dict to collect plot paths.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths.
    """
    if visualization_paths is None:
        visualization_paths = {}

    field_count_summary = create_field_count_comparison(
        original_df=original_df,
        transformed_df=transformed_df,
        operation_name=operation_name,
        output_path=task_dir,
    )

    bar_data: Dict[str, int] = {
        "Original Fields": field_count_summary["original_field_count"],
        "Transformed Fields": field_count_summary["transformed_field_count"],
        "Added Fields": len(field_count_summary["added_fields"]),
        "Removed Fields": len(field_count_summary["removed_fields"]),
    }

    bar_chart_path: Path = (
        task_dir
        / f"{field_label}_{operation_name}_field_count_comparison_{timestamp}.png"
    )

    bar_chart_result_path = create_bar_plot(
        data=bar_data,
        output_path=bar_chart_path,
        title=f"Field Count Comparison for '{field_label}'",
        x_label="Field Count Type",
        y_label="Count",
        orientation="v",
        showlegend=False,
        theme=theme,
        backend=backend,
        strict=strict,
        **kwargs,
    )

    visualization_paths["field_count_comparison_bar_chart"] = bar_chart_result_path

    return visualization_paths


def sample_large_dataset(
    data: Union[pd.Series, pd.DataFrame],
    max_samples: int = 10000,
    random_state: int = 42,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Sample a large Series or DataFrame to a manageable size for visualization.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        Original large dataset
    max_samples : int, optional
        Maximum number of samples to return (default: 10000)
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Sampled subset of the original data
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame")

    if len(data) <= max_samples:
        return data

    return data.sample(n=max_samples, random_state=random_state)

