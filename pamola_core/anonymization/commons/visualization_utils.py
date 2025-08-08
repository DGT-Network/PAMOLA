"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Anonymization Visualization Utilities
Description: Helper utilities for creating visualizations of anonymization metrics
Author: PAMOLA Core Team
Created: 2024-01
Modified: 2025-01
Version: 2.1.0
License: BSD 3-Clause

This module provides helper utilities for visualizing anonymization metrics
calculated by metric_utils.py and privacy_metric_utils.py. It serves as a
thin wrapper around pamola_core.utils.visualization with privacy-specific context.

Version History:
   1.0.0 (2024-01): Initial implementation with basic utilities
   2.0.0 (2024-12): Refactored for integration with ops framework
                    - Removed path management (using task_dir directly)
                    - Focused on metric visualization only
                    - Added wrappers for core visualization functions
   2.1.0 (2025-01): Added categorical anonymization visualizations
                    - Added create_category_distribution_comparison()
                    - Added create_hierarchy_sunburst()
                    - Enhanced support for categorical metrics
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Union, List

import numpy as np
import pandas as pd

# Import core visualization functions
from pamola_core.utils.visualization import (
    create_bar_plot,
    create_histogram,
    create_sunburst_chart,
)

logger = logging.getLogger(__name__)

# Constants for visualization
DEFAULT_MAX_SAMPLES = 10000
DEFAULT_MAX_CATEGORIES = 20
DEFAULT_HISTOGRAM_BINS = 30
DEFAULT_TOP_CATEGORIES_FOR_SUNBURST = 50


def generate_visualization_filename(
    field_name: str,
    operation_name: str,
    visualization_type: str,
    timestamp: Optional[str] = None,
    extension: str = "png",
) -> str:
    """
    Generate a standardized filename for a visualization.

    Parameters:
    -----------
    field_name : str
        Name of the field being visualized
    operation_name : str
        Name of the operation creating the visualization
    visualization_type : str
        Type of visualization (e.g., "histogram", "distribution", "comparison")
    timestamp : str, optional
        Timestamp for file naming. If None, current timestamp is used.
    extension : str, optional
        File extension (default: "png")

    Returns:
    --------
    str
        Standardized filename following pattern: {field}_{operation}_{visType}_{timestamp}.{ext}
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean field name to avoid path issues
    field_name_clean = field_name.replace("/", "_").replace("\\", "_")

    return f"{field_name_clean}_{operation_name}_{visualization_type}_{timestamp}.{extension}"


def register_visualization_artifact(
    result: Any,
    reporter: Any,
    path: Path,
    field_name: str,
    visualization_type: str,
    description: Optional[str] = None,
) -> None:
    """
    Register a visualization artifact with the result and reporter.

    Parameters:
    -----------
    result : OperationResult
        Operation result to add the artifact to
    reporter : Reporter
        Reporter to add the artifact to (can be None)
    path : Path
        Path to the visualization file
    field_name : str
        Name of the field being visualized
    visualization_type : str
        Type of visualization
    description : str, optional
        Custom description of the visualization
    """
    if description is None:
        description = f"{field_name} {visualization_type} visualization"

    # Add to result
    if hasattr(result, "add_artifact"):
        result.add_artifact(
            artifact_type="png",
            path=path,
            description=description,
            category="visualization",
            tags=["metrics", visualization_type],
        )

    # Add to reporter
    if reporter and hasattr(reporter, "add_artifact"):
        reporter.add_artifact("png", str(path), description)


def sample_large_dataset(
    data: pd.Series, max_samples: int = DEFAULT_MAX_SAMPLES, random_state: int = 42
) -> pd.Series:
    """
    Sample a large dataset to a manageable size for visualization.

    Parameters:
    -----------
    data : pd.Series
        Original dataset
    max_samples : int, optional
        Maximum number of samples to return (default: 10000)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.Series
        Sampled dataset if original exceeds max_samples, otherwise original
    """
    if len(data) <= max_samples:
        return data

    logger.info(f"Sampling {max_samples} from {len(data)} records for visualization")
    return data.sample(n=max_samples, random_state=random_state)


def prepare_comparison_data(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> Tuple[Dict[str, Any], str]:
    """
    Prepare data for before/after comparison visualizations.

    Parameters:
    -----------
    original_data : pd.Series
        Original data before anonymization
    anonymized_data : pd.Series
        Data after anonymization
    max_categories : int, optional
        Maximum categories to show for categorical data

    Returns:
    --------
    Tuple[Dict[str, Any], str]
        (prepared_data, data_type) where data_type is 'int', 'float', or 'categorical'
    """
    original_data_clean = original_data.dropna()
    anonymized_data_clean = anonymized_data.dropna()

    # Determine data type
    orig_is_numeric = pd.api.types.is_numeric_dtype(original_data_clean)

    # === Combined Case: Original is numeric ===
    if orig_is_numeric:
        numeric_type = (
            "int" if pd.api.types.is_integer_dtype(original_data_clean) else "float"
        )
        # Even if anonymized is string (e.g., bucketed like "20–30"), we treat as numeric category
        return {
            "Original": original_data_clean.tolist(),
            "Anonymized": anonymized_data_clean.tolist(),
        }, numeric_type

    # === Default categorical fallback ===
    # For categorical, get value counts
    orig_counts = original_data.value_counts().head(max_categories)
    anon_counts = anonymized_data.value_counts().head(max_categories)

    # Get all unique categories from both
    all_categories = sorted(
        set(orig_counts.index) | set(anon_counts.index),
        key=lambda x: -orig_counts.get(x, 0),
    )[:max_categories]

    # Create dictionaries with aligned categories
    orig_dict = {str(cat): int(orig_counts.get(cat, 0)) for cat in all_categories}
    anon_dict = {str(cat): int(anon_counts.get(cat, 0)) for cat in all_categories}

    return {"Original": orig_dict, "Anonymized": anon_dict}, "categorical"


def calculate_optimal_bins(
    data: pd.Series, min_bins: int = 10, max_bins: int = DEFAULT_HISTOGRAM_BINS
) -> int:
    """
    Calculate optimal number of bins for histograms using Sturges' rule.

    Parameters:
    -----------
    data : pd.Series
        Data to calculate bins for
    min_bins : int, optional
        Minimum number of bins (default: 10)
    max_bins : int, optional
        Maximum number of bins (default: 30)

    Returns:
    --------
    int
        Optimal number of bins
    """
    n = len(data.dropna())
    if n == 0:
        return min_bins

    # Sturges' rule: k = 1 + log2(n)
    sturges = int(np.ceil(1 + np.log2(n)))

    # Square root rule as alternative: k = sqrt(n)
    sqrt_rule = int(np.ceil(np.sqrt(n)))

    # Use the average of both rules
    optimal = (sturges + sqrt_rule) // 2

    return max(min_bins, min(optimal, max_bins))


def create_metric_visualization(
    metric_name: str,
    metric_data: Union[Dict[str, Any], pd.Series, List],
    task_dir: Path,
    field_name: str,
    operation_name: str,
    timestamp: Optional[str] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Create a visualization for a specific metric using appropriate chart type.

    Parameters:
    -----------
    metric_name : str
        Name of the metric (e.g., 'k_anonymity_distribution', 'generalization_level')
    metric_data : Union[Dict, pd.Series, List]
        The metric data to visualize
    task_dir : Path
        Task directory where visualization will be saved
    field_name : str
        Field name for the visualization
    operation_name : str
        Operation name
    timestamp : str, optional
        Timestamp for consistency
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns
    -------
    Dict[str, Any]
        Updated visualization_paths dictionary.
    """
    try:
        # Determine visualization type based on metric name
        if "distribution" in metric_name:
            viz_type = "distribution"
            filename = generate_visualization_filename(
                field_name, operation_name, viz_type, timestamp
            )
            output_path = task_dir / filename

            # Create histogram for distributions
            if isinstance(metric_data, dict):
                # If dict, assume it's already binned data
                result = create_bar_plot(
                    data=metric_data,
                    output_path=output_path,
                    title=f"{field_name} - {metric_name.replace('_', ' ').title()}",
                    x_label="Value",
                    y_label="Count",
                    theme=theme,
                    backend=backend or "plotly",
                    strict=strict,
                    **kwargs,
                )
            else:
                # Create histogram from raw data
                result = create_histogram(
                    data=metric_data,
                    output_path=output_path,
                    title=f"{field_name} - {metric_name.replace('_', ' ').title()}",
                    bins=calculate_optimal_bins(pd.Series(metric_data)),
                    theme=theme,
                    backend=backend or "plotly",
                    strict=strict,
                    **kwargs,
                )

        elif "level" in metric_name or "score" in metric_name:
            viz_type = "levels"
            filename = generate_visualization_filename(
                field_name, operation_name, viz_type, timestamp
            )
            output_path = task_dir / filename

            # Bar plot for levels/scores
            result = create_bar_plot(
                data=metric_data,
                output_path=output_path,
                title=f"{field_name} - {metric_name.replace('_', ' ').title()}",
                orientation="h",
                theme=theme,
                backend=backend or "plotly",
                strict=strict,
                **kwargs,
            )

        else:
            # Default to bar plot
            viz_type = "metrics"
            filename = generate_visualization_filename(
                field_name, operation_name, viz_type, timestamp
            )
            output_path = task_dir / filename

            result = create_bar_plot(
                data=metric_data,
                output_path=output_path,
                title=f"{field_name} - {metric_name.replace('_', ' ').title()}",
                theme=theme,
                backend=backend or "plotly",
                strict=strict,
                **kwargs,
            )

        if not result.startswith("Error"):
            return output_path
        else:
            logger.error(f"Failed to create visualization: {result}")
            return None

    except Exception as e:
        logger.error(f"Error creating metric visualization: {str(e)}")
        return None


def create_comparison_visualization(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    task_dir: Path,
    field_name: str,
    operation_name: str,
    timestamp: Optional[str] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Optional[Path]:
    """
    Create a before/after comparison visualization.

    Parameters
    ----------
    original_data : pd.Series
        Original column values.
    anonymized_data : pd.Series
        Transformed/anonymized column values.
    task_dir : Path
        Directory to save the plot.
    field_name : str
        Name of the field being visualized.
    operation_name : str
        Name of the operation (e.g., masking, generalization).
    timestamp : Optional[str]
        Timestamp for filename uniqueness.
    theme : Optional[str]
        Visualization theme (for plotly/matplotlib).
    backend : Optional[str]
        Plotting backend, defaults to "plotly".
    strict : bool
        If True, raise exception on plotting error.
    kwargs : Any
        Extra arguments passed to plotter.

    Returns
    -------
    Optional[Path]
        Path to saved plot, or None if failed.
    """
    try:
        # Step 1: Prepare data and infer type (int, float, str, category, etc.)
        comparison_data, data_type = prepare_comparison_data(
            original_data, anonymized_data
        )

        # Step 2: Generate flattened count dict {label: count}
        flat_counts = generate_flat_counts(comparison_data, data_type)

        # Step 3: Output path
        filename = generate_visualization_filename(
            field_name, operation_name, "comparison", timestamp
        )
        output_path = task_dir / filename

        # Step 4: Create bar chart with flat_counts (dict)
        result = create_bar_plot(
            data=flat_counts,
            output_path=output_path,
            title=f"{field_name} - Before/After Comparison",
            x_label=(
                "Source | Value Range" if data_type in ("int", "float") else "Category"
            ),
            y_label="Count",
            theme=theme,
            backend=backend or "plotly",
            strict=strict,
            **kwargs,
        )

        if isinstance(result, str) and not result.startswith("Error"):
            return output_path

        logger.error(
            f"[{field_name}] Failed to create comparison visualization: {result}"
        )
        return None

    except Exception as e:
        logger.exception(
            f"[{field_name}] Exception during comparison visualization: {e}"
        )
        return None


def generate_flat_counts(data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
    """
    Flatten the prepared data into a Dict for bar chart plotting.

    Parameters
    ----------
    data : Dict[str, Any]
        Prepared data from prepare_comparison_data.
    data_type : str
        One of: 'int', 'float', 'categorical'

    Returns
    -------
    Dict[str, Any]
        Flattened dict of form {"Original | 23": 1, "Anonymized | 23–26": 2, ...}
    """
    flat_counts: Dict[str, Any] = {}

    if data_type in ["int", "float"]:
        orig_series = pd.Series(data["Original"])
        anon_series = pd.Series(data["Anonymized"])

        orig_counts = orig_series.value_counts().sort_index().to_dict()
        anon_counts = anon_series.value_counts().sort_index().to_dict()

        all_labels = sorted(set(orig_counts.keys()).union(set(anon_counts.keys())), key=lambda x: str(x))

        for label in all_labels:
            label_str = str(label)

            orig_val = int(orig_counts.get(label, 0))
            anon_val = int(anon_counts.get(label, 0))

            if orig_val > 0:
                flat_counts[f"Original | {label_str}"] = orig_val
            if anon_val > 0:
                flat_counts[f"Anonymized | {label_str}"] = anon_val

    else:
        # Categorical type — pre-counted already
        for source in ["Original", "Anonymized"]:
            for label, value in data.get(source, {}).items():
                if int(value) > 0:
                    label_str = str(label)
                    flat_counts[f"{source} | {label_str}"] = int(value)

    return flat_counts


def create_distribution_visualization(
    data: Union[pd.Series, Dict[str, int]],
    task_dir: Path,
    field_name: str,
    operation_name: str,
    metric_name: str,
    timestamp: Optional[str] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Optional[Path]:
    """
    Create a distribution visualization for metrics like k-anonymity or risk scores.

    Parameters:
    -----------
    data : Union[pd.Series, Dict[str, int]]
        Distribution data (e.g., {k_value: count} or Series of values)
    task_dir : Path
        Task directory
    field_name : str
        Field name
    operation_name : str
        Operation name
    metric_name : str
        Name of the metric being visualized
    timestamp : str, optional
        Timestamp for consistency
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns:
    --------
    Optional[Path]
        Path to visualization or None if failed
    """
    try:
        # Generate filename
        viz_type = f"{metric_name}_distribution"
        filename = generate_visualization_filename(
            field_name, operation_name, viz_type, timestamp
        )
        output_path = task_dir / filename

        # Create the visualization
        if isinstance(data, dict):
            # Already aggregated data
            result = create_bar_plot(
                data=data,
                output_path=output_path,
                title=f"{field_name} - {metric_name.replace('_', ' ').title()} Distribution",
                x_label=metric_name.replace("_", " ").title(),
                y_label="Count",
                sort_by="key" if "k_anonymity" in metric_name else "value",
                theme=theme,
                backend=backend or "plotly",
                strict=strict,
                **kwargs,
            )
        else:
            # Raw data - create histogram
            result = create_histogram(
                data=data,
                output_path=output_path,
                title=f"{field_name} - {metric_name.replace('_', ' ').title()} Distribution",
                x_label=metric_name.replace("_", " ").title(),
                bins=calculate_optimal_bins(data),
                theme=theme,
                backend=backend or "plotly",
                strict=strict,
                **kwargs,
            )

        if not result.startswith("Error"):
            return output_path
        else:
            logger.error(f"Failed to create distribution: {result}")
            return None

    except Exception as e:
        logger.error(f"Error creating distribution visualization: {str(e)}")
        return None


def create_category_distribution_comparison(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    task_dir: Path,
    field_name: str,
    operation_name: str,
    max_categories: int = 15,
    show_percentages: bool = True,
    timestamp: Optional[str] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Optional[Path]:
    """
    Create a specialized comparison visualization for categorical distributions.

    Shows top categories before and after anonymization, with counts and optionally
    percentages. Particularly useful for merge_low_freq and hierarchy strategies.

    Parameters:
    -----------
    original_data : pd.Series
        Original categorical data
    anonymized_data : pd.Series
        Anonymized categorical data
    task_dir : Path
        Task directory
    field_name : str
        Field name
    operation_name : str
        Operation name
    max_categories : int, optional
        Maximum number of categories to show (default: 15)
    show_percentages : bool, optional
        Whether to show percentage distribution (default: True)
    timestamp : str, optional
        Timestamp for consistency
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns:
    --------
    Optional[Path]
        Path to visualization or None if failed
    """
    try:
        # Get value counts
        orig_counts = original_data.value_counts()
        anon_counts = anonymized_data.value_counts()

        # Calculate category reduction
        orig_unique = len(orig_counts)
        anon_unique = len(anon_counts)
        reduction_pct = (
            ((orig_unique - anon_unique) / orig_unique * 100) if orig_unique > 0 else 0
        )

        # Prepare data for visualization
        if show_percentages:
            # Convert to percentages
            orig_total = orig_counts.sum()
            anon_total = anon_counts.sum()

            orig_pct = (orig_counts / orig_total * 100).round(1)
            anon_pct = (anon_counts / anon_total * 100).round(1)

            # Get top categories from anonymized data (to show groupings)
            top_anon_cats = anon_pct.head(max_categories).index

            # Create comparison data
            comparison_data = {}
            for cat in top_anon_cats:
                comparison_data[str(cat)] = {
                    "Original": float(orig_pct.get(cat, 0)),
                    "Anonymized": float(anon_pct.get(cat, 0)),
                }

            y_label = "Percentage (%)"
        else:
            # Use raw counts
            top_anon_cats = anon_counts.head(max_categories).index

            comparison_data = {}
            for cat in top_anon_cats:
                comparison_data[str(cat)] = {
                    "Original": int(orig_counts.get(cat, 0)),
                    "Anonymized": int(anon_counts.get(cat, 0)),
                }

            y_label = "Count"

        # Generate filename
        viz_type = "category_distribution"
        filename = generate_visualization_filename(
            field_name, operation_name, viz_type, timestamp
        )
        output_path = task_dir / filename

        # Create title with reduction info
        title = (
            f"{field_name} - Category Distribution (Reduced by {reduction_pct:.1f}%)"
        )

        # Create the bar plot
        result = create_bar_plot(
            data=comparison_data,
            output_path=output_path,
            title=title,
            x_label="Category",
            y_label=y_label,
            orientation="v",
            theme=theme,
            backend=backend or "plotly",
            strict=strict,
            **kwargs,
        )

        if not result.startswith("Error"):
            logger.info(f"Created category distribution comparison: {output_path}")
            return output_path
        else:
            logger.error(f"Failed to create category distribution: {result}")
            return None

    except Exception as e:
        logger.error(f"Error creating category distribution comparison: {str(e)}")
        return None


def create_hierarchy_sunburst(
    hierarchy_data: Dict[str, Any],
    task_dir: Path,
    field_name: str,
    operation_name: str,
    max_depth: int = 3,
    max_categories: int = DEFAULT_TOP_CATEGORIES_FOR_SUNBURST,
    timestamp: Optional[str] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Optional[Path]:
    """
    Create a sunburst visualization for hierarchical category structure.

    Useful for visualizing generalization hierarchies and understanding
    the impact of hierarchy-based anonymization.

    Parameters:
    -----------
    hierarchy_data : Dict[str, Any]
        Hierarchical structure data. Can be:
        - Nested dict: {"parent": {"child1": count, "child2": count}}
        - Flat dict with parent info: {"child": {"parent": "parent_name", "count": 10}}
    task_dir : Path
        Task directory
    field_name : str
        Field name
    operation_name : str
        Operation name
    max_depth : int, optional
        Maximum hierarchy depth to visualize (default: 3)
    max_categories : int, optional
        Maximum leaf categories to include (default: 50)
    timestamp : str, optional
        Timestamp for consistency
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns:
    --------
    Optional[Path]
        Path to visualization or None if failed
    """
    try:
        # Generate filename
        viz_type = "hierarchy_sunburst"
        filename = generate_visualization_filename(
            field_name, operation_name, viz_type, timestamp
        )
        output_path = task_dir / filename

        # Process hierarchy data based on format
        if not hierarchy_data:
            logger.warning("Empty hierarchy data provided")
            # Create a simple placeholder
            processed_data = {"No Hierarchy": {"No Data": 1}}
        elif _is_nested_hierarchy(hierarchy_data):
            # Already in nested format
            processed_data = _limit_hierarchy_size(
                hierarchy_data, max_categories, max_depth
            )
        else:
            # Convert flat format to nested
            processed_data = _convert_flat_to_nested_hierarchy(
                hierarchy_data, max_categories, max_depth
            )

        # Create the sunburst chart
        result = create_sunburst_chart(
            data=processed_data,
            output_path=output_path,
            title=f"{field_name} - Generalization Hierarchy",
            maxdepth=max_depth,
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )

        if not result.startswith("Error"):
            logger.info(f"Created hierarchy sunburst: {output_path}")
            return output_path
        else:
            logger.error(f"Failed to create hierarchy sunburst: {result}")
            return None

    except Exception as e:
        logger.error(f"Error creating hierarchy sunburst: {str(e)}")
        return None


def _is_nested_hierarchy(data: Dict[str, Any]) -> bool:
    """
    Check if data is in nested hierarchy format.

    Parameters:
    -----------
    data : Dict[str, Any]
        Data to check

    Returns:
    --------
    bool
        True if nested hierarchy format
    """
    if not data:
        return False

    # Check if any value is a dict (indicating nesting)
    for value in data.values():
        if isinstance(value, dict):
            return True
    return False


def _limit_hierarchy_size(
    hierarchy: Dict[str, Any],
    max_categories: int,
    max_depth: int,
    current_depth: int = 0,
) -> Dict[str, Any]:
    """
    Limit the size and depth of a hierarchy for visualization.

    Parameters:
    -----------
    hierarchy : Dict[str, Any]
        Nested hierarchy data
    max_categories : int
        Maximum leaf nodes to keep
    max_depth : int
        Maximum depth
    current_depth : int
        Current recursion depth

    Returns:
    --------
    Dict[str, Any]
        Limited hierarchy
    """
    if current_depth >= max_depth:
        return {}

    limited = {}
    category_count = 0

    # Sort by size (if values are dicts, count their children)
    sorted_items = sorted(
        hierarchy.items(),
        key=lambda x: len(x[1]) if isinstance(x[1], dict) else x[1],
        reverse=True,
    )

    for key, value in sorted_items:
        if category_count >= max_categories:
            break

        if isinstance(value, dict):
            # Recursive case
            sub_hierarchy = _limit_hierarchy_size(
                value, max_categories - category_count, max_depth, current_depth + 1
            )
            if sub_hierarchy:
                limited[key] = sub_hierarchy
                category_count += _count_leaves(sub_hierarchy)
        else:
            # Leaf node
            limited[key] = value
            category_count += 1

    return limited


def _count_leaves(hierarchy: Dict[str, Any]) -> int:
    """
    Count leaf nodes in a hierarchy.

    Parameters:
    -----------
    hierarchy : Dict[str, Any]
        Hierarchy to count

    Returns:
    --------
    int
        Number of leaf nodes
    """
    count = 0
    for value in hierarchy.values():
        if isinstance(value, dict):
            count += _count_leaves(value)
        else:
            count += 1
    return count


def _convert_flat_to_nested_hierarchy(
    flat_data: Dict[str, Any], max_categories: int, max_depth: int
) -> Dict[str, Dict[str, int]]:
    """
    Convert flat or tree-based hierarchy data to nested {parent: {child: count}} format.

    Parameters:
    -----------
    flat_data : Dict[str, Any]
        Either:
        - Flat format: {"child": {"parent": "parent_name", "count": 10}}
        - Tree format: {"name": ..., "children": [...]}
    max_categories : int
        Limit number of top entries
    max_depth : int
        Not used directly (reserved for future depth limiting)

    Returns:
    --------
    Dict[str, Dict[str, int]]
        Nested dictionary suitable for visualization input
    """
    nested = {}

    # Case 1: Handle tree-like input with 'name' + 'children'
    if isinstance(flat_data, dict) and "name" in flat_data and "children" in flat_data:

        def traverse(node: Dict[str, Any]):
            parent_name = node.get("name")
            children = node.get("children", [])
            if not children:
                return
            for child in children:
                child_name = child.get("name")
                count = child.get("value", 1)
                if parent_name not in nested:
                    nested[parent_name] = {}
                nested[parent_name][child_name] = count
                # Recurse if deeper
                if "children" in child:
                    traverse(child)

        traverse(flat_data)
        return nested

    # Case 2: Flat input
    if not isinstance(flat_data, dict):
        raise ValueError("Input hierarchy data must be a dictionary.")

    # Sanitize and sort flat data
    sorted_items = sorted(
        ((k, v) for k, v in flat_data.items() if isinstance(v, dict) and "parent" in v),
        key=lambda x: x[1].get("count", 0),
        reverse=True,
    )[:max_categories]

    for child, info in sorted_items:
        parent = info.get("parent", "Root")
        count = info.get("count", 1)
        if parent not in nested:
            nested[parent] = {}
        nested[parent][child] = count

    return nested


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


def create_metrics_overview_visualization(
    metrics: Dict[str, Any],
    task_dir: Path,
    field_name: str,
    operation_name: str,
    timestamp: Optional[str] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Create an overview visualization for categorical generalization metrics using core visualization functions.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Categorical generalization metrics
    task_dir : Path
        Task directory
    field_name : str
        Field name
    operation_name : str
        Operation name
    timestamp : str, optional
        Timestamp for consistency
    theme : Optional[str]
        Visualization theme (e.g. "light", "dark").
    backend : Optional[str]
        Visualization backend (e.g. "matplotlib", "plotly").
    strict : bool
        Whether to enforce strict type checking.
    **kwargs : Any
        Additional keyword arguments for visualization functions.

    Returns:
    --------
    Optional[List[Path]]
        List of paths to visualization files or None if failed
    """
    try:
        output_paths = {}

        # Privacy metrics overview (bar plot)
        if "privacy_metric_overview" in metrics:
            privacy_data = metrics["privacy_metric_overview"]
            if isinstance(privacy_data, dict):
                labels = [
                    "Min K",
                    "Avg Suppression %",
                    "Avg Coverage %",
                    "Avg Generalization %",
                ]
                values = [
                    privacy_data.get("min_k_overall", 0),
                    privacy_data.get("avg_suppression_rate", 0) * 100,
                    privacy_data.get("avg_coverage", 0) * 100,
                    privacy_data.get("avg_generalization_level", 0) * 100,
                ]
                privacy_dict = dict(zip(labels, values))
                filename = generate_visualization_filename(
                    field_name, operation_name, "privacy_metric_overview", timestamp
                )
                output_path = task_dir / filename
                result = create_bar_plot(
                    data=privacy_dict,
                    output_path=output_path,
                    title=f"{field_name} - Privacy Metrics Overview",
                    y_label="Value",
                    theme=theme,
                    backend=backend or "plotly",
                    strict=strict,
                    **kwargs,
                )
                if not str(result).startswith("Error"):
                    output_paths["privacy_metric_overview"] = output_path
            else:
                logger.warning(
                    f"[PRIVACY] Skipped privacy summary chart: {privacy_data}"
                )

        # Information loss summary (bar plot)
        if "info_loss_summary" in metrics:
            info_loss = metrics["info_loss_summary"]
            labels = ["Precision Loss", "Entropy Loss", "Category Reduction"]
            values = [
                info_loss.get("avg_precision_loss", 0) * 100,
                info_loss.get("avg_entropy_loss", 0) * 100,
                info_loss.get("avg_category_reduction", 0) * 100,
            ]
            info_dict = dict(zip(labels, values))
            filename = generate_visualization_filename(
                field_name, operation_name, "info_loss_summary", timestamp
            )
            output_path = task_dir / filename
            result = create_bar_plot(
                data=info_dict,
                output_path=output_path,
                title=f"{field_name} - Information Loss Breakdown",
                y_label="Percentage (%)",
                theme=theme,
                backend=backend or "plotly",
                strict=strict,
                **kwargs,
            )
            if not str(result).startswith("Error"):
                output_paths["info_loss_summary"] = output_path

        # Privacy metrics over batches (histogram)
        if "privacy_metrics" in metrics and isinstance(
            metrics["privacy_metrics"], dict
        ):
            history = metrics["privacy_metrics"]

            # Normalize values to lists
            min_k_values = history.get("min_k", [])
            if not isinstance(min_k_values, list):
                min_k_values = [min_k_values]

            coverage_values = history.get("total_coverage", [])
            if not isinstance(coverage_values, list):
                coverage_values = [coverage_values]
            coverage_values = [v * 100 for v in coverage_values]

            # Only create plots if there are multiple data points (at least 2)
            if len(min_k_values) > 1:
                # Min K over batches
                filename = generate_visualization_filename(
                    field_name, operation_name, "min_k_over_batches", timestamp
                )
                output_path = task_dir / filename
                result = create_histogram(
                    data=min_k_values,
                    output_path=output_path,
                    title=f"{field_name} - Min K Across Batches",
                    x_label="Batch Number",
                    y_label="Min K Value",
                    theme=theme,
                    backend=backend or "plotly",
                    strict=strict,
                    **kwargs,
                )
                if not str(result).startswith("Error"):
                    output_paths["min_k_over_batches"] = output_path

                # Coverage over batches
                filename = generate_visualization_filename(
                    field_name, operation_name, "coverage_over", timestamp
                )
                output_path = task_dir / filename
                result = create_histogram(
                    data=coverage_values,
                    output_path=output_path,
                    title=f"{field_name} - Coverage % Across Batches",
                    x_label="Batch Number",
                    y_label="Coverage %",
                    theme=theme,
                    backend=backend or "plotly",
                    strict=strict,
                    **kwargs,
                )
                if not str(result).startswith("Error"):
                    output_paths["coverage_over_batches"] = output_path

        # Performance summary (bar plot)
        if "performance_summary" in metrics:
            perf = metrics["performance_summary"]
            labels = [
                "Total Records",
                "Total Duration (s)",
                "Avg Speed (records/sec)",
            ]
            values = [
                perf.get("total_records_processed", 0),
                perf.get("total_duration_seconds", 0),
                perf.get("avg_records_per_second", 0),
            ]
            perf_dict = dict(zip(labels, values))
            filename = generate_visualization_filename(
                field_name, operation_name, "performance_summary", timestamp
            )
            output_path = task_dir / filename
            result = create_bar_plot(
                data=perf_dict,
                output_path=output_path,
                title=f"{field_name} - Performance Summary",
                y_label="Value",
                theme=theme,
                backend=backend or "plotly",
                strict=strict,
                **kwargs,
            )
            if not str(result).startswith("Error"):
                output_paths["performance_summary"] = output_path

        if output_paths:
            logger.info(f"Created metrics summary visualizations: {output_paths}")
            return output_paths
        else:
            logger.error("No metrics summary visualizations were created.")
            return None

    except Exception as e:
        logger.error(f"Failed to create metrics summary visualization: {e}")
        return None
