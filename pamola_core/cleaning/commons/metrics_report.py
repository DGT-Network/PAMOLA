"""
This module provides functionality to generate a comprehensive markdown report
from metrics collected during data processing steps such as data cleaning,
transformation, or anonymization.

The generated report includes summaries of original and generated data,
null analysis, transformation details, distribution similarity, performance
statistics, and dictionary metadata if available.
"""

from pathlib import Path
from typing import Dict, Any, Union
from pamola_core.common.constants import Constants

def generate_metrics_report(
    metrics: Dict[str, Any],
    output_path: Union[str, Path] = None,
    op_type: str = None,
    field_name: str = None,
) -> str:
    """
    Generates a markdown report from metrics data.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing various metrics (original, generated, performance, etc.)
    output_path : str or Path, optional
        If provided, writes the output report to file.
    op_type : str, optional
        Type of processing operation (e.g. 'cleaning', 'anonymization').
    field_name : str, optional
        Name of the field this report pertains to.

    Returns:
    --------
    str
        The generated markdown report as a string.
    """
    report = f"""# Data Cleaning Metrics Report

    ## Summary

    {_format_field_metrics(metrics)}
    {_format_original_data(metrics)}
    {_format_generated_data(metrics)}
    {_format_transformation(metrics)}
    {_format_distribution(metrics)}
    {_format_performance(metrics)}
    {_format_dictionary(metrics)}
    """

    if output_path:
        output_path = Path(output_path)
        if op_type and field_name and output_path.is_dir():
            output_path = output_path / f"{op_type}_{field_name}_metrics_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report

def _format_field_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format the field-level null metrics section.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing 'field_metrics' key with field-related stats.

    Returns:
    --------
    str
        Markdown-formatted string for field metrics section.
    """
    field = metrics.get("field_metrics", {})
    return (
        f"""### Field Data
    - Field: {field.get('field_name', 'N/A')}
    - Nulls (Original): {field.get('total_nulls_original', 'N/A')}
    - Nulls (Generated): {field.get('total_nulls_generated', 'N/A')}\n
    """
        if field
        else ""
    )


def _format_original_data(metrics: Dict[str, Any]) -> str:
    """
    Format original dataset statistics and null analysis.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing 'original_data' key with related statistics.

    Returns:
    --------
    str
        Markdown-formatted string for original data section.
    """
    orig = metrics.get("original_data", {})
    length = orig.get("length_stats", {})
    null_pattern = orig.get("null_pattern", {})

    null_pattern_str = (
        "\n".join([f"  - {k}: {v}" for k, v in null_pattern.items()])
        if null_pattern
        else ""
    )

    return (
        f"""### Original Data
    - Total records: {orig.get('total_records', 'N/A')}
    - Unique values: {orig.get('unique_values', 'N/A')}
    - Length: min={length.get('min', 'N/A')}, max={length.get('max', 'N/A')}, mean={length.get('mean', 0):.2f}, median={length.get('median', 0):.2f}

    ### Null Analysis For Original Data
    - Null count: {orig.get('null_count', 'N/A')}
    - Null ratio: {orig.get('null_ratio', 'N/A')}
    - Rows removed: {orig.get('removed_rows', 'N/A')}
    - Field has null: {orig.get('field_has_null', 'N/A')}
    - Null pattern:
    {null_pattern_str}
    """
        if orig
        else ""
    )


def _format_generated_data(metrics: Dict[str, Any]) -> str:
    """
    Format generated dataset statistics.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing 'generated_data' key with related statistics.

    Returns:
    --------
    str
        Markdown-formatted string for generated data section.
    """
    gen = metrics.get("generated_data", {})
    length = gen.get("length_stats", {})
    null_pattern = gen.get("null_pattern", {})

    null_pattern_str = (
        "\n".join([f"  - {k}: {v}" for k, v in null_pattern.items()])
        if null_pattern
        else ""
    )

    return (
        f"""### Generated Data
    - Total records: {gen.get('total_records', 'N/A')}
    - Unique values: {gen.get('unique_values', 'N/A')}
    - Length: min={length.get('min', 'N/A')}, max={length.get('max', 'N/A')}, mean={length.get('mean', 0):.2f}, median={length.get('median', 0):.2f}

    ### Null Analysis For Generated Data
    - Null count: {gen.get('null_count', 'N/A')}
    - Null ratio: {gen.get('null_ratio', 'N/A')}
    - Rows removed: {gen.get('removed_rows', 'N/A')}
    - Field has null: {gen.get('field_has_null', 'N/A')}
    - Null pattern:
    {null_pattern_str}
    """
        if gen
        else ""
    )


def _format_transformation(metrics: Dict[str, Any]) -> str:
    """
    Format transformation strategy and metrics.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing 'transformation_metrics' key with transformation stats.

    Returns:
    --------
    str
        Markdown-formatted string for transformation section.
    """
    trans = metrics.get("transformation_metrics", {})
    return (
        f"""### Transformation
    - Replacement strategy: {trans.get('replacement_strategy', 'N/A')}
    - Total replacements: {trans.get('total_replacements', 'N/A')}
    - Null values replaced: {trans.get('null_values_replaced', 'N/A')}
    - Mapping collisions: {trans.get('mapping_collisions', 'N/A')}
    - Reversibility rate: {trans.get('reversibility_rate', 0):.2%}
    """
        if trans
        else ""
    )


def _format_distribution(metrics: Dict[str, Any]) -> str:
    """
    Format distribution comparison and statistical similarity metrics.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing 'distribution_metrics' key with distribution statistics.

    Returns:
    --------
    str
        Markdown-formatted string for distribution section.
    """
    dist = metrics.get("distribution_metrics", {})
    if not dist:
        return ""

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    dist_str = "\n".join(
        [f"- {label}: {fmt(dist[key])}" for key, label in Constants.DISTRIBUTION_LABELS.items() if key in dist]
    )

    return f"""### Distribution Metrics
    {dist_str}
    """


def _format_performance(metrics: Dict[str, Any]) -> str:
    """
    Format performance timing and throughput stats.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing 'performance' key with performance metrics.

    Returns:
    --------
    str
        Markdown-formatted string for performance section.
    """
    perf = metrics.get("performance", {})
    return (
        f"""### Performance
    - Generation time: {perf.get('generation_time', 0):.2f} seconds
    - Records per second: {perf.get('records_per_second', 'N/A')}
    - Records processed: {perf.get('records_processed', 'N/A')}
    - Memory usage: {perf.get('memory_usage_mb', 0):.2f} MB
    """
        if perf
        else "### Performance\nNo performance metrics available.\n"
    )


def _format_dictionary(metrics: Dict[str, Any]) -> str:
    """
    Format dictionary statistics if available.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing 'dictionary_metrics' key with metadata.

    Returns:
    --------
    str
        Markdown-formatted string for dictionary section.
    """
    dict_metrics = metrics.get("dictionary_metrics", {})
    if not dict_metrics:
        return "### Dictionary\nNo dictionary metrics available.\n"
    return f"""### Dictionary
    - Total entries: {dict_metrics.get('total_dictionary_entries', 'N/A')}
    - Language variants: {', '.join(dict_metrics.get('language_variants', []))}
    - Last update: {dict_metrics.get('last_update', 'N/A')}
    """
