"""
Metrics collection and analysis for fake data generation.

This module provides tools for measuring the quality and statistical
properties of generated fake data, including distribution comparison,
format validation, and performance assessment.
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from pamola_core.utils.visualization import (
    create_bar_plot,
    create_combined_chart,
    create_pie_chart
)

# Configure logger
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Base class for metrics collectors."""

    def collect_metrics(self,
                        orig_data: Optional[pd.Series] = None,
                        gen_data: Optional[pd.Series] = None,
                        field_name: str = None,
                        operation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Collects comprehensive metrics about original and generated data.

        Parameters:
        -----------
        orig_data : pd.Series, optional
            Original data series
        gen_data : pd.Series, optional
            Generated data series
        field_name : str, optional
            Name of the field being processed
        operation_params : Dict[str, Any], optional
            Parameters used in the operation

        Returns:
        --------
        Dict[str, Any]
            Dictionary with collected metrics
        """
        metrics = {}

        # Original data metrics
        if orig_data is not None:
            metrics["original_data"] = self.collect_data_stats(orig_data)

        # Generated data metrics
        if gen_data is not None:
            metrics["generated_data"] = self.collect_data_stats(gen_data)

        # Transformation metrics
        if orig_data is not None and gen_data is not None:
            metrics["transformation_metrics"] = self.collect_transformation_metrics(
                orig_data, gen_data, operation_params
            )

        # Performance metrics (can be added by the operation)
        metrics["performance"] = {}

        # Dictionary metrics (can be added by the operation)
        metrics["dictionary_metrics"] = {}

        return metrics

    def collect_data_stats(self, data: pd.Series) -> Dict[str, Any]:
        """
        Collects basic statistical metrics for a data series.

        Parameters:
        -----------
        data : pd.Series
            Data series to analyze

        Returns:
        --------
        Dict[str, Any]
            Dictionary with basic statistical metrics
        """
        # Filter out null values for analysis
        non_null_data = data.dropna()

        # Basic counts
        total_records = len(data)
        unique_values = len(non_null_data.unique())

        # Value distribution (top 20 most common values)
        value_counts = non_null_data.value_counts(normalize=True)
        top_values = value_counts.head(20).to_dict()

        # Length statistics for string data
        length_stats = {}
        if non_null_data.dtype == object:
            try:
                # Convert to string and get lengths
                lengths = non_null_data.astype(str).str.len()

                length_stats = {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": float(lengths.mean()),
                    "median": float(lengths.median())
                }
            except Exception as e:
                logger.warning(f"Could not calculate length statistics: {str(e)}")
                length_stats = {"error": str(e)}

        return {
            "total_records": total_records,
            "unique_values": unique_values,
            "value_distribution": top_values,
            "length_stats": length_stats
        }

    @staticmethod
    def collect_transformation_metrics(
            orig_data: pd.Series,
            gen_data: pd.Series,
            operation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Collect comprehensive metrics about data transformation.

        Parameters:
        -----------
        orig_data : pd.Series
            Original data series
        gen_data : pd.Series
            Generated data series
        operation_params : Dict[str, Any], optional
            Parameters used in the transformation operation

        Returns:
        --------
        Dict[str, Any]
            Transformation metrics
        """
        # Ensure operation_params is a dictionary
        operation_params = operation_params or {}

        try:
            # Null value masks
            orig_null_mask = orig_data.isna().values
            gen_null_mask = gen_data.isna().values

            # Null value replacements
            # Count where original was null but generated is not
            null_values_replaced = np.sum(~orig_null_mask & gen_null_mask)

            # Prepare for value comparison
            # Find indices where both original and generated are non-null
            comparison_mask = ~orig_null_mask & ~gen_null_mask

            # Calculate value replacements
            value_replacements = 0
            if np.any(comparison_mask):
                # Compare values where both are non-null
                value_changes = (
                        orig_data.values[comparison_mask] !=
                        gen_data.values[comparison_mask]
                )
                value_replacements = np.sum(value_changes)

            # Total replacements include both null replacements and value changes
            total_replacements = value_replacements + null_values_replaced

            # Extract replacement strategy
            replacement_strategy = operation_params.get(
                "consistency_mechanism", "unknown"
            )

            # Mapping collisions
            mapping_collisions = 0
            try:
                if (replacement_strategy == "mapping" and
                        "mapping_store" in operation_params):
                    mapping_store = operation_params.get("mapping_store")
                    if hasattr(mapping_store, "get_collision_count"):
                        field_name = operation_params.get("field_name", "unknown")
                        mapping_collisions = mapping_store.get_collision_count(field_name)
            except Exception:
                mapping_collisions = 0

            # Reversibility rate calculation
            reversibility_rate = 0.0
            try:
                # Filter out null values
                gen_data_clean = gen_data[~gen_null_mask]

                if len(gen_data_clean) > 0:
                    # Count values appearing only once
                    value_counts = gen_data_clean.value_counts()
                    unique_values = np.sum(value_counts == 1)

                    # Calculate reversibility rate
                    reversibility_rate = unique_values / len(gen_data_clean)
            except Exception:
                reversibility_rate = 0.0

            # Return metrics
            return {
                "null_values_replaced": int(null_values_replaced),
                "total_replacements": int(total_replacements),
                "replacement_strategy": replacement_strategy,
                "mapping_collisions": int(mapping_collisions),
                "reversibility_rate": float(reversibility_rate)
            }

        except Exception:
            # Fallback to default metrics
            return {
                "null_values_replaced": 0,
                "total_replacements": 0,
                "replacement_strategy": "unknown",
                "mapping_collisions": 0,
                "reversibility_rate": 0.0
            }

    def compare_distributions(self,
                              orig_data: pd.Series,
                              gen_data: pd.Series) -> Dict[str, float]:
        """
        Compares distributions in original and synthetic data.

        Parameters:
        -----------
        orig_data : pd.Series
            Original data series
        gen_data : pd.Series
            Generated data series

        Returns:
        --------
        Dict[str, float]
            Dictionary with distribution comparison metrics
        """
        # Get value counts
        orig_counts = orig_data.value_counts(normalize=True)
        gen_counts = gen_data.value_counts(normalize=True)

        # Merge indexes
        all_values = set(orig_counts.index) | set(gen_counts.index)

        # Calculate Jensen-Shannon divergence
        kl_div = 0.0
        try:
            # Fill missing values with 0
            orig_probs = np.array([orig_counts.get(val, 0) for val in all_values])
            gen_probs = np.array([gen_counts.get(val, 0) for val in all_values])

            # Add a small value to avoid zeros
            orig_probs = orig_probs + 1e-10
            gen_probs = gen_probs + 1e-10

            # Normalize
            orig_probs = orig_probs / orig_probs.sum()
            gen_probs = gen_probs / gen_probs.sum()

            # Calculate midpoint distribution
            mid_probs = (orig_probs + gen_probs) / 2

            # Calculate KL divergence for orig vs mid and gen vs mid
            kl_orig_mid = np.sum(orig_probs * np.log(orig_probs / mid_probs))
            kl_gen_mid = np.sum(gen_probs * np.log(gen_probs / mid_probs))

            # Calculate Jensen-Shannon divergence
            kl_div = (kl_orig_mid + kl_gen_mid) / 2
        except Exception as e:
            logger.warning(f"Error calculating Jensen-Shannon divergence: {str(e)}")

        # Calculate uniqueness preservation
        uniqueness_preservation = 0.0
        try:
            orig_unique_ratio = len(orig_data.unique()) / len(orig_data)
            gen_unique_ratio = len(gen_data.unique()) / len(gen_data)
            uniqueness_preservation = gen_unique_ratio / orig_unique_ratio
        except Exception as e:
            logger.warning(f"Error calculating uniqueness preservation: {str(e)}")

        return {
            "distribution_similarity_score": 1.0 - min(1.0, float(kl_div)),
            "uniqueness_preservation": float(uniqueness_preservation)
        }

    def visualize_metrics(self,
                          metrics: Dict[str, Any],
                          field_name: str,
                          output_dir: Union[str, Path],
                          op_type: str,
                          **kwargs) -> Dict[str, Path]:
        """
        Creates visualizations for metrics data.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Metrics data to visualize
        field_name : str
            Name of the field being processed
        output_dir : str or Path
            Directory to save visualizations
        op_type : str
            Type of operation

        Returns:
        --------
        Dict[str, Path]
            Dictionary mapping visualization names to file paths
        """
        output_dir = Path(output_dir)
        visualizations = {}

        # Visualize value distributions if available
        if ("original_data" in metrics and "value_distribution" in metrics["original_data"] and
                "generated_data" in metrics and "value_distribution" in metrics["generated_data"]):

            orig_dist = metrics["original_data"]["value_distribution"]
            gen_dist = metrics["generated_data"]["value_distribution"]

            # Only visualize if there's data
            if orig_dist and gen_dist:
                # Find common top values
                common_keys = set(orig_dist.keys()) | set(gen_dist.keys())
                top_n = min(10, len(common_keys))

                # Sort by original data frequency
                common_keys = sorted(common_keys,
                                     key=lambda k: orig_dist.get(k, 0),
                                     reverse=True)[:top_n]

                # Create data for combined chart
                orig_values = [orig_dist.get(k, 0) for k in common_keys]
                gen_values = [gen_dist.get(k, 0) for k in common_keys]

                # Create combined chart for value distribution
                dist_path = output_dir / f"{op_type}_{field_name}_value_distribution.png"
                try:
                    vis_path = create_combined_chart(
                        primary_data=dict(zip(common_keys, orig_values)),
                        secondary_data=dict(zip(common_keys, gen_values)),
                        output_path=dist_path,
                        title=f"Value Distribution Comparison for {field_name}",
                        primary_type="bar",
                        secondary_type="bar",
                        x_label="Value",
                        primary_y_label="Original Frequency",
                        secondary_y_label="Generated Frequency",
                        primary_color="royalblue",
                        secondary_color="crimson",
                        **kwargs
                    )
                    if not vis_path.startswith("Error"):
                        visualizations["value_distribution"] = Path(vis_path)
                    else:    
                        logger.warning(f"Error creating replacement rate visualization: {str(e)}")                                        
                except Exception as e:
                    logger.warning(f"Error creating value distribution visualization: {str(e)}")

        # Visualize length distribution for string data
        if ("original_data" in metrics and "length_stats" in metrics["original_data"] and
                "generated_data" in metrics and "length_stats" in metrics["generated_data"]):

            orig_stats = metrics["original_data"]["length_stats"]
            gen_stats = metrics["generated_data"]["length_stats"]

            # Only visualize if there's length data
            if orig_stats and gen_stats and "min" in orig_stats and "min" in gen_stats:
                # Create data for combined chart
                stats_keys = ["min", "max", "mean", "median"]
                orig_stats_values = [orig_stats.get(k, 0) for k in stats_keys]
                gen_stats_values = [gen_stats.get(k, 0) for k in stats_keys]

                # Create combined chart for length stats
                length_path = output_dir / f"{op_type}_{field_name}_length_stats.png"
                try:
                    vis_path = create_bar_plot(
                        data={
                            "Original": dict(zip(stats_keys, orig_stats_values)),
                            "Generated": dict(zip(stats_keys, gen_stats_values))
                        },
                        output_path=length_path,
                        title=f"Length Statistics for {field_name}",
                        x_label="Statistic",
                        y_label="Value",
                        **kwargs
                    )
                    if not vis_path.startswith("Error"):
                        visualizations["length_stats"] = Path(vis_path)
                    else:    
                        logger.warning(f"Error creating replacement rate visualization: {str(e)}")
                    
                except Exception as e:
                    logger.warning(f"Error creating length stats visualization: {str(e)}")

        # Visualize transformation metrics
        if "transformation_metrics" in metrics:
            trans_metrics = metrics["transformation_metrics"]

            # Create pie chart for replacements
            if "total_replacements" in trans_metrics:
                total = metrics["original_data"]["total_records"]
                replaced = trans_metrics["total_replacements"]
                preserved = total - replaced

                pie_data = {
                    "Replaced Values": replaced,
                    "Preserved Values": preserved
                }

                replace_path = output_dir / f"{op_type}_{field_name}_replacements.png"
                try:
                    vis_path = create_pie_chart(
                        data=pie_data,
                        output_path=replace_path,
                        title=f"Value Replacement Rate for {field_name}",
                        show_percentages=True,
                        **kwargs
                    )
                    if not vis_path.startswith("Error"):
                        visualizations["replacement_rate"] = Path(vis_path)
                    else:    
                        logger.warning(f"Error creating replacement rate visualization: {str(e)}")                    
                except Exception as e:
                    logger.warning(f"Error creating replacement rate visualization: {str(e)}")

        # Return all visualization paths
        return visualizations


def create_metrics_collector() -> MetricsCollector:
    """
    Factory function to create a metrics collector.

    Returns:
    --------
    MetricsCollector
        Instance of a metrics collector
    """
    return MetricsCollector()


def generate_metrics_report(metrics: Dict[str, Any], output_path: Union[str, Path] = None, op_type: str = None,
                            field_name: str = None) -> str:
    """
    Generates a markdown report from metrics data.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics data to include in the report
    output_path : str or Path, optional
        Path to save the report. If None, report is returned as a string.
    op_type : str, optional
        Type of operation
    field_name : str, optional
        Name of the field being processed

    Returns:
    --------
    str
        Markdown report as a string
    """
    report = []

    # Add report header
    report.append("# Fake Data Generation Metrics Report")
    report.append("")

    # Add summary section
    report.append("## Summary")
    report.append("")

    # Add original data summary
    if "original_data" in metrics:
        orig = metrics["original_data"]
        report.append("### Original Data")
        report.append("")
        report.append(f"- Total records: {orig.get('total_records', 'N/A')}")
        report.append(f"- Unique values: {orig.get('unique_values', 'N/A')}")

        # Add length stats if available
        if "length_stats" in orig and orig["length_stats"]:
            length = orig["length_stats"]
            report.append(f"- Length: min={length.get('min', 'N/A')}, max={length.get('max', 'N/A')}, " +
                          f"mean={length.get('mean', 'N/A'):.2f}, median={length.get('median', 'N/A'):.2f}")
        report.append("")

    # Add generated data summary
    if "generated_data" in metrics:
        gen = metrics["generated_data"]
        report.append("### Generated Data")
        report.append("")
        report.append(f"- Total records: {gen.get('total_records', 'N/A')}")
        report.append(f"- Unique values: {gen.get('unique_values', 'N/A')}")

        # Add length stats if available
        if "length_stats" in gen and gen["length_stats"]:
            length = gen["length_stats"]
            report.append(f"- Length: min={length.get('min', 'N/A')}, max={length.get('max', 'N/A')}, " +
                          f"mean={length.get('mean', 'N/A'):.2f}, median={length.get('median', 'N/A'):.2f}")
        report.append("")

    # Add transformation metrics
    if "transformation_metrics" in metrics:
        trans = metrics["transformation_metrics"]
        report.append("### Transformation")
        report.append("")
        report.append(f"- Replacement strategy: {trans.get('replacement_strategy', 'N/A')}")
        report.append(f"- Total replacements: {trans.get('total_replacements', 'N/A')}")
        report.append(f"- Null values replaced: {trans.get('null_values_replaced', 'N/A')}")
        report.append(f"- Mapping collisions: {trans.get('mapping_collisions', 'N/A')}")
        report.append(f"- Reversibility rate: {trans.get('reversibility_rate', 'N/A'):.2%}")
        report.append("")

    # Add performance metrics
    if "performance" in metrics:
        perf = metrics["performance"]
        report.append("### Performance")
        report.append("")
        if perf:
            report.append(f"- Generation time: {perf.get('generation_time', 'N/A'):.2f} seconds")
            report.append(f"- Records per second: {perf.get('records_per_second', 'N/A')}")
            report.append(f"- Memory usage: {perf.get('memory_usage_mb', 'N/A'):.2f} MB")
        else:
            report.append("No performance metrics available.")
        report.append("")

    # Add dictionary metrics
    if "dictionary_metrics" in metrics:
        dict_metrics = metrics["dictionary_metrics"]
        report.append("### Dictionary")
        report.append("")
        if dict_metrics:
            report.append(f"- Total entries: {dict_metrics.get('total_dictionary_entries', 'N/A')}")
            report.append(f"- Language variants: {', '.join(dict_metrics.get('language_variants', []))}")
            report.append(f"- Last update: {dict_metrics.get('last_update', 'N/A')}")
        else:
            report.append("No dictionary metrics available.")
        report.append("")

    # Join report lines
    report_text = "\n".join(report)

    # Save report if output path is provided
    if output_path:
        output_path = Path(output_path)

        # If op_type and field_name are provided, use them in the filename
        if op_type and field_name and output_path.is_dir():
            output_path = output_path / f"{op_type}_{field_name}_metrics_report.md"

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(report_text)

    return report_text