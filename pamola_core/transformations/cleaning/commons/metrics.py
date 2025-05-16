"""
Metrics collection and analysis for clean data generation.

This module provides tools for measuring the quality and statistical
properties of generated clean data, including distribution comparison,
format validation, and performance assessment.
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from plotly import graph_objects as go

from pamola_core.utils.visualization import (
    create_bar_plot,
    create_combined_chart,
    create_pie_chart,
)

# Configure logger
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Base class for metrics collectors."""

    def collect_metrics(
        self,
        orig_data: Optional[pd.Series] = None,
        gen_data: Optional[pd.Series] = None,
        field_name: str = None,
        operation_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
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

        if field_name is not None:
            metrics["field_metrics"] = {
                "field_name": field_name,
                "total_nulls_original": (
                    int(orig_data.isna().sum()) if orig_data is not None else 0
                ),
                "total_nulls_generated": (
                    int(gen_data.isna().sum()) if gen_data is not None else 0
                ),
            }

        metrics["operation_params"] = operation_params or {}

        # Performance metrics (if available in operation_params)
        duration = (
            operation_params.get("execution_time_sec") if operation_params else None
        )
        if duration and orig_data is not None:
            total_records = len(orig_data)
            records_per_second = total_records / duration if duration > 0 else None
            mem_usage = operation_params.get("mem_usage") if operation_params else None
            metrics["performance"] = {
                "generation_time": round(duration, 4),
                "records_processed": total_records,
                "records_per_second": (
                    round(records_per_second, 2) if records_per_second else None
                ),
                "memory_usage_mb": round(mem_usage, 2) if mem_usage > 0 else None,
            }
        else:
            metrics["performance"] = {}

        # Dictionary metrics (if available in operation_params)
        if operation_params and "dictionary_info" in operation_params:
            dict_info = operation_params["dictionary_info"]
            metrics["dictionary_metrics"] = {
                "total_dictionary_entries": dict_info.get("total_entries"),
                "language_variants": dict_info.get("languages", []),
                "last_update": dict_info.get("last_update"),
            }
        else:
            metrics["dictionary_metrics"] = {}

        return metrics

    def collect_data_stats(self, data: pd.Series) -> Dict[str, Any]:
        """
        Collects basic statistical metrics for a data series, including null analysis and patterns.

        Parameters:
        -----------
        data : pd.Series
            Data series to analyze

        Returns:
        --------
        Dict[str, Any]
            Dictionary with basic statistical metrics
        """
        # Initial stats
        total_records = len(data)
        null_count = data.isna().sum()
        non_null_count = total_records - null_count
        null_ratio = null_count / total_records if total_records > 0 else 0
        removed_rows = null_count

        # Basic stats before removal (including nulls)
        stats_before_removal = {}
        try:
            if pd.api.types.is_numeric_dtype(data):
                stats_before_removal = {
                    "min": float(data.min(skipna=True)),
                    "max": float(data.max(skipna=True)),
                    "mean": float(data.mean(skipna=True)),
                    "median": float(data.median(skipna=True)),
                    "std": float(data.std(skipna=True)),
                }
        except Exception as e:
            logger.warning(f"Could not compute stats before removal: {str(e)}")

        # Filter out null values
        non_null_data = data.dropna()

        # Stats after removal
        stats_after_removal = {}
        try:
            if pd.api.types.is_numeric_dtype(non_null_data):
                stats_after_removal = {
                    "min": float(non_null_data.min()),
                    "max": float(non_null_data.max()),
                    "mean": float(non_null_data.mean()),
                    "median": float(non_null_data.median()),
                    "std": float(non_null_data.std()),
                }
        except Exception as e:
            logger.warning(f"Could not compute stats after removal: {str(e)}")

        # Value distribution
        unique_values = len(non_null_data.unique())
        value_counts = non_null_data.value_counts(normalize=True)
        top_values = value_counts.head(20).to_dict()

        # Length stats
        length_stats = {}
        try:
            if pd.api.types.is_object_dtype(non_null_data):
                lengths = non_null_data.astype(str).str.len()
                length_stats = {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": float(lengths.mean()),
                    "median": float(lengths.median()),
                }
        except Exception as e:
            logger.warning(f"Could not calculate length statistics: {str(e)}")
            length_stats = {"error": str(e)}

        # Detect if this field has any nulls
        field_has_null = null_count > 0

        # Null pattern (for series, it's just if null or not)
        null_pattern_summary = {
            "Nulls Present": int(null_count),
            "Non-Nulls": int(non_null_count),
        }

        return {
            "total_records": total_records,
            "null_count": null_count,
            "non_null_count": non_null_count,
            "null_ratio": round(null_ratio, 4),
            "removed_rows": removed_rows,
            "field_has_null": field_has_null,
            "null_pattern": null_pattern_summary,
            "unique_values": unique_values,
            "value_distribution": top_values,
            "length_stats": length_stats,
            "stats_before_removal": stats_before_removal,
            "stats_after_removal": stats_after_removal,
            "numeric_stats": stats_after_removal
        }

    @staticmethod
    def collect_transformation_metrics(
        orig_data: pd.Series,
        gen_data: pd.Series,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            Dictionary with metrics:
            - null_values_replaced: number of nulls filled
            - total_replacements: all values changed (null or not)
            - replacement_strategy: method used (from params)
            - mapping_collisions: # of collisions in mapping store (if available)
            - reversibility_rate: % of generated values that are unique
        """
        operation_params = operation_params or {}

        try:
            orig_null_mask = orig_data.isna().values
            gen_null_mask = gen_data.isna().values

            # Correct logic: replaced null = was null in orig, not null in gen
            null_values_replaced = np.sum(orig_null_mask & ~gen_null_mask)

            comparison_mask = ~orig_null_mask & ~gen_null_mask
            value_replacements = 0
            if np.any(comparison_mask):
                value_changes = (
                    orig_data.values[comparison_mask]
                    != gen_data.values[comparison_mask]
                )
                value_replacements = np.sum(value_changes)

            total_replacements = value_replacements + null_values_replaced

            replacement_strategy = operation_params.get(
                "consistency_mechanism", "unknown"
            )

            # Mapping collisions
            mapping_collisions = 0
            if (
                replacement_strategy == "mapping"
                and "mapping_store" in operation_params
            ):
                mapping_store = operation_params.get("mapping_store")
                if hasattr(mapping_store, "get_collision_count"):
                    field_name = operation_params.get("field_name", "unknown")
                    try:
                        mapping_collisions = mapping_store.get_collision_count(
                            field_name
                        )
                    except Exception:
                        mapping_collisions = 0

            # Reversibility rate = ratio of values that only appear once (cannot reverse many-to-one)
            reversibility_rate = 0.0
            try:
                gen_data_clean = gen_data[~gen_null_mask]
                if len(gen_data_clean) > 0:
                    value_counts = gen_data_clean.value_counts()
                    unique_values = np.sum(value_counts == 1)
                    reversibility_rate = unique_values / len(gen_data_clean)
            except Exception:
                reversibility_rate = 0.0

            return {
                "null_values_replaced": int(null_values_replaced),
                "total_replacements": int(total_replacements),
                "replacement_strategy": replacement_strategy,
                "mapping_collisions": int(mapping_collisions),
                "reversibility_rate": float(reversibility_rate),
            }

        except Exception:
            return {
                "null_values_replaced": 0,
                "total_replacements": 0,
                "replacement_strategy": "unknown",
                "mapping_collisions": 0,
                "reversibility_rate": 0.0,
            }

    def compare_distributions(
        self, orig_data: pd.Series, gen_data: pd.Series
    ) -> Dict[str, float]:
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
        metrics = {}

        # Get normalized value counts
        orig_counts = orig_data.value_counts(normalize=True)
        gen_counts = gen_data.value_counts(normalize=True)

        # Merge indexes
        all_values = set(orig_counts.index) | set(gen_counts.index)

        try:
            # Prepare probability arrays with smoothing
            orig_probs = (
                np.array([orig_counts.get(val, 0) for val in all_values]) + 1e-10
            )
            gen_probs = np.array([gen_counts.get(val, 0) for val in all_values]) + 1e-10

            orig_probs /= orig_probs.sum()
            gen_probs /= gen_probs.sum()
            mid_probs = (orig_probs + gen_probs) / 2

            # Jensen-Shannon divergence
            kl_orig_mid = np.sum(orig_probs * np.log(orig_probs / mid_probs))
            kl_gen_mid = np.sum(gen_probs * np.log(gen_probs / mid_probs))
            jsd = (kl_orig_mid + kl_gen_mid) / 2

            metrics["distribution_similarity_score"] = 1.0 - min(1.0, float(jsd))
            metrics["kl_divergence_orig_mid"] = float(kl_orig_mid)
            metrics["kl_divergence_gen_mid"] = float(kl_gen_mid)
        except Exception as e:
            logger.warning(f"Error calculating Jensen-Shannon divergence: {str(e)}")

        # Uniqueness preservation
        try:
            orig_unique_ratio = (
                len(orig_data.unique()) / len(orig_data) if len(orig_data) > 0 else 0
            )
            gen_unique_ratio = (
                len(gen_data.unique()) / len(gen_data) if len(gen_data) > 0 else 0
            )
            metrics["uniqueness_preservation"] = (
                gen_unique_ratio / orig_unique_ratio if orig_unique_ratio > 0 else 0
            )
        except Exception as e:
            logger.warning(f"Error calculating uniqueness preservation: {str(e)}")

        # Entropy
        try:
            from scipy.stats import entropy

            metrics["entropy_original"] = float(entropy(orig_probs))
            metrics["entropy_generated"] = float(entropy(gen_probs))
        except Exception as e:
            logger.warning(f"Error calculating entropy: {str(e)}")

        # Top-N value overlap
        try:
            top_k = 10
            orig_top = set(orig_counts.head(top_k).index)
            gen_top = set(gen_counts.head(top_k).index)
            overlap_score = len(orig_top & gen_top) / top_k
            metrics["top_value_overlap@10"] = round(overlap_score, 2)
        except Exception as e:
            logger.warning(f"Error calculating top value overlap: {str(e)}")

        # Gini coefficient
        try:

            def gini(array):
                array = np.sort(array)
                n = len(array)
                cumvals = np.cumsum(array)
                return (
                    (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
                    if cumvals[-1] != 0
                    else 0.0
                )

            metrics["gini_original"] = float(gini(orig_probs))
            metrics["gini_generated"] = float(gini(gen_probs))
        except Exception as e:
            logger.warning(f"Error calculating Gini coefficient: {str(e)}")

        return metrics

    def visualize_metrics(
        self,
        metrics: Dict[str, Any],
        field_name: str,
        output_dir: Union[str, Path],
        op_type: str,
    ) -> Dict[str, Path]:
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
        if (
            "original_data" in metrics
            and "value_distribution" in metrics["original_data"]
            and "generated_data" in metrics
            and "value_distribution" in metrics["generated_data"]
        ):

            orig_dist = metrics["original_data"]["value_distribution"]
            gen_dist = metrics["generated_data"]["value_distribution"]

            # Only visualize if there's data
            if orig_dist and gen_dist:
                # Find common top values
                common_keys = set(orig_dist.keys()) | set(gen_dist.keys())
                top_n = min(10, len(common_keys))

                # Sort by original data frequency
                common_keys = sorted(
                    common_keys, key=lambda k: orig_dist.get(k, 0), reverse=True
                )[:top_n]

                # Create data for combined chart
                orig_values = [orig_dist.get(k, 0) for k in common_keys]
                gen_values = [gen_dist.get(k, 0) for k in common_keys]

                # Create combined chart for value distribution
                dist_path = (
                    output_dir / f"{op_type}_{field_name}_value_distribution.png"
                )
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
                    )
                    visualizations["value_distribution"] = Path(vis_path)
                except Exception as e:
                    logger.warning(
                        f"Error creating value distribution visualization: {str(e)}"
                    )

        # Visualize length distribution for string or numberic data

        # --- Original Data ---
        if (
            "original_data" in metrics 
            and (
                "length_stats" in metrics["original_data"]
                or "stats_after_removal" in metrics["original_data"]
            )
        ):
            orig_stats = {}
            stats_type = None

            if "length_stats" in metrics["original_data"] and metrics["original_data"]["length_stats"]:
                orig_stats = metrics["original_data"]["length_stats"]
                stats_type = "length"
            elif "numeric_stats" in metrics["original_data"] and metrics["original_data"]["numeric_stats"]:
                orig_stats = metrics["original_data"]["numeric_stats"]
                stats_type = "numeric"

            if orig_stats and "min" in orig_stats:
                stats_keys = ["min", "max", "mean", "median"]
                orig_data = {stat: orig_stats.get(stat, 0) for stat in stats_keys}
                orig_stats_path = output_dir / f"{op_type}_{field_name}_original_{stats_type}_stats.png"

                try:
                    vis_path = create_bar_plot(
                        data=orig_data,
                        output_path=orig_stats_path,
                        title=f"Original Data {stats_type.capitalize()} Statistics for {field_name}",
                        x_label="Statistic",
                        y_label="Value",
                        orientation="v",
                        showlegend=False
                    )
                    visualizations[f"original_{stats_type}_stats"] = Path(vis_path)
                except Exception as e:
                    logger.warning(f"Error creating original {stats_type} stats visualization: {str(e)}")

        # --- Generated Data ---
        if (
            "generated_data" in metrics 
            and (
                "length_stats" in metrics["generated_data"]
                or "stats_after_removal" in metrics["generated_data"]
            )
        ):
            gen_stats = {}
            stats_type = None

            if "length_stats" in metrics["generated_data"] and metrics["generated_data"]["length_stats"]:
                gen_stats = metrics["generated_data"]["length_stats"]
                stats_type = "length"
            elif "numeric_stats" in metrics["generated_data"] and metrics["generated_data"]["numeric_stats"]:
                gen_stats = metrics["generated_data"]["numeric_stats"]
                stats_type = "numeric"

            if gen_stats and "min" in gen_stats:
                stats_keys = ["min", "max", "mean", "median"]
                gen_data = {stat: gen_stats.get(stat, 0) for stat in stats_keys}
                gen_stats_path = output_dir / f"{op_type}_{field_name}_generated_{stats_type}_stats.png"

                try:
                    vis_path = create_bar_plot(
                        data=gen_data,
                        output_path=gen_stats_path,
                        title=f"Generated Data {stats_type.capitalize()} Statistics for {field_name}",
                        x_label="Statistic",
                        y_label="Value",
                        orientation="v",
                        showlegend=False
                    )
                    visualizations[f"generated_{stats_type}_stats"] = Path(vis_path)
                except Exception as e:
                    logger.warning(f"Error creating generated {stats_type} stats visualization: {str(e)}")
                    
        # Visualize transformation metrics
        if "transformation_metrics" in metrics:
            trans_metrics = metrics["transformation_metrics"]

            # Create pie chart for replacements
            if "total_replacements" in trans_metrics:
                total = metrics["original_data"]["total_records"]
                replaced = trans_metrics["total_replacements"]
                preserved = total - replaced

                pie_data = {"Replaced Values": replaced, "Preserved Values": preserved}

                replace_path = output_dir / f"{op_type}_{field_name}_replacements.png"
                try:
                    vis_path = create_pie_chart(
                        data=pie_data,
                        output_path=replace_path,
                        title=f"Value Replacement Rate for {field_name}",
                        show_percentages=True,
                    )
                    visualizations["replacement_rate"] = Path(vis_path)
                except Exception as e:
                    logger.warning(
                        f"Error creating replacement rate visualization: {str(e)}"
                    )

        # Visualize distribution metrics
        if "distribution_metrics" in metrics:
            dist_metrics = metrics["distribution_metrics"]

            # Extract only numeric values
            chart_data = {
                k: v for k, v in dist_metrics.items() if isinstance(v, (int, float))
            }

            if chart_data:
                dist_chart_path = (
                    output_dir / f"{op_type}_{field_name}_distribution_metrics.png"
                )
                try:
                    vis_path = create_bar_plot(
                        data={"Metrics": chart_data},
                        output_path=dist_chart_path,
                        title=f"Distribution Comparison Metrics for {field_name}",
                        x_label="Metric",
                        y_label="Score",
                    )
                    visualizations["distribution_metrics"] = Path(vis_path)
                except Exception as e:
                    logger.warning(
                        f"Error creating distribution metrics visualization: {str(e)}"
                    )

        # Visualize null distribution using create_pie_chart original_data
        if (
            "original_data" in metrics
            and "null_count" in metrics["original_data"]
            and "total_records" in metrics["original_data"]
        ):
            # Original Data null distribution
            orig_null_count = metrics["original_data"]["null_count"]
            orig_total = metrics["original_data"]["total_records"]
            orig_non_null_count = orig_total - orig_null_count

            orig_pie_data = {
                "Non-Null": orig_non_null_count,
                "Null": orig_null_count,
            }

            orig_null_chart_path = output_dir / f"{op_type}_{field_name}_original_null_distribution.png"

            try:
                vis_path = create_pie_chart(
                    data=orig_pie_data,
                    output_path=orig_null_chart_path,
                    title=f"Null Value Distribution for Original Data - {field_name}",
                    show_percentages=True,
                    pull_largest=True,
                )
                visualizations["original_null_distribution"] = Path(vis_path)
            except Exception as e:
                logger.warning(f"Error creating original null distribution chart: {str(e)}")

        # Visualize null distribution using create_pie_chart generated_data
        if (
            "generated_data" in metrics
            and "null_count" in metrics["generated_data"]
            and "total_records" in metrics["generated_data"]
        ):
            # Generated Data null distribution
            gen_null_count = metrics["generated_data"]["null_count"]
            gen_total = metrics["generated_data"]["total_records"]
            gen_non_null_count = gen_total - gen_null_count

            gen_pie_data = {
                "Non-Null": gen_non_null_count,
                "Null": gen_null_count,
            }

            gen_null_chart_path = output_dir / f"{op_type}_{field_name}_generated_null_distribution.png"

            try:
                vis_path = create_pie_chart(
                    data=gen_pie_data,
                    output_path=gen_null_chart_path,
                    title=f"Null Value Distribution for Generated Data - {field_name}",
                    show_percentages=True,
                    pull_largest=True,
                )
                visualizations["generated_null_distribution"] = Path(vis_path)
            except Exception as e:
                logger.warning(f"Error creating generated null distribution chart: {str(e)}")

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
