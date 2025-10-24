"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Group Analyzer Operation
Package:       pamola.pamola_core.profiling.analyzers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides operations for analyzing variability within grouped data for anonymization.
  It evaluates variance and duplication within groups of records sharing the same identifier,
  supporting decision-making about which groups can be safely aggregated during anonymization.

Key Features:
  - Calculates variance and duplication metrics for fields within groups
  - Generates visualizations of variance distribution and field variability
  - Determines aggregation candidates based on configurable thresholds
  - Supports different algorithms for text comparison (MD5, MinHash)
  - Analyzes fields with different weights to produce weighted variance scores
  - Efficient chunked, parallel, and Dask-based processing for large datasets
  - Robust error handling, progress tracking, and operation logging
  - Caching and efficient repeated analysis
  - Integration with PAMOLA.CORE operation framework for standardized input/output
"""

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pamola_core.profiling.commons.group_utils import (
    analyze_group,
    calculate_field_metrics,
)
from pamola_core.profiling.schemas.group_config import GroupAnalyzerOperationConfig
from pamola_core.utils.helpers import filter_used_kwargs
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import (
    OperationArtifact,
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import create_histogram, create_heatmap
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.common.constants import Constants

# Configure module logger
logger = logging.getLogger(__name__)


class GroupAnalyzer:
    def analyze(
        self,
        df: pd.DataFrame,
        field_name: str,
        fields_config: Dict[str, int],
        fields_to_analyze: List[str],
        text_length_threshold: int = 100,
        variance_threshold: float = 0.2,
        large_group_threshold: int = 100,
        large_group_variance_threshold: float = 0.05,
        hash_algorithm: str = "md5",
        minhash_similarity_threshold: float = 0.7,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze grouped data to compute variance and duplication metrics for anonymization.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame to analyze.
        field_name : str
            The column name to group data by.
        fields_config : Dict[str, int]
            Dictionary mapping field names to their weights for variance calculation.
        fields_to_analyze : List[str]
            List of field names to analyze within each group.
        text_length_threshold : int, optional
            Minimum length for text fields to be considered for hashing (default: 100).
        variance_threshold : float, optional
            Variance threshold for aggregation decision (default: 0.2).
        large_group_threshold : int, optional
            Threshold for considering a group as large (default: 100).
        large_group_variance_threshold : float, optional
            Variance threshold for large groups (default: 0.05).
        hash_algorithm : str, optional
            Algorithm for text comparison: "md5" or "minhash" (default: "md5").
        minhash_similarity_threshold : float, optional
            Similarity threshold for MinHash signatures (default: 0.7).
        progress_tracker : Optional[HierarchicalProgressTracker], optional
            Progress tracker for reporting progress (default: None).
        task_logger : Optional[logging.Logger], optional
            Logger for logging messages (default: None).
        **kwargs
            Additional keyword arguments passed to downstream utilities.

        Returns:
        --------
        Dict[str, Any]
        """
        if task_logger:
            logger = task_logger

        # Step 2: Group data
        if progress_tracker:
            progress_tracker.update(1, {"step": "Grouping data"})

        logger.info(f"Grouping data by field: {field_name}")

        # Group data by field_name
        grouped = df.groupby(field_name)
        group_keys = list(grouped.groups.keys())

        # Step 3: Analyze groups
        if progress_tracker:
            progress_tracker.update(1, {"step": "Analyzing groups"})

        logger.info(f"Analyzing {len(group_keys)} groups")

        # Initialize group analysis dictionary for storing metrics
        group_metrics = {}

        # Track variance distribution for visualization
        variance_distribution = {
            "below_0.1": 0,
            "0.1_to_0.2": 0,
            "0.2_to_0.5": 0,
            "0.5_to_0.8": 0,
            "above_0.8": 0,
        }

        # Track field variances for visualization
        field_variances = {field: [] for field in fields_to_analyze}

        # Calculate metrics for each group
        for i, group_key in enumerate(group_keys):
            if progress_tracker and i % 100 == 0:
                progress_tracker.update(
                    0,
                    {
                        "step": "Analyzing groups",
                        "processed": i,
                        "total": len(group_keys),
                    },
                )

            group_df = grouped.get_group(group_key)
            group_metrics[str(group_key)] = analyze_group(
                group_df=group_df,
                fields=fields_to_analyze,
                fields_config=fields_config,
                text_length_threshold=text_length_threshold,
                hash_algorithm=hash_algorithm,
                use_minhash=(hash_algorithm == "minhash"),
                minhash_similarity_threshold=minhash_similarity_threshold,
                minhash_cache={},
                large_group_threshold=large_group_threshold,
                large_group_variance_threshold=large_group_variance_threshold,
                variance_threshold=variance_threshold,
            )

            # Update variance distribution
            weighted_variance = group_metrics[str(group_key)]["weighted_variance"]
            if weighted_variance < 0.1:
                variance_distribution["below_0.1"] += 1
            elif weighted_variance < 0.2:
                variance_distribution["0.1_to_0.2"] += 1
            elif weighted_variance < 0.5:
                variance_distribution["0.2_to_0.5"] += 1
            elif weighted_variance < 0.8:
                variance_distribution["0.5_to_0.8"] += 1
            else:
                variance_distribution["above_0.8"] += 1

            # Update field variances
            for field in fields_to_analyze:
                field_variances[field].append(
                    group_metrics[str(group_key)]["field_variances"].get(field, 0)
                )

        # Calculate overall metrics
        field_metrics = calculate_field_metrics(group_metrics, fields_to_analyze)

        # Calculate average variance across all fields and groups
        all_variances = []
        for metrics in group_metrics.values():
            all_variances.append(metrics["weighted_variance"])

        avg_variance = np.mean(all_variances) if all_variances else 0
        max_variance = np.max(all_variances) if all_variances else 0

        # Count groups that can be aggregated
        groups_to_aggregate = 0
        for metrics in group_metrics.values():
            if metrics["should_aggregate"]:
                groups_to_aggregate += 1

        # Build metrics object
        metrics = {
            "field": field_name,
            "total_groups": len(group_keys),
            "total_records": len(df),
            "avg_variance": float(avg_variance),
            "max_variance": float(max_variance),
            "groups_to_aggregate": groups_to_aggregate,
            "field_metrics": field_metrics,
            "group_metrics": group_metrics,
            "threshold_metrics": variance_distribution,
            "algorithm_info": {
                "hash_algorithm_used": hash_algorithm,
                "text_length_threshold": text_length_threshold,
                "variance_threshold": variance_threshold,
                "large_group_threshold": large_group_threshold,
                "large_group_variance_threshold": large_group_variance_threshold,
                "minhash_similarity_threshold": minhash_similarity_threshold,
            },
            "fields_analyzed": fields_to_analyze,
        }
        return metrics



@register(version="1.0.0")
class GroupAnalyzerOperation(FieldOperation):
    """
    Operation for analyzing variability within groups of records.

    This operation analyzes variability within groups of records with the same
    identifier in a specific data subset, calculating metrics for use in
    anonymization decision-making.
    """

    def __init__(
        self,
        field_name: str,
        fields_config: Dict[str, int],
        text_length_threshold: int = 100,
        variance_threshold: float = 0.2,
        large_group_threshold: int = 100,
        large_group_variance_threshold: float = 0.05,
        hash_algorithm: str = "md5",
        minhash_similarity_threshold: float = 0.7,
        **kwargs,
    ):
        """
        Initialize the group analyzer operation.

        Parameters
        ----------
        field_name : str
            The name of the field to group by.
        fields_config : Dict[str, int]
            Dictionary mapping field names to weights.
        text_length_threshold : int
            Threshold for long text fields (default: 100).
        variance_threshold : float
            Threshold for aggregation decision (default: 0.2).
        large_group_threshold : int
            Minimum size to consider a group as "large".
        large_group_variance_threshold : float
            Variance threshold for large groups (default: 0.05).
        hash_algorithm : str
            Hash algorithm for comparison ('md5' or 'minhash').
        minhash_similarity_threshold : float
            Threshold for MinHash similarity (default: 0.7).
        **kwargs : dict
            Additional parameters passed to FieldOperation.
        """

        # --- Default description fallback ---
        kwargs.setdefault(
            "description",
            f"Group variance analysis for field '{field_name}'",
        )

        # --- Build unified config object ---
        config = GroupAnalyzerOperationConfig(
            field_name=field_name,
            fields_config=fields_config,
            text_length_threshold=text_length_threshold,
            variance_threshold=variance_threshold,
            large_group_threshold=large_group_threshold,
            large_group_variance_threshold=large_group_variance_threshold,
            hash_algorithm=hash_algorithm.lower(),
            minhash_similarity_threshold=minhash_similarity_threshold,
            **kwargs,
        )

        # Inject config into kwargs
        kwargs["config"] = config

        # Initialize base FieldOperation
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # --- Save config attributes to self ---
        for key, value in config.to_dict().items():
            setattr(self, key, value)

        # --- Analyzer binding ---
        self.analyzer = GroupAnalyzer()
        self.use_minhash = self.hash_algorithm == "minhash"
        self.minhash_cache = {}

        # Operation metadata---
        self.operation_name = self.__class__.__name__

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the email analysis operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            if kwargs.get("logger"):
                self.logger = kwargs["logger"]

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(f"Starting group analysis for field: {self.field_name}")

            # Initialize result object
            result = OperationResult(status=OperationStatus.PENDING)

            # Save configuration
            self.save_config(task_dir)

            # Create data writer
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Preparation, Cache Check, Data Loading, Grouping, Visualizations, Saving results
            total_steps = 5 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            current_steps = 0

            # Step 1: Preparation
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(
                    current_steps,
                    {"step": "Starting group analysis", "field": self.field_name},
                )

            # Step 2: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                if progress_tracker:
                    current_steps += 1
                    progress_tracker.update(current_steps, {"step": "Checking Cache"})

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(data_source, dataset_name)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(
                            total_steps, {"step": "Complete (cached)"}
                        )

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Clean invalid values (from cache)",
                            details={"cached": True},
                        )
                    return cache_result

            dirs = self._prepare_directories(task_dir)
            visualizations_dir = dirs["visualizations"]
            output_dir = dirs["output"]

            # Step 3: Load data
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Loading data"})

            self.logger.info(f"Loading data for: {self.field_name}")
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )
            df = load_data_operation(data_source, dataset_name, **settings_operation)

            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source",
                )

            # Check if field_name exists
            if self.field_name not in df.columns:
                error_message = f"Field '{self.field_name}' not found in DataFrame"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Validate fields to analyze exist in the DataFrame
            fields_to_analyze = list(self.fields_config.keys())
            missing_fields = [
                field for field in fields_to_analyze if field not in df.columns
            ]

            if missing_fields:
                error_message = f"Fields not found in DataFrame: {missing_fields}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Step 4: Group data by the specified field
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Grouping data"})

            safe_kwargs = filter_used_kwargs(kwargs, GroupAnalyzer.analyze)
            metrics = self.analyzer.analyze(
                df=df,
                field_name=self.field_name,
                fields_config=self.fields_config,
                fields_to_analyze=fields_to_analyze,
                hash_algorithm=self.hash_algorithm,
                large_group_threshold=self.large_group_threshold,
                large_group_variance_threshold=self.large_group_variance_threshold,
                minhash_similarity_threshold=self.minhash_similarity_threshold,
                text_length_threshold=self.text_length_threshold,
                variance_threshold=self.variance_threshold,
                progress_tracker=progress_tracker,
                task_logger=self.logger,
                **safe_kwargs,
            )

            # Step 5: Generate visualizations
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(
                    current_steps, {"step": "Generating visualizations"}
                )

            self.logger.info(f"Generating visualizations")
            encryption_kwargs = {
                "use_encryption": self.use_encryption,
                "encryption_key": self.encryption_key if self.use_encryption else None,
            }
            self._handle_visualizations(
                threshold_metrics=metrics["threshold_metrics"],
                field_metrics=metrics["field_metrics"],
                vis_dir=visualizations_dir,
                result=result,
                reporter=reporter,
                vis_theme=self.visualization_theme,
                vis_backend=self.visualization_backend,
                vis_strict=self.visualization_strict,
                vis_timeout=self.visualization_timeout,
                progress_tracker=progress_tracker,
                operation_timestamp=operation_timestamp,
                **encryption_kwargs,
            )

            # Step 6: Save results
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Saving results"})

            self.logger.info(f"Saving metrics")

            # Save metrics
            metrics_result = writer.write_metrics(
                metrics=metrics,
                name=f"{self.__class__.__name__}_{self.field_name}_metrics",
                timestamp_in_name=False,
                encryption_key=self.encryption_key if self.use_encryption else None,
            )

            # Add metrics to operation result
            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    result.add_metric(key, value)

            # Register artifacts
            result.add_artifact(
                artifact_type="json",
                path=metrics_result.path,
                description=f"{self.__class__.__name__} {self.field_name} group analysis metrics",
                category=Constants.Artifact_Category_Metrics,
            )

            # Set success status
            result.status = OperationStatus.SUCCESS

            self.logger.info(
                f"Group analysis for {self.field_name} completed successfully"
            )

            # Group data by field_name
            grouped = df.groupby(self.field_name)
            group_keys = list(grouped.groups.keys())

            # Report to reporter if available
            if reporter:
                reporter.add_operation(
                    f"Group analysis for {self.field_name}",
                    details={
                        "total_groups": len(group_keys),
                        "groups_to_aggregate": metrics["groups_to_aggregate"],
                        "avg_variance": metrics["avg_variance"],
                    },
                )

                reporter.add_artifact(
                    "json",
                    str(metrics_result.path),
                    f"{self.__class__.__name__} {self.field_name} group analysis metrics",
                )

            return result

        except Exception as e:
            error_message = f"Error executing group analysis: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message, exception=e
            )

    def _calculate_field_variance(self, field_series: pd.Series) -> Tuple[float, float]:
        """
        Calculate the variance and duplication ratio for a single field.

        Parameters:
        -----------
        field_series : pd.Series
            Series containing values for a single field

        Returns:
        --------
        Tuple[float, float]
            (variance, duplication_ratio)
        """
        total_records = len(field_series)

        if total_records <= 1:
            return 0.0, 0.0

        # Get unique values
        unique_values, value_counts = self._get_unique_values_improved(field_series)
        unique_count = len(unique_values)

        # Calculate variance: (unique_count - 1) / (total_records - 1)
        # This gives 0 when all values are identical, 1 when all values are unique
        # Low variance means many duplicates (high similarity), high variance means few duplicates
        variance = (unique_count - 1) / (total_records - 1) if total_records > 1 else 0

        # Calculate duplication ratio: total_records / unique_count
        duplication_ratio = total_records / unique_count if unique_count > 0 else 0

        return float(variance), float(duplication_ratio)

    def _get_unique_values_improved(
        self, field_series: pd.Series
    ) -> Tuple[Set[Any], Dict[Any, int]]:
        """
        Get unique values from a field series, handling text and NULL values.
        Improved version with better categorical data handling.

        Parameters:
        -----------
        field_series : pd.Series
            Series containing values for a single field

        Returns:
        --------
        Tuple[Set[Any], Dict[Any, int]]
            Set of unique values and dictionary with value counts
        """
        unique_keys = set()
        counts = {}

        # Process each value individually to avoid categorical data issues
        for raw_val in field_series:
            # Handle NULL values
            if pd.isna(raw_val):
                key = None  # Use None as the key for NaN values
            # Handle string values with potential hashing
            elif isinstance(raw_val, str):
                text = raw_val.strip()
                if len(text) > self.text_length_threshold:
                    if self.use_minhash:
                        # Lazy import MinHash only if needed
                        if not hasattr(self, "_minhash_imported"):
                            # Initialize flag and function variable before try block
                            self._minhash_imported = False
                            compute_minhash = (
                                None  # Define variable before try to satisfy linter
                            )
                            compute_minhash_func = None

                            try:
                                # Dynamically import minhash module when needed
                                from pamola_core.utils.nlp.minhash import (
                                    compute_minhash,
                                )

                                compute_minhash_func = (
                                    compute_minhash  # Assign the imported function
                                )
                                self._minhash_imported = (
                                    True  # Set flag only on success
                                )
                            except ImportError:
                                # If import fails (e.g., no pymorphy2), we end up here
                                compute_minhash = (
                                    None  # Explicitly define in except block for linter
                                )
                                self.logger.warning(
                                    "MinHash library not available, falling back to MD5"
                                )
                                self.use_minhash = False  # Make sure we switch to MD5
                                self._minhash_imported = (
                                    False  # Explicitly mark import as failed
                                )
                                compute_minhash_func = (
                                    lambda _: []
                                )  # Assign a dummy function with unused parameter

                            # Save reference to function (imported or dummy) in object attribute
                            self._compute_minhash = compute_minhash_func

                        if self.use_minhash and self._minhash_imported:
                            # Use cached MinHash signatures when possible
                            if text not in self.minhash_cache:
                                self.minhash_cache[text] = self._compute_minhash(text)

                            # Use the first 8 elements of the signature as the key
                            signature = self.minhash_cache[text]
                            signature_str = str(
                                signature[:8]
                            )  # Convert to string for consistent keys
                            key = signature_str
                        else:
                            # Fallback to MD5 if MinHash is not available
                            key = hashlib.md5(text.encode("utf-8")).hexdigest()
                    else:
                        # Use MD5 for exact matching of long texts
                        key = hashlib.md5(text.encode("utf-8")).hexdigest()
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
        if (
            self.use_minhash
            and hasattr(self, "_minhash_imported")
            and self._minhash_imported
        ):
            minhash_keys = [
                k for k in counts.keys() if isinstance(k, str) and k.startswith("[")
            ]

            if len(minhash_keys) > 1:
                # Group similar signatures
                clusters = self._cluster_minhash_signatures_from_keys(minhash_keys)

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

    def _cluster_minhash_signatures_from_keys(
        self, signature_keys: List[str]
    ) -> List[List[str]]:
        """
        Cluster similar MinHash signature keys.

        Parameters:
        -----------
        signature_keys : List[str]
            List of MinHash signature keys to cluster

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
                nums = key.strip("[]").split(",")
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
            for key2, sig2 in parsed_signatures[i + 1 :]:
                if key2 in processed:
                    continue

                # Calculate Jaccard similarity
                similarity = self._calculate_simple_jaccard(sig1, sig2)

                # If similar enough, add to the cluster
                if similarity >= self.minhash_similarity_threshold:
                    cluster.append(key2)
                    processed.add(key2)

            # Add the cluster to the results
            clusters.append(cluster)

        return clusters

    def _calculate_simple_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
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

    def _generate_variance_distribution(
        self, variance_distribution: Dict[str, int], output_path: Path, **kwargs
    ):
        """
        Generate a histogram of variance distribution.

        Parameters:
        -----------
        variance_distribution : Dict[str, int]
            Dictionary with variance ranges and their counts
        output_path : Path
            Path where to save the visualization
        """
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create ordered categories for better visualization
        categories = [
            "below_0.1",
            "0.1_to_0.2",
            "0.2_to_0.5",
            "0.5_to_0.8",
            "above_0.8",
        ]

        # Create labels mapping
        labels_map = {
            "below_0.1": "< 0.1",
            "0.1_to_0.2": "0.1 - 0.2",
            "0.2_to_0.5": "0.2 - 0.5",
            "0.5_to_0.8": "0.5 - 0.8",
            "above_0.8": "> 0.8",
        }

        # Create data for visualization
        data = {labels_map[cat]: variance_distribution[cat] for cat in categories}

        try:
            # Check if the visualization function accepts dictionary input
            # If not, convert to appropriate format
            try:
                # Try with direct dictionary (preferred if supported)
                create_histogram(
                    data=data,
                    output_path=str(output_path),
                    title=f"Variability Distribution ({self.field_name})",
                    x_label="Variability Range",
                    y_label="Number of Groups",
                    bins=len(categories),
                    kde=False,
                    **kwargs,
                )
            except (TypeError, ValueError) as e:
                # Fallback to matplotlib for proper histogram
                plt.figure(figsize=(10, 6))
                plt.bar(list(data.keys()), list(data.values()), color="skyblue")
                plt.title(f"Variability Distribution ({self.field_name})")
                plt.xlabel("Variability Range")
                plt.ylabel("Number of Groups")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()

            self.logger.info(
                f"Generated variance distribution visualization at {output_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Error generating variance distribution visualization: {str(e)}"
            )

            # Fallback to basic matplotlib in case of error
            try:
                plt.figure(figsize=(10, 6))
                plt.bar(list(data.keys()), list(data.values()), color="skyblue")
                plt.title(f"Variability Distribution ({self.field_name})")
                plt.xlabel("Variability Range")
                plt.ylabel("Number of Groups")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()
                self.logger.info(
                    f"Generated fallback variance distribution visualization at {output_path}"
                )
            except Exception as e2:
                self.logger.error(
                    f"Failed to generate fallback visualization: {str(e2)}"
                )

    def _generate_field_heatmap(
        self, field_metrics: Dict[str, Dict[str, float]], output_path: Path, **kwargs
    ):
        """
        Generate a heatmap of field variances.

        Parameters:
        -----------
        field_metrics : Dict[str, Dict[str, float]]
            Dictionary with metrics for each field
        output_path : Path
            Path where to save the visualization
        """
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare data for heatmap
            field_names = list(field_metrics.keys())
            metric_types = ["avg_variance", "max_variance", "avg_duplication_ratio"]

            # Format data suitable for heatmap function
            data = {}
            for metric in metric_types:
                data[metric] = {}
                for field in field_names:
                    data[metric][field] = field_metrics[field][metric]

            # Convert to DataFrame for better compatibility
            df_data = pd.DataFrame(
                {
                    metric: [data[metric][field] for field in field_names]
                    for metric in metric_types
                },
                index=field_names,
            )

            try:
                # Try with the pamola core visualization utility
                create_heatmap(
                    data=df_data,
                    output_path=str(output_path),
                    title=f"Field Variability Metrics ({self.field_name})",
                    x_label="Metrics",
                    y_label="Fields",
                    colorscale="Viridis",
                    annotate=True,
                    annotation_format=".3f",
                    **kwargs,
                )
            except (TypeError, ValueError) as e:
                # Fallback to matplotlib for the heatmap
                plt.figure(figsize=(12, len(field_names) * 0.8))
                im = plt.imshow(df_data.values, cmap="viridis")

                # Add labels
                plt.colorbar(im)
                plt.xticks(range(len(metric_types)), metric_types, rotation=45)
                plt.yticks(range(len(field_names)), field_names)

                plt.title(f"Field Variability Metrics ({self.field_name})")

                # Add annotations
                for i in range(len(field_names)):
                    for j in range(len(metric_types)):
                        text = f"{df_data.iloc[i, j]:.3f}"
                        plt.text(j, i, text, ha="center", va="center", color="white")

                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()

            self.logger.info(f"Generated field variability heatmap at {output_path}")
        except Exception as e:
            self.logger.error(f"Error generating field variability heatmap: {str(e)}")

            # Fallback to basic matplotlib in case of error
            try:
                # Create a simple bar chart instead of a heatmap
                field_avg_variances = {
                    field: metrics["avg_variance"]
                    for field, metrics in field_metrics.items()
                }

                plt.figure(figsize=(10, 6))
                plt.bar(
                    field_avg_variances.keys(),
                    field_avg_variances.values(),
                    color="skyblue",
                )
                plt.title(f"Field Average Variances ({self.field_name})")
                plt.xlabel("Field")
                plt.ylabel("Average Variance")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()

                self.logger.info(
                    f"Generated fallback field variance visualization at {output_path}"
                )
            except Exception as e2:
                self.logger.error(
                    f"Failed to generate fallback visualization: {str(e2)}"
                )

    def _check_cache(
        self,
        data_source: DataSource,
        data_source_name: str = "main",
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        data_source : DataSource
            Data source for the operation
        task_dir : Path
            Task directory
        data_source_name: str
            Dataset name

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Get DataFrame from data source
            df = load_data_operation(data_source, data_source_name)
            if df is None:
                self.logger.warning("No valid DataFrame found in data source")
                return None

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        cached_result.add_metric(key, value)

                # Add cached artifacts to result
                artifacts = cached_data.get("artifacts", [])
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        artifact_type = artifact.get("artifact_type", "")
                        artifact_path = artifact.get("path", "")
                        artifact_name = artifact.get("description", "")
                        artifact_category = artifact.get("category", "output")
                        cached_result.add_artifact(
                            artifact_type,
                            artifact_path,
                            artifact_name,
                            artifact_category,
                        )

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(
        self,
        original_df: pd.DataFrame,
        artifacts: List[OperationArtifact],
        metrics: Dict[str, Any],
        task_dir: Path,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original input data
        processed_df : pd.DataFrame
            Processed DataFrame
        metrics : dict
            The metrics of operation
        task_dir : Path
            Task directory

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache or (not artifacts and not metrics):
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Generate cache key
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            artifacts_for_cache = [artifact.to_dict() for artifact in artifacts]

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "artifacts": artifacts_for_cache,
                "artifacts_test": artifacts,
                "metrics": metrics,
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(f"Successfully saved results to cache")
            else:
                self.logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        from pamola_core.utils.ops.op_cache import operation_cache

        # Get operation parameters
        parameters = self._get_operation_parameters()

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(df)

        # Use the operation_cache utility to generate a consistent cache key
        return operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash,
        )

    def _get_operation_parameters(self) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Get basic operation parameters
        parameters = {
            "field_name": self.field_name,
            "fields_config": self.fields_config,
            "text_length_threshold": self.text_length_threshold,
            "variance_threshold": self.variance_threshold,
            "large_group_threshold": self.large_group_threshold,
            "large_group_variance_threshold": self.large_group_variance_threshold,
            "hash_algorithm": self.hash_algorithm,
            "minhash_similarity_threshold": self.minhash_similarity_threshold,
            "encryption_key": self.encryption_key,
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Generate a hash representing the key characteristics of the data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Hash string representing the data
        """
        import hashlib

        try:
            # Create data characteristics
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _handle_visualizations(
        self,
        threshold_metrics: Dict[str, Any],
        field_metrics: Dict[str, Any],
        vis_dir: Path,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        operation_timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        threshold_metrics : Dict[str, Any]
            Dictionary containing variance distribution metrics (e.g., group counts by variance range).
        field_metrics : Dict[str, Any]
            Dictionary containing per-field metrics (e.g., avg/max variance, duplication ratios).
        vis_dir : Path
            Directory to save visualizations
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        self.logger.info(
            f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s"
        )

        try:
            import threading
            import contextvars

            visualization_paths = {}
            visualization_error = None

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                self.logger.info(
                    f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}"
                )
                self.logger.info(
                    f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
                )

                start_time = time.time()

                try:
                    # Log context variables
                    self.logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        self.logger.info(
                            f"[DIAG] Context vars count: {len(list(current_context))}"
                        )
                    except Exception as ctx_e:
                        self.logger.warning(
                            f"[DIAG] Could not inspect context: {ctx_e}"
                        )

                    # Generate visualizations with visualization context parameters
                    self.logger.info(f"[DIAG] Calling _generate_visualizations...")
                    # Create child progress tracker for visualization if available
                    total_steps = 3  # prepare data, create viz, save
                    viz_progress = None
                    if progress_tracker and hasattr(progress_tracker, "create_subtask"):
                        try:
                            viz_progress = progress_tracker.create_subtask(
                                total=total_steps,
                                description="Generating visualizations",
                                unit="steps",
                            )
                        except Exception as e:
                            self.logger.debug(
                                f"Could not create child progress tracker: {e}"
                            )

                    # Generate visualizations
                    # Generate variance distribution histogram
                    variance_dist_filename = f"{self.__class__.__name__}_{self.field_name}_variance_dist_{operation_timestamp}.png"
                    variance_dist_path = vis_dir / variance_dist_filename
                    self._generate_variance_distribution(
                        threshold_metrics, variance_dist_path, **kwargs
                    )
                    visualization_paths.update(
                        {"variance_dist_path": variance_dist_path}
                    )
                    # Generate field variability heatmap
                    field_heatmap_filename = f"{self.__class__.__name__}_{self.field_name}_field_heatmap_{operation_timestamp}.png"
                    field_heatmap_path = vis_dir / field_heatmap_filename
                    self._generate_field_heatmap(
                        field_metrics, field_heatmap_path, **kwargs
                    )
                    visualization_paths.update(
                        {"field_heatmap_path": field_heatmap_path}
                    )
                    # Close visualization progress tracker
                    if viz_progress:
                        try:
                            viz_progress.close()
                        except:
                            pass

                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"[DIAG] Visualization completed in {elapsed:.2f}s, generated {len(visualization_paths)} files"
                    )
                except Exception as e:
                    elapsed = time.time() - start_time
                    visualization_error = e
                    self.logger.error(
                        f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                    )
                    self.logger.error(f"[DIAG] Stack trace:", exc_info=True)

            # Copy context for the thread
            self.logger.info(f"[DIAG] Preparing to launch visualization thread...")
            ctx = contextvars.copy_context()

            # Create thread with context
            viz_thread = threading.Thread(
                target=ctx.run,
                args=(generate_viz_with_diagnostics,),
                name=f"VizThread-",
                daemon=False,  # Changed from True to ensure proper cleanup
            )

            self.logger.info(
                f"[DIAG] Starting visualization thread with timeout={vis_timeout}s"
            )
            thread_start_time = time.time()
            viz_thread.start()

            # Log periodic status while waiting
            check_interval = 5  # seconds
            elapsed = 0
            while viz_thread.is_alive() and elapsed < vis_timeout:
                viz_thread.join(timeout=check_interval)
                elapsed = time.time() - thread_start_time
                if viz_thread.is_alive():
                    self.logger.info(
                        f"[DIAG] Visualization thread still running after {elapsed:.1f}s..."
                    )

            if viz_thread.is_alive():
                self.logger.error(
                    f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout"
                )
                self.logger.error(
                    f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}"
                )
                visualization_paths = {}
            elif visualization_error:
                self.logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = {}
            else:
                total_time = time.time() - thread_start_time
                self.logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                self.logger.info(
                    f"[DIAG] Generated visualizations: {list(visualization_paths.keys())}"
                )
        except Exception as e:
            self.logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            self.logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = {}

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type="png",
                    path=str(path),
                    description=f"{viz_type} visualization",
                )

        return visualization_paths
