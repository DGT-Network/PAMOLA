"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Group Analyzer Operation
Description: Operation for analyzing variability within grouped data for anonymization
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This operation analyzes vertical subsets of resume data produced by the Scissors module,
evaluating the variance and duplication within groups of records that share the same
resume_id. It supports decision-making about which groups can be safely aggregated
during anonymization.

Key features:
- Calculates variance and duplication metrics for fields within groups
- Generates visualizations of variance distribution
- Determines aggregation candidates based on configurable thresholds
- Supports different algorithms for text comparison (MD5, MinHash)
- Analyzes fields with different weights to produce weighted variance scores
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import only the essential base classes and utilities
# Note: MinHash-related imports are moved to conditional imports inside methods
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import create_histogram, create_heatmap
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.common.constants import Constants

# Configure module logger
logger = logging.getLogger(__name__)


class GroupAnalyzerConfig(OperationConfig):
    """Configuration for GroupAnalyzerOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "fields_config": {
                "type": "object",
                "additionalProperties": {"type": "number"}
            },
            "text_length_threshold": {"type": "integer", "minimum": 10},
            "variance_threshold": {"type": "number", "minimum": 0, "maximum": 1.0},
            "large_group_threshold": {"type": "integer", "minimum": 1},
            "large_group_variance_threshold": {"type": "number", "minimum": 0, "maximum": 1.0},
            "hash_algorithm": {"type": "string", "enum": ["md5", "minhash"]},
            "minhash_similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1.0},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]}           
        }
    }


class GroupAnalyzerOperation(BaseOperation):
    """
    Operation for analyzing variability within groups of records.

    This operation analyzes variability within groups of records with the same
    identifier in a specific data subset, calculating metrics for use in
    anonymization decision-making.
    """

    def __init__(self,
                 field_name: str,
                 fields_config: Dict[str, int],
                 text_length_threshold: int = 100,
                 variance_threshold: float = 0.2,
                 large_group_threshold: int = 100,
                 large_group_variance_threshold: float = 0.05,
                 hash_algorithm: str = "md5",
                 minhash_similarity_threshold: float = 0.7,  # Added for MinHash comparison
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 name: str = "group_analyzer",
                 description: str = ""):
        """
        Initialize the group analyzer operation.

        Parameters:
        -----------
        fields_config : Dict[str, int]
            Dictionary mapping field names to their weights
        text_length_threshold : int, optional
            Threshold for long text fields (default: 100)
        variance_threshold : float, optional
            Threshold for aggregation decision (default: 0.2)
        large_group_threshold : int, optional
            Threshold for large group size (default: 100)
        large_group_variance_threshold : float, optional
            Variance threshold for large groups (default: 0.05)
        hash_algorithm : str, optional
            Algorithm for text comparison: "md5" or "minhash" (default: "md5")
        minhash_similarity_threshold : float, optional
            Threshold for considering MinHash signatures as similar (default: 0.7)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : Optional[Union[str, Path]], optional
            Encryption key for securing outputs (default: None)
        field_name : str
            Field name to group data by
        name : str, optional
            Operation name (default: "group_analyzer")
        description : str, optional
            Operation description (default: "")
        """
        # Create configuration
        config = GroupAnalyzerConfig(
            field_name=field_name,
            fields_config=fields_config,
            text_length_threshold=text_length_threshold,
            variance_threshold=variance_threshold,
            large_group_threshold=large_group_threshold,
            large_group_variance_threshold=large_group_variance_threshold,
            hash_algorithm=hash_algorithm,
            minhash_similarity_threshold=minhash_similarity_threshold,
            use_encryption=use_encryption,
            encryption_key=encryption_key,     
        )

        # Use a default description if none provided
        if not description:
            description = f"Group variance analysis for field '{field_name}'"

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )

        # Store parameters
        self.field_name = field_name
        self.fields_config = fields_config
        self.text_length_threshold = text_length_threshold
        self.variance_threshold = variance_threshold
        self.large_group_threshold = large_group_threshold
        self.large_group_variance_threshold = large_group_variance_threshold
        self.hash_algorithm = hash_algorithm.lower()
        self.minhash_similarity_threshold = minhash_similarity_threshold
        self.use_minhash = self.hash_algorithm == "minhash"     

        # Cache for MinHash signatures to avoid recomputation - only initialized if needed
        self.minhash_cache = {}

        # Semantic versioning for caching
        self.version = "1.1.0"  # Bumped version due to improvements

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}")

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the group analyzer operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[ProgressTracker]
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        self.logger.info(f"Starting group analysis for field: {self.field_name}")

        # Initialize result object
        result = OperationResult(status=OperationStatus.PENDING)

        # Create data writer
        writer = DataWriter(task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker)

        # Set up progress tracking
        total_steps = 5  # Data loading, grouping, analysis, visualization, saving
        if progress_tracker:
            progress_tracker.total = total_steps
            progress_tracker.update(0, {"step": "Starting group analysis", "field": self.field_name})

        try:
            dirs = self._prepare_directories(task_dir)
            visualizations_dir = dirs['visualizations']
            output_dir = dirs['output']

            # Step 1: Load data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Loading data"})

            self.logger.info(f"Loading data for: {self.field_name}")
            # Use the subset name instead of "main" to get the correct dataframe
            dataset_name = kwargs.get('dataset_name', 'main')
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)

            # Save configuration
            self.save_config(task_dir)

            # Check if field_name exists
            if self.field_name not in df.columns:
                error_message = f"Field '{self.field_name}' not found in DataFrame"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Validate fields to analyze exist in the DataFrame
            fields_to_analyze = list(self.fields_config.keys())
            missing_fields = [field for field in fields_to_analyze if field not in df.columns]

            if missing_fields:
                error_message = f"Fields not found in DataFrame: {missing_fields}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Step 2: Group data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Grouping data"})

            self.logger.info(f"Grouping data by field: {self.field_name}")

            # Group data by field_name
            grouped = df.groupby(self.field_name)
            group_keys = list(grouped.groups.keys())

            # Step 3: Analyze groups
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analyzing groups"})

            self.logger.info(f"Analyzing {len(group_keys)} groups")

            # Initialize group analysis dictionary for storing metrics
            group_metrics = {}

            # Track variance distribution for visualization
            variance_distribution = {
                "below_0.1": 0,
                "0.1_to_0.2": 0,
                "0.2_to_0.5": 0,
                "0.5_to_0.8": 0,
                "above_0.8": 0
            }

            # Track field variances for visualization
            field_variances = {field: [] for field in fields_to_analyze}

            # Calculate metrics for each group
            for i, group_key in enumerate(group_keys):
                if progress_tracker and i % 100 == 0:
                    progress_tracker.update(0, {"step": "Analyzing groups", "processed": i, "total": len(group_keys)})

                group_df = grouped.get_group(group_key)
                group_metrics[str(group_key)] = self._analyze_group(group_df, fields_to_analyze)

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
                    field_variances[field].append(group_metrics[str(group_key)]["field_variances"].get(field, 0))

            # Calculate overall metrics
            field_metrics = self._calculate_field_metrics(group_metrics, fields_to_analyze)

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
                "field": self.field_name,
                "total_groups": len(group_keys),
                "total_records": len(df),
                "avg_variance": float(avg_variance),
                "max_variance": float(max_variance),
                "groups_to_aggregate": groups_to_aggregate,
                "field_metrics": field_metrics,
                "group_metrics": group_metrics,
                "threshold_metrics": variance_distribution,
                "algorithm_info": {
                    "hash_algorithm_used": self.hash_algorithm,
                    "text_length_threshold": self.text_length_threshold,
                    "variance_threshold": self.variance_threshold,
                    "large_group_threshold": self.large_group_threshold,
                    "large_group_variance_threshold": self.large_group_variance_threshold,
                    "minhash_similarity_threshold": self.minhash_similarity_threshold
                },
                "fields_analyzed": fields_to_analyze
            }

            # Step 4: Generate visualizations
            if progress_tracker:
                progress_tracker.update(1, {"step": "Generating visualizations"})

            self.logger.info(f"Generating visualizations")

            # Generate variance distribution histogram
            variance_dist_path = visualizations_dir / f"{self.__class__.__name__}_{self.field_name}_variance_dist.png"
            self._generate_variance_distribution(variance_distribution, variance_dist_path, **kwargs)

            # Generate field variability heatmap
            field_heatmap_path = visualizations_dir / f"{self.__class__.__name__}_{self.field_name}_field_heatmap.png"
            self._generate_field_heatmap(field_metrics, field_heatmap_path, **kwargs)

            # Step 5: Save results
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saving results"})

            self.logger.info(f"Saving metrics")

            # Save metrics
            metrics_result = writer.write_metrics(
                metrics=metrics,
                name=f"{self.__class__.__name__}_{self.field_name}_metrics",
                timestamp_in_name=False,
                encryption_key=self.encryption_key if self.use_encryption else None
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
                category=Constants.Artifact_Category_Metrics
            )

            result.add_artifact(
                artifact_type="png",
                path=variance_dist_path,
                description=f"{self.__class__.__name__} {self.field_name} variance distribution histogram",
                category=Constants.Artifact_Category_Visualization
            )

            result.add_artifact(
                artifact_type="png",
                path=field_heatmap_path,
                description=f"{self.__class__.__name__} {self.field_name} field variability heatmap",
                category=Constants.Artifact_Category_Visualization
            )

            # Set success status
            result.status = OperationStatus.SUCCESS

            self.logger.info(f"Group analysis for {self.field_name} completed successfully")

            # Report to reporter if available
            if reporter:
                reporter.add_operation(
                    f"Group analysis for {self.field_name}",
                    details={
                        "total_groups": len(group_keys),
                        "groups_to_aggregate": groups_to_aggregate,
                        "avg_variance": avg_variance
                    }
                )

                reporter.add_artifact(
                    "json",
                    str(metrics_result.path),
                    f"{self.__class__.__name__} {self.field_name} group analysis metrics"
                )

                reporter.add_artifact(
                    "png",
                    str(variance_dist_path),
                    f"{self.__class__.__name__} {self.field_name} variance distribution histogram"
                )

                reporter.add_artifact(
                    "png",
                    str(field_heatmap_path),
                    f"{self.__class__.__name__} {self.field_name} field variability heatmap"
                )

            return result

        except Exception as e:
            error_message = f"Error executing group analysis: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

    def _analyze_group(self, group_df: pd.DataFrame, fields: List[str]) -> Dict[str, Any]:
        """
        Analyze a single group of records.

        Parameters:
        -----------
        group_df : pd.DataFrame
            DataFrame containing records for a single group
        fields : List[str]
            Fields to analyze

        Returns:
        --------
        Dict[str, Any]
            Dictionary with group metrics
        """
        # Initialize metrics
        field_variances = {}
        duplication_ratios = {}

        # Calculate variance and duplication ratio for each field
        for field in fields:
            # Calculate variance for this field
            variance, duplications = self._calculate_field_variance(group_df[field])
            field_variances[field] = variance
            duplication_ratios[field] = duplications

        # Calculate weighted variance
        weighted_variance = self._calculate_weighted_variance(field_variances)

        # Determine if this group should be aggregated
        should_aggregate = self._should_aggregate(weighted_variance, len(group_df))

        # Get max field variance
        max_field_variance = max(field_variances.values()) if field_variances else 0

        return {
            "weighted_variance": weighted_variance,
            "max_field_variance": max_field_variance,
            "total_records": len(group_df),
            "field_variances": field_variances,
            "duplication_ratios": duplication_ratios,
            "should_aggregate": should_aggregate
        }

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

    def _get_unique_values_improved(self, field_series: pd.Series) -> Tuple[Set[Any], Dict[Any, int]]:
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
                        if not hasattr(self, '_minhash_imported'):
                            # Initialize flag and function variable before try block
                            self._minhash_imported = False
                            compute_minhash = None  # Define variable before try to satisfy linter
                            compute_minhash_func = None

                            try:
                                # Dynamically import minhash module when needed
                                from pamola_core.utils.nlp.minhash import compute_minhash
                                compute_minhash_func = compute_minhash  # Assign the imported function
                                self._minhash_imported = True  # Set flag only on success
                            except ImportError:
                                # If import fails (e.g., no pymorphy2), we end up here
                                compute_minhash = None  # Explicitly define in except block for linter
                                self.logger.warning("MinHash library not available, falling back to MD5")
                                self.use_minhash = False  # Make sure we switch to MD5
                                self._minhash_imported = False  # Explicitly mark import as failed
                                compute_minhash_func = lambda _: []  # Assign a dummy function with unused parameter

                            # Save reference to function (imported or dummy) in object attribute
                            self._compute_minhash = compute_minhash_func

                        if self.use_minhash and self._minhash_imported:
                            # Use cached MinHash signatures when possible
                            if text not in self.minhash_cache:
                                self.minhash_cache[text] = self._compute_minhash(text)

                            # Use the first 8 elements of the signature as the key
                            signature = self.minhash_cache[text]
                            signature_str = str(signature[:8])  # Convert to string for consistent keys
                            key = signature_str
                        else:
                            # Fallback to MD5 if MinHash is not available
                            key = hashlib.md5(text.encode('utf-8')).hexdigest()
                    else:
                        # Use MD5 for exact matching of long texts
                        key = hashlib.md5(text.encode('utf-8')).hexdigest()
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
        if self.use_minhash and hasattr(self, '_minhash_imported') and self._minhash_imported:
            minhash_keys = [k for k in counts.keys() if isinstance(k, str) and k.startswith('[')]

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

    def _cluster_minhash_signatures_from_keys(self, signature_keys: List[str]) -> List[List[str]]:
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
                nums = key.strip('[]').split(',')
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
            for key2, sig2 in parsed_signatures[i + 1:]:
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

    def _calculate_weighted_variance(self, field_variances: Dict[str, float]) -> float:
        """
        Calculate weighted variance across all fields.

        Parameters:
        -----------
        field_variances : Dict[str, float]
            Dictionary mapping field names to their variance values

        Returns:
        --------
        float
            Weighted variance across all fields
        """
        weighted_sum = 0
        total_weight = 0

        for field, variance in field_variances.items():
            weight = self.fields_config.get(field, 1)
            weighted_sum += variance * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _should_aggregate(self, weighted_variance: float, group_size: int) -> bool:
        """
        Determine if a group should be aggregated based on variance and size.

        Parameters:
        -----------
        weighted_variance : float
            Weighted variance of the group
        group_size : int
            Number of records in the group

        Returns:
        --------
        bool
            True if the group should be aggregated, False otherwise
        """
        # Apply different thresholds based on group size
        if group_size > self.large_group_threshold:
            return weighted_variance <= self.large_group_variance_threshold
        else:
            return weighted_variance <= self.variance_threshold

    def _calculate_field_metrics(self, group_metrics: Dict[str, Dict[str, Any]], fields: List[str]) -> Dict[
        str, Dict[str, float]]:
        """
        Calculate metrics for each field across all groups.

        Parameters:
        -----------
        group_metrics : Dict[str, Dict[str, Any]]
            Dictionary with metrics for each group
        fields : List[str]
            List of fields to analyze

        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary with metrics for each field
        """
        field_metrics = {}

        for field in fields:
            # Initialize metrics for this field
            field_metrics[field] = {
                "avg_variance": 0,
                "max_variance": 0,
                "avg_duplication_ratio": 0,
                "unique_values_total": 0
            }

            # Collect variances and duplication ratios for this field
            variances = []
            duplication_ratios = []
            unique_values_count = 0

            for group_data in group_metrics.values():
                if field in group_data["field_variances"]:
                    variances.append(group_data["field_variances"][field])

                if field in group_data["duplication_ratios"]:
                    duplication_ratios.append(group_data["duplication_ratios"][field])

                    # Estimate unique values from duplication ratio: total_records / duplication_ratio
                    if group_data["duplication_ratios"][field] > 0:
                        unique_values_count += group_data["total_records"] / group_data["duplication_ratios"][field]

            # Calculate average and max values
            field_metrics[field]["avg_variance"] = float(np.mean(variances)) if variances else 0
            field_metrics[field]["max_variance"] = float(np.max(variances)) if variances else 0
            field_metrics[field]["avg_duplication_ratio"] = float(
                np.mean(duplication_ratios)) if duplication_ratios else 0
            field_metrics[field]["unique_values_total"] = int(unique_values_count)

        return field_metrics

    def _generate_variance_distribution(self, variance_distribution: Dict[str, int], output_path: Path, **kwargs):
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
        categories = ["below_0.1", "0.1_to_0.2", "0.2_to_0.5", "0.5_to_0.8", "above_0.8"]

        # Create labels mapping
        labels_map = {
            "below_0.1": "< 0.1",
            "0.1_to_0.2": "0.1 - 0.2",
            "0.2_to_0.5": "0.2 - 0.5",
            "0.5_to_0.8": "0.5 - 0.8",
            "above_0.8": "> 0.8"
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
                    **kwargs
                )
            except (TypeError, ValueError) as e:
                # Fallback to matplotlib for proper histogram
                plt.figure(figsize=(10, 6))
                plt.bar(list(data.keys()), list(data.values()), color='skyblue')
                plt.title(f"Variability Distribution ({self.field_name})")
                plt.xlabel("Variability Range")
                plt.ylabel("Number of Groups")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()

            self.logger.info(f"Generated variance distribution visualization at {output_path}")
        except Exception as e:
            self.logger.error(f"Error generating variance distribution visualization: {str(e)}")

            # Fallback to basic matplotlib in case of error
            try:
                plt.figure(figsize=(10, 6))
                plt.bar(list(data.keys()), list(data.values()), color='skyblue')
                plt.title(f"Variability Distribution ({self.field_name})")
                plt.xlabel("Variability Range")
                plt.ylabel("Number of Groups")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()
                self.logger.info(f"Generated fallback variance distribution visualization at {output_path}")
            except Exception as e2:
                self.logger.error(f"Failed to generate fallback visualization: {str(e2)}")

    def _generate_field_heatmap(self, field_metrics: Dict[str, Dict[str, float]], output_path: Path, **kwargs):
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
            df_data = pd.DataFrame({
                metric: [data[metric][field] for field in field_names]
                for metric in metric_types
            }, index=field_names)

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
                    **kwargs
                )
            except (TypeError, ValueError) as e:
                # Fallback to matplotlib for the heatmap
                plt.figure(figsize=(12, len(field_names) * 0.8))
                im = plt.imshow(df_data.values, cmap='viridis')

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
                field_avg_variances = {field: metrics["avg_variance"] for field, metrics in field_metrics.items()}

                plt.figure(figsize=(10, 6))
                plt.bar(field_avg_variances.keys(), field_avg_variances.values(), color='skyblue')
                plt.title(f"Field Average Variances ({self.field_name})")
                plt.xlabel("Field")
                plt.ylabel("Average Variance")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()

                self.logger.info(f"Generated fallback field variance visualization at {output_path}")
            except Exception as e2:
                self.logger.error(f"Failed to generate fallback visualization: {str(e2)}")


# Register the operation so it's discoverable
register_operation(GroupAnalyzerOperation)