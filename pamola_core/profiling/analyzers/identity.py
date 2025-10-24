"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Identity Field Profiler Operation
Package:       pamola.pamola_core.profiling.analyzers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides analyzers and operations for profiling identity fields in tabular datasets.
  It includes analysis of identifier consistency, distribution of records per identifier,
  and cross-matching of identifiers for privacy and data quality assessment.

Key Features:
  - Identifier consistency and uniqueness analysis
  - Distribution analysis of records per identifier
  - Cross-matching of identifiers and reference fields
  - Visualization generation for identifier statistics and consistency
  - Robust error handling, progress tracking, and operation logging
  - Caching and efficient repeated analysis
  - Integration with PAMOLA.CORE operation framework for standardized input/output
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd

from pamola_core.profiling.commons.identity_utils import (
    analyze_identifier_distribution,
    analyze_identifier_consistency,
    find_cross_matches,
    compute_identifier_stats,
    generate_consistency_analysis_vis,
    generate_cross_match_distribution_vis,
    generate_field_distribution_vis,
    generate_identifier_statistics_vis,
)
from pamola_core.utils.io import (
    load_data_operation,
    load_settings_operation,
)
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.common.constants import Constants

# Configure logger
logger = logging.getLogger(__name__)


class IdentityAnalyzer:
    """
    Analyzer for identity fields.

    This analyzer provides methods for analyzing identity fields, including
    identifier consistency, distribution of records per identifier, and
    cross-matching of identifiers.
    """

    @staticmethod
    def analyze_identifier_distribution(
        df: pd.DataFrame,
        id_field: str,
        entity_field: Optional[str] = None,
        top_n: int = 15,
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
        return analyze_identifier_distribution(df, id_field, entity_field, top_n)

    @staticmethod
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
        return analyze_identifier_consistency(df, id_field, reference_fields)

    @staticmethod
    def find_cross_matches(
        df: pd.DataFrame,
        id_field: str,
        reference_fields: List[str],
        min_similarity: float = 0.8,
        fuzzy_matching: bool = False,
    ) -> Dict[str, Any]:
        """
        Find cases where reference fields match but identifiers differ.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        id_field : str
            Identifier field to analyze
        reference_fields : List[str]
            Fields that define an entity's identity
        min_similarity : float
            Minimum similarity for fuzzy matching
        fuzzy_matching : bool
            Whether to use fuzzy matching

        Returns:
        --------
        Dict[str, Any]
            Cross-matching analysis results
        """
        return find_cross_matches(
            df, id_field, reference_fields, min_similarity, fuzzy_matching
        )

    @staticmethod
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
        return compute_identifier_stats(df, id_field, entity_field)


class IdentityAnalysisOperationConfig(OperationConfig):
    """Configuration for IdentityAnalysisOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields from BaseOperationConfig
            {
                "type": "object",
                "properties": {
                    # --- Operation-specific fields ---
                    "uid_field": {"type": "string"},
                    "reference_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "id_field": {"type": ["string", "null"], "default": None},
                    "top_n": {"type": "integer", "minimum": 1, "default": 15},
                    "check_cross_matches": {"type": "boolean", "default": True},
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                    },
                    "fuzzy_matching": {"type": "boolean", "default": False},
                },
                "required": ["uid_field", "reference_fields"],
            },
        ],
    }


@register(version="1.0.0")
class IdentityAnalysisOperation(FieldOperation):
    """
    Operation for analyzing identity fields and their consistency.

    This operation analyzes:
    1. Distribution of records per identifier
    2. Consistency between identifiers and reference fields
    3. Cross-matching of identifiers
    """

    def __init__(
        self,
        uid_field: str,
        reference_fields: List[str],
        id_field: Optional[str] = None,
        top_n: int = 15,
        check_cross_matches: bool = True,
        min_similarity: float = 0.8,
        fuzzy_matching: bool = False,
        **kwargs,
    ):
        """
        Initialize the identity analysis operation.

        Parameters
        ----------
        uid_field : str
            Primary identifier field to analyze (e.g., 'UID').
        reference_fields : List[str]
            Fields used to identify an entity (e.g., ['first_name', 'last_name']).
        id_field : Optional[str]
            Entity-level identifier field (optional).
        top_n : int
            Number of top entries to include (default: 15).
        check_cross_matches : bool
            Whether to check for cross-matching (default: True).
        min_similarity : float
            Minimum similarity threshold for fuzzy matching (default: 0.8).
        fuzzy_matching : bool
            Whether to use fuzzy matching (default: False).
        **kwargs : dict
            Additional parameters passed to FieldOperation.
        """
        # --- Default description fallback ---
        kwargs.setdefault(
            "description",
            f"Analysis of identity field '{uid_field}'",
        )

        # --- Build unified config ---
        config = IdentityAnalysisOperationConfig(
            uid_field=uid_field,
            reference_fields=reference_fields or [],
            id_field=id_field,
            top_n=top_n,
            check_cross_matches=check_cross_matches,
            min_similarity=min_similarity,
            fuzzy_matching=fuzzy_matching,
            **kwargs,
        )

        # Inject config into kwargs for parent constructor
        kwargs["config"] = config

        # --- Initialize parent FieldOperation ---
        super().__init__(
            field_name=uid_field,
            **kwargs,
        )

        # --- Save config attributes to instance ---
        for key, value in config.to_dict().items():
            setattr(self, key, value)

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
        Execute the identity analysis operation.

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
            # Initialize global analysis dictionaries
            global identifier_stats, consistency_analysis, distribution_analysis, cross_match_analysis
            identifier_stats = {}
            consistency_analysis = {}
            distribution_analysis = {}
            cross_match_analysis = {}

            # Initialize timing and result
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(f"Starting {self.name} operation at {self.start_time}")
            df = None
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Save configuration to task directory
            self.save_config(task_dir)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Cache check, 2. Validation, 3. Data loading, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
            TOTAL_MAIN_STEPS = 6 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            main_progress = progress_tracker
            current_steps = 0
            if main_progress:
                self.logger.info(
                    f"Setting up progress tracker with {TOTAL_MAIN_STEPS} main steps"
                )
                try:
                    main_progress.total = TOTAL_MAIN_STEPS
                    self._update_progress_tracker(
                        TOTAL_MAIN_STEPS,
                        current_steps,
                        {
                            "step": "Starting identity analysis",
                        },
                        main_progress,
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # Get DataFrame from data source
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )

            # Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                # Step 1: Check if we have a cached result
                if main_progress:
                    current_steps += 1
                    self._update_progress_tracker(
                        TOTAL_MAIN_STEPS, current_steps, "Checking cache", main_progress
                    )

                # Load left dataset for check cache
                df = load_data_operation(
                    data_source, dataset_name, **settings_operation
                )
                if df is None:
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message="No valid DataFrame found in data source",
                    )

                self.logger.info(
                    f"Field: '{self.field_name}' loaded with {len(df)} records."
                )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df=df, reporter=reporter)
                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if main_progress:
                        self._update_progress_tracker(
                            TOTAL_MAIN_STEPS,
                            current_steps,
                            "Complete (cached)",
                            main_progress,
                        )

                    return cache_result

            # Step 2: Data Loading
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Data Loading", main_progress
                )

            try:
                # Load DataFrame
                if df is None:
                    df = load_data_operation(
                        data_source, dataset_name, **settings_operation
                    )
                    if df is None:
                        return OperationResult(
                            status=OperationStatus.ERROR,
                            error_message="No valid DataFrame found in data source",
                        )
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 3: Validation
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Validation", main_progress
                )

            # Validate input parameters
            # Validate main field
            if not self._field_exists(df, self.field_name):
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame",
                )

            # Validate reference fields
            valid_refs, missing_refs = self._validate_reference_fields(df)
            if not valid_refs:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"None of the reference fields {self.reference_fields} found in DataFrame",
                )
            if missing_refs:
                self._log_and_report_missing(
                    reporter,
                    label="reference fields",
                    field_list=missing_refs,
                    context=f"Missing reference fields for {self.field_name}",
                )

            # Validate ID field
            if self.id_field and not self._field_exists(df, self.id_field):
                self._log_and_report_missing(
                    reporter,
                    label="ID field",
                    field_list=[self.id_field],
                    context=f"Missing ID field {self.id_field}",
                )
                valid_id_field = None
            else:
                valid_id_field = self.id_field

            # Log analysis metadata
            reporter.add_operation(
                f"Analyzing identity field: {self.field_name}",
                details={
                    "field_name": self.field_name,
                    "reference_fields": valid_refs,
                    "id_field": valid_id_field,
                    "operation_type": "identity_analysis",
                },
            )

            # Step 4: Processing progress tracker
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Processing data", main_progress
                )

            try:
                self.logger.info(f"Processing with field_name: {self.field_name}")

                # Create child progress tracker for chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=4,
                            description="Identity analysis processing",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Basic identifier statistics
                # Update progress
                if data_tracker:
                    data_tracker.update(
                        1, {"step": "Basic identifier statistics processing"}
                    )
                self.logger.info("Basic identifier statistics processing")

                identifier_stats = IdentityAnalyzer.compute_identifier_stats(
                    df, self.field_name, self.id_field if valid_id_field else None
                )

                self.logger.info("Basic identifier statistics processing complete")

                # Analyze identifier consistency
                # Update progress
                if data_tracker:
                    data_tracker.update(
                        2, {"step": "Analyze identifier consistency processing"}
                    )
                self.logger.info("Analyze identifier consistency processing")

                consistency_analysis = IdentityAnalyzer.analyze_identifier_consistency(
                    df, self.field_name, valid_refs
                )

                self.logger.info("Analyze identifier consistency processing complete")

                # Analyze identifier distribution
                # Update progress
                if data_tracker:
                    data_tracker.update(
                        3, {"step": "Analyze identifier distribution processing"}
                    )
                self.logger.info("Analyze identifier distribution processing")

                if valid_id_field:
                    distribution_analysis = (
                        IdentityAnalyzer.analyze_identifier_distribution(
                            df, self.field_name, self.id_field, self.top_n
                        )
                    )
                    self.logger.info(
                        "Analyze identifier distribution processing complete"
                    )
                else:
                    logger.warning(
                        f"Skipping distribution analysis. ID field not found: {self.id_field}"
                    )
                    reporter.add_operation(
                        f"Skipping distribution analysis for {self.field_name}",
                        status="warning",
                        details={"reason": f"ID field {self.id_field} not found"},
                    )

                # Cross-matching analysis
                # Update progress
                if data_tracker:
                    data_tracker.update(
                        4, {"step": "Analyze cross-matching processing"}
                    )
                self.logger.info("Analyze cross-matching processing")

                if self.check_cross_matches and valid_refs:
                    cross_match_analysis = IdentityAnalyzer.find_cross_matches(
                        df,
                        self.field_name,
                        valid_refs,
                        self.min_similarity,
                        self.fuzzy_matching,
                    )
                    self.logger.info("Analyze cross-matching processing complete")
                else:
                    if not self.check_cross_matches:
                        logger.warning(
                            "Skipping cross-match analysis as per configuration"
                        )
                    elif not valid_refs:
                        logger.warning(
                            "Skipping cross-match analysis. No valid reference fields"
                        )

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass

                self.logger.info(
                    f"Processed data: {len(df)} records, dtype: {df.dtypes}"
                )

                # Log sample of processed data
                if len(df) > 0:
                    self.logger.debug(
                        f"Sample of processed data (first 5 rows): {df.head(5).to_dict(orient='records')}"
                    )
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 5: Metrics Calculation
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS,
                    current_steps,
                    "Metrics Calculation",
                    main_progress,
                )

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Initialize metrics in scope
            metrics = {}

            try:
                # Save identifier statistics
                if identifier_stats and "error" not in identifier_stats:
                    # Generate metrics file name
                    identifier_filename = f"{self.field_name}_{self.name}_identifier_metrics_{operation_timestamp}"

                    # Write metrics to persistent storage/artifact repository
                    identifier_metrics_result = writer.write_metrics(
                        metrics=identifier_stats,
                        name=identifier_filename,
                        timestamp_in_name=False,
                        encryption_key=(
                            self.encryption_key if self.use_encryption else None
                        ),
                    )

                    # Add simple metrics (int, float, str, bool) to the result object
                    for key, value in identifier_stats.items():
                        if isinstance(value, (int, float, str, bool)):
                            result.add_metric(key, value)

                    result.add_artifact(
                        artifact_type="json",
                        path=identifier_metrics_result.path,
                        description=f"{self.name} profiling on {self.field_name} identifier metrics",
                        category=Constants.Artifact_Category_Metrics,
                    )

                    # Add identifier stats to metrics dictionary
                    metrics["identifier_stats"] = identifier_stats
                    metrics["identifier_metrics_result_path"] = (
                        identifier_metrics_result.path
                    )

                # Save consistency analysis results
                if consistency_analysis and "error" not in consistency_analysis:
                    # Generate metrics file name
                    consistency_filename = f"{self.field_name}_{self.name}_consistency_metrics_{operation_timestamp}"

                    consistency_metrics_result = writer.write_metrics(
                        metrics=consistency_analysis,
                        name=consistency_filename,
                        timestamp_in_name=False,
                        encryption_key=(
                            self.encryption_key if self.use_encryption else None
                        ),
                    )

                    # Add simple metrics (int, float, str, bool) to the result object
                    for key, value in consistency_analysis.items():
                        if isinstance(value, (int, float, str, bool)):
                            result.add_metric(key, value)

                    result.add_artifact(
                        artifact_type="json",
                        path=consistency_metrics_result.path,
                        description=f"{self.name} profiling on {self.field_name} consistency metrics",
                        category=Constants.Artifact_Category_Metrics,
                    )

                    # Add consistency analysis to metrics dictionary
                    metrics["consistency_analysis"] = consistency_analysis
                    metrics["consistency_metrics_result_path"] = (
                        consistency_metrics_result.path
                    )

                # Save distribution analysis results
                if distribution_analysis and "error" not in distribution_analysis:
                    # Generate file name for distribution metrics
                    distribution_filename = f"{self.field_name}_{self.name}_distribution_metrics_{operation_timestamp}"

                    # Write metrics to persistent storage/artifact repository
                    distribution_metrics_result = writer.write_metrics(
                        metrics=distribution_analysis,
                        name=distribution_filename,
                        timestamp_in_name=False,
                        encryption_key=(
                            self.encryption_key if self.use_encryption else None
                        ),
                    )

                    # Add simple metrics (int, float, str, bool) to the result object
                    for key, value in distribution_analysis.items():
                        if isinstance(value, (int, float, str, bool)):
                            result.add_metric(key, value)

                    result.add_artifact(
                        artifact_type="json",
                        path=distribution_metrics_result.path,
                        description=f"{self.name} profiling on {self.field_name} distribution metrics",
                        category=Constants.Artifact_Category_Metrics,
                    )

                    # Add distribution analysis to metrics dictionary
                    metrics["distribution_analysis"] = distribution_analysis
                    metrics["distribution_metrics_result_path"] = (
                        distribution_metrics_result.path
                    )

                # Save cross-match analysis results
                if cross_match_analysis and "error" not in cross_match_analysis:
                    # Generate file name for cross-match metrics
                    cross_match_filename = f"{self.field_name}_{self.name}_cross_match_metrics_{operation_timestamp}"

                    # Write metrics to persistent storage/artifact repository
                    cross_match_metrics_result = writer.write_metrics(
                        metrics=cross_match_analysis,
                        name=cross_match_filename,
                        timestamp_in_name=False,
                        encryption_key=(
                            self.encryption_key if self.use_encryption else None
                        ),
                    )

                    # Add simple metrics (int, float, str, bool) to the result object
                    for key, value in cross_match_analysis.items():
                        if isinstance(value, (int, float, str, bool)):
                            result.add_metric(key, value)

                    result.add_artifact(
                        artifact_type="json",
                        path=cross_match_metrics_result.path,
                        description=f"{self.name} profiling on {self.field_name} cross-match metrics",
                        category=Constants.Artifact_Category_Metrics,
                    )

                    # Add cross-match analysis to metrics dictionary
                    metrics["cross_match_analysis"] = cross_match_analysis
                    metrics["cross_match_metrics_result_path"] = (
                        cross_match_metrics_result.path
                    )

                # Report the metrics artifact to the reporter if available
                if reporter:
                    # Add final operation status to reporter
                    reporter.add_operation(
                        f"Analysis of {self.field_name} completed",
                        details={
                            "unique_identifiers": identifier_stats.get(
                                "unique_identifiers", 0
                            ),
                            "consistency_percentage": consistency_analysis.get(
                                "match_percentage", 0
                            ),
                            "reference_fields_used": valid_refs,
                            "cross_matches_found": cross_match_analysis.get(
                                "total_cross_matches", 0
                            ),
                        },
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Visualizations
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS,
                    current_steps,
                    "Generating Visualizations",
                    main_progress,
                )
            # Generate visualizations if required
            # Initialize visualization paths dictionary
            visualization_paths = {}
            if self.generate_visualization and self.visualization_backend is not None:
                try:
                    kwargs_encryption = {
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    }
                    visualization_paths = self._handle_visualizations(
                        identifier_stats=identifier_stats,
                        consistency_analysis=consistency_analysis,
                        distribution_analysis=distribution_analysis,
                        cross_match_analysis=cross_match_analysis,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        vis_theme=self.visualization_theme,
                        vis_backend=self.visualization_backend,
                        vis_strict=self.visualization_strict,
                        vis_timeout=self.visualization_timeout,
                        operation_timestamp=operation_timestamp,
                        **kwargs_encryption,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical
            else:
                self.logger.info(
                    "Skipping visualizations as generate_visualization is False or backend is not set"
                )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=df,
                        metrics=metrics,
                        visualization_paths=visualization_paths,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Cleanup memory
            self._cleanup_memory(df)

            # Record end time
            self.end_time = time.time()

            # Report completion
            if reporter:
                # Create the details dictionary with checks for all values
                details = {
                    "records_processed": self.process_count,
                    "execution_time": (
                        self.end_time - self.start_time
                        if self.end_time and self.start_time
                        else None
                    ),
                }

                # Add the operation to the reporter
                reporter.add_operation(
                    f"Analyzing identity of {self.name} completed",
                    details=details,
                )

            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
            )

            # Set success status
            result.status = OperationStatus.SUCCESS
            return result

        except Exception as e:
            # Handle any unexpected errors
            error_message = f"Error in analyzing identity operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing identity field {self.field_name}: {str(e)}",
            )

    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        # Get basic operation parameters
        parameters = self._get_basic_parameters()

        # Add operation-specific parameters (could be overridden by subclasses)
        parameters.update(self._get_cache_parameters())

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(data)

        # Use the operation_cache utility to generate a consistent cache key
        return self.operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash,
        )

    def _get_basic_parameters(self) -> Dict[str, str]:
        """Get the basic parameters for the cache key generation."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        This method should be overridden by subclasses to provide
        operation-specific parameters for caching.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Base implementation returns minimal parameters
        params = {
            "uid_field": self.uid_field,
            "reference_fields": self.reference_fields or [],
            "id_field": self.id_field,
            "top_n": self.top_n,
            "check_cross_matches": self.check_cross_matches,
            "min_similarity": self.min_similarity,
            "fuzzy_matching": self.fuzzy_matching,
            "use_cache": self.use_cache,
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
            "visualization_theme": self.visualization_theme,
            "visualization_backend": self.visualization_backend,
            "visualization_strict": self.visualization_strict,
            "visualization_timeout": self.visualization_timeout,
            "force_recalculation": self.force_recalculation,
            "generate_visualization": self.generate_visualization,
        }

        return params

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Generate a hash representing the key characteristics of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data for the operation

        Returns
        -------
        str
            Hash string representing the data
        """
        import hashlib
        import json

        try:
            # Generate summary statistics for all columns (numeric and non-numeric)
            characteristics = df.describe(include="all")

            # Convert to JSON string with consistent formatting (ISO for dates)
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback: use length and column data types
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _save_to_cache(
        self,
        original_data: pd.DataFrame,
        metrics: Dict[str, Any],
        visualization_paths: Dict[str, Path],
        task_dir: Path,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_data : pd.DataFrame
            Original input data
        metrics : Dict[str, Any]
            Metrics collected during the operation
        visualization_paths : Dict[str, Path]
            Paths to generated visualizations
        task_dir : Path
            Task directory

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(original_data[self.field_name])

            # Prepare metadata for cache
            operation_params = self._get_basic_parameters()
            operation_params.update(self._get_cache_parameters())

            self.logger.debug(f"Operation parameters for cache: {operation_params}")

            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "parameters": operation_params,
                "data_info": {
                    "original_length": len(original_data),
                    "original_null_count": int(original_data.isna().sum().sum()),
                },
                "visualizations": {
                    k: str(v) for k, v in visualization_paths.items()
                },  # Paths to visualizations
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")

            success = self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(
                    f"Successfully saved {self.field_name} profiling results to cache"
                )
            else:
                self.logger.warning(
                    f"Failed to save {self.field_name} profiling results to cache"
                )

            return success

        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _check_cache(
        self, df: pd.DataFrame, reporter: Any
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for this operation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame for the operation
        reporter : Any
            Reporter object for tracking progress and artifacts

        Returns
        -------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            if self.field_name not in df.columns:
                self.logger.warning(
                    f"Field '{self.field_name}' not found in DataFrame."
                )
                return None

            cache_key = self._generate_cache_key(df[self.field_name])
            self.logger.debug(f"Checking cache for key: {cache_key}")

            cached_result = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(
                f"Using cached result for {self.field_name} of {self.name} profiling"
            )

            result = OperationResult(status=OperationStatus.SUCCESS)
            # Restore cached data
            self._add_cached_metrics(result, cached_result)
            artifacts_restored = self._restore_cached_artifacts(
                result, cached_result, reporter
            )

            # Add cache metadata
            result.add_metric("cached", True)
            result.add_metric("cache_key", cache_key)
            result.add_metric(
                "cache_timestamp", cached_result.get("timestamp", "unknown")
            )
            result.add_metric("artifacts_restored", artifacts_restored)

            if reporter:
                reporter.add_operation(
                    f"{self.name} profiling of {self.field_name} (cached)",
                    details={
                        "field_name": self.field_name,
                        "cached": True,
                        "artifacts_restored": artifacts_restored,
                    },
                )

            self.logger.info(
                f"Cache hit successful: restored {artifacts_restored} artifacts"
            )
            return result

        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _add_cached_metrics(self, result: OperationResult, cached: dict):
        """
        Add cached scalar metrics (int, float, str, bool) to the OperationResult.

        Parameters
        ----------
        result : OperationResult
            The result object to update.
        cached : dict
            Cached result dictionary from cache manager.
        """
        for key, value in cached.get("metrics", {}).items():
            if isinstance(value, (int, float, str, bool)):
                result.add_metric(key, value)

    def _restore_cached_artifacts(
        self, result: OperationResult, cached: dict, reporter: Optional[Any]
    ) -> int:
        """
        Restore artifacts (output, metrics, visualizations) from cached result if files exist.

        Parameters
        ----------
        result : OperationResult
            OperationResult object to update with restored artifacts.
        cached : dict
            Cached result dictionary from cache manager.
        reporter : Optional[Any]
            Optional reporter object for tracking operation-level artifacts.

        Returns
        -------
        int
            Number of artifacts successfully restored.
        """
        artifacts_restored = 0

        def restore_file_artifact(
            path: Union[str, Path], artifact_type: str, desc_suffix: str, category: str
        ):
            """
            Restore a single artifact from a file path if it exists.

            Parameters
            ----------
            path : Union[str, Path]
                Path to the artifact file.
            artifact_type : str
                Type of the artifact (e.g., 'json', 'csv', 'png').
            desc_suffix : str
                Description suffix (e.g., 'visualization', 'metrics').
            category : str
                Artifact category (e.g., output, metrics, visualization).
            """
            nonlocal artifacts_restored
            if not path:
                return

            artifact_path = Path(path)
            if artifact_path.exists():
                result.add_artifact(
                    artifact_type=artifact_type,
                    path=artifact_path,
                    description=f"{self.field_name} {desc_suffix} (cached)",
                    category=category,
                )
                artifacts_restored += 1

                if reporter:
                    reporter.add_operation(
                        f"{self.field_name} {desc_suffix} (cached)",
                        details={
                            "artifact_type": artifact_type,
                            "path": str(artifact_path),
                        },
                    )
            else:
                self.logger.warning(f"Cached file not found: {artifact_path}")

        # Restore visualizations
        for viz_type, path_str in cached.get("visualizations", {}).items():
            restore_file_artifact(
                path_str,
                "png",
                f"{viz_type} visualization",
                Constants.Artifact_Category_Visualization,
            )

        return artifacts_restored

    def _handle_visualizations(
        self,
        identifier_stats: Dict[str, Any],
        consistency_analysis: Dict[str, Any],
        distribution_analysis: Dict[str, Any],
        cross_match_analysis: Dict[str, Any],
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate and save visualizations with thread-safe context support.

        Parameters:
        -----------
        identifier_stats : Dict[str, Any]
            Statistics related to the identifier
        consistency_analysis : Dict[str, Any]
            Analysis results for consistency
        distribution_analysis : Dict[str, Any]
            Analysis results for distribution
        cross_match_analysis : Dict[str, Any]
            Analysis results for cross-matching
        task_dir : Path
            The task directory
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        operation_timestamp : str, optional
            Timestamp for the operation (default: current time)
        **kwargs: Any
            Additional keyword arguments for visualization functions.
        """
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
                    f"[DIAG] Field: {self.field_name}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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

                    # Generate visualizations with context parameters
                    visualization_paths = self._generate_visualizations(
                        identifier_stats=identifier_stats,
                        consistency_analysis=consistency_analysis,
                        distribution_analysis=distribution_analysis,
                        cross_match_analysis=cross_match_analysis,
                        task_dir=task_dir,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend or "plotly",
                        vis_strict=vis_strict,
                        progress_tracker=viz_progress,
                        timestamp=operation_timestamp,  # Pass the same timestamp
                        **kwargs,
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
                name=f"VizThread-{self.field_name}",
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
                description=f"{self.field_name} {viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_operation(
                    f"{self.field_name} {viz_type} visualization",
                    details={"artifact_type": "png", "path": str(path)},
                )

        return visualization_paths

    def _generate_visualizations(
        self,
        identifier_stats: Dict[str, Any],
        consistency_analysis: Dict[str, Any],
        distribution_analysis: Dict[str, Any],
        cross_match_analysis: Dict[str, Any],
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Generate required visualizations for the merge operation using visualization utilities.

        Parameters
        ----------
        identifier_stats : Dict[str, Any]
            Statistics related to the identifier.
        consistency_analysis : Dict[str, Any]
            Analysis results for consistency.
        distribution_analysis : Dict[str, Any]
            Analysis results for distribution.
        cross_match_analysis : Dict[str, Any]
            Analysis results for cross-matching.
        task_dir : Path
            The base directory where all task-related outputs (including visualizations) will be saved.
        vis_theme : Optional[str]
            The theme to use for visualizations.
        vis_backend : Optional[str]
            The backend to use for rendering visualizations.
        vis_strict : bool
            Whether to enforce strict visualization rules.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Tracker for monitoring progress of the visualization generation.
        timestamp : Optional[str]
            Timestamp to include in visualization filenames.
        **kwargs : Any
            Additional keyword arguments for visualization functions.

        Returns
        -------
        dict
            A dictionary mapping visualization types to their corresponding file paths.
        """
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        visualization_paths = {}

        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if visualization should be skipped
        if vis_backend is None:
            self.logger.info(
                f"Skipping visualization for {self.field_name} (backend=None)"
            )
            return visualization_paths

        self.logger.info(
            f"[VIZ] Starting visualization generation for {self.field_name}"
        )
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: "
                f"identifier_stats: {len(identifier_stats)}, "
                f"consistency_analysis: {len(consistency_analysis)}, "
                f"distribution_analysis: {len(distribution_analysis)}, "
                f"cross_match_analysis: {len(cross_match_analysis)}"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # 1. Identifier Statistics (bar chart)
            if identifier_stats and "error" not in identifier_stats:
                visualization_paths.update(
                    generate_identifier_statistics_vis(
                        identifier_stats,
                        field_label=self.field_name,
                        operation_name=self.name,
                        task_dir=viz_dir,
                        timestamp=timestamp,
                        theme=vis_theme,
                        backend=vis_backend,
                        strict=vis_strict,
                        visualization_paths=visualization_paths,
                        **kwargs,
                    )
                )

            # 2. Consistency analysis (bar chart)
            if consistency_analysis and "error" not in consistency_analysis:
                visualization_paths.update(
                    generate_consistency_analysis_vis(
                        consistency_analysis,
                        field_label=self.field_name,
                        operation_name=self.name,
                        task_dir=viz_dir,
                        timestamp=timestamp,
                        theme=vis_theme,
                        backend=vis_backend,
                        strict=vis_strict,
                        visualization_paths=visualization_paths,
                        **kwargs,
                    )
                )

            # 3. Distribution analysis (bar chart)
            if distribution_analysis and "error" not in distribution_analysis:
                visualization_paths.update(
                    generate_field_distribution_vis(
                        distribution_analysis,
                        field_label=self.field_name,
                        operation_name=self.name,
                        task_dir=viz_dir,
                        timestamp=timestamp,
                        top_n=self.top_n,
                        theme=vis_theme,
                        backend=vis_backend,
                        strict=vis_strict,
                        visualization_paths=visualization_paths,
                        **kwargs,
                    )
                )

            # 4. Cross-match distribution (bar chart)
            if cross_match_analysis and "error" not in cross_match_analysis:
                visualization_paths.update(
                    generate_cross_match_distribution_vis(
                        cross_match_analysis,
                        field_label=self.field_name,
                        operation_name=self.name,
                        task_dir=viz_dir,
                        timestamp=timestamp,
                        theme=vis_theme,
                        backend=vis_backend,
                        strict=vis_strict,
                        visualization_paths=visualization_paths,
                        **kwargs,
                    )
                )

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(3, {"step": "Finalizing visualizations"})

        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")

        return visualization_paths

    def _update_progress_tracker(
        self,
        TOTAL_MAIN_STEPS: int,
        n: int,
        step_name: str,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> None:
        """
        Helper to update progress tracker for the step.
        """
        if progress_tracker:
            progress_tracker.total = TOTAL_MAIN_STEPS  # Ensure total steps is set
            progress_tracker.update(
                n,
                {
                    "step": step_name,
                    "operation": f"{self.name}",
                    "uid_field": f"{self.uid_field}",
                    "reference_fields": f"{self.reference_fields}",
                },
            )

    def _field_exists(self, df: pd.DataFrame, field: str) -> bool:
        """
        Check if a given field exists in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to check.
        - field (str): The field/column name.

        Returns:
        - bool: True if field exists, False otherwise.
        """
        return field in df.columns

    def _validate_reference_fields(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """
        Validate which reference fields exist in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - Tuple[List[str], List[str]]: A tuple containing the list of valid and missing reference fields.
        """
        valid = [f for f in self.reference_fields if f in df.columns]
        missing = list(set(self.reference_fields) - set(valid))
        return valid, missing

    def _log_and_report_missing(
        self, reporter, label: str, field_list: List[str], context: str
    ):
        """
        Log a warning and report missing fields.

        Parameters:
        - reporter: Reporter object to log and track operations.
        - label (str): A human-readable label for the type of fields (e.g. "reference fields", "ID field").
        - field_list (List[str]): List of missing fields.
        - context (str): Contextual message for the reporter log.
        """
        logger.warning(f"{label.capitalize()} are missing: {field_list}")
        reporter.add_operation(
            context, status="warning", details={"missing_fields": field_list}
        )

    def _cleanup_memory(
        self,
        original_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        references and optionally calling garbage collection.

        Parameters:
        -----------
        original_df : pd.DataFrame, optional
            Original DataFrame to clear from memory
        """
        # Delete references
        if original_df is not None:
            del original_df

        # Clear operation cache
        if hasattr(self, "operation_cache"):
            self.operation_cache = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Optional: Force garbage collection for large datasets
        # Uncomment if memory pressure is an issue
        # import gc
        # gc.collect()

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare directories for artifacts following PAMOLA.CORE conventions.

        Parameters:
        -----------
        task_dir : Path
            Root task directory

        Returns:
        --------
        Dict[str, Path]
            Dictionary with prepared directories
        """
        directories = {}

        # Create standard directories following PAMOLA.CORE conventions
        directories["root"] = task_dir
        directories["output"] = task_dir / "output"
        directories["dictionaries"] = task_dir / "dictionaries"
        directories["logs"] = task_dir / "logs"
        directories["cache"] = task_dir / "cache"

        # Ensure all directories exist
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories


def analyze_identities(
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    identity_fields: Dict[str, Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, OperationResult]:
    """
    Analyze multiple identity fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    identity_fields : Dict[str, Dict[str, Any]], optional
        Dictionary mapping field names to their configuration. Each configuration
        should include 'reference_fields' (list) and optionally 'id_field' (str).
    **kwargs : dict
        Additional parameters for the operations:
        - top_n: int, number of top entries to include (default: 15)
        - check_cross_matches: bool, whether to analyze cross matches (default: True)

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    dataset_name = kwargs.get("dataset_name", "main")
    settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
    df = load_data_operation(data_source, dataset_name, **settings_operation)
    # Use get_dataframe safely
    if df is None:
        reporter.add_operation(
            "Identity fields analysis",
            status="error",
            details={"error": "No valid DataFrame found in data source"},
        )
        return {}

    # If no identity fields specified, try to detect them (this is a simplified approach)
    if identity_fields is None:
        identity_fields = {}

        # Look for potential ID fields
        potential_id_fields = [
            col
            for col in df.columns
            if "id" in col.lower() or "uuid" in col.lower() or "uid" in col.lower()
        ]

        # Look for potential reference fields (name fields, dates, etc.)
        potential_reference_fields = [
            col
            for col in df.columns
            if "name" in col.lower()
            or "date" in col.lower()
            or "birth" in col.lower()
            or "gender" in col.lower()
        ]

        # Create a simple configuration for detected ID fields
        for id_field in potential_id_fields:
            entity_field = None
            # Try to find a related entity field for this ID field
            for other_id in potential_id_fields:
                if other_id != id_field:
                    entity_field = other_id
                    break

            identity_fields[id_field] = {
                "reference_fields": potential_reference_fields,
                "id_field": entity_field,
            }

    # Report on fields to be analyzed
    reporter.add_operation(
        "Identity fields analysis",
        details={
            "fields_count": len(identity_fields),
            "fields": list(identity_fields.keys()),
            "parameters": {
                k: v
                for k, v in kwargs.items()
                if isinstance(v, (str, int, float, bool))
            },
        },
    )

    # Track progress if enabled
    track_progress = kwargs.get("track_progress", True)
    overall_tracker = None

    if track_progress and identity_fields:
        overall_tracker = HierarchicalProgressTracker(
            total=len(identity_fields),
            description=f"Analyzing {len(identity_fields)} identity fields",
            unit="fields",
            track_memory=True,
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, (field, config) in enumerate(identity_fields.items()):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(
                        0,
                        {"field": field, "progress": f"{i + 1}/{len(identity_fields)}"},
                    )

                logger.info(f"Analyzing identity field: {field}")

                # Get configuration for this field
                reference_fields = config.get("reference_fields", [])
                id_field = config.get("id_field")

                # Create and execute operation
                operation = IdentityAnalysisOperation(
                    field, reference_fields=reference_fields, id_field=id_field
                )
                result = operation.execute(data_source, task_dir, reporter, **kwargs)

                # Store result
                results[field] = result

                # Update overall tracker after successful analysis
                if overall_tracker:
                    if result.status == OperationStatus.SUCCESS:
                        overall_tracker.update(
                            1, {"field": field, "status": "completed"}
                        )
                    else:
                        overall_tracker.update(
                            1,
                            {
                                "field": field,
                                "status": "error",
                                "error": result.error_message,
                            },
                        )

            except Exception as e:
                logger.error(
                    f"Error analyzing identity field {field}: {e}", exc_info=True
                )

                reporter.add_operation(
                    f"Analyzing {field} field",
                    status="error",
                    details={"error": str(e)},
                )

                # Update overall tracker in case of error
                if overall_tracker:
                    overall_tracker.update(1, {"field": field, "status": "error"})

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(
        1 for r in results.values() if r.status == OperationStatus.SUCCESS
    )
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    reporter.add_operation(
        "Identity fields analysis completed",
        details={
            "fields_analyzed": len(results),
            "successful": success_count,
            "failed": error_count,
        },
    )

    return results
