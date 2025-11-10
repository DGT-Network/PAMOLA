"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Categorical Generalization Operation (Facade)
Package:       pamola_core.anonymization.generalization
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
Updated:       2025-01-24
License:       BSD 3-Clause

Description:
   This module provides a minimal facade for categorical generalization operations,
   delegating strategy implementation to specialized modules while maintaining
   compatibility with the PAMOLA.CORE operation framework.

Purpose:
   Acts as the main entry point for categorical generalization, handling:
   - Configuration validation and management
   - Strategy selection and execution
   - Metrics collection and reporting
   - Integration with the operation framework
   - Dask support for large datasets

Key Features:
   - Minimal facade pattern for maintainability
   - Delegates to strategy implementations
   - Thread-safe operation with proper state management
   - Automatic Dask switching for large datasets
   - State reset after execution
   - Category mapping export as artifacts

Dependencies:
   - pandas: Data manipulation
   - dask: Distributed processing (optional)
   - Framework utilities from pamola_core.utils.ops
   - Strategy implementations from categorical_strategies
   - Configuration from categorical_config
"""

import hashlib
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import dask.dataframe as dd

# Base class
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Configuration and strategies from commons
from pamola_core.anonymization.commons.categorical_config import (
    DEFAULT_FREQ_THRESHOLD,
    DEFAULT_MAX_CATEGORIES_FOR_DIVERSITY,
    DEFAULT_MAX_HIERARCHY_CHILDREN_DISPLAY,
    DEFAULT_MAX_SUPPRESSION_RATE,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_GROUP_SIZE,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS,
    MAX_CATEGORIES,
    OPERATION_NAME,
    GeneralizationStrategy,
    GroupRareAs,
    OperationMode,
    NullStrategy,
    TextNormalization,
    get_strategy_params,
)
from pamola_core.anonymization.commons.categorical_strategies import (
    _apply_rare_value_template,
    apply_frequency_based,
    apply_hierarchy,
    apply_merge_low_freq,
)
from pamola_core.anonymization.commons.category_utils import (
    process_dataframe_using_chunk,
    process_dataframe_using_dask,
    process_dataframe_using_joblib,
)

# Commons - categories
from pamola_core.anonymization.commons.category_mapping import (
    DEFAULT_UNKNOWN_TEMPLATE,
    DEFAULT_UNKNOWN_VALUE,
)
from pamola_core.anonymization.commons.category_utils import (
    analyze_category_distribution,
    calculate_semantic_diversity_safe,
)
from pamola_core.anonymization.commons.data_utils import process_nulls
from pamola_core.anonymization.commons.hierarchy_dictionary import HierarchyDictionary

# Commons - metrics
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
    get_process_summary_message,
    calculate_categorical_information_loss,
    calculate_generalization_height,
)

# Commons - privacy metrics
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_batch_metrics,
    check_anonymization_thresholds,
    get_process_summary,
    calculate_suppression_rate,
    calculate_generalization_level,
)

# Validation utilities
from pamola_core.anonymization.commons.validation_utils import (
    validate_categorical_field,
    FieldValueError,
)

# Commons - visualization
from pamola_core.anonymization.commons.visualization_utils import (
    create_metrics_overview_visualization,
    create_category_distribution_comparison,
    create_comparison_visualization,
    create_hierarchy_sunburst,
    sample_large_dataset,
)

# Framework utilities
from pamola_core.anonymization.schemas.categorical_op_schema import CategoricalGeneralizationConfig
from pamola_core.common.constants import Constants
from pamola_core.utils.io import load_settings_operation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.helpers import filter_used_kwargs


@register(version="1.0.0")
class CategoricalGeneralizationOperation(AnonymizationOperation):
    """
    Categorical generalization operation for data anonymization.

    This class provides a facade for categorical generalization strategies,
    delegating actual generalization logic to strategy implementations while
    handling framework integration, configuration validation, and state management.

    The operation supports multiple generalization strategies:
    - Hierarchy-based generalization using external dictionaries
    - Frequency-based category merging
    - Low-frequency category grouping
    """

    def __init__(
        self,
        # Required fields
        field_name: str,
        strategy: str = GeneralizationStrategy.HIERARCHY.value,
        # Dictionary parameters
        external_dictionary_path: Optional[str] = None,
        dictionary_format: str = "auto",
        hierarchy_level: int = 1,
        # Frequency-based parameters
        merge_low_freq: bool = False,
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
        freq_threshold: float = DEFAULT_FREQ_THRESHOLD,
        max_categories: int = MAX_CATEGORIES,
        # Unknown handling
        allow_unknown: bool = True,
        unknown_value: str = "OTHER",
        group_rare_as: str = GroupRareAs.OTHER.value,
        rare_value_template: str = "OTHER_{n}",
        # Text processing
        text_normalization: str = TextNormalization.BASIC.value,
        case_sensitive: bool = False,
        fuzzy_matching: bool = False,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        # Privacy thresholds
        privacy_check_enabled: bool = True,
        min_acceptable_k: int = 5,
        max_acceptable_disclosure_risk: float = 0.2,
        quasi_identifiers: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize categorical generalization operation.

        Parameters
        ----------
        field_name : str
            Target field name for generalization
        strategy : str, default="hierarchy"
            Generalization strategy (hierarchy|merge_low_freq|frequency_based)
        external_dictionary_path : str, optional
            Path to external hierarchy dictionary file
        dictionary_format : str, default="auto"
            Dictionary file format (auto|json|csv)
        hierarchy_level : int, default=1
            Target hierarchy level (1 to MAX_HIERARCHY_LEVELS)
        merge_low_freq : bool, default=False
            Enable low-frequency category merging
        min_group_size : int, default=DEFAULT_MIN_GROUP_SIZE
            Minimum group size for privacy
        freq_threshold : float, default=DEFAULT_FREQ_THRESHOLD
            Frequency threshold for category preservation (0-1)
        max_categories : int, default=MAX_CATEGORIES
            Maximum number of categories to preserve
        allow_unknown : bool, default=True
            Allow unknown values in output
        unknown_value : str, default="OTHER"
            Placeholder for unknown values
        group_rare_as : str, default="OTHER"
            Rare category grouping strategy
        rare_value_template : str, default="OTHER_{n}"
            Template for numbered rare values (must contain {n})
        text_normalization : str, default="basic"
            Text normalization level
        case_sensitive : bool, default=False
            Case-sensitive category matching
        fuzzy_matching : bool, default=False
            Enable fuzzy string matching
        similarity_threshold : float, default=0.85
            Similarity threshold for fuzzy matching (0-1)
        privacy_check_enabled : bool, default=True
            Enable privacy validation checks
        min_acceptable_k : int, default=5
            Minimum k-anonymity threshold (≥2)
        max_acceptable_disclosure_risk : float, default=0.2
            Maximum disclosure risk threshold (0-1)
        quasi_identifiers : List[str], optional
            List of quasi-identifier field names
        **kwargs
            Additional keyword arguments passed to AnonymizationOperation.
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Categorical generalization for '{field_name}' using {strategy} strategy",
        )

        # Build config object (if used for schema/validation)
        config = CategoricalGeneralizationConfig(
            field_name=field_name,
            strategy=strategy,
            external_dictionary_path=external_dictionary_path,
            dictionary_format=dictionary_format,
            hierarchy_level=hierarchy_level,
            merge_low_freq=merge_low_freq,
            min_group_size=min_group_size,
            freq_threshold=freq_threshold,
            max_categories=max_categories,
            allow_unknown=allow_unknown,
            unknown_value=unknown_value,
            group_rare_as=group_rare_as,
            rare_value_template=rare_value_template,
            text_normalization=text_normalization,
            case_sensitive=case_sensitive,
            fuzzy_matching=fuzzy_matching,
            similarity_threshold=similarity_threshold,
            privacy_check_enabled=privacy_check_enabled,
            min_acceptable_k=min_acceptable_k,
            max_acceptable_disclosure_risk=max_acceptable_disclosure_risk,
            quasi_identifiers=quasi_identifiers,
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize base AnonymizationOperation
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # Save config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)
            self.process_kwargs[k] = v

        # Extract strategy-specific parameters
        self.strategy_params: Dict[str, Any] = get_strategy_params(config._params)

        # Operation metadata
        self.operation_id = self._generate_trace_id()
        self.operation_name = self.__class__.__name__

        self._hierarchy_cache: Dict[str, Any] = {}
        self._category_mapping: Optional[Dict[str, str]] = {}
        self._hierarchy_info: Optional[Dict[str, Any]] = {}
        self._fuzzy_matches: int = 0
        self._unknown_values: set[str] = set()

        # Prepare processing parameters
        self._prepare_process_kwargs()
        self._metrics = {}

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the operation with timing and error handling.

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
            # Start timing
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(
                f"Starting {self.operation_name} operation at {self.start_time}"
            )

            df = None
            # Initialize result object
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Create writer for consistent output handling
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Save operation configuration
            self.save_config(task_dir)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Load settings operation
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Cache check, 2. Data Loading & Validation, 3. Prepare output field, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
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
                    main_progress.update(
                        current_steps,
                        {
                            "step": f"Starting {self.operation_name}",
                            "field": self.field_name,
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # Step 1: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                try:
                    if main_progress:
                        current_steps += 1
                        main_progress.update(
                            current_steps,
                            {"step": "Checking cache", "field": self.field_name},
                        )
                    # Load data for cache check
                    df = self._validate_and_get_dataframe(
                        data_source, dataset_name, **settings_operation
                    )

                    self.logger.info("Checking operation cache...")
                    cache_result = self._check_cache(df, reporter)

                    if cache_result:
                        self.logger.info(
                            f"Using cached result for {self.field_name} generalization"
                        )

                        # Update progress
                        if main_progress:
                            main_progress.update(
                                current_steps,
                                {"step": "Complete (cached)", "field": self.field_name},
                            )

                        # Report cache hit to reporter
                        if reporter:
                            reporter.add_operation(
                                f"Categorical generalization of {self.field_name} (from cache)",
                                details={"cached": True},
                            )

                        return cache_result
                except Exception as e:
                    error_message = f"Error checking cache: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            # Step 2: Data Loading & Validation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Data Loading", "field": self.field_name},
                )

            # Validate and get dataframe
            try:
                if df is None:
                    self.logger.info(f"Loading data for field '{self.field_name}'")
                    df = self._validate_and_get_dataframe(
                        data_source, dataset_name, **settings_operation
                    )

                # Validate field is suitable for categorical operations
                is_valid, validation_details = validate_categorical_field(
                    df,
                    self.field_name,
                    allow_null=(self.null_strategy != NullStrategy.ERROR.value),
                    check_distribution=True,
                    logger_instance=self.logger,
                )

                if not is_valid:
                    errors = validation_details.get(
                        "errors", ["Unknown validation error"]
                    )
                    raise FieldValueError(
                        self.field_name,
                        reason="; ".join(errors),
                        invalid_count=validation_details.get("invalid_count"),
                        examples=validation_details.get("sample_invalid"),
                    )

                # Log validation warnings if any
                if validation_details.get("warnings"):
                    for warning in validation_details["warnings"]:
                        self.logger.warning(f"Validation warning: {warning}")
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 3: Prepare output field
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Preparing output field", "field": self.field_name},
                )

            try:
                self.output_field_name = self._prepare_output_field(df)
                self.logger.info(f"Prepared output_field: '{self.output_field_name}'")
                self._report_operation_details(reporter, self.output_field_name)
            except Exception as e:
                error_message = f"Preparing output field error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 4: Processing
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Processing", "field": self.field_name}
                )

            try:
                # Copy original data for processing
                original_data = df[self.field_name].copy(deep=True)

                # Create child progress tracker for Chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=3,
                            description="Processing dataframe",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Apply conditional filtering
                self.filter_mask, filtered_df = self._apply_conditional_filtering(df)

                # Process the filtered data only if not empty
                if not filtered_df.empty:
                    processed_df = self._process_data_with_config(
                        df=filtered_df,
                        is_use_batch_dask=True,
                        progress_tracker=data_tracker,
                    )
                else:
                    self.logger.warning(
                        "Filtered DataFrame is empty. Skipping _process_data_with_config."
                    )
                    processed_df = df.copy(deep=True)
                    processed_df[self.output_field_name] = original_data

                # Handle vulnerable records if k-anonymity is enabled
                if self.ka_risk_field and self.ka_risk_field in df.columns:
                    processed_df = self._handle_vulnerable_records(
                        processed_df, self.output_field_name
                    )

                # Get the anonymized data
                anonymized_data = processed_df[self.output_field_name]

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Record end time after processing metrics
            self.end_time = time.time()
            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Step 5: Metrics Calculation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Metrics Calculation", "field": self.field_name},
                )

            # Initialize metrics in scope
            metrics = {}

            try:
                metrics = self._collect_all_metrics(
                    original_data, anonymized_data, self.filter_mask
                )

                # Generate metrics file name (in self.name existed field_name)
                metrics_file_name = f"{self.field_name}_{self.operation_name}_metrics_{operation_timestamp}"

                # Save metrics using writer
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        self.encryption_key if self.use_encryption else None
                    ),
                )

                # Add metrics to result
                for key, value in metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        result.add_metric(key, value)

                # Register metrics artifact
                result.add_artifact(
                    artifact_type="json",
                    path=metrics_result.path,
                    description=f"{self.field_name} generalization metrics",
                    category=Constants.Artifact_Category_Metrics,
                )

                # Report artifact
                if reporter:
                    reporter.add_operation(
                        f"{self.field_name} generalization metrics",
                        details={
                            "artifact_type": "json",
                            "path": str(metrics_result.path),
                        },
                    )

                # Log summary
                summary = get_process_summary(metrics.get("privacy_metrics", {}))
                for key, message in summary.items():
                    self.logger.info(f"{key}: {message}")

                # Save category mapping artifact if available
                mapping_result_path = None
                if self._category_mapping:
                    try:
                        mapping_result_path = self._save_mapping_artifact(writer)
                        if mapping_result_path:
                            result.add_artifact(
                                artifact_type="json",
                                path=mapping_result_path,
                                description=f"Category mapping for {self.field_name}",
                                category=Constants.Artifact_Category_Mapping,
                                tags=["mapping", "categories", self.field_name],
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to save category mapping: {e}")
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Visualizations
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Generating Visualizations", "field": self.field_name},
                )

            # Generate visualizations if required
            # Initialize visualization paths dictionary
            visualization_paths = {}
            if self.generate_visualization and self.visualization_backend is not None:
                try:
                    # Store metrics for later use
                    self._metrics = metrics

                    kwargs_encryption = {
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    }
                    visualization_paths = self._handle_visualizations(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
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

            # Step 7: Save Output Data
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Save Output Data", "field": self.field_name},
                )

            # Save output data if required
            output_result_path = None
            if self.save_output:
                try:
                    safe_kwargs = filter_used_kwargs(kwargs, self._save_output_data)
                    output_result_path = self._save_output_data(
                        result_df=processed_df,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        timestamp=operation_timestamp,
                        use_encryption=self.use_encryption,
                        **safe_kwargs,
                    )
                except Exception as e:
                    error_message = f"Error saving output data: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
                        metrics=metrics,
                        task_dir=task_dir,
                        visualization_paths=visualization_paths,
                        metrics_result_path=str(metrics_result.path),
                        output_result_path=output_result_path,
                        mapping_result_path=mapping_result_path,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                processed_df=processed_df,
                original_data=original_data,
                anonymized_data=anonymized_data,
            )

            # Finalize timing
            self.end_time = time.time()

            # Report completion
            if reporter:
                reporter.add_operation(
                    f"Categorical generalization of {self.field_name} completed",
                    details={
                        "records_processed": self.process_count,
                        "execution_time": self.end_time - self.start_time,
                        "records_filtered": len(filtered_df),
                        "vulnerable_records_handled": metrics.get(
                            "vulnerable_records", 0
                        ),
                    },
                )

            # Set success status
            result.status = OperationStatus.SUCCESS
            self.logger.info(
                f"Processing completed {self.operation_name} operation in {self.end_time - self.start_time:.2f} seconds"
            )

            return result
        except Exception as e:
            error_message = f"Error in categorical generalization operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_message,
                exception=e,
            )
        finally:
            # Always reset state after execution
            self.reset_state()

    def _process_data_with_config(
        self,
        df: pd.DataFrame,
        is_use_batch_dask: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
    ) -> pd.DataFrame:
        """
        Handle processing of the dataframe, including chunk-wise or full processing.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
        is_use_batch_dask : bool
            Whether to use Dask for batch processing
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        pd.DataFrame
            The processed dataframe
        """
        # Check if dataframe is empty
        if len(df) == 0:
            self.logger.warning("Empty DataFrame provided, returning as is")
            return df

        # Handle null values based on strategy
        if self.null_strategy != "PRESERVE":
            df[self.field_name] = process_nulls(
                df[self.field_name],
                strategy=self.null_strategy.upper(),
                anonymize_value=self.unknown_value,
            )

        processed_df = None
        flag_processed = False
        self.logger.info("Process with config")

        # For larger dataframes, check if we should use parallel processing
        if not flag_processed and self.use_dask:
            try:
                self.logger.info("Parallel Enabled")
                self.logger.info("Parallel Engine: Dask")
                self.logger.info(f"Parallel Workers: {self.npartitions}")
                self.logger.info(
                    f"Using dask processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Setting up dask processing"})

                self.logger.info("Process using Dask")

                # Process with Dask - delegate to subclass method
                if is_use_batch_dask:
                    (
                        processed_df,
                        flag_processed,
                        all_category_mapping,
                        all_hierarchy_info,
                        all_hierarchy_cache,
                        all_fuzzy_matches,
                        all_unknown_values,
                    ) = process_dataframe_using_dask(
                        df=df,
                        process_function=self.process_batch_dask,
                        is_use_batch_dask=is_use_batch_dask,
                        progress_tracker=progress_tracker,
                        task_logger=self.logger,
                        **self.process_kwargs,
                    )
                else:
                    (
                        processed_df,
                        flag_processed,
                        all_category_mapping,
                        all_hierarchy_info,
                        all_hierarchy_cache,
                        all_fuzzy_matches,
                        all_unknown_values,
                    ) = process_dataframe_using_dask(
                        df=df,
                        process_function=self.process_batch,
                        is_use_batch_dask=is_use_batch_dask,
                        progress_tracker=progress_tracker,
                        task_logger=self.logger,
                        **self.process_kwargs,
                    )

                if flag_processed:
                    self.logger.info("Completed using Dask")

            except Exception as e:
                self.logger.warning(
                    f"Error in dask processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and self.use_vectorization:
            try:
                self.logger.info("Parallel Enabled")
                self.logger.info("Parallel Engine: Joblib")
                self.logger.info(f"Parallel Workers: {self.parallel_processes}")
                self.logger.info(
                    f"Using vectorized processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(
                        0, {"step": "Setting up vectorized processing"}
                    )

                self.logger.info("Process using Joblib")

                (
                    processed_df,
                    flag_processed,
                    all_category_mapping,
                    all_hierarchy_info,
                    all_hierarchy_cache,
                    all_fuzzy_matches,
                    all_unknown_values,
                ) = process_dataframe_using_joblib(
                    df=df,
                    process_function=self.process_batch,
                    progress_tracker=progress_tracker,
                    task_logger=self.logger,
                    **self.process_kwargs,
                )

                if flag_processed:
                    self.logger.info("Completed using Joblib")

            except Exception as e:
                self.logger.warning(
                    f"Error in vectorized processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and self.chunk_size > 1:
            try:
                # Regular chunk processing
                self.logger.info(
                    f"Processing in chunks with chunk size {self.chunk_size}"
                )
                total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
                self.logger.info(f"Total chunks to process: {total_chunks}")
                if progress_tracker:
                    progress_tracker.update(
                        0,
                        {
                            "step": "Processing in chunks",
                            "total_chunks": total_chunks,
                        },
                    )
                self.logger.info("Process using chunk")
                (
                    processed_df,
                    flag_processed,
                    all_category_mapping,
                    all_hierarchy_info,
                    all_hierarchy_cache,
                    all_fuzzy_matches,
                    all_unknown_values,
                ) = process_dataframe_using_chunk(
                    df=df,
                    process_function=self.process_batch,
                    progress_tracker=progress_tracker,
                    task_logger=self.logger,
                    **self.process_kwargs,
                )
                if flag_processed:
                    self.logger.info("Completed using chunk")
            except Exception as e:
                self.logger.warning(f"Error in chunk processing: {e}")

        if not flag_processed:
            self.logger.info("Fallback process as usual")
            (
                processed_df,
                all_category_mapping,
                all_hierarchy_info,
                all_hierarchy_cache,
                all_fuzzy_matches,
                all_unknown_values,
            ) = self.process_batch(df, **self.process_kwargs)
            flag_processed = True

        # Update internal state
        self._category_mapping = all_category_mapping
        self._hierarchy_info = all_hierarchy_info
        self._hierarchy_cache = all_hierarchy_cache
        self._fuzzy_matches = all_fuzzy_matches
        self._unknown_values = all_unknown_values
        if self.use_vectorization and "current" in self._hierarchy_cache:
            current = self._hierarchy_cache["current"]
            if not hasattr(current, "_lock") or current._lock is None:
                setattr(current, "_lock", threading.RLock())

        # Update process count
        self.process_count += len(df)
        return processed_df

    @classmethod
    def process_batch(
        cls, batch: pd.DataFrame, **process_kwargs
    ) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        """
        Process a batch of data using the configured strategy.

        Returns:
            Tuple containing:
            - processed batch
            - category_mapping dict
            - hierarchy_info dict
            - hierarchy_cache dict
        """
        try:
            # Prepare context for processing
            context = cls._prepare_context(batch, process_kwargs)

            # Apply generalization strategy
            result = cls._apply_generalization_strategy(batch, context)

            # Apply rare value template
            rare_value_template = context["rare_value_template"]
            if rare_value_template and "{n}" in rare_value_template:
                result = _apply_rare_value_template(
                    result, rare_value_template, None  # Use default pattern
                )

            # Update batch with result
            cls._update_batch_with_result(batch, result, context)
            use_vectorization = context.get("use_vectorization", False)
            category_mapping = context.get("category_mapping", {})
            hierarchy_info = context.get("hierarchy_info", {})
            hierarchy_cache = context.get("hierarchy_cache", {})
            fuzzy_matches = context.get("fuzzy_matches", 0)
            unknown_values = context.get("unknown_values", set())
            if use_vectorization and "current" in hierarchy_cache:
                current = hierarchy_cache["current"]
                if hasattr(current, "_lock"):
                    del current._lock

            # Return everything needed
            return (
                batch,
                category_mapping,
                hierarchy_info,
                hierarchy_cache,
                fuzzy_matches,
                unknown_values,
            )

        except Exception as e:
            raise

    @staticmethod
    def _prepare_context(
        batch: pd.DataFrame, process_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "batch_df": batch,
            "mode": process_kwargs.get("mode", OperationMode.REPLACE.value),
            "field_name": process_kwargs.get("field_name"),
            "output_field_name": process_kwargs.get("output_field_name", "generalized"),
            "strategy": process_kwargs.get(
                "strategy", GeneralizationStrategy.HIERARCHY.value
            ),
            "null_strategy": process_kwargs.get(
                "null_strategy", NullStrategy.PRESERVE.value
            ),
            "use_vectorization": process_kwargs.get("use_vectorization", False),
            "strategy_params": process_kwargs.get("strategy_params", {}),
            "unknown_value": process_kwargs.get("unknown_value", DEFAULT_UNKNOWN_VALUE),
            "rare_value_template": process_kwargs.get(
                "rare_value_template", DEFAULT_UNKNOWN_TEMPLATE
            ),
            "case_sensitive": process_kwargs.get("case_sensitive", False),
            "unknown_values": set(),
            "fuzzy_matches": 0,
            "category_mapping": process_kwargs.get("_category_mapping", {}),
            "hierarchy_info": process_kwargs.get("_hierarchy_info", {}),
            "hierarchy_cache": process_kwargs.get("_hierarchy_cache", {}),
        }

    @classmethod
    def _apply_generalization_strategy(
        cls,
        batch: pd.DataFrame,
        context: Dict[str, Any],
    ) -> pd.Series:
        strategy = context["strategy"]
        field_name = context["field_name"]
        strategy_params = context["strategy_params"]

        if strategy == GeneralizationStrategy.HIERARCHY.value:
            if "hierarchy" not in context:
                hierarchy = cls._load_hierarchy(strategy_params)
                context["hierarchy_cache"]["current"] = hierarchy
                context["hierarchy"] = hierarchy

            return apply_hierarchy(batch[field_name], strategy_params, context)

        elif strategy == GeneralizationStrategy.MERGE_LOW_FREQ.value:
            return apply_merge_low_freq(batch[field_name], strategy_params, context)

        elif strategy == GeneralizationStrategy.FREQUENCY_BASED.value:
            return apply_frequency_based(batch[field_name], strategy_params, context)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _update_batch_with_result(
        batch: pd.DataFrame, result: pd.Series, context: Dict[str, Any]
    ):
        if context["mode"] == OperationMode.REPLACE.value:
            batch[context["field_name"]] = result
        else:
            batch[context["output_field_name"]] = result

    @classmethod
    def process_batch_dask(cls, ddf: dd.DataFrame, **process_kwargs) -> dd.DataFrame:
        """
        Process Dask DataFrame. Should be overridden by subclasses for optimal performance.

        Parameters:
        -----------
        ddf : dd.DataFrame
            Dask DataFrame to process
        kwargs : Any
            Additional keyword arguments for processing

        Returns:
        --------
        dd.DataFrame
            Processed Dask DataFrame
        """

        # Default implementation: process each partition with process_batch
        def process_partition(partition):
            return cls.process_batch(partition.copy(deep=True), **process_kwargs)

        return ddf.map_partitions(process_partition)

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect operation-specific metrics.

        """
        # Prepare timing information
        timing_info = {
            "start_time": self.start_time,
            "end_time": self.end_time or time.time(),
            "batch_count": getattr(self, "process_count", 0) // self.chunk_size,
        }

        # Prepare operation parameters
        operation_params: Dict[str, Any] = {
            "strategy": self.strategy,
            **self.strategy_params,
        }

        # Add mapping and hierarchy info if available
        if self._category_mapping:
            operation_params["category_mapping"] = self._category_mapping
        if self._hierarchy_info:
            operation_params["hierarchy_info"] = self._hierarchy_info

        # Collect base metrics
        metrics = collect_operation_metrics(
            operation_type="generalization",
            original_data=original_data,
            processed_data=anonymized_data,
            operation_params=operation_params,
            timing_info=timing_info,
        )

        # Add category distribution analysis
        dist_original = analyze_category_distribution(
            original_data, top_n=DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS
        )
        dist_anonymized = analyze_category_distribution(
            anonymized_data, top_n=DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS
        )

        metrics["distribution_metrics"] = {
            "original": {
                "unique_values": dist_original["total_categories"],
                "entropy": dist_original.get("entropy", 0.0),
                "top_categories": dist_original.get("top_categories", {}),
            },
            "anonymized": {
                "unique_values": dist_anonymized["total_categories"],
                "entropy": dist_anonymized.get("entropy", 0.0),
                "top_categories": dist_anonymized.get("top_categories", {}),
            },
        }

        # Calculate information loss of data analysis type
        info_loss = calculate_categorical_information_loss(
            original_data, anonymized_data
        )
        metrics["categorical_info_loss"] = info_loss

        # Calculate average information loss
        metrics["info_loss_summary"] = {
            "avg_precision_loss": round(info_loss.get("precision_loss", 0.0), 4),
            "avg_entropy_loss": round(info_loss.get("entropy_loss", 0.0), 4),
            "avg_category_reduction": round(
                info_loss.get("category_reduction_ratio", 0.0), 4
            ),
        }

        # Add strategy-specific metrics
        if self.strategy == GeneralizationStrategy.HIERARCHY.value:
            # Calculate dictionary coverage
            coverage = self._calculate_dictionary_coverage(original_data)

            # Prepare hierarchy info for generalization height calculation
            hierarchy_info = self._hierarchy_info or {
                "hierarchy_level": self.strategy_params.get("hierarchy_level", 1),
                "levels": [],  # Can be populated with actual levels if available
            }

            # Calculate generalization height with correct signature
            gen_height = calculate_generalization_height(
                original_data, anonymized_data, hierarchy_info
            )

            metrics["hierarchy_metrics"] = {
                "hierarchy_level": self.strategy_params.get("hierarchy_level", 1),
                "dictionary_coverage": coverage,
                "fuzzy_matches": self._fuzzy_matches,
                "unknown_values": len(self._unknown_values),
                "generalization_height": gen_height,
            }

        elif self.strategy in [
            GeneralizationStrategy.MERGE_LOW_FREQ.value,
            GeneralizationStrategy.FREQUENCY_BASED.value,
        ]:
            # Metrics for frequency-based strategies
            metrics["frequency_metrics"] = {
                "min_group_size": self.strategy_params.get("min_group_size", 10),
                "freq_threshold": self.strategy_params.get("freq_threshold", 0.01),
                "groups_created": dist_anonymized["total_categories"],
                "rare_categories_before": self._count_rare_categories(dist_original),
                "rare_categories_after": self._count_rare_categories(dist_anonymized),
            }

        # Privacy metrics
        if self.quasi_identifiers:
            # Create proper DataFrames for privacy metrics
            original_batch = pd.DataFrame(
                {col: original_data for col in [self.field_name]}
            )
            anonymized_batch = pd.DataFrame(
                {col: anonymized_data for col in [self.output_field_name]}
            )

            privacy_metrics = calculate_batch_metrics(
                original_batch=original_batch,
                anonymized_batch=anonymized_batch,
                original_field_name=self.field_name,
                anonymized_field_name=self.output_field_name,
                quasi_identifiers=self.quasi_identifiers,
            )
            metrics["privacy_metrics"] = privacy_metrics

            # Calculate privacy metrics overview
            metrics["privacy_metric_overview"] = {
                "min_k_overall": privacy_metrics.get("min_k", 0),
                "avg_suppression_rate": round(
                    privacy_metrics.get("suppression_rate", 0.0), 4
                ),
                "avg_coverage": round(privacy_metrics.get("total_coverage", 0.0), 4),
                "avg_generalization_level": round(
                    privacy_metrics.get("generalization_level", 0.0), 4
                ),
            }

            # Check thresholds
            if self.privacy_check_enabled:
                privacy_thresholds = {
                    "min_k": self.min_acceptable_k,
                    "max_suppression": DEFAULT_MAX_SUPPRESSION_RATE,
                    "min_coverage": DEFAULT_MIN_COVERAGE,
                    "max_vulnerable_ratio": self.max_acceptable_disclosure_risk,
                }

                threshold_results = check_anonymization_thresholds(
                    privacy_metrics, privacy_thresholds
                )
                metrics["privacy_threshold_checks"] = threshold_results

                # Get privacy summary using correct function
                metrics["privacy_summary"] = get_process_summary_message(metrics)

            # Add generalization level for privacy
            metrics["generalization_level"] = calculate_generalization_level(
                original_data, anonymized_data
            )
        else:
            metrics["privacy_metrics"] = {
                "status": "SKIPPED",
                "reason": "no_quasi_identifiers",
            }
            metrics["privacy_summary"] = (
                "Privacy checks skipped: no quasi-identifiers specified"
            )

        # NULL strategy metrics
        if self.null_strategy == NullStrategy.ANONYMIZE.value:
            original_nulls = original_data.isna().sum()
            anonymized_nulls = anonymized_data.isna().sum()

            metrics["null_anonymization"] = {
                "original_nulls": int(original_nulls),
                "anonymized_nulls": int(anonymized_nulls),
                "null_anonymization_rate": calculate_suppression_rate(
                    anonymized_data, original_nulls
                ),
            }

        # Semantic diversity for reasonable cardinality
        if anonymized_data.nunique() <= DEFAULT_MAX_CATEGORIES_FOR_DIVERSITY:
            metrics["semantic_diversity"] = calculate_semantic_diversity_safe(
                list(anonymized_data.unique())
            )

        # Memory usage metrics
        if hasattr(self, "chunk_size") and self.chunk_size != self.original_chunk_size:
            metrics["memory_optimization"] = {
                "original_chunk_size": self.original_chunk_size,
                "adjusted_chunk_size": self.chunk_size,
                "adaptive_chunk_size": self.adaptive_chunk_size,
            }

        return metrics

    def _generate_visualizations(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate visualizations using the core visualization utilities with thread-safe context support.

        This is a base implementation that provides a basic distribution comparison.
        Subclasses should override to provide operation-specific visualizations.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing
        task_dir : Path
            Task directory for saving visualizations
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter object for tracking artifacts
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        visualization_paths = {}
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

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
            f"[VIZ] Starting visualization generation for {self.field_name} using {self.null_strategy} strategy"
        )
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            # Sample large datasets for visualization
            if len(original_data) > DEFAULT_SAMPLE_SIZE:
                self.logger.info(
                    f"[VIZ] Sampling large dataset: {len(original_data)} -> {DEFAULT_SAMPLE_SIZE} samples"
                )
                original_for_viz = sample_large_dataset(
                    original_data, max_samples=DEFAULT_SAMPLE_SIZE
                )
                anonymized_for_viz = sample_large_dataset(
                    anonymized_data, max_samples=DEFAULT_SAMPLE_SIZE
                )
            else:
                original_for_viz = original_data
                anonymized_for_viz = anonymized_data

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: {len(original_for_viz)} samples"
            )
            self.logger.debug(
                f"[VIZ] Original data type: {original_for_viz.dtype}, Anonymized data type: {anonymized_for_viz.dtype}"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # 1. Category distribution comparison
            dist_path = create_category_distribution_comparison(
                original_data=original_for_viz,
                anonymized_data=anonymized_for_viz,
                task_dir=viz_dir,
                field_name=self.field_name,
                operation_name=OPERATION_NAME,
                max_categories=DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS,
                timestamp=timestamp,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )
            if dist_path:
                visualization_paths["distribution"] = dist_path
                self.logger.info(
                    f"Created distribution visualization: {dist_path.name}"
                )

            # 2. Hierarchy sunburst if applicable
            if (
                self.strategy == GeneralizationStrategy.HIERARCHY.value
                and self._category_mapping
            ):
                hierarchy_dict = self._convert_mapping_to_hierarchy(
                    self._category_mapping
                )

                sunburst_path = create_hierarchy_sunburst(
                    hierarchy_data=hierarchy_dict,
                    task_dir=viz_dir,
                    field_name=self.field_name,
                    operation_name=OPERATION_NAME,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )
                if sunburst_path:
                    visualization_paths["hierarchy"] = sunburst_path
                    self.logger.info(
                        f"Created hierarchy visualization: {sunburst_path}"
                    )

            # 3. Metrics heatmap/visualization
            if self._metrics:
                metrics_viz_path = create_metrics_overview_visualization(
                    metrics=self._metrics,
                    task_dir=viz_dir,
                    field_name=self.field_name,
                    operation_name=OPERATION_NAME,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )
                if metrics_viz_path:
                    visualization_paths.update(metrics_viz_path)
                    for path in metrics_viz_path:
                        self.logger.info(
                            f"Created metrics visualization: {Path(path).name}"
                        )

            # 5. General comparison (order 5 as per SRS)
            comp_path = create_comparison_visualization(
                original_data=original_for_viz,
                anonymized_data=anonymized_for_viz,
                task_dir=viz_dir,
                field_name=self.field_name,
                operation_name=OPERATION_NAME,
                timestamp=timestamp,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )
            if comp_path:
                visualization_paths["comparison"] = comp_path
                self.logger.info(f"Created comparison visualization: {comp_path.name}")

            self.logger.info(
                f"Generated {len(visualization_paths)} visualizations successfully"
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to generate visualizations: {e}", exc_info=True
            )

        return visualization_paths

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for numeric generalization
        """
        params = dict(
            field_name=self.field_name,
            strategy=self.strategy,
            external_dictionary_path=self.external_dictionary_path,
            dictionary_format=self.dictionary_format,
            hierarchy_level=self.hierarchy_level,
            # Frequency-based parameters
            merge_low_freq=self.merge_low_freq,
            min_group_size=self.min_group_size,
            freq_threshold=self.freq_threshold,
            max_categories=self.max_categories,
            # Unknown handling
            allow_unknown=self.allow_unknown,
            unknown_value=self.unknown_value,
            group_rare_as=self.group_rare_as,
            rare_value_template=self.rare_value_template,
            # Text processing
            text_normalization=self.text_normalization,
            case_sensitive=self.case_sensitive,
            fuzzy_matching=self.fuzzy_matching,
            similarity_threshold=self.similarity_threshold,
            # Privacy thresholds
            privacy_check_enabled=self.privacy_check_enabled,
            min_acceptable_k=self.min_acceptable_k,
            max_acceptable_disclosure_risk=self.max_acceptable_disclosure_risk,
            mode=self.mode,
            output_field_name=self.output_field_name,
            column_prefix=self.column_prefix,
            quasi_identifiers=self.quasi_identifiers,
            condition_field=self.condition_field,
            condition_values=self.condition_values,
            condition_operator=self.condition_operator,
            ka_risk_field=self.ka_risk_field,
            risk_threshold=self.risk_threshold,
            vulnerable_record_strategy=self.vulnerable_record_strategy,
            optimize_memory=self.optimize_memory,
            adaptive_chunk_size=self.adaptive_chunk_size,
            chunk_size=self.chunk_size,
            use_dask=self.use_dask,
            npartitions=self.npartitions,
            dask_partition_size=self.dask_partition_size,
            use_vectorization=self.use_vectorization,
            parallel_processes=self.parallel_processes,
            use_cache=self.use_cache,
            use_encryption=self.use_encryption,
            encryption_mode=self.encryption_mode,
            encryption_key=self.encryption_key,
            visualization_theme=self.visualization_theme,
            visualization_backend=self.visualization_backend,
            visualization_strict=self.visualization_strict,
            visualization_timeout=self.visualization_timeout,
            output_format=self.output_format,
            force_recalculation=self.force_recalculation,
            generate_visualization=self.generate_visualization,
            save_output=self.save_output,
        )

        # Add strategy-specific parameters
        if self.strategy == GeneralizationStrategy.HIERARCHY.value:
            # Include dictionary hash if configured
            if self.use_cache and self._hierarchy_cache.get("current"):
                try:
                    file_hash = self._calculate_dictionary_hash()
                    if file_hash:
                        params["dictionary_hash"] = file_hash
                    else:
                        self.logger.warning(
                            "Dictionary file hash not available for cache key"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to calculate dictionary hash: {e}")

        return params

    def reset_state(self):
        """
        Reset operation state after execution.
        """
        # Clear cached data
        self._hierarchy_cache.clear()
        self._category_mapping = None
        self._hierarchy_info = None
        self._fuzzy_matches = 0
        self._unknown_values = set()
        self._metrics = {}

        # Reset chunk size if it was adjusted
        if hasattr(self, "original_chunk_size"):
            self.chunk_size = self.original_chunk_size

        self.logger.debug(f"State reset completed for operation {self.operation_id}")

    # Private helper methods
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID for operation."""
        return f"{OPERATION_NAME}_{uuid.uuid4().hex[:8]}"

    def _prepare_process_kwargs(self) -> None:
        """
        Prepare or update keyword arguments for processing methods.

        Consolidates configuration, parameters, and state into
        process_kwargs for easy passing to processing functions.
        """
        if not hasattr(self, "process_kwargs"):
            self.process_kwargs = {}

        self.process_kwargs.update(
            {
                # Core identification
                "operation_id": self.operation_id,
                # Strategy parameters (pre-extracted)
                "strategy_params": self.strategy_params,
                # State references (mutable, allows tracking)
                "_hierarchy_cache": self._hierarchy_cache,
                "_category_mapping": self._category_mapping,
                "_hierarchy_info": self._hierarchy_info,
            }
        )

    @staticmethod
    def _load_hierarchy(strategy_params: Dict[str, Any]) -> HierarchyDictionary:
        """
        Load hierarchy dictionary for hierarchy strategy.

        """
        if not strategy_params.get("external_dictionary_path"):
            raise ValueError("Hierarchy strategy requires external_dictionary_path")

        hierarchy = HierarchyDictionary()
        hierarchy.load_from_file(
            strategy_params["external_dictionary_path"],
            strategy_params.get("dictionary_format", "auto"),
            strategy_params["case_sensitive"],
        )

        # Validate structure
        is_valid, errors = hierarchy.validate_structure()
        if not is_valid:
            raise ValueError(f"Invalid hierarchy dictionary: {', '.join(errors)}")

        return hierarchy

    def _save_mapping_artifact(self, writer: DataWriter) -> Optional[Path]:
        """
        Save category mapping as artifact.

        """
        if not self._category_mapping:
            return None

        try:
            mapping_data = {
                "operation_id": self.operation_id,
                "field_name": self.field_name,
                "strategy": self.strategy,
                "timestamp": datetime.now().isoformat(),
                "total_mappings": len(self._category_mapping),
                "mappings": self._category_mapping,
            }

            result = writer.write_json(
                data=mapping_data,
                name=f"{self.field_name}_category_mapping",
                subdir="mappings",
                timestamp_in_name=True,
            )

            # If we reach here, the write was successful
            self.logger.info(f"Saved category mapping to {result.path}")
            return result.path

        except Exception as e:
            self.logger.warning(f"Error saving category mapping: {e}")
            return None

    def _calculate_dictionary_coverage(self, data: pd.Series) -> float:
        """
        Calculate dictionary coverage for hierarchy strategy.

        """
        if self.strategy != GeneralizationStrategy.HIERARCHY.value:
            return 0.0

        hierarchy = self._hierarchy_cache.get("current")
        if not hierarchy:
            return 0.0

        coverage_info = hierarchy.get_coverage(
            data.dropna().unique().tolist(),
            normalize=not self.strategy_params.get("case_sensitive", False),
        )

        return coverage_info.get("coverage_percent", 0.0)

    def _calculate_dictionary_hash(self) -> Optional[str]:
        """Calculate hash of dictionary file for cache key."""
        dict_path = self.strategy_params.get("external_dictionary_path")
        if not dict_path:
            return None

        try:
            path = Path(dict_path)
            if not path.exists():
                return None

            # Calculate file hash
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]  # Use first 16 chars

        except Exception as e:
            self.logger.debug(f"Failed to calculate dictionary hash: {e}")
            return None

    def _count_rare_categories(self, distribution: Dict[str, Any]) -> int:
        """Count rare categories based on threshold."""
        threshold = self.strategy_params.get("freq_threshold", 0.01)
        total_count = distribution.get("total_count", 0)

        if total_count == 0:
            return 0

        rare_count = 0
        for category, count in distribution.get("frequency_counts", {}).items():
            if count / total_count < threshold:
                rare_count += 1

        return rare_count

    def _convert_mapping_to_hierarchy(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Convert flat mapping to hierarchical structure for visualization."""
        hierarchy = {"name": "root", "children": []}

        # Group by generalized values
        groups = defaultdict(list)
        for original, generalized in mapping.items():
            groups[generalized].append(original)

        # Create hierarchy structure
        for generalized, originals in groups.items():
            group_node = {
                "name": generalized,
                "children": [
                    {"name": orig, "value": 1}
                    for orig in originals[:DEFAULT_MAX_HIERARCHY_CHILDREN_DISPLAY]
                ],
            }
            if len(originals) > DEFAULT_MAX_HIERARCHY_CHILDREN_DISPLAY:
                group_node["children"].append(
                    {
                        "name": f"... and {len(originals) - DEFAULT_MAX_HIERARCHY_CHILDREN_DISPLAY} more",
                        "value": len(originals)
                        - DEFAULT_MAX_HIERARCHY_CHILDREN_DISPLAY,
                    }
                )
            hierarchy["children"].append(group_node)

        return hierarchy


# Factory function for backward compatibility
def create_categorical_generalization_operation(
    field_name: str, **kwargs
) -> CategoricalGeneralizationOperation:
    """
    Create a categorical generalization operation.

    Parameters:
    -----------
    field_name : str
        Field to generalize
    **kwargs : dict
        Additional parameters for configuration

    Returns:
    --------
    CategoricalGeneralizationOperation
        Configured operation instance
    """
    return CategoricalGeneralizationOperation(field_name=field_name, **kwargs)
