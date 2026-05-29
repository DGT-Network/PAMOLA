"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Hash-based Pseudonymization Operation
Package:       pamola_core.anonymization.pseudonymization
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-10
Updated:       2026-04-15
License:       BSD 3-Clause

Description:
    This module implements hash-based pseudonymization for sensitive data fields.
    It provides irreversible transformation of identifiers using cryptographic
    hash functions (SHA3-256/512) with configurable salt and pepper values.

Key Features:
    - SHA3-256/512 hash algorithms for strong cryptographic security
    - Configurable salt (per-field or global) and pepper (per-session)
    - Support for REPLACE and ENRICH modes
    - Batch processing with caching for performance
    - Risk-based processing for vulnerable records
    - Multiple output formats (hex, base64, uuid-style)
    - Compound identifier support for multi-field pseudonymization
    - Comprehensive metrics collection and visualization
    - Full integration with PAMOLA framework lifecycle (7-step pattern)

Security Considerations:
    - Uses SHA3 family for quantum resistance
    - Salt prevents rainbow table attacks
    - Pepper provides additional session-specific security
    - Secure memory handling for sensitive data
    - No storage of original-to-pseudonym mappings (irreversible)

Changelog:
    2.0.0 (2026-04-15):
        - Refactored to match PAMOLA operation lifecycle (7-step pattern)
        - Removed inline HashBasedPseudonymizationConfig (moved to schemas/)
        - Constructor follows config→super→setattr pattern
        - execute() follows 7-step lifecycle matching numeric_op.py
        - _validate_configuration() extracted as standalone method
    1.0.1 (2025-06-15):
        - Fixed import issues with validation framework
        - Updated to use validation_utils facade
        - Improved error handling and logging
    1.0.0 (2025-01-20):
        - Initial implementation with full framework integration
"""

import base64
import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Base anonymization operation import
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Schema import — config class lives in schemas/, not inline
from pamola_core.anonymization.schemas.hash_based_op_core_schema import (
    HashBasedPseudonymizationConfig,
)

# Metric utilities imports
from pamola_core.anonymization.commons.metric_utils import (
    calculate_anonymization_effectiveness,
    calculate_process_performance,
    collect_operation_metrics,
)

# Privacy metric utilities imports
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_batch_metrics,
    calculate_simple_disclosure_risk,
    calculate_suppression_rate,
    get_process_summary,
)

# Pseudonymization utilities imports
from pamola_core.anonymization.commons.pseudonymization_utils import (
    PseudonymizationCache,
    estimate_collision_probability,
    format_pseudonym_output,
    generate_session_pepper,
    load_salt_configuration,
)

# Validation imports
from pamola_core.anonymization.commons.validation_utils import (
    check_field_exists,
    create_field_validator,
)

# Crypto helpers imports
from pamola_core.common.helpers.data_helper import DataHelper
from pamola_core.utils.crypto_helpers.pseudonymization import (
    HashGenerator,
    SecureBytes,
)

# Operation framework imports
from pamola_core.utils.io import load_settings_operation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_field_utils import (
    create_composite_key,
)
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.helpers import filter_used_kwargs
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.errors.codes import ErrorCode
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.exceptions import (
    FieldNotFoundError,
    InvalidParameterError,
)


@register(version="1.0.0")
class HashBasedPseudonymizationOperation(AnonymizationOperation):
    """
    Hash-based pseudonymization operation for irreversible data transformation.

    Applies cryptographic hash functions (SHA3-256/512) to transform sensitive
    identifiers into pseudonyms that cannot be reversed without the original
    salt and pepper values. Follows PAMOLA 7-step operation lifecycle.
    """

    # Tells AnonymizationOperation.process_data() to use the pseudonymization
    # null-anonymize placeholder ("*REDACTED*") instead of "SUPPRESSED".
    _is_pseudonymization: bool = True

    def __init__(
        self,
        field_name: str,
        additional_fields: Optional[List[str]] = None,
        algorithm: str = "sha3_256",
        salt_config: Optional[Dict[str, Any]] = None,
        salt_file: Optional[Path] = None,
        use_pepper: bool = True,
        pepper_length: int = 32,
        hash_output_format: str = "hex",
        output_length: Optional[int] = None,
        output_prefix: Optional[str] = None,
        output_suffix: Optional[str] = None,
        quasi_identifiers: Optional[List[str]] = None,
        compound_mode: bool = False,
        compound_separator: str = "|",
        compound_null_handling: str = "skip",
        **kwargs,
    ):
        """
        Initialize hash-based pseudonymization operation.

        Parameters
        -----------
        field_name : str
            Primary field to pseudonymize
        additional_fields : Optional[List[str]]
            Additional fields for compound pseudonymization
        algorithm : str
            Hash algorithm: "sha3_256" or "sha3_512" (default: "sha3_256")
        salt_config : Optional[Dict[str, Any]]
            Salt configuration with 'source' and 'value' or 'field_name'
        salt_file : Optional[Path]
            Path to salt file (for file-based salt)
        use_pepper : bool
            Whether to use session pepper (default: True)
        pepper_length : int
            Length of pepper in bytes (default: 32)
        hash_output_format : str
            Hash output format: "hex", "base64", "base32", "base58", or "uuid" (default: "hex")
        output_length : Optional[int]
            Truncate output to specified length (default: None)
        output_prefix : Optional[str]
            Prefix for pseudonyms (default: None)
        output_suffix : Optional[str]
            Suffix for pseudonyms (default: None)
        compound_mode : bool
            Whether to create compound identifiers (default: False)
        compound_separator : str
            Separator for compound identifiers (default: "|")
        compound_null_handling : str
            How to handle nulls in compounds (default: "skip")
        **kwargs : dict
            Additional parameters passed to AnonymizationOperation
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Hash-based pseudonymization for '{field_name}' using {algorithm}",
        )

        # Normalize defaults
        if salt_config is None:
            salt_config = {"source": "parameter", "value": "0" * 64}
        if additional_fields is None:
            additional_fields = []

        # Build config object (if used for schema/validation)
        config = HashBasedPseudonymizationConfig(
            field_name=field_name,
            additional_fields=additional_fields,
            algorithm=algorithm,
            salt_config=salt_config,
            salt_file=str(salt_file) if salt_file else None,
            use_pepper=use_pepper,
            pepper_length=pepper_length,
            hash_output_format=hash_output_format,
            output_length=output_length,
            output_prefix=output_prefix,
            output_suffix=output_suffix,
            quasi_identifiers=quasi_identifiers,
            compound_mode=compound_mode,
            compound_separator=compound_separator,
            compound_null_handling=compound_null_handling,
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize base AnonymizationOperation
        super().__init__(field_name=field_name, **kwargs)

        # Copy config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # Initialize internal state for crypto components, caching, and metrics
        self._salt = None
        self._pepper = None
        self._session_id: Optional[str] = None
        self._hash_generator = None
        self._pseudonym_cache = PseudonymizationCache() if self.use_cache else None
        self._collision_tracker: set = set()
        self._collision_probability = 0.0
        self._collision_count = 0
        self._cache_hits = 0
        self._hash_computation_time = 0.0

        # Operation metadata
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
        Execute the hash-based pseudonymization operation using 7-step lifecycle.

        Parameters
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

        Returns
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Start timing
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(
                f"Starting: {self.operation_name} operation at {self.start_time}"
            )

            # Initialize result object
            result = OperationResult(status=OperationStatus.PENDING)

            # Initialize dataframe
            df = None

            # Extract dataset name from kwargs
            dataset_name = kwargs.get("dataset_name", "main")

            # Generate timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare directories for artifacts
            dirs = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(cache_dir=dirs["cache"])

            # Initialize error handler early so outer except can use it
            self.error_handler = ErrorHandler(
                logger=self.logger,
                operation_name=self.operation_name,
            )

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir,
                logger=self.logger,
                progress_tracker=progress_tracker,
            )

            # Save configuration to task directory
            self.save_config(task_dir)

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, "
                f"backend={self.visualization_backend}, "
                f"strict={self.visualization_strict}, "
                f"timeout={self.visualization_timeout}s"
            )

            # Load settings operation
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )

            # Set up progress tracking
            TOTAL_MAIN_STEPS = 6 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            main_progress = progress_tracker
            current_steps = 0
            if main_progress:
                try:
                    main_progress.total = TOTAL_MAIN_STEPS
                    main_progress.update(
                        current_steps,
                        {
                            "step": "Starting hash-based pseudonymization",
                            "field": self.field_name,
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # -------------------------------------------------------------------
            # Step 1: Data Loading & Validation
            # -------------------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Data Loading", "field": self.field_name},
                )

            # Initialize crypto (salt, pepper, hash generator)
            try:
                self._initialize_crypto_components()
            except Exception as e:
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.CRYPTO_ERROR,
                    context={"step": "crypto_init", "operation": self.operation_name},
                    message_kwargs={
                        "reason": str(e),
                    },
                )

            try:
                # Validate configuration before loading data to fail fast on misconfiguration
                self._validate_configuration()
                self.logger.info(
                    f"Operation: {self.operation_name}, Load data and validate input parameters"
                )
                df = self._validate_and_get_dataframe(
                    data_source, dataset_name, **settings_operation
                )

                # Validate all required fields exist
                all_fields = [self.field_name] + list(self.additional_fields or [])
                for field in all_fields:
                    if not check_field_exists(df, field):
                        raise FieldNotFoundError(
                            field_name=field,
                            available_fields=list(df.columns),
                        )
                    # Warn on validation issues but do not fail
                    try:
                        validator = create_field_validator("text")
                        val_result = validator.validate(df[field], field_name=field)
                        if not val_result.is_valid:
                            self.logger.warning(
                                f"Field '{field}' validation warnings: {val_result.warnings}"
                            )
                    except Exception as ve:
                        self.logger.warning(f"Could not validate field '{field}': {ve}")

            except Exception as e:
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.DATA_LOAD_FAILED,
                    context={"dataset": dataset_name, "operation": self.operation_name},
                    message_kwargs={"source": dataset_name, "reason": str(e)},
                )

            # -------------------------------------------------------------------
            # Step 2: Cache check
            # -------------------------------------------------------------------
            if self.use_cache and not self.force_recalculation:
                if main_progress:
                    current_steps += 1
                    main_progress.update(
                        current_steps,
                        {"step": "Checking cache", "field": self.field_name},
                    )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df, reporter)
                if cache_result:
                    self.logger.info(
                        f"Using cached result for {self.field_name} pseudonymization"
                    )
                    if main_progress:
                        main_progress.update(
                            current_steps,
                            {"step": "Complete (cached)", "field": self.field_name},
                        )
                    if reporter:
                        reporter.add_operation(
                            f"Hash pseudonymization of {self.field_name} (cached)",
                            details={"cached": True},
                        )
                    return cache_result

            # -------------------------------------------------------------------
            # Step 3: Prepare output field
            # -------------------------------------------------------------------
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
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.PROCESSING_FAILED,
                    context={"step": "prepare_output_field", "field": self.field_name},
                    message_kwargs={
                        "field_name": self.field_name,
                        "operation": self.operation_name,
                        "reason": str(e),
                    },
                )

            # -------------------------------------------------------------------
            # Step 4: Processing
            # -------------------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Processing", "field": self.field_name},
                )

            try:
                # Normalize integer dtype if required
                df[self.field_name] = DataHelper.normalize_int_dtype_vectorized(
                    df[self.field_name], safe_mode=False
                )

                # Capture original data for metrics (before transformation)
                all_fields = [self.field_name] + list(self.additional_fields or [])
                if self.compound_mode:
                    original_data = create_composite_key(
                        df,
                        all_fields,
                        self.compound_separator,
                        self.compound_null_handling,
                    )
                else:
                    original_data = df[self.field_name].copy(deep=True)

                # Create child progress tracker for batch processing
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

                # Process filtered data
                if not filtered_df.empty:
                    processed_df = self._process_data_with_config(
                        df=filtered_df,
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

                # Calculate collision probability after processing
                # Use actual unique pseudonyms count (not total rows) for accurate estimate
                self._collision_probability = estimate_collision_probability(
                    len(self._collision_tracker),
                    256 if self.algorithm == "sha3_256" else 512,
                )

                # Get the anonymized data for metrics
                # In REPLACE compound mode, additional_fields are nulled out by process_batch,
                # so use primary field directly (it contains the pseudonymized compound value)
                if self.mode == "REPLACE":
                    anonymized_data = processed_df[self.field_name].copy(deep=True)
                else:
                    anonymized_data = processed_df[self.output_field_name].copy(
                        deep=True
                    )

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except Exception:
                        pass

            except Exception as e:
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.PROCESSING_FAILED,
                    context={"step": "processing", "field": self.field_name},
                    message_kwargs={
                        "field_name": self.field_name,
                        "operation": self.operation_name,
                        "reason": str(e),
                    },
                )

            # Record end time after processing
            self.end_time = time.time()
            if self.end_time and self.start_time:
                self.execution_time = self.end_time - self.start_time

            # -------------------------------------------------------------------
            # Step 5: Metrics Calculation
            # -------------------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Metrics Calculation", "field": self.field_name},
                )

            metrics = {}
            try:
                metrics = self._collect_comprehensive_metrics(
                    original_data, anonymized_data, processed_df
                )

                metrics_file_name = (
                    f"{self.field_name}_anonymization_hash_pseudonymization_metrics_"
                    f"{operation_timestamp}"
                )
                self._save_metrics(
                    metrics=metrics,
                    writer=writer,
                    result=result,
                    reporter=reporter,
                    progress_tracker=progress_tracker,
                    operation_timestamp=operation_timestamp,
                    file_name=metrics_file_name,
                )

                summary = get_process_summary(metrics.get("privacy_metrics", {}))
                for key, message in summary.items():
                    self.logger.info(f"{key}: {message}")

            except Exception as e:
                self.logger.warning(f"Error calculating metrics: {str(e)}")
                # Metrics failure is non-critical — continue

            # -------------------------------------------------------------------
            # Step 6: Visualizations
            # -------------------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Generating Visualizations", "field": self.field_name},
                )

            if self.generate_visualization and self.visualization_backend is not None:
                try:
                    kwargs_encryption = {
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    }
                    self._handle_visualizations(
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
                    self.logger.warning(f"Error generating visualizations: {str(e)}")
                    # Visualization failure is non-critical — continue
            else:
                self.logger.info(
                    "Skipping visualizations as generate_visualization is False or backend is not set"
                )

            # -------------------------------------------------------------------
            # Step 7: Save Output Data
            # -------------------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Save Output Data", "field": self.field_name},
                )

            if self.save_output:
                try:
                    safe_kwargs = filter_used_kwargs(
                        kwargs, HashBasedPseudonymizationOperation._save_output_data
                    )
                    self._save_output_data(
                        result_df=processed_df,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        timestamp=operation_timestamp,
                        **safe_kwargs,
                    )
                except Exception as e:
                    return self.error_handler.handle_error(
                        error=e,
                        error_code=ErrorCode.ARTIFACT_WRITE_FAILED,
                        context={"step": "save_output", "field": self.field_name},
                        message_kwargs={
                            "path": str(task_dir / "output"),
                            "reason": str(e),
                        },
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
                        result=result,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                processed_df=processed_df,
                original_data=original_data,
                anonymized_data=anonymized_data,
            )

            # Report completion
            if reporter:
                reporter.add_operation(
                    f"Hash pseudonymization of {self.field_name} completed",
                    details={
                        "records_processed": self.process_count,
                        "execution_time": self.execution_time,
                        "algorithm": self.algorithm,
                    },
                )

            # Set success status
            result.status = OperationStatus.SUCCESS
            result.execution_time = self.execution_time
            self.logger.info(
                f"Processing completed {self.operation_name} operation in "
                f"{self.execution_time:.2f} seconds"
            )
            return result

        except Exception as e:
            self.logger.exception(f"Error in {self.operation_name}: {str(e)}")
            return self.error_handler.handle_error(
                error=e,
                error_code=ErrorCode.PROCESSING_FAILED,
                context={"operation": self.operation_name, "field": self.field_name},
                message_kwargs={
                    "field_name": self.field_name,
                    "operation": self.operation_name,
                    "reason": str(e),
                },
            )
        finally:
            # Always clean up secure crypto memory
            if self._pepper:
                self._pepper.clear()
                self._pepper = None
            self._salt = None
            self._hash_generator = None
            self._session_id = None
            self.logger.debug("Cleared cryptographic components from memory")

            # Clear pseudonym cache only when pepper is enabled: pepper changes every
            # run, so cached hashes from the previous run would be wrong next run.
            # When pepper is disabled, hashes are deterministic — cache can be reused.
            if self.use_pepper and self._pseudonym_cache:
                self._pseudonym_cache.clear()

            # Reset per-execution counters after every run (success or failure)
            # so the next execute() call starts from a clean state.
            self._collision_tracker = set()
            self._collision_probability = 0.0
            self._collision_count = 0
            self._cache_hits = 0
            self._hash_computation_time = 0.0

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data to pseudonymize values.

        Parameters
        -----------
        batch : pd.DataFrame
            DataFrame batch to process

        Returns
        --------
        pd.DataFrame
            Processed DataFrame with pseudonymized values
        """
        # Validate all fields exist in batch (should already be validated in execute, but double-check for safety).
        # Use defensive `list(... or [])` guard so this method is safe to call
        # independently (additional_fields may be None when config is reloaded).
        all_fields = [self.field_name] + list(self.additional_fields or [])
        for field in all_fields:
            if field not in batch.columns:
                raise FieldNotFoundError(
                    field_name=field,
                    available_fields=list(batch.columns),
                )

        # Build working Series for pseudonymization: compound mode merges fields, single mode uses field directly
        if self.compound_mode:
            working_series = create_composite_key(
                batch, all_fields, self.compound_separator, self.compound_null_handling
            )
        else:
            working_series = batch[self.field_name].copy(deep=True)

        # Pseudonymize non-null values
        non_null_mask = working_series.notna()
        non_null_values = working_series[non_null_mask]

        if len(non_null_values) > 0:
            unique_values = non_null_values.unique()
            pseudonym_map = {}

            for value in unique_values:
                # Check cache first
                if self.use_cache and self._pseudonym_cache:
                    cached = self._pseudonym_cache.get(str(value))
                    if cached:
                        pseudonym_map[value] = cached
                        self._cache_hits += 1
                        continue

                # Generate new pseudonym
                start_hash_time = time.perf_counter()
                pseudonym = self._generate_pseudonym(str(value))
                self._hash_computation_time += time.perf_counter() - start_hash_time

                # Check for collisions (set lookup O(1), half the memory of dict)
                if pseudonym in self._collision_tracker:
                    self._collision_count += 1
                    self.logger.warning(
                        f"Hash collision detected for '{value}' → '{pseudonym}'"
                    )
                else:
                    self._collision_tracker.add(pseudonym)

                pseudonym_map[value] = pseudonym

                # Cache the result
                if self.use_cache and self._pseudonym_cache:
                    self._pseudonym_cache.put(str(value), pseudonym)

            # Apply pseudonyms to working series
            working_series[non_null_mask] = non_null_values.map(pseudonym_map)

        # Write result to DataFrame based on mode
        if self.mode == "REPLACE":
            if self.compound_mode:
                # Replace primary field with pseudonymized compound; null out additional fields
                batch[self.field_name] = working_series
                for field in self.additional_fields or []:
                    batch[field] = None
            else:
                batch[self.field_name] = working_series
        else:  # ENRICH
            batch[self.output_field_name] = working_series

        return batch

    def _validate_configuration(self):
        """Validate operation configuration before execution."""
        if self.algorithm not in ["sha3_256", "sha3_512"]:
            raise InvalidParameterError(
                param_name="algorithm",
                param_value=self.algorithm,
                reason=f"Unsupported hash algorithm: {self.algorithm}. Must be 'sha3_256' or 'sha3_512'",
            )

        if self.hash_output_format not in ["hex", "base64", "base32", "base58", "uuid"]:
            raise InvalidParameterError(
                param_name="hash_output_format",
                param_value=self.hash_output_format,
                reason=f"Unsupported hash output format: {self.hash_output_format}",
            )

        if self.output_length is not None and self.output_length < 8:
            raise InvalidParameterError(
                param_name="output_length",
                param_value=self.output_length,
                reason="output_length must be at least 8 characters",
            )

        if self.compound_mode and not self.additional_fields:
            raise InvalidParameterError(
                param_name="additional_fields",
                param_value=self.additional_fields,
                reason="compound_mode requires additional_fields to be specified",
            )

        # Reject the all-zero default salt when pepper is disabled — with no
        # per-session entropy, identical inputs produce identical pseudonyms
        # across every PAMOLA install, defeating rainbow-table resistance.
        if (
            not self.use_pepper
            and self.salt_config.get("source") == "parameter"
            and self._is_weak_salt_value(self.salt_config.get("value"))
        ):
            raise InvalidParameterError(
                param_name="salt_config",
                param_value=self.salt_config,
                reason=(
                    "Default/weak salt (all-zero or empty) detected with "
                    "use_pepper=False. Provide a non-trivial salt value, "
                    "use salt_config['source']='file', or enable use_pepper=True."
                ),
            )

    @staticmethod
    def _is_weak_salt_value(value: Any) -> bool:
        """Return True for salt values that provide no entropy (empty or all-zero)."""
        if value is None or value == "":
            return True
        if isinstance(value, str):
            stripped = value.strip()
            return stripped == "" or set(stripped) <= {"0"}
        if isinstance(value, (bytes, bytearray)):
            return len(value) == 0 or all(b == 0 for b in value)
        return False

    def _collect_comprehensive_metrics(
        self,
        original_series: pd.Series,
        processed_series: pd.Series,
        full_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Collect all metrics using commons utilities.

        Parameters
        -----------
        original_series : pd.Series
            Original data before pseudonymization
        processed_series : pd.Series
            Data after pseudonymization
        full_df : pd.DataFrame
            Full processed DataFrame

        Returns
        --------
        Dict[str, Any]
            Comprehensive metrics dictionary
        """
        # Timing info derived from operation state
        timing_info = {
            "start_time": self.start_time,
            "end_time": (
                self.end_time
                if hasattr(self, "end_time") and self.end_time
                else time.time()
            ),
            "batch_count": max(
                1, (len(full_df) + self.chunk_size - 1) // self.chunk_size
            ),
        }

        # Use metric_utils to collect base operation metrics
        operation_metrics = collect_operation_metrics(
            operation_type="pseudonymization",
            original_data=original_series,
            processed_data=processed_series,
            operation_params={
                "algorithm": self.algorithm,
                "hash_output_format": self.hash_output_format,
                "compound_mode": self.compound_mode,
                "use_pepper": self.use_pepper,
            },
            timing_info=timing_info,
        )

        # Anonymization effectiveness
        effectiveness = calculate_anonymization_effectiveness(
            original_series, processed_series
        )
        operation_metrics["effectiveness"] = effectiveness

        # Pseudonymization-specific metrics.
        # NOTE: `rows_processed` counts every row passed through the op (matches
        # base class process_count). `unique_values_hashed` is the count of
        # distinct inputs that produced a hash (collision_tracker holds unique
        # output pseudonyms; add collision_count for the inputs that mapped to
        # an already-seen pseudonym).
        # `pseudonymization_rate` is the fraction of non-null originals that
        # actually changed value (i.e., received a pseudonym). This drops below
        # 1.0 when conditional filtering, k-anonymity suppression, or other
        # row-level skips leave some non-null inputs untouched. 0.0 when there
        # are no non-null inputs to pseudonymize.
        non_null_original = int(original_series.notna().sum())
        if non_null_original > 0:
            changed_mask = original_series.notna() & (
                original_series.astype(str) != processed_series.astype(str)
            )
            pseudonymization_rate = round(int(changed_mask.sum()) / non_null_original, 4)
        else:
            pseudonymization_rate = 0.0

        operation_metrics["pseudonymization"] = {
            "algorithm": self.algorithm,
            "hash_output_format": self.hash_output_format,
            "collision_probability": self._collision_probability,
            "collision_count": self._collision_count,
            "estimated_collision_count": int(
                self._collision_probability * self.process_count
            ),
            "unique_pseudonyms": processed_series.nunique(),
            "pseudonymization_rate": pseudonymization_rate,
            "rows_processed": self.process_count,
            "unique_values_hashed": len(self._collision_tracker) + self._collision_count,
            "salt_source": self.salt_config.get("source", "unknown"),
            "hash_computation_time": round(self._hash_computation_time, 4),
            "cache_hits": self._cache_hits,
        }

        # Cache metrics if caching is enabled
        if self.use_cache and self._pseudonym_cache:
            operation_metrics["cache"] = self._pseudonym_cache.get_statistics()

        # Privacy metrics if quasi-identifiers are available
        output_col = (
            self.output_field_name
            if self.mode == "ENRICH" and hasattr(self, "output_field_name")
            else self.field_name
        )
        if self.quasi_identifiers and all(
            qi in full_df.columns for qi in self.quasi_identifiers
        ):
            privacy_metrics = calculate_batch_metrics(
                original_batch=full_df[
                    [self.field_name] + list(self.quasi_identifiers)
                ],
                anonymized_batch=full_df[[output_col] + list(self.quasi_identifiers)],
                original_field_name=self.field_name,
                anonymized_field_name=output_col,
                quasi_identifiers=list(self.quasi_identifiers),
            )
            privacy_metrics["disclosure_risk"] = calculate_simple_disclosure_risk(
                full_df, list(self.quasi_identifiers)
            )
            privacy_metrics["suppression_rate"] = calculate_suppression_rate(
                processed_series, original_series.isna().sum()
            )
            operation_metrics["privacy_metrics"] = privacy_metrics

        # Performance metrics
        operation_metrics["performance"] = calculate_process_performance(
            timing_info["start_time"],
            timing_info["end_time"],
            len(full_df),
            timing_info["batch_count"],
        )

        return operation_metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """Get operation-specific parameters for cache key generation."""
        params = {
            "algorithm": self.algorithm,
            "hash_output_format": self.hash_output_format,
            "output_length": self.output_length or 0,
            "output_prefix": self.output_prefix or "",
            "output_suffix": self.output_suffix or "",
            "compound_mode": self.compound_mode,
            "compound_fields": (
                tuple(sorted(self.additional_fields)) if self.compound_mode else ()
            ),
            "salt_source": self.salt_config.get("source", ""),
        }

        # Add salt file field if file-based salt
        if self.salt_config.get("source") == "file":
            params["salt_field"] = self.salt_config.get("field_name", "")

        # Add salt digest for cache consistency (not the full salt for security)
        if self._salt:
            params["salt_digest"] = hashlib.sha256(self._salt).hexdigest()[:16]

        params["use_pepper"] = bool(self.use_pepper)

        # When pepper is enabled, pepper bytes are generated per-run and CANNOT
        # be embedded in the cache key (security) — so we must force a unique
        # cache key per process invocation. _session_id is set in
        # _initialize_crypto_components() when use_pepper=True.
        if self.use_pepper and getattr(self, "_session_id", None):
            params["session_id"] = self._session_id
        return params

    def _initialize_crypto_components(self):
        """Initialize salt, pepper, and hash generator for cryptographic operations."""
        salt_file_path = Path(self.salt_file) if self.salt_file else None
        self._salt = load_salt_configuration(self.salt_config, salt_file_path)
        self.logger.info(f"Loaded {len(self._salt)}-byte salt")

        self._hash_generator = HashGenerator(algorithm=self.algorithm)

        if self.use_pepper:
            self._pepper = generate_session_pepper(self.pepper_length)
            # Per-run session id makes the disk cache key unique per execution
            # so the previous run's pseudonyms (computed with a different
            # pepper) cannot be served back. See _get_cache_parameters().
            self._session_id = uuid.uuid4().hex
            self.logger.info(f"Generated {self.pepper_length}-byte session pepper")

    def _generate_pseudonym(self, value: str) -> str:
        """
        Generate a pseudonym for a single value.

        Parameters
        -----------
        value : str
            Value to pseudonymize

        Returns
        --------
        str
            Generated pseudonym
        """
        # Apply hash function with salt and optional pepper
        if self.use_pepper and self._pepper:
            hash_bytes = self._hash_generator.hash_with_salt_and_pepper(
                value.encode("utf-8"), self._salt, self._pepper.get()
            )
        else:
            hash_bytes = self._hash_generator.hash_with_salt(
                value.encode("utf-8"), self._salt
            )

        # Encode output in requested format
        if self.hash_output_format == "hex":
            pseudonym = hash_bytes.hex()
        elif self.hash_output_format == "base64":
            pseudonym = base64.urlsafe_b64encode(hash_bytes).decode("utf-8").rstrip("=")
        elif self.hash_output_format == "base32":
            pseudonym = base64.b32encode(hash_bytes).decode("utf-8").rstrip("=")
        elif self.hash_output_format == "base58":
            try:
                import base58

                pseudonym = base58.b58encode(hash_bytes).decode("utf-8")
            except ImportError:
                self.logger.warning("base58 not available, falling back to hex")
                pseudonym = hash_bytes.hex()
        elif self.hash_output_format == "uuid":
            uuid_bytes = hash_bytes[:16]
            pseudonym = str(uuid.UUID(bytes=uuid_bytes))
        else:
            raise InvalidParameterError(
                param_name="hash_output_format",
                param_value=self.hash_output_format,
                reason=f"Unknown hash output format: {self.hash_output_format}",
            )

        # Apply length limit if specified
        if self.output_length and len(pseudonym) > self.output_length:
            pseudonym = pseudonym[: self.output_length]

        # Apply prefix/suffix decorators
        pseudonym = format_pseudonym_output(
            pseudonym, self.output_prefix, self.output_suffix
        )
        return pseudonym

    def __getstate__(self):
        # Threading locks inside SecureBytes / HashGenerator / PseudonymizationCache
        # can't be pickled. Strip the wrappers but preserve raw pepper BYTES so
        # every worker process produces identical hashes (determinism requirement).
        state = self.__dict__.copy()
        # Preserve pepper bytes (not the SecureBytes wrapper that contains the lock)
        pepper = state.pop("_pepper", None)
        if pepper is not None:
            try:
                state["_pepper_bytes"] = pepper.get()
            except Exception:
                state["_pepper_bytes"] = None
        # HashGenerator is trivially reconstructible from self.algorithm
        state.pop("_hash_generator", None)
        # Cache is worker-local; workers get a fresh one
        state.pop("_pseudonym_cache", None)
        return state

    def __setstate__(self, state):
        pepper_bytes = state.pop("_pepper_bytes", None)
        self.__dict__.update(state)
        # Reconstruct hash generator from already-restored self.algorithm
        self._hash_generator = HashGenerator(algorithm=self.algorithm)
        # Reconstruct pepper wrapper from raw bytes (same bytes => same hashes)
        self._pepper = SecureBytes(pepper_bytes) if pepper_bytes else None
        # Fresh per-worker cache (caches are local; no cross-worker sharing needed)
        self._pseudonym_cache = (
            PseudonymizationCache() if getattr(self, "use_cache", False) else None
        )


# Factory function
def create_hash_pseudonymization_operation(
    field_name: str, **kwargs
) -> HashBasedPseudonymizationOperation:
    """
    Create a hash-based pseudonymization operation with default settings.

    Parameters
    -----------
    field_name : str
        Field to pseudonymize
    **kwargs : dict
        Additional parameters to override defaults

    Returns
    --------
    HashBasedPseudonymizationOperation
        Configured hash pseudonymization operation
    """
    return HashBasedPseudonymizationOperation(field_name=field_name, **kwargs)
