"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Consistent Mapping Pseudonymization Operation
Package:       pamola_core.anonymization.pseudonymization
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-05-20
Updated:       2025-06-20
License:       BSD 3-Clause

Description:
    This module implements consistent mapping pseudonymization for sensitive data fields.
    It provides reversible transformation of identifiers into pseudonyms using encrypted
    mapping storage, enabling data re-identification when necessary with proper authorization.

Key Features:
    - Reversible pseudonymization with encrypted mapping storage
    - Support for UUID, sequential, and random string pseudonym generation
    - Atomic file operations for mapping persistence
    - Thread-safe concurrent processing
    - Batch processing with automatic persistence
    - Support for REPLACE and ENRICH modes
    - Compound identifier support for multi-field pseudonymization
    - Integration with PAMOLA framework standards (7-step lifecycle)
    - Comprehensive metrics collection and visualization

Security Considerations:
    - All mappings are encrypted using AES-256-GCM
    - Encryption keys must be 256-bit (32 bytes)
    - Atomic file operations prevent corruption
    - Thread-safe operations for concurrent access
    - No plaintext mappings in memory or logs

Changelog:
    1.0.0 (2025-01-20): Initial implementation
    1.0.1 (2025-06-15): Updated imports to use validation_utils facade
    1.0.2 (2025-06-15): P-1 through P-9 fixes
    1.1.0 (2025-06-20): Refactored to 7-step lifecycle; imported config from schema module
"""

import secrets
import string
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from pamola_core.common.helpers.data_helper import DataHelper
from pamola_core.errors.codes import ErrorCode
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.exceptions import (
    PseudonymizationError,
    FieldNotFoundError,
    InvalidParameterError,
    PamolaFileNotFoundError,
    ValidationError,
)

# Base anonymization operation import
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Config schema import (replaces inline class)
from pamola_core.anonymization.schemas.mapping_op_core_schema import (
    ConsistentMappingPseudonymizationConfig,
)

# Mapping storage import
from pamola_core.anonymization.commons.mapping_storage import MappingStorage

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
    get_process_summary,
)

# Pseudonymization utilities imports
from pamola_core.anonymization.commons.pseudonymization_utils import (
    format_pseudonym_output,
)

# Validation imports
from pamola_core.anonymization.commons.validation_utils import check_field_exists

# Crypto helpers imports
from pamola_core.utils.crypto_helpers.pseudonymization import (
    PseudonymGenerator,
    validate_key_size,
)

# IO / framework imports
from pamola_core.utils.io import load_settings_operation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_field_utils import (
    create_composite_key,
)
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.helpers import filter_used_kwargs


@register(version="1.0.0")
class ConsistentMappingPseudonymizationOperation(AnonymizationOperation):
    """
    Consistent mapping pseudonymization with encrypted storage.

    This operation maintains a bidirectional mapping between original values
    and generated pseudonyms, enabling reversibility when needed. All mappings
    are stored encrypted using AES-256-GCM.

    Follows the standard PAMOLA 7-step operation lifecycle.
    """

    # Tells AnonymizationOperation.process_data() to use the pseudonymization
    # null-anonymize placeholder ("*REDACTED*") instead of "SUPPRESSED".
    _is_pseudonymization: bool = True

    def __init__(
        self,
        field_name: str,
        mapping_encryption_key: Union[str, bytes],
        additional_fields: Optional[List[str]] = None,
        mapping_file: Optional[Union[str, Path]] = None,
        mapping_format: str = "csv",
        pseudonym_type: str = "uuid",
        pseudonym_prefix: Optional[str] = None,
        pseudonym_suffix: Optional[str] = None,
        pseudonym_length: int = 36,
        create_if_not_exists: bool = True,
        backup_on_update: bool = True,
        persist_frequency: int = 1000,
        quasi_identifiers: Optional[List[str]] = None,
        compound_mode: bool = False,
        compound_separator: str = "|",
        compound_null_handling: str = "skip",
        **kwargs,
    ):
        """
        Initialize consistent mapping pseudonymization operation.

        Parameters
        -----------
        field_name : str
            Primary field to pseudonymize
        mapping_encryption_key : Union[str, bytes]
            256-bit encryption key (hex string or bytes) for mapping storage
        additional_fields : Optional[List[str]]
            Additional fields for compound pseudonymization
        mapping_file : Optional[Union[str, Path]]
            Custom mapping file name (auto-generated if None)
        mapping_format : str
            Format for mapping storage: "csv" or "json" (default: "csv")
        pseudonym_type : str
            Type: "uuid", "sequential", or "random_string" (default: "uuid")
        pseudonym_prefix : Optional[str]
            Prefix for pseudonyms (default: None)
        pseudonym_suffix : Optional[str]
            Suffix for pseudonyms (default: None)
        pseudonym_length : int
            Length for random_string type (default: 36)
        create_if_not_exists : bool
            Create mapping file if missing (default: True)
        backup_on_update : bool
            Backup before updates (default: True)
        persist_frequency : int
            Save after N new mappings (default: 1000)
        quasi_identifiers : Optional[List[str]]
            Quasi-identifiers for privacy metrics
        compound_mode : bool
            Whether to create compound identifiers (default: False)
        compound_separator : str
            Separator for compound identifiers (default: "|")
        compound_null_handling : str
            How to handle nulls in compounds (default: "skip")
        **kwargs : dict
            Additional parameters passed to AnonymizationOperation
            (mode, null_strategy, condition_field, ka_risk_field, etc.)
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Consistent mapping pseudonymization for '{field_name}' using {pseudonym_type}",
        )

        # Validate and normalize mapping encryption key
        if isinstance(mapping_encryption_key, str):
            try:
                _mapping_encryption_key = bytes.fromhex(mapping_encryption_key)
            except ValueError as e:
                raise ValidationError(f"Invalid hex encryption key: {e}")
        else:
            _mapping_encryption_key = mapping_encryption_key

        # Validate key size
        try:
            validate_key_size(_mapping_encryption_key, 256)
        except Exception as e:
            raise ValidationError("Invalid encryption key size") from e

        # Normalize list-typed params at the source so the setattr loop below
        # propagates real lists to self (avoids None vs [] inconsistency).
        additional_fields = additional_fields or []
        quasi_identifiers = quasi_identifiers or []

        # Build config object (store key as hex for serialization)
        config = ConsistentMappingPseudonymizationConfig(
            field_name=field_name,
            additional_fields=additional_fields,
            mapping_file=str(mapping_file) if mapping_file else None,
            mapping_format=mapping_format,
            pseudonym_type=pseudonym_type,
            pseudonym_prefix=pseudonym_prefix,
            pseudonym_suffix=pseudonym_suffix,
            pseudonym_length=pseudonym_length,
            mapping_encryption_key=_mapping_encryption_key.hex(),
            create_if_not_exists=create_if_not_exists,
            backup_on_update=backup_on_update,
            persist_frequency=persist_frequency,
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

        # Operation metadata
        self.operation_name = self.__class__.__name__
        self.version = "1.1.0"

        # Store encryption key as private bytes; remove the public hex attr set by
        # the config setattr loop above to prevent accidental serialization/logging.
        # (Defense-in-depth: ConsistentMappingPseudonymizationConfig.SENSITIVE_KEYS
        # also redacts this key when save_config() writes config.json to disk.)
        self._mapping_encryption_key = _mapping_encryption_key
        if hasattr(self, "mapping_encryption_key"):
            delattr(self, "mapping_encryption_key")

        # Initialize components
        self._pseudonym_generator = PseudonymGenerator(pseudonym_type)

        # Runtime state (initialized during execute)
        self._mapping_storage: Optional[MappingStorage] = None
        self._mapping: Dict[str, str] = {}
        self._reverse_mapping: Dict[str, str] = {}
        self._new_mappings_count = 0
        self._total_new_mappings = (
            0  # cumulative; never reset (unlike _new_mappings_count)
        )
        self._total_lookups = 0
        self._mapping_hits = 0
        self._mapping_lock = threading.RLock()
        self._mapping_path: Optional[Path] = None
        self._sequential_counter = 0
        self._collision_count = 0
        self._generated_pseudonyms: set = set()
        # Welford running mean for lookup timing (O(1) memory vs list-based)
        self._lookup_time_count = 0
        self._lookup_time_mean = 0.0
        self._persist_count = 0

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the consistent mapping pseudonymization operation.

        Follows the standard 7-step PAMOLA lifecycle:
        1. Data Loading & Validation
        2. Cache check
        3. Prepare output field
        4. Processing (with mapping initialization)
        5. Metrics
        6. Visualization
        7. Save output

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
        **kwargs
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
            df = None

            dataset_name = kwargs.get("dataset_name", "main")
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare directories and infrastructure
            dirs = self._prepare_directories(task_dir)

            self.operation_cache = OperationCache(cache_dir=dirs["cache"])

            # Initialize error handler early so outer except can use it
            self.error_handler = ErrorHandler(
                logger=self.logger,
                operation_name=self.operation_name,
            )

            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            self.save_config(task_dir)

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, "
                f"backend={self.visualization_backend}, strict={self.visualization_strict}, "
                f"timeout={self.visualization_timeout}s"
            )

            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )

            # Progress setup (7 main steps)
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
                            "step": "Starting mapping pseudonymization",
                            "field": self.field_name,
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # ----------------------------------------------------------
            # Step 1: Data Loading & Validation
            # ----------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Data Loading", "field": self.field_name}
                )

            # Initialize mapping storage
            try:
                self._initialize_mapping(task_dir)
            except Exception as e:
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.MAPPING_ERROR,
                    context={"operation": self.operation_name, "field": self.field_name},
                    message_kwargs={
                        "context": f"initialize_mapping:{self.field_name}",
                        "reason": str(e),
                    },
                )

            try:
                self._validate_configuration()
                self.logger.info(
                    f"Operation: {self.operation_name}, Load data and validate input parameters"
                )
                df = self._validate_and_get_dataframe(
                    data_source, dataset_name, **settings_operation
                )

                # Validate all required fields exist
                all_fields = [self.field_name] + self.additional_fields
                for field in all_fields:
                    if not check_field_exists(df, field):
                        raise FieldNotFoundError(
                            field_name=field,
                            available_fields=list(df.columns),
                        )
            except Exception as e:
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.DATA_LOAD_FAILED,
                    context={"dataset": dataset_name, "operation": self.operation_name},
                    message_kwargs={"source": dataset_name, "reason": str(e)},
                )

            # ----------------------------------------------------------
            # Step 2: Cache check
            # ----------------------------------------------------------
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
                            f"Mapping pseudonymization of {self.field_name} (cached)",
                            details={"cached": True},
                        )
                    return cache_result

            # ----------------------------------------------------------
            # Step 3: Prepare output field
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # Step 4: Processing
            # ----------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Processing", "field": self.field_name}
                )

            try:
                # Normalize integer dtype if required
                df[self.field_name] = DataHelper.normalize_int_dtype_vectorized(
                    df[self.field_name], safe_mode=False
                )

                # Store original data for metrics
                all_fields = [self.field_name] + self.additional_fields
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

                # Apply conditional filtering then process
                self.filter_mask, filtered_df = self._apply_conditional_filtering(df)

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

                # Persist any remaining new mappings
                if self._new_mappings_count > 0:
                    self._persist_mappings()

                # Handle vulnerable records if k-anonymity is enabled
                if self.ka_risk_field and self.ka_risk_field in df.columns:
                    processed_df = self._handle_vulnerable_records(
                        processed_df, self.output_field_name
                    )

                # Get anonymized data for metrics
                # In REPLACE compound mode, additional_fields are nulled out by process_batch,
                # so use primary field directly
                if self.mode == "REPLACE":
                    anonymized_data = processed_df[self.field_name].copy(deep=True)
                else:
                    anonymized_data = processed_df[self.output_field_name].copy(
                        deep=True
                    )

                if data_tracker:
                    try:
                        data_tracker.close()
                    except Exception:
                        pass

            except Exception as e:
                # Attempt to persist any completed mappings before returning error
                try:
                    if self._new_mappings_count > 0:
                        self._persist_mappings()
                except Exception:
                    pass
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

            # Record end time
            self.end_time = time.time()
            if self.end_time and self.start_time:
                self.execution_time = self.end_time - self.start_time

            # ----------------------------------------------------------
            # Step 5: Metrics Calculation
            # ----------------------------------------------------------
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

                metrics_file_name = f"{self.field_name}_mapping_pseudonymization_metrics_{operation_timestamp}"
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
                # Non-critical — continue execution

            # ----------------------------------------------------------
            # Step 6: Visualization
            # ----------------------------------------------------------
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
            else:
                self.logger.info(
                    "Skipping visualizations as generate_visualization is False or backend is not set"
                )

            # ----------------------------------------------------------
            # Step 7: Save Output Data
            # ----------------------------------------------------------
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Save Output Data", "field": self.field_name},
                )

            if self.save_output:
                try:
                    safe_kwargs = filter_used_kwargs(
                        kwargs,
                        ConsistentMappingPseudonymizationOperation._save_output_data,
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

            # Cache result if enabled
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

            # Cleanup memory
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                processed_df=processed_df,
                original_data=original_data,
                anonymized_data=anonymized_data,
            )

            if reporter:
                reporter.add_operation(
                    f"Mapping pseudonymization of {self.field_name} completed",
                    details={
                        "records_processed": self.process_count,
                        "execution_time": self.execution_time,
                        "total_mappings": len(self._mapping),
                        "new_mappings": self._total_new_mappings,
                    },
                )

            result.status = OperationStatus.SUCCESS
            result.execution_time = self.execution_time
            self.logger.info(
                f"Processing completed {self.operation_name} operation "
                f"in {self.execution_time:.2f} seconds"
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
            # Release mapping storage reference to free memory between runs
            self._mapping_storage = None
            # Reset per-execution counters after every run (success or failure)
            # so the next execute() call starts from a clean state.
            self._new_mappings_count = 0
            self._total_new_mappings = 0
            self._total_lookups = 0
            self._mapping_hits = 0
            self._collision_count = 0
            self._lookup_time_count = 0
            self._lookup_time_mean = 0.0
            self._persist_count = 0

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
        # Defensive `list(... or [])` guard so this public method is safe to
        # call independently (additional_fields may be None after config reload).
        all_fields = [self.field_name] + list(self.additional_fields or [])
        for field in all_fields:
            if field not in batch.columns:
                raise FieldNotFoundError(
                    field_name=field,
                    available_fields=list(batch.columns),
                )

        # Create working series based on mode
        if self.compound_mode:
            working_series = create_composite_key(
                batch, all_fields, self.compound_separator, self.compound_null_handling
            )
        else:
            working_series = batch[self.field_name].copy(deep=True)

        # Null handling is done by base class _process_data_with_config() before calling process_batch

        # Pseudonymize non-null values
        non_null_mask = working_series.notna()
        non_null_values = working_series[non_null_mask]

        if len(non_null_values) > 0:
            for idx, value in non_null_values.items():
                str_value = str(value)

                with self._mapping_lock:
                    # Measure only actual lookup/generation time, not lock wait
                    lookup_start = time.perf_counter()
                    self._total_lookups += 1

                    if str_value in self._mapping:
                        pseudonym = self._mapping[str_value]
                        self._mapping_hits += 1
                    else:
                        pseudonym = self._generate_unique_pseudonym()
                        self._mapping[str_value] = pseudonym
                        self._reverse_mapping[pseudonym] = str_value
                        self._new_mappings_count += 1
                        self._total_new_mappings += 1

                    # Welford running mean: update inside lock to avoid race on mean state
                    lookup_time = time.perf_counter() - lookup_start
                    self._lookup_time_count += 1
                    delta = lookup_time - self._lookup_time_mean
                    self._lookup_time_mean += delta / self._lookup_time_count

                working_series.at[idx] = pseudonym

                # Persist at frequency threshold (within batch loop)
                with self._mapping_lock:
                    if self._new_mappings_count >= self.persist_frequency:
                        self._persist_mappings()

        # Update the DataFrame using self.output_field_name (set in Step 3)
        if self.mode == "REPLACE":
            if self.compound_mode:
                # Replace primary field; null out additional fields
                batch[self.field_name] = working_series
                for field in self.additional_fields or []:
                    batch[field] = None
            else:
                batch[self.field_name] = working_series
        else:  # ENRICH
            batch[self.output_field_name] = working_series

        return batch

    def _validate_configuration(self) -> None:
        """Validate operation configuration before execution."""
        if self.pseudonym_type not in ["uuid", "sequential", "random_string"]:
            raise InvalidParameterError(
                param_name="pseudonym_type",
                param_value=self.pseudonym_type,
                reason=f"Unknown pseudonym type: {self.pseudonym_type}. "
                f"Must be one of: uuid, sequential, random_string",
            )

        if self.mapping_format not in ["csv", "json"]:
            raise InvalidParameterError(
                param_name="mapping_format",
                param_value=self.mapping_format,
                reason=f"Unsupported mapping format: {self.mapping_format}. "
                f"Must be one of: csv, json",
            )

        if self.pseudonym_type == "random_string":
            prefix_len = len(self.pseudonym_prefix) if self.pseudonym_prefix else 0
            suffix_len = len(self.pseudonym_suffix) if self.pseudonym_suffix else 0
            if self.pseudonym_length - prefix_len - suffix_len < 4:
                raise InvalidParameterError(
                    param_name="pseudonym_length",
                    param_value=self.pseudonym_length,
                    reason=(
                        f"Effective pseudonym length after prefix/suffix "
                        f"({self.pseudonym_length - prefix_len - suffix_len}) must be at least 4"
                    ),
                )

        if self.compound_mode and not self.additional_fields:
            raise InvalidParameterError(
                param_name="compound_mode",
                param_value=self.compound_mode,
                reason="compound_mode requires additional_fields to be specified",
            )

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """Return parameters that determine cache key uniqueness."""
        return dict(
            pseudonym_type=self.pseudonym_type,
            pseudonym_prefix=self.pseudonym_prefix,
            pseudonym_length=self.pseudonym_length,
            mapping_format=self.mapping_format,
            compound_mode=self.compound_mode,
        )

    def _initialize_mapping(self, task_dir: Path) -> None:
        """Initialize mapping storage and load existing mappings."""
        maps_dir = task_dir / "maps"
        maps_dir.mkdir(exist_ok=True)

        # Determine mapping file path
        if not self.mapping_file:
            operation_name = f"{self.field_name}_mapping"
            if self.compound_mode:
                operation_name = f"compound_{operation_name}"
            self.mapping_file = f"{operation_name}.{self.mapping_format}.enc"

        self._mapping_path = maps_dir / self.mapping_file
        self.logger.info(f"Mapping file path: {self._mapping_path}")

        self._mapping_storage = MappingStorage(
            mapping_file=self._mapping_path,
            encryption_key=self._mapping_encryption_key,
            format=self.mapping_format,
            backup_on_update=self.backup_on_update,
        )

        try:
            # Load with metadata support
            loaded_data = self._mapping_storage.load()

            if isinstance(loaded_data, dict) and "_metadata" in loaded_data:
                self._mapping = loaded_data.get("mappings", {})
                metadata = loaded_data["_metadata"]
                if self.pseudonym_type == "sequential":
                    self._sequential_counter = metadata.get("last_sequential", 0)
                    self.logger.info(
                        f"Restored sequential counter: {self._sequential_counter}"
                    )
            else:
                # Legacy format
                self._mapping = loaded_data
                if self.pseudonym_type == "sequential" and self._mapping:
                    self._calculate_sequential_counter()

            self._reverse_mapping = {v: k for k, v in self._mapping.items()}
            self._generated_pseudonyms = set(self._reverse_mapping.keys())
            self.logger.info(f"Loaded {len(self._mapping)} existing mappings")

        except (PamolaFileNotFoundError, FileNotFoundError):
            if self.create_if_not_exists:
                self.logger.info("Creating new mapping file")
                self._mapping = {}
                self._reverse_mapping = {}
                self._sequential_counter = 0
                self._generated_pseudonyms = set()
            else:
                raise

    def _persist_mappings(self) -> None:
        """Save current mappings to encrypted file."""
        with self._mapping_lock:
            try:
                # Save with metadata
                save_data = {
                    "mappings": self._mapping,
                    "_metadata": {
                        "last_sequential": self._sequential_counter,
                        "total_mappings": len(self._mapping),
                        "last_updated": datetime.now().isoformat(),
                        "pseudonym_type": self.pseudonym_type,
                        "version": self.version,
                    },
                }
                self._mapping_storage.save(save_data)
                self.logger.info(
                    f"Persisted {len(self._mapping)} mappings ({self._new_mappings_count} new)"
                )
                self._new_mappings_count = 0
                self._persist_count += 1
            except Exception as e:
                self.logger.error(f"Failed to persist mappings: {e}")
                raise

    def _calculate_sequential_counter(self) -> None:
        """Calculate sequential counter from existing mappings."""
        max_seq = 0
        for pseudonym in self._reverse_mapping.keys():
            num_part = pseudonym
            if self.pseudonym_prefix:
                num_part = num_part.replace(self.pseudonym_prefix, "")
            if self.pseudonym_suffix:
                num_part = num_part.replace(self.pseudonym_suffix, "")
            try:
                seq_num = int(num_part)
                max_seq = max(max_seq, seq_num)
            except ValueError:
                continue
        self._sequential_counter = max_seq

    def _generate_unique_pseudonym(self) -> str:
        """
        Generate a unique pseudonym that doesn't exist in current mappings.

        Returns
        --------
        str
            Unique pseudonym
        """
        if self.pseudonym_type == "uuid":
            pseudonym = self._pseudonym_generator.generate_unique(
                self._generated_pseudonyms, prefix=self.pseudonym_prefix
            )
        elif self.pseudonym_type == "sequential":
            self._sequential_counter += 1
            # zfill width grows automatically beyond 999999 — no silent truncation
            width = max(6, len(str(self._sequential_counter)))
            pseudonym = str(self._sequential_counter).zfill(width)
            pseudonym = format_pseudonym_output(
                pseudonym, self.pseudonym_prefix, self.pseudonym_suffix
            )
        elif self.pseudonym_type == "random_string":
            characters = string.ascii_letters + string.digits
            prefix_len = len(self.pseudonym_prefix) if self.pseudonym_prefix else 0
            suffix_len = len(self.pseudonym_suffix) if self.pseudonym_suffix else 0
            random_len = self.pseudonym_length - prefix_len - suffix_len

            attempts = 0
            while True:
                random_part = "".join(
                    secrets.choice(characters) for _ in range(random_len)
                )
                pseudonym = format_pseudonym_output(
                    random_part, self.pseudonym_prefix, self.pseudonym_suffix
                )
                if pseudonym not in self._generated_pseudonyms:
                    break
                attempts += 1
                if attempts > 10:
                    self._collision_count += 1
                    self.logger.warning(
                        "High collision rate detected for random_string generation"
                    )
                if attempts > 100:
                    raise PseudonymizationError(
                        field_name=self.field_name,
                        reason="unable to generate unique pseudonym after 100 attempts",
                        pseudonym_type=self.pseudonym_type,
                        attempts=attempts,
                        max_attempts=100,
                    )
        else:
            raise InvalidParameterError(
                param_name="pseudonym_type",
                param_value=self.pseudonym_type,
                reason=f"Unknown pseudonym type: {self.pseudonym_type}",
            )

        self._generated_pseudonyms.add(pseudonym)
        return pseudonym

    def _collect_comprehensive_metrics(
        self,
        original_series: pd.Series,
        processed_series: pd.Series,
        full_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Collect all metrics using commons utilities."""
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

        operation_metrics = collect_operation_metrics(
            operation_type="pseudonymization",
            original_data=original_series,
            processed_data=processed_series,
            operation_params={
                "pseudonym_type": self.pseudonym_type,
                "compound_mode": self.compound_mode,
                "reversible": True,
                "encryption": "AES-256-GCM",
            },
            timing_info=timing_info,
        )

        # Effectiveness metrics
        effectiveness = calculate_anonymization_effectiveness(
            original_series, processed_series
        )
        operation_metrics["effectiveness"] = effectiveness

        # Average lookup time from Welford running mean (O(1) memory, no list needed)
        avg_lookup_time_ms = self._lookup_time_mean * 1000

        # Hit rate and percent new (use cumulative counter, not reset-on-persist one)
        hit_rate = (
            self._mapping_hits / self._total_lookups if self._total_lookups > 0 else 0.0
        )
        percent_new = (
            (self._total_new_mappings / self._total_lookups * 100)
            if self._total_lookups > 0
            else 0.0
        )

        metadata = self._mapping_storage.get_metadata() if self._mapping_storage else {}

        mapping_metrics: Dict[str, Any] = {
            "pseudonym_type": self.pseudonym_type,
            "total_mappings": len(self._mapping),
            "new_mappings_created": self._total_new_mappings,
            "mapping_file_size": metadata.get("size_bytes", 0),
            "mapping_file_path": str(self._mapping_path),
            "encryption_algorithm": "AES-256-GCM",
            "persist_frequency": self.persist_frequency,
            "reversible": True,
            "lookup_time_avg": round(avg_lookup_time_ms, 4),
            "mapping_hit_rate": round(hit_rate, 4),
            "percent_new_mappings": round(percent_new, 2),
            "total_lookups": self._total_lookups,
            "mapping_hits": self._mapping_hits,
            "persistence_count": self._persist_count,
        }

        if self.pseudonym_type == "random_string":
            mapping_metrics["collision_count"] = self._collision_count

        operation_metrics["mapping"] = mapping_metrics

        # Privacy metrics if quasi-identifiers available
        output_col = (
            self.output_field_name if self.mode == "ENRICH" else self.field_name
        )
        if self.quasi_identifiers and all(
            qi in full_df.columns for qi in self.quasi_identifiers
        ):
            privacy_metrics = calculate_batch_metrics(
                original_batch=full_df[[self.field_name] + self.quasi_identifiers],
                anonymized_batch=full_df[[output_col] + self.quasi_identifiers],
                original_field_name=self.field_name,
                anonymized_field_name=output_col,
                quasi_identifiers=self.quasi_identifiers,
            )
            operation_metrics["privacy_metrics"] = privacy_metrics
            privacy_metrics["disclosure_risk"] = calculate_simple_disclosure_risk(
                full_df, self.quasi_identifiers
            )

        # Performance metrics
        performance = calculate_process_performance(
            timing_info["start_time"],
            timing_info["end_time"],
            len(full_df),
            timing_info["batch_count"],
        )
        operation_metrics["performance"] = performance

        return operation_metrics

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def get_reverse_mapping(self, pseudonym: str) -> Optional[str]:
        """
        Get original value for a pseudonym (for authorized reversal).

        Parameters
        -----------
        pseudonym : str
            Pseudonym to reverse

        Returns
        --------
        Optional[str]
            Original value if found, None otherwise
        """
        with self._mapping_lock:
            return self._reverse_mapping.get(pseudonym)

    def export_mappings(self, output_path: Path, include_metadata: bool = True) -> None:
        """
        Export mappings in encrypted form (for backup/transfer).

        Parameters
        -----------
        output_path : Path
            Path to export to
        include_metadata : bool
            Whether to include metadata
        """
        with self._mapping_lock:
            # Always include metadata for proper restore
            export_data = {
                "mappings": self._mapping,
                "_metadata": (
                    {
                        "field_name": self.field_name,
                        "pseudonym_type": self.pseudonym_type,
                        "total_mappings": len(self._mapping),
                        "last_sequential": self._sequential_counter,
                        "export_timestamp": datetime.now().isoformat(),
                        "version": self.version,
                        "mapping_format": self.mapping_format,
                    }
                    if include_metadata
                    else {}
                ),
            }

            # Use mapping storage with correct format
            temp_storage = MappingStorage(
                mapping_file=output_path,
                encryption_key=self._mapping_encryption_key,
                format=self.mapping_format,
                backup_on_update=False,
            )
            temp_storage.save(export_data)

    def __getstate__(self):
        # Exclude non-picklable attrs containing threading locks so Dask's
        # _normalize_pickle determinism check can serialize this object.
        # Actual execution uses the original in-memory instance (not the unpickled one).
        state = self.__dict__.copy()
        state.pop("_mapping_lock", None)  # threading.RLock — not picklable
        state.pop("_mapping_storage", None)  # MappingStorage — contains threading.RLock
        state.pop(
            "_pseudonym_generator", None
        )  # PseudonymGenerator — contains threading.Lock
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mapping_lock = threading.RLock()
        self._mapping_storage = None
        self._pseudonym_generator = None


# Factory function
def create_mapping_pseudonymization_operation(
    field_name: str, mapping_encryption_key: Union[str, bytes], **kwargs
) -> ConsistentMappingPseudonymizationOperation:
    """
    Create a consistent mapping pseudonymization operation with default settings.

    Parameters
    -----------
    field_name : str
        Field to pseudonymize
    mapping_encryption_key : Union[str, bytes]
        256-bit encryption key for mapping storage
    **kwargs
        Additional parameters to override defaults

    Returns
    --------
    ConsistentMappingPseudonymizationOperation
        Configured mapping pseudonymization operation
    """
    return ConsistentMappingPseudonymizationOperation(
        field_name=field_name, mapping_encryption_key=mapping_encryption_key, **kwargs
    )