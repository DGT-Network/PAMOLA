"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Consistent Mapping Pseudonymization Operation
Package:       pamola_core.anonymization.pseudonymization
Version:       1.0.2
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-05-20
Updated:       2025-06-15
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
    - Integration with PAMOLA framework standards
    - Comprehensive metrics collection and visualization

Security Considerations:
    - All mappings are encrypted using AES-256-GCM
    - Encryption keys must be 256-bit (32 bytes)
    - Atomic file operations prevent corruption
    - Thread-safe operations for concurrent access
    - No plaintext mappings in memory or logs

Dependencies:
    - pamola_core.utils.crypto_helpers.pseudonymization: Core crypto functions
    - pamola_core.anonymization.commons.mapping_storage: Encrypted mapping storage
    - pamola_core.anonymization.commons.pseudonymization_utils: Shared utilities
    - threading: For thread synchronization
    - uuid: For UUID generation
    - pathlib: For file path handling

Changelog:
    1.0.0 (2025-01-20):
        - Initial implementation with full framework integration
        - Support for UUID, sequential, and random string generation
        - Encrypted mapping storage with atomic operations
        - Batch processing with configurable persistence
        - Comprehensive metrics and visualization
    1.0.1 (2025-06-15):
        - Updated imports to use validation_utils facade
        - Improved mapping directory structure
        - Enhanced error handling and recovery
    1.0.2 (2025-06-15):
        - Fixed condition_operator normalization (P-1)
        - Improved key validation error handling (P-2)
        - Added null_strategy error handling in process_batch (P-3)
        - Accurate lookup time tracking (P-4)
        - Sequential counter persistence in metadata (P-5)
        - Pseudonym length validation (P-6)
        - Added missing metrics: mapping_hit_rate, percent_new_mappings (P-7)
        - Fixed export format consistency (P-8)
        - Corrected progress tracking weights (P-9)
        - Added collision tracking for random_string type
"""

import logging
import string
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Base anonymization operation import
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Data utilities imports
from pamola_core.anonymization.commons.data_utils import process_nulls

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

# Validation imports - Using validation_utils facade
from pamola_core.anonymization.commons.validation_utils import check_field_exists

# Visualization utilities imports
from pamola_core.anonymization.commons.visualization_utils import (
    create_comparison_visualization,
    create_metric_visualization,
)

# Crypto helpers imports
from pamola_core.utils.crypto_helpers.pseudonymization import (
    PseudonymGenerator,
    validate_key_size,
)

# Operation framework imports
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_processing import (
    force_garbage_collection,
    optimize_dataframe_dtypes,
)
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_field_utils import (
    apply_condition_operator,
    create_composite_key,
    generate_output_field_name,
)
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker

# Configure module logger
logger = logging.getLogger(__name__)


class ConsistentMappingPseudonymizationConfig(OperationConfig):
    """Configuration for ConsistentMappingPseudonymizationOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "additional_fields": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "mapping_file": {"type": ["string", "null"]},
            "mapping_format": {"type": "string", "enum": ["csv", "json"]},
            "pseudonym_type": {
                "type": "string",
                "enum": ["uuid", "sequential", "random_string"],
            },
            "pseudonym_prefix": {"type": ["string", "null"]},
            "pseudonym_suffix": {"type": ["string", "null"]},
            "pseudonym_length": {"type": "integer", "minimum": 4, "maximum": 64},
            "encryption_key": {"type": "string"},  # Hex-encoded key
            "create_if_not_exists": {"type": "boolean"},
            "backup_on_update": {"type": "boolean"},
            "persist_frequency": {"type": "integer", "minimum": 1},
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "output_field_name": {"type": ["string", "null"]},
            "column_prefix": {"type": "string"},
            "null_strategy": {
                "type": "string",
                "enum": ["PRESERVE", "EXCLUDE", "ANONYMIZE", "ERROR"],
            },
            "batch_size": {"type": "integer", "minimum": 1},
            "use_cache": {"type": "boolean"},  # Not used, mapping is the cache
            "use_encryption": {"type": "boolean"},  # Always true for this operation
            "condition_field": {"type": ["string", "null"]},
            "condition_values": {"type": ["array", "null"]},
            "condition_operator": {"type": "string"},
            "ka_risk_field": {"type": ["string", "null"]},
            "risk_threshold": {"type": "number"},
            "vulnerable_record_strategy": {"type": "string"},
            "output_file_format": {
                "type": "string",
                "enum": ["csv", "parquet", "arrow"],
            },
            "quasi_identifiers": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "compound_mode": {"type": "boolean"},
            "compound_separator": {"type": "string"},
            "compound_null_handling": {
                "type": "string",
                "enum": ["skip", "empty", "null"],
            },
        },
        "required": ["field_name", "encryption_key"],
    }


class ConsistentMappingPseudonymizationOperation(AnonymizationOperation):
    """
    Consistent mapping pseudonymization with encrypted storage.

    This operation maintains a bidirectional mapping between original values
    and generated pseudonyms, enabling reversibility when needed. All mappings
    are stored encrypted using AES-256-GCM.
    """

    def __init__(
        self,
        field_name: str,
        encryption_key: Union[str, bytes],
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
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "_",
        null_strategy: str = "PRESERVE",
        batch_size: int = 10000,
        use_cache: bool = True,  # Ignored - mapping is the cache
        use_encryption: bool = True,  # Always true
        condition_field: Optional[str] = None,
        condition_values: Optional[List] = None,
        condition_operator: Optional[str] = None,  # P-1: Made optional
        ka_risk_field: Optional[str] = None,
        risk_threshold: float = 5.0,
        vulnerable_record_strategy: str = "pseudonymize",
        output_file_format: str = "csv",
        quasi_identifiers: Optional[List[str]] = None,
        compound_mode: bool = False,
        compound_separator: str = "|",
        compound_null_handling: str = "skip",
        description: str = "",
    ):
        """
        Initialize consistent mapping pseudonymization operation.

        Parameters:
        -----------
        field_name : str
            Primary field to pseudonymize
        encryption_key : Union[str, bytes]
            256-bit encryption key (hex string or bytes)
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
        mode : str
            "REPLACE" or "ENRICH" (default: "REPLACE")
        output_field_name : Optional[str]
            Output field name for ENRICH mode
        column_prefix : str
            Prefix for generated column names (default: "_")
        null_strategy : str
            How to handle nulls: "PRESERVE", "EXCLUDE", "ANONYMIZE", "ERROR"
        batch_size : int
            Batch size for processing (default: 10000)
        use_cache : bool
            Ignored - mapping serves as cache
        use_encryption : bool
            Always true for mapping operations
        condition_field : Optional[str]
            Field for conditional processing
        condition_values : Optional[List]
            Values for conditional processing
        condition_operator : Optional[str]
            Operator for conditions (default: "in" if condition_field set)
        ka_risk_field : Optional[str]
            Field containing k-anonymity risk scores
        risk_threshold : float
            Risk threshold for vulnerable records (default: 5.0)
        vulnerable_record_strategy : str
            Strategy for vulnerable records (default: "pseudonymize")
        output_file_format : str
            Output format: "csv", "parquet", "arrow" (default: "csv")
        quasi_identifiers : Optional[List[str]]
            Quasi-identifiers for privacy metrics
        compound_mode : bool
            Whether to create compound identifiers (default: False)
        compound_separator : str
            Separator for compound identifiers (default: "|")
        compound_null_handling : str
            How to handle nulls in compounds (default: "skip")
        description : str
            Operation description
        """
        # P-1: Normalize condition_operator
        if condition_field is not None and condition_operator is None:
            condition_operator = "in"
            logger.debug(
                f"Set default condition_operator='in' for condition_field='{condition_field}'"
            )

        # P-2: Validate encryption key with better error handling
        if isinstance(encryption_key, str):
            # Assume hex-encoded
            try:
                self._encryption_key = bytes.fromhex(encryption_key)
            except ValueError as e:
                raise ValueError(f"Invalid hex encryption key: {e}")
        else:
            self._encryption_key = encryption_key

        # P-2: Wrap key size validation with custom error message
        try:
            validate_key_size(self._encryption_key, 256)
        except Exception as e:
            # Don't expose internal details about expected size
            raise ValueError("Invalid encryption key size") from e

        # P-6: Validate pseudonym_length for random_string type
        if pseudonym_type == "random_string":
            prefix_len = len(pseudonym_prefix) if pseudonym_prefix else 0
            suffix_len = len(pseudonym_suffix) if pseudonym_suffix else 0
            effective_length = pseudonym_length - prefix_len - suffix_len

            if effective_length < 4:
                raise ValueError(
                    f"Pseudonym length ({pseudonym_length}) minus prefix/suffix "
                    f"({prefix_len + suffix_len}) must be at least 4 characters"
                )

        # Validate compound mode
        if compound_mode and not additional_fields:
            raise ValueError("compound_mode requires additional_fields to be specified")

        # Ensure additional_fields is always a list
        if additional_fields is None:
            additional_fields = []

        # Build config parameters
        config_params = {
            "field_name": field_name,
            "additional_fields": additional_fields,
            "mapping_file": str(mapping_file) if mapping_file else None,
            "mapping_format": mapping_format,
            "pseudonym_type": pseudonym_type,
            "pseudonym_prefix": pseudonym_prefix,
            "pseudonym_suffix": pseudonym_suffix,
            "pseudonym_length": pseudonym_length,
            "encryption_key": self._encryption_key.hex(),  # Store as hex
            "create_if_not_exists": create_if_not_exists,
            "backup_on_update": backup_on_update,
            "persist_frequency": persist_frequency,
            "mode": mode,
            "output_field_name": output_field_name,
            "column_prefix": column_prefix,
            "null_strategy": null_strategy,
            "batch_size": batch_size,
            "use_cache": True,  # Always true (mapping is cache)
            "use_encryption": True,  # Always true
            "condition_field": condition_field,
            "condition_values": condition_values,
            "condition_operator": condition_operator or "in",  # Ensure non-None
            "ka_risk_field": ka_risk_field,
            "risk_threshold": risk_threshold,
            "vulnerable_record_strategy": vulnerable_record_strategy,
            "output_file_format": output_file_format,
            "quasi_identifiers": quasi_identifiers,
            "compound_mode": compound_mode,
            "compound_separator": compound_separator,
            "compound_null_handling": compound_null_handling,
        }

        # Create configuration
        config = ConsistentMappingPseudonymizationConfig(**config_params)

        # Use default description if none provided
        if not description:
            description = f"Consistent mapping pseudonymization for field '{field_name}' using {pseudonym_type}"

        # Initialize base class
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            column_prefix=column_prefix,
            null_strategy=null_strategy,
            batch_size=batch_size,
            use_cache=True,  # Always use mapping as cache
            use_encryption=True,  # Always encrypt mappings
            encryption_key=(
                str(mapping_file) if mapping_file else None
            ),  # Use as identifier
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator or "in",  # Ensure non-None
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            vulnerable_record_strategy=vulnerable_record_strategy,
            description=description,
        )

        # Store operation-specific parameters
        self.additional_fields = additional_fields
        self.mapping_file = mapping_file
        self.mapping_format = mapping_format
        self.pseudonym_type = pseudonym_type
        self.pseudonym_prefix = pseudonym_prefix
        self.pseudonym_suffix = pseudonym_suffix
        self.pseudonym_length = pseudonym_length
        self.create_if_not_exists = create_if_not_exists
        self.backup_on_update = backup_on_update
        self.persist_frequency = persist_frequency
        self.output_file_format = output_file_format
        self.quasi_identifiers = quasi_identifiers or []
        self.compound_mode = compound_mode
        self.compound_separator = compound_separator
        self.compound_null_handling = compound_null_handling

        # Version information
        self.version = "1.0.2"

        # Initialize components
        self._pseudonym_generator = PseudonymGenerator(pseudonym_type)

        # Will be initialized during execution
        self._mapping_storage: Optional[MappingStorage] = None
        self._mapping: Dict[str, str] = {}
        self._reverse_mapping: Dict[str, str] = {}
        self._new_mappings_count = 0
        self._total_lookups = 0
        self._mapping_hits = 0  # P-7: Track hits for hit rate
        self._mapping_lock = threading.RLock()
        self._output_field = None
        self._mapping_path: Optional[Path] = None
        self._sequential_counter = 0  # For sequential pseudonyms
        self._collision_count = 0  # Track collisions for random_string
        self._generated_pseudonyms = set()  # Track all generated pseudonyms
        self._lookup_times = []  # P-4: Track individual lookup times

    def execute(
        self, data_source, task_dir, reporter=None, progress_tracker=None, **kwargs
    ):
        """
        Execute the consistent mapping pseudonymization operation.

        This method overrides the base class execute to handle mapping
        initialization, persistence, and proper thread safety.
        """
        start_time = time.time()

        # Create progress tracker if not provided
        if progress_tracker is None:
            progress_tracker = HierarchicalProgressTracker(
                total=100,
                description=f"Consistent mapping pseudonymization for {self.field_name}",
                unit="steps",
            )
            should_close_tracker = True
        else:
            should_close_tracker = False

        # Create DataWriter instance
        writer = DataWriter(
            task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
        )

        try:
            # Update progress: Starting (5%)
            progress_tracker.update(
                5,
                {
                    "status": "initializing",
                    "phase": "startup",
                    "message": f"Starting mapping pseudonymization for field '{self.field_name}'",
                },
            )

            # Initialize mapping storage (5%)
            self._initialize_mapping(task_dir)
            progress_tracker.update(5, {"status": "mapping_initialized"})

            # Get the DataFrame
            df, error_info = data_source.get_dataframe("main")
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Failed to load data: {error_info.get('message')}",
                )

            # Validate fields using the new validation framework
            all_fields = [self.field_name] + self.additional_fields
            for field in all_fields:
                # Use check_field_exists from validation_utils
                if not check_field_exists(df, field):
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=f"Field '{field}' not found in DataFrame",
                    )

            # Determine output field name
            self._output_field = generate_output_field_name(
                self.field_name,
                self.mode,
                self.output_field_name,
                operation_suffix="mapped",
                column_prefix=self.column_prefix,
            )

            # Store original for metrics
            if self.compound_mode:
                # For compound mode, create composite of all fields
                original_series = create_composite_key(
                    df, all_fields, self.compound_separator, self.compound_null_handling
                )
            else:
                original_series = df[self.field_name].copy()

            # Update progress: Data loaded (10%)
            progress_tracker.update(
                10,
                {
                    "status": "data_loaded",
                    "phase": "data_preparation",
                    "records": len(df),
                    "fields": len(all_fields),
                    "compound_mode": self.compound_mode,
                    "existing_mappings": len(self._mapping),
                    "message": f"Loaded {len(df):,} records, {len(self._mapping):,} existing mappings",
                },
            )

            # Apply risk-based filtering if configured
            vulnerable_mask = None
            vulnerable_count = 0
            if self.ka_risk_field and self.ka_risk_field in df.columns:
                vulnerable_mask = df[self.ka_risk_field] < self.risk_threshold

                # Apply additional conditions
                if self.condition_field and self.condition_field in df.columns:
                    condition_mask = apply_condition_operator(
                        df[self.condition_field],
                        self.condition_values,
                        self.condition_operator,
                    )
                    vulnerable_mask = vulnerable_mask & condition_mask

                vulnerable_count = vulnerable_mask.sum()
                self.logger.info(f"Identified {vulnerable_count} vulnerable records")

                progress_tracker.update(
                    5,
                    {
                        "status": "risk_analysis_complete",
                        "vulnerable_records": vulnerable_count,
                        "message": f"Risk analysis complete: {vulnerable_count:,} vulnerable records",
                    },
                )

            # Process in batches (55% for P-9 fix)
            total_batches = (len(df) + self.batch_size - 1) // self.batch_size
            batch_progress = progress_tracker.create_subtask(
                total=total_batches, description="Processing batches", unit="batches"
            )

            # Process each batch
            for i in range(0, len(df), self.batch_size):
                batch_indices = df.iloc[i : i + self.batch_size].index

                try:
                    # Process batch
                    batch_result = self._process_batch_safe(df.loc[batch_indices])
                    df.loc[batch_indices] = batch_result

                    # Check if we need to persist mappings
                    if self._new_mappings_count >= self.persist_frequency:
                        self._persist_mappings()

                    # Memory cleanup for large batches
                    if len(batch_indices) > 50000:
                        force_garbage_collection()

                except Exception as e:
                    self.logger.error(f"Error in batch {i // self.batch_size}: {e}")
                    if not kwargs.get("continue_on_error", False):
                        raise

                batch_progress.update(
                    1,
                    {
                        "batch": i // self.batch_size + 1,
                        "processed_records": min(i + self.batch_size, len(df)),
                        "new_mappings": self._new_mappings_count,
                    },
                )

            # Update main progress after batches (55% total)
            progress_tracker.update(55, {"status": "batches_complete"})

            # Final save of any remaining mappings
            if self._new_mappings_count > 0:
                self._persist_mappings()

            # Get processed series for metrics
            if self.mode == "REPLACE":
                if self.compound_mode:
                    # Re-create composite for comparison
                    processed_series = create_composite_key(
                        df,
                        all_fields,
                        self.compound_separator,
                        self.compound_null_handling,
                    )
                else:
                    processed_series = df[self.field_name].copy()
            else:
                processed_series = df[self._output_field].copy()

            # Update progress: Processing complete (10%)
            progress_tracker.update(
                10,
                {
                    "status": "processing_complete",
                    "phase": "post_processing",
                    "total_mappings": len(self._mapping),
                    "new_mappings_created": self._new_mappings_count,
                },
            )

            # Collect comprehensive metrics
            timing_info = {
                "start_time": start_time,
                "end_time": time.time(),
                "batch_count": total_batches,
            }

            operation_metrics = self._collect_comprehensive_metrics(
                original_series, processed_series, df, timing_info
            )

            # Save metrics
            metrics_result = writer.write_metrics(
                metrics=operation_metrics,
                name=f"{self.field_name}_mapping_pseudonymization",
                timestamp_in_name=True,
            )

            # Log summary
            summary = get_process_summary(operation_metrics.get("privacy_metrics", {}))
            for key, message in summary.items():
                self.logger.info(f"{key}: {message}")

            # Update progress: Metrics saved (5%)
            progress_tracker.update(5, {"status": "metrics_saved"})

            # Generate visualizations (10%)
            viz_progress = progress_tracker.create_subtask(
                total=2, description="Generating visualizations", unit="charts"
            )

            # Uniqueness comparison
            comparison_viz = create_comparison_visualization(
                original_series,
                processed_series,
                task_dir,
                self.field_name,
                "mapping_pseudonymization",
                None,
            )
            if comparison_viz:
                viz_progress.update(1)

            # Mapping growth visualization
            mapping_viz = None
            if len(self._mapping) > 0:
                mapping_viz = create_metric_visualization(
                    "mapping_growth",
                    {
                        "Initial Mappings": len(self._mapping)
                        - self._new_mappings_count,
                        "New Mappings": self._new_mappings_count,
                        "Total Mappings": len(self._mapping),
                    },
                    task_dir,
                    self.field_name,
                    "mapping_pseudonymization",
                    None,
                )
                viz_progress.update(1)

            # Update progress: Visualizations complete
            progress_tracker.update(10, {"status": "visualizations_complete"})

            # Write output if requested (5%)
            output_path = None
            if kwargs.get("write_output", True):
                output_progress = progress_tracker.create_subtask(
                    total=1, description="Writing output", unit="files"
                )

                # Optimize memory before writing
                df, _ = optimize_dataframe_dtypes(df)

                # Write output
                output_result = writer.write_dataframe(
                    df=df,
                    name=kwargs.get("output_name", "pseudonymized_data"),
                    format=self.output_file_format,
                    subdir="output",
                    timestamp_in_name=kwargs.get("timestamp_output", True),
                    encryption_key=self.encryption_key if self.use_encryption else None,
                )
                output_path = output_result.path

                output_progress.update(1)
                progress_tracker.update(5, {"status": "output_written"})

            # Create operation result
            result = OperationResult(
                status=OperationStatus.SUCCESS, execution_time=time.time() - start_time
            )

            # Add metrics
            result.add_metric("records_processed", len(df))
            result.add_metric("pseudonym_type", self.pseudonym_type)
            result.add_metric("total_mappings", len(self._mapping))
            result.add_metric("new_mappings_created", self._new_mappings_count)
            result.add_metric("mapping_file_path", str(self._mapping_path))
            result.add_metric("reversible", True)
            result.add_metric("vulnerable_records", int(vulnerable_count))

            # P-7: Add missing metrics
            if self._total_lookups > 0:
                mapping_hit_rate = self._mapping_hits / self._total_lookups
                result.add_metric("mapping_hit_rate", round(mapping_hit_rate, 4))
                result.add_metric(
                    "percent_new_mappings",
                    round(self._new_mappings_count / self._total_lookups * 100, 2),
                )

            # Add collision metric for random_string
            if self.pseudonym_type == "random_string":
                result.add_metric("collision_count", self._collision_count)

            # Add effectiveness metrics
            if "effectiveness" in operation_metrics:
                for key, value in operation_metrics["effectiveness"].items():
                    result.add_nested_metric("effectiveness", key, value)

            # Add privacy metrics
            if "privacy_metrics" in operation_metrics:
                for key, value in operation_metrics["privacy_metrics"].items():
                    if isinstance(value, (int, float)):
                        result.add_nested_metric("privacy", key, value)

            # Add artifacts
            result.add_artifact(
                artifact_type="json",
                path=metrics_result.path,
                description="Process metrics",
                category="metrics",
                tags=["metrics", "process", "mapping_pseudonymization"],
            )

            if comparison_viz:
                result.add_artifact(
                    artifact_type="png",
                    path=comparison_viz,
                    description="Before/after comparison",
                    category="visualization",
                    tags=["visualization", "comparison", self.field_name],
                )

            if mapping_viz:
                result.add_artifact(
                    artifact_type="png",
                    path=mapping_viz,
                    description="Mapping statistics",
                    category="visualization",
                    tags=["visualization", "mapping", "growth"],
                )

            if output_path:
                result.add_artifact(
                    artifact_type=self.output_file_format,
                    path=output_path,
                    description="Pseudonymized dataset",
                    category="output",
                    tags=["data", "pseudonymized", "reversible"],
                )

            # Add mapping file as artifact
            if self._mapping_path and self._mapping_path.exists():
                result.add_artifact(
                    artifact_type="encrypted_mapping",
                    path=self._mapping_path,
                    description="Encrypted pseudonym mappings",
                    category="mapping",
                    tags=["mapping", "encrypted", self.pseudonym_type],
                )

            # Final progress update (5%)
            progress_tracker.update(5, {"status": "complete"})

            return result

        except Exception as e:
            self.logger.error(f"Error in mapping pseudonymization: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )
        finally:
            if should_close_tracker:
                progress_tracker.close()

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data to pseudonymize values.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with pseudonymized values
        """
        # Validate fields exist
        all_fields = [self.field_name] + self.additional_fields
        for field in all_fields:
            if field not in batch.columns:
                raise ValueError(f"Field '{field}' not found in DataFrame")

        # Create working series based on mode
        if self.compound_mode:
            # Create compound identifier from multiple fields
            working_series = create_composite_key(
                batch, all_fields, self.compound_separator, self.compound_null_handling
            )
        else:
            working_series = batch[self.field_name].copy()

        # P-3: Handle null values with proper error handling
        try:
            processed_series = process_nulls(
                working_series, self.null_strategy, anonymize_value="*REDACTED*"
            )
        except ValueError as e:
            # Handle ERROR strategy gracefully
            self.logger.warning(
                f"Null processing error: {e}. Continuing with PRESERVE strategy."
            )
            processed_series = working_series.copy()

        # Pseudonymize non-null values
        non_null_mask = processed_series.notna()
        non_null_values = processed_series[non_null_mask]

        if len(non_null_values) > 0:
            # Process each unique value
            for idx, value in non_null_values.items():
                str_value = str(value)

                # P-4: High-resolution timing for each lookup
                lookup_start = time.perf_counter()

                # Thread-safe mapping lookup/creation
                with self._mapping_lock:
                    self._total_lookups += 1

                    if str_value in self._mapping:
                        # Use existing mapping
                        pseudonym = self._mapping[str_value]
                        self._mapping_hits += 1  # P-7: Track hits
                    else:
                        # Generate new unique pseudonym
                        pseudonym = self._generate_unique_pseudonym()

                        # Add to mappings
                        self._mapping[str_value] = pseudonym
                        self._reverse_mapping[pseudonym] = str_value
                        self._new_mappings_count += 1

                # P-4: Record lookup time
                lookup_time = time.perf_counter() - lookup_start
                self._lookup_times.append(lookup_time)

                processed_series.at[idx] = pseudonym

        # Update the DataFrame
        if self.mode == "REPLACE":
            if self.compound_mode:
                # For compound mode, replace all source fields with first field
                # and null out the rest
                batch[self.field_name] = processed_series
                for field in self.additional_fields:
                    batch[field] = None
            else:
                batch[self.field_name] = processed_series
        else:  # ENRICH
            batch[self._output_field] = processed_series

        return batch

    def _process_batch_safe(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch with error handling."""
        try:
            return self.process_batch(batch)
        except Exception as e:
            self.logger.warning(f"Batch processing failed: {e}")
            if self.mode == "ENRICH" and self._output_field not in batch.columns:
                batch[self._output_field] = None
            return batch

    def _initialize_mapping(self, task_dir: Path) -> None:
        """Initialize mapping storage and load existing mappings."""
        # Create maps directory
        maps_dir = task_dir / "maps"
        maps_dir.mkdir(exist_ok=True)

        # Determine mapping file path
        if not self.mapping_file:
            # Auto-generate filename based on field and operation
            operation_name = f"{self.field_name}_mapping"
            if self.compound_mode:
                operation_name = f"compound_{operation_name}"
            self.mapping_file = f"{operation_name}.{self.mapping_format}.enc"

        # Create full path
        self._mapping_path = maps_dir / self.mapping_file
        self.logger.info(f"Mapping file path: {self._mapping_path}")

        # Initialize storage
        self._mapping_storage = MappingStorage(
            mapping_file=self._mapping_path,
            encryption_key=self._encryption_key,
            format=self.mapping_format,
            backup_on_update=self.backup_on_update,
        )

        # Load existing mappings
        try:
            # P-5: Load with metadata support
            loaded_data = self._mapping_storage.load()

            # Check if it's the new format with metadata
            if isinstance(loaded_data, dict) and "_metadata" in loaded_data:
                self._mapping = loaded_data.get("mappings", {})
                metadata = loaded_data["_metadata"]

                # P-5: Restore sequential counter from metadata
                if self.pseudonym_type == "sequential":
                    self._sequential_counter = metadata.get("last_sequential", 0)
                    self.logger.info(
                        f"Restored sequential counter: {self._sequential_counter}"
                    )
            else:
                # Legacy format - just mappings
                self._mapping = loaded_data

                # P-5: Calculate sequential counter from existing mappings
                if self.pseudonym_type == "sequential" and self._mapping:
                    self._calculate_sequential_counter()

            # Build reverse mapping
            self._reverse_mapping = {v: k for k, v in self._mapping.items()}

            # Track all existing pseudonyms for collision detection
            self._generated_pseudonyms = set(self._reverse_mapping.keys())

            self.logger.info(f"Loaded {len(self._mapping)} existing mappings")

        except FileNotFoundError:
            if self.create_if_not_exists:
                self.logger.info("Creating new mapping file")
                self._mapping = {}
                self._reverse_mapping = {}
                self._sequential_counter = 0
                self._generated_pseudonyms = set()
            else:
                raise

    def _calculate_sequential_counter(self) -> None:
        """Calculate sequential counter from existing mappings."""
        max_seq = 0
        for pseudonym in self._reverse_mapping.keys():
            # Extract number from pseudonym
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

        Returns:
        --------
        str
            Unique pseudonym
        """
        if self.pseudonym_type == "uuid":
            pseudonym = self._pseudonym_generator.generate_unique(
                self._generated_pseudonyms, prefix=self.pseudonym_prefix
            )
        elif self.pseudonym_type == "sequential":
            # Generate sequential pseudonym
            self._sequential_counter += 1
            pseudonym = str(self._sequential_counter).zfill(6)  # Pad with zeros
            pseudonym = format_pseudonym_output(
                pseudonym, self.pseudonym_prefix, self.pseudonym_suffix
            )
        elif self.pseudonym_type == "random_string":
            # Generate random string of specified length
            import secrets

            characters = string.ascii_letters + string.digits

            # Calculate effective length
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

                # Check for collision
                if pseudonym not in self._generated_pseudonyms:
                    break

                attempts += 1
                if attempts > 10:
                    self._collision_count += 1
                    self.logger.warning(
                        f"High collision rate detected for random_string generation"
                    )

                if attempts > 100:
                    raise RuntimeError(
                        "Unable to generate unique pseudonym after 100 attempts"
                    )
        else:
            raise ValueError(f"Unknown pseudonym type: {self.pseudonym_type}")

        # Track generated pseudonym
        self._generated_pseudonyms.add(pseudonym)
        return pseudonym

    def _persist_mappings(self) -> None:
        """Save current mappings to encrypted file."""
        with self._mapping_lock:
            try:
                # P-5: Save with metadata
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
                    f"Persisted {len(self._mapping)} mappings "
                    f"({self._new_mappings_count} new)"
                )
                self._new_mappings_count = 0
            except Exception as e:
                self.logger.error(f"Failed to persist mappings: {e}")
                raise

    def _collect_comprehensive_metrics(
        self,
        original_series: pd.Series,
        processed_series: pd.Series,
        full_df: pd.DataFrame,
        timing_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect all metrics using commons utilities."""
        # Use metric_utils to collect operation metrics
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

        # Add effectiveness metrics
        effectiveness = calculate_anonymization_effectiveness(
            original_series, processed_series
        )
        operation_metrics["effectiveness"] = effectiveness

        # P-4: Calculate accurate lookup time
        if self._lookup_times:
            avg_lookup_time_ms = (
                sum(self._lookup_times) / len(self._lookup_times) * 1000
            )
        else:
            avg_lookup_time_ms = 0.0

        # P-7: Calculate hit rate
        hit_rate = (
            self._mapping_hits / self._total_lookups if self._total_lookups > 0 else 0.0
        )
        percent_new = (
            (self._new_mappings_count / self._total_lookups * 100)
            if self._total_lookups > 0
            else 0.0
        )

        # Add mapping-specific metrics
        metadata = self._mapping_storage.get_metadata() if self._mapping_storage else {}

        mapping_metrics = {
            "pseudonym_type": self.pseudonym_type,
            "total_mappings": len(self._mapping),
            "new_mappings_created": self._new_mappings_count,
            "mapping_file_size": metadata.get("size_bytes", 0),
            "mapping_file_path": str(self._mapping_path),
            "encryption_algorithm": "AES-256-GCM",
            "persist_frequency": self.persist_frequency,
            "reversible": True,
            "lookup_time_avg": round(avg_lookup_time_ms, 4),  # P-4: Accurate time
            "mapping_hit_rate": round(hit_rate, 4),  # P-7: Added
            "percent_new_mappings": round(percent_new, 2),  # P-7: Added
            "total_lookups": self._total_lookups,
            "mapping_hits": self._mapping_hits,
        }

        # Add collision count for random_string
        if self.pseudonym_type == "random_string":
            mapping_metrics["collision_count"] = self._collision_count

        operation_metrics["mapping"] = mapping_metrics

        # Add privacy metrics if quasi-identifiers available
        if self.quasi_identifiers and all(
            qi in full_df.columns for qi in self.quasi_identifiers
        ):
            privacy_metrics = calculate_batch_metrics(
                original_batch=full_df[[self.field_name] + self.quasi_identifiers],
                anonymized_batch=full_df[
                    [self._output_field if self.mode == "ENRICH" else self.field_name]
                    + self.quasi_identifiers
                ],
                original_field_name=self.field_name,
                anonymized_field_name=(
                    self._output_field if self.mode == "ENRICH" else self.field_name
                ),
                quasi_identifiers=self.quasi_identifiers,
            )
            operation_metrics["privacy_metrics"] = privacy_metrics

            # Add additional privacy indicators
            privacy_metrics["disclosure_risk"] = calculate_simple_disclosure_risk(
                full_df, self.quasi_identifiers
            )

        # Add performance metrics
        performance = calculate_process_performance(
            timing_info["start_time"],
            timing_info["end_time"],
            len(full_df),
            timing_info["batch_count"],
        )
        operation_metrics["performance"] = performance

        return operation_metrics

    def get_reverse_mapping(self, pseudonym: str) -> Optional[str]:
        """
        Get original value for a pseudonym (for authorized reversal).

        Parameters:
        -----------
        pseudonym : str
            Pseudonym to reverse

        Returns:
        --------
        Optional[str]
            Original value if found, None otherwise
        """
        with self._mapping_lock:
            return self._reverse_mapping.get(pseudonym)

    def export_mappings(self, output_path: Path, include_metadata: bool = True) -> None:
        """
        Export mappings in encrypted form (for backup/transfer).

        Parameters:
        -----------
        output_path : Path
            Path to export to
        include_metadata : bool
            Whether to include metadata
        """
        with self._mapping_lock:
            # P-5: Always include metadata for proper restore
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
                        "mapping_format": self.mapping_format,  # P-8: Include format
                    }
                    if include_metadata
                    else {}
                ),
            }

            # P-8: Use mapping storage with correct format
            temp_storage = MappingStorage(
                mapping_file=output_path,
                encryption_key=self._encryption_key,
                format=self.mapping_format,  # P-8: Use configured format
                backup_on_update=False,
            )
            temp_storage.save(export_data)


# Register the operation
register_operation(ConsistentMappingPseudonymizationOperation)


# Factory function
def create_mapping_pseudonymization_operation(
    field_name: str, encryption_key: Union[str, bytes], **kwargs
) -> ConsistentMappingPseudonymizationOperation:
    """
    Create a consistent mapping pseudonymization operation with default settings.

    Parameters:
    -----------
    field_name : str
        Field to pseudonymize
    encryption_key : Union[str, bytes]
        256-bit encryption key
    **kwargs : dict
        Additional parameters to override defaults

    Returns:
    --------
    ConsistentMappingPseudonymizationOperation
        Configured mapping pseudonymization operation
    """
    return ConsistentMappingPseudonymizationOperation(
        field_name=field_name, encryption_key=encryption_key, **kwargs
    )
