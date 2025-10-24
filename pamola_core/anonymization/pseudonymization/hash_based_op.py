"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Hash-based Pseudonymization Operation
Package:       pamola_core.anonymization.pseudonymization
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-10
Updated:       2025-06-15
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
    - Full integration with PAMOLA framework

Security Considerations:
    - Uses SHA3 family for quantum resistance
    - Salt prevents rainbow table attacks
    - Pepper provides additional session-specific security
    - Secure memory handling for sensitive data
    - No storage of original-to-pseudonym mappings (irreversible)

Dependencies:
    - pamola_core.utils.crypto_helpers.pseudonymization: Core crypto functions
    - pamola_core.anonymization.commons.pseudonymization_utils: Shared utilities
    - hashlib: For SHA3 algorithms
    - base64: For base64 encoding
    - uuid: For UUID-style formatting

Changelog:
    1.0.0 (2025-01-20):
        - Initial implementation with full framework integration
        - Support for SHA3-256/512 algorithms
        - Batch processing with caching
        - Risk-based processing
        - Comprehensive metrics and visualization
    1.0.1 (2025-06-15):
        - Fixed import issues with validation framework
        - Updated to use validation_utils facade
        - Improved error handling and logging
"""

import base64
import base58
import hashlib
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Base anonymization operation import
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Data utilities imports
from pamola_core.anonymization.commons.data_utils import process_nulls

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

# Validation imports - FIXED: Using validation_utils facade
from pamola_core.anonymization.commons.validation_utils import (
    check_field_exists,
    create_field_validator,
)

# Visualization utilities imports
from pamola_core.anonymization.commons.visualization_utils import (
    create_comparison_visualization,
    create_metric_visualization,
)

# Crypto helpers imports
from pamola_core.utils.crypto_helpers.pseudonymization import HashGenerator

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
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker

# Configure module logger
logger = logging.getLogger(__name__)


class HashBasedPseudonymizationConfig(OperationConfig):
    """Configuration for HashBasedPseudonymizationOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "additional_fields": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "algorithm": {"type": "string", "enum": ["sha3_256", "sha3_512"]},
            "salt_config": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": ["parameter", "file"]},
                    "value": {"type": ["string", "null"]},
                    "field_name": {"type": ["string", "null"]},
                },
                "required": ["source"],
            },
            "use_pepper": {"type": "boolean"},
            "pepper_length": {"type": "integer", "minimum": 16, "default": 32},
            "output_format": {
                "type": "string",
                "enum": ["hex", "base64", "base32", "base58", "uuid"],
            },
            "output_length": {"type": ["integer", "null"], "minimum": 8},
            "prefix": {"type": ["string", "null"]},
            "suffix": {"type": ["string", "null"]},
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "output_field_name": {"type": ["string", "null"]},
            "column_prefix": {"type": "string"},
            "null_strategy": {
                "type": "string",
                "enum": ["PRESERVE", "EXCLUDE", "ANONYMIZE", "ERROR"],
            },
            "batch_size": {"type": "integer", "minimum": 1},
            "use_cache": {"type": "boolean"},
            "cache_size": {"type": "integer", "minimum": 1000},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]},
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
        "required": ["field_name", "algorithm", "salt_config"],
    }


@register(version="1.0.0")
class HashBasedPseudonymizationOperation(AnonymizationOperation):
    """
    Hash-based pseudonymization operation for irreversible data transformation.

    This operation applies cryptographic hash functions to transform sensitive
    identifiers into pseudonyms that cannot be reversed without the original
    salt and pepper values.
    """

    def __init__(
        self,
        field_name: str,
        additional_fields: Optional[List[str]] = None,
        algorithm: str = "sha3_256",
        salt_config: Optional[Dict[str, Any]] = None,
        salt_file: Optional[Path] = None,
        use_pepper: bool = True,
        pepper_length: int = 32,
        output_format: str = "hex",
        output_length: Optional[int] = None,
        output_prefix: Optional[str] = None,
        output_suffix: Optional[str] = None,
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "_",
        null_strategy: str = "PRESERVE",
        batch_size: int = 10000,
        use_cache: bool = True,
        cache_size: int = 100000,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        condition_field: Optional[str] = None,
        condition_values: Optional[List] = None,
        condition_operator: str = "in",
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
        Initialize hash-based pseudonymization operation.

        Parameters:
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
        output_format : str
            Output format: "hex", "base64", "base32", "base58", or "uuid" (default: "hex")
        output_length : Optional[int]
            Truncate output to specified length (default: None)
        output_prefix : Optional[str]
            Prefix for pseudonyms (default: None)
        output_suffix : Optional[str]
            Suffix for pseudonyms (default: None)
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
            Whether to cache pseudonyms (default: True)
        cache_size : int
            Maximum cache size (default: 100000)
        use_encryption : bool
            Whether to encrypt output files (default: False)
        encryption_key : Optional[Union[str, Path]]
            Encryption key for output files
        condition_field : Optional[str]
            Field for conditional processing
        condition_values : Optional[List]
            Values for conditional processing
        condition_operator : str
            Operator for conditions (default: "in")
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
        # Validate parameters
        if output_length is not None and output_length < 8:
            raise ValueError("output_length must be at least 8 characters")

        if compound_mode and not additional_fields:
            raise ValueError("compound_mode requires additional_fields to be specified")

        # Default salt config if not provided
        if salt_config is None:
            salt_config = {
                "source": "parameter",
                "value": "0" * 64,  # 32-byte default salt as hex
            }

        # Ensure additional_fields is always a list
        if additional_fields is None:
            additional_fields = []

        # Build config parameters
        config_params = {
            "field_name": field_name,
            "additional_fields": additional_fields,
            "algorithm": algorithm,
            "salt_config": salt_config,
            "use_pepper": use_pepper,
            "pepper_length": pepper_length,
            "output_format": output_format,
            "output_length": output_length,
            "output_prefix": output_prefix,
            "output_suffix": output_suffix,
            "mode": mode,
            "output_field_name": output_field_name,
            "column_prefix": column_prefix,
            "null_strategy": null_strategy,
            "batch_size": batch_size,
            "use_cache": use_cache,
            "cache_size": cache_size,
            "use_encryption": use_encryption,
            "encryption_key": encryption_key,
            "condition_field": condition_field,
            "condition_values": condition_values,
            "condition_operator": condition_operator,
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
        config = HashBasedPseudonymizationConfig(**config_params)

        # Use default description if none provided
        if not description:
            description = f"Hash-based pseudonymization for field '{field_name}' using {algorithm}"

        # Initialize base class
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            column_prefix=column_prefix,
            null_strategy=null_strategy,
            batch_size=batch_size,
            use_cache=use_cache,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator,
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            vulnerable_record_strategy=vulnerable_record_strategy,
            description=description,
        )

        # Store operation-specific parameters
        self.additional_fields = additional_fields
        self.algorithm = algorithm.lower()
        self.salt_config = salt_config
        self.salt_file = Path(salt_file) if salt_file else None
        self.use_pepper = use_pepper
        self.pepper_length = pepper_length
        self.output_format = output_format.lower()
        self.output_length = output_length
        self.output_prefix = output_prefix
        self.output_suffix = output_suffix
        self.output_file_format = output_file_format
        self.quasi_identifiers = quasi_identifiers or []
        self.compound_mode = compound_mode
        self.compound_separator = compound_separator
        self.compound_null_handling = compound_null_handling

        # Version information
        self.version = "1.0.1"

        # Initialize internal state
        self._cache = None
        self._salt = None
        self._pepper = None
        self._hash_generator = None
        self._output_field = None
        self._processed_count = 0
        self._cache_hits = 0
        self._collision_count = 0
        self._collision_probability = 0.0
        self._collision_tracker = {}  # Track actual collisions
        self._hash_computation_time = 0.0  # Total time spent hashing

        # Initialize cache if enabled
        if self.use_cache:
            self._cache = PseudonymizationCache(max_size=cache_size)

    def execute(
        self, data_source, task_dir, reporter=None, progress_tracker=None, **kwargs
    ):
        """
        Execute the hash-based pseudonymization operation.

        This method overrides the base class execute to handle salt/pepper
        initialization and proper cleanup of secure memory.
        """
        start_time = time.time()

        # Create progress tracker if not provided
        if progress_tracker is None:
            progress_tracker = HierarchicalProgressTracker(
                total=100,
                description=f"Hash-based pseudonymization for {self.field_name}",
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
            # Update progress: Starting
            progress_tracker.update(
                5,
                {
                    "status": "initializing",
                    "phase": "startup",
                    "message": f"Starting hash-based pseudonymization for field '{self.field_name}'",
                },
            )

            # Initialize salt and pepper
            self._initialize_crypto_components()
            progress_tracker.update(5, {"status": "crypto_initialized"})

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

                # Additional validation using field validator
                # Create appropriate validator based on expected field type
                try:
                    # For pseudonymization, we typically expect text fields
                    validator = create_field_validator("text")
                    result = validator.validate(df[field], field_name=field)
                    if not result.is_valid:
                        self.logger.warning(
                            f"Field '{field}' validation warnings: {result.warnings}"
                        )
                except Exception as e:
                    self.logger.warning(f"Could not validate field '{field}': {e}")

            # Determine output field name
            self._output_field = generate_output_field_name(
                self.field_name,
                self.mode,
                self.output_field_name,
                operation_suffix="pseudonymized",
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

            # Update progress: Data loaded
            progress_tracker.update(
                10,
                {
                    "status": "data_loaded",
                    "phase": "data_preparation",
                    "records": len(df),
                    "fields": len(all_fields),
                    "compound_mode": self.compound_mode,
                    "message": f"Loaded {len(df):,} records",
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
                    10,
                    {
                        "status": "risk_analysis_complete",
                        "vulnerable_records": vulnerable_count,
                        "message": f"Risk analysis complete: {vulnerable_count:,} vulnerable records",
                    },
                )

            # Process in batches
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
                    },
                )

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

            # Update progress: Processing complete
            progress_tracker.update(
                60,
                {
                    "status": "processing_complete",
                    "phase": "post_processing",
                    "processed_count": self._processed_count,
                    "cache_hits": self._cache_hits if self.use_cache else 0,
                },
            )

            # Calculate collision probability
            self._collision_probability = estimate_collision_probability(
                self._processed_count, 256 if self.algorithm == "sha3_256" else 512
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
                name=f"{self.field_name}_hash_pseudonymization",
                timestamp_in_name=True,
            )

            # Log summary
            summary = get_process_summary(operation_metrics.get("privacy_metrics", {}))
            for key, message in summary.items():
                self.logger.info(f"{key}: {message}")

            # Update progress: Metrics saved
            progress_tracker.update(70, {"status": "metrics_saved"})

            # Generate visualizations
            viz_progress = progress_tracker.create_subtask(
                total=2, description="Generating visualizations", unit="charts"
            )

            # Uniqueness comparison
            comparison_viz = create_comparison_visualization(
                original_series,
                processed_series,
                task_dir,
                self.field_name,
                "hash_pseudonymization",
                None,
            )
            if comparison_viz:
                viz_progress.update(1)

            # Cache performance visualization if using cache
            cache_viz = None
            if self.use_cache and self._cache:
                cache_stats = self._cache.get_statistics()
                if cache_stats["total_requests"] > 0:
                    cache_viz = create_metric_visualization(
                        "cache_performance",
                        {
                            "Hits": cache_stats["hits"],
                            "Misses": cache_stats["misses"],
                            "Hit Rate": cache_stats["hit_rate"] * 100,
                        },
                        task_dir,
                        self.field_name,
                        "hash_pseudonymization",
                        None,
                    )
                    viz_progress.update(1)

            # Update progress: Visualizations complete
            progress_tracker.update(80, {"status": "visualizations_complete"})

            # Write output if requested
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
                progress_tracker.update(95, {"status": "output_written"})

            # Create operation result
            result = OperationResult(
                status=OperationStatus.SUCCESS, execution_time=time.time() - start_time
            )

            # Add metrics
            result.add_metric("records_processed", self._processed_count)
            result.add_metric("algorithm", self.algorithm)
            result.add_metric("output_format", self.output_format)
            result.add_metric("collision_probability", self._collision_probability)
            result.add_metric("vulnerable_records", int(vulnerable_count))

            if self.use_cache and self._cache:
                cache_stats = self._cache.get_statistics()
                result.add_nested_metric("cache", "hit_rate", cache_stats["hit_rate"])
                result.add_nested_metric("cache", "size", cache_stats["size"])
                result.add_nested_metric(
                    "cache", "cache_hit_rate", cache_stats["hit_rate"]
                )  # Add required metric name

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
                tags=["metrics", "process", "hash_pseudonymization"],
            )

            if comparison_viz:
                result.add_artifact(
                    artifact_type="png",
                    path=comparison_viz,
                    description="Before/after comparison",
                    category="visualization",
                    tags=["visualization", "comparison", self.field_name],
                )

            if cache_viz:
                result.add_artifact(
                    artifact_type="png",
                    path=cache_viz,
                    description="Cache performance",
                    category="visualization",
                    tags=["visualization", "cache", "performance"],
                )

            if output_path:
                result.add_artifact(
                    artifact_type=self.output_file_format,
                    path=output_path,
                    description="Pseudonymized dataset",
                    category="output",
                    tags=["data", "pseudonymized", self.algorithm],
                )

            # Final progress update
            progress_tracker.update(100, {"status": "complete"})

            return result

        except Exception as e:
            self.logger.error(f"Error in hash pseudonymization: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )
        finally:
            # Clean up secure memory
            self._cleanup_crypto_components()
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

        # Handle null values
        processed_series = process_nulls(
            working_series, self.null_strategy, anonymize_value="*REDACTED*"
        )

        # Pseudonymize non-null values
        non_null_mask = processed_series.notna()
        non_null_values = processed_series[non_null_mask]

        if len(non_null_values) > 0:
            # Process each unique value
            unique_values = non_null_values.unique()
            pseudonym_map = {}

            for value in unique_values:
                # Check cache first
                if self.use_cache and self._cache:
                    cached = self._cache.get(str(value))
                    if cached:
                        pseudonym_map[value] = cached
                        self._cache_hits += 1
                        continue

                # Generate new pseudonym
                start_hash_time = time.time()
                pseudonym = self._generate_pseudonym(str(value))
                self._hash_computation_time += time.time() - start_hash_time

                # Check for collisions
                if pseudonym in self._collision_tracker:
                    if self._collision_tracker[pseudonym] != str(value):
                        self._collision_count += 1
                        self.logger.warning(
                            f"Hash collision detected: '{value}' and '{self._collision_tracker[pseudonym]}' "
                            f"both map to '{pseudonym}'"
                        )
                else:
                    self._collision_tracker[pseudonym] = str(value)

                pseudonym_map[value] = pseudonym

                # Cache the result
                if self.use_cache and self._cache:
                    self._cache.put(str(value), pseudonym)

                self._processed_count += 1

            # Apply pseudonyms
            processed_series[non_null_mask] = non_null_values.map(pseudonym_map)

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

    def _initialize_crypto_components(self):
        """Initialize salt and pepper for cryptographic operations."""
        # Load salt
        self._salt = load_salt_configuration(self.salt_config, self.salt_file)
        self.logger.info(f"Loaded {len(self._salt)}-byte salt")

        # Initialize hash generator
        self._hash_generator = HashGenerator(algorithm=self.algorithm)

        # Generate session pepper if enabled
        if self.use_pepper:
            self._pepper = generate_session_pepper(self.pepper_length)
            self.logger.info(f"Generated {self.pepper_length}-byte session pepper")

    def _cleanup_crypto_components(self):
        """Clean up secure memory used for salt and pepper."""
        if self._pepper:
            self._pepper.clear()
            self._pepper = None

        # Clear salt from memory (not SecureBytes, so manual clear)
        if self._salt:
            self._salt = None

        self.logger.debug("Cleared cryptographic components from memory")

    def _generate_pseudonym(self, value: str) -> str:
        """
        Generate a pseudonym for a single value.

        Parameters:
        -----------
        value : str
            Value to pseudonymize

        Returns:
        --------
        str
            Generated pseudonym
        """
        # Apply hash function with salt and pepper
        if self.use_pepper and self._pepper:
            hash_bytes = self._hash_generator.hash_with_salt_and_pepper(
                value.encode("utf-8"), self._salt, self._pepper.get()
            )
        else:
            hash_bytes = self._hash_generator.hash_with_salt(
                value.encode("utf-8"), self._salt
            )

        # Format output
        if self.output_format == "hex":
            pseudonym = hash_bytes.hex()
        elif self.output_format == "base64":
            pseudonym = base64.urlsafe_b64encode(hash_bytes).decode("utf-8").rstrip("=")
        elif self.output_format == "base32":
            pseudonym = base64.b32encode(hash_bytes).decode("utf-8").rstrip("=")
        elif self.output_format == "base58":
            # Import base58 if available
            try:

                pseudonym = base58.b58encode(hash_bytes).decode("utf-8")
            except ImportError:
                # Fallback to hex
                self.logger.warning("base58 not available, falling back to hex")
                pseudonym = hash_bytes.hex()
        elif self.output_format == "uuid":
            # Use first 16 bytes for UUID format
            uuid_bytes = hash_bytes[:16]
            pseudonym = str(uuid.UUID(bytes=uuid_bytes))
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")

        # Apply length limit if specified
        if self.output_length and len(pseudonym) > self.output_length:
            pseudonym = pseudonym[: self.output_length]

        # Apply prefix/suffix
        pseudonym = format_pseudonym_output(
            pseudonym, self.output_prefix, self.output_suffix
        )

        return pseudonym

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
                "algorithm": self.algorithm,
                "output_format": self.output_format,
                "compound_mode": self.compound_mode,
                "use_pepper": self.use_pepper,
            },
            timing_info=timing_info,
        )

        # Add effectiveness metrics
        effectiveness = calculate_anonymization_effectiveness(
            original_series, processed_series
        )
        operation_metrics["effectiveness"] = effectiveness

        # Add pseudonymization-specific metrics
        pseudo_metrics = {
            "algorithm": self.algorithm,
            "output_format": self.output_format,
            "collision_probability": self._collision_probability,
            "collision_count": self._collision_count,
            "estimated_collision_count": int(
                self._collision_probability * self._processed_count
            ),
            "unique_pseudonyms": processed_series.nunique(),
            "pseudonymization_rate": 1.0 if self._processed_count > 0 else 0.0,
            "values_pseudonymized": self._processed_count,
            "salt_source": self.salt_config.get("source", "unknown"),
            "hash_computation_time": round(self._hash_computation_time, 4),
        }
        operation_metrics["pseudonymization"] = pseudo_metrics

        # Add cache metrics if using cache
        if self.use_cache and self._cache:
            cache_stats = self._cache.get_statistics()
            operation_metrics["cache"] = cache_stats

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
            privacy_metrics["suppression_rate"] = calculate_suppression_rate(
                processed_series, original_series.isna().sum()
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

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """Get operation-specific parameters for cache key generation."""
        params = {
            "algorithm": self.algorithm,
            "output_format": self.output_format,
            "output_length": self.output_length or 0,
            "output_prefix": self.output_prefix or "",
            "output_suffix": self.output_suffix or "",
            "compound_mode": self.compound_mode,
            "compound_fields": (
                tuple(sorted(self.additional_fields)) if self.compound_mode else ()
            ),
            "version": self.version,
            "salt_source": self.salt_config.get("source", ""),
        }

        # Add salt configuration (but not the actual salt value for security)
        if self.salt_config.get("source") == "file":
            params["salt_field"] = self.salt_config.get("field_name", "")

        # Add salt digest for cache consistency
        if self._salt:
            params["salt_digest"] = hashlib.sha256(self._salt).hexdigest()[:16]

        params["use_pepper"] = bool(self.use_pepper)

        return params


# Factory function
def create_hash_pseudonymization_operation(
    field_name: str, **kwargs
) -> HashBasedPseudonymizationOperation:
    """
    Create a hash-based pseudonymization operation with default settings.

    Parameters:
    -----------
    field_name : str
        Field to pseudonymize
    **kwargs : dict
        Additional parameters to override defaults

    Returns:
    --------
    HashBasedPseudonymizationOperation
        Configured hash pseudonymization operation
    """
    return HashBasedPseudonymizationOperation(field_name=field_name, **kwargs)
