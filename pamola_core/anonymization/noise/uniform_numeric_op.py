"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Numeric Noise Operation
Package:       pamola_core.anonymization.noise
Version:       1.0.0
Status:        development
Author:        PAMOLA Core Team
Created:       2025-01-20
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
   This module implements the UniformNumericNoiseOperation class for adding
   uniformly distributed random noise to numeric fields. It supports both
   additive and multiplicative noise, with configurable bounds and constraints.

   The operation integrates with the PAMOLA.CORE framework, providing secure
   random generation, comprehensive metrics collection, and support for both
   pandas and Dask processing engines.

Key Features:
   - Uniform noise addition with configurable range (symmetric/asymmetric)
   - Additive and multiplicative noise types
   - Output bounds enforcement (min/max constraints)
   - Zero value preservation option
   - Automatic integer type preservation
   - Statistical scaling based on field standard deviation
   - Cryptographically secure or reproducible random generation
   - Full Dask support for distributed processing
   - Comprehensive noise impact metrics
   - Integration with framework utilities

Framework Integration:
   - Inherits from AnonymizationOperation base class
   - Uses SecureRandomGenerator from noise_utils
   - Integrates with validation_utils for parameter validation
   - Uses metric_utils and statistical_utils for metrics
   - Supports DataWriter for output and metrics
   - Compatible with ProgressTracker for monitoring

Changelog:
   1.0.0 (2025-06-15):
      - Initial implementation following noise SRS specifications
      - Full integration with commons utilities
      - Dask support for large-scale processing
      - Comprehensive metrics and validation
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import base class
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Import noise utilities
from pamola_core.anonymization.commons.noise_utils import (
    SecureRandomGenerator,
    calculate_noise_impact,
    calculate_distribution_preservation,
    analyze_noise_effectiveness,
)

# Import statistical utilities
from pamola_core.anonymization.commons.statistical_utils import (
    calculate_utility_metrics,
    analyze_noise_uniformity,
)

# Import validation utilities
from pamola_core.anonymization.commons.validation import (
    InvalidParameterError,
)

# Import framework utilities
from pamola_core.anonymization.commons.validation_utils import validate_numeric_field
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_registry import register

# Constants
EPSILON = 1e-10


class UniformNumericNoiseConfig(OperationConfig):
    """Configuration schema for UniformNumericNoiseOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {
                "type": "string",
            },
            "noise_range": {
                "oneOf": [
                    {"type": "number"},
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                ]
            },
            "noise_type": {"type": "string", "enum": ["additive", "multiplicative"]},
            "output_min": {"type": ["number", "null"]},
            "output_max": {"type": ["number", "null"]},
            "preserve_zero": {"type": "boolean"},
            "round_to_integer": {"type": ["boolean", "null"]},
            "scale_by_std": {
                "type": "boolean",
            },
            "scale_factor": {"type": "number", "minimum": 0},
            "random_seed": {"type": ["integer", "null"]},
            "use_secure_random": {
                "type": "boolean",
            },
            # Conditional processing parameters
            "condition_field": {"type": ["string", "null"]},
            "condition_values": {"type": ["array", "null"]},
            "condition_operator": {"type": "string"},
            # Multi-field conditions
            "multi_conditions": {"type": ["array", "null"]},
            "condition_logic": {"type": "string"},
            # K-anonymity integration
            "ka_risk_field": {"type": ["string", "null"]},
            "risk_threshold": {"type": "number"},
            "vulnerable_record_strategy": {"type": "string"},
            # Memory optimization
            "optimize_memory": {"type": "boolean"},
            "adaptive_chunk_size": {"type": "boolean"},
            # Standard parameters
            "description": {"type": "string"},
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "column_prefix": {"type": "string"},
            "output_field_name": {"type": ["string", "null"]},
            "null_strategy": {
                "type": "string",
                "enum": ["PRESERVE", "EXCLUDE", "ANONYMIZE", "ERROR"],
            },
            "chunk_size": {"type": "integer"},
            "use_dask": {"type": "boolean"},
            "npartitions": {"type": ["integer", "null"]},
            "dask_partition_size": {"type": ["string", "null"]},
            "use_vectorization": {"type": "boolean"},
            "parallel_processes": {"type": ["integer", "null"]},
            "use_cache": {"type": "boolean"},
            "output_format": {
                "type": "string",
                "enum": ["csv", "parquet", "arrow"],
                "default": "csv",
            },
            "visualization_theme": {"type": ["string", "null"]},
            "visualization_backend": {"type": ["string", "null"]},
            "visualization_strict": {"type": "boolean"},
            "visualization_timeout": {"type": "integer"},
            "use_encryption": {"type": "boolean"},
            "encryption_mode": {"type": ["string", "null"]},
            "encryption_key": {"type": ["string", "null"]},
        },
        "required": ["field_name", "noise_range"],
    }


@register(version="1.0.0")
class UniformNumericNoiseOperation(AnonymizationOperation):
    """
    Operation for adding uniform random noise to numeric fields.

    This operation implements REQ-UNIFORM-001 through REQ-UNIFORM-007 from the
    PAMOLA.CORE Noise Operations Sub-Specification.

    The operation adds uniformly distributed random noise within specified bounds,
    supporting both additive and multiplicative noise types with various constraints
    for maintaining data utility while providing privacy protection.

    Attributes:
        noise_range: Symmetric float or (min, max) tuple for noise range
        noise_type: Type of noise application ('additive' or 'multiplicative')
        output_min: Minimum allowed output value
        output_max: Maximum allowed output value
        preserve_zero: Whether to keep zero values unchanged
        round_to_integer: Whether to round results to integers
        scale_by_std: Whether to scale noise by field standard deviation
        scale_factor: Additional scaling factor for noise
        random_seed: Seed for reproducible noise (if not using secure random)
        use_secure_random: Whether to use cryptographically secure random
    """

    def __init__(
        self,
        field_name: str,
        # Noise parameters
        noise_range: Union[float, Tuple[float, float]],
        noise_type: str = "additive",
        # Bounds and constraints
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        preserve_zero: bool = False,
        # Integer handling
        round_to_integer: Optional[bool] = None,
        # Statistical parameters
        scale_by_std: bool = False,
        scale_factor: float = 1.0,
        # Reproducibility
        random_seed: Optional[int] = None,
        use_secure_random: bool = True,
        # Conditional processing parameters
        condition_field: Optional[str] = None,
        condition_values: Optional[List] = None,
        condition_operator: str = "in",
        # Multi-field conditions
        multi_conditions: Optional[List[Dict[str, Any]]] = None,
        condition_logic: str = "AND",
        # K-anonymity integration
        ka_risk_field: Optional[str] = None,
        risk_threshold: float = 5.0,
        vulnerable_record_strategy: str = "suppress",
        # Memory optimization
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,
        # Standard parameters
        description: str = "",
        mode: str = "REPLACE",
        column_prefix: str = "_",
        output_field_name: Optional[str] = None,
        null_strategy: str = "PRESERVE",
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        dask_partition_size: Optional[str] = None,
        use_vectorization: bool = False,
        parallel_processes: Optional[int] = None,
        use_cache: bool = False,
        output_format: str = "csv",
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        use_encryption: bool = False,
        encryption_mode: Optional[str] = None,
        encryption_key: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize the uniform numeric noise operation.

        Args:
            field_name: Field to add noise to
            noise_range: Symmetric range (float) or asymmetric range (tuple)
            noise_type: 'additive' or 'multiplicative' noise
            output_min: Minimum allowed output value
            output_max: Maximum allowed output value
            preserve_zero: Don't add noise to zero values
            round_to_integer: Round to integers (auto-detected if None)
            scale_by_std: Scale noise by field standard deviation
            scale_factor: Additional scaling factor
            random_seed: Seed for reproducibility (ignored if use_secure_random=True)
            use_secure_random: Use cryptographically secure random
            condition_field: Field name for conditional processing
            condition_values: Values to match for conditional processing
            condition_operator: Operator for condition evaluation
            multi_conditions: List of conditions for complex filtering
            condition_logic: Logic operator for conditions
            ka_risk_field: Field containing k-anonymity risk scores
            risk_threshold: Minimum risk score threshold
            vulnerable_record_strategy: Strategy for high-risk records
            optimize_memory: Enable memory optimization
            adaptive_chunk_size: Auto-adjust chunk size
            description: Operation description
            mode: 'REPLACE' or 'ENRICH' output mode
            column_prefix: Prefix for new columns in ENRICH mode
            output_field_name: Custom output field name
            null_strategy: Strategy for handling null values
            chunk_size: Records per batch
            use_dask: Enable Dask for distributed processing
            npartitions: Number of Dask partitions
            dask_partition_size: Target size per partition
            use_vectorization: Enable vectorized operations
            parallel_processes: Number of parallel processes
            use_cache: Enable result caching
            output_format: Output file format
            visualization_theme: Theme for visualizations
            visualization_backend: Visualization rendering backend
            visualization_strict: Strict visualization validation
            visualization_timeout: Visualization generation timeout
            use_encryption: Enable output encryption
            encryption_mode: Encryption algorithm
            encryption_key: Encryption key or key file path

            **kwargs: Additional parameters for base class
        """
        noise_range_for_config = (
            list(noise_range) if isinstance(noise_range, tuple) else noise_range
        )

        config = UniformNumericNoiseConfig(
            field_name=field_name,
            # Noise parameters
            noise_range=noise_range_for_config,
            noise_type=noise_type,
            # Bounds and constraints
            output_min=output_min,
            output_max=output_max,
            preserve_zero=preserve_zero,
            # Integer handling
            round_to_integer=round_to_integer,
            # Statistical parameters
            scale_by_std=scale_by_std,
            scale_factor=scale_factor,
            # Reproducibility
            random_seed=random_seed,
            use_secure_random=use_secure_random,
            # Conditional processing parameters
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator,
            # Multi-field conditions
            multi_conditions=multi_conditions,
            condition_logic=condition_logic,
            # K-anonymity integration
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            vulnerable_record_strategy=vulnerable_record_strategy,
            # Memory optimization
            optimize_memory=optimize_memory,
            adaptive_chunk_size=adaptive_chunk_size,
            # Standard parameters
            description=description,
            mode=mode,
            column_prefix=column_prefix,
            output_field_name=output_field_name,
            null_strategy=null_strategy,
            chunk_size=chunk_size,
            use_dask=use_dask,
            npartitions=npartitions,
            dask_partition_size=dask_partition_size,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            use_cache=use_cache,
            output_format=output_format,
            visualization_theme=visualization_theme,
            visualization_backend=visualization_backend,
            visualization_strict=visualization_strict,
            visualization_timeout=visualization_timeout,
            use_encryption=use_encryption,
            encryption_mode=encryption_mode,
            encryption_key=encryption_key,
        )

        # Initialize base class
        super().__init__(
            field_name=field_name,
            # Conditional processing parameters
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator,
            # Multi-field conditions
            multi_conditions=multi_conditions,
            condition_logic=condition_logic,
            # K-anonymity integration
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            vulnerable_record_strategy=vulnerable_record_strategy,
            # Memory optimization
            optimize_memory=optimize_memory,
            adaptive_chunk_size=adaptive_chunk_size,
            # Standard parameters
            description=f"Uniform noise addition for field '{field_name}'",
            mode=mode,
            column_prefix=column_prefix,
            output_field_name=output_field_name,
            null_strategy=null_strategy,
            chunk_size=chunk_size,
            use_dask=use_dask,
            npartitions=npartitions,
            dask_partition_size=dask_partition_size,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            use_cache=use_cache,
            output_format=output_format,
            visualization_theme=visualization_theme,
            visualization_backend=visualization_backend,
            visualization_strict=visualization_strict,
            visualization_timeout=visualization_timeout,
            use_encryption=use_encryption,
            encryption_mode=encryption_mode,
            encryption_key=encryption_key,
            **kwargs,
        )

        # Save config attributes to self
        for k, v in config._params.items():
            setattr(self, k, v)
            self.process_kwargs[k] = v

        self.config = config
        self.noise_type = noise_type.lower()

        # Validate and store noise parameters
        self._validate_noise_parameters(noise_range, noise_type, output_min, output_max)

        # Initialize generator (will be created per execution)
        self._generator: Optional[SecureRandomGenerator] = None
        self._scale_factor_calculated: Optional[float] = None

        # Assign process_kwargs
        self.process_kwargs["noise_type"] = self.noise_type
        self.process_kwargs["_generator"] = self._generator
        self.process_kwargs["_scale_factor_calculated"] = self._scale_factor_calculated

        # Version
        self.version = "1.0.0"

    def _validate_noise_parameters(
        self,
        noise_range: Union[float, Tuple[float, float]],
        noise_type: str,
        output_min: Optional[float],
        output_max: Optional[float],
    ) -> None:
        """
        Validate noise operation parameters.

        Args:
            noise_range: Noise range to validate
            noise_type: Noise type to validate
            output_min: Minimum bound to validate
            output_max: Maximum bound to validate

        Raises:
            InvalidParameterError: If parameters are invalid
        """
        # Validate noise type
        if noise_type.lower() not in ["additive", "multiplicative"]:
            raise InvalidParameterError(
                param_name="noise_type",
                param_value=noise_type,
                reason="must be 'additive' or 'multiplicative'",
            )

        # Validate noise range
        if isinstance(noise_range, (tuple, list)):
            if len(noise_range) != 2:
                raise InvalidParameterError(
                    param_name="noise_range",
                    param_value=noise_range,
                    reason="tuple must have exactly 2 elements (min, max)",
                )
            if noise_range[0] >= noise_range[1]:
                raise InvalidParameterError(
                    param_name="noise_range",
                    param_value=noise_range,
                    reason=f"invalid range: min ({noise_range[0]}) >= max ({noise_range[1]})",
                )
        elif not isinstance(noise_range, (int, float)):
            raise InvalidParameterError(
                param_name="noise_range",
                param_value=noise_range,
                reason="must be a number or tuple of numbers",
            )

        # Validate output bounds
        if output_min is not None and output_max is not None:
            if output_min >= output_max:
                raise InvalidParameterError(
                    param_name="output_min",
                    param_value=output_min,
                    reason=f"must be less than output_max ({output_max})",
                )

    @staticmethod
    def _generate_noise(size: int, **kwargs) -> np.ndarray:
        """
        Generate uniform noise values.

        Args:
            size: Number of noise values to generate
            **kwargs: Additional parameters (e.g., noise_range, scale_factor)

        Returns:
            Array of noise values
        """
        # Get parameters from kwargs
        noise_range = kwargs.get("noise_range", [])
        use_secure_random = kwargs.get("use_secure_random", True)
        random_seed = kwargs.get("random_seed", None)
        scale_factor = kwargs.get("scale_factor", 1.0)
        _generator = kwargs.get("_generator", None)
        _scale_factor_calculated = kwargs.get("_scale_factor_calculated", 0.0)

        if _generator is None:
            _generator = SecureRandomGenerator(
                use_secure=use_secure_random, seed=random_seed
            )

        # Determine noise bounds
        if isinstance(noise_range, (tuple, list)):
            min_noise, max_noise = noise_range
        else:
            min_noise, max_noise = -noise_range, noise_range

        # Generate uniform noise
        if _generator is not None:
            noise = _generator.uniform(min_noise, max_noise, size)
        else:
            # Fallback if generator fails to initialize
            noise = np.random.uniform(min_noise, max_noise, size)

        # Apply scale factor
        if _scale_factor_calculated is not None:
            noise *= _scale_factor_calculated
        else:
            noise *= scale_factor

        return np.asarray(noise)

    @staticmethod
    def _calculate_scale_factor(series: pd.Series, **kwargs) -> float:
        """
        Calculate noise scale factor based on data statistics.

        Args:
            series: Data series to analyze
            **kwargs: Additional parameters (e.g., scale_by_std, scale_factor)

        Returns:
            Calculated scale factor
        """
        # Get parameters from kwargs
        scale_by_std = kwargs.get("scale_by_std", False)
        scale_factor = kwargs.get("scale_factor", 1.0)

        if not scale_by_std:
            return scale_factor

        # Calculate standard deviation
        std = series.std()
        if std > 0:
            return scale_factor * std
        else:
            return scale_factor

    @staticmethod
    def _apply_noise(values: pd.Series, noise: np.ndarray, **kwargs) -> pd.Series:
        """
        Apply noise to values with constraints.

        Args:
            values: Original values
            noise: Noise values to apply
            **kwargs: Additional parameters (e.g., output_min, output_max,
                      round_to_integer, preserve_zero)

        Returns:
            Series with noise applied
        """
        # Get parameters from kwargs
        noise_type = kwargs.get("noise_type", "additive")
        output_min = kwargs.get("output_min", None)
        output_max = kwargs.get("output_max", None)
        preserve_zero = kwargs.get("preserve_zero", False)
        round_to_integer = kwargs.get("round_to_integer", False)

        if len(values) != len(noise):
            raise ValueError("Length of values and noise must match.")

        # Handle noise type
        if noise_type == "additive":
            noisy_values = values + noise
        elif noise_type == "multiplicative":
            noisy_values = values * (1 + noise)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Apply bounds if specified
        if output_min is not None:
            noisy_values = np.maximum(noisy_values, output_min)
        if output_max is not None:
            noisy_values = np.minimum(noisy_values, output_max)

        # Handle integer fields
        if round_to_integer:
            noisy_values = np.round(noisy_values).astype(values.dtype)

        result = pd.Series(noisy_values, index=values.index)

        # Preserve zeros if requested
        if preserve_zero:
            zero_mask = values == 0
            noisy_values[zero_mask] = 0

        return result

    @classmethod
    def process_batch(cls, batch: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process a batch of data by adding uniform noise.

        Parameters:
        -----------
            batch : pd.DataFrame
                DataFrame batch to process
            **kwargs : Any
                Additional keyword arguments for processing

        Returns:
            Processed DataFrame with noise added
        """
        # Extract parameters from kwargs
        field_name = kwargs.get("field_name")
        output_field_name = kwargs.get("output_field_name", f"{field_name}_generalized")
        _scale_factor_calculated = kwargs.get("_scale_factor_calculated", None)
        round_to_integer = kwargs.get("round_to_integer", None)
        scale_by_std = kwargs.get("scale_by_std", False)
        mode = kwargs.get("mode", "REPLACE")
        null_strategy = kwargs.get("null_strategy", "PRESERVE")

        result = batch.copy(deep=True)

        # Validate numeric field
        if not validate_numeric_field(
            batch, field_name, allow_null=(null_strategy != "ERROR")
        ):
            raise ValueError(
                f"Field '{field_name}' is not numeric or validation failed"
            )

        # Get values
        values = batch[field_name]

        # Auto-detect integer type if not specified
        if round_to_integer is None:
            kwargs["round_to_integer"] = pd.api.types.is_integer_dtype(values)

        # Calculate scale factor if needed (once per operation)
        if scale_by_std and _scale_factor_calculated is None:
            _scale_factor_calculated = cls._calculate_scale_factor(values, **kwargs)

        # Handle nulls based on strategy
        non_null_mask = values.notna()
        non_null_values = values[non_null_mask]
        if len(non_null_values) > 0:
            # Generate and apply noise
            noise = cls._generate_noise(len(non_null_values), **kwargs)
            noisy_values = cls._apply_noise(non_null_values, noise, **kwargs)

            # Update result
            if mode == "REPLACE":
                result.loc[non_null_mask, field_name] = noisy_values
            else:  # ENRICH
                result[output_field_name] = values.copy()
                result.loc[non_null_mask, output_field_name] = noisy_values
        else:
            # No non-null values to process
            if mode == "ENRICH":
                result[output_field_name] = values.copy()

        return result

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect uniform noise specific metrics.

        Args:
            original_data: Original data series
            anonymized_data: Data series after noise addition

        Returns:
            Dictionary of noise-specific metrics
        """
        # Calculate actual noise added
        actual_noise = anonymized_data - original_data

        # Basic noise statistics
        metrics = {
            "noise_type": self.noise_type,
            "noise_range": self.noise_range,
            "actual_noise_mean": float(actual_noise.mean()),
            "actual_noise_std": float(actual_noise.std()),
            "actual_noise_min": float(actual_noise.min()),
            "actual_noise_max": float(actual_noise.max()),
            "signal_to_noise_ratio": (
                float(original_data.std() / actual_noise.std())
                if actual_noise.std() > 0
                else float("inf")
            ),
            "values_at_bounds": {
                "at_min": (
                    int((anonymized_data == self.output_min).sum())
                    if self.output_min
                    else 0
                ),
                "at_max": (
                    int((anonymized_data == self.output_max).sum())
                    if self.output_max
                    else 0
                ),
            },
            "preserved_zeros": (
                int((original_data == 0).sum()) if self.preserve_zero else None
            ),
            "secure_random": self.use_secure_random,
        }

        # Add noise impact metrics
        impact_metrics = calculate_noise_impact(original_data, anonymized_data)
        metrics["noise_impact"] = impact_metrics

        # Add distribution preservation metrics
        dist_metrics = calculate_distribution_preservation(
            original_data, anonymized_data
        )
        metrics["distribution_preservation"] = dist_metrics

        # Add uniformity analysis
        if isinstance(self.noise_range, (tuple, list)):
            expected_min, expected_max = self.noise_range
        else:
            expected_min, expected_max = -self.noise_range, self.noise_range

        uniformity_metrics = analyze_noise_uniformity(
            actual_noise,
            expected_min * self.scale_factor,
            expected_max * self.scale_factor,
        )
        metrics["uniformity_analysis"] = uniformity_metrics

        # Add noise effectiveness analysis
        effectiveness = analyze_noise_effectiveness(
            original_data,
            anonymized_data,
            privacy_metric="snr",
            target_value=10.0,  # Target SNR of 10
        )
        metrics["noise_effectiveness"] = effectiveness

        # Add utility metrics
        try:
            utility = calculate_utility_metrics(
                original_data, anonymized_data, metric_set="standard"
            )
            metrics["utility_metrics"] = utility
        except NotImplementedError:
            # Function not yet implemented
            self.logger.debug("Utility metrics calculation not yet implemented")

        return metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
            Dictionary of parameters affecting the operation output
        """
        params = super()._get_cache_parameters()

        # Add noise-specific parameters
        params.update(
            {
                "field_name": self.field_name,
                "noise_range": self.noise_range,
                "noise_type": self.noise_type,
                "output_min": self.output_min,
                "output_max": self.output_max,
                "preserve_zero": self.preserve_zero,
                "round_to_integer": self.round_to_integer,
                "scale_by_std": self.scale_by_std,
                "scale_factor": self.scale_factor,
                "random_seed": self.random_seed if not self.use_secure_random else None,
                "use_secure_random": self.use_secure_random,
                "condition_field": self.condition_field,
                "condition_values": self.condition_values,
                "condition_operator": self.condition_operator,
                "multi_conditions": self.multi_conditions,
                "condition_logic": self.condition_logic,
                "ka_risk_field": self.ka_risk_field,
                "risk_threshold": self.risk_threshold,
                "vulnerable_record_strategy": self.vulnerable_record_strategy,
                "optimize_memory": self.optimize_memory,
                "adaptive_chunk_size": self.adaptive_chunk_size,
                "mode": self.mode,
                "column_prefix": self.column_prefix,
                "output_field_name": self.output_field_name,
                "null_strategy": self.null_strategy,
                "chunk_size": self.chunk_size,
                "use_dask": self.use_dask,
                "npartitions": self.npartitions,
                "dask_partition_size": self.dask_partition_size,
                "use_vectorization": self.use_vectorization,
                "parallel_processes": self.parallel_processes,
                "use_cache": self.use_cache,
                "output_format": self.output_format,
                "visualization_theme": self.visualization_theme,
                "visualization_backend": self.visualization_backend,
                "visualization_strict": self.visualization_strict,
                "visualization_timeout": self.visualization_timeout,
                "use_encryption": self.use_encryption,
                "encryption_mode": self.encryption_mode,
                "encryption_key": self.encryption_key,
            }
        )

        return params

    def __repr__(self) -> str:
        """String representation of the operation."""
        range_str = (
            f"±{self.noise_range}"
            if isinstance(self.noise_range, (int, float))
            else str(self.noise_range)
        )
        return (
            f"UniformNumericNoiseOperation("
            f"field='{self.field_name}', "
            f"range={range_str}, "
            f"type='{self.noise_type}', "
            f"secure={self.use_secure_random})"
        )
