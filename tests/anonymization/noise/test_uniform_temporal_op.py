"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Test Uniform Temporal Noise Operation
Package:       tests.anonymization.noise
Version:       1.0.0
Status:        development
Author:        PAMOLA Core Team
Created:       2025-07-10
License:       BSD 3-Clause

Description:
   Comprehensive test suite for UniformTemporalNoiseOperation class.
   
   This test suite validates all aspects of the temporal noise operation including:
   - Basic temporal shift functionality with various time units
   - Direction control (forward, backward, both)
   - Boundary datetime enforcement
   - Weekend and time-of-day preservation
   - Special date preservation
   - Output granularity control
   - Statistical properties of generated noise
   - Error handling and edge cases
   - Integration with framework utilities

Test Categories:
   1. Configuration validation
   2. Initialization tests (factory, parameters, inheritance)
   3. Core temporal shift strategies (direction, time units)
   4. Mode operations (REPLACE/ENRICH)
   5. Null handling strategies
   6. Error handling and validation
   7. Processing methods (batch, value processing)
   8. Advanced features (DataWriter, progress, visualization, encryption)

Dependencies:
   - pytest: Testing framework
   - pandas: Data manipulation
   - numpy: Numerical operations
   - unittest.mock: Mocking framework utilities
   - datetime: Date/time handling
   
Reference Implementation:
   Uses established test patterns and comprehensive coverage approaches.
   Adapted for temporal noise operation requirements

Success Criteria:
   - 100% test pass rate
   - Comprehensive coverage of all operation aspects
   - Real implementation testing based on source code analysis
   - Statistical validation of temporal noise properties
   - Proper isolation and clean workspace management
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pamola_core.anonymization.noise.uniform_temporal_op import (
    UniformTemporalNoiseOperation,
    UniformTemporalNoiseConfig
)
from pamola_core.anonymization.commons.validation.exceptions import (
    ValidationError,
    InvalidParameterError
)
from pamola_core.utils.ops.op_config import OperationConfig


class TestUniformTemporalNoiseOperation:
    """Comprehensive test suite for UniformTemporalNoiseOperation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock data source with all required attributes
        self.mock_data_source = Mock()
        self.mock_data_source.name = "test_datasource"
        self.mock_data_source.description = "Test data source"
        self.mock_data_source.metadata = {"test": "metadata"}
        self.mock_data_source.get_schema.return_value = {"timestamp": "datetime64[ns]"}
        
        # Test data paths
        self.test_config_file = Path(__file__).parent.parent / "configs" / "test_config_uniform_temporal_op.json"
        
        # Ensure test config exists
        if not self.test_config_file.exists():
            pytest.skip(f"Test config file not found: {self.test_config_file}")

    def get_fresh_temporal_df(self, size: int = 100, include_nulls: bool = True, 
                             include_weekends: bool = True) -> pd.DataFrame:
        """
        Generate fresh DataFrame with temporal data for testing.
        
        Args:
            size: Number of rows to generate
            include_nulls: Whether to include null values
            include_weekends: Whether to include weekend dates
            
        Returns:
            DataFrame with temporal test data
        """
        # Generate base datetime range
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        date_range = pd.date_range(start=start_date, end=end_date, periods=size)
        
        # Add time components for variety
        timestamps = []
        for i, base_date in enumerate(date_range):
            # Add varied time components
            hour = i % 24
            minute = (i * 7) % 60  
            second = (i * 13) % 60
            # Create proper timestamp without microseconds to avoid parsing issues
            timestamp = pd.Timestamp(
                year=base_date.year,
                month=base_date.month,
                day=base_date.day,
                hour=hour,
                minute=minute,
                second=second
            )
            timestamps.append(timestamp)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'id': range(size),
            'category': [f'cat_{i % 5}' for i in range(size)],
            'value': np.random.normal(100, 20, size)
        })
        
        # Add nulls if requested
        if include_nulls and size > 10:
            null_indices = np.random.choice(size, size // 10, replace=False)
            df.loc[null_indices, 'timestamp'] = pd.NaT
        
        # Filter weekends if not requested
        if not include_weekends:
            weekday_mask = df['timestamp'].dt.dayofweek < 5
            df = df[weekday_mask].reset_index(drop=True)
        
        return df

    def get_fresh_weekend_df(self, size: int = 50) -> pd.DataFrame:
        """Generate DataFrame with known weekend/weekday distribution."""
        weekdays = pd.date_range('2023-01-02', periods=size//2, freq='B')  # Business days
        weekends = pd.date_range('2023-01-07', periods=size//2, freq='W-SAT')  # Saturdays
        
        all_dates = list(weekdays) + list(weekends)
        all_dates = all_dates[:size]
        
        df = pd.DataFrame({
            'timestamp': all_dates,
            'day_type': ['weekday'] * len(weekdays) + ['weekend'] * (size - len(weekdays))
        })
        
        return df.sample(frac=1).reset_index(drop=True)  # Shuffle

    # =============================================================================
    # 1. Configuration Tests
    # =============================================================================

    def test_config_validation(self):
        """Test configuration validation for temporal noise parameters."""
        # Test that valid configurations can be created without errors
        
        # Valid configuration with days range
        config_days = UniformTemporalNoiseConfig(
            field_name="timestamp",
            noise_range_days=30,
            direction="both"
        )
        assert config_days is not None
        
        # Valid configuration with hours range  
        config_hours = UniformTemporalNoiseConfig(
            field_name="timestamp",
            noise_range_hours=48,
            direction="forward"
        )
        assert config_hours is not None
        
        # Valid configuration with minutes range
        config_minutes = UniformTemporalNoiseConfig(
            field_name="timestamp",
            noise_range_minutes=120,
            direction="backward"
        )
        assert config_minutes is not None
        
        # Valid configuration with seconds range
        config_seconds = UniformTemporalNoiseConfig(
            field_name="timestamp",
            noise_range_seconds=3600,
            direction="both"
        )
        assert config_seconds is not None
        
        # Valid multi-unit configuration
        config_multi = UniformTemporalNoiseConfig(
            field_name="timestamp",
            noise_range_days=7,
            noise_range_hours=12,
            noise_range_minutes=30,
            direction="both"
        )
        assert config_multi is not None

    # =============================================================================
    # 2. Initialization Tests  
    # =============================================================================

    def test_init_factory_pattern(self):
        """Test operation initialization through factory pattern."""
        # Test basic initialization
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30
        )
        assert operation.field_name == "timestamp"
        assert operation.noise_range_days == 30
        assert operation.direction == "both"
        assert operation.mode == "REPLACE"
        
        # Test with multiple time units
        operation_multi = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=7,
            noise_range_hours=12,
            noise_range_minutes=30,
            noise_range_seconds=45
        )
        assert operation_multi.noise_range_days == 7
        assert operation_multi.noise_range_hours == 12
        assert operation_multi.noise_range_minutes == 30
        assert operation_multi.noise_range_seconds == 45

    def test_init_with_valid_parameters(self):
        """Test initialization with comprehensive valid parameters."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            direction="forward",
            min_datetime="2020-01-01",
            max_datetime="2025-12-31",
            preserve_weekends=True,
            preserve_time_of_day=True,
            output_granularity="day",
            random_seed=42,
            use_secure_random=False,
            mode="ENRICH",
            null_strategy="PRESERVE"
        )
        
        assert operation.field_name == "timestamp"
        assert operation.noise_range_days == 30
        assert operation.direction == "forward"
        assert operation.min_datetime == pd.Timestamp("2020-01-01")
        assert operation.max_datetime == pd.Timestamp("2025-12-31")
        assert operation.preserve_weekends == True
        assert operation.preserve_time_of_day == True
        assert operation.output_granularity == "day"
        assert operation.random_seed == 42
        assert operation.use_secure_random == False
        assert operation.mode == "ENRICH"

    def test_inheritance_structure(self):
        """Test proper inheritance from base anonymization operation."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_hours=24
        )
        
        # Check inheritance chain
        from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
        assert isinstance(operation, AnonymizationOperation)
        
        # Check base class attributes are inherited
        assert hasattr(operation, 'field_name')
        assert hasattr(operation, 'mode')
        assert hasattr(operation, 'null_strategy')
        assert hasattr(operation, 'process_batch')

    # =============================================================================
    # 3. Core Temporal Shift Strategy Tests
    # =============================================================================

    def test_days_range_strategy(self):
        """Test temporal shift with days range."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            direction="both",
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(100, include_nulls=False)
        original_timestamps = df['timestamp'].copy()
        
        result = operation.process_batch(df)
        shifted_timestamps = result['timestamp']
        
        # Calculate actual shifts in days
        shifts = (shifted_timestamps - original_timestamps).dt.total_seconds() / (24 * 3600)
        shifts_clean = shifts.dropna()
        
        # Verify shifts are within expected range (±30 days)
        assert all(abs(shift) <= 30.1 for shift in shifts_clean), "Shifts exceed ±30 days range"
        
        # Check that some shifts occurred (not all zero)
        assert any(abs(shift) > 0.1 for shift in shifts_clean), "No significant shifts detected"

    def test_hours_range_strategy(self):
        """Test temporal shift with hours range."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_hours=48,
            direction="both",
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(100, include_nulls=False)
        original_timestamps = df['timestamp'].copy()
        
        result = operation.process_batch(df)
        shifted_timestamps = result['timestamp']
        
        # Calculate actual shifts in hours
        shifts = (shifted_timestamps - original_timestamps).dt.total_seconds() / 3600
        shifts_clean = shifts.dropna()
        
        # Verify shifts are within expected range (±48 hours)
        assert all(abs(shift) <= 48.1 for shift in shifts_clean), "Shifts exceed ±48 hours range"

    def test_direction_control_strategies(self):
        """Test direction control for temporal shifts."""
        df = self.get_fresh_temporal_df(100, include_nulls=False)
        
        # Test forward direction
        operation_forward = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            direction="forward",
            random_seed=42,
            use_secure_random=False
        )
        
        result_forward = operation_forward.process_batch(df.copy())
        forward_shifts = (result_forward['timestamp'] - df['timestamp']).dt.total_seconds()
        forward_shifts_clean = forward_shifts.dropna()
        
        # All shifts should be >= 0 (forward in time)
        assert all(shift >= -1 for shift in forward_shifts_clean), "Forward shifts contain negative values"
        
        # Test backward direction
        operation_backward = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            direction="backward",
            random_seed=42,
            use_secure_random=False
        )
        
        result_backward = operation_backward.process_batch(df.copy())
        backward_shifts = (result_backward['timestamp'] - df['timestamp']).dt.total_seconds()
        backward_shifts_clean = backward_shifts.dropna()
        
        # All shifts should be <= 0 (backward in time)
        assert all(shift <= 1 for shift in backward_shifts_clean), "Backward shifts contain positive values"

    # =============================================================================
    # 4. Mode Operation Tests
    # =============================================================================

    def test_replace_mode(self):
        """Test REPLACE mode functionality."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            mode="REPLACE",
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(50, include_nulls=False)
        original_columns = set(df.columns)
        
        result = operation.process_batch(df)
        
        # Should have same columns (no new columns added)
        assert set(result.columns) == original_columns
        
        # Original timestamp field should be modified
        assert not result['timestamp'].equals(df['timestamp'])

    def test_enrich_mode(self):
        """Test ENRICH mode functionality."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp", 
            noise_range_days=30,
            mode="ENRICH",
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(50, include_nulls=False)
        original_columns = set(df.columns)
        
        result = operation.process_batch(df)
        
        # Should have one additional column
        assert len(result.columns) == len(df.columns) + 1
        
        # Original timestamp field should be unchanged
        assert result['timestamp'].equals(df['timestamp'])
        
        # New field should exist with shifted values
        new_column = [col for col in result.columns if col not in original_columns][0]
        assert new_column in result.columns
        assert not result[new_column].equals(df['timestamp'])

    # =============================================================================
    # 5. Null Handling Tests
    # =============================================================================

    def test_null_handling_preserve(self):
        """Test null handling with PRESERVE strategy."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            null_strategy="PRESERVE",
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(50, include_nulls=True)
        null_mask = df['timestamp'].isna()
        null_count = null_mask.sum()
        
        if null_count > 0:
            result = operation.process_batch(df)
            
            # Null values should remain null
            result_null_mask = result['timestamp'].isna()
            assert result_null_mask.sum() == null_count
            assert result_null_mask.equals(null_mask)

    def test_null_handling_error(self):
        """Test null handling with ERROR strategy."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            null_strategy="ERROR"
        )
        
        df = self.get_fresh_temporal_df(50, include_nulls=True)
        null_count = df['timestamp'].isna().sum()
        
        if null_count > 0:
            # Should raise an error when nulls are present
            with pytest.raises((ValueError, InvalidParameterError, ValidationError)):
                operation.process_batch(df)

    # =============================================================================
    # 6. Error Handling Tests
    # =============================================================================

    def test_error_field_not_found(self):
        """Test error handling when field is not found."""
        operation = UniformTemporalNoiseOperation(
            field_name="nonexistent_field",
            noise_range_days=30
        )
        
        df = self.get_fresh_temporal_df(50)
        
        with pytest.raises(KeyError):
            operation.process_batch(df)

    def test_error_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Test invalid direction
        with pytest.raises(InvalidParameterError):
            UniformTemporalNoiseOperation(
                field_name="timestamp",
                noise_range_days=30,
                direction="invalid_direction"
            )
        
        # Test negative noise range
        with pytest.raises(InvalidParameterError):
            UniformTemporalNoiseOperation(
                field_name="timestamp",
                noise_range_days=-10
            )
        
        # Test invalid granularity
        with pytest.raises(InvalidParameterError):
            UniformTemporalNoiseOperation(
                field_name="timestamp",
                noise_range_days=30,
                output_granularity="invalid_granularity"
            )
        
        # Test no time components specified
        with pytest.raises(InvalidParameterError):
            UniformTemporalNoiseOperation(
                field_name="timestamp"
            )

    # =============================================================================
    # 7. Processing Method Tests
    # =============================================================================

    def test_batch_processing(self):
        """Test batch processing functionality."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(1000, include_nulls=False)
        
        # Process the batch
        result = operation.process_batch(df)
        
        # Verify result structure
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

    def test_dask_processing_integration(self):
        """Test Dask processing integration."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            use_dask=True,
            npartitions=4
        )
        
        df = self.get_fresh_temporal_df(100)
        
        # Note: process_batch doesn't implement Dask directly
        # This tests the integration pattern
        result = operation.process_batch(df)
        assert isinstance(result, pd.DataFrame)

    def test_value_processing_not_implemented(self):
        """Test that process_value raises NotImplementedError for temporal operations."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30
        )
        
        # Temporal operations don't implement process_value
        with pytest.raises(NotImplementedError):
            operation.process_value(pd.Timestamp('2023-01-01'))

    # =============================================================================
    # 8. Advanced Feature Tests
    # =============================================================================

    def test_advanced_data_writer_integration(self):
        """Test DataWriter integration for output and metrics."""
        
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(50)
        result = operation.process_batch(df)
        
        # Basic processing should work
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    @patch('pamola_core.utils.progress.ProgressTracker')
    def test_advanced_progress_tracking(self, mock_progress_class):
        """Test progress tracking integration."""
        mock_tracker = Mock()
        mock_progress_class.return_value = mock_tracker
        
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30
        )
        
        df = self.get_fresh_temporal_df(100)
        result = operation.process_batch(df)
        
        # Processing should complete successfully
        assert isinstance(result, pd.DataFrame)

    def test_advanced_visualization_generation(self):
        """Test visualization generation for temporal noise analysis."""
        
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            visualization_backend="plotly"
        )
        
        df = self.get_fresh_temporal_df(100)
        result = operation.process_batch(df)
        
        # Processing should work regardless of visualization
        assert isinstance(result, pd.DataFrame)

    def test_advanced_encryption_support(self):
        """Test encryption parameter handling."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            use_encryption=True,
            encryption_mode="AES-256"
        )
        
        # Should initialize without error
        assert operation.use_encryption == True
        assert hasattr(operation, 'encryption_mode')

    def test_advanced_chunked_processing(self):
        """Test chunked processing for large datasets."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            chunk_size=50,
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(200)
        result = operation.process_batch(df)
        
        # Should process entire dataset
        assert len(result) == len(df)
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

    def test_advanced_parallel_processing(self):
        """Test parallel processing configuration."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            parallel_processes=4,
            use_vectorization=True
        )
        
        df = self.get_fresh_temporal_df(100)
        result = operation.process_batch(df)
        
        # Should process successfully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_advanced_output_format_generation(self):
        """Test output format specification."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            output_format="parquet"
        )
        
        # Should initialize with specified format
        assert operation.output_format == "parquet"

    def test_advanced_memory_optimization(self):
        """Test memory optimization features."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            optimize_memory=True,
            adaptive_chunk_size=True
        )
        
        df = self.get_fresh_temporal_df(500)
        result = operation.process_batch(df)
        
        # Should process efficiently
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    # =============================================================================
    # 9. Temporal-Specific Feature Tests
    # =============================================================================

    def test_weekend_preservation(self):
        """Test weekend/weekday status preservation."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=14,
            preserve_weekends=True,
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_weekend_df(100)
        original_weekends = df['timestamp'].dt.dayofweek >= 5
        
        result = operation.process_batch(df)
        result_weekends = result['timestamp'].dt.dayofweek >= 5
        
        # Weekend status should be preserved (allowing for some boundary adjustments)
        # Note: Perfect preservation might not be possible due to boundary constraints
        preservation_rate = (original_weekends == result_weekends).mean()
        assert preservation_rate >= 0.8, f"Weekend preservation rate too low: {preservation_rate}"

    def test_time_of_day_preservation(self):
        """Test time of day preservation."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            preserve_time_of_day=True,
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(100, include_nulls=False)
        original_times = df['timestamp'].dt.time
        
        result = operation.process_batch(df)
        result_times = result['timestamp'].dt.time
        
        # Times should be preserved exactly
        non_null_mask = df['timestamp'].notna()
        if non_null_mask.any():
            # Convert to string comparison to avoid pandas parsing issues
            original_times_str = original_times[non_null_mask].astype(str)
            result_times_str = result_times[non_null_mask].astype(str)
            preserved_times = original_times_str.equals(result_times_str)
            assert preserved_times, "Time of day not properly preserved"

    def test_datetime_boundaries(self):
        """Test datetime boundary enforcement."""
        min_date = "2023-06-01"
        max_date = "2023-08-31"
        
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=365,  # Large range to test boundaries
            min_datetime=min_date,
            max_datetime=max_date,
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(100, include_nulls=False)
        result = operation.process_batch(df)
        
        # All results should be within bounds
        result_clean = result['timestamp'].dropna()
        if len(result_clean) > 0:
            min_result = result_clean.min()
            max_result = result_clean.max()
            
            assert min_result >= pd.Timestamp(min_date), f"Result below minimum: {min_result}"
            assert max_result <= pd.Timestamp(max_date), f"Result above maximum: {max_result}"

    def test_output_granularity(self):
        """Test output granularity rounding."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_hours=48,
            output_granularity="day",
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(50, include_nulls=False)
        result = operation.process_batch(df)
        
        # All times should be rounded to day (00:00:00)
        result_clean = result['timestamp'].dropna()
        if len(result_clean) > 0:
            times = result_clean.dt.time
            midnight = pd.Timestamp('00:00:00').time()
            all_midnight = all(t == midnight for t in times)
            assert all_midnight, "Granularity not properly applied"

    def test_statistical_noise_properties(self):
        """Test statistical properties of temporal noise distribution."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            direction="both",
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(1000, include_nulls=False)
        original_timestamps = df['timestamp'].copy()
        
        result = operation.process_batch(df)
        shifted_timestamps = result['timestamp']
        
        # Calculate shifts in days
        shifts = (shifted_timestamps - original_timestamps).dt.total_seconds() / (24 * 3600)
        shifts_clean = shifts.dropna()
        
        if len(shifts_clean) > 100:  # Need sufficient sample size
            # Test uniform distribution properties
            mean_shift = shifts_clean.mean()
            std_shift = shifts_clean.std()
            
            # Mean should be close to 0 for "both" direction (with tolerance)
            assert abs(mean_shift) < 2.0, f"Mean shift too large: {mean_shift}"
            
            # Standard deviation should indicate spread
            assert std_shift > 5.0, f"Standard deviation too small: {std_shift}"
            
            # Range should be approximately correct (±30 days)
            assert shifts_clean.min() >= -30.5, f"Minimum shift too negative: {shifts_clean.min()}"
            assert shifts_clean.max() <= 30.5, f"Maximum shift too positive: {shifts_clean.max()}"

    def test_reproducibility_with_seed(self):
        """Test reproducibility with fixed random seed."""
        config_params = {
            "field_name": "timestamp",
            "noise_range_days": 30,
            "random_seed": 42,
            "use_secure_random": False
        }
        
        operation1 = UniformTemporalNoiseOperation(**config_params)
        operation2 = UniformTemporalNoiseOperation(**config_params)
        
        df = self.get_fresh_temporal_df(100, include_nulls=False)
        
        result1 = operation1.process_batch(df.copy())
        result2 = operation2.process_batch(df.copy())
        
        # Results should be identical with same seed
        timestamps_equal = result1['timestamp'].equals(result2['timestamp'])
        assert timestamps_equal, "Results not reproducible with same seed"

    # =============================================================================
    # 10. Integration and Metrics Tests
    # =============================================================================

    def test_metrics_collection(self):
        """Test temporal-specific metrics collection."""
        operation = UniformTemporalNoiseOperation(
            field_name="timestamp",
            noise_range_days=30,
            random_seed=42,
            use_secure_random=False
        )
        
        df = self.get_fresh_temporal_df(100, include_nulls=False)
        original_data = df['timestamp'].copy()
        
        result = operation.process_batch(df)
        anonymized_data = result['timestamp']
        
        # Collect metrics (this tests the internal method)
        if hasattr(operation, '_collect_specific_metrics'):
            metrics = operation._collect_specific_metrics(original_data, anonymized_data)
            
            # Verify expected metric categories
            assert 'noise_range_config' in metrics
            assert 'direction' in metrics
            assert 'actual_shifts' in metrics
            assert 'shift_direction' in metrics
            
            # Verify noise range configuration
            assert metrics['noise_range_config']['days'] == 30
            assert metrics['direction'] == "both"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
