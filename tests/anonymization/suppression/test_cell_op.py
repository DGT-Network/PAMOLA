"""
Tests for CellSuppressionOperation.

This module contains comprehensive tests for the CellSuppressionOperation class,
covering all functionality including basic suppression strategies, conditional suppression,
outlier detection, rare value detection, group-based strategies, and advanced features.

Test Architecture:
- Configuration tests: Parameter validation and initialization
- Core functionality tests: All suppression strategies and conditions
- Mode tests: REPLACE vs ENRICH mode behavior
- Conditional suppression tests: Field-based and value-based filtering
- Group-based tests: Group mean and mode strategies
- Processing tests: Pandas, Dask, and Joblib processing methods
- Advanced feature tests: Encryption, visualization, caching, and parallel processing
- Error handling tests: Invalid inputs and edge cases
"""

import json
import os
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import numpy as np

# Import the operation and related classes
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.anonymization.commons.validation.exceptions import FieldNotFoundError, FieldTypeError
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker


class TestCellSuppressionOperation:
    """Test suite for CellSuppressionOperation."""

    @pytest.fixture
    def test_config_data(self):
        """Load test configuration data."""
        config_path = Path(__file__).parent.parent / "configs" / "test_config_cell_op.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def op_config(self):
        """Create a mock OperationConfig."""
        config = Mock(spec=OperationConfig)
        config.get_operation_config.return_value = {
            "field_name": "test_field",
            "suppression_strategy": "null",
            "mode": "REPLACE"
        }
        config.get_io_config.return_value = {
            "input_format": "csv",
            "output_format": "csv",
            "encryption_enabled": False
        }
        config.get_processing_config.return_value = {
            "use_dask": False,
            "use_vectorization": False,
            "chunk_size": 10000
        }
        return config

    @pytest.fixture
    def mock_data_source(self):
        """Create a mock DataSource."""
        data_source = Mock()  # Remove spec=DataSource to allow any method
        # Create fresh DataFrame directly in fixture
        fresh_df = pd.DataFrame({
            'test_field': ['value1', 'value2', 'value3', 'value4', 'value5', None, 'value7', 'value8'],
            'numeric_field': [10, 20, 30, 40, 50, 60, 1000, 80],  # 1000 is outlier
            'categorical_field': ['A', 'B', 'A', 'C', 'A', 'B', 'D', 'D'],  # D is rare
            'category': ['group1', 'group1', 'group2', 'group2', 'group1', 'group2', 'group1', 'group2'],
            'sensitive_field': ['secret1', 'public', 'secret2', 'public', 'secret3', 'public', 'secret4', 'public'],
            'score': [85, 95, 75, 90, 80, 100, 70, 88],
            'value_field': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        })
        # DataSource.get_dataframe returns a tuple (df, error_info)
        data_source.get_dataframe.return_value = (fresh_df, None)
        data_source.get_field_info.return_value = {
            "name": "test_field",
            "type": "object",
            "nullable": True
        }
        return data_source

    @pytest.fixture
    def task_dir(self):
        """Create a temporary task directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def get_fresh_cell_df(self) -> pd.DataFrame:
        """Create a fresh DataFrame for cell suppression testing."""
        return pd.DataFrame({
            'test_field': ['value1', 'value2', 'value3', 'value4', 'value5', None, 'value7', 'value8'],
            'numeric_field': [10, 20, 30, 40, 50, 60, 1000, 80],  # 1000 is outlier
            'categorical_field': ['A', 'B', 'A', 'C', 'A', 'B', 'D', 'D'],  # D is rare
            'category': ['group1', 'group1', 'group2', 'group2', 'group1', 'group2', 'group1', 'group2'],
            'sensitive_field': ['secret1', 'public', 'secret2', 'public', 'secret3', 'public', 'secret4', 'public'],
            'score': [85, 95, 75, 90, 80, 100, 70, 88],
            'value_field': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        })

    def get_fresh_numeric_df(self) -> pd.DataFrame:
        """Create a fresh DataFrame with numeric data for testing."""
        return pd.DataFrame({
            'numeric_field': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'category': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })

    def get_fresh_outlier_df(self) -> pd.DataFrame:
        """Create a fresh DataFrame with outliers for testing."""
        return pd.DataFrame({
            'numeric_field': [10, 20, 30, 40, 50, 1000, 70, 80, 90, 100],  # 1000 is outlier
            'category': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })

    # =======================
    # CONFIGURATION TESTS
    # =======================

    def test_config_validation_all_strategies(self, test_config_data):
        """Test that all suppression strategies from config are valid."""
        config = test_config_data["basic_null_suppression"]["config"]
        
        # Test all valid suppression strategies
        valid_strategies = ["null", "mean", "median", "mode", "constant", "group_mean", "group_mode"]
        for strategy in valid_strategies:
            config_copy = config.copy()
            config_copy["suppression_strategy"] = strategy
            
            # Add required parameters for specific strategies
            if strategy == "constant":
                config_copy["suppression_value"] = "TEST"
            elif strategy in ["group_mean", "group_mode"]:
                config_copy["group_by_field"] = "category"
                
            # Should not raise any exception
            operation = CellSuppressionOperation(**config_copy)
            assert operation.suppression_strategy == strategy

    # =======================
    # INITIALIZATION TESTS
    # =======================

    def test_factory_initialization(self, test_config_data):
        """Test initialization from factory method."""
        config = test_config_data["basic_null_suppression"]["config"]
        operation = CellSuppressionOperation(**config)
        
        assert operation.field_name == "test_field"
        assert operation.suppression_strategy == "null"
        assert operation.mode == "REPLACE"

    def test_initialization_with_all_parameters(self, test_config_data):
        """Test initialization with comprehensive parameters."""
        config = test_config_data["mean_suppression"]["config"]
        operation = CellSuppressionOperation(
            field_name="numeric_field",
            suppression_strategy="mean",
            mode="REPLACE",
            suppress_if="outlier",
            outlier_method="iqr",
            outlier_threshold=1.5,
            chunk_size=5000,
            use_cache=True,
            generate_visualization=True
        )
        
        assert operation.field_name == "numeric_field"
        assert operation.suppression_strategy == "mean"
        assert operation.suppress_if == "outlier"
        assert operation.outlier_method == "iqr"
        assert operation.outlier_threshold == 1.5
        assert operation.chunk_size == 5000

    def test_inheritance_from_base_class(self):
        """Test that CellSuppressionOperation inherits from AnonymizationOperation."""
        operation = CellSuppressionOperation(field_name="test_field")
        
        # Check inheritance
        from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
        assert isinstance(operation, AnonymizationOperation)
        
        # Check that base class methods are available
        assert hasattr(operation, 'execute')
        assert hasattr(operation, 'process_batch')

    # =======================
    # SUPPRESSION STRATEGY TESTS
    # =======================

    def test_null_suppression_strategy(self):
        """Test null suppression strategy."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null"
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        # All non-null values should be replaced with None
        assert result["test_field"].isna().sum() > df["test_field"].isna().sum()

    def test_mean_suppression_strategy(self):
        """Test mean suppression strategy."""
        operation = CellSuppressionOperation(
            field_name="numeric_field",
            suppression_strategy="mean"
        )
        
        df = self.get_fresh_numeric_df()
        original_mean = df["numeric_field"].mean()
        result = operation.process_batch(df)
        
        # All values should be replaced with the mean
        assert all(result["numeric_field"] == original_mean)

    def test_median_suppression_strategy(self):
        """Test median suppression strategy."""
        operation = CellSuppressionOperation(
            field_name="numeric_field",
            suppression_strategy="median"
        )
        
        df = self.get_fresh_numeric_df()
        original_median = df["numeric_field"].median()
        result = operation.process_batch(df)
        
        # All values should be replaced with the median
        assert all(result["numeric_field"] == original_median)

    def test_mode_suppression_strategy(self):
        """Test mode suppression strategy."""
        operation = CellSuppressionOperation(
            field_name="categorical_field",
            suppression_strategy="mode"
        )
        
        df = self.get_fresh_cell_df()
        original_mode = df["categorical_field"].mode().iloc[0]
        result = operation.process_batch(df)
        
        # All values should be replaced with the mode
        assert all(result["categorical_field"] == original_mode)

    def test_constant_suppression_strategy(self):
        """Test constant suppression strategy."""
        constant_value = "REDACTED"
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="constant",
            suppression_value=constant_value
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        # All non-null values should be replaced with the constant value
        non_null_mask = df["test_field"].notna()
        assert all(result.loc[non_null_mask, "test_field"] == constant_value)

    def test_group_mean_suppression_strategy(self):
        """Test group-based mean suppression strategy."""
        operation = CellSuppressionOperation(
            field_name="numeric_field",
            suppression_strategy="group_mean",
            group_by_field="category",
            min_group_size=2
        )
        
        df = self.get_fresh_numeric_df()
        result = operation.process_batch(df)
        
        # Values should be replaced with group means
        group_means = df.groupby("category")["numeric_field"].mean()
        for category in df["category"].unique():
            category_mask = df["category"] == category
            expected_mean = group_means[category]
            assert all(result.loc[category_mask, "numeric_field"] == expected_mean)

    def test_group_mode_suppression_strategy(self):
        """Test group-based mode suppression strategy."""
        operation = CellSuppressionOperation(
            field_name="categorical_field",
            suppression_strategy="group_mode",
            group_by_field="category",
            min_group_size=2
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        # Values should be replaced with group modes
        for category in df["category"].unique():
            category_mask = df["category"] == category
            group_data = df.loc[category_mask, "categorical_field"]
            if len(group_data) >= 2:  # min_group_size
                expected_mode = group_data.mode().iloc[0] if not group_data.mode().empty else group_data.iloc[0]
                assert all(result.loc[category_mask, "categorical_field"] == expected_mode)

    # =======================
    # MODE TESTS
    # =======================

    def test_replace_mode(self):
        """Test REPLACE mode behavior."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="constant",
            suppression_value="MASKED",
            mode="REPLACE"
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        # Original field should be modified
        assert "test_field" in result.columns
        # No new field should be created
        assert "test_field_masked" not in result.columns

    def test_enrich_mode(self):
        """Test ENRICH mode behavior."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="constant",
            suppression_value="MASKED",
            mode="ENRICH",
            output_field_name="test_field_masked"
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        # Original field should be preserved
        assert "test_field" in result.columns
        # New field should be created
        assert "test_field_masked" in result.columns
        # Original values should be unchanged
        pd.testing.assert_series_equal(result["test_field"], df["test_field"])

    # =======================
    # CONDITIONAL SUPPRESSION TESTS
    # =======================

    def test_conditional_suppression_in_operator(self):
        """Test conditional suppression with 'in' operator."""
        operation = CellSuppressionOperation(
            field_name="sensitive_field",
            suppression_strategy="constant",
            suppression_value="REDACTED",
            condition_field="category",
            condition_values=["group1"],
            condition_operator="in"
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        # Only values where category is 'group1' should be suppressed
        group1_mask = df["category"] == "group1"
        assert all(result.loc[group1_mask, "sensitive_field"] == "REDACTED")
        
        # Values where category is not 'group1' should be unchanged
        group2_mask = df["category"] == "group2"
        original_group2_values = df.loc[group2_mask, "sensitive_field"]
        result_group2_values = result.loc[group2_mask, "sensitive_field"]
        pd.testing.assert_series_equal(result_group2_values, original_group2_values)

    def test_outlier_suppression_iqr_method(self):
        """Test outlier suppression using IQR method."""
        operation = CellSuppressionOperation(
            field_name="numeric_field",
            suppression_strategy="median",
            suppress_if="outlier",
            outlier_method="iqr",
            outlier_threshold=1.5
        )
        
        df = self.get_fresh_outlier_df()
        result = operation.process_batch(df)
        
        # The outlier value (1000) should be replaced with median
        median_val = df["numeric_field"].median()
        
        # Check that outlier was detected and replaced
        assert 1000 not in result["numeric_field"].values
        assert median_val in result["numeric_field"].values

    def test_outlier_suppression_zscore_method(self):
        """Test outlier suppression using Z-score method."""
        operation = CellSuppressionOperation(
            field_name="numeric_field",
            suppression_strategy="mean",
            suppress_if="outlier",
            outlier_method="zscore",
            outlier_threshold=2.0
        )
        
        df = self.get_fresh_outlier_df()
        result = operation.process_batch(df)
        
        # The outlier value (1000) should be replaced with mean
        mean_val = df["numeric_field"].mean()
        
        # Check that outlier was detected and replaced
        assert 1000 not in result["numeric_field"].values

    def test_rare_value_suppression(self):
        """Test rare value suppression."""
        operation = CellSuppressionOperation(
            field_name="categorical_field",
            suppression_strategy="mode",
            suppress_if="rare",
            rare_threshold=3  # Values appearing less than 3 times are rare
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        # Count occurrences of each value
        value_counts = df["categorical_field"].value_counts()
        rare_values = value_counts[value_counts < 3].index
        
        # Rare values should be replaced with mode
        mode_val = df["categorical_field"].mode().iloc[0]
        for rare_val in rare_values:
            # Check that rare values are no longer present (replaced with mode)
            rare_mask = df["categorical_field"] == rare_val
            if rare_mask.any():
                assert all(result.loc[rare_mask, "categorical_field"] == mode_val)

    # =======================
    # ERROR HANDLING TESTS
    # =======================

    def test_field_not_found_error(self):
        """Test error when field is not found."""
        operation = CellSuppressionOperation(
            field_name="nonexistent_field",
            suppression_strategy="null"
        )
        
        df = self.get_fresh_cell_df()
        
        with pytest.raises(FieldNotFoundError):
            operation.process_batch(df)

    def test_invalid_suppression_strategy(self):
        """Test error with invalid suppression strategy."""
        with pytest.raises(ValueError, match="Invalid suppression_strategy"):
            CellSuppressionOperation(
                field_name="test_field",
                suppression_strategy="invalid_strategy"
            )

    def test_constant_strategy_without_value(self):
        """Test error when constant strategy is used without suppression_value."""
        with pytest.raises(ValueError, match="suppression_value required"):
            CellSuppressionOperation(
                field_name="test_field",
                suppression_strategy="constant"
            )

    def test_group_strategy_without_group_field(self):
        """Test error when group strategy is used without group_by_field."""
        with pytest.raises(ValueError, match="group_by_field required"):
            CellSuppressionOperation(
                field_name="test_field",
                suppression_strategy="group_mean"
            )

    def test_invalid_suppress_if_parameter(self):
        """Test error with invalid suppress_if parameter."""
        with pytest.raises(ValueError, match="Invalid suppress_if"):
            CellSuppressionOperation(
                field_name="test_field",
                suppression_strategy="null",
                suppress_if="invalid_condition"
            )

    def test_numeric_strategy_on_non_numeric_field(self):
        """Test error when numeric strategy is applied to non-numeric field."""
        operation = CellSuppressionOperation(
            field_name="test_field",  # Non-numeric field
            suppression_strategy="mean"
        )
        
        df = self.get_fresh_cell_df()
        
        with pytest.raises(FieldTypeError):
            operation.process_batch(df)

    # =======================
    # PROCESSING METHOD TESTS
    # =======================

    @patch('pamola_core.anonymization.suppression.cell_op.dd')
    def test_process_batch_pandas(self, mock_dd):
        """Test pandas batch processing."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            use_dask=False,
            use_vectorization=False
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_process_with_dask(self):
        """Test Dask distributed processing."""
        # Mock both the module-level import and the method-level import
        with patch('pamola_core.anonymization.suppression.cell_op.dd') as mock_dd_module, \
             patch('dask.dataframe.from_pandas') as mock_from_pandas:
            
            # Mock Dask DataFrame with all required attributes
            mock_ddf = Mock()
            mock_from_pandas.return_value = mock_ddf
            mock_ddf.map_partitions.return_value = mock_ddf
            
            # Create a proper mock compute result with _suppression_mask_
            result_df = self.get_fresh_cell_df()
            result_df["_suppression_mask_"] = pd.Series([True, False, True, False, True, False, True, False])
            mock_ddf.compute.return_value = result_df
            
            operation = CellSuppressionOperation(
                field_name="test_field",
                suppression_strategy="null",
                use_dask=True,
                npartitions=2
            )
            
            df = self.get_fresh_cell_df()
            
            # Ensure the attributes are set correctly
            assert operation.use_dask == True
            assert operation.npartitions == 2
            assert operation.use_vectorization == False  # This should be default False
            
            # Test the _process_data method directly to ensure Dask path is taken
            suppression_mask, result = operation._process_data(df)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(df)
            assert isinstance(suppression_mask, pd.Series)
            
            # Check that the Dask functions were called
            mock_from_pandas.assert_called_once()
            mock_ddf.map_partitions.assert_called_once()
            mock_ddf.compute.assert_called_once()

    def test_process_with_joblib(self):
        """Test parallel processing with joblib."""
        with patch('joblib.Parallel') as mock_parallel:
            
            # Mock joblib Parallel - it should return a list of tuples
            mock_parallel.return_value.return_value = [
                (self.get_fresh_cell_df(), pd.Series([True, False, True, False, True, False, True, False]))
            ]
            
            operation = CellSuppressionOperation(
                field_name="test_field",
                suppression_strategy="null",
                use_vectorization=True,
                parallel_processes=2
            )
            
            df = self.get_fresh_cell_df()
            
            # Ensure the attributes are set correctly
            assert operation.use_vectorization == True
            assert operation.parallel_processes == 2
            assert operation.use_dask == False  # This should be default False
            
            # Test the _process_data method directly to ensure joblib path is taken
            suppression_mask, result = operation._process_data(df)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(df)
            assert isinstance(suppression_mask, pd.Series)
            
            # Check that joblib Parallel was called
            mock_parallel.assert_called()

    # =======================
    # ADVANCED FEATURE TESTS
    # =======================

    @patch('pamola_core.anonymization.suppression.cell_op.DataWriter')
    def test_complex_parameters_with_data_writer(self, mock_data_writer):
        """Test operation with complex parameters and DataWriter."""
        mock_writer = Mock()
        mock_data_writer.return_value = mock_writer
        
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="constant",
            suppression_value="MASKED",
            output_format="json",
            use_encryption=True,
            encryption_mode="AES"
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    @patch('pamola_core.anonymization.suppression.cell_op.HierarchicalProgressTracker')
    def test_progress_tracking(self, mock_progress_tracker):
        """Test progress tracking functionality."""
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null"
        )
        
        df = self.get_fresh_cell_df()
        mock_data_source = Mock()
        mock_data_source.get_data.return_value = df
        mock_data_source.get_dataframe.return_value = (df, None)
        mock_data_source.get_field_info.return_value = {
            "name": "test_field",
            "type": "object",
            "nullable": True
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            result = operation.execute(mock_data_source, task_dir, progress_tracker=mock_tracker)
            
            assert result.status == OperationStatus.SUCCESS

    @patch('pamola_core.anonymization.suppression.cell_op.create_histogram')
    @patch('pamola_core.anonymization.suppression.cell_op.create_comparison_visualization')
    def test_visualization_generation(self, mock_comparison, mock_histogram, mock_data_source, task_dir):
        """Test visualization generation."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            generate_visualization=True,
            visualization_backend="matplotlib"
        )
        
        result = operation.execute(mock_data_source, task_dir)
        
        assert result.status == OperationStatus.SUCCESS

    @patch('pamola_core.anonymization.suppression.cell_op.crypto_utils')
    def test_encryption_handling(self, mock_crypto, mock_data_source, task_dir):
        """Test encryption functionality."""
        mock_crypto.get_encryption_mode.return_value = "AES"
        
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            use_encryption=True,
            encryption_key="test_key",
            encryption_mode="AES"
        )
        
        result = operation.execute(mock_data_source, task_dir)
        
        assert result.status == OperationStatus.SUCCESS
        
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            result = operation.execute(mock_data_source, task_dir)
            
            assert result.status == OperationStatus.SUCCESS

    def test_chunked_processing(self, mock_data_source, task_dir):
        """Test chunked processing for large datasets."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            optimize_memory=True,
            chunk_size=3  # Small chunk size for testing
        )
        
        result = operation.execute(mock_data_source, task_dir)
        
        assert result.status == OperationStatus.SUCCESS

    def test_parallel_processing_configuration(self):
        """Test parallel processing configuration."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            use_vectorization=True,
            parallel_processes=2,
            chunk_size=5
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_output_generation_with_metrics(self, mock_data_source, task_dir):
        """Test output generation with metrics collection."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            save_output=True,
            output_format="csv"
        )
        
        result = operation.execute(mock_data_source, task_dir)
        
        assert result.status == OperationStatus.SUCCESS
        assert result.metrics is not None
        assert "operation_type" in result.metrics
        assert result.metrics["operation_type"] == "cell_suppression"

    # =======================
    # INTEGRATION TESTS
    # =======================

    def test_full_execution_cycle(self, mock_data_source, task_dir):
        """Test full execution cycle from start to finish."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="constant",
            suppression_value="REDACTED",
            save_output=True,
            generate_visualization=False,  # Disable for simpler testing
            use_cache=False
        )
        
        result = operation.execute(mock_data_source, task_dir)
        
        assert result.status == OperationStatus.SUCCESS
        assert result.execution_time >= 0  # Can be 0 for very fast operations
        assert result.metrics is not None
        assert "cells_suppressed" in result.metrics
        assert "suppression_rate" in result.metrics

    def test_error_handling_in_execution(self, mock_data_source, task_dir):
        """Test error handling during execution."""
        operation = CellSuppressionOperation(
            field_name="nonexistent_field",
            suppression_strategy="null"
        )
        
        result = operation.execute(mock_data_source, task_dir)
        
        assert result.status == OperationStatus.ERROR
        assert result.error_message is not None

    def test_caching_behavior(self, mock_data_source, task_dir):
        """Test caching behavior."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            use_cache=True,
            force_recalculation=False
        )
        
        # First execution
        result1 = operation.execute(mock_data_source, task_dir)
        assert result1.status == OperationStatus.SUCCESS
        
        # Second execution should use cache
        result2 = operation.execute(mock_data_source, task_dir)
        assert result2.status == OperationStatus.SUCCESS

    def test_memory_optimization(self):
        """Test memory optimization features."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="null",
            optimize_memory=True,
            adaptive_chunk_size=True,
            chunk_size=2
        )
        
        df = self.get_fresh_cell_df()
        result = operation.process_batch(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_null_strategy_handling(self):
        """Test null strategy handling."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="mode",
            null_strategy="PRESERVE"
        )
        
        df = self.get_fresh_cell_df()
        original_null_count = df["test_field"].isna().sum()
        result = operation.process_batch(df)
        
        # Null count should be preserved
        assert result["test_field"].isna().sum() == original_null_count

    def test_metrics_collection_comprehensive(self, mock_data_source, task_dir):
        """Test comprehensive metrics collection."""
        operation = CellSuppressionOperation(
            field_name="test_field",
            suppression_strategy="constant",
            suppression_value="MASKED",
            suppress_if="rare",
            rare_threshold=5
        )
        
        result = operation.execute(mock_data_source, task_dir)
        
        assert result.status == OperationStatus.SUCCESS
        assert result.metrics is not None
        
        # Check specific metrics
        expected_metrics = [
            "operation_type",
            "suppression_strategy",
            "cells_suppressed",
            "suppression_rate",
            "total_cells_processed",
            "non_null_cells_processed"
        ]
        
        for metric in expected_metrics:
            assert metric in result.metrics
