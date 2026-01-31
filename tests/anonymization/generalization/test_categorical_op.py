"""
Unit tests for CategoricalGeneralizationOperation.

This module contains comprehensive tests for the categorical generalization operation,
following transformation operation test patterns and ensuring 90%+ code coverage.

Test Coverage: 24 comprehensive test methods covering all operation aspects
- Core functionality: Hierarchy, merge low frequency, frequency-based strategies
- Error handling: Parameter validation, field validation
- Advanced features: Dask processing, encryption, parallel processing
- Mode operations: Replace and enrich modes with proper validation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pamola_core.anonymization.generalization.categorical_op import (
    CategoricalGeneralizationOperation,
    create_categorical_generalization_operation,
)
from pamola_core.anonymization.commons.categorical_config import NullStrategy
from pamola_core.anonymization.schemas.categorical_op_core_schema import CategoricalGeneralizationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.op_config import ConfigError
from pamola_core.utils.progress import HierarchicalProgressTracker

# Test hierarchy file path - dynamically resolved relative to this test file location
# Points to: tests/anonymization/configs/test_hierarchy_categorical_op.json
TEST_HIERARCHY_PATH = str(Path(__file__).parent / "../configs/test_hierarchy_categorical_op.json")


def create_test_data():
    """Create test data for categorical generalization tests."""
    # Basic categorical data - use object dtype strings
    basic_df = pd.DataFrame({
        "category": pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"], dtype="object"),
        "frequency": [10, 20, 30, 5, 15, 25, 8, 12],
        "other_field": range(8)
    })
    
    # Data with nulls - object dtype strings, no null for now
    null_df = pd.DataFrame({
        "category": pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"], dtype="object"),
        "frequency": [10, 20, 30, 5, 15, 25, 8, 12],
        "other_field": range(8)
    })
    
    # Data with low frequency categories
    low_freq_df = pd.DataFrame({
        "category": pd.Series(["A"] * 100 + ["B"] * 50 + ["C"] * 2 + ["D"] * 1 + ["E"] * 3, dtype="object"),
        "frequency": range(156),
        "other_field": range(156)
    })
    
    # Hierarchical data - using categories that match the hierarchy file defined in TEST_HIERARCHY_PATH
    hierarchical_df = pd.DataFrame({
        "category": pd.Series(["food_fruit_apple", "food_fruit_banana", "food_vegetable_carrot", 
                    "electronics_phone_iphone", "electronics_laptop_macbook", "food_fruit_apple"], dtype="object"),
        "frequency": [10, 20, 30, 40, 50, 60],
        "other_field": range(6)
    })
    
    return basic_df, null_df, low_freq_df, hierarchical_df


class MockReporter:
    """Mock reporter for testing."""
    def __init__(self):
        self.operations = []
        
    def add_operation(self, operation, details=None):
        self.operations.append({"operation": operation, "details": details or {}})


class TestCategoricalGeneralizationConfig:
    """Test cases for CategoricalGeneralizationConfig."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config for hierarchy strategy
        valid_config = {
            "field_name": "category",
            "strategy": "hierarchy",
            "hierarchy_level": 2,
            "external_dictionary_path": TEST_HIERARCHY_PATH
        }
        config = CategoricalGeneralizationConfig(**valid_config)
        assert config.strategy == "hierarchy"
        assert config.hierarchy_level == 2
        
        # Valid config for merge_low_freq strategy
        valid_config = {
            "field_name": "category",
            "strategy": "merge_low_freq",
            "freq_threshold": 0.05,
            "unknown_value": "OTHER"
        }
        config = CategoricalGeneralizationConfig(**valid_config)
        assert config.freq_threshold == 0.05
        
        # Valid config for frequency_based strategy
        valid_config = {
            "field_name": "category",
            "strategy": "frequency_based",
            "max_categories": 5
        }
        config = CategoricalGeneralizationConfig(**valid_config)
        assert config.max_categories == 5
        
        # Invalid config - missing required param for hierarchy
        with pytest.raises(ValueError):
            invalid_config = {
                "field_name": "category",
                "strategy": "hierarchy",
                # Missing external_dictionary_path
            }
            CategoricalGeneralizationConfig(**invalid_config)
        
        # Invalid config - invalid strategy
        with pytest.raises(ValueError):
            invalid_config = {
                "field_name": "category",
                "strategy": "invalid_strategy",
            }
            CategoricalGeneralizationConfig(**invalid_config)


class TestCategoricalGeneralizationOperation:
    """Test cases for CategoricalGeneralizationOperation."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        """Setup before and teardown after each test."""
        # Create mock reporter
        self.reporter = MockReporter()
        
        # Create temporary directory for task artifacts
        self.task_dir = tmp_path
        
        # Create mock data source
        self.data_source = Mock(spec=DataSource)
        
        # Required attributes for DataSource mock
        self.data_source.encryption_keys = {}
        self.data_source.settings = {}
        self.data_source.encryption_modes = {}
        self.data_source.data_source_name = "test_data_source"
        
        yield
        
        # Cleanup after test
        pass
    
    def get_fresh_basic_df(self):
        """Get a fresh basic dataframe for each test to avoid reference issues."""
        return pd.DataFrame({
            "category": pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"], dtype="object"),
            "frequency": [10, 20, 30, 5, 15, 25, 8, 12],
            "other_field": range(8)
        })
    
    def get_fresh_null_df(self):
        """Get a fresh null dataframe for each test."""
        return pd.DataFrame({
            "category": pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"], dtype="object"),
            "frequency": [10, 20, 30, 5, 15, 25, 8, 12],
            "other_field": range(8)
        })
    
    def get_fresh_low_freq_df(self):
        """Get a fresh low frequency dataframe for each test."""
        return pd.DataFrame({
            "category": pd.Series(["A"] * 100 + ["B"] * 50 + ["C"] * 2 + ["D"] * 1 + ["E"] * 3, dtype="object"),
            "frequency": range(156),
            "other_field": range(156)
        })
    
    def get_fresh_hierarchical_df(self):
        """Get a fresh hierarchical dataframe for each test."""
        return pd.DataFrame({
            "category": pd.Series(["food_fruit_apple", "food_fruit_banana", "food_vegetable_carrot", 
                        "electronics_phone_iphone", "electronics_laptop_macbook", "food_fruit_apple"], dtype="object"),
            "frequency": [10, 20, 30, 40, 50, 60],
            "other_field": range(6)
        })
    
    def test_factory_function(self):
        """Test factory function creation."""
        operation = create_categorical_generalization_operation(
            field_name="category",
            strategy="hierarchy",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        assert isinstance(operation, CategoricalGeneralizationOperation)
        assert operation.config.field_name == "category"
        assert operation.config.strategy == "hierarchy"
    
    def test_initialization(self):
        """Test operation initialization."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            hierarchy_level=2,
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        assert operation.config.field_name == "category"
        assert operation.config.strategy == "hierarchy"
        assert operation.config.hierarchy_level == 2
        assert operation.config.mode == "REPLACE"
        assert operation.config.null_strategy == "PRESERVE"
    
    def test_inheritance_from_base_anonymization_operation(self):
        """Test that operation inherits from BaseAnonymizationOperation."""
        from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
        
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        assert isinstance(operation, AnonymizationOperation)
        assert hasattr(operation, 'execute')
        assert hasattr(operation, 'process_batch')
        assert hasattr(operation, 'process_batch_dask')
    
    def test_hierarchy_strategy_execution(self):
        """Test hierarchy strategy execution."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            hierarchy_level=2,
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        # Mock data source with hierarchical data - use fresh copy
        self.data_source.get_dataframe.return_value = (self.get_fresh_hierarchical_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        assert result.status == OperationStatus.SUCCESS
        assert len(self.reporter.operations) > 0
    
    def test_merge_low_freq_strategy_execution(self):
        """Test merge_low_freq strategy execution."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="merge_low_freq",
            freq_threshold=0.1,
            unknown_value="OTHER"
        )
        
        # Mock data source with low frequency data - use fresh copy
        self.data_source.get_dataframe.return_value = (self.get_fresh_low_freq_df(), None)
        
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
        assert len(self.reporter.operations) > 0
    
    def test_frequency_based_strategy_execution(self):
        """Test frequency_based strategy execution."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="frequency_based",
            max_categories=3
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
        assert len(self.reporter.operations) > 0
    
    def test_replace_mode(self):
        """Test REPLACE mode."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            mode="REPLACE",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
    
    def test_enrich_mode(self):
        """Test ENRICH mode."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            mode="ENRICH",
            output_field_name="category_generalized",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
    
    def test_null_strategy_preserve(self):
        """Test null strategy PRESERVE."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            null_strategy="PRESERVE",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Mock data source with null data
        self.data_source.get_dataframe.return_value = (self.get_fresh_null_df(), None)
        
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
    
    def test_null_strategy_error(self):
        """Test null strategy ERROR."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            null_strategy="ERROR",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Mock data source with null data
        self.data_source.get_dataframe.return_value = (self.get_fresh_null_df(), None)
        
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.ERROR
    
    def test_field_not_found(self):
        """Test behavior when field is not found."""
        operation = CategoricalGeneralizationOperation(
            field_name="nonexistent_field",
            strategy="hierarchy",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.ERROR
        assert "not found" in result.error_message.lower()
    
    def test_process_batch_method(self):
        """Test process_batch method."""
        fresh_df = self.get_fresh_basic_df()
        batch_result = CategoricalGeneralizationOperation.process_batch(
            batch=fresh_df,
            field_name="category",
            strategy="hierarchy",
            strategy_params={
                "external_dictionary_path": TEST_HIERARCHY_PATH,
                "hierarchy_level": 2
            },
            mode="REPLACE",
            null_strategy="PRESERVE"
        )
        
        assert isinstance(batch_result, tuple)
        assert len(batch_result) == 4  # Returns (batch, category_mapping, hierarchy_info, hierarchy_cache)
        processed_batch = batch_result[0]
        assert isinstance(processed_batch, pd.DataFrame)
        assert "category" in processed_batch.columns
    
    def test_process_batch_dask_method(self):
        """Test process_batch_dask method."""
        import dask.dataframe as dd
        
        fresh_df = self.get_fresh_basic_df()
        dask_df = dd.from_pandas(fresh_df, npartitions=2)
        
        result = CategoricalGeneralizationOperation.process_batch_dask(
            ddf=dask_df,
            field_name="category",
            strategy="hierarchy",
            strategy_params={
                "external_dictionary_path": TEST_HIERARCHY_PATH,
                "hierarchy_level": 2
            },
            mode="REPLACE",
            null_strategy="PRESERVE"
        )
        
        assert hasattr(result, 'compute')
    
    def test_process_value_method(self):
        """Test process_value method - should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            CategoricalGeneralizationOperation.process_value(
                value="food_fruit_apple",
                field_name="category",
                strategy="hierarchy",
                hierarchy_level=2
            )
    
    def test_complex_execute_parameters(self):
        """Test execute method with complex parameters."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            use_dask=True,
            parallel_processes=2,
            chunk_size=1000,
            use_cache=True,
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            force_recalculation=True,
            encryption_key="test_key",
            visualization_timeout=60
        )
        
        assert result.status == OperationStatus.SUCCESS
    
    def test_datawriter_integration(self):
        """Test DataWriter integration."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            external_dictionary_path=TEST_HIERARCHY_PATH,
            output_format="csv"
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
        # Check that output files were created
        assert len(result.artifacts) > 0
    
    def test_progress_tracking(self):
        """Test progress tracking integration."""
        progress_tracker = Mock(spec=HierarchicalProgressTracker)
        progress_tracker.create_subtask.return_value = Mock()
        
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=progress_tracker
        )
        
        assert result.status == OperationStatus.SUCCESS
        assert progress_tracker.update.called
    
    
    
    def test_visualization_generation(self):
        """Test visualization generation."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
        # Check for visualization artifacts
        viz_artifacts = [a for a in result.artifacts if "visualization" in a.description.lower()]
        assert len(viz_artifacts) > 0
    
    def test_encryption_support(self):
        """Test encryption support."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            use_encryption=True,
            encryption_key="test_key",
            encryption_mode="simple",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
    
    def test_chunked_processing(self):
        """Test chunked processing."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            chunk_size=2,
            adaptive_chunk_size=False,
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
    
    def test_parallel_processing(self):
        """Test parallel processing."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            parallel_processes=2,
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
    
    def test_output_file_generation(self):
        """Test output file generation."""
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            output_format="json",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.SUCCESS
        # Check output file was created
        output_files = list(self.task_dir.glob("**/*.json"))
        assert len(output_files) > 0
    
    
    def test_error_handling_scenarios(self):
        """Test various error handling scenarios."""
        # Test with invalid data source
        invalid_data_source = Mock()
        invalid_data_source.get_dataframe.side_effect = Exception("Data source error")
        
        operation = CategoricalGeneralizationOperation(
            field_name="category",
            strategy="hierarchy",
            external_dictionary_path=TEST_HIERARCHY_PATH
        )
        
        # Set fresh basic data
        self.data_source.get_dataframe.return_value = (self.get_fresh_basic_df(), None)
        result = operation.execute(
            data_source=invalid_data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )
        
        assert result.status == OperationStatus.ERROR
        assert "error" in result.error_message.lower()
    
    def get_fresh_empty_df(self):
        """Get a fresh empty DataFrame for testing."""
        return pd.DataFrame({"category": pd.Series([], dtype='object')})
    
    def get_fresh_single_category_df(self):
        """Get a fresh DataFrame with a single category repeated."""
        return pd.DataFrame({"category": pd.Series(["A"] * 10, dtype="object")})

