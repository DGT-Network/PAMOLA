"""
Tests for AttributeSuppressionOperation.

This module contains comprehensive tests for the AttributeSuppressionOperation class,
covering all functionality including basic suppression, conditional suppression,
risk-based suppression, performance optimization, and integration features.

Test Architecture:
- Configuration tests: Parameter validation and initialization
- Core functionality tests: Basic and advanced suppression logic
- Conditional suppression tests: Field-based and risk-based filtering
- Integration tests: Dask, DataWriter, visualization, and encryption
- Performance tests: Memory optimization and parallel processing
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
from pamola_core.anonymization.suppression.attribute_op import AttributeSuppressionOperation
from pamola_core.anonymization.commons.validation.exceptions import FieldNotFoundError
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker


class TestAttributeSuppressionOperation:
    """Test suite for AttributeSuppressionOperation."""

    @pytest.fixture
    def test_config_data(self):
        """Load test configuration data."""
        config_path = Path(__file__).parent.parent / "configs" / "test_config_attribute_op.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def op_config(self):
        """Create a mock OperationConfig."""
        config = Mock(spec=OperationConfig)
        config.get_operation_config.return_value = {
            "field_name": "test_field",
            "mode": "REMOVE",
            "save_suppressed_schema": True
        }
        return config

    @pytest.fixture
    def data_source(self):
        """Create a mock DataSource."""
        ds = Mock(spec=DataSource)
        ds.get_settings.return_value = {}
        ds.get_data_path.return_value = Path("test_data.csv")
        ds.get_format.return_value = "csv"
        ds.get_dataset_config.return_value = {}
        ds.get_connection_config.return_value = {}
        return ds

    @staticmethod
    def get_fresh_suppression_df():
        """Create a fresh DataFrame for attribute suppression testing."""
        return pd.DataFrame({
            'sensitive_field': ['secret1', 'secret2', 'secret3', 'secret4', 'secret5'],
            'primary_field': ['A', 'B', 'C', 'D', 'E'],
            'secondary_field': [1, 2, 3, 4, 5],
            'tertiary_field': [10.1, 20.2, 30.3, 40.4, 50.5],
            'category': ['high_risk', 'low_risk', 'confidential', 'public', 'high_risk'],
            'k_anonymity_score': [2.0, 5.0, 1.5, 8.0, 3.5],
            'status': ['active', 'inactive', 'active', 'active', 'inactive'],
            'risk_level': [0.8, 0.3, 0.9, 0.2, 0.7],
            'score': [0.6, 0.4, 0.75, 0.3, 0.85],
            'large_field': ['large_data1', 'large_data2', 'large_data3', 'large_data4', 'large_data5'],
            'distributed_field': ['dist1', 'dist2', 'dist3', 'dist4', 'dist5'],
            'parallel_field': ['para1', 'para2', 'para3', 'para4', 'para5'],
            'visual_field': ['viz1', 'viz2', 'viz3', 'viz4', 'viz5'],
            'encrypted_field': ['enc1', 'enc2', 'enc3', 'enc4', 'enc5'],
            'cached_field': ['cache1', 'cache2', 'cache3', 'cache4', 'cache5'],
            'numeric_field': [100, 200, 300, 400, 500],
            'minimal_field': ['min1', 'min2', 'min3', 'min4', 'min5']
        })

    # Configuration Tests
    def test_operation_initialization(self, test_config_data):
        """Test operation initialization with various configurations."""
        config = test_config_data["basic_attribute_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.field_name == "sensitive_field"
        assert op.mode == "REMOVE"
        assert op.save_suppressed_schema == True
        assert op.additional_fields == []
        assert op._suppression_count == 0
        assert op._original_column_count == 0

    def test_config_validation(self, test_config_data):
        """Test configuration validation with valid parameters."""
        config = test_config_data["multiple_fields_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.field_name == "primary_field"
        assert op.additional_fields == ["secondary_field", "tertiary_field"]
        assert op.mode == "REMOVE"
        assert op.save_suppressed_schema == True

    def test_invalid_mode(self, test_config_data):
        """Test that invalid mode raises ValueError."""
        config = test_config_data["invalid_mode"]["config"]
        
        with pytest.raises(ValueError) as excinfo:
            AttributeSuppressionOperation(**config)
        
        assert "only supports mode='REMOVE'" in str(excinfo.value)

    def test_config_access_methods(self, test_config_data):
        """Test configuration access methods."""
        config = test_config_data["conditional_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.field_name == "sensitive_field"
        assert op.condition_field == "category"
        assert op.condition_values == ["high_risk", "confidential"]
        assert op.condition_operator == "in"

    # Core Functionality Tests
    def test_basic_suppression(self, test_config_data):
        """Test basic attribute suppression functionality."""
        config = test_config_data["basic_attribute_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            
            # Create a mock data source with get_settings method
            mock_data_source = Mock()
            mock_data_source.get_settings.return_value = {}
            
            # Mock the execute method components
            with patch.object(op, '_handle_preparation') as mock_prep, \
                 patch.object(op, '_load_data_and_validate_input_parameters') as mock_load, \
                 patch.object(op, '_get_cache', return_value=None), \
                 patch.object(op, '_handle_metrics'), \
                 patch.object(op, '_save_output'), \
                 patch.object(op, '_handle_visualizations'), \
                 patch.object(op, '_save_cache'):
                
                # Configure mocks
                mock_prep.return_value = {"task_dir": task_dir}
                df = self.get_fresh_suppression_df()
                mock_load.return_value = (df, True)
                
                result = op.execute(mock_data_source, task_dir)
                
                assert result.status == OperationStatus.SUCCESS
                assert mock_prep.called
                assert mock_load.called

    def test_multiple_fields_suppression(self, test_config_data):
        """Test suppression of multiple fields."""
        config = test_config_data["multiple_fields_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test the core suppression logic
        mask, result = op._process_data(df)
        
        # Verify fields were suppressed
        assert "primary_field" not in result.columns
        assert "secondary_field" not in result.columns  
        assert "tertiary_field" not in result.columns
        
        # Verify other fields remain
        assert "sensitive_field" in result.columns
        assert "category" in result.columns
        
        # Verify suppression count
        assert op._suppression_count == 3

    def test_conditional_suppression(self, test_config_data):
        """Test conditional suppression based on field values."""
        config = test_config_data["conditional_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test the core suppression logic
        mask, result = op._process_data(df)
        
        # Verify that only rows matching condition are in result
        # Based on condition: category in ["high_risk", "confidential"]
        expected_mask = df['category'].isin(["high_risk", "confidential"])
        
        assert len(result) == expected_mask.sum()
        assert "sensitive_field" not in result.columns

    def test_risk_based_suppression(self, test_config_data):
        """Test risk-based suppression using k-anonymity threshold."""
        config = test_config_data["risk_based_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test the core suppression logic  
        mask, result = op._process_data(df)
        
        # Verify that rows with k_anonymity_score < 3.0 are processed
        # identity_field should be suppressed
        assert len(result) > 0
        assert op._suppression_count == 1

    def test_complex_conditional_suppression(self, test_config_data):
        """Test complex conditional suppression with multiple conditions."""
        config = test_config_data["complex_conditional_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test the core suppression logic
        mask, result = op._process_data(df)
        
        # Verify suppression occurred
        assert op._suppression_count == 1
        assert "sensitive_field" not in result.columns

    # Data Validation Tests
    def test_field_not_found_error(self):
        """Test handling of non-existent fields."""
        op = AttributeSuppressionOperation(
            field_name="non_existent_field",
            mode="REMOVE"
        )
        
        df = self.get_fresh_suppression_df()
        
        with pytest.raises(FieldNotFoundError):
            op._process_data(df)

    def test_null_handling_with_schema(self, test_config_data):
        """Test handling of null values in suppressed fields."""
        config = test_config_data["basic_attribute_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Add null values to test field
        df.loc[0, 'sensitive_field'] = None
        df.loc[1, 'sensitive_field'] = pd.NA
        
        # Test the core suppression logic
        mask, result = op._process_data(df)
        
        # Verify suppression occurred despite null values
        assert "sensitive_field" not in result.columns
        assert op._suppression_count == 1

    # Performance Tests
    def test_performance_optimization(self, test_config_data):
        """Test performance optimization features."""
        config = test_config_data["performance_optimized"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.optimize_memory == True
        assert op.adaptive_chunk_size == True
        assert op.chunk_size == 5000
        assert op.save_suppressed_schema == False

    def test_parallel_processing(self, test_config_data):
        """Test parallel processing configuration."""
        config = test_config_data["parallel_processing"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.use_vectorization == True
        assert op.parallel_processes == 2
        assert op.chunk_size == 1000

    # Integration Tests
    def test_dask_processing_integration(self, test_config_data):
        """Test integration with Dask for distributed processing."""
        config = test_config_data["dask_distributed"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.use_dask == True
        assert op.npartitions == 4
        assert op.dask_partition_size == "100MB"
        
        # Test Dask processing logic (mocked)
        with patch('pamola_core.anonymization.suppression.attribute_op.DASK_AVAILABLE', True):
            with patch('pamola_core.anonymization.suppression.attribute_op.dd') as mock_dd:
                mock_ddf = Mock()
                mock_ddf.columns = ['distributed_field', 'other_field']
                mock_ddf.drop.return_value = mock_ddf
                
                result = op._process_batch_dask(mock_ddf)
                
                assert result is not None
                mock_ddf.drop.assert_called_once_with(columns=['distributed_field'])

    def test_advanced_data_writer_integration(self, test_config_data):
        """Test advanced DataWriter integration with multiple formats."""
        config = test_config_data["with_encryption"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.use_encryption == True
        assert op.encryption_mode == "AES"
        assert op.save_output == True
        assert op.output_format == "csv"

    def test_advanced_visualization_generation(self, test_config_data):
        """Test advanced visualization generation capabilities."""
        config = test_config_data["with_visualization"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.generate_visualization == True
        assert op.visualization_backend == "plotly"
        assert op.visualization_theme == "light"

    def test_advanced_caching_support(self, test_config_data):
        """Test advanced caching support."""
        config = test_config_data["cache_enabled"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.use_cache == True
        assert op.force_recalculation == False

    def test_advanced_memory_optimization(self, test_config_data):
        """Test advanced memory optimization features."""
        config = test_config_data["performance_optimized"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test memory optimization doesn't break functionality
        mask, result = op._process_data(df)
        
        assert len(result.columns) < len(df.columns)
        assert op._suppression_count > 0

    # Schema and Metadata Tests
    def test_suppressed_schema_collection(self, test_config_data):
        """Test collection of suppressed schema metadata."""
        config = test_config_data["basic_attribute_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test metadata collection
        op._collect_suppressed_metadata(df, ['sensitive_field'])
        
        assert 'sensitive_field' in op._suppressed_schema
        schema = op._suppressed_schema['sensitive_field']
        assert 'dtype' in schema
        assert 'null_count' in schema
        assert 'unique_count' in schema

    def test_numeric_field_metadata(self, test_config_data):
        """Test metadata collection for numeric fields."""
        config = test_config_data["basic_attribute_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test metadata collection for numeric field
        op._collect_suppressed_metadata(df, ['numeric_field'])
        
        assert 'numeric_field' in op._suppressed_schema
        schema = op._suppressed_schema['numeric_field']
        assert 'min' in schema
        assert 'max' in schema
        assert 'mean' in schema
        assert 'std' in schema

    def test_range_condition_suppression(self, test_config_data):
        """Test range-based conditional suppression."""
        config = test_config_data["range_condition"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test the core suppression logic
        mask, result = op._process_data(df)
        
        # Verify suppression occurred
        assert op._suppression_count == 1
        assert "numeric_field" not in result.columns

    def test_minimal_configuration(self, test_config_data):
        """Test minimal configuration with only required parameters."""
        config = test_config_data["minimal_config"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        assert op.field_name == "minimal_field"
        assert op.mode == "REMOVE"
        assert op.additional_fields == []
        assert op.save_suppressed_schema == True  # Default value

    def test_duplicate_fields_handling(self):
        """Test handling of duplicate fields in suppression list."""
        op = AttributeSuppressionOperation(
            field_name="sensitive_field",
            additional_fields=["sensitive_field", "primary_field", "primary_field"],
            mode="REMOVE"
        )
        
        df = self.get_fresh_suppression_df()
        
        # Test the core suppression logic
        mask, result = op._process_data(df)
        
        # Verify deduplication occurred
        assert op._suppression_count == 2  # Only unique fields counted
        assert "sensitive_field" not in result.columns
        assert "primary_field" not in result.columns

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        op = AttributeSuppressionOperation(
            field_name="sensitive_field",
            mode="REMOVE"
        )
        
        # Create empty DataFrame with columns
        df = pd.DataFrame(columns=['sensitive_field', 'other_field'])
        
        # Test the core suppression logic
        mask, result = op._process_data(df)
        
        # Verify suppression occurred on empty DataFrame
        assert len(result) == 0
        assert "sensitive_field" not in result.columns
        assert "other_field" in result.columns

    def test_string_representation(self, test_config_data):
        """Test string representation of the operation."""
        config = test_config_data["multiple_fields_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        repr_str = repr(op)
        assert "AttributeSuppressionOperation" in repr_str
        assert "primary_field" in repr_str
        assert "secondary_field" in repr_str
        assert "tertiary_field" in repr_str

    def test_total_steps_computation(self, test_config_data):
        """Test computation of total steps for progress tracking."""
        config = test_config_data["with_visualization"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        
        # Test with various configurations
        steps = op._compute_total_steps(
            use_cache=True,
            force_recalculation=False,
            save_output=True,
            generate_visualization=True
        )
        
        assert steps >= 5  # At least preparation, load, process, metrics, and cache steps

    def test_error_handling_in_metadata_collection(self):
        """Test error handling during metadata collection."""
        op = AttributeSuppressionOperation(
            field_name="sensitive_field",
            mode="REMOVE"
        )
        
        df = self.get_fresh_suppression_df()
        
        # Test with problematic column (shouldn't crash)
        with patch('pandas.Series.nunique', side_effect=Exception("Test error")):
            op._collect_suppressed_metadata(df, ['sensitive_field'])
            
            # Should have error recorded in schema
            assert 'sensitive_field' in op._suppressed_schema
            assert 'error' in op._suppressed_schema['sensitive_field']

    def test_build_suppression_mask_method(self, test_config_data):
        """Test the _build_suppression_mask method functionality."""
        config = test_config_data["conditional_suppression"]["config"]
        
        op = AttributeSuppressionOperation(**config)
        df = self.get_fresh_suppression_df()
        
        # Test mask building
        mask = op._build_suppression_mask(df)
        
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert len(mask) == len(df)
        
        # Verify mask matches expected condition
        expected_mask = df['category'].isin(["high_risk", "confidential"])
        # Compare values directly to avoid name/index comparison issues
        assert mask.values.tolist() == expected_mask.values.tolist()
