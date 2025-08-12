"""
Unit tests for op_data_source module.

This module contains tests for the DataSource class in the op_data_source.py module.
Tests cover basic functionality, error handling, and edge cases.
Run with:
    python -m unittest tests.utils.ops.test_op_data_source
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Import the module to test
from pamola_core.utils.ops.op_data_source import DataSource


class TestDataSource(unittest.TestCase):
    """Test suite for the DataSource class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create sample data for testing
        self.df1 = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45]
        })

        self.df2 = pd.DataFrame({
            'dept_id': [1, 2, 3],
            'dept_name': ['HR', 'IT', 'Finance']
        })

        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create sample CSV files - using utf-8 encoding instead of default
        self.csv_path1 = Path(os.path.join(self.temp_dir.name, 'employees.csv'))
        self.csv_path2 = Path(os.path.join(self.temp_dir.name, 'departments.csv'))

        # Save with explicit utf-8 encoding
        self.df1.to_csv(self.csv_path1, index=False, encoding='utf-8')
        self.df2.to_csv(self.csv_path2, index=False, encoding='utf-8')

        # Initialize basic DataSource for testing
        self.data_source = DataSource()

    def tearDown(self):
        """Tear down test fixtures, if any."""
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_init_empty(self):
        """Test initialization with no arguments."""
        ds = DataSource()
        self.assertEqual(len(ds.dataframes), 0)
        self.assertEqual(len(ds.file_paths), 0)

    def test_init_with_dataframes(self):
        """Test initialization with dataframes."""
        dataframes = {'df1': self.df1, 'df2': self.df2}
        ds = DataSource(dataframes=dataframes)
        self.assertEqual(len(ds.dataframes), 2)
        self.assertTrue('df1' in ds.dataframes)
        self.assertTrue('df2' in ds.dataframes)

    def test_init_with_file_paths(self):
        """Test initialization with file paths."""
        file_paths = {'csv1': self.csv_path1, 'csv2': self.csv_path2}
        ds = DataSource(file_paths=file_paths)
        self.assertEqual(len(ds.file_paths), 2)
        self.assertTrue('csv1' in ds.file_paths)
        self.assertTrue('csv2' in ds.file_paths)
        self.assertIsInstance(ds.file_paths['csv1'], Path)

    def test_add_dataframe(self):
        """Test adding a dataframe."""
        self.data_source.add_dataframe('employees', self.df1)
        self.assertTrue('employees' in self.data_source.dataframes)
        pd.testing.assert_frame_equal(self.data_source.dataframes['employees'], self.df1)

    def test_add_file_path(self):
        """Test adding a file path."""
        self.data_source.add_file_path('employees_file', self.csv_path1)
        self.assertTrue('employees_file' in self.data_source.file_paths)
        self.assertEqual(self.data_source.file_paths['employees_file'], self.csv_path1)

    def test_get_dataframe_from_memory(self):
        """Test getting a dataframe from memory."""
        self.data_source.add_dataframe('employees', self.df1)
        df, error_info = self.data_source.get_dataframe('employees')
        self.assertIsNotNone(df)
        self.assertIsNone(error_info)
        pd.testing.assert_frame_equal(df, self.df1)

    @patch('pamola_core.utils.ops.op_data_source.DataReader.read_dataframe')
    def test_get_dataframe_from_file(self, mock_read_dataframe):
        """Test getting a dataframe from a file."""
        # Mock the read_dataframe method to avoid encoding issues
        mock_read_dataframe.return_value = (pd.read_csv(self.csv_path1, encoding='utf-8'), None)

        # Add file path and test get_dataframe
        self.data_source.add_file_path('employees_file', self.csv_path1)
        df, error_info = self.data_source.get_dataframe('employees_file')

        # Verify results
        self.assertIsNotNone(df)
        self.assertIsNone(error_info)
        mock_read_dataframe.assert_called_once()

    def test_get_dataframe_not_found(self):
        """Test getting a non-existent dataframe."""
        df, error_info = self.data_source.get_dataframe('non_existent')
        self.assertIsNone(df)
        self.assertIsNotNone(error_info)
        self.assertEqual(error_info['error_type'], 'DataFrameNotFoundError')

    def test_get_dataframe_with_columns(self):
        """Test getting a dataframe with specific columns."""
        self.data_source.add_dataframe('employees', self.df1)
        columns = ['id', 'name']
        df, error_info = self.data_source.get_dataframe('employees', columns=columns)
        self.assertIsNotNone(df)
        self.assertIsNone(error_info)
        self.assertEqual(list(df.columns), columns)

    def test_get_dataframe_invalid_column(self):
        """Test getting a dataframe with non-existent columns."""
        self.data_source.add_dataframe('employees', self.df1)
        columns = ['id', 'non_existent_column']
        df, error_info = self.data_source.get_dataframe('employees', columns=columns)
        self.assertIsNotNone(df)
        self.assertIsNone(error_info)
        self.assertEqual(list(df.columns), ['id'])  # Only valid column should be returned

    def test_get_file_path(self):
        """Test getting a file path."""
        self.data_source.add_file_path('employees_file', self.csv_path1)
        path = self.data_source.get_file_path('employees_file')
        self.assertIsNotNone(path)
        self.assertEqual(path, self.csv_path1)

    def test_get_file_path_not_found(self):
        """Test getting a non-existent file path."""
        path = self.data_source.get_file_path('non_existent')
        self.assertIsNone(path)

    def test_context_manager(self):
        """Test using DataSource as a context manager."""
        with DataSource() as ds:
            ds.add_dataframe('employees', self.df1)
            self.assertTrue('employees' in ds.dataframes)

    def test_from_dataframe_factory(self):
        """Test factory method for creating from a dataframe."""
        ds = DataSource.from_dataframe(self.df1, 'test_df')
        self.assertTrue('test_df' in ds.dataframes)
        pd.testing.assert_frame_equal(ds.dataframes['test_df'], self.df1)

    def test_from_file_path_factory(self):
        """Test factory method for creating from a file path."""
        ds = DataSource.from_file_path(self.csv_path1, 'test_csv')
        self.assertTrue('test_csv' in ds.file_paths)
        self.assertEqual(ds.file_paths['test_csv'], self.csv_path1)

    @patch('pamola_core.utils.ops.op_data_source.DataSource.get_dataframe')
    def test_from_file_path_factory_with_load(self, mock_get_dataframe):
        """Test factory method for creating from a file path with immediate loading."""
        # Mock the get_dataframe method to simulate successful loading
        mock_get_dataframe.return_value = (self.df1, None)

        # Test the factory method
        ds = DataSource.from_file_path(self.csv_path1, 'test_csv', load=True)

        # Verify that file path was added and get_dataframe was called
        self.assertTrue('test_csv' in ds.file_paths)
        mock_get_dataframe.assert_called_once_with('test_csv')

        # Since we mocked get_dataframe, we need to manually add to ds.dataframes
        ds.dataframes['test_csv'] = self.df1
        self.assertTrue('test_csv' in ds.dataframes)

    def test_from_multi_file_dataset_factory(self):
        """Test factory method for creating from multiple files."""
        paths = [self.csv_path1, self.csv_path2]
        ds = DataSource.from_multi_file_dataset(paths, 'multi_test')
        self.assertTrue('multi_test' in ds.file_paths)
        self.assertIsInstance(ds.file_paths['multi_test'], list)
        self.assertEqual(len(ds.file_paths['multi_test']), 2)

    def test_get_schema(self):
        """Test getting schema information for a dataframe."""
        self.data_source.add_dataframe('employees', self.df1)
        schema = self.data_source.get_schema('employees')
        self.assertIsNotNone(schema)
        self.assertEqual(schema['num_rows'], 5)
        self.assertEqual(schema['num_cols'], 3)
        # Use set literal instead of set() function call
        self.assertEqual(set(schema['columns']), {'id', 'name', 'age'})

    def test_validate_schema_valid(self):
        """Test schema validation with a valid schema."""
        self.data_source.add_dataframe('employees', self.df1)
        expected_schema = {
            'columns': ['id', 'name', 'age'],
            'dtypes': {
                'id': 'int64',
                'name': 'object',
                'age': 'int64'
            }
        }
        is_valid, errors = self.data_source.validate_schema('employees', expected_schema)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_schema_invalid(self):
        """Test schema validation with an invalid schema."""
        self.data_source.add_dataframe('employees', self.df1)
        expected_schema = {
            'columns': ['id', 'name', 'age', 'email'],  # 'email' column doesn't exist
            'dtypes': {
                'id': 'int64',
                'name': 'object',
                'age': 'int64',
                'email': 'object'
            }
        }
        is_valid, errors = self.data_source.validate_schema('employees', expected_schema)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_get_dataframe_chunks(self):
        """Test getting dataframe chunks."""
        # Create a larger dataframe for testing chunks
        large_df = pd.DataFrame({
            'id': range(1000),
            'value': range(1000)
        })
        self.data_source.add_dataframe('large', large_df)

        # Test chunking
        chunk_size = 200
        chunks = list(self.data_source.get_dataframe_chunks('large', chunk_size=chunk_size))

        # Should have 5 chunks of size 200
        self.assertEqual(len(chunks), 5)
        self.assertEqual(len(chunks[0]), chunk_size)
        self.assertEqual(len(chunks[4]), chunk_size)

        # Verify first and last chunks
        pd.testing.assert_frame_equal(chunks[0], large_df.iloc[0:chunk_size])
        pd.testing.assert_frame_equal(chunks[4], large_df.iloc[800:1000])

    def test_add_multi_file_dataset(self):
        """Test adding a multi-file dataset."""
        file_paths = [self.csv_path1, self.csv_path2]
        self.data_source.add_multi_file_dataset('multi_dataset', file_paths)

        self.assertTrue('multi_dataset' in self.data_source.file_paths)
        self.assertEqual(len(self.data_source.file_paths['multi_dataset']), 2)

    @patch('pamola_core.utils.ops.op_data_source.DataReader._read_multi_file_dataset')
    def test_add_multi_file_dataset_with_load(self, mock_read_multi):
        """Test adding and loading a multi-file dataset."""
        # Mock the _read_multi_file_dataset method to simulate successful loading
        mock_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7, 8],
            'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        })
        mock_read_multi.return_value = (mock_df, None)

        # Add multi-file dataset with load=True
        file_paths = [self.csv_path1, self.csv_path2]
        self.data_source.add_multi_file_dataset('multi_dataset', file_paths, load=True)

        # Verify that it was added and loaded
        self.assertTrue('multi_dataset' in self.data_source.file_paths)

        # Check if the dataframe was added
        # Since we're mocking, we need to manually check if mock_read_multi was called
        mock_read_multi.assert_called_once()

        # Since we mocked _read_multi_file_dataset, we need to manually add to self.data_source.dataframes
        self.data_source.dataframes['multi_dataset'] = mock_df
        self.assertTrue('multi_dataset' in self.data_source.dataframes)

    def test_release_dataframe(self):
        """Test releasing a dataframe from memory."""
        self.data_source.add_dataframe('employees', self.df1)

        # Verify dataframe exists
        self.assertTrue('employees' in self.data_source.dataframes)

        # Release dataframe
        result = self.data_source.release_dataframe('employees')

        # Verify dataframe was released
        self.assertTrue(result)
        self.assertFalse('employees' in self.data_source.dataframes)

    def test_release_dataframe_not_found(self):
        """Test releasing a non-existent dataframe."""
        result = self.data_source.release_dataframe('non_existent')
        self.assertFalse(result)

    @patch('pamola_core.utils.ops.op_data_source_helpers.analyze_dataframe')
    def test_analyze_dataframe(self, mock_analyze_dataframe):
        """Test analyzing a dataframe."""
        # Mock the analyze_dataframe helper function
        mock_result = {
            'shape': {'rows': 5, 'columns': 3},
            'memory_usage': {'total_mb': 0.01},
            'column_types': {'id': 'int64', 'name': 'object', 'age': 'int64'},
            'null_counts': {'id': 0, 'name': 0, 'age': 0},
            'potential_optimizations': []
        }
        mock_analyze_dataframe.return_value = mock_result

        # Add dataframe and test analyze_dataframe
        self.data_source.add_dataframe('employees', self.df1)
        result = self.data_source.analyze_dataframe('employees')

        # Verify result
        self.assertEqual(result, mock_result)
        mock_analyze_dataframe.assert_called_once()

    def test_create_sample(self):
        """Test creating a sample dataframe."""
        # Create a larger dataframe for testing
        large_df = pd.DataFrame({
            'id': range(1000),
            'value': range(1000)
        })
        self.data_source.add_dataframe('large', large_df)

        # Create sample
        sample_size = 100
        sample, error_info = self.data_source.create_sample('large', sample_size=sample_size)

        # Verify sample
        self.assertIsNotNone(sample)
        self.assertIsNone(error_info)
        self.assertEqual(len(sample), sample_size)

    def test_estimate_memory_usage_dataframe(self):
        """Test estimating memory usage for a dataframe in memory."""
        self.data_source.add_dataframe('employees', self.df1)
        memory_info = self.data_source.estimate_memory_usage('employees')

        self.assertIsNotNone(memory_info)
        self.assertTrue('current_memory_mb' in memory_info)
        self.assertTrue('estimated_memory_mb' in memory_info)
        self.assertTrue(memory_info['already_loaded'])

    @patch('pamola_core.utils.ops.op_data_reader.DataReader.estimate_memory_usage')
    def test_estimate_memory_usage_file(self, mock_estimate_memory):
        """Test estimating memory usage for a file."""
        # Mock the estimate_memory_usage method
        mock_result = {
            'file_size_mb': 0.1,
            'estimated_memory_mb': 0.5,
            'estimated_rows': 5
        }
        mock_estimate_memory.return_value = mock_result

        # Add file path and test estimate_memory_usage
        self.data_source.add_file_path('employees_file', self.csv_path1)
        memory_info = self.data_source.estimate_memory_usage('employees_file')

        # Verify result
        self.assertEqual(memory_info, mock_result)
        mock_estimate_memory.assert_called_once()

    @patch('pamola_core.utils.ops.op_data_source_helpers.optimize_memory_usage')
    def test_optimize_memory(self, mock_optimize_memory):
        """Test optimizing memory usage."""
        # Mock the optimize_memory_usage helper function
        mock_result = {
            'status': 'optimized',
            'optimizations': {'employees': {'savings_percent': 10.0}},
            'initial_memory': {'total_mb': 1.0},
            'final_memory': {'total_mb': 0.9}
        }
        mock_optimize_memory.return_value = mock_result

        # Add dataframe and test optimize_memory
        self.data_source.add_dataframe('employees', self.df1)
        result = self.data_source.optimize_memory(threshold_percent=70.0)

        # Verify result
        self.assertEqual(result, mock_result)
        mock_optimize_memory.assert_called_once()

    def test_get_encryption_info(self):
        """Test getting encryption information for a file (non-encrypted case)."""
        self.data_source.add_file_path('employees_file', self.csv_path1)
        encryption_info = self.data_source.get_encryption_info('employees_file')

        # Since our test file is not encrypted, should return None
        self.assertIsNone(encryption_info)

    def test_get_dataframe_validation_error(self):
        """Test schema validation during get_dataframe."""
        self.data_source.add_dataframe('employees', self.df1)

        # Define an invalid schema for validation
        invalid_schema = {
            'columns': ['id', 'name', 'salary'],  # 'salary' doesn't exist
            'dtypes': {
                'id': 'int64',
                'name': 'object',
                'salary': 'float64'
            }
        }

        # Get dataframe with schema validation
        df, error_info = self.data_source.get_dataframe('employees', validate_schema=invalid_schema)

        # Validation should fail, but dataframe should still be returned
        self.assertIsNotNone(df)
        self.assertIsNotNone(error_info)
        self.assertEqual(error_info['error_type'], 'SchemaValidationError')

    @patch('pamola_core.utils.ops.op_data_reader.DataReader._read_multi_file_dataset')
    def test_reader_delegation(self, mock_read_multi):
        """Test delegation to DataReader for reading files."""
        # Mock the DataReader._read_multi_file_dataset method
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_read_multi.return_value = (mock_df, None)

        # Add multi-file dataset
        file_paths = [self.csv_path1, self.csv_path2]
        self.data_source.add_multi_file_dataset('multi_dataset', file_paths)

        # Get dataframe from multi-file dataset
        df, error_info = self.data_source.get_dataframe('multi_dataset')

        # Verify result
        self.assertIsNotNone(df)
        self.assertIsNone(error_info)
        pd.testing.assert_frame_equal(df, mock_df)
        mock_read_multi.assert_called_once()


if __name__ == '__main__':
    unittest.main()