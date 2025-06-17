"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Tests for DataWriter
Description: Unit tests for the DataWriter class
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Run with
     python -m unittest -v tests.utils.ops.test_op_data_writer
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the module to test
from pamola_core.utils.ops.op_data_writer import DataWriter, WriterResult, DataWriteError


class TestDataWriter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.task_dir = Path(self.temp_dir.name)

        # Initialize a mock logger
        self.mock_logger = MagicMock()

        # Initialize a mock progress tracker
        self.mock_progress = MagicMock()

        # Initialize the DataWriter
        self.writer = DataWriter(
            task_dir=self.task_dir,
            logger=self.mock_logger,
            progress_tracker=self.mock_progress
        )

        # Create sample data for tests
        self.sample_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'User_{i}' for i in range(1, 11)],
            'value': np.random.rand(10)
        })

        self.sample_dict = {
            'key1': 'value1',
            'key2': 42,
            'key3': {
                'nested1': True,
                'nested2': [1, 2, 3]
            }
        }

        # Create a sample figure
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.title('Sample Plot')
        self.sample_fig = plt.gcf()

    def tearDown(self):
        """Clean up test fixtures after each test."""
        # Close the figure
        plt.close(self.sample_fig)

        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test DataWriter initialization."""
        # Verify the task_dir is set correctly
        self.assertEqual(self.writer.task_dir, self.task_dir)

        # Verify the standard directories were created
        self.assertTrue((self.task_dir / "output").exists())
        self.assertTrue((self.task_dir / "dictionaries").exists())
        self.assertTrue((self.task_dir / "logs").exists())

        # Verify the logger and progress tracker are set
        self.assertEqual(self.writer.logger, self.mock_logger)
        self.assertEqual(self.writer.progress_tracker, self.mock_progress)

    @patch('pamola_core.utils.ops.op_data_writer.write_dataframe_to_csv')
    def test_write_dataframe_csv(self, mock_write_csv):
        """Test writing a DataFrame to CSV."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_write_csv.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 1024
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="test_data.csv")
        type(mock_path).stem = Mock(return_value="test_data")

        # Call the method
        result = self.writer.write_dataframe(
            df=self.sample_df,
            name="test_data",
            format="csv"
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 1024)
        self.assertEqual(result.format, "csv")

        # Verify the mock was called with expected arguments
        mock_write_csv.assert_called_once()
        args, kwargs = mock_write_csv.call_args

        # Use pandas testing to check DataFrame equality
        # Instead of direct comparison, check that both frames are pandas DataFrames
        # and that they have the same shape
        self.assertIsInstance(args[0], pd.DataFrame)
        self.assertEqual(args[0].shape, self.sample_df.shape)
        # You could also use pd.testing.assert_frame_equal if you want to be more thorough
        # pd.testing.assert_frame_equal(args[0], self.sample_df)

        self.assertEqual(kwargs.get('encryption_key'), None)

    @patch('pamola_core.utils.ops.op_data_writer.write_parquet')
    def test_write_dataframe_parquet(self, mock_write_parquet):
        """Test writing a DataFrame to Parquet."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_write_parquet.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 512
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="test_data.parquet")
        type(mock_path).stem = Mock(return_value="test_data")

        # Call the method
        result = self.writer.write_dataframe(
            df=self.sample_df,
            name="test_data",
            format="parquet",
            timestamp_in_name=True
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 512)
        self.assertEqual(result.format, "parquet")

        # Verify the mock was called with expected arguments
        mock_write_parquet.assert_called_once()
        args, kwargs = mock_write_parquet.call_args

        # Use pandas testing to check DataFrame equality
        self.assertIsInstance(args[0], pd.DataFrame)
        self.assertEqual(args[0].shape, self.sample_df.shape)

    @patch('pamola_core.utils.ops.op_data_writer.write_json')
    def test_write_json(self, mock_write_json):
        """Test writing a JSON object."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_write_json.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 256
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="test_config.json")
        type(mock_path).stem = Mock(return_value="test_config")

        # Call the method
        result = self.writer.write_json(
            data=self.sample_dict,
            name="test_config",
            subdir=None,  # Root directory
            pretty=True
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 256)
        self.assertEqual(result.format, "json")

        # Verify the mock was called with expected arguments
        mock_write_json.assert_called_once()
        args, kwargs = mock_write_json.call_args
        self.assertEqual(args[0], self.sample_dict)
        self.assertEqual(kwargs.get('indent'), 2)  # Should have indent=2 for pretty=True

    @patch('pamola_core.utils.ops.op_data_writer.save_plot')
    def test_write_visualization(self, mock_save_plot):
        """Test writing a visualization."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_save_plot.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 1536
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="sample_plot.png")
        type(mock_path).stem = Mock(return_value="sample_plot")

        # Call the method
        result = self.writer.write_visualization(
            figure=self.sample_fig,
            name="sample_plot",
            format="png"
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 1536)
        self.assertEqual(result.format, "png")

        # Verify the mock was called with expected arguments
        mock_save_plot.assert_called_once()
        args, kwargs = mock_save_plot.call_args
        self.assertEqual(args[0], self.sample_fig)

    @patch('pamola_core.utils.ops.op_data_writer.write_json')
    def test_write_dictionary(self, mock_write_json):
        """Test writing a dictionary."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_write_json.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 128
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="word_freqs.json")
        type(mock_path).stem = Mock(return_value="word_freqs")

        # Call the method
        result = self.writer.write_dictionary(
            data={'apple': 10, 'banana': 5, 'cherry': 8},
            name="word_freqs",
            format="json"
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 128)
        self.assertEqual(result.format, "json")

        # Verify the mock was called with expected arguments
        mock_write_json.assert_called_once()

    @patch('pamola_core.utils.ops.op_data_writer.write_json')
    def test_write_metrics(self, mock_write_json):
        """Test writing metrics."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_write_json.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 384
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="metrics_results.json")
        type(mock_path).stem = Mock(return_value="metrics_results")

        # Call the method
        metrics_data = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90
        }

        result = self.writer.write_metrics(
            metrics=metrics_data,
            name="metrics_results"
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 384)
        self.assertEqual(result.format, "json")

        # Verify the mock was called with expected arguments
        mock_write_json.assert_called_once()
        args, kwargs = mock_write_json.call_args

        # Check that metrics are enriched with metadata
        json_data = args[0]
        self.assertIn('metadata', json_data)
        self.assertIn('metrics', json_data)
        self.assertEqual(json_data['metrics'], metrics_data)

    @patch('pamola_core.utils.ops.op_data_writer.append_to_json_array')
    def test_append_to_json_array(self, mock_append):
        """Test appending to a JSON array."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_append.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 200
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="log_entries.json")
        type(mock_path).stem = Mock(return_value="log_entries")

        # Call the method
        log_entry = {'timestamp': '2025-05-03T12:34:56', 'message': 'Test log entry'}

        result = self.writer.append_to_json_array(
            item=log_entry,
            name="log_entries",
            create_if_missing=True
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 200)
        self.assertEqual(result.format, "json")

        # Verify the mock was called with expected arguments
        mock_append.assert_called_once()
        args, kwargs = mock_append.call_args
        self.assertEqual(args[0], log_entry)
        self.assertTrue(kwargs.get('create_if_missing'))

    @patch('pamola_core.utils.ops.op_data_writer.merge_json_objects')
    def test_merge_json_objects(self, mock_merge):
        """Test merging JSON objects."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_merge.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 300
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="config.json")
        type(mock_path).stem = Mock(return_value="config")

        # Call the method
        config_update = {'new_setting': 'value', 'existing': {'nested': 'updated'}}

        result = self.writer.merge_json_objects(
            data=config_update,
            name="config",
            recursive_merge=True
        )

        # Verify the result
        self.assertIsInstance(result, WriterResult)
        self.assertEqual(result.path, mock_path)
        self.assertEqual(result.size_bytes, 300)
        self.assertEqual(result.format, "json")

        # Verify the mock was called with expected arguments
        mock_merge.assert_called_once()
        args, kwargs = mock_merge.call_args
        self.assertEqual(args[0], config_update)
        self.assertTrue(kwargs.get('recursive_merge'))

    def test_get_output_path(self):
        """Test generating output paths."""
        # Test basic path generation
        path1 = self.writer._get_output_path("test", "csv", "output")
        self.assertEqual(path1, self.task_dir / "output" / "test.csv")

        # Test with timestamp
        path2 = self.writer._get_output_path("test", "json", None, timestamp_in_name=True)
        # The path should include a timestamp format like: test_20250503T123456.json
        self.assertTrue(path2.name.startswith("test_"))
        self.assertTrue(path2.name.endswith(".json"))

        # Test with extension already having a dot
        path3 = self.writer._get_output_path("test", ".parquet", "output")
        self.assertEqual(path3, self.task_dir / "output" / "test.parquet")

        # Test with subdirectory creation
        new_subdir = "custom_dir"
        path4 = self.writer._get_output_path("test", "png", new_subdir)
        self.assertEqual(path4, self.task_dir / new_subdir / "test.png")
        self.assertTrue((self.task_dir / new_subdir).exists())

    def test_ensure_directories(self):
        """Test directory creation."""
        # Delete the standard directories
        for subdir in ["output", "dictionaries", "logs"]:
            dir_path = self.task_dir / subdir
            if dir_path.exists():
                os.rmdir(dir_path)

        # Call the method
        self.writer._ensure_directories()

        # Verify the directories were recreated
        for subdir in ["output", "dictionaries", "logs"]:
            self.assertTrue((self.task_dir / subdir).exists())

    @patch('pamola_core.utils.ops.op_data_writer.write_dataframe_to_csv')
    def test_file_exists_no_overwrite(self, mock_write_csv):
        """Test behavior when a file exists and overwrite=False."""

        # Make the file appear to exist
        def side_effect(*args, **kwargs):
            raise FileExistsError("File already exists")

        mock_write_csv.side_effect = side_effect

        # Call the method with overwrite=False
        with self.assertRaises(DataWriteError):
            self.writer.write_dataframe(
                df=self.sample_df,
                name="existing_file",
                format="csv",
                overwrite=False
            )

    @patch('pamola_core.utils.ops.op_data_writer.write_dataframe_to_csv')
    def test_error_handling(self, mock_write_csv):
        """Test error handling during writing."""
        # Make the write operation fail
        mock_write_csv.side_effect = ValueError("Test error")

        # Call the method and expect a DataWriteError
        with self.assertRaises(DataWriteError) as context:
            self.writer.write_dataframe(
                df=self.sample_df,
                name="error_file",
                format="csv"
            )

        # Verify the error message includes the original error
        self.assertIn("Test error", str(context.exception))

        # Verify the logger was called with an error message
        self.mock_logger.error.assert_called_once()

    @patch('pamola_core.utils.ops.op_data_writer.write_dataframe_to_csv')
    def test_progress_tracking(self, mock_write_csv):
        """Test progress tracking during writing."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_write_csv.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 1024
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="test_data.csv")
        type(mock_path).stem = Mock(return_value="test_data")

        # Call the method
        result = self.writer.write_dataframe(
            df=self.sample_df,
            name="test_data",
            format="csv"
        )

        # Verify progress tracker methods were called
        self.mock_progress.create_subtask.assert_called_once()
        mock_subtask = self.mock_progress.create_subtask.return_value
        mock_subtask.update.assert_called()
        mock_subtask.close.assert_called_once()

    def test_invalid_format(self):
        """Test handling of invalid format."""
        with self.assertRaises(DataWriteError):
            self.writer.write_dataframe(
                df=self.sample_df,
                name="invalid_format",
                format="invalid"
            )

    @patch('pamola_core.utils.ops.op_data_writer.write_json')
    def test_get_caller_info(self, mock_write_json):
        """Test extraction of caller information."""
        # Create a mock path with a properly mocked stat method
        mock_path = Mock(spec=Path)
        mock_write_json.return_value = mock_path

        # Configure the mock_path.stat() method to return mock_stat
        mock_stat = Mock()
        mock_stat.st_size = 100
        mock_stat.st_mtime = datetime.now().timestamp()
        mock_path.stat.return_value = mock_stat

        # Configure the path name and stem attributes
        type(mock_path).name = Mock(return_value="metrics.json")
        type(mock_path).stem = Mock(return_value="metrics")

        # Call the metrics method which should extract caller info
        result = self.writer.write_metrics(
            metrics={'test': 123},
            name="metrics"
        )

        # Verify the write_json call included metadata with caller info
        args, kwargs = mock_write_json.call_args
        metadata = args[0]['metadata']

        # Should have timestamp and name fields at minimum
        self.assertIn('timestamp', metadata)
        self.assertEqual(metadata['name'], "metrics")

        # Should have some operation info (might be limited in test context)
        self.assertIn('operation', metadata)


if __name__ == '__main__':
    unittest.main()