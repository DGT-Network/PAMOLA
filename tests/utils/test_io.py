"""
Unit tests for the pamola_core/utils/io.py module.

This test suite covers the main functionality of the IO module including:
- File and directory operations
- CSV reading and writing with various options
- Data format conversions and transformations
- Encryption/decryption integration
- Memory optimization features
- Multi-file handling
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import pandas as pd
import numpy as np
import json

# Import module under test
from pamola_core.utils import io
from pamola_core.utils.io_helpers import error_utils


class TestIODirectoryManagement(unittest.TestCase):
    """Test directory management functions in the IO module."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_subdir = os.path.join(self.temp_dir, "test_subdir")

    def tearDown(self):
        """Clean up temporary files after tests."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_directory(self):
        """Test creating a directory if it doesn't exist."""
        # Test creating a new directory
        result = io.ensure_directory(self.test_subdir)
        self.assertTrue(os.path.exists(self.test_subdir))
        self.assertTrue(os.path.isdir(self.test_subdir))
        self.assertEqual(result, Path(self.test_subdir))

        # Test with an existing directory (should not raise error)
        result2 = io.ensure_directory(self.test_subdir)
        self.assertEqual(result2, Path(self.test_subdir))

    def test_get_timestamped_filename(self):
        """Test generating timestamped filenames."""
        # Test with timestamp
        filename = io.get_timestamped_filename("test", "csv", True)
        self.assertTrue(filename.startswith("test_"))
        self.assertTrue(filename.endswith(".csv"))

        # Test without timestamp
        filename = io.get_timestamped_filename("test", "json", False)
        self.assertEqual(filename, "test.json")

    def test_list_directory_contents(self):
        """Test listing directory contents with various patterns."""
        # Create test files
        os.makedirs(self.test_subdir)
        test_files = [
            os.path.join(self.temp_dir, "file1.txt"),
            os.path.join(self.temp_dir, "file2.csv"),
            os.path.join(self.test_subdir, "file3.txt")
        ]
        for file_path in test_files:
            with open(file_path, 'w') as f:
                f.write("test content")

        # Test non-recursive listing
        files = io.list_directory_contents(self.temp_dir, "*.txt", False)
        self.assertEqual(len(files), 1)
        self.assertEqual(os.path.basename(str(files[0])), "file1.txt")

        # Test recursive listing
        files = io.list_directory_contents(self.temp_dir, "*.txt", True)
        self.assertEqual(len(files), 2)

    def test_clear_directory(self):
        """Test clearing directory contents."""
        # Create test files
        test_files = [
            os.path.join(self.temp_dir, "file1.txt"),
            os.path.join(self.temp_dir, "file2.csv")
        ]
        for file_path in test_files:
            with open(file_path, 'w') as f:
                f.write("test content")

        # Test clearing without confirm prompt
        count = io.clear_directory(self.temp_dir, confirm=False)
        self.assertEqual(count, 2)
        self.assertEqual(len(os.listdir(self.temp_dir)), 0)


class TestIOCSVFunctions(unittest.TestCase):
    """Test CSV reading and writing functions."""

    def setUp(self):
        """Set up test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.temp_dir, "test.csv")

        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'id': range(1, 101),
            'name': [f'Item {i}' for i in range(1, 101)],
            'value': np.random.rand(100)
        })

        # Write test CSV file
        self.test_df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        """Clean up temporary files after tests."""
        shutil.rmtree(self.temp_dir)

    def test_read_full_csv(self):
        """Test reading a full CSV file into DataFrame."""
        df = io.read_full_csv(self.test_csv, encoding='utf-8')
        self.assertEqual(len(df), 100)
        self.assertEqual(list(df.columns), ['id', 'name', 'value'])

        # Test with columns parameter
        df = io.read_full_csv(self.test_csv, encoding='utf-8', columns=['id', 'name'])
        self.assertEqual(list(df.columns), ['id', 'name'])

        # Test with nrows parameter
        df = io.read_full_csv(self.test_csv, encoding='utf-8', nrows=10)
        self.assertEqual(len(df), 10)

        # Test with skiprows parameter
        df = io.read_full_csv(self.test_csv, encoding='utf-8', skiprows=5)
        self.assertEqual(len(df), 95)

    def test_read_csv_in_chunks(self):
        """Test reading a CSV file in chunks."""
        chunks = list(io.read_csv_in_chunks(self.test_csv, chunk_size=20, encoding='utf-8'))
        self.assertEqual(len(chunks), 5)  # 100 rows / 20 rows per chunk = 5 chunks

        # Verify total rows read equals original DataFrame rows
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, 100)

    def test_write_dataframe_to_csv(self):
        """Test writing a DataFrame to CSV."""
        output_csv = os.path.join(self.temp_dir, "output.csv")
        result = io.write_dataframe_to_csv(self.test_df, output_csv, encoding='utf-8')

        # Verify file was created
        self.assertTrue(os.path.exists(output_csv))

        # Read back and verify content
        df_read = pd.read_csv(output_csv)
        self.assertEqual(len(df_read), 100)

        # Verify result is the path
        self.assertEqual(result, Path(output_csv))

    def test_write_chunks_to_csv(self):
        """Test writing chunks to a CSV file."""
        # Split DataFrame into chunks
        chunk_size = 25
        # Create chunks as iterator of dataframes
        chunks = iter([self.test_df.iloc[i:i + chunk_size] for i in range(0, len(self.test_df), chunk_size)])

        output_csv = os.path.join(self.temp_dir, "chunks.csv")
        result = io.write_chunks_to_csv(chunks, output_csv, encoding='utf-8')

        # Verify file was created and contains all data
        df_read = pd.read_csv(output_csv)
        self.assertEqual(len(df_read), 100)

        # Verify result is the path
        self.assertEqual(result, Path(output_csv))


class TestIOEncryptionSupport(unittest.TestCase):
    """Test encryption and decryption integration in IO operations."""

    def setUp(self):
        """Set up test data and mock crypto functions."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.temp_dir, "test.csv")
        self.test_encrypted = os.path.join(self.temp_dir, "test.enc")

        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'Item {i}' for i in range(1, 11)],
            'value': np.random.rand(10)
        })

        # Write test CSV file
        self.test_df.to_csv(self.test_csv, index=False)

        # Also create encrypted file for testing
        shutil.copy(self.test_csv, self.test_encrypted)

        # Set up mocks
        patcher1 = mock.patch('pamola_core.utils.io_helpers.crypto_utils.decrypt_file')
        patcher2 = mock.patch('pamola_core.utils.io_helpers.crypto_utils.encrypt_file')
        patcher3 = mock.patch('pamola_core.utils.io_helpers.directory_utils.get_temp_file_for_decryption')
        patcher4 = mock.patch('pamola_core.utils.io_helpers.directory_utils.get_temp_file_for_encryption')
        patcher5 = mock.patch('pamola_core.utils.io_helpers.directory_utils.safe_remove_temp_file')

        self.mock_decrypt = patcher1.start()
        self.mock_encrypt = patcher2.start()
        self.mock_get_temp_decrypt = patcher3.start()
        self.mock_get_temp_encrypt = patcher4.start()
        self.mock_safe_remove = patcher5.start()

        # Configure mocks
        self.mock_decrypt.side_effect = lambda source_path, destination_path, key: shutil.copy(source_path,
                                                                                               destination_path)
        self.mock_encrypt.side_effect = lambda source_path, destination_path, key: shutil.copy(source_path,
                                                                                               destination_path)
        self.mock_get_temp_decrypt.return_value = os.path.join(self.temp_dir, "temp_decrypted.csv")
        self.mock_get_temp_encrypt.return_value = os.path.join(self.temp_dir, "temp_encrypted.csv")

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)
        self.addCleanup(patcher4.stop)
        self.addCleanup(patcher5.stop)

    def tearDown(self):
        """Clean up temporary files after tests."""
        shutil.rmtree(self.temp_dir)

    def test_read_with_encryption(self):
        """Test reading an encrypted CSV file."""
        # Test reading with encryption key
        df = io.read_full_csv(
            self.test_encrypted,
            encoding='utf-8',
            encryption_key="test_key"
        )

        # Verify crypto_utils.decrypt_file was called
        self.mock_decrypt.assert_called()

        # Verify DataFrame was correctly loaded
        self.assertEqual(len(df), 10)
        self.assertEqual(list(df.columns), ['id', 'name', 'value'])

    def test_write_with_encryption(self):
        """Test writing a DataFrame to an encrypted CSV file."""
        output_csv = os.path.join(self.temp_dir, "encrypted_output.csv")

        # Write with encryption key
        result = io.write_dataframe_to_csv(
            self.test_df,
            output_csv,
            encoding='utf-8',
            encryption_key="test_key"
        )

        # Verify crypto_utils.encrypt_file was called
        self.mock_encrypt.assert_called()

        # Verify result is the path
        self.assertEqual(result, Path(output_csv))

        # Verify file exists
        self.assertTrue(os.path.exists(output_csv))


class TestIOMultiFileSupport(unittest.TestCase):
    """Test multi-file dataset handling."""

    def setUp(self):
        """Set up test data with multiple CSV files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create multiple test CSV files
        self.csv_files = []
        for i in range(3):
            df = pd.DataFrame({
                'id': range(1, 11),
                'file_num': [i + 1] * 10,
                'value': np.random.rand(10)
            })
            file_path = os.path.join(self.temp_dir, f"file{i + 1}.csv")
            df.to_csv(file_path, index=False)
            self.csv_files.append(file_path)

    def tearDown(self):
        """Clean up temporary files after tests."""
        shutil.rmtree(self.temp_dir)

    @mock.patch('pamola_core.utils.io.multi_file_utils')
    def test_read_multi_csv(self, mock_multi_file_utils):
        """Test reading multiple CSV files and combining them."""
        # Set up the mock to return a combined DataFrame
        combined_df = pd.DataFrame({
            'id': list(range(1, 11)) * 3,
            'file_num': [1] * 10 + [2] * 10 + [3] * 10,
            'value': np.random.rand(30)
        })
        mock_multi_file_utils.read_multi_csv.return_value = combined_df

        # Call the function
        result = io.read_multi_csv(
            self.csv_files,
            encoding='utf-8',
            columns=['id', 'file_num']
        )

        # Verify multi_file_utils.read_multi_csv was called with correct args
        mock_multi_file_utils.read_multi_csv.assert_called_once()
        # Check the arguments passed
        call_args = mock_multi_file_utils.read_multi_csv.call_args
        self.assertEqual(call_args[1]['file_paths'], self.csv_files)
        self.assertEqual(call_args[1]['encoding'], 'utf-8')
        self.assertEqual(call_args[1]['columns'], ['id', 'file_num'])

        # Verify result is the combined DataFrame
        self.assertIs(result, combined_df)

    @mock.patch('pamola_core.utils.io.multi_file_utils')
    def test_read_similar_files(self, mock_multi_file_utils):
        """Test reading similar files from a directory."""
        # Set up the mock to return a combined DataFrame
        combined_df = pd.DataFrame({
            'id': list(range(1, 11)) * 3,
            'file_num': [1] * 10 + [2] * 10 + [3] * 10,
            'value': np.random.rand(30)
        })
        mock_multi_file_utils.read_similar_files.return_value = combined_df

        # Call the function
        result = io.read_similar_files(
            self.temp_dir,
            pattern="*.csv",
            recursive=False
        )

        # Verify multi_file_utils.read_similar_files was called with correct args
        mock_multi_file_utils.read_similar_files.assert_called_once()
        # Check the arguments passed
        call_args = mock_multi_file_utils.read_similar_files.call_args
        self.assertEqual(call_args[1]['directory'], self.temp_dir)
        self.assertEqual(call_args[1]['pattern'], "*.csv")
        self.assertEqual(call_args[1]['recursive'], False)

        # Verify result is the combined DataFrame
        self.assertIs(result, combined_df)


class TestIOOtherFormats(unittest.TestCase):
    """Test reading and writing other file formats."""

    def setUp(self):
        """Set up test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'Item {i}' for i in range(1, 11)],
            'value': np.random.rand(10)
        })

        # Create test JSON file
        self.test_json_path = os.path.join(self.temp_dir, "test.json")
        self.test_json_data = {
            "name": "Test Object",
            "values": [1, 2, 3, 4, 5],
            "metadata": {
                "author": "Test Author",
                "version": 1.0
            }
        }
        with open(self.test_json_path, 'w') as f:
            json.dump(self.test_json_data, f) # type: ignore

    def tearDown(self):
        """Clean up temporary files after tests."""
        shutil.rmtree(self.temp_dir)

    @mock.patch('pamola_core.utils.io_helpers.format_utils.check_pyarrow_available')
    @mock.patch('pathlib.Path.exists')
    @mock.patch('pandas.read_parquet')
    def test_parquet_functions(self, mock_read_parquet, mock_path_exists, mock_check_pyarrow):
        """Test Parquet reading and writing."""
        # Setup mocks
        mock_check_pyarrow.return_value = None
        mock_path_exists.return_value = True  # Mock Path.exists() to always return True
        mock_read_parquet.return_value = self.test_df

        # Test reading Parquet
        test_parquet = os.path.join(self.temp_dir, "test.parquet")
        df = io.read_parquet(test_parquet, columns=['id', 'name'])

        # Verify pandas.read_parquet was called correctly
        mock_read_parquet.assert_called_once()

        # Test writing Parquet
        with mock.patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            output_parquet = os.path.join(self.temp_dir, "output.parquet")
            result = io.write_parquet(
                self.test_df,
                output_parquet,
                compression="snappy"
            )

            # Verify DataFrame.to_parquet was called correctly
            mock_to_parquet.assert_called_once()

    def test_json_functions(self):
        """Test JSON reading and writing."""
        # Test reading JSON
        data = io.read_json(self.test_json_path)
        self.assertEqual(data['name'], "Test Object")
        self.assertEqual(data['metadata']['author'], "Test Author")

        # Test writing JSON
        output_json = os.path.join(self.temp_dir, "output.json")
        result = io.write_json(data, output_json)

        # Verify file was created
        self.assertTrue(os.path.exists(output_json))

        # Read back and verify
        with open(output_json, 'r') as f:
            written_data = json.load(f)
        self.assertEqual(written_data['name'], "Test Object")


class TestIOMemoryManagement(unittest.TestCase):
    """Test memory management functions."""

    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.temp_dir, "test.csv")

        # Create test DataFrame with various types
        self.test_df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.rand(1000),
            'str_col': [f'Item {i}' for i in range(1000)]
        })

        # Write test CSV file
        self.test_df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        """Clean up temporary files after tests."""
        shutil.rmtree(self.temp_dir)

    @mock.patch('pamola_core.utils.io_helpers.memory_utils.estimate_file_memory')
    def test_estimate_file_memory(self, mock_estimate):
        """Test estimating memory requirements for loading a file."""
        # Set up mock to return test data
        mock_estimate.return_value = {
            "file_size_bytes": 10000,
            "file_size_mb": 0.01,
            "estimated_memory_mb": 0.05,
            "memory_factor": 5.0,
            "file_type": "csv"
        }

        # Call the function
        result = io.estimate_file_memory(self.test_csv)

        # Verify memory_utils.estimate_file_memory was called correctly
        mock_estimate.assert_called_once_with(self.test_csv)

        # Verify result contains expected keys
        self.assertEqual(result["file_type"], "csv")
        self.assertEqual(result["memory_factor"], 5.0)

    @mock.patch('pamola_core.utils.io_helpers.memory_utils.optimize_dataframe_memory')
    def test_optimize_dataframe_memory(self, mock_optimize):
        """Test optimizing memory usage of a DataFrame."""
        # Set up mock to return optimized DataFrame and stats
        optimized_df = self.test_df.copy()
        optimization_info = {
            "before_bytes": 100000,
            "after_bytes": 50000,
            "savings_percent": 50.0,
            "conversions": {
                "int_col": "int32 -> int8",
                "float_col": "float64 -> float32"
            }
        }
        mock_optimize.return_value = (optimized_df, optimization_info)

        # Call the function
        df, info = io.optimize_dataframe_memory(
            self.test_df,
            categorical_threshold=0.5
        )

        # Verify memory_utils.optimize_dataframe_memory was called correctly
        mock_optimize.assert_called_once()
        args, kwargs = mock_optimize.call_args
        self.assertEqual(kwargs['categorical_threshold'], 0.5)

        # Verify results
        self.assertIs(df, optimized_df)
        self.assertEqual(info["savings_percent"], 50.0)


class TestIOErrorHandling(unittest.TestCase):
    """Test error handling behavior."""

    def setUp(self):
        """Set up test paths."""
        self.temp_dir = tempfile.mkdtemp()
        self.nonexistent_file = os.path.join(self.temp_dir, "does_not_exist.csv")

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_file_not_found_handling(self):
        """Test handling of FileNotFoundError."""
        # Attempt to read non-existent file, expect exception
        with self.assertRaises(FileNotFoundError):
            io.read_full_csv(self.nonexistent_file)

    def test_helper_module_error_propagation(self):
        """Test that errors from helper modules are properly propagated."""
        # Create a mock helper that returns an error_info object
        error_info = error_utils.create_error_info(
            "TestError",
            "Test error message",
            "Test resolution",
            self.nonexistent_file
        )

        with mock.patch('pamola_core.utils.io.multi_file_utils.read_multi_csv',
                        return_value=error_info):
            # Call function, should return the error_info from helper
            result = io.read_multi_csv([self.nonexistent_file])

            # Verify result is the error_info object
            self.assertTrue(error_utils.is_error_info(result))
            self.assertEqual(result["error_type"], "TestError")
            self.assertEqual(result["message"], "Test error message")


if __name__ == '__main__':
    unittest.main()