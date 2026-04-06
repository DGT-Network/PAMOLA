"""
Unit tests for directory_utils module in pamola_core/utils/io_helpers.

Tests cover directory operations, temporary file management, path utilities,
and security-related functionality.
"""

import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pamola_core.errors.exceptions import (
    PathSecurityError,
    PathValidationError,
    PamolaFileNotFoundError,
    ValidationError,
)
from pamola_core.utils.io_helpers import directory_utils


class TestEnsureDirectory(unittest.TestCase):
    """Test ensure_directory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_directory_creates_new_dir(self):
        """Test ensure_directory creates a new directory."""
        new_dir = os.path.join(self.temp_dir, "new_dir")
        result = directory_utils.ensure_directory(new_dir)

        self.assertTrue(os.path.exists(new_dir))
        self.assertEqual(result, Path(new_dir))

    def test_ensure_directory_with_path_object(self):
        """Test ensure_directory with Path object."""
        new_dir = Path(self.temp_dir) / "path_obj_dir"
        result = directory_utils.ensure_directory(new_dir)

        self.assertTrue(new_dir.exists())
        self.assertEqual(result, new_dir)

    def test_ensure_directory_existing(self):
        """Test ensure_directory with existing directory."""
        existing_dir = os.path.join(self.temp_dir, "existing")
        os.makedirs(existing_dir)

        result = directory_utils.ensure_directory(existing_dir)

        self.assertTrue(os.path.exists(existing_dir))
        self.assertEqual(result, Path(existing_dir))

    def test_ensure_directory_nested(self):
        """Test ensure_directory creates nested directories."""
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        result = directory_utils.ensure_directory(nested_dir)

        self.assertTrue(os.path.exists(nested_dir))
        self.assertEqual(result, Path(nested_dir))


class TestListDirectoryContents(unittest.TestCase):
    """Test list_directory_contents function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        self.files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"file{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"content {i}")
            self.files.append(file_path)

        # Create subdirectory with files
        self.subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(self.subdir)
        subfile = os.path.join(self.subdir, "subfile.txt")
        with open(subfile, 'w') as f:
            f.write("subcontent")

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_list_directory_non_recursive(self):
        """Test listing directory non-recursively."""
        files = directory_utils.list_directory_contents(self.temp_dir, "*.txt", recursive=False)

        self.assertEqual(len(files), 3)

    def test_list_directory_recursive(self):
        """Test listing directory recursively."""
        files = directory_utils.list_directory_contents(self.temp_dir, "*.txt", recursive=True)

        self.assertEqual(len(files), 4)  # 3 in root + 1 in subdir

    def test_list_directory_pattern_filtering(self):
        """Test pattern filtering in list_directory_contents."""
        # Create CSV files
        for i in range(2):
            csv_path = os.path.join(self.temp_dir, f"file{i}.csv")
            with open(csv_path, 'w') as f:
                f.write("csv content")

        # List only CSV files
        files = directory_utils.list_directory_contents(self.temp_dir, "*.csv", recursive=False)

        self.assertEqual(len(files), 2)

    def test_list_directory_nonexistent(self):
        """Test listing non-existent directory raises error."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent")

        with self.assertRaises(PathValidationError):
            directory_utils.list_directory_contents(nonexistent)

    def test_list_directory_file_not_dir(self):
        """Test listing a file (not directory) raises error."""
        file_path = os.path.join(self.temp_dir, "file0.txt")

        with self.assertRaises(PathValidationError):
            directory_utils.list_directory_contents(file_path)


class TestClearDirectory(unittest.TestCase):
    """Test clear_directory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"file{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"content {i}")

        # Create subdirectory
        self.subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(self.subdir)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_clear_directory_all_items(self):
        """Test clearing all items in directory."""
        count = directory_utils.clear_directory(self.temp_dir, confirm=False)

        self.assertEqual(count, 4)  # 3 files + 1 directory
        self.assertEqual(len(os.listdir(self.temp_dir)), 0)

    def test_clear_directory_with_ignore_patterns(self):
        """Test clearing with ignore patterns."""
        count = directory_utils.clear_directory(
            self.temp_dir,
            ignore_patterns=["*.txt"],
            confirm=False
        )

        self.assertEqual(count, 1)  # Only subdir, not txt files
        self.assertGreater(len(os.listdir(self.temp_dir)), 0)  # Files still exist

    def test_clear_directory_nonexistent(self):
        """Test clearing non-existent directory raises error."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent")

        with self.assertRaises(PathValidationError):
            directory_utils.clear_directory(nonexistent)

    def test_clear_directory_file_not_dir(self):
        """Test clearing a file (not directory) raises error."""
        file_path = os.path.join(self.temp_dir, "file0.txt")

        with self.assertRaises(PathValidationError):
            directory_utils.clear_directory(file_path)

    def test_clear_directory_empty(self):
        """Test clearing empty directory."""
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)

        count = directory_utils.clear_directory(empty_dir, confirm=False)

        self.assertEqual(count, 0)


class TestEnsureParentDirectory(unittest.TestCase):
    """Test ensure_parent_directory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_parent_directory_creates_parent(self):
        """Test ensure_parent_directory creates parent directories."""
        file_path = os.path.join(self.temp_dir, "subdir", "file.txt")
        result = directory_utils.ensure_parent_directory(file_path)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.dirname(file_path)))

    def test_ensure_parent_directory_existing_parent(self):
        """Test ensure_parent_directory with existing parent."""
        file_path = os.path.join(self.temp_dir, "file.txt")
        result = directory_utils.ensure_parent_directory(file_path)

        self.assertTrue(result)

    def test_ensure_parent_directory_nested_creation(self):
        """Test ensure_parent_directory creates nested parents."""
        file_path = os.path.join(self.temp_dir, "level1", "level2", "level3", "file.txt")
        result = directory_utils.ensure_parent_directory(file_path)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.dirname(file_path)))


class TestGetFileStats(unittest.TestCase):
    """Test get_file_stats function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")

        with open(self.test_file, 'w') as f:
            f.write("test content")

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_get_file_stats_returns_dict(self):
        """Test get_file_stats returns a dictionary."""
        stats = directory_utils.get_file_stats(self.test_file)

        self.assertIsInstance(stats, dict)

    def test_get_file_stats_contains_expected_keys(self):
        """Test get_file_stats contains expected keys."""
        stats = directory_utils.get_file_stats(self.test_file)

        expected_keys = [
            'name', 'path', 'size_bytes', 'size_mb',
            'creation_time', 'modification_time', 'extension',
            'is_directory', 'is_file', 'is_symlink',
            'permissions', 'owner_readable', 'owner_writable', 'owner_executable'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

    def test_get_file_stats_file_not_found(self):
        """Test get_file_stats with non-existent file."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")

        with self.assertRaises(PamolaFileNotFoundError):
            directory_utils.get_file_stats(nonexistent)

    def test_get_file_stats_size_values(self):
        """Test get_file_stats size calculations."""
        stats = directory_utils.get_file_stats(self.test_file)

        self.assertGreater(stats['size_bytes'], 0)
        self.assertGreater(stats['size_mb'], 0)

    def test_get_file_stats_file_type(self):
        """Test get_file_stats correctly identifies file type."""
        stats = directory_utils.get_file_stats(self.test_file)

        self.assertTrue(stats['is_file'])
        self.assertFalse(stats['is_directory'])


class TestGetTimestampedFilename(unittest.TestCase):
    """Test get_timestamped_filename function."""

    def test_with_timestamp(self):
        """Test get_timestamped_filename with timestamp included."""
        filename = directory_utils.get_timestamped_filename("test", "csv", include_timestamp=True)

        self.assertTrue(filename.startswith("test_"))
        self.assertTrue(filename.endswith(".csv"))
        # Check for timestamp pattern (YYYYMMDD_HHMMSS)
        self.assertRegex(filename, r"test_\d{8}_\d{6}\.csv")

    def test_without_timestamp(self):
        """Test get_timestamped_filename without timestamp."""
        filename = directory_utils.get_timestamped_filename("test", "json", include_timestamp=False)

        self.assertEqual(filename, "test.json")

    def test_default_extension(self):
        """Test get_timestamped_filename with default extension."""
        filename = directory_utils.get_timestamped_filename("myfile", include_timestamp=False)

        self.assertEqual(filename, "myfile.csv")

    def test_various_extensions(self):
        """Test get_timestamped_filename with various extensions."""
        extensions = ["txt", "json", "parquet", "xlsx"]

        for ext in extensions:
            filename = directory_utils.get_timestamped_filename("file", ext, include_timestamp=False)
            self.assertTrue(filename.endswith(f".{ext}"))


class TestGetUniquePath(unittest.TestCase):
    """Test get_unique_filename and make_unique_path functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_get_unique_filename_new(self):
        """Test get_unique_filename with new file."""
        result = directory_utils.get_unique_filename(
            self.temp_dir, "newfile", "txt", include_timestamp=False
        )

        self.assertTrue(str(result).endswith("newfile.txt"))

    def test_make_unique_path_nonexistent(self):
        """Test make_unique_path with non-existent path."""
        path = os.path.join(self.temp_dir, "nonexistent.txt")
        result = directory_utils.make_unique_path(path)

        self.assertEqual(result, Path(path))

    def test_make_unique_path_existing_file(self):
        """Test make_unique_path with existing file."""
        file_path = os.path.join(self.temp_dir, "existing.txt")
        with open(file_path, 'w') as f:
            f.write("test")

        result = directory_utils.make_unique_path(file_path)

        # Should append counter
        self.assertNotEqual(result, Path(file_path))
        self.assertTrue(str(result).endswith(".txt"))

    def test_make_unique_path_existing_directory(self):
        """Test make_unique_path with existing directory."""
        dir_path = os.path.join(self.temp_dir, "existing_dir")
        os.makedirs(dir_path)

        result = directory_utils.make_unique_path(dir_path)

        # Should append counter
        self.assertNotEqual(result, Path(dir_path))


class TestSafeRemoveTempFile(unittest.TestCase):
    """Test safe_remove_temp_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "temp.txt")

        with open(self.test_file, 'w') as f:
            f.write("temp content")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_safe_remove_temp_file_success(self):
        """Test safe_remove_temp_file removes file successfully."""
        self.assertTrue(os.path.exists(self.test_file))

        result = directory_utils.safe_remove_temp_file(self.test_file)

        self.assertTrue(result)
        self.assertFalse(os.path.exists(self.test_file))

    def test_safe_remove_temp_file_nonexistent(self):
        """Test safe_remove_temp_file with non-existent file."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")

        result = directory_utils.safe_remove_temp_file(nonexistent)

        # Should return True (success) because file doesn't exist to remove
        self.assertTrue(result)

    def test_safe_remove_temp_file_none(self):
        """Test safe_remove_temp_file with None path."""
        result = directory_utils.safe_remove_temp_file(None)

        self.assertTrue(result)

    def test_safe_remove_temp_file_with_logger(self):
        """Test safe_remove_temp_file with custom logger."""
        mock_logger = mock.Mock()

        result = directory_utils.safe_remove_temp_file(self.test_file, logger_obj=mock_logger)

        self.assertTrue(result)
        mock_logger.debug.assert_called()

    def test_safe_remove_temp_file_removal_from_tracking(self):
        """Test safe_remove_temp_file removes from tracking."""
        # Add file to tracking
        directory_utils._temp_resources["files"].add(self.test_file)

        result = directory_utils.safe_remove_temp_file(self.test_file)

        self.assertTrue(result)
        self.assertNotIn(self.test_file, directory_utils._temp_resources["files"])


class TestTempFileCreation(unittest.TestCase):
    """Test temporary file creation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_file = os.path.join(self.temp_dir, "original.txt")

        with open(self.original_file, 'w') as f:
            f.write("original content")

    def tearDown(self):
        """Clean up temporary files."""
        # Clear all tracked temp resources
        directory_utils._temp_resources["files"].clear()
        directory_utils._temp_resources["directories"].clear()

        shutil.rmtree(self.temp_dir)

    def test_get_temp_file_creation(self):
        """Test get_temp_file creates temporary file."""
        temp_file = directory_utils.get_temp_file(self.original_file)

        self.assertIsInstance(temp_file, Path)
        self.assertTrue(os.path.exists(temp_file))
        self.assertIn(str(temp_file), directory_utils._temp_resources["files"])

    def test_get_temp_file_suffix(self):
        """Test get_temp_file uses custom suffix."""
        temp_file = directory_utils.get_temp_file(self.original_file, suffix=".backup")

        self.assertTrue(str(temp_file).endswith(".backup"))

    def test_get_temp_file_no_create(self):
        """Test get_temp_file with create=False."""
        temp_file = directory_utils.get_temp_file(self.original_file, create=False)

        # Path should be registered but not created
        self.assertIn(str(temp_file), directory_utils._temp_resources["files"])

    def test_get_temp_file_for_encryption(self):
        """Test get_temp_file_for_encryption."""
        temp_file = directory_utils.get_temp_file_for_encryption(self.original_file)

        self.assertIsInstance(temp_file, Path)
        self.assertTrue(str(temp_file).endswith(".enc"))

    def test_get_temp_file_for_decryption(self):
        """Test get_temp_file_for_decryption."""
        temp_file = directory_utils.get_temp_file_for_decryption(self.original_file)

        self.assertIsInstance(temp_file, Path)
        self.assertTrue(str(temp_file).endswith(".dec"))


class TestNormalizePath(unittest.TestCase):
    """Test normalize_path function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_normalize_path_basic(self):
        """Test normalize_path converts to Path object."""
        path_str = os.path.join(self.temp_dir, "test.txt")
        result = directory_utils.normalize_path(path_str)

        self.assertIsInstance(result, Path)

    def test_normalize_path_absolute(self):
        """Test normalize_path with make_absolute=True."""
        relative_path = "relative/path.txt"
        result = directory_utils.normalize_path(relative_path, make_absolute=True)

        self.assertTrue(result.is_absolute())

    def test_normalize_path_symlinks(self):
        """Test normalize_path with resolve_symlinks=True."""
        file_path = os.path.join(self.temp_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write("test")

        result = directory_utils.normalize_path(file_path, resolve_symlinks=True)

        self.assertIsInstance(result, Path)


if __name__ == '__main__':
    unittest.main()
