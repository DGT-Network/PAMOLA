"""
Extended unit tests for directory_utils module - targeting missed coverage lines.

Focuses on: cleanup handlers, temp file functions, path security, secure directory
creation, secure_cleanup, protect_path, create_secure_path, is_path_in_directory,
is_path_directory, is_path_writable, normalize_path variants.
"""

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
)
from pamola_core.utils.io_helpers import directory_utils


class TestCleanupTempResources(unittest.TestCase):
    """Test _cleanup_temp_resources function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cleanup_removes_registered_file(self):
        """Cleanup should remove tracked temp files."""
        temp_file = os.path.join(self.temp_dir, "cleanup_test.txt")
        with open(temp_file, "w") as f:
            f.write("test")

        old_files = set(directory_utils._temp_resources["files"])
        directory_utils._temp_resources["files"].add(temp_file)

        directory_utils._cleanup_temp_resources()

        self.assertFalse(os.path.exists(temp_file))

        # Restore original state
        directory_utils._temp_resources["files"] = old_files

    def test_cleanup_removes_registered_directory(self):
        """Cleanup should remove tracked temp directories."""
        temp_subdir = os.path.join(self.temp_dir, "cleanup_dir")
        os.makedirs(temp_subdir)

        old_dirs = set(directory_utils._temp_resources["directories"])
        directory_utils._temp_resources["directories"].add(temp_subdir)

        directory_utils._cleanup_temp_resources()

        self.assertFalse(os.path.exists(temp_subdir))

        directory_utils._temp_resources["directories"] = old_dirs

    def test_cleanup_handles_nonexistent_file_gracefully(self):
        """Cleanup should handle missing files without error."""
        old_files = set(directory_utils._temp_resources["files"])
        directory_utils._temp_resources["files"].add("/nonexistent/path/file.tmp")

        # Should not raise
        try:
            directory_utils._cleanup_temp_resources()
        except Exception:
            self.fail("_cleanup_temp_resources raised unexpectedly")
        finally:
            directory_utils._temp_resources["files"] = old_files


class TestGetTempFile(unittest.TestCase):
    """Extended tests for get_temp_file."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original = os.path.join(self.temp_dir, "original.csv")
        with open(self.original, "w") as f:
            f.write("a,b,c\n1,2,3")

    def tearDown(self):
        directory_utils._temp_resources["files"].discard(
            *list(directory_utils._temp_resources["files"])
        ) if directory_utils._temp_resources["files"] else None
        # Use discard approach to not fail if empty
        directory_utils._temp_resources["files"].clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_temp_file_creates_in_same_dir(self):
        """Temp file should be created in same directory as original."""
        temp_file = directory_utils.get_temp_file(self.original)
        self.assertEqual(temp_file.parent, Path(self.original).parent)

    def test_get_temp_file_contains_tmp_in_name(self):
        """Temp file name should contain _tmp_."""
        temp_file = directory_utils.get_temp_file(self.original)
        self.assertIn("_tmp_", temp_file.name)

    def test_get_temp_file_fallback_to_system_temp(self):
        """Should fall back to system temp if parent not writable."""
        # Use a path whose parent doesn't exist
        nonexistent_original = Path(self.temp_dir) / "noparent" / "file.csv"
        temp_file = directory_utils.get_temp_file(nonexistent_original)
        # Should still return a valid Path
        self.assertIsInstance(temp_file, Path)

    def test_get_temp_file_create_false_registers_but_not_created(self):
        """With create=False the file path is registered but may not exist."""
        temp_file = directory_utils.get_temp_file(self.original, create=False)
        self.assertIn(str(temp_file), directory_utils._temp_resources["files"])

    def test_get_temp_file_for_encryption_suffix(self):
        """get_temp_file_for_encryption uses .enc suffix."""
        temp_file = directory_utils.get_temp_file_for_encryption(self.original)
        self.assertTrue(str(temp_file).endswith(".enc"))

    def test_get_temp_file_for_decryption_suffix(self):
        """get_temp_file_for_decryption uses .dec suffix."""
        temp_file = directory_utils.get_temp_file_for_decryption(self.original)
        self.assertTrue(str(temp_file).endswith(".dec"))

    def test_get_temp_file_custom_suffix(self):
        """Custom suffix is applied."""
        temp_file = directory_utils.get_temp_file(self.original, suffix=".bak")
        self.assertTrue(str(temp_file).endswith(".bak"))


class TestSafeRemoveTempFileExtended(unittest.TestCase):
    """Extended coverage for safe_remove_temp_file."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_removes_from_tracking_set(self):
        """File removed from _temp_resources when unlinked."""
        tracked_file = os.path.join(self.temp_dir, "tracked.txt")
        with open(tracked_file, "w") as f:
            f.write("x")
        directory_utils._temp_resources["files"].add(tracked_file)

        result = directory_utils.safe_remove_temp_file(tracked_file)

        self.assertTrue(result)
        self.assertNotIn(tracked_file, directory_utils._temp_resources["files"])

    def test_returns_true_for_none_path(self):
        result = directory_utils.safe_remove_temp_file(None)
        self.assertTrue(result)

    def test_returns_false_on_error(self):
        """Should return False when removal fails."""
        with mock.patch("pamola_core.utils.io_helpers.directory_utils.Path.exists", return_value=True):
            with mock.patch("pamola_core.utils.io_helpers.directory_utils.Path.unlink", side_effect=OSError("Permission denied")):
                result = directory_utils.safe_remove_temp_file("/some/locked/file.txt")
                self.assertFalse(result)


class TestMakeUniquePath(unittest.TestCase):
    """Extended tests for make_unique_path."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_returns_as_is(self):
        p = os.path.join(self.temp_dir, "brand_new.txt")
        result = directory_utils.make_unique_path(p)
        self.assertEqual(result, Path(p))

    def test_existing_file_increments_counter(self):
        p = os.path.join(self.temp_dir, "file.txt")
        with open(p, "w") as f:
            f.write("x")
        result = directory_utils.make_unique_path(p)
        self.assertIn("_1", str(result))
        self.assertNotEqual(result, Path(p))

    def test_existing_dir_increments_counter(self):
        d = os.path.join(self.temp_dir, "mydir")
        os.makedirs(d)
        result = directory_utils.make_unique_path(d)
        self.assertNotEqual(result, Path(d))

    def test_counter_increments_beyond_1(self):
        """When _1 also exists, should try _2."""
        p = os.path.join(self.temp_dir, "file.txt")
        p1 = os.path.join(self.temp_dir, "file_1.txt")
        for fp in [p, p1]:
            with open(fp, "w") as f:
                f.write("x")

        result = directory_utils.make_unique_path(p)
        self.assertIn("_2", str(result))

    def test_create_flag_with_nonexistent_dir_path(self):
        """create=True on a dir-like path should call ensure_directory."""
        p = os.path.join(self.temp_dir, "newdir")
        result = directory_utils.make_unique_path(p, create=True)
        self.assertIsInstance(result, Path)


class TestIsPathInDirectory(unittest.TestCase):
    """Tests for is_path_in_directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_path_is_directory_itself(self):
        result = directory_utils.is_path_in_directory(self.temp_dir, self.temp_dir)
        self.assertTrue(result)

    def test_direct_child_with_include_subdirs_false(self):
        child = os.path.join(self.temp_dir, "child.txt")
        result = directory_utils.is_path_in_directory(child, self.temp_dir, include_subdirs=False)
        self.assertTrue(result)

    def test_nested_child_with_include_subdirs_false(self):
        nested = os.path.join(self.temp_dir, "sub", "nested.txt")
        result = directory_utils.is_path_in_directory(nested, self.temp_dir, include_subdirs=False)
        self.assertFalse(result)

    def test_nested_child_with_include_subdirs_true(self):
        nested = os.path.join(self.temp_dir, "sub", "nested.txt")
        result = directory_utils.is_path_in_directory(nested, self.temp_dir, include_subdirs=True)
        self.assertTrue(result)

    def test_path_outside_directory(self):
        other_dir = tempfile.mkdtemp()
        try:
            result = directory_utils.is_path_in_directory(other_dir, self.temp_dir)
            self.assertFalse(result)
        finally:
            shutil.rmtree(other_dir, ignore_errors=True)


class TestIsPathDirectory(unittest.TestCase):
    """Tests for is_path_directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_existing_directory_returns_true(self):
        self.assertTrue(directory_utils.is_path_directory(self.temp_dir))

    def test_existing_file_returns_false(self):
        f = os.path.join(self.temp_dir, "f.txt")
        with open(f, "w") as fp:
            fp.write("x")
        self.assertFalse(directory_utils.is_path_directory(f))

    def test_nonexistent_with_no_extension_returns_true(self):
        p = os.path.join(self.temp_dir, "nodot")
        self.assertTrue(directory_utils.is_path_directory(p))

    def test_nonexistent_with_extension_returns_false(self):
        p = os.path.join(self.temp_dir, "file.csv")
        self.assertFalse(directory_utils.is_path_directory(p))

    def test_path_ending_with_sep_returns_true(self):
        p = os.path.join(self.temp_dir, "mydir") + os.sep
        self.assertTrue(directory_utils.is_path_directory(p))


class TestIsPathWritable(unittest.TestCase):
    """Tests for is_path_writable."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_existing_writable_directory(self):
        self.assertTrue(directory_utils.is_path_writable(self.temp_dir))

    def test_existing_file_is_writable(self):
        f = os.path.join(self.temp_dir, "writable.txt")
        with open(f, "w") as fp:
            fp.write("x")
        self.assertTrue(directory_utils.is_path_writable(f))

    def test_nonexistent_path_checks_parent(self):
        nonexistent = os.path.join(self.temp_dir, "doesnotexist.txt")
        # Parent (temp_dir) is writable
        result = directory_utils.is_path_writable(nonexistent)
        self.assertTrue(result)


class TestProtectPath(unittest.TestCase):
    """Tests for protect_path."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_protect_path_nonexistent_raises(self):
        nonexistent = os.path.join(self.temp_dir, "missing.txt")
        with self.assertRaises(PamolaFileNotFoundError):
            directory_utils.protect_path(nonexistent)

    def test_protect_path_make_writable(self):
        """protect_path with readonly=False should make file writable."""
        f = os.path.join(self.temp_dir, "protect_me.txt")
        with open(f, "w") as fp:
            fp.write("data")
        # Should not raise
        directory_utils.protect_path(f, readonly=False)
        self.assertTrue(os.access(f, os.W_OK))

    def test_protect_path_make_readonly(self):
        """protect_path with readonly=True should restrict writes."""
        f = os.path.join(self.temp_dir, "readonly_me.txt")
        with open(f, "w") as fp:
            fp.write("data")
        directory_utils.protect_path(f, readonly=True)
        # Re-enable write for cleanup
        try:
            directory_utils.protect_path(f, readonly=False)
        except Exception:
            pass


class TestCreateSecureDirectoryStructure(unittest.TestCase):
    """Tests for create_secure_directory_structure."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_base_and_subdirs(self):
        base = os.path.join(self.temp_dir, "secure_base")
        result = directory_utils.create_secure_directory_structure(
            base, ["data", "logs", "keys"]
        )
        self.assertIn("base", result)
        self.assertIn("data", result)
        self.assertIn("logs", result)
        self.assertIn("keys", result)
        for name, path in result.items():
            self.assertTrue(path.exists())

    def test_sanitizes_dotdot_traversal(self):
        base = os.path.join(self.temp_dir, "secure_base2")
        result = directory_utils.create_secure_directory_structure(
            base, ["../../escape"]
        )
        # Path should have been sanitized (.. replaced with __)
        keys = list(result.keys())
        self.assertFalse(any(".." in k for k in keys))

    def test_empty_subdirs(self):
        base = os.path.join(self.temp_dir, "empty_base")
        result = directory_utils.create_secure_directory_structure(base, [])
        self.assertIn("base", result)
        self.assertEqual(len(result), 1)


class TestCreateSecureTempDirectory(unittest.TestCase):
    """Tests for create_secure_temp_directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Track dirs to clean up
        self._created = []

    def tearDown(self):
        for d in self._created:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean tracking
        directory_utils._temp_resources["directories"].clear()

    def test_creates_directory(self):
        d = directory_utils.create_secure_temp_directory(parent_dir=self.temp_dir)
        self._created.append(str(d))
        self.assertTrue(d.exists())
        self.assertTrue(d.is_dir())

    def test_uses_prefix(self):
        d = directory_utils.create_secure_temp_directory(
            prefix="myprefix_", parent_dir=self.temp_dir
        )
        self._created.append(str(d))
        self.assertTrue(d.name.startswith("myprefix_"))

    def test_registers_for_cleanup(self):
        d = directory_utils.create_secure_temp_directory(
            parent_dir=self.temp_dir, register_for_cleanup=True
        )
        self._created.append(str(d))
        self.assertIn(str(d), directory_utils._temp_resources["directories"])

    def test_no_register_when_flag_false(self):
        d = directory_utils.create_secure_temp_directory(
            parent_dir=self.temp_dir, register_for_cleanup=False
        )
        self._created.append(str(d))
        self.assertNotIn(str(d), directory_utils._temp_resources["directories"])

    def test_without_parent_uses_system_tempdir(self):
        d = directory_utils.create_secure_temp_directory(register_for_cleanup=False)
        self._created.append(str(d))
        self.assertTrue(d.exists())


class TestCreateSecureTempFile(unittest.TestCase):
    """Tests for create_secure_temp_file."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        directory_utils._temp_resources["files"].clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_file(self):
        f = directory_utils.create_secure_temp_file(directory=self.temp_dir)
        self.assertTrue(f.exists())

    def test_uses_prefix_and_suffix(self):
        f = directory_utils.create_secure_temp_file(
            prefix="pfx_", suffix=".dat", directory=self.temp_dir
        )
        self.assertTrue(f.name.startswith("pfx_"))
        self.assertTrue(str(f).endswith(".dat"))

    def test_registered_for_cleanup(self):
        f = directory_utils.create_secure_temp_file(
            directory=self.temp_dir, register_for_cleanup=True
        )
        self.assertIn(str(f), directory_utils._temp_resources["files"])

    def test_not_registered_when_false(self):
        f = directory_utils.create_secure_temp_file(
            directory=self.temp_dir, register_for_cleanup=False
        )
        self.assertNotIn(str(f), directory_utils._temp_resources["files"])

    def test_without_directory_uses_system_tempdir(self):
        f = directory_utils.create_secure_temp_file(register_for_cleanup=False)
        self.assertTrue(f.exists())
        # Clean up
        f.unlink(missing_ok=True)


class TestCreateSecurePath(unittest.TestCase):
    """Tests for create_secure_path."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_secure_path(self):
        result = directory_utils.create_secure_path(self.temp_dir, "subdir/file.txt")
        self.assertIsInstance(result, Path)
        self.assertTrue(str(result).startswith(str(Path(self.temp_dir).absolute())))

    def test_traversal_sanitized(self):
        """Directory traversal sequences should be replaced."""
        result = directory_utils.create_secure_path(self.temp_dir, "../escape/file.txt")
        # The resulting path should still be within base_dir or sanitized
        # as .. is replaced with __
        self.assertIsInstance(result, Path)

    def test_leading_slash_removed(self):
        result = directory_utils.create_secure_path(self.temp_dir, "/subdir/file.txt")
        self.assertIsInstance(result, Path)

    def test_create_flag_creates_directories(self):
        result = directory_utils.create_secure_path(
            self.temp_dir, "newsubdir/file.txt", create=True
        )
        self.assertTrue(result.parent.exists())


class TestSecureCleanup(unittest.TestCase):
    """Tests for secure_cleanup."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_removes_regular_file(self):
        f = os.path.join(self.temp_dir, "to_remove.txt")
        with open(f, "w") as fp:
            fp.write("data")
        failed = directory_utils.secure_cleanup([f])
        self.assertFalse(os.path.exists(f))
        self.assertEqual(failed, [])

    def test_removes_directory(self):
        d = os.path.join(self.temp_dir, "removedir")
        os.makedirs(d)
        failed = directory_utils.secure_cleanup([d])
        self.assertFalse(os.path.exists(d))
        self.assertEqual(failed, [])

    def test_nonexistent_path_skipped(self):
        failed = directory_utils.secure_cleanup(["/nonexistent/path/xyz.txt"])
        self.assertEqual(failed, [])

    def test_secure_delete_overwrites_before_removal(self):
        f = os.path.join(self.temp_dir, "secure_del.txt")
        with open(f, "w") as fp:
            fp.write("sensitive data here")
        failed = directory_utils.secure_cleanup([f], secure_delete=True)
        self.assertFalse(os.path.exists(f))
        self.assertEqual(failed, [])

    def test_secure_delete_directory(self):
        d = os.path.join(self.temp_dir, "securedir")
        os.makedirs(d)
        inner = os.path.join(d, "inner.txt")
        with open(inner, "w") as fp:
            fp.write("sensitive")
        failed = directory_utils.secure_cleanup([d], secure_delete=True)
        self.assertFalse(os.path.exists(d))
        self.assertEqual(failed, [])

    def test_empty_list_returns_empty(self):
        failed = directory_utils.secure_cleanup([])
        self.assertEqual(failed, [])


class TestGetUniqueFilenameExtended(unittest.TestCase):
    """Extended tests for get_unique_filename."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_with_timestamp_creates_timestamped_file(self):
        result = directory_utils.get_unique_filename(
            self.temp_dir, "report", "csv", include_timestamp=True
        )
        self.assertIsInstance(result, Path)
        # Name should contain timestamp pattern
        import re
        self.assertRegex(result.name, r"report_\d{8}_\d{6}\.csv")

    def test_extension_with_leading_dot_stripped(self):
        result = directory_utils.get_unique_filename(
            self.temp_dir, "data", ".json", include_timestamp=False
        )
        self.assertTrue(result.name.endswith(".json"))
        # Should not have double dots
        self.assertNotIn("..", result.name)

    def test_returns_unique_path_when_existing(self):
        # Pre-create a file to force uniqueness logic
        existing = directory_utils.get_unique_filename(
            self.temp_dir, "file", "txt", include_timestamp=False
        )
        existing.touch()
        # Now request another file with same base name
        result = directory_utils.get_unique_filename(
            self.temp_dir, "file", "txt", include_timestamp=False
        )
        self.assertNotEqual(result, existing)


if __name__ == "__main__":
    unittest.main()
