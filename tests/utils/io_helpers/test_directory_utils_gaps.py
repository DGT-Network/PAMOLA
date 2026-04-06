"""
Gap tests for directory_utils module.
Covers missed lines: ensure_directory, safe_remove_temp_file, get_temp_dir,
list_directory_contents, clear_directory, get_file_stats, get_timestamped_filename,
get_unique_filename, get_temp_file, get_temp_file_for_encryption,
get_temp_file_for_decryption, make_unique_path, normalize_path, is_path_in_directory,
is_path_directory, is_path_writable, protect_path, create_secure_directory_structure,
create_secure_temp_directory, create_secure_temp_file, create_secure_path,
secure_cleanup, with_secure_temp_directory, ensure_parent_directory.
"""

import os
import pytest
from pathlib import Path

from pamola_core.utils.io_helpers.directory_utils import (
    ensure_directory,
    ensure_parent_directory,
    list_directory_contents,
    clear_directory,
    get_file_stats,
    get_timestamped_filename,
    get_unique_filename,
    get_temp_file,
    get_temp_file_for_encryption,
    get_temp_file_for_decryption,
    safe_remove_temp_file,
    make_unique_path,
    normalize_path,
    is_path_in_directory,
    is_path_directory,
    is_path_writable,
    protect_path,
    create_secure_directory_structure,
    create_secure_temp_directory,
    create_secure_temp_file,
    create_secure_path,
    secure_cleanup,
    with_secure_temp_directory,
)
from pamola_core.errors.exceptions import (
    PathValidationError,
    PamolaFileNotFoundError,
    PathSecurityError,
)


# ---------------------------------------------------------------------------
# ensure_directory
# ---------------------------------------------------------------------------

def test_ensure_directory_creates_new(tmp_path):
    new_dir = tmp_path / "new_subdir"
    result = ensure_directory(new_dir)
    assert result.exists()
    assert result.is_dir()


def test_ensure_directory_existing(tmp_path):
    result = ensure_directory(tmp_path)
    assert result == tmp_path


def test_ensure_directory_returns_path_obj(tmp_path):
    result = ensure_directory(str(tmp_path))
    assert isinstance(result, Path)


def test_ensure_directory_nested(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    result = ensure_directory(nested)
    assert result.exists()


# ---------------------------------------------------------------------------
# ensure_parent_directory
# ---------------------------------------------------------------------------

def test_ensure_parent_directory_creates_parent(tmp_path):
    file_path = tmp_path / "subdir" / "file.txt"
    result = ensure_parent_directory(file_path)
    assert result is True
    assert file_path.parent.exists()


def test_ensure_parent_directory_already_exists(tmp_path):
    file_path = tmp_path / "file.txt"
    result = ensure_parent_directory(file_path)
    assert result is True


# ---------------------------------------------------------------------------
# list_directory_contents
# ---------------------------------------------------------------------------

def test_list_directory_contents_basic(tmp_path):
    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "file2.txt").write_text("b")
    items = list_directory_contents(tmp_path)
    assert len(items) == 2


def test_list_directory_contents_with_pattern(tmp_path):
    (tmp_path / "data.csv").write_text("a,b")
    (tmp_path / "notes.txt").write_text("hello")
    items = list_directory_contents(tmp_path, pattern="*.csv")
    assert len(items) == 1
    assert items[0].suffix == ".csv"


def test_list_directory_contents_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("deep")
    (tmp_path / "top.txt").write_text("top")
    items = list_directory_contents(tmp_path, pattern="*.txt", recursive=True)
    assert len(items) == 2


def test_list_directory_contents_nonexistent_raises():
    with pytest.raises(PathValidationError):
        list_directory_contents(Path("/nonexistent/path/xyz"))


# ---------------------------------------------------------------------------
# clear_directory
# ---------------------------------------------------------------------------

def test_clear_directory_removes_files(tmp_path):
    (tmp_path / "f1.txt").write_text("x")
    (tmp_path / "f2.txt").write_text("y")
    count = clear_directory(tmp_path, confirm=False)
    assert count == 2
    assert len(list(tmp_path.iterdir())) == 0


def test_clear_directory_with_ignore_patterns(tmp_path):
    (tmp_path / "keep.log").write_text("keep")
    (tmp_path / "remove.txt").write_text("remove")
    count = clear_directory(tmp_path, ignore_patterns=["*.log"], confirm=False)
    assert count == 1
    assert (tmp_path / "keep.log").exists()


def test_clear_directory_empty_dir(tmp_path):
    count = clear_directory(tmp_path, confirm=False)
    assert count == 0


def test_clear_directory_nonexistent_raises():
    with pytest.raises(PathValidationError):
        clear_directory(Path("/nonexistent/path/xyz"), confirm=False)


# ---------------------------------------------------------------------------
# get_file_stats
# ---------------------------------------------------------------------------

def test_get_file_stats_basic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    stats = get_file_stats(f)
    assert stats["name"] == "test.txt"
    assert stats["size_bytes"] > 0
    assert stats["is_file"] is True
    assert stats["is_directory"] is False


def test_get_file_stats_nonexistent_raises():
    with pytest.raises(PamolaFileNotFoundError):
        get_file_stats(Path("/nonexistent/file.txt"))


def test_get_file_stats_directory(tmp_path):
    stats = get_file_stats(tmp_path)
    assert stats["is_directory"] is True


# ---------------------------------------------------------------------------
# get_timestamped_filename
# ---------------------------------------------------------------------------

def test_get_timestamped_filename_with_timestamp():
    name = get_timestamped_filename("report", "csv", include_timestamp=True)
    assert name.startswith("report_")
    assert name.endswith(".csv")


def test_get_timestamped_filename_without_timestamp():
    name = get_timestamped_filename("report", "csv", include_timestamp=False)
    assert name == "report.csv"


def test_get_timestamped_filename_custom_extension():
    name = get_timestamped_filename("data", "parquet", include_timestamp=False)
    assert name == "data.parquet"


# ---------------------------------------------------------------------------
# get_unique_filename
# ---------------------------------------------------------------------------

def test_get_unique_filename_basic(tmp_path):
    path = get_unique_filename(tmp_path, "output", "csv", include_timestamp=False)
    assert isinstance(path, Path)
    assert path.suffix == ".csv"
    assert path.parent == tmp_path


def test_get_unique_filename_creates_dir(tmp_path):
    new_dir = tmp_path / "new_output"
    path = get_unique_filename(new_dir, "data", "json", include_timestamp=False)
    assert new_dir.exists()


# ---------------------------------------------------------------------------
# get_temp_file
# ---------------------------------------------------------------------------

def test_get_temp_file_creates_file(tmp_path):
    orig = tmp_path / "original.csv"
    orig.write_text("data")
    temp = get_temp_file(orig)
    assert temp.exists()
    assert temp.suffix == ".tmp"
    # cleanup
    temp.unlink(missing_ok=True)


def test_get_temp_file_no_create(tmp_path):
    orig = tmp_path / "original.csv"
    orig.write_text("data")
    temp = get_temp_file(orig, create=False)
    assert not temp.exists()


def test_get_temp_file_custom_suffix(tmp_path):
    orig = tmp_path / "data.csv"
    orig.write_text("data")
    temp = get_temp_file(orig, suffix=".bak")
    assert temp.suffix == ".bak"
    temp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# get_temp_file_for_encryption / decryption
# ---------------------------------------------------------------------------

def test_get_temp_file_for_encryption(tmp_path):
    orig = tmp_path / "plain.csv"
    orig.write_text("data")
    temp = get_temp_file_for_encryption(orig)
    assert temp.suffix == ".enc"
    temp.unlink(missing_ok=True)


def test_get_temp_file_for_decryption(tmp_path):
    orig = tmp_path / "encrypted.enc"
    orig.write_text("data")
    temp = get_temp_file_for_decryption(orig)
    assert temp.suffix == ".dec"
    temp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# safe_remove_temp_file
# ---------------------------------------------------------------------------

def test_safe_remove_temp_file_existing(tmp_path):
    f = tmp_path / "temp.tmp"
    f.write_text("content")
    result = safe_remove_temp_file(f)
    assert result is True
    assert not f.exists()


def test_safe_remove_temp_file_nonexistent(tmp_path):
    result = safe_remove_temp_file(tmp_path / "missing.tmp")
    assert result is True  # nonexistent is considered success


def test_safe_remove_temp_file_none():
    result = safe_remove_temp_file(None)
    assert result is True


# ---------------------------------------------------------------------------
# make_unique_path
# ---------------------------------------------------------------------------

def test_make_unique_path_nonexistent(tmp_path):
    path = tmp_path / "newfile.txt"
    result = make_unique_path(path)
    assert result == path  # no conflict


def test_make_unique_path_existing_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = make_unique_path(f)
    assert result != f
    assert result.stem.endswith("_1")


def test_make_unique_path_existing_dir(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    result = make_unique_path(d)
    assert result != d


# ---------------------------------------------------------------------------
# normalize_path
# ---------------------------------------------------------------------------

def test_normalize_path_basic():
    result = normalize_path("/tmp/test")
    assert isinstance(result, Path)


def test_normalize_path_make_absolute():
    result = normalize_path("relative/path", make_absolute=True)
    assert result.is_absolute()


def test_normalize_path_resolve_symlinks(tmp_path):
    result = normalize_path(tmp_path, resolve_symlinks=True)
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# is_path_in_directory
# ---------------------------------------------------------------------------

def test_is_path_in_directory_true(tmp_path):
    child = tmp_path / "child" / "file.txt"
    assert is_path_in_directory(child, tmp_path) is True


def test_is_path_in_directory_false(tmp_path):
    other = Path("/other/path/file.txt")
    assert is_path_in_directory(other, tmp_path) is False


def test_is_path_in_directory_same_path(tmp_path):
    assert is_path_in_directory(tmp_path, tmp_path) is True


def test_is_path_in_directory_no_subdirs(tmp_path):
    direct_child = tmp_path / "file.txt"
    nested_child = tmp_path / "sub" / "file.txt"
    assert is_path_in_directory(direct_child, tmp_path, include_subdirs=False) is True
    assert is_path_in_directory(nested_child, tmp_path, include_subdirs=False) is False


# ---------------------------------------------------------------------------
# is_path_directory
# ---------------------------------------------------------------------------

def test_is_path_directory_existing_dir(tmp_path):
    assert is_path_directory(tmp_path) is True


def test_is_path_directory_existing_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    assert is_path_directory(f) is False


def test_is_path_directory_nonexistent_no_extension():
    result = is_path_directory(Path("/nonexistent/directory_without_ext"))
    assert result is True


def test_is_path_directory_nonexistent_with_extension():
    result = is_path_directory(Path("/nonexistent/file.txt"))
    assert result is False


# ---------------------------------------------------------------------------
# is_path_writable
# ---------------------------------------------------------------------------

def test_is_path_writable_existing_dir(tmp_path):
    assert is_path_writable(tmp_path) is True


def test_is_path_writable_existing_file(tmp_path):
    f = tmp_path / "writable.txt"
    f.write_text("x")
    assert is_path_writable(f) is True


def test_is_path_writable_nonexistent_in_writable_parent(tmp_path):
    nonexistent = tmp_path / "newfile.txt"
    assert is_path_writable(nonexistent) is True


# ---------------------------------------------------------------------------
# protect_path
# ---------------------------------------------------------------------------

def test_protect_path_readonly(tmp_path):
    f = tmp_path / "protected.txt"
    f.write_text("content")
    protect_path(f, readonly=True)
    # Restore so cleanup works
    protect_path(f, readonly=False)


def test_protect_path_nonexistent_raises():
    with pytest.raises(PamolaFileNotFoundError):
        protect_path(Path("/nonexistent/file.txt"))


# ---------------------------------------------------------------------------
# create_secure_directory_structure
# ---------------------------------------------------------------------------

def test_create_secure_directory_structure(tmp_path):
    result = create_secure_directory_structure(tmp_path, ["data", "logs", "temp"])
    assert "base" in result
    assert "data" in result
    assert result["data"].exists()


def test_create_secure_directory_structure_sanitizes_traversal(tmp_path):
    result = create_secure_directory_structure(tmp_path, ["../escape"])
    # traversal should be sanitized
    assert "__/escape" in result or "__escape" in result or any("__" in k for k in result)


# ---------------------------------------------------------------------------
# create_secure_temp_directory
# ---------------------------------------------------------------------------

def test_create_secure_temp_directory_default():
    d = create_secure_temp_directory()
    assert d.exists()
    assert d.is_dir()
    # cleanup
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_create_secure_temp_directory_with_parent(tmp_path):
    d = create_secure_temp_directory(parent_dir=tmp_path)
    assert d.exists()
    assert d.parent == tmp_path


def test_create_secure_temp_directory_custom_prefix(tmp_path):
    d = create_secure_temp_directory(prefix="mytest_", parent_dir=tmp_path)
    assert d.name.startswith("mytest_")


def test_create_secure_temp_directory_no_cleanup_registration(tmp_path):
    d = create_secure_temp_directory(parent_dir=tmp_path, register_for_cleanup=False)
    assert d.exists()


# ---------------------------------------------------------------------------
# create_secure_temp_file
# ---------------------------------------------------------------------------

def test_create_secure_temp_file_default():
    f = create_secure_temp_file()
    assert f.exists()
    f.unlink(missing_ok=True)


def test_create_secure_temp_file_custom_prefix_suffix(tmp_path):
    f = create_secure_temp_file(prefix="test_", suffix=".dat", directory=tmp_path)
    assert f.name.startswith("test_")
    assert f.suffix == ".dat"
    f.unlink(missing_ok=True)


def test_create_secure_temp_file_no_registration():
    f = create_secure_temp_file(register_for_cleanup=False)
    assert f.exists()
    f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# create_secure_path
# ---------------------------------------------------------------------------

def test_create_secure_path_valid(tmp_path):
    result = create_secure_path(tmp_path, "subdir/file.txt")
    assert isinstance(result, Path)
    assert str(tmp_path) in str(result)


def test_create_secure_path_create_dirs(tmp_path):
    result = create_secure_path(tmp_path, "new_subdir/", create=True)
    assert result.parent.exists() or result.exists()


def test_create_secure_path_traversal_sanitized(tmp_path):
    # Should not raise — traversal is replaced with __ instead
    result = create_secure_path(tmp_path, "../attempt")
    assert str(tmp_path) in str(result)


# ---------------------------------------------------------------------------
# secure_cleanup
# ---------------------------------------------------------------------------

def test_secure_cleanup_file(tmp_path):
    f = tmp_path / "to_delete.txt"
    f.write_text("remove me")
    failed = secure_cleanup([f])
    assert failed == []
    assert not f.exists()


def test_secure_cleanup_directory(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    (d / "file.txt").write_text("x")
    failed = secure_cleanup([d])
    assert failed == []
    assert not d.exists()


def test_secure_cleanup_nonexistent(tmp_path):
    # should not raise, just skip
    failed = secure_cleanup([tmp_path / "nonexistent.txt"])
    assert failed == []


def test_secure_cleanup_secure_delete_file(tmp_path):
    f = tmp_path / "secret.txt"
    f.write_text("secret data")
    failed = secure_cleanup([f], secure_delete=True)
    assert failed == []
    assert not f.exists()


def test_secure_cleanup_ignore_errors(tmp_path):
    failed = secure_cleanup(
        [tmp_path / "nonexistent.txt"], ignore_errors=True
    )
    assert isinstance(failed, list)


# ---------------------------------------------------------------------------
# with_secure_temp_directory (decorator)
# ---------------------------------------------------------------------------

def test_with_secure_temp_directory_decorator():
    @with_secure_temp_directory
    def my_func(temp_dir=None):
        assert temp_dir is not None
        assert temp_dir.exists()
        return "done"

    result = my_func()
    assert result == "done"
