"""
Tests for the directory_manager module in the pamola_core/utils/tasks package.

These tests ensure that the TaskDirectoryManager class properly implements directory creation, path resolution, artifact management, cleanup, and error handling.
"""

import os
import shutil
import tempfile
import pytest
import logging
from pathlib import Path
from unittest import mock
from pamola_core.utils.tasks.directory_manager import (
    TaskDirectoryManager,
    DirectoryManagerError,
    PathValidationError,
    DirectoryCreationError,
    create_directory_manager,
)

# --- Fixtures and Mocks ---

class DummyTaskConfig:
    def __init__(self, task_id="test_task", project_root=None, log_directory=None, suffixes=None, clean_temp_on_exit=False):
        self.task_id = task_id
        self.project_root = Path(project_root or tempfile.gettempdir())
        self._task_dir = self.project_root / "DATA" / "processed" / self.task_id
        self.log_directory = log_directory
        self.task_dir_suffixes = suffixes or ["input", "output", "temp", "logs"]
        self.clean_temp_on_exit = clean_temp_on_exit
    def get_task_dir(self):
        return self._task_dir
    def get_reports_dir(self):
        return self.project_root / "reports"

class DummyProgressManager:
    def __init__(self):
        self.logs = []
    def create_operation_context(self, name, total, description=None, unit=None, leave=None):
        return mock.MagicMock().__enter__.return_value
    def log_info(self, msg):
        self.logs.append(("info", msg))
    def log_error(self, msg):
        self.logs.append(("error", msg))
    def log_warning(self, msg):
        self.logs.append(("warning", msg))
    def log_debug(self, msg):
        self.logs.append(("debug", msg))

@pytest.fixture
def temp_root(tmp_path):
    return tmp_path

@pytest.fixture
def dummy_config(temp_root):
    return DummyTaskConfig(project_root=temp_root)

@pytest.fixture
def directory_mgr(dummy_config):
    return TaskDirectoryManager(dummy_config, logger=logging.getLogger("test"))

@pytest.fixture
def directory_mgr_with_progress(dummy_config):
    return TaskDirectoryManager(dummy_config, logger=logging.getLogger("test"), progress_manager=DummyProgressManager())

# --- Tests for TaskDirectoryManager ---

class TestTaskDirectoryManager:
    def test_initialization_and_resolve_task_dir(self, directory_mgr):
        assert directory_mgr.task_id == "test_task"
        assert directory_mgr.task_dir.name == "test_task"
        assert directory_mgr.project_root.exists()

    def test_ensure_directories_creates_all(self, directory_mgr):
        dirs = directory_mgr.ensure_directories()
        for suffix in directory_mgr.directory_suffixes:
            assert (directory_mgr.task_dir / suffix).exists()
            assert dirs[suffix] == directory_mgr.task_dir / suffix
        assert dirs["task"] == directory_mgr.task_dir

    def test_ensure_directories_with_progress(self, directory_mgr_with_progress):
        dirs = directory_mgr_with_progress.ensure_directories()
        for suffix in directory_mgr_with_progress.directory_suffixes:
            assert (directory_mgr_with_progress.task_dir / suffix).exists()
            assert dirs[suffix] == directory_mgr_with_progress.task_dir / suffix
        assert dirs["task"] == directory_mgr_with_progress.task_dir

    def test_get_directory_valid_and_invalid(self, directory_mgr):
        directory_mgr.ensure_directories()
        assert directory_mgr.get_directory("input").exists()
        with pytest.raises(DirectoryManagerError):
            directory_mgr.get_directory("not_a_dir")

    def test_get_artifact_path_and_creation(self, directory_mgr):
        directory_mgr.ensure_directories()
        path = directory_mgr.get_artifact_path("artifact", artifact_type="txt", subdir="output", include_timestamp=False)
        assert path.parent.name == "output"
        assert path.name.startswith("artifact.")
        # Should create subdir if missing
        new_path = directory_mgr.get_artifact_path("new_artifact", subdir="newdir", include_timestamp=False)
        assert new_path.parent.exists()
        assert new_path.name.startswith("new_artifact.")

    def test_get_artifact_path_invalid_type(self, directory_mgr):
        directory_mgr.ensure_directories()
        with pytest.raises(PathValidationError):
            with mock.patch("pamola_core.utils.tasks.directory_manager.validate_path_security", return_value=False):
                directory_mgr.get_artifact_path("bad", subdir="output")

    def test_clean_temp_directory_standard(self, directory_mgr):
        directory_mgr.ensure_directories()
        temp_dir = directory_mgr.get_directory("temp")
        # Create files and dirs
        (temp_dir / "file1.txt").write_text("abc")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").write_text("def")
        assert any(temp_dir.iterdir())
        assert directory_mgr.clean_temp_directory() is True
        assert not any(temp_dir.iterdir())

    def test_clean_temp_directory_with_progress(self, directory_mgr_with_progress):
        directory_mgr_with_progress.ensure_directories()
        temp_dir = directory_mgr_with_progress.get_directory("temp")
        (temp_dir / "file1.txt").write_text("abc")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").write_text("def")
        assert directory_mgr_with_progress.clean_temp_directory() is True
        assert not any(temp_dir.iterdir())

    def test_clean_temp_directory_missing(self, directory_mgr):
        # Remove temp dir
        temp_dir = directory_mgr.get_directory("temp")
        shutil.rmtree(temp_dir)
        assert directory_mgr.clean_temp_directory() is True

    def test_get_timestamped_filename(self, directory_mgr):
        name = directory_mgr.get_timestamped_filename("foo", extension="csv")
        assert name.startswith("foo_") and name.endswith(".csv")

    def test_validate_directory_structure(self, directory_mgr):
        directory_mgr.ensure_directories()
        results = directory_mgr.validate_directory_structure()
        for k, v in results.items():
            assert v is True

    def test_list_artifacts(self, directory_mgr):
        directory_mgr.ensure_directories()
        out_dir = directory_mgr.get_directory("output")
        f1 = out_dir / "a.txt"
        f2 = out_dir / "b.txt"
        f1.write_text("1")
        f2.write_text("2")
        files = directory_mgr.list_artifacts(subdir="output", pattern="*.txt")
        assert f1 in files and f2 in files

    def test_list_artifacts_missing_dir(self, directory_mgr):
        files = directory_mgr.list_artifacts(subdir="notadir")
        assert files == []

    def test_import_external_file_valid(self, directory_mgr, tmp_path):
        directory_mgr.ensure_directories()
        src = tmp_path / "src.txt"
        src.write_text("data")
        imported = directory_mgr.import_external_file(src, subdir="input")
        assert imported.exists()
        assert imported.read_text() == "data"

    def test_import_external_file_invalid_path(self, directory_mgr, tmp_path):
        directory_mgr.ensure_directories()
        src = tmp_path / "src.txt"
        src.write_text("data")
        with mock.patch("pamola_core.utils.tasks.directory_manager.validate_path_security", return_value=False):
            with pytest.raises(PathValidationError):
                directory_mgr.import_external_file(src, subdir="input")

    def test_import_external_file_missing(self, directory_mgr):
        directory_mgr.ensure_directories()
        with pytest.raises(DirectoryManagerError):
            directory_mgr.import_external_file("/not/exist.txt", subdir="input")

    def test_import_external_file_raises_on_copy(self, directory_mgr, tmp_path):
        directory_mgr.ensure_directories()
        src = tmp_path / "src.txt"
        src.write_text("data")
        with mock.patch("shutil.copy2", side_effect=Exception("fail")):
            with pytest.raises(DirectoryManagerError):
                directory_mgr.import_external_file(src, subdir="input")

    def test_normalize_and_validate_path(self, directory_mgr):
        directory_mgr.ensure_directories()
        rel = directory_mgr.normalize_and_validate_path("input/foo.txt")
        abs_path = directory_mgr.normalize_and_validate_path(rel)
        assert rel.is_absolute() and abs_path.is_absolute()

    def test_normalize_and_validate_path_invalid(self, directory_mgr):
        with mock.patch("pamola_core.utils.tasks.directory_manager.validate_path_security", return_value=False):
            with pytest.raises(PathValidationError):
                directory_mgr.normalize_and_validate_path("input/foo.txt")

    def test_cleanup(self, directory_mgr):
        directory_mgr.config.clean_temp_on_exit = True
        directory_mgr.ensure_directories()
        temp_dir = directory_mgr.get_directory("temp")
        (temp_dir / "file.txt").write_text("abc")
        assert directory_mgr.cleanup() is True
        assert not any(temp_dir.iterdir())

    def test_context_manager_enter_exit(self, directory_mgr):
        directory_mgr.config.clean_temp_on_exit = True
        with directory_mgr as mgr:
            temp_dir = mgr.get_directory("temp")
            (temp_dir / "file.txt").write_text("abc")
        assert not any(temp_dir.iterdir())

    def test_create_directory_manager_helper(self, dummy_config):
        mgr = create_directory_manager(dummy_config, logger=logging.getLogger("test"))
        assert isinstance(mgr, TaskDirectoryManager)
        # Error case
        with mock.patch("pamola_core.utils.tasks.directory_manager.ensure_directory", side_effect=Exception("fail")):
            with pytest.raises(DirectoryManagerError):
                create_directory_manager(dummy_config, logger=logging.getLogger("test"))

    def test_resolve_task_dir_raises_exception(self, directory_mgr, monkeypatch):
        # Simulate get_task_dir raising an exception
        monkeypatch.setattr(directory_mgr.config, "get_task_dir", lambda: (_ for _ in ()).throw(Exception("fail")))
        directory_mgr._initialized = False
        with pytest.raises(DirectoryManagerError):
            directory_mgr._resolve_task_dir()

    def test_ensure_directories_with_progress_path_security_error(self, directory_mgr_with_progress, monkeypatch):
        monkeypatch.setattr("pamola_core.utils.tasks.directory_manager.validate_path_security", lambda path: False)
        with pytest.raises(DirectoryCreationError):
            directory_mgr_with_progress._ensure_directories_with_progress()

    def test_ensure_directories_with_progress_generic_exception(self, directory_mgr_with_progress, monkeypatch):
        monkeypatch.setattr("pamola_core.utils.tasks.directory_manager.ensure_directory", lambda path: (_ for _ in ()).throw(Exception("fail")))
        with pytest.raises(DirectoryCreationError):
            directory_mgr_with_progress._ensure_directories_with_progress()

    def test_ensure_directories_standard_path_security_error(self, directory_mgr, monkeypatch):
        monkeypatch.setattr("pamola_core.utils.tasks.directory_manager.validate_path_security", lambda path: False)
        with pytest.raises(DirectoryCreationError):
            directory_mgr._ensure_directories_standard()

    def test_ensure_directories_standard_generic_exception(self, directory_mgr, monkeypatch):
        monkeypatch.setattr("pamola_core.utils.tasks.directory_manager.ensure_directory", lambda path: (_ for _ in ()).throw(Exception("fail")))
        with pytest.raises(DirectoryCreationError):
            directory_mgr._ensure_directories_standard()

    def test_get_artifact_path_ensure_directory_raises(self, directory_mgr, monkeypatch):
        monkeypatch.setattr("pamola_core.utils.tasks.directory_manager.ensure_directory", lambda path: (_ for _ in ()).throw(Exception("fail")))
        with pytest.raises(Exception):
            directory_mgr.get_artifact_path("artifact", subdir="newdir2")

    def test_clean_temp_directory_exception(self, directory_mgr, monkeypatch):
        monkeypatch.setattr(directory_mgr, "get_directory", lambda name: (_ for _ in ()).throw(Exception("fail")))
        assert directory_mgr.clean_temp_directory() is False

    def test_clean_temp_directory_with_progress_item_removal_error(self, directory_mgr_with_progress, monkeypatch):
        directory_mgr_with_progress.ensure_directories()
        temp_dir = directory_mgr_with_progress.get_directory("temp")
        (temp_dir / "file1.txt").write_text("abc")
        orig_unlink = Path.unlink
        def bad_unlink(self):
            raise Exception("fail")
        monkeypatch.setattr(Path, "unlink", bad_unlink)
        assert directory_mgr_with_progress.clean_temp_directory() is False
        monkeypatch.setattr(Path, "unlink", orig_unlink)

    def test_clean_temp_directory_standard_item_removal_error(self, directory_mgr, monkeypatch):
        directory_mgr.ensure_directories()
        temp_dir = directory_mgr.get_directory("temp")
        (temp_dir / "file1.txt").write_text("abc")
        orig_unlink = Path.unlink
        def bad_unlink(self):
            raise Exception("fail")
        monkeypatch.setattr(Path, "unlink", bad_unlink)
        assert directory_mgr.clean_temp_directory() is False
        monkeypatch.setattr(Path, "unlink", orig_unlink)

    def test_list_artifacts_exception(self, directory_mgr, monkeypatch):
        monkeypatch.setattr(directory_mgr, "get_directory", lambda subdir: (_ for _ in ()).throw(Exception("fail")))
        assert directory_mgr.list_artifacts("output") == []

    def test_cleanup_exception(self, directory_mgr, monkeypatch):
        directory_mgr.config.clean_temp_on_exit = True
        monkeypatch.setattr(directory_mgr, "clean_temp_directory", lambda: (_ for _ in ()).throw(Exception("fail")))
        with pytest.raises(Exception):
            directory_mgr.cleanup()

    def test_create_directory_manager_no_logger_progress_manager(self, monkeypatch, dummy_config):
        # Remove logger and progress_manager, simulate error
        monkeypatch.setattr("pamola_core.utils.tasks.directory_manager.TaskDirectoryManager.__init__", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))
        from pamola_core.utils.tasks import directory_manager
        with pytest.raises(DirectoryManagerError):
            directory_manager.create_directory_manager(dummy_config, logger=None, progress_manager=None)

if __name__ == "__main__":
    pytest.main()
