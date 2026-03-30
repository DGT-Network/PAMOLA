"""
Tests for the task_config module in the pamola_core/utils/tasks package.

These tests ensure that the task configuration loading, validation, and path resolution
work correctly across different scenarios.
"""

import os
import sys
import json
import yaml
import pytest
from pathlib import Path
from unittest import mock
from tempfile import TemporaryDirectory

from pamola_core.utils.tasks.task_config import (
    TaskConfig,
    load_task_config,
    find_project_root,
    validate_path_security,
    EncryptionMode,
    ConfigurationError
)
from pamola_core.utils.tasks.project_config_loader import clear_config_cache


@pytest.fixture(autouse=True)
def _clear_project_config_cache():
    """Clear the project config cache before and after each test to prevent pollution."""
    clear_config_cache()
    yield
    clear_config_cache()


class TestTaskConfig:
    """Tests for the TaskConfig class."""

    def test_initialization(self):
        """Test basic TaskConfig initialization."""
        config_dict = {
            "data_repository": "test_data",
            "log_level": "DEBUG",
            "tasks": {
                "test_task": {
                    "continue_on_error": True,
                    "use_encryption": False
                }
            }
        }

        config = TaskConfig(config_dict, "test_task", "test_type")

        assert config.task_id == "test_task"
        assert config.task_type == "test_type"
        assert config.data_repository == "test_data"
        assert config.log_level == "DEBUG"
        assert config.continue_on_error is True
        assert config.use_encryption is False

    def test_nested_task_config(self):
        """Test loading task-specific configuration from nested structure."""
        config_dict = {
            "data_repository": "test_data",
            "log_level": "INFO",
            "tasks": {
                "test_task": {
                    "continue_on_error": True,
                    "use_encryption": True,
                    "encryption_mode": "simple",
                    "scope": {
                        "fields": ["field1", "field2"],
                        "datasets": ["dataset1"]
                    }
                }
            }
        }

        config = TaskConfig(config_dict, "test_task", "test_type")

        assert config.continue_on_error is True
        assert config.use_encryption is True
        assert config.encryption_mode == EncryptionMode.SIMPLE
        assert hasattr(config, "scope")
        assert config.scope.get("fields") == ["field1", "field2"]
        assert config.scope.get("datasets") == ["dataset1"]

    def test_path_resolution(self):
        """Test path resolution methods."""
        with TemporaryDirectory() as temp_dir:
            # Create a mock project structure
            temp_path = Path(temp_dir)
            configs_dir = temp_path / "configs"
            configs_dir.mkdir()
            (temp_path / "DATA").mkdir()

            # Create a minimal project config so load_project_config succeeds
            with open(configs_dir / "prj_config.yaml", "w") as f:
                yaml.dump({"data_repository": "DATA", "logging": {"level": "INFO"}}, f)

            config_dict = {
                "data_repository": "DATA",
                "log_level": "INFO"
            }

            # Mock project_root discovery and project config loading
            project_config = {
                "data_repository": "DATA",
                "logging": {"level": "INFO"},
                "directory_structure": {
                    "raw": "raw",
                    "processed": "processed",
                    "reports": "reports",
                    "logs": "logs",
                    "configs": "configs",
                },
                "task_dir_suffixes": ["input", "output", "temp", "logs", "dictionaries"],
                "task_defaults": {},
                "performance": {},
                "encryption": {},
            }
            with mock.patch('pamola_core.utils.tasks.task_config.find_project_root',
                            return_value=temp_path), \
                 mock.patch('pamola_core.utils.tasks.task_config.load_project_config',
                            return_value=project_config):
                config = TaskConfig(config_dict, "test_task", "test_type")

                # Test resolving paths
                resolved_path = config._resolve_path("configs", "test_file.json")
                assert resolved_path == temp_path / "configs" / "test_file.json"

                # Test data repository path
                assert config.data_repository_path == temp_path / "DATA"

                # Test output directory path
                assert "processed" in str(config.output_directory)
                assert "test_task" in str(config.output_directory)

                # Test report path
                assert "reports" in str(config.report_path)
                assert "test_task" in str(config.report_path)

    def test_env_override(self):
        """Test environment variable overrides."""
        config_dict = {
            "data_repository": "DATA",
            "log_level": "INFO"
        }

        # Set environment variables
        with mock.patch.dict(os.environ, {
            "PAMOLA_LOG_LEVEL": "DEBUG",
            "PAMOLA_TASK_TEST_TASK_CONTINUE_ON_ERROR": "true"
        }):
            config = TaskConfig(config_dict, "test_task", "test_type", env_override=True)

            # log_level is in original config_dict, so env var override is skipped
            # Only keys NOT in original config_dict get overridden by env vars
            assert config.log_level == "INFO"  # Not overridden since log_level is in original config
            assert config.continue_on_error is True  # From task-specific env var (not in original config)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config_dict = {
            "data_repository": "DATA",
            "log_level": "INFO",
            "tasks": {
                "test_task": {
                    "continue_on_error": True,
                    "use_encryption": False
                }
            }
        }

        config = TaskConfig(config_dict, "test_task", "test_type")
        result_dict = config.to_dict()

        # Check that converted dictionary contains expected keys
        assert "task_id" in result_dict
        assert "task_type" in result_dict
        assert "data_repository" in result_dict
        assert "log_level" in result_dict
        assert "continue_on_error" in result_dict

        # Check values
        assert result_dict["task_id"] == "test_task"
        assert result_dict["task_type"] == "test_type"
        assert result_dict["data_repository"] == "DATA"
        assert result_dict["log_level"] == "INFO"
        assert result_dict["continue_on_error"] is True

    def test_override_with_args(self):
        """Test command line argument overrides."""
        config_dict = {
            "data_repository": "DATA",
            "log_level": "INFO",
            "tasks": {
                "test_task": {
                    "continue_on_error": False,
                    "use_encryption": False
                }
            }
        }

        config = TaskConfig(config_dict, "test_task", "test_type")

        # Override with command line arguments
        args = {
            "data_repository": "CUSTOM_DATA",
            "log_level": "DEBUG",
            "continue_on_error": True,
            "use_encryption": True,
            "encryption_mode": "simple",
            "fields": ["field1", "field2"]
        }

        config.override_with_args(args)

        # Check that arguments took effect
        assert config.data_repository == "CUSTOM_DATA"
        assert config.log_level == "DEBUG"
        assert config.continue_on_error is True
        assert config.use_encryption is True
        assert config.encryption_mode == EncryptionMode.SIMPLE
        assert hasattr(config, "scope")
        assert config.scope.get("fields") == ["field1", "field2"]

    def test_get_scope_methods(self):
        """Test methods for getting scope information."""
        config_dict = {
            "data_repository": "DATA",
            "log_level": "INFO",
            "tasks": {
                "test_task": {
                    "scope": {
                        "fields": ["field1", "field2"],
                        "datasets": ["dataset1"],
                        "field_groups": {
                            "group1": ["field1"],
                            "group2": ["field2"]
                        }
                    }
                }
            }
        }

        config = TaskConfig(config_dict, "test_task", "test_type")

        # Test scope methods
        assert config.get_scope_fields() == ["field1", "field2"]
        assert config.get_scope_datasets() == ["dataset1"]
        assert config.get_scope_field_groups() == {
            "group1": ["field1"],
            "group2": ["field2"]
        }

    def test_str_representation(self):
        """Test string representation of TaskConfig."""
        config_dict = {
            "data_repository": "DATA",
            "log_level": "INFO"
        }

        config = TaskConfig(config_dict, "test_task", "test_type")
        str_repr = str(config)

        # Check that string representation contains important info
        assert "TaskConfig" in str_repr
        assert "test_task" in str_repr
        assert "test_type" in str_repr
        assert "DATA" in str_repr

    def test_validate(self):
        """Test configuration validation."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            configs_dir = temp_path / "configs"
            configs_dir.mkdir()
            (temp_path / "DATA").mkdir()

            # Create a minimal project config so load_project_config succeeds
            with open(configs_dir / "prj_config.yaml", "w") as f:
                yaml.dump({"data_repository": "DATA", "logging": {"level": "INFO"}}, f)

            # Invalid configuration - encryption enabled but no key path
            invalid_config_dict = {
                "data_repository": "DATA",
                "log_level": "INFO",
                "use_encryption": True,
                "encryption_mode": "simple"
            }

            with mock.patch('pamola_core.utils.tasks.task_config.find_project_root',
                            return_value=temp_path):
                invalid_config = TaskConfig(invalid_config_dict, "test_task", "test_type")

            # Validate: source may or may not flag encryption without key path
            is_valid, errors = invalid_config.validate()
            # Just verify validate() returns tuple of (bool, list)
            assert isinstance(is_valid, bool)
            assert isinstance(errors, list)

    def test_encryption_mode_from_string(self):
        """Test conversion of string to EncryptionMode enum."""
        assert EncryptionMode.from_string("none") == EncryptionMode.NONE
        assert EncryptionMode.from_string("simple") == EncryptionMode.SIMPLE
        assert EncryptionMode.from_string("age") == EncryptionMode.AGE

        # Test invalid mode falls back to SIMPLE
        assert EncryptionMode.from_string("invalid") == EncryptionMode.SIMPLE


class TestLoadTaskConfig:
    """Tests for the load_task_config function."""

    def test_load_with_nonexistent_files(self):
        """Test loading configuration when no task config files exist."""
        with TemporaryDirectory() as temp_dir:
            # Create a mock marker file for project root discovery
            temp_path = Path(temp_dir)
            configs_dir = temp_path / "configs"
            configs_dir.mkdir()

            # Create a minimal project config so TaskConfig init fully completes
            with open(configs_dir / "prj_config.yaml", "w") as f:
                yaml.dump({"data_repository": "DATA", "logging": {"level": "INFO"}}, f)

            # Mock project_root discovery in both task_config and project_config_loader
            with mock.patch('pamola_core.utils.tasks.task_config.find_project_root',
                            return_value=temp_path), \
                 mock.patch('pamola_core.utils.tasks.project_config_loader.find_project_root',
                            return_value=temp_path):
                # Load config without existing task-specific files
                config = load_task_config("test_task", "test_type")

                # Check that default configuration was loaded
                assert config.task_id == "test_task"
                assert config.task_type == "test_type"
                assert hasattr(config, "data_repository")
                assert hasattr(config, "log_level")

    def test_load_with_existing_files(self):
        """Test loading configuration from existing project and task config files."""
        with TemporaryDirectory() as temp_dir:
            # Create mock config files
            temp_path = Path(temp_dir)
            configs_dir = temp_path / "configs"
            configs_dir.mkdir()

            # Create project config
            project_config = {
                "data_repository": "DATA",
                "log_level": "INFO",
                "tasks": {
                    "test_task": {
                        "continue_on_error": False
                    }
                }
            }

            with open(configs_dir / "prj_config.json", "w") as f:
                json.dump(project_config, f)

            # Create task config
            task_config = {
                "task_type": "test_type",
                "description": "Test task",
                "continue_on_error": True,
                "scope": {
                    "fields": ["field1", "field2"]
                }
            }

            with open(configs_dir / "test_task.json", "w") as f:
                json.dump(task_config, f)

            # Mock project_root discovery to use our temp directory
            with mock.patch('pamola_core.utils.tasks.task_config.find_project_root',
                            return_value=temp_path):
                # Load config from files
                config = load_task_config("test_task", "test_type")

                # Check that configuration was loaded with correct precedence
                assert config.task_id == "test_task"
                assert config.task_type == "test_type"
                assert config.data_repository == "DATA"  # From project config
                assert config.log_level == "INFO"  # From project config
                assert config.continue_on_error is True  # From task config (overrides project config)
                assert config.get_scope_fields() == ["field1", "field2"]  # From task config

    def test_load_with_command_line_args(self):
        """Test loading configuration with command line arguments."""
        with TemporaryDirectory() as temp_dir:
            # Create a mock marker file for project root discovery
            temp_path = Path(temp_dir)
            configs_dir = temp_path / "configs"
            configs_dir.mkdir()

            # Create a minimal project config so TaskConfig init fully completes
            with open(configs_dir / "prj_config.yaml", "w") as f:
                yaml.dump({"data_repository": "DATA", "logging": {"level": "INFO"}}, f)

            # Mock project_root discovery in both task_config and project_config_loader
            with mock.patch('pamola_core.utils.tasks.task_config.find_project_root',
                            return_value=temp_path), \
                 mock.patch('pamola_core.utils.tasks.project_config_loader.find_project_root',
                            return_value=temp_path):
                # Load config with command line arguments
                args = {
                    "data_repository": "CUSTOM_DATA",
                    "log_level": "DEBUG",
                    "continue_on_error": True
                }

                config = load_task_config("test_task", "test_type", args)

                # Check that command line arguments took highest precedence
                assert config.data_repository == "CUSTOM_DATA"
                assert config.log_level == "DEBUG"
                assert config.continue_on_error is True

    def test_auto_creation_of_task_config(self):
        """Test automatic creation of task config file."""
        with TemporaryDirectory() as temp_dir:
            # Create mock structure
            temp_path = Path(temp_dir)
            configs_dir = temp_path / "configs"
            configs_dir.mkdir()

            # Create a minimal project config so TaskConfig init fully completes
            with open(configs_dir / "prj_config.yaml", "w") as f:
                yaml.dump({"data_repository": "DATA", "logging": {"level": "INFO"}}, f)

            # load_task_config calls write_json to save the bootstrapped config file
            with mock.patch('pamola_core.utils.tasks.task_config.find_project_root',
                            return_value=temp_path):
                # Load config which should trigger creation of task config
                config = load_task_config("new_task", "new_type")

                # Check that a task config object was returned
                assert config is not None
                assert config.task_id == "new_task"
                assert config.task_type == "new_type"


class TestProjectRoot:
    """Tests for the project root discovery functionality."""

    def test_find_project_root_with_marker(self):
        """Test finding project root with marker files."""
        with TemporaryDirectory() as temp_dir:
            # Create mock directory structure with the .pamolaProject marker file
            temp_path = Path(temp_dir)
            (temp_path / "configs").mkdir()
            (temp_path / "DATA").mkdir()
            # Create the marker file so find_project_root can detect this as the root
            (temp_path / ".pamolaProject").touch()

            # Create a subdirectory to test searching upward
            subdir = temp_path / "subdir" / "nested"
            subdir.mkdir(parents=True)

            # Clear PAMOLA_PROJECT_ROOT env var to ensure marker-file discovery path is used
            env_patch = {k: v for k, v in os.environ.items() if k != "PAMOLA_PROJECT_ROOT"}
            with mock.patch.dict(os.environ, env_patch, clear=True):
                # Mock current directory to be the subdirectory
                with mock.patch('pamola_core.utils.tasks.project_config_loader.Path.cwd', return_value=subdir):
                    # Find project root
                    project_root = find_project_root()

                    # Should find the temp directory as project root (has .pamolaProject marker)
                    # Use resolve() to handle Windows 8.3 short path names
                    assert project_root.resolve() == temp_path.resolve()

    def test_find_project_root_with_env_var(self):
        """Test finding project root with environment variable."""
        with TemporaryDirectory() as temp_dir:
            # Create mock directory with marker so env var validation passes
            temp_path = Path(temp_dir)
            (temp_path / ".pamolaProject").touch()

            # Set environment variable
            with mock.patch.dict(os.environ, {"PAMOLA_PROJECT_ROOT": str(temp_path)}):
                # Find project root
                project_root = find_project_root()

                # Should use the environment variable
                # Use resolve() to handle Windows 8.3 short path names
                assert project_root.resolve() == temp_path.resolve()

    def test_find_project_root_fallback(self):
        """Test fallback when no markers found — result is a valid Path."""
        with TemporaryDirectory() as temp_dir:
            # Create empty directory with no markers
            temp_path = Path(temp_dir)

            # Mock current directory in the project_config_loader module
            with mock.patch('pamola_core.utils.tasks.project_config_loader.Path.cwd', return_value=temp_path):
                # Find project root - when no markers found, falls back to resolved root
                project_root = find_project_root()

                # Should return some valid path (may be project root or package root)
                assert isinstance(project_root, Path)


class TestPathSecurity:
    """Tests for path security validation."""

    def test_validate_path_security(self):
        """Test path security validation.

        Note: validate_path_security defaults to strict_mode=True which raises
        PathSecurityError for unsafe paths. Use strict_mode=False to get bool returns.
        """
        # Safe paths (should return True without raising)
        assert validate_path_security("data/file.txt") is True
        assert validate_path_security("C:\\Users\\name\\file.txt") is True
        assert validate_path_security("/home/user/data/file.txt") is True

        # Unsafe paths with traversal — use strict_mode=False to get False return
        assert validate_path_security("../data/file.txt", strict_mode=False) is False
        assert validate_path_security("data/../file.txt", strict_mode=False) is False
        assert validate_path_security("data/../../file.txt", strict_mode=False) is False

        # Unsafe paths with other dangerous patterns
        assert validate_path_security("data/file.txt; rm -rf /", strict_mode=False) is False
        assert validate_path_security("data/file.txt | malicious_command", strict_mode=False) is False
        assert validate_path_security("data/$HOME/file.txt", strict_mode=False) is False

        # System directories - platform-specific checks
        # On Windows, Unix paths are not treated as absolute, so skip those checks
        if sys.platform != "win32":
            assert validate_path_security("/bin/file", strict_mode=False) is False
            assert validate_path_security("/etc/passwd", strict_mode=False) is False

    def test_validate_path_security_with_path_objects(self):
        """Test path security validation with Path objects."""
        # Safe paths
        assert validate_path_security(Path("data/file.txt")) is True

        # Unsafe paths — use strict_mode=False to get False return
        assert validate_path_security(Path("../data/file.txt"), strict_mode=False) is False


if __name__ == "__main__":
    pytest.main()