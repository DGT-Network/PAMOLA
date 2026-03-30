"""
Unit tests for pamola_core.configs.settings module.

Tests cover:
- Configuration loading and caching
- Default configuration values
- Nested dictionary updates
- Environment variable support
- Data repository detection
- Configuration persistence

Run with: pytest -s tests/configs/test_settings.py
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock


from pamola_core.configs.settings import (
    load_config,
    get_config,
    get_data_repository,
    set_data_repository,
    get_directory_structure,
    get_performance_settings,
    get_logging_settings,
    save_config,
    update_nested_dict,
    get_config_file_paths,
    DEFAULT_CONFIG,
)


class TestDefaultConfig:
    """Test default configuration structure."""

    def test_default_config_has_required_keys(self):
        """DEFAULT_CONFIG should have required top-level keys."""
        required = {"data_repository", "directory_structure", "logging", "performance"}
        assert required.issubset(set(DEFAULT_CONFIG.keys()))

    def test_directory_structure_defaults(self):
        """Directory structure should have default values."""
        dirs = DEFAULT_CONFIG["directory_structure"]
        required = {"raw", "processed", "logs", "configs"}
        assert required.issubset(set(dirs.keys()))

    def test_logging_defaults(self):
        """Logging config should have required keys."""
        logging_cfg = DEFAULT_CONFIG["logging"]
        required = {"level", "file", "format"}
        assert required.issubset(set(logging_cfg.keys()))

    def test_performance_defaults(self):
        """Performance config should have required keys."""
        perf = DEFAULT_CONFIG["performance"]
        required = {"chunk_size", "default_encoding", "default_delimiter"}
        assert required.issubset(set(perf.keys()))


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_config_returns_dict(self):
        """load_config should return dict."""
        # Reset global config first
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        config = load_config()
        assert isinstance(config, dict)

    def test_load_config_has_defaults(self):
        """Loaded config should include default values."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        config = load_config()
        assert "directory_structure" in config
        assert "logging" in config
        assert "performance" in config

    def test_load_config_caching(self):
        """Repeated calls should return cached config."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        config1 = load_config()
        config2 = load_config()
        assert config1 is config2

    def test_load_config_from_file(self):
        """Should load configuration from file."""
        import pamola_core.configs.settings as settings_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            custom_config = {
                "data_repository": "/custom/path",
                "performance": {"chunk_size": 50000}
            }
            config_file.write_text(json.dumps(custom_config))

            settings_mod._config = None
            config = load_config(config_file)

            assert config["data_repository"] == "/custom/path"
            assert config["performance"]["chunk_size"] == 50000

    def test_load_config_missing_file_uses_defaults(self):
        """Should use defaults if config file doesn't exist."""
        import pamola_core.configs.settings as settings_mod

        settings_mod._config = None
        config = load_config("/nonexistent/path/config.json")

        assert isinstance(config, dict)
        assert "performance" in config

    def test_load_config_invalid_json_uses_defaults(self):
        """Should use defaults if JSON is invalid."""
        import pamola_core.configs.settings as settings_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "bad_config.json"
            config_file.write_text("invalid json {[")

            settings_mod._config = None
            config = load_config(config_file)

            assert isinstance(config, dict)
            assert "performance" in config


class TestGetConfig:
    """Test get_config() function."""

    def test_get_config_returns_dict(self):
        """get_config should return dict."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        config = get_config()
        assert isinstance(config, dict)

    def test_get_config_lazy_loads(self):
        """get_config should lazy-load if not cached."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        config = get_config()
        assert config is not None

    def test_get_config_caching(self):
        """Repeated calls should return same object."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestDataRepository:
    """Test data repository management."""

    def test_get_data_repository_returns_path(self):
        """get_data_repository should return Path object."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        repo = get_data_repository()
        assert isinstance(repo, Path)

    def test_set_data_repository(self):
        """Should set data repository path."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        test_path = "/test/data/repo"
        set_data_repository(test_path)

        config = get_config()
        # Compare as Path objects to handle OS-specific path separators
        assert Path(config["data_repository"]) == Path(test_path)

    def test_set_data_repository_with_path_object(self):
        """Should accept Path objects."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        test_path = Path("/test/data/repo")
        set_data_repository(test_path)

        repo = get_data_repository()
        assert isinstance(repo, Path)
        assert repo.name == "repo"


class TestGetDirectoryStructure:
    """Test directory structure access."""

    def test_get_directory_structure_returns_dict(self):
        """Should return directory structure dict."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        dirs = get_directory_structure()
        assert isinstance(dirs, dict)

    def test_directory_structure_has_defaults(self):
        """Directory structure should have expected keys."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        dirs = get_directory_structure()
        assert "raw" in dirs
        assert "processed" in dirs
        assert "logs" in dirs
        assert "configs" in dirs


class TestGetPerformanceSettings:
    """Test performance settings access."""

    def test_get_performance_settings_returns_dict(self):
        """Should return performance settings dict."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        perf = get_performance_settings()
        assert isinstance(perf, dict)

    def test_performance_settings_has_defaults(self):
        """Performance settings should have expected keys."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        perf = get_performance_settings()
        assert "chunk_size" in perf
        assert "default_encoding" in perf
        assert "default_delimiter" in perf


class TestGetLoggingSettings:
    """Test logging settings access."""

    def test_get_logging_settings_returns_dict(self):
        """Should return logging settings dict."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        logging_cfg = get_logging_settings()
        assert isinstance(logging_cfg, dict)

    def test_logging_settings_has_defaults(self):
        """Logging settings should have expected keys."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        logging_cfg = get_logging_settings()
        assert "level" in logging_cfg
        assert "file" in logging_cfg
        assert "format" in logging_cfg


class TestUpdateNestedDict:
    """Test nested dictionary update function."""

    def test_update_flat_dict(self):
        """Should update flat dictionary."""
        base = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        result = update_nested_dict(base, update)

        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_update_nested_dict(self):
        """Should update nested dictionaries."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        update = {"a": {"y": 20, "z": 30}}
        result = update_nested_dict(base, update)

        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 20
        assert result["a"]["z"] == 30
        assert result["b"] == 3

    def test_update_deep_nested_dict(self):
        """Should update deeply nested dictionaries."""
        base = {"a": {"b": {"c": 1}}}
        update = {"a": {"b": {"c": 2, "d": 3}}}
        result = update_nested_dict(base, update)

        assert result["a"]["b"]["c"] == 2
        assert result["a"]["b"]["d"] == 3

    def test_update_overwrites_non_dict(self):
        """Should overwrite non-dict values with dicts."""
        base = {"a": "string"}
        update = {"a": {"nested": "value"}}
        result = update_nested_dict(base, update)

        assert isinstance(result["a"], dict)
        assert result["a"]["nested"] == "value"

    def test_update_preserves_base(self):
        """Should not modify base dict."""
        base = {"a": 1}
        update = {"b": 2}
        result = update_nested_dict(base, update)

        assert "b" not in base
        assert "b" in result


class TestSaveConfig:
    """Test configuration persistence."""

    def test_save_config_creates_file(self):
        """Should create config file."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            saved_path = save_config(config_path)

            assert saved_path.exists()
            assert saved_path.name == "config.json"

    def test_save_config_valid_json(self):
        """Saved config should be valid JSON."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            save_config(config_path)

            # Should be able to load as JSON
            content = json.loads(config_path.read_text())
            assert isinstance(content, dict)

    def test_save_config_includes_current_settings(self):
        """Saved config should include current settings."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            save_config(config_path)

            saved = json.loads(config_path.read_text())
            assert "performance" in saved
            assert "logging" in saved
            assert "directory_structure" in saved


class TestGetConfigFilePaths:
    """Test configuration file path resolution."""

    def test_get_config_file_paths_returns_list(self):
        """Should return list of Path objects."""
        paths = get_config_file_paths()
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_config_paths_from_env_included(self):
        """Should include PAMOLA_CONFIG_PATH if set."""
        env_path = "/env/config.json"
        with mock.patch.dict(os.environ, {"PAMOLA_CONFIG_PATH": env_path}):
            paths = get_config_file_paths()
            # Compare normalised Path objects OR raw string suffixes to handle
            # OS-specific path separators (e.g. Windows turns /env/ into \env\)
            env_path_obj = Path(env_path)
            assert any(
                p == env_path_obj or str(p).endswith("env" + os.sep + "config.json")
                for p in paths
            )

    def test_config_paths_not_empty(self):
        """Should return at least one config path."""
        paths = get_config_file_paths()
        assert len(paths) > 0


class TestEnvironmentVariableSupport:
    """Test environment variable configuration."""

    def test_data_repository_from_env(self):
        """Should use PAMOLA_DATA_REPOSITORY env var when no config file sets data_repository."""
        import pamola_core.configs.settings as settings_mod
        settings_mod._config = None

        # Patch get_config_file_paths to return no existing files so env var is used
        with mock.patch("pamola_core.configs.settings.get_config_file_paths", return_value=[]):
            with mock.patch.dict(os.environ, {"PAMOLA_DATA_REPOSITORY": "/env/data"}):
                settings_mod._config = None
                config = load_config()
                assert config["data_repository"] == "/env/data"

    def test_config_path_from_env(self):
        """Should use PAMOLA_CONFIG_PATH env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "custom.json"
            config_file.write_text(json.dumps({"data_repository": "/custom"}))

            with mock.patch.dict(os.environ, {"PAMOLA_CONFIG_PATH": str(config_file)}):
                import pamola_core.configs.settings as settings_mod
                settings_mod._config = None
                config = load_config()

                assert config["data_repository"] == "/custom"
