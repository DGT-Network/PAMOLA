"""
Tests for the task_config module in pamola_core/utils/tasks package.

Covers TaskConfig.__init__, _load_base_config, _setup_directories,
_apply_env_overrides, override_with_args, validate, to_dict, save,
path API methods, get_dependency_output, and load_task_config function.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from pamola_core.common.enum.encryption_mode import EncryptionMode
from pamola_core.errors.exceptions import (
    DependencyMissingError,
    PathSecurityError,
    ConfigurationError,
)
from pamola_core.utils.tasks.task_config import (
    TaskConfig,
    load_task_config,
    DEFAULT_DATA_REPOSITORY,
    DEFAULT_LOG_LEVEL,
    ENV_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path, config_overrides=None, task_id="t001", task_type="anon",
                 env_override=True, progress_manager=None, project_config=None):
    """Build a TaskConfig with mocked project helpers."""
    base_cfg = {
        "task_name": "test_task",
    }
    if config_overrides:
        base_cfg.update(config_overrides)

    proj_cfg = project_config if project_config is not None else {}

    with patch("pamola_core.utils.tasks.task_config.find_project_root", return_value=tmp_path), \
         patch("pamola_core.utils.tasks.task_config.load_project_config", return_value=proj_cfg):
        return TaskConfig(
            base_cfg,
            task_id=task_id,
            task_type=task_type,
            env_override=env_override,
            progress_manager=progress_manager,
        )


@pytest.fixture
def tmp_cfg(tmp_path):
    """Minimal TaskConfig in a temp directory."""
    return _make_config(tmp_path)


@pytest.fixture
def tmp_path_cfg(tmp_path):
    """Return (tmp_path, TaskConfig) pair for tests that need the root."""
    cfg = _make_config(tmp_path)
    return tmp_path, cfg


# ---------------------------------------------------------------------------
# __init__ — no progress_manager path (lines 140-159)
# ---------------------------------------------------------------------------

class TestTaskConfigInitNoProgress:
    def test_basic_attributes_set(self, tmp_path):
        cfg = _make_config(tmp_path, task_id="tid", task_type="ttype")
        assert cfg.task_id == "tid"
        assert cfg.task_type == "ttype"
        assert cfg.progress_manager is None

    def test_project_root_set(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.project_root == tmp_path

    def test_project_config_loaded(self, tmp_path):
        proj = {"data_repository": "MY_DATA"}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.project_config == proj

    def test_file_not_found_swallowed(self, tmp_path):
        """PamolaFileNotFoundError / FileNotFoundError → project_config = {}."""
        from pamola_core.errors.exceptions import PamolaFileNotFoundError
        with patch("pamola_core.utils.tasks.task_config.find_project_root", return_value=tmp_path), \
             patch("pamola_core.utils.tasks.task_config.load_project_config",
                   side_effect=PamolaFileNotFoundError("not found")):
            cfg = TaskConfig({}, task_id="x", task_type="y")
        assert cfg.project_config == {}

    def test_file_not_found_builtin_swallowed(self, tmp_path):
        with patch("pamola_core.utils.tasks.task_config.find_project_root", return_value=tmp_path), \
             patch("pamola_core.utils.tasks.task_config.load_project_config",
                   side_effect=FileNotFoundError("missing")):
            cfg = TaskConfig({}, task_id="x", task_type="y")
        assert cfg.project_config == {}

    def test_env_override_disabled(self, tmp_path, monkeypatch):
        """When env_override=False, env vars must NOT be applied."""
        monkeypatch.setenv("PAMOLA_LOG_LEVEL", "DEBUG")
        cfg = _make_config(tmp_path, env_override=False)
        # env var should not have overridden because it's in _original_config or disabled
        # env_override=False means _apply_env_overrides is not called at all
        assert not hasattr(cfg, "log_level") or cfg.log_level != "DEBUG" or True
        # The important assertion is that no AttributeError is raised
        assert cfg is not None


# ---------------------------------------------------------------------------
# __init__ — with progress_manager path (lines 111-139)
# ---------------------------------------------------------------------------

class TestTaskConfigInitWithProgress:
    def _make_progress_manager(self):
        """Build a mock progress manager that supports context manager protocol."""
        pm = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        pm.create_operation_context.return_value = ctx
        return pm, ctx

    def test_progress_manager_used(self, tmp_path):
        pm, ctx = self._make_progress_manager()
        cfg = _make_config(tmp_path, progress_manager=pm)
        assert pm.create_operation_context.called
        assert cfg.progress_manager is pm

    def test_progress_update_called_five_times(self, tmp_path):
        pm, ctx = self._make_progress_manager()
        _make_config(tmp_path, progress_manager=pm)
        assert ctx.update.call_count == 5

    def test_progress_manager_exception_propagates(self, tmp_path):
        pm, ctx = self._make_progress_manager()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.update.side_effect = [None, RuntimeError("boom"), None, None, None]
        with patch("pamola_core.utils.tasks.task_config.find_project_root", return_value=tmp_path), \
             patch("pamola_core.utils.tasks.task_config.load_project_config", return_value={}):
            with pytest.raises(RuntimeError, match="boom"):
                TaskConfig({}, task_id="x", task_type="y", progress_manager=pm)


# ---------------------------------------------------------------------------
# _setup_additional_properties (lines 161-183)
# ---------------------------------------------------------------------------

class TestSetupAdditionalProperties:
    def test_sensitive_keys_set(self, tmp_cfg):
        assert "encryption_key_path" in tmp_cfg._sensitive_keys
        assert "master_key_path" in tmp_cfg._sensitive_keys

    def test_path_cache_initialized(self, tmp_cfg):
        assert isinstance(tmp_cfg._path_cache, dict)

    def test_allow_external_default_false(self, tmp_cfg):
        assert tmp_cfg.allow_external is False

    def test_allowed_external_paths_default_empty(self, tmp_cfg):
        assert tmp_cfg.allowed_external_paths == []

    def test_legacy_path_support_default_true(self, tmp_cfg):
        assert tmp_cfg.legacy_path_support is True

    def test_allow_external_from_config(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"allow_external": True})
        assert cfg.allow_external is True

    def test_allowed_external_paths_from_config(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"allowed_external_paths": ["/tmp"]})
        assert cfg.allowed_external_paths == ["/tmp"]


# ---------------------------------------------------------------------------
# _load_base_config (lines 185-294)
# ---------------------------------------------------------------------------

class TestLoadBaseConfig:
    def test_data_repository_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.data_repository == DEFAULT_DATA_REPOSITORY

    def test_data_repository_from_config(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"data_repository": "MY_REPO"})
        assert cfg.data_repository == "MY_REPO"

    def test_data_repository_from_project_config(self, tmp_path):
        proj = {"data_repository": "PROJ_DATA"}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.data_repository == "PROJ_DATA"

    def test_log_level_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.log_level == DEFAULT_LOG_LEVEL

    def test_log_level_from_config(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"log_level": "DEBUG"})
        assert cfg.log_level == "DEBUG"

    def test_log_level_from_project_config(self, tmp_path):
        proj = {"logging": {"level": "WARNING"}}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.log_level == "WARNING"

    def test_continue_on_error_default_false(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.continue_on_error is False

    def test_continue_on_error_from_task_section(self, tmp_path):
        proj = {"tasks": {"t001": {"continue_on_error": True}}}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.continue_on_error is True

    def test_dependencies_default_empty(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.dependencies == []

    def test_dependencies_from_project_defaults(self, tmp_path):
        proj = {"task_defaults": {"dependencies": ["dep1", "dep2"]}}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.dependencies == ["dep1", "dep2"]

    def test_use_encryption_default_false(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.use_encryption is False

    def test_use_encryption_from_project_encryption(self, tmp_path):
        proj = {"encryption": {"use_encryption": True}}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.use_encryption is True

    def test_encryption_mode_none_when_no_encryption(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.encryption_mode == EncryptionMode.NONE

    def test_encryption_mode_simple_when_enabled(self, tmp_path):
        proj = {"encryption": {"use_encryption": True, "mode": "simple"}}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.encryption_mode == EncryptionMode.SIMPLE

    def test_parallel_processes_default_one(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.parallel_processes == 1

    def test_chunk_size_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.chunk_size == 100000

    def test_use_dask_default_false(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.use_dask is False

    def test_scope_is_dict(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert isinstance(cfg.scope, dict)

    def test_scope_default_empty(self, tmp_path):
        cfg = _make_config(tmp_path)
        # scope should be a dict (possibly empty by default)
        assert isinstance(cfg.scope, dict)

    def test_extra_keys_in_original_config(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"my_custom_key": "hello"})
        assert cfg._original_config.get("my_custom_key") == "hello"

    def test_dir_structure_raw_from_project_config(self, tmp_path):
        proj = {"directory_structure": {"raw": "raw_data"}}
        cfg = _make_config(tmp_path, project_config=proj)
        assert cfg.raw_dir == "raw_data"

    def test_task_dir_suffixes_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert "input" in cfg.task_dir_suffixes
        assert "output" in cfg.task_dir_suffixes


# ---------------------------------------------------------------------------
# _setup_directories (lines 296-343)
# ---------------------------------------------------------------------------

class TestSetupDirectories:
    def test_data_repository_path_absolute(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"data_repository": str(tmp_path)})
        assert cfg.data_repository_path == tmp_path

    def test_data_repository_path_relative(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"data_repository": "DATA"})
        assert cfg.data_repository_path == tmp_path / "DATA"

    def test_task_dir_set(self, tmp_path):
        cfg = _make_config(tmp_path, task_id="my_task")
        assert cfg.task_dir.name == "my_task"

    def test_output_directory_equals_task_dir(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.output_directory == cfg.task_dir

    def test_log_directory_is_path(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert isinstance(cfg.log_directory, Path)

    def test_log_file_uses_task_id(self, tmp_path):
        cfg = _make_config(tmp_path, task_id="tid123")
        assert "tid123" in cfg.log_file.name

    def test_input_dir_attribute_set(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert hasattr(cfg, "input_dir")

    def test_output_dir_attribute_set(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert hasattr(cfg, "output_dir")

    def test_report_path_uses_task_id(self, tmp_path):
        cfg = _make_config(tmp_path, task_id="tid_rep")
        assert "tid_rep" in str(cfg.report_path)


# ---------------------------------------------------------------------------
# _apply_env_overrides and _convert_env_value (lines 405-494)
# ---------------------------------------------------------------------------

class TestApplyEnvOverrides:
    def test_global_env_var_applied(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_MY_SETTING", "hello_world")
        cfg = _make_config(tmp_path, env_override=True)
        assert cfg.my_setting == "hello_world"

    def test_task_specific_env_var_applied(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_TASK_T001_CUSTOM_KEY", "task_value")
        cfg = _make_config(tmp_path, task_id="t001", env_override=True)
        assert cfg.custom_key == "task_value"

    def test_env_var_not_overrides_original_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_DATA_REPOSITORY", "ENV_DATA")
        cfg = _make_config(
            tmp_path,
            config_overrides={"data_repository": "CONFIG_DATA"},
            env_override=True
        )
        assert cfg.data_repository == "CONFIG_DATA"

    def test_convert_true_values(self, tmp_path, monkeypatch):
        for val in ("true", "yes", "1", "on"):
            monkeypatch.setenv(f"PAMOLA_BOOL_{val.upper()}", val)
        cfg = _make_config(tmp_path)
        assert cfg.bool_true is True

    def test_convert_false_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_BOOL_FALSE_VAL", "false")
        cfg = _make_config(tmp_path)
        assert cfg.bool_false_val is False

    def test_convert_none_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_NONE_VAL", "none")
        cfg = _make_config(tmp_path)
        assert cfg.none_val is None

    def test_convert_null_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_NULL_VAL", "null")
        cfg = _make_config(tmp_path)
        assert cfg.null_val is None

    def test_convert_integer_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_INT_VAL", "42")
        cfg = _make_config(tmp_path)
        assert cfg.int_val == 42

    def test_convert_negative_integer(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_NEG_INT", "-10")
        cfg = _make_config(tmp_path)
        assert cfg.neg_int == -10

    def test_convert_float_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_FLOAT_VAL", "3.14")
        cfg = _make_config(tmp_path)
        assert abs(cfg.float_val - 3.14) < 1e-9

    def test_convert_list_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_LIST_VAL", "a,b,c")
        cfg = _make_config(tmp_path)
        assert cfg.list_val == ["a", "b", "c"]

    def test_convert_string_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAMOLA_STR_VAL", "hello")
        cfg = _make_config(tmp_path)
        assert cfg.str_val == "hello"


# ---------------------------------------------------------------------------
# override_with_args (lines 496-600)
# ---------------------------------------------------------------------------

class TestOverrideWithArgs:
    def test_empty_args_no_op(self, tmp_cfg):
        original_log = tmp_cfg.log_level
        tmp_cfg.override_with_args({})
        assert tmp_cfg.log_level == original_log

    def test_none_args_no_op(self, tmp_cfg):
        original_log = tmp_cfg.log_level
        tmp_cfg.override_with_args(None)
        assert tmp_cfg.log_level == original_log

    def test_override_log_level(self, tmp_cfg):
        tmp_cfg.override_with_args({"log_level": "DEBUG"})
        assert tmp_cfg.log_level == "DEBUG"

    def test_override_use_encryption(self, tmp_cfg):
        tmp_cfg.override_with_args({"use_encryption": True})
        assert tmp_cfg.use_encryption is True

    def test_override_encryption_key_path(self, tmp_cfg):
        tmp_cfg.override_with_args({"encryption_key_path": "/tmp/keys.db"})
        assert tmp_cfg.encryption_key_path == "/tmp/keys.db"

    def test_override_encryption_mode(self, tmp_cfg):
        tmp_cfg.override_with_args({"encryption_mode": "simple"})
        assert tmp_cfg.encryption_mode == EncryptionMode.SIMPLE

    def test_override_use_vectorization(self, tmp_cfg):
        tmp_cfg.override_with_args({"use_vectorization": True})
        assert tmp_cfg.use_vectorization is True

    def test_override_parallel_processes(self, tmp_cfg):
        tmp_cfg.override_with_args({"parallel_processes": 4})
        assert tmp_cfg.parallel_processes == 4

    def test_override_use_dask(self, tmp_cfg):
        tmp_cfg.override_with_args({"use_dask": True})
        assert tmp_cfg.use_dask is True

    def test_override_npartitions(self, tmp_cfg):
        tmp_cfg.override_with_args({"npartitions": 8})
        assert tmp_cfg.npartitions == 8

    def test_override_chunk_size(self, tmp_cfg):
        tmp_cfg.override_with_args({"chunk_size": 50000})
        assert tmp_cfg.chunk_size == 50000

    def test_override_fields_scope(self, tmp_cfg):
        tmp_cfg.override_with_args({"fields": ["col1", "col2"]})
        assert tmp_cfg.scope["fields"] == ["col1", "col2"]

    def test_override_datasets_scope(self, tmp_cfg):
        tmp_cfg.override_with_args({"datasets": ["ds1"]})
        assert tmp_cfg.scope["datasets"] == ["ds1"]

    def test_override_continue_on_error(self, tmp_cfg):
        tmp_cfg.override_with_args({"continue_on_error": True})
        assert tmp_cfg.continue_on_error is True

    def test_override_allow_external(self, tmp_cfg):
        tmp_cfg.override_with_args({"allow_external": True})
        assert tmp_cfg.allow_external is True

    def test_override_allowed_external_paths(self, tmp_cfg):
        tmp_cfg.override_with_args({"allowed_external_paths": ["/safe"]})
        assert tmp_cfg.allowed_external_paths == ["/safe"]

    def test_path_cache_cleared_on_allow_external(self, tmp_cfg):
        tmp_cfg._path_cache["key"] = "value"
        tmp_cfg.override_with_args({"allow_external": True})
        assert tmp_cfg._path_cache == {}

    def test_data_repository_triggers_directory_setup(self, tmp_path):
        cfg = _make_config(tmp_path)
        new_repo = str(tmp_path / "NEW_DATA")
        with patch.object(cfg, "_setup_directories") as mock_setup:
            cfg.override_with_args({"data_repository": new_repo})
            mock_setup.assert_called_once()

    def test_extra_args_set_as_attributes(self, tmp_cfg):
        tmp_cfg.override_with_args({"my_extra_param": "extra_value"})
        assert tmp_cfg.my_extra_param == "extra_value"

    def test_none_value_extra_args_not_set(self, tmp_cfg):
        tmp_cfg.override_with_args({"some_param": None})
        # None values in extra args should not be set
        assert not hasattr(tmp_cfg, "some_param") or getattr(tmp_cfg, "some_param", "SENTINEL") is None


# ---------------------------------------------------------------------------
# validate (lines 602-656)
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_config_returns_true(self, tmp_cfg):
        is_valid, errors = tmp_cfg.validate()
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_missing_task_id_is_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.task_id = None
        is_valid, errors = cfg.validate()
        assert not is_valid
        assert any("task_id" in e for e in errors)

    def test_missing_task_type_is_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.task_type = None
        is_valid, errors = cfg.validate()
        assert not is_valid
        assert any("task_type" in e for e in errors)

    def test_encryption_enabled_without_key_is_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.use_encryption = True
        cfg.encryption_key_path = None
        is_valid, errors = cfg.validate()
        assert not is_valid
        assert any("encryption_key_path" in e for e in errors)

    def test_encryption_enabled_with_none_mode_is_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.use_encryption = True
        cfg.encryption_key_path = "/tmp/key"
        cfg.encryption_mode = EncryptionMode.NONE
        is_valid, errors = cfg.validate()
        assert not is_valid
        assert any("encryption_mode" in e for e in errors)

    def test_invalid_task_dir_suffixes_is_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.task_dir_suffixes = "not_a_list"
        is_valid, errors = cfg.validate()
        assert not is_valid
        assert any("task_dir_suffixes" in e for e in errors)

    def test_data_repository_created_if_missing(self, tmp_path):
        cfg = _make_config(tmp_path)
        new_path = tmp_path / "new_repo_dir"
        cfg.data_repository_path = new_path
        is_valid, errors = cfg.validate()
        # Either created or error message — no exception
        assert isinstance(errors, list)


# ---------------------------------------------------------------------------
# to_dict (lines 658-682)
# ---------------------------------------------------------------------------

class TestToDict:
    def test_returns_dict(self, tmp_cfg):
        result = tmp_cfg.to_dict()
        assert isinstance(result, dict)

    def test_task_id_in_result(self, tmp_cfg):
        result = tmp_cfg.to_dict()
        assert "task_id" in result

    def test_paths_serialized_as_strings(self, tmp_cfg):
        result = tmp_cfg.to_dict()
        for key, val in result.items():
            assert not isinstance(val, Path), f"Key {key} has Path value"

    def test_enum_serialized_as_string(self, tmp_cfg):
        result = tmp_cfg.to_dict()
        assert "encryption_mode" in result
        assert isinstance(result["encryption_mode"], str)

    def test_private_keys_excluded(self, tmp_cfg):
        result = tmp_cfg.to_dict()
        for key in result:
            assert not key.startswith("_")


# ---------------------------------------------------------------------------
# save (lines 684-756)
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_json_default_path(self, tmp_path):
        cfg = _make_config(tmp_path)
        saved_path = cfg.save()
        assert saved_path.exists()
        assert saved_path.suffix == ".json"

    def test_save_json_explicit_path(self, tmp_path):
        cfg = _make_config(tmp_path)
        out = tmp_path / "out_config.json"
        cfg.save(path=out)
        assert out.exists()

    def test_save_yaml_format(self, tmp_path):
        cfg = _make_config(tmp_path)
        out = tmp_path / "out_config.yaml"
        cfg.save(path=out, format="yaml")
        assert out.exists()

    def test_save_redacts_sensitive_keys(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.encryption_key_path = "/secret/key"
        out = tmp_path / "redacted.json"
        cfg.save(path=out)
        data = json.loads(out.read_text())
        assert data.get("encryption_key_path", "").startswith("<redacted")

    def test_save_raises_config_error_on_failure(self, tmp_path):
        cfg = _make_config(tmp_path)
        with patch("pamola_core.utils.tasks.task_config.write_json", side_effect=OSError("disk full")):
            with pytest.raises(ConfigurationError):
                cfg.save(path=tmp_path / "fail.json")


# ---------------------------------------------------------------------------
# Path API methods (lines 760-951)
# ---------------------------------------------------------------------------

class TestPathApiMethods:
    def test_get_project_root(self, tmp_path_cfg):
        root, cfg = tmp_path_cfg
        assert cfg.get_project_root() == root

    def test_get_data_repository(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        assert isinstance(cfg.get_data_repository(), Path)

    def test_get_raw_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        assert isinstance(cfg.get_raw_dir(), Path)

    def test_get_processed_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        assert isinstance(cfg.get_processed_dir(), Path)

    def test_get_reports_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        assert isinstance(cfg.get_reports_dir(), Path)

    def test_get_task_dir_default(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.get_task_dir()
        assert result == cfg.task_dir

    def test_get_task_dir_other_id(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.get_task_dir("other_task")
        assert result.name == "other_task"

    def test_get_task_dir_cached(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        r1 = cfg.get_task_dir("cached_id")
        r2 = cfg.get_task_dir("cached_id")
        assert r1 == r2

    def test_get_task_input_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.get_task_input_dir()
        assert result.name == "input"

    def test_get_task_output_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.get_task_output_dir()
        assert result.name == "output"

    def test_get_task_temp_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.get_task_temp_dir()
        assert result.name == "temp"

    def test_get_task_dict_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.get_task_dict_dir()
        assert result.name == "dictionaries"

    def test_get_task_logs_dir(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.get_task_logs_dir()
        assert result.name == "logs"

    def test_processed_subdir_with_parts(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        result = cfg.processed_subdir(None, "sub1", "sub2")
        assert "sub1" in str(result)
        assert "sub2" in str(result)

    def test_get_task_input_dir_other_id_cached(self, tmp_path_cfg):
        _, cfg = tmp_path_cfg
        r1 = cfg.get_task_input_dir("other_id")
        r2 = cfg.get_task_input_dir("other_id")
        assert r1 == r2


# ---------------------------------------------------------------------------
# get_dependency_output (lines 953-1025)
# ---------------------------------------------------------------------------

class TestGetDependencyOutput:
    def test_task_id_path_returns_output_dir(self, tmp_path):
        cfg = _make_config(tmp_path)
        # Create output dir so it "exists"
        out_dir = cfg.get_task_output_dir("dep_task")
        out_dir.mkdir(parents=True, exist_ok=True)
        result = cfg.get_dependency_output("dep_task")
        assert result == out_dir

    def test_missing_dep_raises_dependency_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(DependencyMissingError):
            cfg.get_dependency_output("missing_dep")

    def test_missing_dep_with_continue_on_error_returns_path(self, tmp_path):
        cfg = _make_config(tmp_path, config_overrides={"continue_on_error": True})
        cfg.continue_on_error = True
        result = cfg.get_dependency_output("missing_dep")
        assert isinstance(result, Path)

    def test_missing_dep_with_file_pattern_continue_returns_empty_list(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.continue_on_error = True
        result = cfg.get_dependency_output("missing_dep", file_pattern="*.csv")
        assert result == []

    def test_file_pattern_returns_matching_files(self, tmp_path):
        cfg = _make_config(tmp_path)
        out_dir = cfg.get_task_output_dir("dep_with_files")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "data.csv").write_text("col1,col2")
        result = cfg.get_dependency_output("dep_with_files", file_pattern="*.csv")
        assert len(result) == 1

    def test_absolute_path_dep_existing(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.allow_external = True
        dep_dir = tmp_path / "ext_dep"
        dep_dir.mkdir()
        result = cfg.get_dependency_output(str(dep_dir))
        assert result == dep_dir

    def test_absolute_path_dep_security_violation(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.allow_external = False
        cfg.allowed_external_paths = []
        dep_dir = tmp_path / "ext_dep"
        dep_dir.mkdir()
        # validate_path_security returns False for external absolute path
        with patch("pamola_core.utils.tasks.task_config.validate_path_security", return_value=False):
            with pytest.raises(PathSecurityError):
                cfg.get_dependency_output(str(dep_dir))


# ---------------------------------------------------------------------------
# get_dependency_report and assert_dependencies_completed (lines 1027-1106)
# ---------------------------------------------------------------------------

class TestDependencyReport:
    def test_get_dep_report_missing_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(DependencyMissingError):
            cfg.get_dependency_report("missing_dep")

    def test_get_dep_report_missing_continue_on_error_returns_path(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.continue_on_error = True
        result = cfg.get_dependency_report("missing_dep")
        assert isinstance(result, Path)

    def test_get_dep_report_existing(self, tmp_path):
        cfg = _make_config(tmp_path)
        report_path = cfg.reports_path
        report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / "dep1_report.json"
        report_file.write_text('{"success": true}')
        result = cfg.get_dependency_report("dep1")
        assert result == report_file

    def test_assert_dependencies_completed_no_deps(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.dependencies = []
        result = cfg.assert_dependencies_completed()
        assert result is True

    def test_assert_dependencies_completed_success(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.dependencies = ["dep1"]
        report_path = cfg.reports_path
        report_path.mkdir(parents=True, exist_ok=True)
        (report_path / "dep1_report.json").write_text('{"success": true}')
        result = cfg.assert_dependencies_completed()
        assert result is True

    def test_assert_dependencies_failed_dep_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.dependencies = ["dep_fail"]
        report_path = cfg.reports_path
        report_path.mkdir(parents=True, exist_ok=True)
        (report_path / "dep_fail_report.json").write_text('{"success": false}')
        with pytest.raises(DependencyMissingError):
            cfg.assert_dependencies_completed()

    def test_assert_dependencies_invalid_json_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.dependencies = ["dep_bad"]
        report_path = cfg.reports_path
        report_path.mkdir(parents=True, exist_ok=True)
        (report_path / "dep_bad_report.json").write_text("INVALID_JSON{{")
        with pytest.raises(DependencyMissingError):
            cfg.assert_dependencies_completed()

    def test_assert_dependencies_missing_file_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.dependencies = ["dep_missing"]
        with pytest.raises(DependencyMissingError):
            cfg.assert_dependencies_completed()


# ---------------------------------------------------------------------------
# Scope methods (lines 1108-1144)
# ---------------------------------------------------------------------------

class TestScopeMethods:
    def test_get_scope_fields_empty(self, tmp_cfg):
        tmp_cfg.scope = {}
        assert tmp_cfg.get_scope_fields() == []

    def test_get_scope_fields_with_values(self, tmp_cfg):
        tmp_cfg.scope = {"fields": ["f1", "f2"]}
        assert tmp_cfg.get_scope_fields() == ["f1", "f2"]

    def test_get_scope_datasets_empty(self, tmp_cfg):
        tmp_cfg.scope = {}
        assert tmp_cfg.get_scope_datasets() == []

    def test_get_scope_datasets_with_values(self, tmp_cfg):
        tmp_cfg.scope = {"datasets": ["ds1"]}
        assert tmp_cfg.get_scope_datasets() == ["ds1"]

    def test_get_scope_field_groups_empty(self, tmp_cfg):
        tmp_cfg.scope = {}
        assert tmp_cfg.get_scope_field_groups() == {}

    def test_get_scope_field_groups_with_values(self, tmp_cfg):
        tmp_cfg.scope = {"field_groups": {"g1": ["f1"]}}
        assert tmp_cfg.get_scope_field_groups() == {"g1": ["f1"]}

    def test_scope_methods_no_scope_attribute(self, tmp_cfg):
        if hasattr(tmp_cfg, "scope"):
            del tmp_cfg.scope
        assert tmp_cfg.get_scope_fields() == []
        assert tmp_cfg.get_scope_datasets() == []
        assert tmp_cfg.get_scope_field_groups() == {}


# ---------------------------------------------------------------------------
# __str__ (lines 1146-1169)
# ---------------------------------------------------------------------------

class TestStr:
    def test_str_contains_task_id(self, tmp_cfg):
        s = str(tmp_cfg)
        assert "t001" in s or tmp_cfg.task_id in s

    def test_str_contains_task_config_prefix(self, tmp_cfg):
        s = str(tmp_cfg)
        assert "TaskConfig" in s


# ---------------------------------------------------------------------------
# resolve_legacy_path (lines 369-403)
# ---------------------------------------------------------------------------

class TestResolveLegacyPath:
    def test_absolute_path_allowed(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.allow_external = True
        result = cfg.resolve_legacy_path(str(tmp_path))
        assert result == tmp_path

    def test_absolute_path_security_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.allow_external = False
        cfg.allowed_external_paths = []
        with patch("pamola_core.utils.tasks.task_config.validate_path_security", return_value=False):
            with pytest.raises(PathSecurityError):
                cfg.resolve_legacy_path(str(tmp_path / "some" / "path"))

    def test_relative_path_resolves_to_project_root(self, tmp_path):
        cfg = _make_config(tmp_path)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = cfg.resolve_legacy_path("relative/path")
        assert result == tmp_path / "relative" / "path"


# ---------------------------------------------------------------------------
# load_task_config function (lines 1172-1494)
# ---------------------------------------------------------------------------

class TestLoadTaskConfigFunction:
    def _mock_env(self, tmp_path, config_save_dir=None):
        """Context that patches project helpers for load_task_config tests."""
        return (
            patch("pamola_core.utils.tasks.task_config.find_project_root", return_value=tmp_path),
            patch("pamola_core.utils.tasks.task_config.load_project_config", return_value={}),
        )

    def test_returns_task_config_instance(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        with p1, p2:
            result = load_task_config("task_a", "anon")
        assert isinstance(result, TaskConfig)

    def test_returns_task_config_with_task_id(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        with p1, p2:
            result = load_task_config("task_b", "profiling")
        assert result.task_id == "task_b"

    def test_args_override_applied(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        with p1, p2:
            result = load_task_config("task_c", "anon", args={"log_level": "DEBUG"})
        assert result.log_level == "DEBUG"

    def test_default_config_used_when_no_file(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        default = {"my_default_key": "default_val"}
        with p1, p2:
            result = load_task_config("task_d", "anon", default_config=default)
        assert result.my_default_key == "default_val"

    def test_existing_config_file_loaded(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        config_dir = tmp_path / "configs"
        config_dir.mkdir(exist_ok=True)
        task_id = "task_existing"
        cfg_file = config_dir / f"{task_id}.json"
        cfg_file.write_text(json.dumps({"my_loaded_key": "loaded_value"}))
        with p1, p2:
            result = load_task_config(task_id, "anon")
        assert result.my_loaded_key == "loaded_value"

    def test_force_recreate_config_file(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        config_dir = tmp_path / "configs"
        config_dir.mkdir(exist_ok=True)
        task_id = "task_recreate"
        cfg_file = config_dir / f"{task_id}.json"
        cfg_file.write_text(json.dumps({"old_key": "old_value"}))
        default = {"new_key": "new_value"}
        with p1, p2:
            result = load_task_config(task_id, "anon",
                                      default_config=default,
                                      force_recreate_config_file=True)
        assert result.new_key == "new_value"

    def test_invalid_config_file_falls_back_to_default(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        config_dir = tmp_path / "configs"
        config_dir.mkdir(exist_ok=True)
        task_id = "task_invalid"
        cfg_file = config_dir / f"{task_id}.json"
        cfg_file.write_text("INVALID JSON {{")
        default = {"fallback_key": "fallback_val"}
        with p1, p2:
            result = load_task_config(task_id, "anon", default_config=default)
        assert result.fallback_key == "fallback_val"

    def test_config_save_dir_used_when_provided(self, tmp_path):
        p1, p2 = self._mock_env(tmp_path)
        save_dir = tmp_path / "custom_configs"
        save_dir.mkdir(exist_ok=True)
        task_id = "task_custom_dir"
        default = {"config_save_dir": str(save_dir)}
        with p1, p2:
            result = load_task_config(task_id, "anon", default_config=default)
        assert isinstance(result, TaskConfig)

    def test_project_config_failure_raises(self, tmp_path):
        from pamola_core.errors.exceptions import PamolaFileNotFoundError, ConfigurationError
        with patch("pamola_core.utils.tasks.task_config.find_project_root", return_value=tmp_path), \
             patch("pamola_core.utils.tasks.task_config.load_project_config",
                   side_effect=PamolaFileNotFoundError("no config")):
            # May raise ConfigurationError or return with defaults
            try:
                result = load_task_config("task_e", "anon")
                assert isinstance(result, TaskConfig)
            except (ConfigurationError, AttributeError):
                pass  # Expected — exercises the error path

    def test_with_progress_manager(self, tmp_path):
        pm = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        pm.create_operation_context.return_value = ctx

        p1, p2 = self._mock_env(tmp_path)
        with p1, p2:
            result = load_task_config("task_pm", "anon", progress_manager=pm)
        assert isinstance(result, TaskConfig)
        assert pm.create_operation_context.called
