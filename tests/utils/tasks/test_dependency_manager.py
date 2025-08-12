import pytest
import logging
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from pamola_core.utils.tasks.dependency_manager import (
    TaskDependencyManager,
    DependencyError,
    DependencyMissingError,
    DependencyFailedError,
    OptionalT1IDependencyManager,
    PathSecurityError
)

class DummyConfig:
    def __init__(self, output_dir, reports_dir, dependencies=None, continue_on_error=False, allow_external=False, allowed_external_paths=None):
        self._output_dir = Path(output_dir)
        self._reports_dir = Path(reports_dir)
        self.dependencies = dependencies or []
        self.continue_on_error = continue_on_error
        self.allow_external = allow_external
        self.allowed_external_paths = allowed_external_paths or []

    def get_task_output_dir(self, dependency_id):
        return self._output_dir / dependency_id

    def get_reports_dir(self):
        return self._reports_dir

@pytest.fixture
def tmp_dirs(tmp_path):
    output_dir = tmp_path / "outputs"
    reports_dir = tmp_path / "reports"
    output_dir.mkdir()
    reports_dir.mkdir()
    return output_dir, reports_dir

@pytest.fixture
def logger():
    return logging.getLogger("test")

@pytest.fixture
def manager(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    config = DummyConfig(output_dir, reports_dir)
    return TaskDependencyManager(config, logger)

@pytest.fixture
def manager_with_deps(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    deps = ["dep1", "dep2"]
    config = DummyConfig(output_dir, reports_dir, dependencies=deps)
    return TaskDependencyManager(config, logger)

@pytest.fixture
def manager_continue_on_error(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    deps = ["dep1"]
    config = DummyConfig(output_dir, reports_dir, dependencies=deps, continue_on_error=True)
    return TaskDependencyManager(config, logger)

def test_get_dependency_output_task_id(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_dir = output_dir / "dep1"
    dep_dir.mkdir()
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_output("dep1")
    assert result == dep_dir

def test_get_dependency_output_task_id_missing(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_output("missing_dep")

def test_get_dependency_output_task_id_missing_continue_on_error(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    config = DummyConfig(output_dir, reports_dir, continue_on_error=True)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_output("missing_dep")
    assert isinstance(result, Path)

def test_get_dependency_output_with_file_pattern(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_dir = output_dir / "dep1"
    dep_dir.mkdir()
    (dep_dir / "file1.txt").write_text("abc")
    (dep_dir / "file2.log").write_text("def")
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    files = mgr.get_dependency_output("dep1", file_pattern="*.txt")
    assert len(files) == 1
    assert files[0].name == "file1.txt"

def test_get_dependency_output_with_file_pattern_no_match(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_dir = output_dir / "dep1"
    dep_dir.mkdir()
    (dep_dir / "file1.txt").write_text("abc")
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    files = mgr.get_dependency_output("dep1", file_pattern="*.log")
    assert files == []

def test_get_dependency_output_absolute_path(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    abs_dir = output_dir / "absdep"
    abs_dir.mkdir()
    config = DummyConfig(output_dir, reports_dir, allow_external=True, allowed_external_paths=[str(output_dir)])
    mgr = TaskDependencyManager(config, logger)
    with patch("pamola_core.utils.tasks.path_security.validate_path_security", return_value=True):
        result = mgr.get_dependency_output(str(abs_dir))
    assert result == abs_dir

def test_get_dependency_output_absolute_path_missing(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    abs_dir = output_dir / "absdep"
    config = DummyConfig(output_dir, reports_dir, allow_external=True, allowed_external_paths=[str(output_dir)])
    mgr = TaskDependencyManager(config, logger)
    with patch("pamola_core.utils.tasks.path_security.validate_path_security", return_value=True):
        with pytest.raises(DependencyMissingError):
            mgr.get_dependency_output(str(abs_dir))

def test_get_dependency_output_absolute_path_missing_continue_on_error(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    abs_dir = output_dir / "absdep"
    config = DummyConfig(output_dir, reports_dir, continue_on_error=True, allow_external=True, allowed_external_paths=[str(output_dir)])
    mgr = TaskDependencyManager(config, logger)
    with patch("pamola_core.utils.tasks.path_security.validate_path_security", return_value=True):
        result = mgr.get_dependency_output(str(abs_dir))
    assert isinstance(result, Path)

def test_get_dependency_report_found(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    report_path = reports_dir / f"{dep_id}_report.json"
    report_path.write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_report(dep_id)
    assert result == report_path

def test_get_dependency_report_alternate(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    alt_path = reports_dir / f"{dep_id}.json"
    alt_path.write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_report(dep_id)
    assert result == alt_path

def test_get_dependency_report_alternate_dir(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    alt_dir = reports_dir / dep_id
    alt_dir.mkdir()
    alt_report = alt_dir / "report.json"
    alt_report.write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_report(dep_id)
    assert result == alt_report

def test_get_dependency_report_missing(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_report(dep_id)

def test_get_dependency_report_missing_continue_on_error(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir, continue_on_error=True)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_report(dep_id)
    assert result.name == "dummy_report.json"

def test_get_dependency_report_missing_continue_on_error_returns_dummy(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir, continue_on_error=True)
    mgr = TaskDependencyManager(config, logger)
    dummy = mgr.get_dependency_report(dep_id)
    assert dummy.name == "dummy_report.json"
    assert dummy.parent == reports_dir

def test_get_dependency_report_absolute_path_error(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(ValueError):
        mgr.get_dependency_report(str(output_dir))

def test_assert_dependencies_completed_all_success(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["dep1", "dep2"]
    for dep in dep_ids:
        (reports_dir / f"{dep}_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = TaskDependencyManager(config, logger)
    assert mgr.assert_dependencies_completed() is True

def test_assert_dependencies_completed_failure(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["dep1", "dep2"]
    (reports_dir / "dep1_report.json").write_text(json.dumps({"success": False, "error_info": {"message": "fail"}}))
    (reports_dir / "dep2_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyFailedError):
        mgr.assert_dependencies_completed()

def test_assert_dependencies_completed_missing_report(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["dep1", "dep2"]
    (reports_dir / "dep2_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.assert_dependencies_completed()

@patch("pamola_core.utils.io.read_json", side_effect=json.JSONDecodeError("msg", "doc", 0))
def test_assert_dependencies_completed_json_decode_error(mock_read_json, tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    (reports_dir / f"{dep_id}_report.json").write_text("not json")
    config = DummyConfig(output_dir, reports_dir, dependencies=[dep_id])
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.assert_dependencies_completed()

@patch("pamola_core.utils.io.read_json", side_effect=FileNotFoundError)
def test_assert_dependencies_completed_file_not_found(mock_read_json, tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir, dependencies=[dep_id])
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.assert_dependencies_completed()

def test_assert_dependencies_completed_continue_on_error(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["dep1", "dep2"]
    (reports_dir / "dep1_report.json").write_text(json.dumps({"success": False, "error_info": {"message": "fail"}}))
    (reports_dir / "dep2_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids, continue_on_error=True)
    mgr = TaskDependencyManager(config, logger)
    assert mgr.assert_dependencies_completed() is True

def test_assert_dependencies_completed_absolute_path(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    abs_dep = str(output_dir / "absdep")
    dep_ids = [abs_dep, "dep2"]
    (reports_dir / "dep2_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = TaskDependencyManager(config, logger)
    assert mgr.assert_dependencies_completed() is True

def test_is_absolute_dependency(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    assert mgr.is_absolute_dependency("/abs/path")
    assert mgr.is_absolute_dependency("C:\\abs\\path")
    assert not mgr.is_absolute_dependency("relpath")

def test_get_dependency_metrics(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    metrics = {"foo": {"bar": 42}, "baz": 1}
    (reports_dir / f"{dep_id}_report.json").write_text(json.dumps({"success": True, "metrics": metrics}))
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_metrics(dep_id)
    assert result == metrics
    result2 = mgr.get_dependency_metrics(dep_id, metric_path="foo.bar")
    assert result2 == {"bar": 42}

@patch("pamola_core.utils.io.read_json", side_effect=json.JSONDecodeError("msg", "doc", 0))
def test_get_dependency_metrics_json_decode_error(mock_read_json, tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    (reports_dir / f"{dep_id}_report.json").write_text("not json")
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_metrics(dep_id)

@patch("pamola_core.utils.io.read_json", side_effect=FileNotFoundError)
def test_get_dependency_metrics_file_not_found(mock_read_json, tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_metrics(dep_id)

def test_get_dependency_metrics_missing(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_metrics(dep_id)

def test_get_dependency_metrics_invalid_json(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    (reports_dir / f"{dep_id}_report.json").write_text("not json")
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_metrics(dep_id)

def test_get_dependency_metrics_keyerror(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    (reports_dir / f"{dep_id}_report.json").write_text(json.dumps({"success": True, "metrics": {"foo": 1}}))
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(KeyError):
        mgr.get_dependency_metrics(dep_id, metric_path="bar.baz")

def test_get_dependency_status(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    report = {"success": True, "execution_time_seconds": 1.2, "completion_time": "now", "error_info": None}
    (reports_dir / f"{dep_id}_report.json").write_text(json.dumps(report))
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    result = mgr.get_dependency_status(dep_id)
    assert result["success"] is True
    assert result["task_id"] == dep_id
    assert result["report_path"].endswith("dep1_report.json")

@patch("pamola_core.utils.io.read_json", side_effect=json.JSONDecodeError("msg", "doc", 0))
def test_get_dependency_status_json_decode_error(mock_read_json, tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    (reports_dir / f"{dep_id}_report.json").write_text("not json")
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_status(dep_id)

@patch("pamola_core.utils.io.read_json", side_effect=FileNotFoundError)
def test_get_dependency_status_file_not_found(mock_read_json, tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_status(dep_id)

def test_get_dependency_status_missing(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_status(dep_id)

def test_get_dependency_status_invalid_json(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_id = "dep1"
    (reports_dir / f"{dep_id}_report.json").write_text("not json")
    config = DummyConfig(output_dir, reports_dir)
    mgr = TaskDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.get_dependency_status(dep_id)

def test_optional_t1i_dependency_manager(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["t_1I", "dep2"]
    (reports_dir / "dep2_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = OptionalT1IDependencyManager(config, logger)
    assert mgr.assert_dependencies_completed() is True

def test_optional_t1i_dependency_manager_missing_t1i(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["t_1I"]
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = OptionalT1IDependencyManager(config, logger)
    # Should not raise, should return True
    assert mgr.assert_dependencies_completed() is True

def test_optional_t1i_dependency_manager_other_missing(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["t_1I", "dep2"]
    # Create a report for t_1I so only dep2 is missing
    (reports_dir / "t_1I_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = OptionalT1IDependencyManager(config, logger)
    with pytest.raises(DependencyMissingError):
        mgr.assert_dependencies_completed()

def test_optional_t1i_dependency_manager_missing_t1i_and_other(tmp_dirs, logger):
    output_dir, reports_dir = tmp_dirs
    dep_ids = ["t_1I", "other"]
    # Create a report for t_1I so only 'other' is missing
    (reports_dir / "t_1I_report.json").write_text(json.dumps({"success": True}))
    config = DummyConfig(output_dir, reports_dir, dependencies=dep_ids)
    mgr = OptionalT1IDependencyManager(config, logger)
    # Should raise for 'other' missing
    with pytest.raises(DependencyMissingError):
        mgr.assert_dependencies_completed()

if __name__ == "__main__":
    pytest.main()