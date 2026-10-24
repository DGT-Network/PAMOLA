"""CVPLLinkageOperation is a first-class operation: catalog, CLI, config validation, schemas."""

import json

from typer.testing import CliRunner

from pamola_core.cli.main import app
from pamola_core.cli.utils.exit_codes import EXIT_OK, EXIT_VALIDATION

SAMPLE = {
    "operation": "CVPLLinkageOperation",
    "parameters": {
        "qids": ["age", "gender"],
        "auxiliary": ["income"],
        "tau": 0.9,
        "seed": 0,
        "ground_truth": {"source_id_column": "pid", "release_id_column": "rid"},
    },
}


def test_cvpl_is_listed_by_list_ops(cli_runner: CliRunner):
    result = cli_runner.invoke(app, ["list-ops", "--format", "json"])
    assert result.exit_code == EXIT_OK
    entries = {e["name"]: e for e in json.loads(result.stdout)}
    assert entries["CVPLLinkageOperation"]["category"] == "metrics"


def test_cvpl_is_listed_in_the_metrics_category(cli_runner: CliRunner):
    result = cli_runner.invoke(app, ["list-ops", "--category", "metrics", "--format", "json"])
    assert result.exit_code == EXIT_OK
    assert "CVPLLinkageOperation" in [e["name"] for e in json.loads(result.stdout)]


def test_validate_config_accepts_a_sample_config(cli_runner: CliRunner, tmp_path):
    path = tmp_path / "cvpl.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    result = cli_runner.invoke(app, ["validate-config", "--config", str(path), "--format", "json"])
    assert result.exit_code == EXIT_OK, result.stdout
    assert json.loads(result.stdout)["valid"] is True


def test_validate_config_reports_the_missing_qids(cli_runner: CliRunner, tmp_path):
    bad = {"operation": "CVPLLinkageOperation", "parameters": {"tau": 0.9}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    result = cli_runner.invoke(app, ["validate-config", "--config", str(path), "--format", "json"])
    assert result.exit_code == EXIT_VALIDATION
    assert any("qids" in e for e in json.loads(result.stdout)["errors"])


def test_schema_builder_exposes_the_four_schema_parts():
    from pamola_core.utils.schema_helpers.schema_builder import get_all_op_configs

    cores = [c[0].__name__ for c in get_all_op_configs()]
    assert "CVPLLinkageConfig" in cores
    core, ui, exclude, tooltips = next(c for c in get_all_op_configs() if c[0].__name__ == "CVPLLinkageConfig")
    assert "qids" in core.schema["allOf"][1]["properties"]
    assert "qids" in ui.schema["allOf"][1]["properties"]
    assert "columns" not in exclude and "name" in exclude
    assert "qids" in tooltips and "ground_truth" in tooltips


def test_task_runner_treats_cvpl_like_the_other_metrics():
    from pamola_core.utils.tasks.task_runner import TaskRunner

    src = TaskRunner.__init__.__code__.co_names
    assert "CVPLLinkageOperation" in src
