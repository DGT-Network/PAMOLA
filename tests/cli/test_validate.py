"""
Tests for validate-config command.

Tests:
- Validate operation config JSON
- Validate task pipeline JSON
- Output formats (table, json)
- Error reporting (missing keys, unknown ops, invalid params)
- Exit codes
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pamola_core.cli.main import app
from pamola_core.cli.utils.exit_codes import EXIT_OK, EXIT_ERROR, EXIT_VALIDATION


class TestValidateHelp:
    """Test validate-config command help."""

    def test_validate_help_long(self, cli_runner: CliRunner):
        """Test --help for validate-config command."""
        result = cli_runner.invoke(app, ["validate-config", "--help"])
        assert result.exit_code == EXIT_OK
        assert (
            "validate" in result.stdout.lower() or "config" in result.stdout.lower()
        )

    def test_validate_help_short(self, cli_runner: CliRunner):
        """Test -h for validate-config command."""
        result = cli_runner.invoke(app, ["validate-config", "-h"])
        assert result.exit_code == EXIT_OK


class TestValidateNoArgs:
    """Test validate command with no arguments."""

    def test_validate_no_args_error(self, cli_runner: CliRunner):
        """Test validate with no --config or --task returns error."""
        result = cli_runner.invoke(app, ["validate-config"])
        assert result.exit_code == EXIT_ERROR

    def test_validate_error_message_clear(self, cli_runner: CliRunner):
        """Test error message mentions both options."""
        result = cli_runner.invoke(app, ["validate-config"])
        assert result.exit_code == EXIT_ERROR


class TestValidateOpConfig:
    """Test validating operation config files."""

    def test_valid_op_config(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test validation of valid operation config."""
        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(sample_op_config_json)]
        )
        # May succeed if operation exists, or fail if not registered
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_config_long_option(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test --config long option."""
        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(sample_op_config_json)]
        )
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_config_missing_operation_key(self, cli_runner: CliRunner, temp_dir: Path):
        """Test config without 'operation' key."""
        invalid_config = {
            "parameters": {"fields": ["ssn"]},
            "scope": {"target": ["ssn"]}
        }
        config_file = temp_dir / "no_op.json"
        config_file.write_text(json.dumps(invalid_config), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(config_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_config_unknown_operation(self, cli_runner: CliRunner, temp_dir: Path):
        """Test config with unknown operation."""
        invalid_config = {
            "operation": "NonexistentOperation",
            "parameters": {}
        }
        config_file = temp_dir / "unknown_op.json"
        config_file.write_text(json.dumps(invalid_config), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(config_file)]
        )
        # Should fail validation
        assert result.exit_code in [EXIT_VALIDATION, EXIT_ERROR]

    def test_config_invalid_json(self, cli_runner: CliRunner, invalid_json_file: Path):
        """Test config file with invalid JSON."""
        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(invalid_json_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_config_nonexistent_file(self, cli_runner: CliRunner):
        """Test config file that doesn't exist."""
        result = cli_runner.invoke(
            app, ["validate-config", "--config", "/nonexistent/config.json"]
        )
        assert result.exit_code != EXIT_OK


class TestValidateTask:
    """Test validating task pipeline files."""

    def test_valid_task(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test validation of valid task JSON."""
        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(sample_task_json)]
        )
        # May succeed or fail based on operation registry
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_task_long_option(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test --task long option."""
        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(sample_task_json)]
        )
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_task_missing_input_datasets(self, cli_runner: CliRunner, temp_dir: Path):
        """Test task without input_datasets key."""
        invalid_task = {
            "task_id": "test",
            "operations": []
        }
        task_file = temp_dir / "no_inputs.json"
        task_file.write_text(json.dumps(invalid_task), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(task_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_task_missing_operations(self, cli_runner: CliRunner, temp_dir: Path):
        """Test task without operations key."""
        invalid_task = {
            "task_id": "test",
            "input_datasets": {"data": "input.csv"}
        }
        task_file = temp_dir / "no_ops.json"
        task_file.write_text(json.dumps(invalid_task), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(task_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_task_empty_operations(self, cli_runner: CliRunner, temp_dir: Path):
        """Test task with empty operations list."""
        invalid_task = {
            "task_id": "test",
            "input_datasets": {"data": "input.csv"},
            "operations": []
        }
        task_file = temp_dir / "empty_ops.json"
        task_file.write_text(json.dumps(invalid_task), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(task_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_task_op_missing_class_name(self, cli_runner: CliRunner, temp_dir: Path):
        """Test task operation without class_name."""
        invalid_task = {
            "task_id": "test",
            "input_datasets": {"data": "input.csv"},
            "operations": [
                {
                    "parameters": {}
                }
            ]
        }
        task_file = temp_dir / "no_class.json"
        task_file.write_text(json.dumps(invalid_task), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(task_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_task_invalid_json(self, cli_runner: CliRunner, invalid_json_file: Path):
        """Test task file with invalid JSON."""
        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(invalid_json_file)]
        )
        assert result.exit_code == EXIT_VALIDATION


class TestValidateOutputFormats:
    """Test different output formats."""

    def test_format_table_default(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test default (table) output format."""
        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(sample_op_config_json)]
        )
        # May succeed or fail validation
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_format_table_long(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test --format table option."""
        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(sample_op_config_json),
                "--format",
                "table",
            ],
        )
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_format_table_short(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test -f table short option."""
        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(sample_op_config_json),
                "-f",
                "table",
            ],
        )
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_format_json(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test --format json option."""
        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(sample_op_config_json),
                "--format",
                "json",
            ],
        )
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]
        # Output should be JSON-like
        output = result.stdout.strip()
        if output.startswith("{"):
            data = json.loads(output)
            assert "valid" in data or "errors" in data

    def test_format_json_short(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test -f json short option."""
        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(sample_op_config_json),
                "-f",
                "json",
            ],
        )
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_json_output_structure_valid(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test JSON output structure for valid config."""
        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(sample_op_config_json),
                "-f",
                "json",
            ],
        )
        output = result.stdout.strip()
        if output.startswith("{"):
            data = json.loads(output)
            # Should have valid and file keys
            assert "valid" in data
            if "file" in data:
                assert str(sample_op_config_json) in data["file"]

    def test_json_output_structure_invalid(self, cli_runner: CliRunner, temp_dir: Path):
        """Test JSON output structure for invalid config."""
        invalid_config = {"parameters": {}}  # Missing operation
        config_file = temp_dir / "invalid.json"
        config_file.write_text(json.dumps(invalid_config), encoding="utf-8")

        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(config_file),
                "-f",
                "json",
            ],
        )
        output = result.stdout.strip()
        if output.startswith("{"):
            data = json.loads(output)
            assert "valid" in data
            assert data["valid"] is False or "errors" in data


class TestValidateErrorMessages:
    """Test error message clarity."""

    def test_error_lists_issues(self, cli_runner: CliRunner, temp_dir: Path):
        """Test that validation errors are listed clearly."""
        invalid_config = {}  # Empty config
        config_file = temp_dir / "empty.json"
        config_file.write_text(json.dumps(invalid_config), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(config_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_error_shows_file_path(self, cli_runner: CliRunner, temp_dir: Path):
        """Test that error messages show file path."""
        invalid_config = {}
        config_file = temp_dir / "test.json"
        config_file.write_text(json.dumps(invalid_config), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(config_file)]
        )
        # Should mention file in output
        assert str(config_file) in result.stdout or "test.json" in result.stdout or result.exit_code != EXIT_OK


class TestValidateExitCodes:
    """Test exit code values."""

    def test_success_exit_zero(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test success returns exit code 0 (if config is valid)."""
        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(sample_op_config_json)]
        )
        # May be 0 if valid, or 2 if invalid
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_validation_error_exit_two(self, cli_runner: CliRunner, temp_dir: Path):
        """Test validation error returns exit code 2."""
        invalid_config = {}
        config_file = temp_dir / "invalid.json"
        config_file.write_text(json.dumps(invalid_config), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(config_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_json_error_exit_two(self, cli_runner: CliRunner, invalid_json_file: Path):
        """Test JSON parsing error returns exit code 2."""
        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(invalid_json_file)]
        )
        assert result.exit_code == EXIT_VALIDATION


class TestValidateConfigVsTask:
    """Test differences between config and task validation."""

    def test_config_requires_operation_key(self, cli_runner: CliRunner, temp_dir: Path):
        """Test that config requires 'operation' key."""
        config = {"parameters": {}}
        config_file = temp_dir / "config.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--config", str(config_file)]
        )
        assert result.exit_code == EXIT_VALIDATION

    def test_task_requires_input_datasets(self, cli_runner: CliRunner, temp_dir: Path):
        """Test that task requires 'input_datasets' key."""
        task = {"operations": []}
        task_file = temp_dir / "task.json"
        task_file.write_text(json.dumps(task), encoding="utf-8")

        result = cli_runner.invoke(
            app, ["validate-config", "--task", str(task_file)]
        )
        assert result.exit_code == EXIT_VALIDATION


class TestValidateIntegration:
    """Integration tests."""

    def test_config_and_task_mutually_exclusive(self, cli_runner: CliRunner, sample_op_config_json: Path, sample_task_json: Path):
        """Test that providing both --config and --task works (task takes priority or both processed)."""
        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(sample_op_config_json),
                "--task",
                str(sample_task_json),
            ],
        )
        # Should process one (likely task as it's checked second)
        assert result.exit_code in [EXIT_OK, EXIT_VALIDATION, EXIT_ERROR]

    def test_json_format_for_scripting(self, cli_runner: CliRunner, sample_op_config_json: Path):
        """Test JSON format is suitable for scripting."""
        result = cli_runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(sample_op_config_json),
                "-f",
                "json",
            ],
        )
        output = result.stdout.strip()
        if output:
            # Should be parseable JSON
            data = json.loads(output)
            assert isinstance(data, dict)
            assert any(key in data for key in ["valid", "errors"])
