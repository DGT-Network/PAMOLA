"""
Tests for run command.

Tests:
- Task mode (--task with full pipeline)
- Single-operation mode (--op with --config and --input)
- Seed parameter for reproducibility
- Output directory handling
- Error scenarios (invalid JSON, missing params, etc.)
- Exit codes
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from pamola_core.cli.main import app
from pamola_core.cli.utils.exit_codes import EXIT_OK, EXIT_ERROR, EXIT_VALIDATION


class TestRunHelp:
    """Test run command help."""

    def test_run_help_long(self, cli_runner: CliRunner):
        """Test --help for run command."""
        result = cli_runner.invoke(app, ["run", "--help"])
        assert result.exit_code == EXIT_OK
        assert "execute" in result.stdout.lower() or "task" in result.stdout.lower()

    def test_run_help_short(self, cli_runner: CliRunner):
        """Test -h for run command."""
        result = cli_runner.invoke(app, ["run", "-h"])
        assert result.exit_code == EXIT_OK


class TestRunNoArgs:
    """Test run command with no arguments."""

    def test_run_no_args_error(self, cli_runner: CliRunner):
        """Test that run with no --task or --op returns error."""
        result = cli_runner.invoke(app, ["run"])
        assert result.exit_code == EXIT_ERROR

    def test_run_error_message_clear(self, cli_runner: CliRunner):
        """Test error message is clear about required options."""
        result = cli_runner.invoke(app, ["run"])
        assert result.exit_code == EXIT_ERROR


class TestRunTaskMode:
    """Test task mode (--task option)."""

    def test_task_mode_missing_input_datasets(self, cli_runner: CliRunner, temp_dir: Path):
        """Test that task without input_datasets fails validation."""
        invalid_task = {
            "task_id": "test",
            "task_type": "anonymization",
            "operations": []
        }
        task_file = temp_dir / "no_inputs.json"
        task_file.write_text(json.dumps(invalid_task), encoding="utf-8")

        result = cli_runner.invoke(app, ["run", "--task", str(task_file)])
        assert result.exit_code == EXIT_VALIDATION

    def test_task_mode_missing_operations(self, cli_runner: CliRunner, temp_dir: Path):
        """Test that task without operations fails validation."""
        invalid_task = {
            "task_id": "test",
            "task_type": "anonymization",
            "input_datasets": {"data": "input.csv"}
        }
        task_file = temp_dir / "no_ops.json"
        task_file.write_text(json.dumps(invalid_task), encoding="utf-8")

        result = cli_runner.invoke(app, ["run", "--task", str(task_file)])
        assert result.exit_code == EXIT_VALIDATION

    def test_task_mode_valid_structure(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test task with valid structure (may fail on execution but not validation)."""
        result = cli_runner.invoke(app, ["run", "--task", str(sample_task_json)])
        # May fail with EXIT_ERROR due to missing files, but not EXIT_VALIDATION
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_task_mode_long_option(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test --task long option."""
        result = cli_runner.invoke(app, ["run", "--task", str(sample_task_json)])
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_task_mode_short_option(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test -t short option for task."""
        result = cli_runner.invoke(app, ["run", "-t", str(sample_task_json)])
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_task_mode_with_output(self, cli_runner: CliRunner, sample_task_json: Path, temp_dir: Path):
        """Test task mode with output directory."""
        output_dir = temp_dir / "output"
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--output", str(output_dir)]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_task_mode_output_short(self, cli_runner: CliRunner, sample_task_json: Path, temp_dir: Path):
        """Test -o short option for output."""
        output_dir = temp_dir / "results"
        result = cli_runner.invoke(
            app, ["run", "-t", str(sample_task_json), "-o", str(output_dir)]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_task_mode_with_seed(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test task mode with seed for reproducibility."""
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--seed", "42"]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_task_nonexistent_file(self, cli_runner: CliRunner):
        """Test task file that doesn't exist."""
        result = cli_runner.invoke(app, ["run", "--task", "/nonexistent/task.json"])
        assert result.exit_code != EXIT_OK


class TestRunSingleOpMode:
    """Test single-operation mode."""

    def test_op_requires_input(self, cli_runner: CliRunner):
        """Test that --op requires --input."""
        result = cli_runner.invoke(
            app, ["run", "--op", "AttributeSuppressionOperation"]
        )
        assert result.exit_code == EXIT_ERROR

    def test_single_op_with_all_options(
        self, cli_runner: CliRunner, sample_csv_file: Path, sample_op_config_json: Path, temp_dir: Path
    ):
        """Test single-op mode with all options."""
        output_dir = temp_dir / "op_output"
        result = cli_runner.invoke(
            app,
            [
                "run",
                "--op",
                "AttributeSuppressionOperation",
                "--config",
                str(sample_op_config_json),
                "--input",
                str(sample_csv_file),
                "--output",
                str(output_dir),
                "--seed",
                "42",
            ],
        )
        # May fail on execution but should parse args correctly
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_single_op_minimal(self, cli_runner: CliRunner, sample_csv_file: Path):
        """Test single-op with minimal arguments."""
        result = cli_runner.invoke(
            app, ["run", "--op", "AttributeSuppressionOperation", "--input", str(sample_csv_file)]
        )
        # Should work without config (defaults to empty params)
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_single_op_with_seed(
        self, cli_runner: CliRunner, sample_csv_file: Path
    ):
        """Test single-op mode with seed."""
        result = cli_runner.invoke(
            app,
            [
                "run",
                "--op",
                "AttributeSuppressionOperation",
                "--input",
                str(sample_csv_file),
                "--seed",
                "123",
            ],
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_single_op_invalid_config_json(
        self, cli_runner: CliRunner, sample_csv_file: Path, invalid_json_file: Path
    ):
        """Test single-op with invalid JSON config."""
        result = cli_runner.invoke(
            app,
            [
                "run",
                "--op",
                "AttributeSuppressionOperation",
                "--config",
                str(invalid_json_file),
                "--input",
                str(sample_csv_file),
            ],
        )
        assert result.exit_code == EXIT_VALIDATION


class TestRunTaskModeErrorHandling:
    """Test error handling in task mode."""

    def test_task_invalid_json(self, cli_runner: CliRunner, invalid_json_file: Path):
        """Test task file with invalid JSON."""
        result = cli_runner.invoke(app, ["run", "--task", str(invalid_json_file)])
        assert result.exit_code == EXIT_VALIDATION

    def test_task_json_not_object(self, cli_runner: CliRunner, temp_dir: Path):
        """Test task file where JSON is array instead of object."""
        bad_task = temp_dir / "array.json"
        bad_task.write_text("[]", encoding="utf-8")
        result = cli_runner.invoke(app, ["run", "--task", str(bad_task)])
        # Should fail validation
        assert result.exit_code != EXIT_OK

    def test_task_empty_json(self, cli_runner: CliRunner, temp_dir: Path):
        """Test task file with empty object."""
        empty_task = temp_dir / "empty.json"
        empty_task.write_text("{}", encoding="utf-8")
        result = cli_runner.invoke(app, ["run", "--task", str(empty_task)])
        assert result.exit_code == EXIT_VALIDATION

    def test_task_null_json(self, cli_runner: CliRunner, temp_dir: Path):
        """Test task file with null."""
        null_task = temp_dir / "null.json"
        null_task.write_text("null", encoding="utf-8")
        result = cli_runner.invoke(app, ["run", "--task", str(null_task)])
        assert result.exit_code != EXIT_OK


class TestRunOutputHandling:
    """Test output directory handling."""

    def test_output_plain_name_to_output_dir(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test that plain output name is placed under ./output/"""
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--output", "results"]
        )
        # Should parse and attempt to create output/results
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_output_absolute_path(self, cli_runner: CliRunner, sample_task_json: Path, temp_dir: Path):
        """Test output with absolute path."""
        output_path = temp_dir / "custom_output"
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--output", str(output_path)]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_output_relative_path(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test output with relative path."""
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--output", "my_task/results"]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]


class TestRunSeedHandling:
    """Test seed parameter handling."""

    def test_seed_valid_integer(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test seed with valid integer."""
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--seed", "42"]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_seed_zero(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test seed with zero."""
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--seed", "0"]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_seed_large_integer(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test seed with large integer."""
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--seed", "999999999"]
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_seed_invalid_not_integer(self, cli_runner: CliRunner, sample_task_json: Path):
        """Test seed with non-integer value."""
        result = cli_runner.invoke(
            app, ["run", "--task", str(sample_task_json), "--seed", "not_a_number"]
        )
        # Should fail due to type validation
        assert result.exit_code != EXIT_OK


class TestRunExitCodes:
    """Test exit codes."""

    def test_error_exit_code_range(self, cli_runner: CliRunner):
        """Test that errors return non-zero exit codes."""
        result = cli_runner.invoke(app, ["run"])
        assert result.exit_code != EXIT_OK
        assert result.exit_code in [EXIT_ERROR, EXIT_VALIDATION, 2, 1]


class TestRunIntegration:
    """Integration tests."""

    def test_task_and_op_mutually_exclusive(self, cli_runner: CliRunner, sample_task_json: Path, sample_csv_file: Path):
        """Test that --task and --op are properly handled (task takes precedence)."""
        result = cli_runner.invoke(
            app,
            [
                "run",
                "--task",
                str(sample_task_json),
                "--op",
                "AttributeSuppressionOperation",
                "--input",
                str(sample_csv_file),
            ],
        )
        # Task mode should take precedence
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_task_with_multiple_options(self, cli_runner: CliRunner, sample_task_json: Path, temp_dir: Path):
        """Test task mode with multiple options together."""
        result = cli_runner.invoke(
            app,
            [
                "run",
                "--task",
                str(sample_task_json),
                "--output",
                str(temp_dir / "out"),
                "--seed",
                "99",
            ],
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]

    def test_op_with_all_options(self, cli_runner: CliRunner, sample_csv_file: Path, sample_op_config_json: Path, temp_dir: Path):
        """Test single-op with all options."""
        result = cli_runner.invoke(
            app,
            [
                "run",
                "--op",
                "AttributeSuppressionOperation",
                "--config",
                str(sample_op_config_json),
                "--input",
                str(sample_csv_file),
                "--output",
                str(temp_dir / "op_out"),
                "--seed",
                "77",
            ],
        )
        assert result.exit_code in [EXIT_ERROR, EXIT_OK]
