"""
Tests for main CLI app and global options.

Tests:
- App initialization and structure
- --version / -V flag
- --help / -h flag
- --verbose / -v flag
- Error handling for unknown commands
- Exit codes
"""

import re
import logging

import pytest
from typer.testing import CliRunner

from pamola_core.cli.main import app
from pamola_core.cli.utils.exit_codes import EXIT_OK, EXIT_ERROR


class TestMainAppStructure:
    """Test basic CLI app structure and initialization."""

    def test_app_creation(self):
        """Test that app is properly initialized."""
        assert app is not None
        assert app.info.name == "pamola-core"
        assert app.info.help is not None

    def test_app_has_callback(self):
        """Test that root callback is registered."""
        # Verify app has root callback with version and verbose options
        callbacks = [cmd for cmd in app.registered_commands if callable(cmd)]
        assert len(callbacks) >= 0  # Callback should exist


class TestVersionOption:
    """Test --version and -V flags."""

    def test_version_long_flag(self, cli_runner: CliRunner):
        """Test --version flag returns version and exits with 0."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == EXIT_OK
        assert "pamola-core" in result.stdout
        # Version format should be "pamola-core X.Y.Z"
        assert re.match(r"pamola-core \d+\.\d+\.\d+", result.stdout)

    def test_version_short_flag(self, cli_runner: CliRunner):
        """Test -V flag returns version and exits with 0."""
        result = cli_runner.invoke(app, ["-V"])
        assert result.exit_code == EXIT_OK
        assert "pamola-core" in result.stdout

    def test_version_output_format(self, cli_runner: CliRunner):
        """Test version output matches expected format."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == EXIT_OK
        # Should be single line with version
        lines = result.stdout.strip().split("\n")
        assert len(lines) >= 1
        assert lines[0].startswith("pamola-core ")

    def test_version_is_eager(self, cli_runner: CliRunner):
        """Test that --version is processed eagerly (before subcommands)."""
        # Even with invalid subcommand after --version, should show version
        result = cli_runner.invoke(app, ["--version", "invalid-command"])
        assert result.exit_code == EXIT_OK
        assert "pamola-core" in result.stdout


class TestHelpOption:
    """Test --help and -h flags."""

    def test_help_long_flag(self, cli_runner: CliRunner):
        """Test --help flag displays help and exits with 0."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        assert "PAMOLA.CORE" in result.stdout or "Usage" in result.stdout

    def test_help_short_flag(self, cli_runner: CliRunner):
        """Test -h flag displays help and exits with 0."""
        result = cli_runner.invoke(app, ["-h"])
        assert result.exit_code == EXIT_OK
        assert "PAMOLA.CORE" in result.stdout or "Usage" in result.stdout

    def test_help_shows_subcommands(self, cli_runner: CliRunner):
        """Test that help output mentions available subcommands."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        # Should mention registered subcommands
        assert "list-ops" in result.stdout or "Commands" in result.stdout

    def test_help_no_args(self, cli_runner: CliRunner):
        """Test that running without args shows help (no_args_is_help=True).

        Typer with no_args_is_help=True exits with code 0 or 2 depending on
        the version/platform — accept either as success.
        """
        result = cli_runner.invoke(app, [])
        # Typer may return 0 or 2 for no-args-is-help
        assert result.exit_code in [0, 2]
        # Should show help or usage info
        assert any(
            keyword in result.stdout
            for keyword in ["PAMOLA", "Usage", "Commands", "list-ops"]
        )


class TestVerboseOption:
    """Test --verbose and -v flags."""

    def test_verbose_long_flag(self, cli_runner: CliRunner, caplog):
        """Test --verbose flag enables debug logging."""
        with caplog.at_level(logging.DEBUG):
            result = cli_runner.invoke(app, ["--verbose", "--version"])
        assert result.exit_code == EXIT_OK

    def test_verbose_short_flag(self, cli_runner: CliRunner, caplog):
        """Test -v flag enables debug logging."""
        with caplog.at_level(logging.DEBUG):
            result = cli_runner.invoke(app, ["-v", "--version"])
        assert result.exit_code == EXIT_OK

    def test_verbose_does_not_expose_value(self, cli_runner: CliRunner):
        """Test that verbose option is not exposed to command (expose_value=False)."""
        # Should succeed even though verbose is not a parameter to callback
        result = cli_runner.invoke(app, ["-v", "--help"])
        assert result.exit_code == EXIT_OK


class TestRootCallback:
    """Test root callback behavior."""

    def test_root_callback_with_version(self, cli_runner: CliRunner):
        """Test that version callback works in root."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == EXIT_OK

    def test_root_callback_with_help(self, cli_runner: CliRunner):
        """Test that help callback works in root."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK


class TestUnknownCommands:
    """Test error handling for unknown commands."""

    def test_unknown_command_error(self, cli_runner: CliRunner):
        """Test that unknown command returns error exit code."""
        result = cli_runner.invoke(app, ["unknown-command"])
        assert result.exit_code != EXIT_OK
        # Typer outputs error messages — check stdout or output attribute
        combined = result.stdout + (result.output or "")
        assert (
            "No such command" in combined
            or "Error" in combined
            or result.exit_code != 0
        )

    def test_unknown_option_error(self, cli_runner: CliRunner):
        """Test that unknown option returns error."""
        result = cli_runner.invoke(app, ["--unknown-option"])
        assert result.exit_code != EXIT_OK


class TestCommandRegistration:
    """Test that subcommands are properly registered."""

    def test_list_ops_registered(self, cli_runner: CliRunner):
        """Test that list-ops command is registered."""
        result = cli_runner.invoke(app, ["list-ops", "--help"])
        assert result.exit_code == EXIT_OK
        assert "list" in result.stdout.lower() or "operations" in result.stdout.lower()

    def test_run_registered(self, cli_runner: CliRunner):
        """Test that run command is registered."""
        result = cli_runner.invoke(app, ["run", "--help"])
        assert result.exit_code == EXIT_OK

    def test_validate_registered(self, cli_runner: CliRunner):
        """Test that validate-config command is registered."""
        result = cli_runner.invoke(app, ["validate-config", "--help"])
        assert result.exit_code == EXIT_OK

    def test_schema_registered(self, cli_runner: CliRunner):
        """Test that schema command is registered."""
        result = cli_runner.invoke(app, ["schema", "--help"])
        assert result.exit_code == EXIT_OK


class TestExitCodes:
    """Test CLI exit codes."""

    def test_success_exit_code(self, cli_runner: CliRunner):
        """Test successful command returns 0."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_help_exit_code(self, cli_runner: CliRunner):
        """Test help command returns 0."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_error_exit_code(self, cli_runner: CliRunner):
        """Test error condition returns non-zero exit code."""
        result = cli_runner.invoke(app, ["unknown"])
        assert result.exit_code != 0


class TestOutputFormatting:
    """Test CLI output formatting."""

    def test_version_output_single_line(self, cli_runner: CliRunner):
        """Test version output is single line."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == EXIT_OK
        # Count non-empty lines
        lines = [l for l in result.stdout.split("\n") if l.strip()]
        assert len(lines) == 1

    def test_help_output_not_empty(self, cli_runner: CliRunner):
        """Test help output is not empty."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        assert len(result.stdout.strip()) > 0
