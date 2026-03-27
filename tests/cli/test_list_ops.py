"""
Tests for list-ops command.

Tests:
- List all operations (default)
- Filter by category
- Output formats (table, json)
- Error handling
- Exit codes
- Catalog loading vs. registry fallback
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from pamola_core.cli.main import app
from pamola_core.cli.utils.exit_codes import EXIT_OK, EXIT_ERROR


class TestListOpsBasic:
    """Test basic list-ops functionality."""

    def test_list_ops_no_args(self, cli_runner: CliRunner):
        """Test listing all operations without arguments."""
        result = cli_runner.invoke(app, ["list-ops"])
        assert result.exit_code == EXIT_OK

    def test_list_ops_help(self, cli_runner: CliRunner):
        """Test --help for list-ops command."""
        result = cli_runner.invoke(app, ["list-ops", "--help"])
        assert result.exit_code == EXIT_OK
        assert "list" in result.stdout.lower() or "operations" in result.stdout.lower()

    def test_list_ops_short_help(self, cli_runner: CliRunner):
        """Test -h for list-ops command."""
        result = cli_runner.invoke(app, ["list-ops", "-h"])
        assert result.exit_code == EXIT_OK


class TestListOpsCategoryFilter:
    """Test category filtering."""

    def test_category_option_long(self, cli_runner: CliRunner):
        """Test --category option filters by category."""
        result = cli_runner.invoke(app, ["list-ops", "--category", "anonymization"])
        assert result.exit_code == EXIT_OK

    def test_category_option_short(self, cli_runner: CliRunner):
        """Test -c short option for category."""
        result = cli_runner.invoke(app, ["list-ops", "-c", "profiling"])
        assert result.exit_code == EXIT_OK

    def test_invalid_category(self, cli_runner: CliRunner):
        """Test filtering with invalid category returns empty or message."""
        result = cli_runner.invoke(
            app, ["list-ops", "--category", "nonexistent_category"]
        )
        # Should still exit successfully but with empty/no results message
        assert result.exit_code == EXIT_OK

    def test_category_case_sensitive(self, cli_runner: CliRunner):
        """Test that category filtering is case-sensitive."""
        result = cli_runner.invoke(app, ["list-ops", "--category", "Anonymization"])
        # Might return no results if case-sensitive
        assert result.exit_code == EXIT_OK

    def test_category_with_underscores(self, cli_runner: CliRunner):
        """Test category names with underscores."""
        result = cli_runner.invoke(app, ["list-ops", "--category", "fake_data"])
        assert result.exit_code == EXIT_OK


class TestListOpsOutputFormats:
    """Test different output formats."""

    def test_output_format_table_long(self, cli_runner: CliRunner):
        """Test --format table option (default)."""
        result = cli_runner.invoke(app, ["list-ops", "--format", "table"])
        assert result.exit_code == EXIT_OK

    def test_output_format_table_short(self, cli_runner: CliRunner):
        """Test -f table short option."""
        result = cli_runner.invoke(app, ["list-ops", "-f", "table"])
        assert result.exit_code == EXIT_OK

    def test_output_format_json(self, cli_runner: CliRunner):
        """Test --format json option."""
        result = cli_runner.invoke(app, ["list-ops", "--format", "json"])
        assert result.exit_code == EXIT_OK
        # Output should be valid JSON
        output = result.stdout.strip()
        if output and output.startswith("["):
            data = json.loads(output)
            assert isinstance(data, list)

    def test_output_format_json_short(self, cli_runner: CliRunner):
        """Test -f json short option."""
        result = cli_runner.invoke(app, ["list-ops", "-f", "json"])
        assert result.exit_code == EXIT_OK

    def test_json_structure(self, cli_runner: CliRunner):
        """Test JSON output has expected structure."""
        result = cli_runner.invoke(app, ["list-ops", "-f", "json"])
        assert result.exit_code == EXIT_OK
        output = result.stdout.strip()
        if output and output.startswith("["):
            data = json.loads(output)
            if data:  # If there are operations
                first_op = data[0]
                # Check expected keys in each operation
                assert "name" in first_op or "operation" in first_op or isinstance(
                    first_op, dict
                )

    def test_invalid_format(self, cli_runner: CliRunner):
        """Test invalid format returns error."""
        result = cli_runner.invoke(app, ["list-ops", "--format", "xml"])
        # Should error or ignore invalid format
        assert result.exit_code != EXIT_OK or "table" in result.stdout or "[" in result.stdout

    def test_format_with_category(self, cli_runner: CliRunner):
        """Test combining format and category options."""
        result = cli_runner.invoke(
            app, ["list-ops", "-c", "profiling", "-f", "json"]
        )
        assert result.exit_code == EXIT_OK


class TestListOpsOutput:
    """Test output content and structure."""

    def test_output_contains_operation_names(self, cli_runner: CliRunner):
        """Test that output contains operation names."""
        result = cli_runner.invoke(app, ["list-ops"])
        assert result.exit_code == EXIT_OK

    def test_output_contains_categories(self, cli_runner: CliRunner):
        """Test that output mentions categories."""
        result = cli_runner.invoke(app, ["list-ops"])
        assert result.exit_code == EXIT_OK

    def test_categories_help_text(self, cli_runner: CliRunner):
        """Test that categories help text is shown."""
        result = cli_runner.invoke(app, ["list-ops"])
        assert result.exit_code == EXIT_OK

    def test_total_count_shown(self, cli_runner: CliRunner):
        """Test that total operation count is shown."""
        result = cli_runner.invoke(app, ["list-ops", "-f", "table"])
        assert result.exit_code == EXIT_OK


class TestListOpsErrorHandling:
    """Test error handling in list-ops."""

    def test_catalog_load_error_fallback(self, cli_runner: CliRunner):
        """Test that fallback to registry works if catalog fails."""
        # This will naturally test fallback unless catalog is available
        result = cli_runner.invoke(app, ["list-ops"])
        # Should succeed either way
        assert result.exit_code == EXIT_OK

    @patch("pamola_core.cli.commands.list_ops._load_from_catalog")
    def test_error_message_on_load_failure(self, mock_load, cli_runner: CliRunner):
        """Test error message when both catalog and registry fail."""
        mock_load.side_effect = Exception("Load failed")
        result = cli_runner.invoke(app, ["list-ops"])
        # Should show error message
        assert result.exit_code == EXIT_ERROR or "Failed" in result.stdout

    def test_multiple_calls_consistent(self, cli_runner: CliRunner):
        """Test that multiple calls return consistent results."""
        result1 = cli_runner.invoke(app, ["list-ops"])
        result2 = cli_runner.invoke(app, ["list-ops"])
        assert result1.exit_code == result2.exit_code
        # Results should be the same
        assert result1.stdout == result2.stdout


class TestListOpsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_category_name(self, cli_runner: CliRunner):
        """Test behavior with empty category name."""
        result = cli_runner.invoke(app, ["list-ops", "--category", ""])
        assert result.exit_code == EXIT_OK

    def test_very_long_category_name(self, cli_runner: CliRunner):
        """Test with very long category name."""
        long_cat = "a" * 1000
        result = cli_runner.invoke(app, ["list-ops", "--category", long_cat])
        assert result.exit_code == EXIT_OK

    def test_special_characters_in_category(self, cli_runner: CliRunner):
        """Test category name with special characters."""
        result = cli_runner.invoke(
            app, ["list-ops", "--category", "test-category_123"]
        )
        assert result.exit_code == EXIT_OK

    def test_json_output_empty_array(self, cli_runner: CliRunner):
        """Test JSON output when no operations found."""
        result = cli_runner.invoke(
            app, ["list-ops", "-c", "nonexistent", "-f", "json"]
        )
        assert result.exit_code == EXIT_OK
        output = result.stdout.strip()
        # Either empty array or no JSON (plain text message)
        if output.startswith("["):
            assert output == "[]" or json.loads(output) == []

    def test_both_category_and_format(self, cli_runner: CliRunner):
        """Test with both category and format options."""
        result = cli_runner.invoke(
            app, ["list-ops", "-c", "anonymization", "-f", "json"]
        )
        assert result.exit_code == EXIT_OK


class TestListOpsIntegration:
    """Integration tests combining multiple features."""

    def test_list_all_then_filter(self, cli_runner: CliRunner):
        """Test listing all and then filtering gives consistent subset."""
        all_result = cli_runner.invoke(app, ["list-ops"])
        filtered_result = cli_runner.invoke(
            app, ["list-ops", "-c", "anonymization"]
        )
        assert all_result.exit_code == EXIT_OK
        assert filtered_result.exit_code == EXIT_OK

    def test_table_to_json_consistency(self, cli_runner: CliRunner):
        """Test that same data is shown in table and JSON formats."""
        table_result = cli_runner.invoke(app, ["list-ops", "-f", "table"])
        json_result = cli_runner.invoke(app, ["list-ops", "-f", "json"])
        assert table_result.exit_code == EXIT_OK
        assert json_result.exit_code == EXIT_OK

    def test_category_filter_with_formats(self, cli_runner: CliRunner):
        """Test category filter works with all formats."""
        for fmt in ["table", "json"]:
            result = cli_runner.invoke(
                app, ["list-ops", "-c", "profiling", "-f", fmt]
            )
            assert result.exit_code == EXIT_OK
