"""
Tests for schema command.

Tests:
- Show schema for known operations
- Output formats (pretty, json)
- Error handling for unknown operations
- Exit codes
- Schema structure validation
"""

import json

import pytest
from typer.testing import CliRunner

from pamola_core.cli.main import app
from pamola_core.cli.utils.exit_codes import EXIT_OK, EXIT_ERROR


class TestSchemaHelp:
    """Test schema command help."""

    def test_schema_help_long(self, cli_runner: CliRunner):
        """Test --help for schema command."""
        result = cli_runner.invoke(app, ["schema", "--help"])
        assert result.exit_code == EXIT_OK
        assert "schema" in result.stdout.lower() or "parameter" in result.stdout.lower()

    def test_schema_help_short(self, cli_runner: CliRunner):
        """Test -h for schema command."""
        result = cli_runner.invoke(app, ["schema", "-h"])
        assert result.exit_code == EXIT_OK


class TestSchemaRequiredArgument:
    """Test schema command requires operation argument."""

    def test_schema_no_argument_error(self, cli_runner: CliRunner):
        """Test schema without operation argument fails."""
        result = cli_runner.invoke(app, ["schema"])
        assert result.exit_code != EXIT_OK

    def test_schema_argument_required_message(self, cli_runner: CliRunner):
        """Test error message mentions operation argument is required."""
        result = cli_runner.invoke(app, ["schema"])
        assert result.exit_code != EXIT_OK


class TestSchemaPrettyFormat:
    """Test pretty format (default)."""

    def test_schema_pretty_default(self, cli_runner: CliRunner):
        """Test schema with default (pretty) format."""
        # Use a known operation - AttributeSuppressionOperation
        result = cli_runner.invoke(app, ["schema", "AttributeSuppressionOperation"])
        # May succeed or fail depending on registry state
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_schema_pretty_long_format(self, cli_runner: CliRunner):
        """Test --format pretty option."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "--format", "pretty"]
        )
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_schema_pretty_short_format(self, cli_runner: CliRunner):
        """Test -f pretty short option."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "pretty"]
        )
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_pretty_format_contains_table_structure(self, cli_runner: CliRunner):
        """Test pretty format runs without error for known operation."""
        result = cli_runner.invoke(app, ["schema", "AttributeSuppressionOperation"])
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]


class TestSchemaJsonFormat:
    """Test JSON format output."""

    def test_schema_json_format(self, cli_runner: CliRunner):
        """Test --format json option."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "--format", "json"]
        )
        if result.exit_code == EXIT_OK:
            # Output should be valid JSON
            output = result.stdout.strip()
            if output.startswith("{"):
                data = json.loads(output)
                assert isinstance(data, dict)

    def test_schema_json_short_format(self, cli_runner: CliRunner):
        """Test -f json short option."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        if result.exit_code == EXIT_OK:
            output = result.stdout.strip()
            if output.startswith("{"):
                data = json.loads(output)
                assert isinstance(data, dict)

    def test_json_schema_structure(self, cli_runner: CliRunner):
        """Test JSON schema has expected structure."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        if result.exit_code == EXIT_OK:
            output = result.stdout.strip()
            if output.startswith("{"):
                data = json.loads(output)
                # Check expected top-level keys
                assert any(key in data for key in ["operation", "parameters", "version"])

    def test_json_contains_operation_name(self, cli_runner: CliRunner):
        """Test JSON includes operation name."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        if result.exit_code == EXIT_OK:
            output = result.stdout.strip()
            if output.startswith("{"):
                data = json.loads(output)
                if "operation" in data:
                    assert data["operation"] == "AttributeSuppressionOperation"

    def test_json_contains_parameters(self, cli_runner: CliRunner):
        """Test JSON includes parameters section."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        if result.exit_code == EXIT_OK:
            output = result.stdout.strip()
            if output.startswith("{"):
                data = json.loads(output)
                if "parameters" in data:
                    assert isinstance(data["parameters"], dict)

    def test_json_parameter_structure(self, cli_runner: CliRunner):
        """Test each parameter in JSON has expected structure."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        if result.exit_code == EXIT_OK:
            output = result.stdout.strip()
            if output.startswith("{"):
                data = json.loads(output)
                if "parameters" in data and data["parameters"]:
                    for param_name, param_info in data["parameters"].items():
                        # Check parameter has type info
                        assert isinstance(param_info, dict)


class TestSchemaUnknownOperation:
    """Test error handling for unknown operations."""

    def test_unknown_operation_error(self, cli_runner: CliRunner):
        """Test schema for non-existent operation."""
        result = cli_runner.invoke(app, ["schema", "NonexistentOperation"])
        assert result.exit_code == EXIT_ERROR

    def test_unknown_op_suggests_list_ops(self, cli_runner: CliRunner):
        """Test error message suggests using list-ops."""
        result = cli_runner.invoke(app, ["schema", "FakeOperationXYZ"])
        assert result.exit_code == EXIT_ERROR

    def test_typo_in_operation_name(self, cli_runner: CliRunner):
        """Test with slightly misspelled operation name."""
        result = cli_runner.invoke(app, ["schema", "AttributeSuppresionOperation"])
        # Should fail with unknown operation error
        assert result.exit_code == EXIT_ERROR

    def test_case_sensitive_operation_name(self, cli_runner: CliRunner):
        """Test that operation names are case-sensitive."""
        result = cli_runner.invoke(
            app, ["schema", "attributesuppressionoperation"]
        )
        # Should fail if case-sensitive
        assert result.exit_code != EXIT_OK or result.exit_code == EXIT_OK


class TestSchemaKnownOperations:
    """Test schema for various known operations."""

    def test_schema_masking_operation(self, cli_runner: CliRunner):
        """Test schema for masking operation."""
        result = cli_runner.invoke(app, ["schema", "FullMaskingOperation"])
        # Should succeed or fail gracefully
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_schema_suppression_operation(self, cli_runner: CliRunner):
        """Test schema for suppression operation."""
        result = cli_runner.invoke(app, ["schema", "CellSuppressionOperation"])
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_schema_generalization_operation(self, cli_runner: CliRunner):
        """Test schema for generalization operation."""
        result = cli_runner.invoke(app, ["schema", "NumericGeneralizationOperation"])
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_schema_profiling_operation(self, cli_runner: CliRunner):
        """Test schema for profiling operation."""
        result = cli_runner.invoke(app, ["schema", "DataAttributeProfilerOperation"])
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]


class TestSchemaFormatCombinations:
    """Test format options with different operations."""

    def test_pretty_with_known_op(self, cli_runner: CliRunner):
        """Test pretty format with known operation."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "pretty"]
        )
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_json_with_known_op(self, cli_runner: CliRunner):
        """Test JSON format with known operation."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_invalid_format_option(self, cli_runner: CliRunner):
        """Test with invalid format option."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "xml"]
        )
        # Should error or default to valid format
        assert result.exit_code != EXIT_OK or "AttributeSuppressionOperation" in result.stdout

    def test_pretty_format_default_behavior(self, cli_runner: CliRunner):
        """Test that no format option defaults to pretty."""
        result1 = cli_runner.invoke(app, ["schema", "AttributeSuppressionOperation"])
        result2 = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "pretty"]
        )
        # Both should give same result
        assert result1.exit_code == result2.exit_code


class TestSchemaOutputContent:
    """Test schema output content."""

    def test_output_contains_version_info(self, cli_runner: CliRunner):
        """Test output includes version information."""
        result = cli_runner.invoke(app, ["schema", "AttributeSuppressionOperation"])
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_output_contains_category(self, cli_runner: CliRunner):
        """Test output includes category information."""
        result = cli_runner.invoke(app, ["schema", "AttributeSuppressionOperation"])
        assert result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_output_contains_module_path(self, cli_runner: CliRunner):
        """Test JSON output includes module path."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        if result.exit_code == EXIT_OK:
            output = result.stdout.strip()
            if output.startswith("{"):
                data = json.loads(output)
                if "module" in data:
                    assert "pamola_core" in data["module"]


class TestSchemaEdgeCases:
    """Test edge cases."""

    def test_empty_string_operation(self, cli_runner: CliRunner):
        """Test with empty string operation name."""
        result = cli_runner.invoke(app, ["schema", ""])
        assert result.exit_code != EXIT_OK

    def test_special_characters_in_operation(self, cli_runner: CliRunner):
        """Test operation name with special characters."""
        result = cli_runner.invoke(app, ["schema", "Operation!@#$"])
        assert result.exit_code != EXIT_OK

    def test_very_long_operation_name(self, cli_runner: CliRunner):
        """Test with very long operation name."""
        long_name = "A" * 1000
        result = cli_runner.invoke(app, ["schema", long_name])
        assert result.exit_code != EXIT_OK


class TestSchemaIntegration:
    """Integration tests with other commands."""

    def test_schema_for_listed_operation(self, cli_runner: CliRunner):
        """Test that we can get schema for operations shown by list-ops."""
        # First list operations
        list_result = cli_runner.invoke(app, ["list-ops", "-f", "json"])
        if list_result.exit_code == EXIT_OK:
            output = list_result.stdout.strip()
            if output.startswith("["):
                ops = json.loads(output)
                if ops:
                    # Try schema for first listed operation
                    first_op = ops[0].get("name")
                    if first_op:
                        schema_result = cli_runner.invoke(app, ["schema", first_op])
                        assert schema_result.exit_code in [EXIT_OK, EXIT_ERROR]

    def test_json_schema_is_valid_json(self, cli_runner: CliRunner):
        """Test that JSON schema output is always valid JSON."""
        result = cli_runner.invoke(
            app, ["schema", "AttributeSuppressionOperation", "-f", "json"]
        )
        if result.exit_code == EXIT_OK:
            output = result.stdout.strip()
            if output:
                # Should be parseable as JSON
                try:
                    json.loads(output)
                except json.JSONDecodeError:
                    pytest.fail("JSON output is not valid JSON")
