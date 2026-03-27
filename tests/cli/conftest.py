"""
Shared pytest fixtures for CLI tests.

Provides CliRunner, temporary files, and mocked registry.
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a CliRunner for testing CLI commands."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test files.

    Uses pytest's tmp_path fixture which handles Windows cleanup correctly,
    avoiding PermissionError on directory removal after tests.
    """
    return tmp_path


@pytest.fixture
def sample_task_json(temp_dir: Path) -> Path:
    """Create a sample task JSON file."""
    task_def = {
        "task_id": "test_task_001",
        "task_type": "anonymization",
        "description": "Test anonymization task",
        "input_datasets": {
            "customers": str(temp_dir / "input.csv")
        },
        "auxiliary_datasets": {},
        "data_types": {},
        "operations": [
            {
                "operation": "single_op",
                "class_name": "AttributeSuppressionOperation",
                "parameters": {
                    "fields": ["ssn", "email"]
                },
                "scope": {
                    "target": ["ssn", "email"]
                },
                "dataset_name": "customers",
                "task_operation_id": "op_001",
                "task_operation_order_index": 1
            }
        ],
        "additional_options": {}
    }

    task_file = temp_dir / "task.json"
    task_file.write_text(json.dumps(task_def, indent=2), encoding="utf-8")
    return task_file


@pytest.fixture
def sample_op_config_json(temp_dir: Path) -> Path:
    """Create a sample operation config JSON file."""
    config = {
        "operation": "AttributeSuppressionOperation",
        "parameters": {
            "fields": ["ssn"]
        },
        "scope": {
            "target": ["ssn"]
        }
    }

    config_file = temp_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_file


@pytest.fixture
def invalid_json_file(temp_dir: Path) -> Path:
    """Create an invalid JSON file for error testing."""
    invalid_file = temp_dir / "invalid.json"
    invalid_file.write_text("{invalid json content", encoding="utf-8")
    return invalid_file


@pytest.fixture
def sample_csv_file(temp_dir: Path) -> Path:
    """Create a sample CSV file for operation testing."""
    csv_content = """id,name,email,ssn
1,John Doe,john@example.com,123-45-6789
2,Jane Smith,jane@example.com,987-65-4321
3,Bob Johnson,bob@example.com,555-55-5555"""

    csv_file = temp_dir / "data.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file
