"""
Unit tests for op_test_helpers module.

These tests verify the functionality of the testing utilities provided by the
op_test_helpers module, including MockDataSource, StubDataWriter, assertion helpers,
and test environment creation utilities.

Run with: pytest -s tests/utils/ops/test_op_test_helpers.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from pamola_core.utils.ops.op_test_helpers import (
    MockDataSource, StubDataWriter, CallRecord,
    assert_artifact_exists, assert_metrics_content,
    create_test_operation_env
)


class TestMockDataSource:
    """Tests for the MockDataSource class."""

    def test_init_empty(self):
        """Test initializing an empty MockDataSource."""
        mock_ds = MockDataSource()
        assert len(mock_ds.dataframes) == 0

        # Try to get non-existent dataframe
        df, error_info = mock_ds.get_dataframe("non_existent")
        assert df is None
        assert error_info is not None
        assert "not found" in error_info["message"]

    def test_init_with_dataframes(self):
        """Test initializing MockDataSource with dataframes."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        mock_ds = MockDataSource({
            "df1": df1,
            "df2": df2
        })

        assert len(mock_ds.dataframes) == 2
        assert "df1" in mock_ds.dataframes
        assert "df2" in mock_ds.dataframes

    def test_from_dataframe(self):
        """Test factory method for creating from a single dataframe."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        # With default name
        mock_ds = MockDataSource.from_dataframe(df)
        assert "main" in mock_ds.dataframes
        assert len(mock_ds.dataframes["main"]) == 3

        # With custom name
        mock_ds2 = MockDataSource.from_dataframe(df, name="custom")
        assert "custom" in mock_ds2.dataframes
        assert "main" not in mock_ds2.dataframes

    def test_add_dataframe(self):
        """Test adding a dataframe to an existing MockDataSource."""
        mock_ds = MockDataSource()
        df = pd.DataFrame({'a': [1, 2, 3]})

        mock_ds.add_dataframe("test_df", df)
        assert "test_df" in mock_ds.dataframes

        # Get the dataframe back
        retrieved_df, error_info = mock_ds.get_dataframe("test_df")
        assert error_info is None
        assert retrieved_df is not None
        assert len(retrieved_df) == 3
        assert 'a' in retrieved_df.columns

        # Verify we get a copy
        assert retrieved_df is not df

    def test_get_schema(self):
        """Test getting schema information from a dataframe."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })

        mock_ds = MockDataSource.from_dataframe(df)
        schema = mock_ds.get_schema("main")

        assert schema is not None
        assert "columns" in schema
        assert "dtypes" in schema
        assert "row_count" in schema

        assert set(schema["columns"]) == {'int_col', 'float_col', 'str_col'}
        assert schema["row_count"] == 3
        assert "int" in schema["dtypes"]["int_col"]
        assert "float" in schema["dtypes"]["float_col"]

        # Test non-existent dataframe
        assert mock_ds.get_schema("non_existent") is None

    def test_has_dataframe(self):
        """Test checking dataframe existence."""
        mock_ds = MockDataSource()
        df = pd.DataFrame({'a': [1, 2, 3]})

        assert not mock_ds.has_dataframe("test_df")

        mock_ds.add_dataframe("test_df", df)
        assert mock_ds.has_dataframe("test_df")

    def test_context_manager(self):
        """Test using MockDataSource as a context manager."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        with MockDataSource.from_dataframe(df) as mock_ds:
            assert mock_ds.has_dataframe("main")
            retrieved_df, _ = mock_ds.get_dataframe("main")
            assert len(retrieved_df) == 3

        # After context exit, dataframes should be cleared
        assert not mock_ds.dataframes


class TestStubDataWriter:
    """Tests for the StubDataWriter class."""

    def test_init_with_tempdir(self):
        """Test initializing StubDataWriter with auto temp directory."""
        writer = StubDataWriter()

        # Check that temp directory was created
        assert writer._temp_dir is not None
        assert writer.task_dir.exists()

        # Check standard directories were created
        assert (writer.task_dir / "output").exists()
        assert (writer.task_dir / "dictionaries").exists()
        assert (writer.task_dir / "logs").exists()

    def test_init_with_specified_dir(self, tmp_path):
        """Test initializing with a specified directory."""
        task_dir = tmp_path / "task_dir"
        writer = StubDataWriter(task_dir)

        # Check that specified directory was used
        assert writer.task_dir == task_dir
        assert writer._temp_dir is None

        # Check standard directories were created
        assert (task_dir / "output").exists()
        assert (task_dir / "dictionaries").exists()
        assert (task_dir / "logs").exists()

    def test_write_dataframe_csv(self, tmp_path):
        """Test writing a dataframe to CSV."""
        writer = StubDataWriter(tmp_path)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        result = writer.write_dataframe(
            df=df,
            name="test_data",
            format="csv",
            subdir="output"
        )

        # Check that file was created
        assert result.path.exists()
        assert result.path.is_file()
        assert result.path.suffix == ".csv"
        assert result.format == "csv"
        assert result.size_bytes > 0

        # Check call was recorded
        calls = writer.get_calls("write_dataframe")
        assert len(calls) == 1
        assert calls[0].method == "write_dataframe"
        assert calls[0].params["name"] == "test_data"
        assert calls[0].params["format"] == "csv"
        assert calls[0].params["subdir"] == "output"

        # Read back the file to verify content
        df_read = pd.read_csv(result.path, index_col=0)
        assert len(df_read) == 3
        assert list(df_read['b']) == [4, 5, 6]

    def test_write_dataframe_with_timestamp(self, tmp_path):
        """Test writing dataframe with timestamp in filename."""
        writer = StubDataWriter(tmp_path)
        df = pd.DataFrame({'a': [1, 2, 3]})

        result = writer.write_dataframe(
            df=df,
            name="test_data",
            format="csv",
            timestamp_in_name=True
        )

        # Check that file was created with timestamp
        assert result.path.exists()

        # Filename should match timestamp pattern
        timestamp_pattern = r"\d{8}T\d{6}_test_data\.csv"
        assert re.match(timestamp_pattern, result.path.name)

        # Check call was recorded
        calls = writer.get_calls("write_dataframe")
        assert len(calls) == 1
        assert calls[0].params["timestamp_in_name"] is True

    def test_write_json(self, tmp_path):
        """Test writing JSON data."""
        writer = StubDataWriter(tmp_path)
        data = {'a': 1, 'b': 2, 'nested': {'c': 3}}

        result = writer.write_json(
            data=data,
            name="test_json",
            subdir=None,  # Root directory
            pretty=True
        )

        # Check that file was created
        assert result.path.exists()
        assert result.path.is_file()
        assert result.path.suffix == ".json"
        assert result.format == "json"

        # Check call was recorded
        calls = writer.get_calls("write_json")
        assert len(calls) == 1
        assert calls[0].method == "write_json"
        assert calls[0].params["name"] == "test_json"
        assert calls[0].params["pretty"] is True
        assert "a" in calls[0].params["data_keys"]

        # Read back the file to verify content
        with open(result.path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data['a'] == 1
        assert loaded_data['nested']['c'] == 3

    def test_write_metrics(self, tmp_path):
        """Test writing metrics data."""
        writer = StubDataWriter(tmp_path)
        metrics = {
            'count': 100,
            'mean': 42.5,
            'stats': {
                'min': 10,
                'max': 90
            }
        }

        result = writer.write_metrics(
            metrics=metrics,
            name="test_metrics"
        )

        # Check that file was created
        assert result.path.exists()
        assert result.path.is_file()
        assert result.path.suffix == ".json"

        # Metrics should be in root directory
        assert result.path.parent == tmp_path

        # Check call was recorded
        calls = writer.get_calls("write_json")  # write_metrics calls write_json
        assert len(calls) == 1

        # Read back the file to verify content
        with open(result.path, 'r') as f:
            loaded_data = json.load(f)

        # Should have metadata and metrics
        assert "metadata" in loaded_data
        assert "metrics" in loaded_data
        assert loaded_data["metrics"]["count"] == 100
        assert loaded_data["metrics"]["stats"]["min"] == 10
        assert loaded_data["metadata"]["stub"] is True

    def test_write_dictionary(self, tmp_path):
        """Test writing a dictionary."""
        writer = StubDataWriter(tmp_path)
        data = {
            'id1': 'value1',
            'id2': 'value2',
            'id3': 'value3'
        }

        result = writer.write_dictionary(
            data=data,
            name="test_dict",
            format="json"
        )

        # Check that file was created in dictionaries subdir
        assert result.path.exists()
        assert result.path.is_file()
        assert result.path.parent.name == "dictionaries"

        # Check call was recorded
        calls = writer.get_calls("write_json")  # write_dictionary calls write_json
        assert len(calls) == 1

        # Read back the file to verify content
        with open(result.path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data['id1'] == 'value1'
        assert loaded_data['id3'] == 'value3'

    def test_get_calls(self, tmp_path):
        """Test getting recorded calls."""
        writer = StubDataWriter(tmp_path)

        # Create various artifacts
        df = pd.DataFrame({'a': [1, 2, 3]})
        writer.write_dataframe(df, name="df1", format="csv")
        writer.write_dataframe(df, name="df2", format="csv")
        writer.write_json({'x': 1}, name="json1")

        # Get all calls
        all_calls = writer.get_calls()
        assert len(all_calls) == 3

        # Get calls by method
        df_calls = writer.get_calls("write_dataframe")
        assert len(df_calls) == 2
        assert all(isinstance(call, CallRecord) for call in df_calls)
        assert all(call.method == "write_dataframe" for call in df_calls)

        json_calls = writer.get_calls("write_json")
        assert len(json_calls) == 1
        assert json_calls[0].method == "write_json"

    def test_clear_calls(self, tmp_path):
        """Test clearing recorded calls."""
        writer = StubDataWriter(tmp_path)

        # Create artifacts
        df = pd.DataFrame({'a': [1, 2, 3]})
        writer.write_dataframe(df, name="df1", format="csv")
        writer.write_json({'x': 1}, name="json1")

        # Verify calls were recorded
        assert len(writer.get_calls()) == 2

        # Clear calls
        writer.clear_calls()
        assert len(writer.get_calls()) == 0

        # Verify we can still record new calls
        writer.write_dataframe(df, name="df2", format="csv")
        assert len(writer.get_calls()) == 1

    def test_context_manager(self):
        """Test using StubDataWriter as a context manager."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        with StubDataWriter() as writer:
            # Write a file
            result = writer.write_dataframe(df, name="test", format="csv")
            assert result.path.exists()

            # Get the temp dir
            temp_dir_path = writer._temp_dir.name

        # After context exit, temp directory should be cleaned up
        assert not Path(temp_dir_path).exists()


class TestAssertionHelpers:
    """Tests for the assertion helper functions."""

    def test_assert_artifact_exists_success(self, tmp_path):
        """Test assertion for existing artifact."""
        # Create a test file
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_file = output_dir / "test_data.csv"
        test_file.write_text("test content")

        # Assert file exists
        result = assert_artifact_exists(tmp_path, "output", r"test_.*\.csv")
        assert result == test_file

    def test_assert_artifact_exists_with_timestamp(self, tmp_path):
        """Test assertion for artifact with timestamp in name."""
        # Create a test file with timestamp
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_file = output_dir / "20250503T142530_test_data.csv"
        test_file.write_text("test content")

        # Assert file exists
        result = assert_artifact_exists(tmp_path, "output", r"\d{8}T\d{6}_test_.*\.csv")
        assert result == test_file

    def test_assert_artifact_exists_failure(self, tmp_path):
        """Test assertion fails for non-existing artifact."""
        # Create directory but no matching file
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Assert raises error
        with pytest.raises(AssertionError) as excinfo:
            assert_artifact_exists(tmp_path, "output", r"test_.*\.csv")

        assert "No file matching pattern" in str(excinfo.value)

    def test_assert_artifact_exists_directory_not_found(self, tmp_path):
        """Test assertion fails when directory doesn't exist."""
        # Directory doesn't exist
        with pytest.raises(AssertionError) as excinfo:
            assert_artifact_exists(tmp_path, "non_existent", r"test_.*\.csv")

        assert "Directory does not exist" in str(excinfo.value)

    def test_assert_metrics_content_success(self, tmp_path):
        """Test assertion for metrics content success."""
        # Create a metrics file
        metrics_data = {
            "metrics": {
                "count": 100,
                "mean": 42.5,
                "stats": {
                    "min": 10,
                    "max": 90
                }
            }
        }

        metrics_file = tmp_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f)

        # Assert partial match works
        result = assert_metrics_content(tmp_path, {
            "count": 100,
            "stats": {"min": 10}
        })

        assert result == metrics_data

    def test_assert_metrics_content_failure(self, tmp_path):
        """Test assertion fails with mismatched content."""
        # Create a metrics file
        metrics_data = {
            "metrics": {
                "count": 100,
                "mean": 42.5
            }
        }

        metrics_file = tmp_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f)

        # Assert fails when expected key is missing
        with pytest.raises(AssertionError) as excinfo:
            assert_metrics_content(tmp_path, {
                "missing_key": "value"
            })

        assert "Missing keys" in str(excinfo.value)

        # Assert fails when value doesn't match
        with pytest.raises(AssertionError) as excinfo:
            assert_metrics_content(tmp_path, {
                "count": 200  # Actual is 100
            })

        assert "Mismatched values" in str(excinfo.value)

    def test_assert_metrics_content_nested_match(self, tmp_path):
        """Test assertion with nested metrics structure."""
        # Create a metrics file with nested structure
        metrics_data = {
            "metrics": {
                "levels": {
                    "level1": {
                        "value": 10,
                        "nested": {
                            "deep": 42
                        }
                    }
                }
            }
        }

        metrics_file = tmp_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f)

        # Assert nested structure matches
        result = assert_metrics_content(tmp_path, {
            "levels": {
                "level1": {
                    "nested": {
                        "deep": 42
                    }
                }
            }
        })

        assert result == metrics_data

        # Assert fails when deep nested value doesn't match
        with pytest.raises(AssertionError) as excinfo:
            assert_metrics_content(tmp_path, {
                "levels": {
                    "level1": {
                        "nested": {
                            "deep": 43  # Actual is 42
                        }
                    }
                }
            })

        assert "Mismatched values" in str(excinfo.value)


class TestCreateTestEnvironment:
    """Tests for the test environment creation helper."""

    def test_create_test_operation_env_basic(self, tmp_path):
        """Test creating a basic test environment."""
        task_dir, config = create_test_operation_env(tmp_path)

        # Check directories were created
        assert task_dir.exists()
        assert (task_dir / "output").exists()
        assert (task_dir / "dictionaries").exists()
        assert (task_dir / "logs").exists()

        # Check config file was created
        config_file = task_dir / "config.json"
        assert config_file.exists()

        # In MockOperationConfig, attributes are just the dict entries
        assert config.to_dict()["operation_name"] == "test_operation"
        assert config.to_dict()["version"] == "1.0.0"

        # Load config file and check content
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        assert config_data["operation_name"] == "test_operation"
        assert "parameters" in config_data
        assert "field_name" in config_data["parameters"]

    def test_create_test_operation_env_with_overrides(self, tmp_path):
        """Test creating environment with config overrides."""
        overrides = {
            "operation_name": "custom_operation",
            "version": "2.0.0",
            "parameters": {
                "custom_param": "custom_value",
                "threshold": 0.75
            }
        }

        task_dir, config = create_test_operation_env(tmp_path, overrides)

        # Check config was overridden
        assert config.to_dict()["operation_name"] == "custom_operation"
        assert config.to_dict()["version"] == "2.0.0"

        # Load config file and check content
        with open(task_dir / "config.json", 'r') as f:
            config_data = json.load(f)

        assert config_data["operation_name"] == "custom_operation"
        assert config_data["parameters"]["custom_param"] == "custom_value"
        assert config_data["parameters"]["threshold"] == 0.75

        # Should still have default parameters not overridden
        assert "field_name" in config_data["parameters"]

    def test_create_test_operation_env_with_nested_overrides(self, tmp_path):
        """Test with deeply nested config overrides."""
        overrides = {
            "parameters": {
                "advanced": {
                    "nested": {
                        "value": 42
                    }
                }
            }
        }

        task_dir, config = create_test_operation_env(tmp_path, overrides)

        # Load config file and check content
        with open(task_dir / "config.json", 'r') as f:
            config_data = json.load(f)

        # Check nested structure was created
        assert "advanced" in config_data["parameters"]
        assert "nested" in config_data["parameters"]["advanced"]
        assert config_data["parameters"]["advanced"]["nested"]["value"] == 42


class TestIntegration:
    """Integration tests using multiple components together."""

    def test_full_testing_workflow(self, tmp_path):
        """Test using all components in a typical test workflow."""
        # Create test data
        df = pd.DataFrame({
            'id': range(1, 11),
            'value': [i * 2 for i in range(1, 11)]
        })

        # Create mocks and test environment
        data_source = MockDataSource.from_dataframe(df)
        task_dir, config = create_test_operation_env(
            tmp_path,
            {
                "operation_name": "test_workflow",
                "parameters": {
                    "threshold": 15
                }
            }
        )
        writer = StubDataWriter(task_dir)

        # Simulate a simple operation execution
        # (In a real test, this would be a real operation instance)
        def mock_operation_execute():
            # Get data
            input_df, _ = data_source.get_dataframe("main")

            # Apply threshold filter
            threshold = 15
            filtered_df = input_df[input_df['value'] > threshold]

            # Write outputs
            writer.write_dataframe(
                filtered_df,
                name="filtered_data",
                format="csv",
                timestamp_in_name=True
            )

            # Write metrics
            writer.write_metrics({
                "total_records": len(input_df),
                "filtered_records": len(filtered_df),
                "threshold": threshold
            }, name="operation_metrics")

            return True

        # Execute mock operation
        result = mock_operation_execute()
        assert result is True

        # Verify outputs using assertion helpers
        output_file = assert_artifact_exists(task_dir, "output", r"\d{8}T\d{6}_filtered_data\.csv")

        # Read back the filtered data and verify
        df_filtered = pd.read_csv(output_file, index_col=0)

        # Fix: with values [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        # there are 3 values > 15: [16, 18, 20]
        assert len(df_filtered) == 3  # Values 16, 18, and 20 exceed threshold 15
        assert all(df_filtered['value'] > 15)

        # Verify metrics
        metrics = assert_metrics_content(task_dir, {
            "total_records": 10,
            "filtered_records": 3,  # Updated to match actual number of filtered records
            "threshold": 15
        })

        # Verify all expected calls were made
        assert len(writer.get_calls()) == 2
        assert len(writer.get_calls("write_dataframe")) == 1
        assert len(writer.get_calls("write_json")) == 1