"""
Unit tests for Operation Result module (op_result.py)

Run with: pytest tests/utils/ops/test_op_result.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from pamola_core.utils.ops.op_result import (
    OperationResult, OperationArtifact, OperationStatus,
    ArtifactGroup, ValidationError
)


class TestOperationArtifact(unittest.TestCase):
    """Test cases for OperationArtifact class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

        # Create a test file
        self.test_file = self.test_path / "test_artifact.csv"
        with open(self.test_file, "w") as f:
            f.write("col1,col2\n1,2\n3,4\n")

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    @patch('pamola_core.utils.ops.op_result.get_file_metadata')
    def test_get_file_size(self, mock_get_file_metadata):
        """Test _get_file_size method."""
        # Mock the get_file_metadata function
        mock_get_file_metadata.return_value = {"size_bytes": 1024}

        # Create artifact - already calls get_file_metadata once
        artifact = OperationArtifact("csv", self.test_file, "Test artifact")

        # Reset the mock to clear the call history
        mock_get_file_metadata.reset_mock()

        # Test size retrieval
        size = artifact._get_file_size()

        # Verify the function was called once after reset
        mock_get_file_metadata.assert_called_once_with(self.test_file)
        self.assertEqual(size, 1024)

    @patch('pamola_core.utils.ops.op_result.calculate_checksum')
    def test_calculate_checksum(self, mock_calculate_checksum):
        """Test calculate_checksum method."""
        # Mock the calculate_checksum function
        mock_checksum = "abc123def456"
        mock_calculate_checksum.return_value = mock_checksum

        # Create artifact
        artifact = OperationArtifact("csv", self.test_file, "Test artifact")

        # Test checksum calculation
        checksum = artifact.calculate_checksum()

        # Verify
        mock_calculate_checksum.assert_called_once_with(self.test_file, 'sha256')
        self.assertEqual(checksum, mock_checksum)
        self.assertEqual(artifact.checksum, mock_checksum)

    def test_exists(self):
        """Test exists method."""
        # Test with existing file
        artifact = OperationArtifact("csv", self.test_file, "Test artifact")
        self.assertTrue(artifact.exists())

        # Test with non-existent file
        non_existent_file = self.test_path / "non_existent_file.csv"
        artifact = OperationArtifact("csv", non_existent_file, "Non-existent artifact")
        self.assertFalse(artifact.exists())

    @patch('pamola_core.utils.ops.op_result.get_file_metadata')
    def test_validate(self, mock_get_file_metadata):
        """Test validate method."""
        # Mock the get_file_metadata function
        mock_get_file_metadata.return_value = {
            "exists": True,
            "size_bytes": 1024,
            "extension": ".csv",
            "created_at": "2025-05-01T12:00:00",
            "modified_at": "2025-05-01T13:00:00"
        }

        # Create artifact - already calls get_file_metadata once
        artifact = OperationArtifact("csv", self.test_file, "Test artifact")
        artifact.checksum = "mock_checksum"  # Set a checksum to avoid calculation

        # Reset the mock to clear the call history
        mock_get_file_metadata.reset_mock()

        # Test validation
        validation_result = artifact.validate()

        # Verify the function was called once after reset
        mock_get_file_metadata.assert_called_once_with(self.test_file)
        self.assertTrue(validation_result["exists"])
        self.assertTrue(validation_result["size_valid"])
        self.assertTrue(validation_result["type_valid"])
        self.assertTrue(validation_result["is_valid"])
        self.assertEqual(validation_result["checksum"], "mock_checksum")
        self.assertEqual(validation_result["created_at"], "2025-05-01T12:00:00")
        self.assertEqual(validation_result["modified_at"], "2025-05-01T13:00:00")

    def test_add_tag(self):
        """Test add_tag method."""
        artifact = OperationArtifact("csv", self.test_file, "Test artifact")

        # Initial state
        self.assertEqual(artifact.tags, [])

        # Add a tag
        artifact.add_tag("test_tag")
        self.assertEqual(artifact.tags, ["test_tag"])

        # Add the same tag again (should not duplicate)
        artifact.add_tag("test_tag")
        self.assertEqual(artifact.tags, ["test_tag"])

        # Add another tag
        artifact.add_tag("another_tag")
        self.assertEqual(artifact.tags, ["test_tag", "another_tag"])

    def test_to_dict(self):
        """Test to_dict method."""
        artifact = OperationArtifact(
            "csv",
            self.test_file,
            "Test artifact",
            "output",
            ["tag1", "tag2"]
        )
        artifact.checksum = "mock_checksum"

        # Get dictionary representation
        dict_repr = artifact.to_dict()

        # Verify
        self.assertEqual(dict_repr["type"], "csv")
        self.assertEqual(dict_repr["path"], str(self.test_file))
        self.assertEqual(dict_repr["description"], "Test artifact")
        self.assertEqual(dict_repr["category"], "output")
        self.assertEqual(dict_repr["tags"], ["tag1", "tag2"])
        self.assertEqual(dict_repr["checksum"], "mock_checksum")
        self.assertIn("creation_time", dict_repr)
        self.assertIn("size", dict_repr)


class TestArtifactGroup(unittest.TestCase):
    """Test cases for ArtifactGroup class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

        # Create a test file
        self.test_file = self.test_path / "test_artifact.csv"
        with open(self.test_file, "w") as f:
            f.write("col1,col2\n1,2\n3,4\n")

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_add_and_get_artifacts(self):
        """Test add_artifact and get_artifacts methods."""
        # Create a group
        group = ArtifactGroup("test_group", "Test group description")

        # Verify initial state
        self.assertEqual(group.name, "test_group")
        self.assertEqual(group.description, "Test group description")
        self.assertEqual(group.artifacts, [])

        # Create artifacts
        artifact1 = OperationArtifact("csv", self.test_file, "Test artifact 1")
        artifact2 = OperationArtifact("json", self.test_path / "test.json", "Test artifact 2")

        # Add artifacts
        group.add_artifact(artifact1)
        group.add_artifact(artifact2)

        # Verify
        self.assertEqual(len(group.artifacts), 2)
        self.assertEqual(group.get_artifacts(), [artifact1, artifact2])

    def test_to_dict(self):
        """Test to_dict method."""
        # Create a group
        group = ArtifactGroup("test_group", "Test group description")

        # Create and add artifacts
        artifact1 = OperationArtifact("csv", self.test_file, "Test artifact 1")
        artifact2 = OperationArtifact("json", self.test_path / "test.json", "Test artifact 2")
        group.add_artifact(artifact1)
        group.add_artifact(artifact2)

        # Get dictionary representation
        dict_repr = group.to_dict()

        # Verify
        self.assertEqual(dict_repr["name"], "test_group")
        self.assertEqual(dict_repr["description"], "Test group description")
        self.assertEqual(len(dict_repr["artifacts"]), 2)
        self.assertEqual(dict_repr["artifacts"][0]["description"], "Test artifact 1")
        self.assertEqual(dict_repr["artifacts"][1]["description"], "Test artifact 2")


class TestOperationResult(unittest.TestCase):
    """Test cases for OperationResult class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

        # Create test files
        self.csv_file = self.test_path / "test.csv"
        with open(self.csv_file, "w") as f:
            f.write("col1,col2\n1,2\n3,4\n")

        self.json_file = self.test_path / "test.json"
        with open(self.json_file, "w") as f:
            json.dump({"key": "value"}, f) # type: ignore

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization."""
        # Default initialization
        result = OperationResult()
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.assertEqual(result.artifacts, [])
        self.assertEqual(result.metrics, {})
        self.assertIsNone(result.error_message)
        self.assertIsNone(result.execution_time)

        # Custom initialization
        result = OperationResult(
            status=OperationStatus.ERROR,
            artifacts=[OperationArtifact("csv", self.csv_file)],
            metrics={"metric1": 10},
            error_message="Test error",
            execution_time=1.5
        )
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual(result.metrics, {"metric1": 10})
        self.assertEqual(result.error_message, "Test error")
        self.assertEqual(result.execution_time, 1.5)

    def test_add_artifact(self):
        """Test add_artifact method."""
        result = OperationResult()

        # Add an artifact
        artifact = result.add_artifact(
            "csv",
            self.csv_file,
            "Test CSV",
            "output",
            ["tag1", "tag2"]
        )

        # Verify
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual(artifact.artifact_type, "csv")
        self.assertEqual(artifact.path, self.csv_file)
        self.assertEqual(artifact.description, "Test CSV")
        self.assertEqual(artifact.category, "output")
        self.assertEqual(artifact.tags, ["tag1", "tag2"])

        # Add to a group
        artifact = result.add_artifact(
            "json",
            self.json_file,
            "Test JSON",
            "output",
            ["tag3"],
            group="test_group"
        )

        # Verify group creation
        self.assertIn("test_group", result.artifact_groups)
        self.assertEqual(len(result.artifact_groups["test_group"].artifacts), 1)
        self.assertEqual(result.artifact_groups["test_group"].artifacts[0], artifact)

    def test_register_artifact_via_writer(self):
        """Test register_artifact_via_writer method."""
        result = OperationResult()

        # Create a mock DataWriter
        mock_writer = MagicMock()
        mock_writer.write_dataframe.return_value = MagicMock(
            path=self.csv_file,
            size_bytes=100,
            timestamp="2025-05-01T12:00:00",
            format="csv"
        )
        mock_writer.write_json.return_value = MagicMock(
            path=self.json_file,
            size_bytes=50,
            timestamp="2025-05-01T12:00:00",
            format="json"
        )

        # Create test data
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        json_data = {"key": "value"}

        # Test with DataFrame
        artifact = result.register_artifact_via_writer(
            mock_writer,
            df,
            "output",
            "test_df",
            description="Test DataFrame",
            tags=["data"]
        )

        # Verify
        mock_writer.write_dataframe.assert_called_once()
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual(artifact.path, self.csv_file)
        self.assertEqual(artifact.artifact_type, "csv")

        # Test with JSON data
        artifact = result.register_artifact_via_writer(
            mock_writer,
            json_data,
            "dictionaries",
            "test_json",
            artifact_type="json",
            description="Test JSON",
            tags=["dict"],
            group="test_group"
        )

        # Verify
        mock_writer.write_json.assert_called_once()
        self.assertEqual(len(result.artifacts), 2)
        self.assertEqual(artifact.path, self.json_file)
        self.assertEqual(artifact.artifact_type, "json")
        self.assertIn("test_group", result.artifact_groups)

        # Test with unsupported combination
        with self.assertRaises(ValidationError):
            result.register_artifact_via_writer(
                mock_writer,
                "unsupported",
                "output",
                "test_unsupported",
                artifact_type="unknown"
            )

    def test_add_artifact_group(self):
        """Test add_artifact_group method."""
        result = OperationResult()

        # Add a group
        group = result.add_artifact_group("test_group", "Test group description")

        # Verify
        self.assertIn("test_group", result.artifact_groups)
        self.assertEqual(group.name, "test_group")
        self.assertEqual(group.description, "Test group description")

        # Add the same group again (should return existing group)
        group2 = result.add_artifact_group("test_group", "New description")
        self.assertIs(group, group2)
        self.assertEqual(group.description, "Test group description")  # Description not updated

    def test_add_metric(self):
        """Test add_metric method."""
        result = OperationResult()

        # Add metrics
        result.add_metric("metric1", 10)
        result.add_metric("metric2", "value")
        result.add_metric("metric3", True)

        # Verify
        self.assertEqual(result.metrics, {
            "metric1": 10,
            "metric2": "value",
            "metric3": True
        })

    def test_add_nested_metric(self):
        """Test add_nested_metric method."""
        result = OperationResult()

        # Add nested metrics
        result.add_nested_metric("category1", "metric1", 10)
        result.add_nested_metric("category1", "metric2", "value")
        result.add_nested_metric("category2", "metric3", True)

        # Verify
        self.assertEqual(result.metrics, {
            "category1": {
                "metric1": 10,
                "metric2": "value"
            },
            "category2": {
                "metric3": True
            }
        })

    def test_get_artifacts_methods(self):
        """Test artifact retrieval methods."""
        result = OperationResult()

        # Add artifacts
        artifact1 = result.add_artifact("csv", self.csv_file, "CSV 1", tags=["tag1", "common"])
        artifact2 = result.add_artifact("csv", self.csv_file, "CSV 2", tags=["tag2", "common"])
        artifact3 = result.add_artifact("json", self.json_file, "JSON", category="metric", tags=["tag3"])

        # Test get_artifacts_by_type
        csv_artifacts = result.get_artifacts_by_type("csv")
        self.assertEqual(len(csv_artifacts), 2)
        self.assertIn(artifact1, csv_artifacts)
        self.assertIn(artifact2, csv_artifacts)

        json_artifacts = result.get_artifacts_by_type("json")
        self.assertEqual(len(json_artifacts), 1)
        self.assertEqual(json_artifacts[0], artifact3)

        # Test get_artifacts_by_tag
        tag1_artifacts = result.get_artifacts_by_tag("tag1")
        self.assertEqual(len(tag1_artifacts), 1)
        self.assertEqual(tag1_artifacts[0], artifact1)

        common_artifacts = result.get_artifacts_by_tag("common")
        self.assertEqual(len(common_artifacts), 2)
        self.assertIn(artifact1, common_artifacts)
        self.assertIn(artifact2, common_artifacts)

        # Test get_artifacts_by_category
        output_artifacts = result.get_artifacts_by_category("output")
        self.assertEqual(len(output_artifacts), 2)
        self.assertIn(artifact1, output_artifacts)
        self.assertIn(artifact2, output_artifacts)

        metric_artifacts = result.get_artifacts_by_category("metric")
        self.assertEqual(len(metric_artifacts), 1)
        self.assertEqual(metric_artifacts[0], artifact3)

    def test_get_artifact_group(self):
        """Test get_artifact_group method."""
        result = OperationResult()

        # Add group
        group = result.add_artifact_group("test_group", "Test group")

        # Test retrieval
        retrieved_group = result.get_artifact_group("test_group")
        self.assertIs(retrieved_group, group)

        # Test non-existent group
        self.assertIsNone(result.get_artifact_group("non_existent"))

    @patch('pamola_core.utils.ops.op_result.OperationArtifact.validate')
    def test_validate_artifacts(self, mock_validate):
        """Test validate_artifacts method."""
        result = OperationResult()

        # Add artifacts
        artifact1 = result.add_artifact("csv", self.csv_file, "CSV 1")
        artifact2 = result.add_artifact("json", self.json_file, "JSON")

        # Mock validate method
        mock_validate.side_effect = [
            {"exists": True, "size_valid": True, "type_valid": True, "is_valid": True},
            {"exists": True, "size_valid": False, "type_valid": True, "is_valid": False}
        ]

        # Test validation
        validation_result = result.validate_artifacts()

        # Verify
        self.assertEqual(mock_validate.call_count, 2)
        self.assertFalse(validation_result["all_valid"])
        self.assertEqual(validation_result["invalid_count"], 1)
        self.assertEqual(validation_result["invalid_artifacts"], [str(self.json_file)])

    def test_to_reporter_details(self):
        """Test to_reporter_details method."""
        result = OperationResult(
            status=OperationStatus.SUCCESS,
            execution_time=2.5,
            error_message=None
        )

        # Add metrics
        result.add_metric("metric1", 10)
        result.add_metric("metric2", "value")
        result.add_nested_metric("category", "nested_metric", 20)

        # Add artifacts
        result.add_artifact("csv", self.csv_file, "CSV 1")
        result.add_artifact("csv", self.csv_file, "CSV 2")
        result.add_artifact("json", self.json_file, "JSON")

        # Get reporter details
        details = result.to_reporter_details()

        # Verify
        self.assertEqual(details["status"], "success")
        self.assertEqual(details["execution_time"], "2.50 seconds")
        self.assertEqual(details["metric_metric1"], 10)
        self.assertEqual(details["metric_metric2"], "value")
        self.assertEqual(details["artifacts_count"], 3)
        self.assertEqual(details["artifacts_csv_count"], 2)
        self.assertEqual(details["artifacts_json_count"], 1)


if __name__ == "__main__":
    unittest.main()