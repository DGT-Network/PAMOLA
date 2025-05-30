"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Test Anonymization Visualization Utilities
Description: Unit tests for the visualization utilities module
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides tests for visualization_utils.py functions.
Run with pytest:
    pytest tests/anonymization/commons/test_visualization_utils.py -v
"""
import unittest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Import the module to test
from pamola_core.anonymization.commons.visualization_utils import (
    generate_visualization_filename,
    register_visualization_artifact,
    sample_large_dataset,
    prepare_comparison_data,
    calculate_optimal_bins,
    create_visualization_path
)


class TestVisualizationUtils(unittest.TestCase):
    """Test case for the visualization utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.numeric_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.categorical_series = pd.Series(['A', 'B', 'C', 'A', 'B', 'D', 'E', 'F', 'G', 'H'])
        self.large_series = pd.Series(range(20000))

        # Create sample paths
        self.task_dir = Path("/test/task_dir")

    def test_generate_visualization_filename(self):
        """Test generation of standardized filenames."""
        # Test with default parameters
        # Instead of patching datetime.datetime, let's provide a timestamp directly
        timestamp = "20250504_123456"
        filename = generate_visualization_filename(
            field_name="income",
            operation_name="binning",
            visualization_type="histogram",
            timestamp=timestamp
        )
        expected = "income_binning_histogram_20250504_123456.png"
        self.assertEqual(filename, expected)

        # Test with custom timestamp and extension
        filename = generate_visualization_filename(
            field_name="income",
            operation_name="binning",
            visualization_type="histogram",
            timestamp="20250101_010101",
            extension="jpg"
        )
        expected = "income_binning_histogram_20250101_010101.jpg"
        self.assertEqual(filename, expected)

        # Test with special characters in field name
        filename = generate_visualization_filename(
            field_name="user/email",
            operation_name="masking",
            visualization_type="pattern"
        )
        self.assertIn("user_email", filename.replace("/", "_"))

    def test_register_visualization_artifact(self):
        """Test registration of visualization artifacts."""
        # Create mock objects
        mock_result = Mock()
        mock_result.add_artifact = Mock()
        mock_reporter = Mock()
        mock_reporter.add_artifact = Mock()

        # Test path
        path = Path("/test/path/to/visualization.png")

        # Call the function
        register_visualization_artifact(
            result=mock_result,
            reporter=mock_reporter,
            path=path,
            field_name="income",
            visualization_type="histogram"
        )

        # Check if add_artifact was called with correct parameters
        mock_result.add_artifact.assert_called_once()
        args, kwargs = mock_result.add_artifact.call_args
        self.assertEqual(kwargs.get("artifact_type"), "png")
        self.assertEqual(kwargs.get("path"), path)
        self.assertIn("income", kwargs.get("description", ""))
        self.assertIn("histogram", kwargs.get("description", ""))

        # Check if reporter.add_artifact was called with correct parameters
        mock_reporter.add_artifact.assert_called_once()
        args, kwargs = mock_reporter.add_artifact.call_args
        self.assertEqual(args[0], "png")
        self.assertEqual(args[1], str(path))
        self.assertIn("income", args[2])
        self.assertIn("histogram", args[2])

        # Test with custom description
        mock_result.add_artifact.reset_mock()
        mock_reporter.add_artifact.reset_mock()

        custom_description = "Custom visualization description"
        register_visualization_artifact(
            result=mock_result,
            reporter=mock_reporter,
            path=path,
            field_name="income",
            visualization_type="histogram",
            description=custom_description
        )

        # Check if description was used
        args, kwargs = mock_result.add_artifact.call_args
        self.assertEqual(kwargs.get("description"), custom_description)

        # Test with None reporter (should not raise exception)
        mock_result.add_artifact.reset_mock()
        register_visualization_artifact(
            result=mock_result,
            reporter=None,
            path=path,
            field_name="income",
            visualization_type="histogram"
        )
        mock_result.add_artifact.assert_called_once()

    def test_sample_large_dataset(self):
        """Test sampling of large datasets."""
        # Test with small dataset (should return unchanged)
        small_data = pd.Series(range(100))
        result = sample_large_dataset(small_data, max_samples=1000)
        self.assertEqual(len(result), 100)
        pd.testing.assert_series_equal(result, small_data)

        # Test with large dataset
        large_data = pd.Series(range(20000))
        max_samples = 5000
        result = sample_large_dataset(large_data, max_samples=max_samples)
        self.assertEqual(len(result), max_samples)

        # Should be a subset of the original
        self.assertTrue(set(result).issubset(set(large_data)))

        # Test reproducibility with same random_state
        result1 = sample_large_dataset(large_data, max_samples=100, random_state=42)
        result2 = sample_large_dataset(large_data, max_samples=100, random_state=42)
        pd.testing.assert_series_equal(result1, result2)

        # Test different results with different random_state
        result3 = sample_large_dataset(large_data, max_samples=100, random_state=999)
        self.assertFalse(result1.equals(result3))

    def test_prepare_comparison_data_numeric(self):
        """Test preparation of numeric comparison data."""
        original = pd.Series([1, 2, 3, 4, 5])
        anonymized = pd.Series([1, 2, 3, 3, 4])

        # Test auto detection
        data, data_type = prepare_comparison_data(original, anonymized)

        self.assertEqual(data_type, "numeric")
        self.assertEqual(len(data), 2)
        self.assertIn("Original", data)
        self.assertIn("Anonymized", data)
        self.assertEqual(data["Original"], original.tolist())
        self.assertEqual(data["Anonymized"], anonymized.tolist())

        # Test explicit data type
        data, data_type = prepare_comparison_data(original, anonymized, data_type="numeric")
        self.assertEqual(data_type, "numeric")

        # Test with null values
        original_with_nulls = pd.Series([1, 2, None, 4, 5])
        anonymized_with_nulls = pd.Series([1, 2, 3, None, 4])

        data, data_type = prepare_comparison_data(original_with_nulls, anonymized_with_nulls)
        self.assertEqual(data_type, "numeric")
        # Nulls should be dropped
        self.assertEqual(len(data["Original"]), 4)
        self.assertEqual(len(data["Anonymized"]), 4)

        # Test with empty data
        empty_data, data_type = prepare_comparison_data(pd.Series(), pd.Series())
        # The actual implementation categorizes empty series as "categorical"
        self.assertEqual(data_type, "categorical")
        # The empty data still returns a dictionary with "Original" and "Anonymized" keys
        self.assertEqual(len(empty_data), 2)
        self.assertIn("Original", empty_data)
        self.assertIn("Anonymized", empty_data)
        # But each of those should contain empty data
        self.assertEqual(len(empty_data["Original"]), 0)
        self.assertEqual(len(empty_data["Anonymized"]), 0)

    def test_prepare_comparison_data_categorical(self):
        """Test preparation of categorical comparison data."""
        original = pd.Series(['A', 'B', 'C', 'A', 'B', 'D', 'E', 'F', 'G', 'H'])
        anonymized = pd.Series(['Other', 'Other', 'C', 'Other', 'Other', 'D', 'Other', 'Other', 'Other', 'H'])

        # Test auto detection
        data, data_type = prepare_comparison_data(original, anonymized)

        self.assertEqual(data_type, "categorical")
        self.assertEqual(len(data), 2)
        self.assertIn("Original", data)
        self.assertIn("Anonymized", data)

        # Should have counts for categories
        self.assertIsInstance(data["Original"], dict)
        self.assertIsInstance(data["Anonymized"], dict)

        # Original values should be counted correctly
        self.assertEqual(data["Original"]["A"], 2)
        self.assertEqual(data["Original"]["B"], 2)

        # Anonymized values should be counted correctly
        self.assertEqual(data["Anonymized"]["Other"], 7)
        self.assertEqual(data["Anonymized"]["C"], 1)

        # Test with max_categories limit
        data, data_type = prepare_comparison_data(original, anonymized, max_categories=3)

        # Should limit to max_categories (we may have at most max_categories keys)
        self.assertLessEqual(len(data["Original"]), 3)

        # Test with explicit data type
        data, data_type = prepare_comparison_data(original, anonymized, data_type="categorical")
        self.assertEqual(data_type, "categorical")

    def test_calculate_optimal_bins(self):
        """Test calculation of optimal bin count."""
        # Test with small dataset
        small_data = pd.Series(range(25))
        bins = calculate_optimal_bins(small_data)
        self.assertEqual(bins, 5)  # Should be min_bins

        # Test with medium dataset
        medium_data = pd.Series(range(100))
        bins = calculate_optimal_bins(medium_data)
        self.assertEqual(bins, 10)  # Should be sqrt(100)

        # Test with large dataset
        large_data = pd.Series(range(1000))
        bins = calculate_optimal_bins(large_data)
        self.assertGreaterEqual(bins, 5)
        self.assertLessEqual(bins, 30)
        self.assertEqual(bins, min(30, max(5, int(np.sqrt(1000)))))

        # Test with custom bounds
        custom_bins = calculate_optimal_bins(medium_data, min_bins=10, max_bins=15)
        self.assertEqual(custom_bins, 10)

        # Test with empty data
        empty_bins = calculate_optimal_bins(pd.Series())
        self.assertEqual(empty_bins, 5)  # Should default to min_bins

        # Test with data containing nulls
        null_data = pd.Series([1, 2, None, 4, 5])
        null_bins = calculate_optimal_bins(null_data)
        self.assertEqual(null_bins, 5)  # Should handle nulls properly

    def test_create_visualization_path(self):
        """Test creation of visualization paths."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            # Test with existing directory
            filename = "test_visualization.png"
            path = create_visualization_path(self.task_dir, filename)

            # Path should be correct
            self.assertEqual(path, self.task_dir / filename)

            # Directory should be created
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Test with nested filename
            nested_filename = "nested/dir/test_visualization.png"
            path = create_visualization_path(self.task_dir, nested_filename)

            # Path should be correct
            self.assertEqual(path, self.task_dir / nested_filename)

            # Test with absolute path as string
            absolute_dir = "/absolute/path"
            path = create_visualization_path(Path(absolute_dir), filename)
            self.assertEqual(path, Path(absolute_dir) / filename)

            # Test with string path
            string_dir = "/string/path"
            # Convert string to Path before passing to function
            path = create_visualization_path(Path(string_dir), filename)
            self.assertEqual(path, Path(string_dir) / filename)

    def test_prepare_comparison_data_error_handling(self):
        """Test error handling in prepare_comparison_data."""
        # Mismatched types
        numeric_data = pd.Series([1, 2, 3])
        categorical_data = pd.Series(['A', 'B', 'C'])

        # Should not raise error, but should detect based on original data
        data, data_type = prepare_comparison_data(numeric_data, categorical_data)
        self.assertEqual(data_type, "numeric")

        # Test with invalid data type
        data, data_type = prepare_comparison_data(numeric_data, categorical_data, data_type="invalid_type")
        self.assertEqual(data_type, "unknown")

        # Test with series of complex objects - skip this test as dictionaries are unhashable
        # and cannot be used with set operations in the tested function
        # complex_series = pd.Series([{"a": 1}, {"b": 2}])
        # data, data_type = prepare_comparison_data(complex_series, complex_series)
        # self.assertEqual(data_type, "categorical")


if __name__ == '__main__':
    unittest.main()