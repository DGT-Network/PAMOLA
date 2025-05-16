import unittest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import numpy as np
import pandas as pd
from pamola_core.fake_data.commons.metrics import MetricsCollector, create_metrics_collector, generate_metrics_report


class TestCollectMetrics(unittest.TestCase):
    def setUp(self):
        self.collector = MetricsCollector()
        # Mock các method nội bộ
        self.collector.collect_data_stats = Mock(side_effect=lambda x: {"count": len(x)})
        self.collector.collect_transformation_metrics = Mock(return_value={"metric1": 1.0})

    def test_collect_metrics_only_orig_data(self):
        orig = pd.Series([1, 2, 3])
        result = self.collector.collect_metrics(orig_data=orig)
        self.collector.collect_data_stats.assert_called_once_with(orig)
        self.collector.collect_transformation_metrics.assert_not_called()
        self.assertIn("original_data", result)
        self.assertNotIn("generated_data", result)
        self.assertNotIn("transformation_metrics", result)
        self.assertEqual(result["original_data"], {"count": 3})
        self.assertEqual(result["performance"], {})
        self.assertEqual(result["dictionary_metrics"], {})

    def test_collect_metrics_only_gen_data(self):
        gen = pd.Series([4, 5])
        result = self.collector.collect_metrics(gen_data=gen)
        self.collector.collect_data_stats.assert_called_once_with(gen)
        self.collector.collect_transformation_metrics.assert_not_called()
        self.assertNotIn("original_data", result)
        self.assertIn("generated_data", result)
        self.assertNotIn("transformation_metrics", result)
        self.assertEqual(result["generated_data"], {"count": 2})

    def test_collect_metrics_both_data(self):
        orig = pd.Series([1, 2])
        gen = pd.Series([3, 4, 5])
        params = {"param1": "value1"}

        result = self.collector.collect_metrics(orig_data=orig, gen_data=gen, operation_params=params)

        calls = self.collector.collect_data_stats.call_args_list
        # Lấy tất cả đối số gọi của collect_data_stats
        called_args = [call.args[0] for call in calls]

        def series_equal(a, b):
            return len(a) == len(b) and (a.values == b.values).all()

        # So sánh giá trị list để tránh lỗi so sánh Series có độ dài khác nhau
        self.assertTrue(any(series_equal(arg, orig) for arg in called_args))
        self.assertTrue(any(series_equal(arg, gen) for arg in called_args))

        self.collector.collect_transformation_metrics.assert_called_once_with(orig, gen, params)
        self.assertIn("original_data", result)
        self.assertIn("generated_data", result)
        self.assertIn("transformation_metrics", result)
        self.assertEqual(result["performance"], {})
        self.assertEqual(result["dictionary_metrics"], {})

    def test_collect_metrics_no_data(self):
        result = self.collector.collect_metrics()
        self.collector.collect_data_stats.assert_not_called()
        self.collector.collect_transformation_metrics.assert_not_called()
        self.assertNotIn("original_data", result)
        self.assertNotIn("generated_data", result)
        self.assertNotIn("transformation_metrics", result)
        self.assertEqual(result["performance"], {})
        self.assertEqual(result["dictionary_metrics"], {})


class TestCollectDataStats(unittest.TestCase):
    def setUp(self):
        self.collector = MetricsCollector()

    def test_collect_data_stats_numeric(self):
        data = pd.Series([1, 2, 2, 3, 4, 4, 4, None])
        result = self.collector.collect_data_stats(data)

        self.assertEqual(result["total_records"], 8)
        self.assertEqual(result["unique_values"], 4)  # 1,2,3,4
        expected_dist = {
            4: 3/7,
            2: 2/7,
            1: 1/7,
            3: 1/7
        }
        for key, val in expected_dist.items():
            self.assertAlmostEqual(result["value_distribution"][key], val)
        self.assertEqual(result["length_stats"], {})  # numeric data ko tính length_stats

    def test_collect_data_stats_string(self):
        data = pd.Series(["apple", "banana", "apple", None, "cherry", "date", "date"])
        result = self.collector.collect_data_stats(data)

        self.assertEqual(result["total_records"], 7)
        self.assertEqual(result["unique_values"], 4)  # apple, banana, cherry, date

        expected_dist = {
            "apple": 2 / 6,
            "date": 2 / 6,
            "banana": 1 / 6,
            "cherry": 1 / 6
        }
        for key, val in expected_dist.items():
            self.assertAlmostEqual(result["value_distribution"][key], val)

        length_stats = result["length_stats"]
        lengths = [5, 6, 5, 6, 4, 4]  # độ dài các chuỗi không null
        self.assertEqual(length_stats["min"], min(lengths))
        self.assertEqual(length_stats["max"], max(lengths))
        self.assertAlmostEqual(length_stats["mean"], sum(lengths) / len(lengths))
        self.assertAlmostEqual(length_stats["median"], np.median(lengths))

    def test_collect_data_stats_with_all_null(self):
        data = pd.Series([None, None])
        result = self.collector.collect_data_stats(data)

        self.assertEqual(result["total_records"], 2)
        self.assertEqual(result["unique_values"], 0)
        self.assertEqual(result["value_distribution"], {})
        self.assertIn("error", result["length_stats"])
        self.assertIsInstance(result["length_stats"]["error"], str)


class FakeMappingStore:
    def __init__(self, collisions=0):
        self._collisions = collisions
    def get_collision_count(self, field_name):
        return self._collisions


class TestCollectTransformationMetrics(unittest.TestCase):

    def test_basic_no_nulls_no_changes(self):
        orig = pd.Series([1, 2, 3])
        gen = pd.Series([1, 2, 3])
        params = {}
        result = MetricsCollector.collect_transformation_metrics(orig, gen, params)

        self.assertEqual(result["null_values_replaced"], 0)
        self.assertEqual(result["total_replacements"], 0)
        self.assertEqual(result["replacement_strategy"], "unknown")
        self.assertEqual(result["mapping_collisions"], 0)
        self.assertAlmostEqual(result["reversibility_rate"], 1.0)  # all unique, each appears once

    def test_nulls_replaced_and_value_changes(self):
        orig = pd.Series([1, None, 3, None])
        gen = pd.Series([2, 5, 3, 7])
        params = {"consistency_mechanism": "custom"}
        result = MetricsCollector.collect_transformation_metrics(orig, gen, params)

        self.assertEqual(result["null_values_replaced"], 0)
        self.assertEqual(result["total_replacements"], 1)
        self.assertEqual(result["replacement_strategy"], "custom")
        self.assertEqual(result["mapping_collisions"], 0)
        self.assertAlmostEqual(result["reversibility_rate"], 1.0)

    def test_with_mapping_collisions(self):
        orig = pd.Series([1, 2, 3])
        gen = pd.Series([4, 5, 6])
        mapping_store = FakeMappingStore(collisions=2)
        params = {
            "consistency_mechanism": "mapping",
            "mapping_store": mapping_store,
            "field_name": "field1"
        }
        result = MetricsCollector.collect_transformation_metrics(orig, gen, params)

        self.assertEqual(result["replacement_strategy"], "mapping")
        self.assertEqual(result["mapping_collisions"], 2)
        self.assertEqual(result["null_values_replaced"], 0)
        self.assertEqual(result["total_replacements"], 3)
        self.assertAlmostEqual(result["reversibility_rate"], 1.0)

    def test_reversibility_rate_with_duplicates_and_nulls(self):
        orig = pd.Series([1, 2, 2, None])
        gen = pd.Series([2, 2, 3, 4])
        params = {}
        result = MetricsCollector.collect_transformation_metrics(orig, gen, params)

        self.assertEqual(result["null_values_replaced"], 0)
        self.assertEqual(result["total_replacements"], 2)
        self.assertAlmostEqual(result["reversibility_rate"], 2/4)

    def test_fallback_on_exception(self):
        result = MetricsCollector.collect_transformation_metrics(None, None, None)
        self.assertEqual(result["null_values_replaced"], 0)
        self.assertEqual(result["total_replacements"], 0)
        self.assertEqual(result["replacement_strategy"], "unknown")
        self.assertEqual(result["mapping_collisions"], 0)
        self.assertEqual(result["reversibility_rate"], 0.0)


class TestCompareDistributions(unittest.TestCase):
    def setUp(self):
        self.metrics_collector = MetricsCollector()

    def test_identical_distributions(self):
        data = pd.Series([1, 2, 3, 4, 5])
        result = self.metrics_collector.compare_distributions(data, data)
        self.assertAlmostEqual(result["distribution_similarity_score"], 1.0)
        self.assertAlmostEqual(result["uniqueness_preservation"], 1.0)

    def test_different_distributions(self):
        orig = pd.Series([1, 1, 2, 2, 3, 3])
        gen = pd.Series([4, 4, 5, 5, 6, 6])
        result = self.metrics_collector.compare_distributions(orig, gen)
        self.assertTrue(0 <= result["distribution_similarity_score"] < 1)
        self.assertTrue(result["uniqueness_preservation"] > 0)

    def test_partial_overlap(self):
        orig = pd.Series([1, 1, 2, 3])
        gen = pd.Series([2, 2, 3, 4])
        result = self.metrics_collector.compare_distributions(orig, gen)
        self.assertTrue(0 <= result["distribution_similarity_score"] <= 1)
        self.assertTrue(result["uniqueness_preservation"] > 0)

    def test_empty_series(self):
        orig = pd.Series([])
        gen = pd.Series([])
        result = self.metrics_collector.compare_distributions(orig, gen)
        self.assertEqual(result["distribution_similarity_score"], 1.0)
        self.assertEqual(result["uniqueness_preservation"], 0.0)

    def test_with_null_values(self):
        orig = pd.Series([1, 2, None, 2, None])
        gen = pd.Series([1, None, None, 2, 3])
        result = self.metrics_collector.compare_distributions(orig, gen)
        self.assertTrue(0 <= result["distribution_similarity_score"] <= 1)
        self.assertTrue(result["uniqueness_preservation"] >= 0)


class TestVisualizeMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics_collector = MetricsCollector()
        self.output_dir = Path("/tmp/test_output")
        self.field_name = "test_field"
        self.op_type = "test_op"

    @patch("pamola_core.fake_data.commons.metrics.create_combined_chart")
    @patch("pamola_core.fake_data.commons.metrics.create_bar_plot")
    @patch("pamola_core.fake_data.commons.metrics.create_pie_chart")
    def test_visualize_metrics_all_visualizations(self, mock_pie, mock_bar, mock_combined):
        # Setup mocks to return a fake path string
        mock_combined.return_value = "/tmp/test_output/combined.png"
        mock_bar.return_value = "/tmp/test_output/bar.png"
        mock_pie.return_value = "/tmp/test_output/pie.png"

        metrics = {
            "original_data": {
                "value_distribution": {"a": 0.4, "b": 0.6},
                "length_stats": {"min": 1, "max": 5, "mean": 3, "median": 3},
                "total_records": 10
            },
            "generated_data": {
                "value_distribution": {"a": 0.3, "b": 0.7},
                "length_stats": {"min": 1, "max": 6, "mean": 3.5, "median": 4}
            },
            "transformation_metrics": {
                "total_replacements": 3
            }
        }

        result = self.metrics_collector.visualize_metrics(metrics, self.field_name, self.output_dir, self.op_type)

        # Check calls to visualization functions
        mock_combined.assert_called_once()
        mock_bar.assert_called_once()
        mock_pie.assert_called_once()

        # Check returned keys and paths
        self.assertIn("value_distribution", result)
        self.assertIn("length_stats", result)
        self.assertIn("replacement_rate", result)

        self.assertEqual(result["value_distribution"], Path(mock_combined.return_value))
        self.assertEqual(result["length_stats"], Path(mock_bar.return_value))
        self.assertEqual(result["replacement_rate"], Path(mock_pie.return_value))

    @patch("pamola_core.utils.visualization.create_combined_chart")
    def test_visualize_metrics_no_value_distribution(self, mock_combined):
        metrics = {
            "original_data": {},
            "generated_data": {},
        }
        result = self.metrics_collector.visualize_metrics(metrics, self.field_name, self.output_dir, self.op_type)
        self.assertNotIn("value_distribution", result)
        mock_combined.assert_not_called()

    @patch("pamola_core.utils.visualization.create_bar_plot")
    def test_visualize_metrics_no_length_stats(self, mock_bar):
        metrics = {
            "original_data": {"value_distribution": {"a": 0.5}},
            "generated_data": {"value_distribution": {"a": 0.5}},
        }
        result = self.metrics_collector.visualize_metrics(metrics, self.field_name, self.output_dir, self.op_type)
        self.assertNotIn("length_stats", result)
        mock_bar.assert_not_called()

    @patch("pamola_core.utils.visualization.create_pie_chart")
    def test_visualize_metrics_no_transformation_metrics(self, mock_pie):
        metrics = {
            "original_data": {"total_records": 10},
            "generated_data": {},
        }
        result = self.metrics_collector.visualize_metrics(metrics, self.field_name, self.output_dir, self.op_type)
        self.assertNotIn("replacement_rate", result)
        mock_pie.assert_not_called()


class TestCreateMetricsCollector(unittest.TestCase):

    def test_create_metrics_collector_returns_instance(self):
        collector = create_metrics_collector()
        self.assertIsNotNone(collector, "Returned collector is None")
        self.assertIsInstance(collector, MetricsCollector, "Returned object is not a MetricsCollector instance")


class TestGenerateMetricsReport(unittest.TestCase):

    def setUp(self):
        self.metrics = {
            "original_data": {
                "total_records": 100,
                "unique_values": 80,
                "length_stats": {"min": 1, "max": 10, "mean": 5.5, "median": 5}
            },
            "generated_data": {
                "total_records": 100,
                "unique_values": 85,
                "length_stats": {"min": 2, "max": 9, "mean": 5.3, "median": 5}
            },
            "transformation_metrics": {
                "replacement_strategy": "simple",
                "total_replacements": 50,
                "null_values_replaced": 10,
                "mapping_collisions": 2,
                "reversibility_rate": 0.95
            },
            "performance": {
                "generation_time": 2.5,
                "records_per_second": 40,
                "memory_usage_mb": 123.45
            },
            "dictionary_metrics": {
                "total_dictionary_entries": 500,
                "language_variants": ["en", "fr"],
                "last_update": "2024-12-01"
            }
        }
        self.output_dir = Path("/tmp/reports")
        self.file_path = self.output_dir / "synthesis_testfield_metrics_report.md"

    def test_generate_metrics_report_returns_string(self):
        report = generate_metrics_report(self.metrics)
        self.assertIsInstance(report, str)
        self.assertIn("# Fake Data Generation Metrics Report", report)
        self.assertIn("## Summary", report)
        self.assertIn("### Original Data", report)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.is_dir", return_value=True)  # patch method is_dir trên Path class
    def test_generate_metrics_report_writes_file(self, mock_is_dir, mock_mkdir, mock_open_func):
        result = generate_metrics_report(
            self.metrics,
            output_path=self.output_dir,
            op_type="synthesis",
            field_name="testfield"
        )

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open_func.assert_called_once_with(self.file_path, "w")
        handle = mock_open_func()
        self.assertIn("Fake Data Generation Metrics Report", handle.write.call_args[0][0])

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_metrics_report_custom_file_path(self, mock_mkdir, mock_open_func):
        # Test trường hợp truyền output_path là 1 file cụ thể
        custom_path = Path("/tmp/reports/custom_report.md")
        result = generate_metrics_report(self.metrics, output_path=custom_path)

        mock_open_func.assert_called_once_with(custom_path, "w")
        handle = mock_open_func()
        self.assertIn("Dictionary", handle.write.call_args[0][0])


if __name__ == "__main__":
    unittest.main()