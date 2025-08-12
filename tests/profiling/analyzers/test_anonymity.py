import unittest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import tempfile
import pandas as pd
from pamola_core.profiling.analyzers.anonymity import KAnonymityProfilerOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error
        self.encryption_keys = {}
        self.encryption_modes = {}
    def get_dataframe(self, dataset_name, **kwargs):
        if self.df is not None:
            return self.df, None
        return None, {"message": self.error or "No data"}

class TestKAnonymityProfilingOperation(unittest.TestCase):
    def setUp(self):
        self.operation = KAnonymityProfilerOperation(
            name = "KAnonymityProfiler",
            description = "K-anonymity profiling and risk assessment",
            quasi_identifiers = None,
            mode = "ANALYZE",
            threshold_k = 5,
            generate_visualizations = True,
            export_metrics = True,
            visualization_theme = "professional",
            max_combinations = 50,
            chunk_size = 100000,
            output_field_suffix = "k_anon",
            quasi_identifier_sets = [
					[
						"Age"
					]
				],
            id_fields = ["ID"],
            use_dask = False,
            use_cache = False,
            use_vectorization = False,
            parallel_processes = 1,
            npartitions = 1,
            visualization_backend = None,
            visualization_strict = False,
            visualization_timeout = 120,
            use_encryption = False,
            encryption_key = None,
            encryption_mode = None
        )
        self.df = pd.DataFrame({
                'KAnonymityProfiler': ["Alice", "Bob", "Alice", "Bob", "Charlie"],
                'ID': [1, 2, 3, 4, 5],
                'Age': ["Alice", "Bob", "Alice", "Bob", "Charlie"]
            })
        self.mock_data_source = DummyDataSource(df=self.df)
        self.mock_task_dir = Path("test_task_dir")
        self.mock_reporter = MagicMock()
        self.mock_progress_tracker = MagicMock()
        self.task_dir = Path("test_task_dir")

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_success(self, mock_load_data_operation):
        # Create a realistic DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.mock_reporter.add_operation.assert_called_with(
            "K-Anonymity Profiling Completed",
            details={
                "mode": 'ANALYZE',
                "combinations_analyzed": 1,
                "threshold_k": 5
            }
        )

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_no_dataframe(self, mock_load_data_operation):
        # Mock no DataFrame returned
        mock_load_data_operation.return_value = None

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "No valid DataFrame found in data source")

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_exception_handling(self, mock_load_data_operation):
        # Mock exception during data loading
        mock_load_data_operation.side_effect = Exception("Test exception")

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("K-anonymity profiling failed", result.error_message)
        self.mock_reporter.add_operation.assert_called_with(
            "K-Anonymity Profiling",
            status="error",
            details={"error": "Test exception"}
        )

    def test_sorting_data(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]


        sorted_data = sorted(data, key=lambda x: x["age"])
        self.assertEqual(sorted_data, [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}])

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_with_progress_tracker(self, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        # Verify progress tracker updates
        self.mock_progress_tracker.update.assert_any_call(ANY, {"step": "Completed", "status": 'success'})

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_generate_field_combinations_when_none(self, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Do not pass fields_combinations (default value is None)
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        self.assertEqual(result.status, OperationStatus.SUCCESS)
    
    @patch.object(KAnonymityProfilerOperation, '_check_cache')    
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.anonymity.get_timestamped_filename", return_value="dummy.csv")
    @patch("pamola_core.profiling.analyzers.anonymity.logger")
    def test_execute_use_cache(
        self, mock_logger, mock_get_timestamped_filename, mock_load_data, mock_cache
    ):
        # Arrange
        df = self.df
        mock_load_data.return_value = df
        mock_cache.return_value = OperationResult(
                            status=OperationStatus.SUCCESS
                        )

        # Act
        self.operation.use_cache = True
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assert
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        
    @patch.object(KAnonymityProfilerOperation, '_check_cache')    
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.anonymity.get_timestamped_filename", return_value="dummy.csv")
    @patch("pamola_core.profiling.analyzers.anonymity.logger")
    def test_execute_use_cache_none_cache(
        self, mock_logger, mock_get_timestamped_filename, mock_load_data, mock_cache
    ):
        # Arrange
        df = self.df
        mock_load_data.return_value = df
        mock_cache.return_value = None

        # Act
        self.operation.use_cache = True
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assert
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        
    @patch.object(KAnonymityProfilerOperation, '_prepare_qi_combinations')    
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.anonymity.get_timestamped_filename", return_value="dummy.csv")
    @patch("pamola_core.profiling.analyzers.anonymity.logger")
    def test_execute_not_qi_combinations(
        self, mock_logger, mock_get_timestamped_filename, mock_load_data, mock_prepare_qi_combinations
    ):
        # Arrange
        df = self.df
        mock_load_data.return_value = df
        mock_prepare_qi_combinations.return_value = None

        # Act
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assert
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "No quasi-identifiers specified or detected")
        
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_with_ENRICH_mode(self, mock_load_data_operation):
        # Mock DataFrame
        mock_df = self.df
        mock_load_data_operation.return_value = mock_df

        # Execute the operation
        op = KAnonymityProfilerOperation(mode = "ENRICH",
                                         quasi_identifier_sets = [
					[
						"Age"
					]
				], id_fields = ["ID"], use_cache = False)
        result = op.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_with_BOTH_mode(self, mock_load_data_operation):
        # Mock DataFrame
        mock_df = self.df
        mock_load_data_operation.return_value = mock_df

        # Execute the operation
        op = KAnonymityProfilerOperation(mode = "BOTH",
                                         quasi_identifier_sets = [
					[
						"Age"
					]
				], id_fields = ["ID"], use_cache = False)
        result = op.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_exception_updates_progress_tracker(self, mock_load_data_operation):
        # Simulate exception when load_data_operation
        mock_load_data_operation.side_effect = Exception("Test error for progress tracker")

        mock_progress_tracker = MagicMock()

        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=mock_progress_tracker
        )

        # Check that progress_tracker.update is called with error info
        mock_progress_tracker.update.assert_any_call(
            0, {"step": "Error", "error": "Test error for progress tracker"}
        )
        self.assertEqual(result.status, OperationStatus.ERROR)
        
    @patch("pamola_core.profiling.analyzers.anonymity.ensure_directory")
    def test_prepare_directories(self, mock_ensure):
        dirs = self.operation._prepare_directories(self.task_dir)
        self.assertIn('output', dirs)
        self.assertIn('visualizations', dirs)
        self.assertIn('dictionaries', dirs)
        self.assertTrue(str(dirs['output']).endswith('output'))
        self.assertTrue(str(dirs['visualizations']).endswith('visualizations'))
        self.assertTrue(str(dirs['dictionaries']).endswith('dictionaries'))
        self.assertEqual(mock_ensure.call_count, 3)

    def test_detect_quasi_identifiers_basic(self):
        # Test with categorical and numeric columns
        df = pd.DataFrame({
            'cat1': ['a', 'b', 'a', 'c'],
            'cat2': ['x', 'y', 'x', 'z'],
            'num1': [1, 2, 1, 2],
            'num2': [10.0, 20.0, 10.0, 20.0],
            'id': [100, 101, 102, 103]
        })
        op = KAnonymityProfilerOperation(max_combinations=10)
        # Patch get_field_statistics to control dtype/unique_count
        with patch('pamola_core.profiling.analyzers.anonymity.get_field_statistics') as mock_stats:
            def fake_stats(series):
                if series.name.startswith('cat'):
                    return {'dtype': 'object', 'unique_count': series.nunique()}
                if series.name == 'num1':
                    return {'dtype': 'int64', 'unique_count': series.nunique()}
                if series.name == 'num2':
                    return {'dtype': 'float64', 'unique_count': series.nunique()}
                return {'dtype': 'int64', 'unique_count': series.nunique()}
            mock_stats.side_effect = fake_stats
            combos = op._detect_quasi_identifiers(df, exclude_fields=['id'])
        # Should generate all 2- and 3-combinations of cat1, cat2, num1, num2
        expected_cols = {'cat1', 'cat2', 'num1', 'num2'}
        for combo in combos:
            self.assertTrue(set(combo).issubset(expected_cols))
            self.assertTrue(2 <= len(combo) <= 3)
        self.assertTrue(len(combos) > 0)

    def test_detect_quasi_identifiers_exclude_fields(self):
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        op = KAnonymityProfilerOperation(max_combinations=5)
        with patch('pamola_core.profiling.analyzers.anonymity.get_field_statistics', return_value={'dtype': 'int64', 'unique_count': 3}):
            combos = op._detect_quasi_identifiers(df, exclude_fields=['b'])
        # Only 'a' and 'c' should be considered
        for combo in combos:
            self.assertNotIn('b', combo)

    def test_detect_quasi_identifiers_cardinality_limit(self):
        df = pd.DataFrame({
            'low_card': [1, 1, 2, 2],
            'high_card': list(range(4)),
            'cat': ['a', 'b', 'a', 'c']
        })
        op = KAnonymityProfilerOperation(max_combinations=10)
        def fake_stats(series):
            if series.name == 'low_card':
                return {'dtype': 'int64', 'unique_count': 2}
            if series.name == 'high_card':
                return {'dtype': 'int64', 'unique_count': 1000}
            if series.name == 'cat':
                return {'dtype': 'object', 'unique_count': 3}
        with patch('pamola_core.profiling.analyzers.anonymity.get_field_statistics', side_effect=fake_stats):
            combos = op._detect_quasi_identifiers(df, exclude_fields=[])
        # 'high_card' should not be included in any combo
        for combo in combos:
            self.assertNotIn('high_card', combo)

    def test_detect_quasi_identifiers_no_valid_fields(self):
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'high_card': list(range(3))
        })
        op = KAnonymityProfilerOperation(max_combinations=5)
        def fake_stats(series):
            return {'dtype': 'int64', 'unique_count': 1000}
        with patch('pamola_core.profiling.analyzers.anonymity.get_field_statistics', side_effect=fake_stats):
            combos = op._detect_quasi_identifiers(df, exclude_fields=['id'])
        self.assertEqual(combos, [])

    def test_calculate_group_sizes_small(self):
        df = pd.DataFrame({
            'A': ['x', 'x', 'y', 'y', 'y'],
            'B': [1, 1, 2, 2, 2]
        })
        op = KAnonymityProfilerOperation()
        k_values = op._calculate_group_sizes(df, ['A', 'B'])
        self.assertTrue((k_values == [1, 1]).sum() == 2)  # No group size 1

    @patch('pamola_core.profiling.analyzers.anonymity.get_dataframe_chunks')
    def test_calculate_group_sizes_large_manual_chunk(self, mock_chunk):
        with patch('pandas.Series.__len__', return_value=10000):
            df = pd.DataFrame({
                'A': ['a'] * 5000 + ['b'] * 5000,
                'B': [1] * 2500 + [2] * 2500 + [1] * 2500 + [2] * 2500
            })
            op = KAnonymityProfilerOperation(chunk_size=1000)
            k_values = op._calculate_group_sizes(df, ['A', 'B'], chunk_size=1000)
            mock_chunk.assert_called()

    def test_calculate_group_sizes_with_nan(self):
        df = pd.DataFrame({
            'A': ['x', 'x', None, None],
            'B': [1, 1, 2, 2]
        })
        op = KAnonymityProfilerOperation()
        k_values = op._calculate_group_sizes(df, ['A', 'B'])
        # Should handle None/NaN as a group
        self.assertIn(2, k_values)

    def test_calculate_group_sizes_vectorization(self):
        df = pd.DataFrame({
            'A': ['x', 'x', 'y', 'y', 'y'],
            'B': [1, 1, 2, 2, 2]
        })
        op = KAnonymityProfilerOperation(chunk_size=40)
        # Patch Parallel and get_dataframe_chunks to simulate vectorization
        with patch('pamola_core.profiling.analyzers.anonymity.Parallel') as mock_parallel, \
             patch('pamola_core.profiling.analyzers.anonymity.get_dataframe_chunks') as mock_get_dataframe_chunks:
             k_values = op._calculate_group_sizes(df, ['A', 'B'], use_vectorization=True, parallel_processes=2)
             mock_parallel.assert_called()
             mock_get_dataframe_chunks.assert_called()
             
    def test_calculate_group_sizes_use_dask(self):
        df = pd.DataFrame({
            'A': ['x', 'x', 'y', 'y', 'y'],
            'B': [1, 1, 2, 2, 2]
        })
        op = KAnonymityProfilerOperation(chunk_size=40)
        with patch('dask.dataframe.from_pandas') as mock_from_pandas:
            k_values = op._calculate_group_sizes(df, ['A', 'B'], use_dask=True)
            mock_from_pandas.assert_called()

    def test_calculate_group_sizes_empty(self):
        df = pd.DataFrame({'A': [], 'B': []})
        op = KAnonymityProfilerOperation()
        k_values = op._calculate_group_sizes(df, ['A', 'B'])
        self.assertEqual(len(k_values), 0)

    def test_calculate_k_anonymity_chunked_basic(self):
        df = pd.DataFrame({
            'A': ['x', 'x', 'y', 'y', 'y'],
            'B': [1, 1, 2, 2, 2]
        })
        op = KAnonymityProfilerOperation(chunk_size=2)
        result = op._calculate_k_anonymity_chunked(df, ['A', 'B'])
        self.assertIn('min_k', result)
        self.assertIn('mean_k', result)
        self.assertIn('k_distribution', result)
        self.assertEqual(result['min_k'], 2)
        self.assertEqual(result['mean_k'], 2.5)

    def test_calculate_k_anonymity_chunked_large(self):
        df = pd.DataFrame({
            'A': ['a'] * 5000 + ['b'] * 5000,
            'B': [1] * 2500 + [2] * 2500 + [1] * 2500 + [2] * 2500
        })
        op = KAnonymityProfilerOperation(chunk_size=1000)
        result = op._calculate_k_anonymity_chunked(df, ['A', 'B'])
        self.assertEqual(result['min_k'], 2500)
        self.assertEqual(result['mean_k'], 2500)

    def test_calculate_k_anonymity_chunked_with_nan(self):
        df = pd.DataFrame({
            'A': ['x', 'x', None, None],
            'B': [1, 1, 2, 2]
        })
        op = KAnonymityProfilerOperation(chunk_size=2)
        result = op._calculate_k_anonymity_chunked(df, ['A', 'B'])
        self.assertGreaterEqual(result['min_k'], 2)
        self.assertIn('k_distribution', result)

    def test_calculate_k_anonymity_chunked_empty(self):
        df = pd.DataFrame({'A': [], 'B': []})
        op = KAnonymityProfilerOperation(chunk_size=2)
        result = op._calculate_k_anonymity_chunked(df, ['A', 'B'])
        self.assertEqual(result['min_k'], 0)
        self.assertEqual(result['mean_k'], 0)
        self.assertEqual(result['k_distribution'], {})

    def test_calculate_k_values_chunked_basic(self):
        df = pd.DataFrame({
            'A': ['x', 'x', 'y', 'y', 'y'],
            'B': [1, 1, 2, 2, 2]
        })
        op = KAnonymityProfilerOperation(chunk_size=2)
        k_values = op._calculate_k_values_chunked(df, ['A', 'B'])
        self.assertTrue((k_values == 2).sum() == 2)
        self.assertTrue((k_values == 3).sum() == 3)
        self.assertEqual(len(k_values), 5)

    def test_calculate_k_values_chunked_large(self):
        df = pd.DataFrame({
            'A': ['a'] * 5000 + ['b'] * 5000,
            'B': [1] * 2500 + [2] * 2500 + [1] * 2500 + [2] * 2500
        })
        op = KAnonymityProfilerOperation(chunk_size=1000)
        k_values = op._calculate_k_values_chunked(df, ['A', 'B'])
        self.assertTrue(all(k == 2500 for k in k_values))
        self.assertEqual(len(k_values), 4*2500)

    def test_calculate_k_values_chunked_with_nan(self):
        df = pd.DataFrame({
            'A': ['x', 'x', None, None],
            'B': [1, 1, 2, 2]
        })
        op = KAnonymityProfilerOperation(chunk_size=2)
        k_values = op._calculate_k_values_chunked(df, ['A', 'B'])
        self.assertTrue(all(k == 2 for k in k_values))
        self.assertEqual(len(k_values), 4)

    def test_calculate_k_values_chunked_empty(self):
        df = pd.DataFrame({'A': [], 'B': []})
        op = KAnonymityProfilerOperation(chunk_size=2)
        k_values = op._calculate_k_values_chunked(df, ['A', 'B'])
        self.assertEqual(len(k_values), 0)
        
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")    
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(KAnonymityProfilerOperation, '_generate_cache_key')
    def test_no_cache(self, mock_cache_key, mock_operation_cache, mock_load_data_operation):
        mock_load_data_operation.return_value = self.df
        mock_cache_key.return_value = 'cache_key'
        out = self.operation._check_cache(
            self.mock_data_source
        )
        self.assertEqual(out, None)
        
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")    
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(KAnonymityProfilerOperation, '_generate_cache_key')
    def test_cache(self, mock_cache_key, mock_operation_cache, mock_get_cache, mock_load_data_operation):
        mock_load_data_operation.return_value = self.df
        mock_cache_key.return_value = 'cache_key'
        mock_operation_cache.get_cache.return_value = {
            'artifacts': [],
            'analysis_results': {}
        }
        mock_operation_cache.return_value = {'main': 'vis_path.png'}
        self.operation.use_cache = True
        out = self.operation._check_cache(self.mock_data_source)
        self.assertEqual(out.status, OperationStatus.SUCCESS)
    
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(KAnonymityProfilerOperation, '_generate_cache_key')
    def test_cache_metrics(self, mock_cache_key, mock_operation_cache, mock_get_cache,
                                               mock_load_data_operation):
        mock_load_data_operation.return_value = self.df
        mock_cache_key.return_value = 'cache_key'
        metric_path = 'metric.json'
        mock_operation_cache.get_cache.return_value = {
            'metrics': {
                'path': metric_path
            }
        }
        self.operation.use_cache = True
        out = self.operation._check_cache(self.mock_data_source)
        self.assertEqual(out.status, OperationStatus.SUCCESS)
        self.assertEqual(out.metrics['path'], metric_path)
        
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(KAnonymityProfilerOperation, '_generate_cache_key')
    def test_cache_artifact(self, mock_cache_key, mock_operation_cache, mock_get_cache,
                                               mock_load_data_operation):
        mock_load_data_operation.return_value = self.df
        mock_cache_key.return_value = 'cache_key'
        metric_path = 'metric.json'
        mock_operation_cache.get_cache.return_value = {
            'artifacts': [{
                'type': 'json',
                'path': metric_path,
                'description': 'description artifact',
                'category': 'output'
            }]
        }
        self.operation.use_cache = True
        out = self.operation._check_cache(self.mock_data_source)
        self.assertEqual(out.status, OperationStatus.SUCCESS)
        self.assertEqual(str(out.artifacts[0].path), metric_path)
                        
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(KAnonymityProfilerOperation, '_generate_cache_key')
    def test_cache_exception(self, mock_cache_key, mock_operation_cache, mock_get_cache, mock_load_data_operation):
        mock_load_data_operation.return_value = self.df
        mock_cache_key.return_value = 'cache_key'
        mock_operation_cache.get_cache.side_effect = Exception("Cache Exception")
        self.operation.use_cache = True
        out = self.operation._check_cache(self.mock_data_source)
        self.assertEqual(out, None)
        
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(KAnonymityProfilerOperation, '_generate_cache_key')
    def test_cache_df_none(self, mock_cache_key, mock_operation_cache, mock_get_cache, mock_load_data_operation):
        mock_load_data_operation.return_value = None
        mock_cache_key.return_value = 'cache_key'
        self.operation.use_cache = True
        out = self.operation._check_cache(self.mock_data_source)
        self.assertEqual(out, None)


if __name__ == "__main__":
    unittest.main()