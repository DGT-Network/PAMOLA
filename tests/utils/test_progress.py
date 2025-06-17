"""
Unit tests for the progress tracking module.

These tests cover the pamola core functionality of the progress tracking utilities,
including simple progress bars, hierarchical trackers, data processing functions,
and error handling mechanisms.
"""

import unittest
import pandas as pd
import numpy as np
import io
import sys
import time
import tempfile
import logging
import warnings
from unittest.mock import patch, MagicMock, call, ANY
from contextlib import contextmanager, ExitStack

from pamola_core.utils.progress import (
    SimpleProgressBar,
    HierarchicalProgressTracker,
    track_operation_safely,
    process_dataframe_in_chunks_enhanced,
    iterate_dataframe_chunks_enhanced,
    process_dataframe_in_parallel_enhanced,
    multi_stage_process,
    configure_logging,
    get_logger,
    # Legacy classes
    ProgressBar,
    ProgressTracker,
    track_operation,
    process_dataframe_in_chunks,
    iterate_dataframe_chunks,
    process_dataframe_in_parallel
)


@contextmanager
def captured_output():
    """Context manager to capture stdout and stderr"""
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class FileClosingTestCase(unittest.TestCase):
    """Base class to properly close file handlers created during tests."""

    def setUp(self):
        super().setUp()
        # Keep track of all loggers created
        self.loggers = []

    def tearDown(self):
        # Close all file handlers
        for logger in self.loggers:
            if isinstance(logger, logging.Logger):
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.close()
        super().tearDown()


class TestLoggingConfiguration(FileClosingTestCase):
    """Test the logging configuration functionality."""

    def setUp(self):
        super().setUp()
        # Reset logger before each test
        from pamola_core.utils import progress
        if hasattr(progress, '_logger'):
            # Close any existing handlers first
            if progress._logger:
                for handler in progress._logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.close()
            # Reset logger
            progress._logger = None

    def test_configure_logging_default(self):
        """Test default logging configuration."""
        with tempfile.NamedTemporaryFile(suffix='.log') as temp_log:
            logger = configure_logging(log_file=temp_log.name)
            self.loggers.append(logger)

            self.assertIsInstance(logger, logging.Logger)
            self.assertEqual(logger.level, logging.INFO)
            self.assertTrue(len(logger.handlers) >= 1)  # At least console handler

    def test_configure_logging_custom(self):
        """Test custom logging configuration."""
        # Create temp file for log
        with tempfile.NamedTemporaryFile(suffix='.log') as temp_log:
            logger = configure_logging(
                level=logging.DEBUG,
                format_str="%(levelname)s: %(message)s",
                log_file=temp_log.name
            )
            self.loggers.append(logger)

            self.assertIsInstance(logger, logging.Logger)
            self.assertEqual(logger.level, logging.DEBUG)

            # Check format
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    self.assertEqual(
                        handler.formatter._fmt,
                        "%(levelname)s: %(message)s"
                    )

    def test_get_logger(self):
        """Test get_logger initializes logger if needed."""
        with tempfile.NamedTemporaryFile(suffix='.log') as temp_log:
            # Configure with temp file
            configure_logging(log_file=temp_log.name)

            logger1 = get_logger()
            logger2 = get_logger()
            self.loggers.append(logger1)

            self.assertIsInstance(logger1, logging.Logger)
            self.assertIs(logger1, logger2)  # Same instance


class TestSimpleProgressBar(unittest.TestCase):
    """Test the SimpleProgressBar class."""

    def test_init(self):
        """Test initialization with different parameters."""
        with patch('pamola_core.utils.progress.tqdm') as mock_tqdm:
            bar = SimpleProgressBar(total=100, description="Test", unit="items")
            mock_tqdm.assert_called_once()
            self.assertEqual(bar.total, 100)
            self.assertEqual(bar.description, "Test")
            self.assertEqual(bar.unit, "items")

    def test_update(self):
        """Test update method."""
        with patch('pamola_core.utils.progress.tqdm') as mock_tqdm:
            mock_instance = mock_tqdm.return_value
            bar = SimpleProgressBar(total=100)
            bar.update(5)
            mock_instance.update.assert_called_once_with(5)

    def test_context_manager(self):
        """Test using as context manager."""
        with patch('pamola_core.utils.progress.tqdm') as mock_tqdm:
            mock_instance = mock_tqdm.return_value
            with SimpleProgressBar(total=100) as bar:
                bar.update(10)
            mock_instance.update.assert_called_once_with(10)
            mock_instance.close.assert_called_once()


class TestHierarchicalProgressTracker(unittest.TestCase):
    """Test the HierarchicalProgressTracker class."""

    def test_init(self):
        """Test initialization."""
        with patch('pamola_core.utils.progress.tqdm') as mock_tqdm:
            tracker = HierarchicalProgressTracker(
                total=100,
                description="Test",
                unit="items",
                level=2,
                track_memory=True
            )
            # Called twice: once in base init, once in hierarchical override
            self.assertEqual(mock_tqdm.call_count, 2)
            self.assertEqual(tracker.level, 2)
            self.assertTrue(tracker.track_memory)

    def test_create_subtask(self):
        """Test creating a subtask."""
        with patch('pamola_core.utils.progress.tqdm'):
            parent = HierarchicalProgressTracker(total=5, description="Parent")
            subtask = parent.create_subtask(total=10, description="Child")

            self.assertEqual(len(parent.children), 1)
            self.assertIs(parent.children[0], subtask)
            self.assertEqual(subtask.level, parent.level + 1)
            self.assertIs(subtask.parent, parent)

    def test_close_with_children(self):
        """Test closing parent closes children."""
        # Простой интеграционный тест без моков
        with captured_output():
            # Создаем реальные объекты без моков
            parent = HierarchicalProgressTracker(total=3, description="Parent")
            child1 = parent.create_subtask(total=10, description="Child1")
            child2 = parent.create_subtask(total=10, description="Child2")

            # Патчим методы close, но не меняем их поведение
            original_close1 = child1.close
            original_close2 = child2.close

            calls1 = []
            calls2 = []

            def mock_close1():
                calls1.append(1)
                return original_close1()

            def mock_close2():
                calls2.append(1)
                return original_close2()

            # Устанавливаем mock методы
            child1.close = mock_close1
            child2.close = mock_close2

            # Закрываем родителя
            parent.close()

            # Проверяем, что дочерние close методы были вызваны
            self.assertEqual(len(calls1), 1, "Child1.close не был вызван")
            self.assertEqual(len(calls2), 1, "Child2.close не был вызван")

    def test_child_updates_parent(self):
        """Test completing subtask updates parent progress."""
        with patch('pamola_core.utils.progress.tqdm'):
            parent = HierarchicalProgressTracker(total=3, description="Parent")
            parent.update = MagicMock()

            child = parent.create_subtask(total=10, description="Child")
            child.close()

            # Check parent was updated
            parent.update.assert_called_once_with(1)


class TestContextManagers(unittest.TestCase):
    """Test the context managers for operation tracking."""

    def test_track_operation_safely_success(self):
        """Test track_operation_safely with successful operation."""
        with patch('pamola_core.utils.progress.HierarchicalProgressTracker') as mock_tracker_cls:
            mock_tracker = mock_tracker_cls.return_value

            with track_operation_safely("Test", 100) as tracker:
                self.assertIs(tracker, mock_tracker)
                tracker.update(50)

            mock_tracker.update.assert_called_once_with(50)
            mock_tracker.close.assert_called_once()

    def test_track_operation_safely_error(self):
        """Test track_operation_safely with error."""
        with patch('pamola_core.utils.progress.HierarchicalProgressTracker') as mock_tracker_cls:
            mock_tracker = mock_tracker_cls.return_value

            on_error_mock = MagicMock()

            with self.assertRaises(ValueError):
                with track_operation_safely("Test", 100, on_error=on_error_mock) as tracker:
                    raise ValueError("Test error")

            # Verify on_error was called
            on_error_mock.assert_called_once()
            self.assertIsInstance(on_error_mock.call_args[0][0], ValueError)

            # Verify tracker was closed
            mock_tracker.close.assert_called_once()


class TestDataFrameProcessing(unittest.TestCase):
    """Test the DataFrame processing functions."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'value': list(range(1000))
        })

    def test_process_dataframe_in_chunks(self):
        """Test processing DataFrame in chunks."""
        with patch('pamola_core.utils.progress.track_operation_safely') as mock_track:
            mock_context = MagicMock()
            mock_track.return_value.__enter__.return_value = mock_context

            # Define a processing function
            def process_func(chunk):
                return chunk['value'].sum()

            # Process the DataFrame
            results = process_dataframe_in_chunks_enhanced(
                df=self.df,
                process_func=process_func,
                description="Test",
                chunk_size=100
            )

            # Check results
            self.assertEqual(len(results), 10)  # 1000 items / 100 chunk size
            self.assertEqual(sum(results), sum(range(1000)))

            # Check tracker was updated 10 times
            self.assertEqual(mock_context.update.call_count, 10)

    def test_iterate_dataframe_chunks(self):
        """Test iterating over DataFrame chunks."""
        with patch('pamola_core.utils.progress.track_operation_safely') as mock_track:
            mock_context = MagicMock()
            mock_track.return_value.__enter__.return_value = mock_context

            chunks = list(iterate_dataframe_chunks_enhanced(
                df=self.df,
                chunk_size=200,
                description="Test"
            ))

            # Check chunks
            self.assertEqual(len(chunks), 5)  # 1000 items / 200 chunk size
            self.assertEqual(len(chunks[0]), 200)
            self.assertEqual(len(chunks[-1]), 200)

            # Check tracker was updated 5 times
            self.assertEqual(mock_context.update.call_count, 5)

    @unittest.skip("Skip parallel processing test that requires joblib")
    def test_process_dataframe_in_parallel(self):
        """Test processing DataFrame in parallel."""
        # Этот тест пропускается, так как требует установки joblib
        # и специфических моков, которые могут быть нестабильны
        pass


class TestMultiStageProcess(unittest.TestCase):
    """Test multi-stage process tracking."""

    def test_multi_stage_process(self):
        """Test creating a multi-stage process tracker."""
        with patch('pamola_core.utils.progress.HierarchicalProgressTracker') as mock_tracker_cls:
            stages = ["Load", "Process", "Save"]
            weights = [0.2, 0.6, 0.2]

            tracker = multi_stage_process(
                total_stages=3,
                stage_descriptions=stages,
                stage_weights=weights,
                track_memory=True
            )

            # Check tracker was created with correct params
            mock_tracker_cls.assert_called_once_with(
                total=3,
                description="Overall progress",
                unit="stages",
                track_memory=True
            )

            # Check returned tracker
            self.assertIs(tracker, mock_tracker_cls.return_value)

    def test_multi_stage_process_validation(self):
        """Test validation in multi-stage process."""
        # Test unequal lengths
        with self.assertRaises(ValueError):
            multi_stage_process(
                total_stages=3,
                stage_descriptions=["A", "B"],
                stage_weights=[0.3, 0.3, 0.4]
            )

        # Test weights don't sum to 1
        with self.assertRaises(ValueError):
            multi_stage_process(
                total_stages=3,
                stage_descriptions=["A", "B", "C"],
                stage_weights=[0.3, 0.3, 0.5]  # Sums to 1.1
            )


class TestLegacyIntegration(unittest.TestCase):
    """Test backwards compatibility with legacy classes."""

    def test_progress_bar_deprecated(self):
        """Test ProgressBar is deprecated but functional."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch('pamola_core.utils.progress.tqdm'):
                # Create and use a ProgressBar
                bar = ProgressBar(total=100, description="Test")
                bar.update(10)
                bar.close()

                # Check deprecation warning
                self.assertTrue(any(issubclass(warning.category, DeprecationWarning)
                                    for warning in w))

    def test_progress_tracker_deprecated(self):
        """Test ProgressTracker is deprecated but functional."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch('pamola_core.utils.progress.tqdm'):
                # Create and use a ProgressTracker
                tracker = ProgressTracker(total=100, description="Test")
                tracker.update(10)

                # Create a subtask
                subtask = tracker.create_subtask(total=50, description="Subtask")
                subtask.update(25)

                tracker.close()

                # Check deprecation warning
                self.assertTrue(any(issubclass(warning.category, DeprecationWarning)
                                    for warning in w))

    def test_legacy_functions_deprecated(self):
        """Test legacy functions are deprecated but functional."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch('pamola_core.utils.progress.process_dataframe_in_chunks_enhanced') as mock_enhanced:
                # Create a sample DataFrame
                df = pd.DataFrame({'value': range(100)})

                # Use legacy function
                process_dataframe_in_chunks(
                    df=df,
                    process_func=lambda x: x,
                    description="Test",
                    chunk_size=10
                )

                # Check enhanced version was called
                mock_enhanced.assert_called_once()

                # Check deprecation warning
                self.assertTrue(any(issubclass(warning.category, DeprecationWarning)
                                    for warning in w))


class TestIntegration(FileClosingTestCase):
    """Integration tests to verify overall functionality."""

    def test_basic_integration(self):
        """Basic integration test with real functionality."""
        # Make a small DataFrame
        df = pd.DataFrame({'value': list(range(30))})

        # Define processing function
        def square_values(chunk):
            return chunk['value'] ** 2

        # Process in chunks with minimal output
        with captured_output():
            results = process_dataframe_in_chunks_enhanced(
                df=df,
                process_func=square_values,
                description="Integration Test",
                chunk_size=10,
            )

        # Verify results without looking at output
        self.assertEqual(len(results), 3)  # 3 chunks
        expected = [sum(x ** 2 for x in range(0, 10)),
                    sum(x ** 2 for x in range(10, 20)),
                    sum(x ** 2 for x in range(20, 30))]
        for i, series in enumerate(results):
            self.assertEqual(series.sum(), expected[i])

    def test_hierarchical_integration(self):
        """Test hierarchical tracking in real operation."""
        # Create a small test scenario
        with captured_output():
            master = HierarchicalProgressTracker(total=2, description="Master")

            # First subtask
            subtask1 = master.create_subtask(total=5, description="Subtask 1")
            for i in range(5):
                subtask1.update(1)

            # Second subtask
            subtask2 = master.create_subtask(total=3, description="Subtask 2")
            for i in range(3):
                subtask2.update(1)

            master.close()

        # We can't easily verify the output, but this test ensures no exceptions


class TestErrorHandling(unittest.TestCase):
    """Test error handling capabilities."""

    def test_error_in_chunk_processing(self):
        """Test handling errors during chunk processing."""
        # Create a DataFrame
        df = pd.DataFrame({'value': list(range(50))})

        # Function that raises on specific values
        def problematic_function(chunk):
            if 25 in chunk['value'].values:
                raise ValueError("Value 25 encountered")
            return chunk['value'].sum()

        # Test fail mode (default)
        with self.assertRaises(ValueError):
            with captured_output():
                process_dataframe_in_chunks_enhanced(
                    df=df,
                    process_func=problematic_function,
                    description="Error Test",
                    chunk_size=10,
                    error_handling="fail"
                )

        # Test ignore mode
        with captured_output():
            results = process_dataframe_in_chunks_enhanced(
                df=df,
                process_func=problematic_function,
                description="Error Test",
                chunk_size=10,
                error_handling="ignore"
            )
            # Should have 4 successful chunks, missing the one with error
            self.assertEqual(len(results), 4)

        # Test log mode - проверяем только количество успешных чанков, а не логи
        with captured_output():
            results = process_dataframe_in_chunks_enhanced(
                df=df,
                process_func=problematic_function,
                description="Error Test",
                chunk_size=10,
                error_handling="log"
            )

            # Should have 4 successful chunks
            self.assertEqual(len(results), 4)


if __name__ == '__main__':
    unittest.main()