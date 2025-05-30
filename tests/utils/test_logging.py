"""
Unit tests for the PAMOLA.CORE logging module (pamola_core/utils/logging.py)

This test suite verifies that the logging module correctly:
- Configures loggers with different levels
- Creates log files in specified directories
- Automatically creates nested directories when needed
- Properly configures task-specific logging
- Creates and retrieves named loggers

Run with:
    python -m unittest tests/utils/test_logging.py
"""

import logging as std_logging
import shutil
import tempfile
import unittest
from pathlib import Path

# Import the module under test
from pamola_core.utils import logging as custom_logging


class TestLogging(unittest.TestCase):
    """Tests for the logging.py module"""

    def setUp(self):
        """Set up test environment - create temporary directories"""
        # Create a temporary directory for tests
        self.test_dir = Path(tempfile.mkdtemp())

        # Create a temporary directory for logs
        self.log_dir = self.test_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Create a temporary directory for tasks
        self.tasks_dir = self.test_dir / "tasks"
        self.tasks_dir.mkdir(exist_ok=True)

        # Create a specific task directory
        self.task_dir = self.tasks_dir / "task_001"
        self.task_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)

    def test_configure_logging_default(self):
        """Test basic configuration without log files"""
        logger = custom_logging.configure_logging()

        # Verify logger was created with the correct name
        self.assertEqual(logger.name, "pamola_core")

        # Verify log level is set correctly
        self.assertEqual(logger.level, std_logging.INFO)

        # Verify there is at least one handler (console)
        self.assertGreaterEqual(len(logger.handlers), 1)

        # Verify the first handler is a console handler
        self.assertIsInstance(logger.handlers[0], std_logging.StreamHandler)

    def test_configure_logging_with_file(self):
        """Test configuration with log file output"""
        log_file = "test.log"
        logger = custom_logging.configure_logging(
            log_file=log_file,
            log_dir=self.log_dir
        )

        # Check if file path is correctly constructed
        log_path = self.log_dir / log_file

        # Write a test message
        test_message = "Test log message"
        logger.info(test_message)

        # Verify the file exists
        self.assertTrue(log_path.exists())

        # Verify the message was written to the file
        with open(log_path, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)

        # Verify the logger has two handlers (console and file)
        self.assertEqual(len(logger.handlers), 2)
        self.assertIsInstance(logger.handlers[1], std_logging.FileHandler)

    def test_configure_logging_custom_level(self):
        """Test setting custom log level"""
        logger = custom_logging.configure_logging(level=std_logging.DEBUG)

        # Verify the level is set correctly
        self.assertEqual(logger.level, std_logging.DEBUG)

    def test_configure_task_logging(self):
        """Test task-specific logging configuration"""
        logger = custom_logging.configure_task_logging(self.task_dir)

        # Verify logger name is correctly formatted
        expected_name = f"pamola_core.task.{self.task_dir.name}"
        self.assertEqual(logger.name, expected_name)

        # Verify logs directory is created
        task_logs_dir = self.tasks_dir / "logs"
        self.assertTrue(task_logs_dir.exists())

        # Verify task log file is created with the correct name
        task_log_file = task_logs_dir / f"{self.task_dir.name}.log"

        # Write a test message
        test_message = "Task log test message"
        logger.info(test_message)

        # Verify the file exists
        self.assertTrue(task_log_file.exists())

        # Verify the message was written to the file
        with open(task_log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)

    def test_get_logger(self):
        """Test getting a logger for a specific module"""
        logger_name = "pamola_core.test_module"
        logger = custom_logging.get_logger(logger_name)

        # Verify logger name is set correctly
        self.assertEqual(logger.name, logger_name)

        # Verify getLogger alias works the same way
        logger2 = custom_logging.getLogger(logger_name)
        self.assertEqual(logger2.name, logger_name)

    def test_log_directory_creation(self):
        """Test automatic creation of log directories"""
        # Create a nested path that doesn't exist yet
        nested_log_dir = self.test_dir / "nested" / "log" / "directory"
        log_file = "deep_log.log"

        # Path should not exist before configure_logging call
        self.assertFalse(nested_log_dir.exists())

        # Configure logging with non-existent path
        logger = custom_logging.configure_logging(
            log_file=log_file,
            log_dir=nested_log_dir
        )

        # Write something to the log
        logger.info("Testing nested directory creation")

        # Verify directory and file were created
        self.assertTrue(nested_log_dir.exists())
        self.assertTrue((nested_log_dir / log_file).exists())


if __name__ == "__main__":
    unittest.main()