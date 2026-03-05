"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Logging Configuration
Description: Standardized logging configuration for PAMOLA projects
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides utilities for configuring and managing logging
throughout the PAMOLA ecosystem, ensuring consistent log formatting and output.

Key features:
- Centralized logging configuration with consistent formatting
- Support for both console and file logging with rotation
- Task-specific logging configuration with context managers
- Thread-safe logger configuration
- Structured logging support (JSON format) for machine parsing
- Process/worker identification for distributed systems
- Compatibility with Python standard logging interface
- Automatic log directory creation and validation

Implementation follows best practices for Python logging with adaptations
for the specific needs of privacy-preserving data operations.
"""

import json
import logging
import sys
import threading
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Union

from pamola_core.errors.exceptions import ConfigurationError

# ======= Thread-safe configuration lock =======
_config_lock = threading.Lock()

# ======= Configured loggers registry =======
_configured_loggers: Dict[str, bool] = {}

# ======= Configurable logging system =======
_DEFAULT_LOG_NAME = "pamola_core"
_DEFAULT_LOG_TASK = "pamola_core.task"
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOG_FORMAT = (
    "%(asctime)s - [%(process)d] - %(levelname)s - %(name)s - %(message)s"
)
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_DEFAULT_BACKUP_COUNT = 5


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects suitable for machine parsing
    and integration with logging aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process_id": record.process,
            "thread_id": record.thread,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add task_id if present
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def _validate_log_level(level: Union[int, str]) -> int:
    """
    Validate and convert log level to integer.

    Parameters:
    -----------
    level : int or str
        Logging level to validate

    Returns:
    --------
    int
        Validated log level as integer

    Raises:
    -------
    ValueError
        If level is invalid
    """
    if isinstance(level, int):
        # Validate it's a known level
        if level not in (
            logging.NOTSET,
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ):
            raise ValueError(f"Invalid log level: {level}")
        return level

    if isinstance(level, str):
        level_upper = level.upper()
        if not hasattr(logging, level_upper):
            raise ValueError(
                f"Invalid log level: {level}. "
                f"Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        return getattr(logging, level_upper)

    raise ValueError(f"Log level must be int or str, got {type(level)}")


def _validate_logger_name(name: Optional[str]) -> str:
    """
    Validate logger name.

    Parameters:
    -----------
    name : str or None
        Logger name to validate

    Returns:
    --------
    str
        Validated logger name

    Raises:
    -------
    ValueError
        If name is invalid (empty or None)
    """
    if not name or (isinstance(name, str) and name.strip() == ""):
        raise ValueError(
            "Logger name cannot be empty. "
            "Provide a specific logger name to avoid configuring the root logger."
        )
    return name


def _validate_log_directory(log_path: Path) -> None:
    """
    Validate that log directory is writable.

    Parameters:
    -----------
    log_path : Path
        Path to log file

    Raises:
    -------
    LoggingConfigError
        If directory is not writable
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Test writability
        test_file = log_path.parent / ".pamola_write_test"
        test_file.touch()
        test_file.unlink()

    except (PermissionError, OSError) as e:
        raise ConfigurationError(
            reason=f"Cannot write to log directory {log_path.parent}: {e}",
            config_key="logging_config",
        ) from e


def configure_logging(
    name: str = _DEFAULT_LOG_NAME,
    level: Union[int, str] = _DEFAULT_LOG_LEVEL,
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    structured: bool = False,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    Configure logging with thread-safe handler management.

    Parameters:
    -----------
    name : str
        Name of the logger to configure (default "pamola_core")
    level : int or str, optional
        Logging level (default INFO). Can be string or numeric level
    log_file : str or Path, optional
        Path to log file or filename. If None, logs only to console
    log_dir : str or Path, optional
        Directory where logs will be stored. Default is current working directory
    max_bytes : int, optional
        Maximum size of log file before rotation (default 10MB)
    backup_count : int, optional
        Number of backup files to keep (default 5)
    structured : bool, optional
        Use JSON structured logging format (default False)
    force_reconfigure : bool, optional
        Force reconfiguration even if logger already configured (default False)

    Returns:
    --------
    logging.Logger
        Configured logger

    Raises:
    -------
    ValueError
        If parameters are invalid
    LoggingConfigError
        If configuration fails

    Examples:
    ---------
    >>> logger = configure_logging("my_app", level="DEBUG")
    >>> logger = configure_logging("my_app", log_file="app.log", log_dir="/var/log")
    >>> logger = configure_logging("my_app", structured=True)
    """
    # Validate inputs
    name = _validate_logger_name(name)
    level = _validate_log_level(level)

    with _config_lock:
        logger = logging.getLogger(name)

        # Check if already configured
        if name in _configured_loggers and not force_reconfigure:
            # Just update level if already configured
            logger.setLevel(level)
            return logger

        # Clear existing handlers if reconfiguring
        if logger.handlers:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

        # Set level
        logger.setLevel(level)
        logger.propagate = False

        # Create formatter
        formatter = (
            StructuredFormatter()
            if structured
            else logging.Formatter(_DEFAULT_LOG_FORMAT)
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            if log_dir:
                log_path = Path(log_dir) / log_file
            else:
                log_path = Path(log_file)

            # Validate directory
            _validate_log_directory(log_path)

            # Use rotating file handler
            file_handler = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Mark as configured
        _configured_loggers[name] = True

        return logger


def configure_task_logging(
    *,  # Force keyword-only arguments
    task_id: str,
    name: Optional[str] = None,
    level: Union[int, str] = _DEFAULT_LOG_LEVEL,
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    structured: bool = False,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    Configure logging for a specific task.

    Parameters:
    -----------
    task_id : str
        Unique identifier for the task (required, keyword-only)
    name : str, optional
        Name of the logger to configure (default "pamola_core.task.{task_id}")
    level : int or str, optional
        Logging level (default INFO)
    log_file : str or Path, optional
        Path to log file. If None and log_dir is set, uses "{task_id}.log"
    log_dir : str or Path, optional
        Directory where logs will be stored
    max_bytes : int, optional
        Maximum size of log file before rotation (default 10MB)
    backup_count : int, optional
        Number of backup files to keep (default 5)
    structured : bool, optional
        Use JSON structured logging format (default False)
    force_reconfigure : bool, optional
        Force reconfiguration even if logger already configured (default False)

    Returns:
    --------
    logging.Logger
        Configured logger with task context

    Raises:
    -------
    ValueError
        If task_id is invalid
    LoggingConfigError
        If configuration fails

    Examples:
    ---------
    >>> logger = configure_task_logging(task_id="anonymize_123")
    >>> logger = configure_task_logging(task_id="task_1", log_dir="/var/log/tasks")
    """
    if not task_id or (isinstance(task_id, str) and task_id.strip() == ""):
        raise ValueError("task_id cannot be empty")

    try:
        # Determine logger name
        if not name:
            logger_name = f"{_DEFAULT_LOG_TASK}.{task_id}"
        else:
            logger_name = f"{name}.{task_id}"

        # Validate level
        level = _validate_log_level(level)

        # Auto-generate log file if log_dir provided without log_file
        if log_dir and not log_file:
            log_file = f"{task_id}.log"

        return configure_logging(
            name=logger_name,
            level=level,
            log_file=log_file,
            log_dir=log_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
            structured=structured,
            force_reconfigure=force_reconfigure,
        )

    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(
            reason=f"Failed to configure task logging for task_id={task_id}: {e}",
            config_key="logging_config",
        ) from e


@contextmanager
def task_logging_context(task_id: str, cleanup: bool = True, **kwargs):
    """
    Context manager for task-specific logging with automatic cleanup.

    Parameters:
    -----------
    task_id : str
        Unique identifier for the task
    cleanup : bool, optional
        Whether to clean up handlers on exit (default True)
    **kwargs
        Additional arguments passed to configure_task_logging

    Yields:
    -------
    logging.Logger
        Configured task logger

    Examples:
    ---------
    >>> with task_logging_context(task_id="process_123", log_dir="/tmp") as logger:
    ...     logger.info("Processing started")
    ...     # Do work
    ...     logger.info("Processing completed")
    """
    logger = configure_task_logging(task_id=task_id, **kwargs)

    try:
        yield logger
    finally:
        if cleanup:
            # Clean up handlers
            with _config_lock:
                for handler in logger.handlers[:]:
                    try:
                        handler.close()
                    except Exception:
                        pass
                    logger.removeHandler(handler)

                # Remove from configured registry
                logger_name = logger.name
                if logger_name in _configured_loggers:
                    del _configured_loggers[logger_name]


def get_configured_logger(name: str) -> Optional[logging.Logger]:
    """
    Get a logger if it has been configured via this module.

    Parameters:
    -----------
    name : str
        Name of the logger

    Returns:
    --------
    logging.Logger or None
        Configured logger if it exists, None otherwise

    Examples:
    ---------
    >>> configure_logging("my_app")
    >>> logger = get_configured_logger("my_app")
    >>> assert logger is not None
    """
    if name in _configured_loggers:
        return logging.getLogger(name)
    return None


def is_configured(name: str) -> bool:
    """
    Check if a logger has been configured via this module.

    Parameters:
    -----------
    name : str
        Name of the logger

    Returns:
    --------
    bool
        True if logger is configured, False otherwise
    """
    return name in _configured_loggers


def reset_logging_config() -> None:
    """
    Reset all logging configuration.

    Removes all handlers from configured loggers and clears the registry.
    Useful for testing or application shutdown.

    Warning: This affects all loggers configured through this module.
    """
    with _config_lock:
        for logger_name in list(_configured_loggers.keys()):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
                logger.removeHandler(handler)

        _configured_loggers.clear()


# Maintain backward compatibility with simple getter
def getLogger(name: str) -> logging.Logger:
    """
    Get a logger by name.

    Note: This returns a standard Python logger that may not be configured
    with PAMOLA's formatting. Use configure_logging() or get_configured_logger()
    to ensure proper configuration.

    Parameters:
    -----------
    name : str
        Name of the logger

    Returns:
    --------
    logging.Logger
        Logger instance (may be unconfigured)

    Examples:
    ---------
    >>> # For configured logger, use:
    >>> logger = configure_logging("my_app")
    >>>
    >>> # For simple logger without PAMOLA formatting:
    >>> logger = getLogger("my_app")
    """
    if not name:
        raise ValueError("Logger name cannot be empty")
    return logging.getLogger(name)
