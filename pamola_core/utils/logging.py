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
- Support for both console and file logging
- Task-specific logging configuration
- Compatibility with Python standard logging interface
- Automatic log directory creation and management

Implementation follows best practices for Python logging with adaptations
for the specific needs of privacy-preserving data operations.

TODO:
- Add support for rotating log files to manage disk space
- Implement log level filtering based on module/component
- Add structured logging options (JSON format) for machine parsing
- Integrate with external logging systems (e.g., ELK stack)
- Add support for colorized console output for better readability
- Implement log anonymization for sensitive data patterns
"""

import logging
import sys
from pathlib import Path
from typing import List, Union, Optional

# ======= Configurable logging system =======
# Default configuration that can be overridden
_DEFAULT_LOG_NAME = "pamola_core"
_DEFAULT_LOG_TASK = "pamola_core.task"
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_DEFAULT_LOG_HANDLERS: Optional[List[logging.Handler]] = None # Will be initialized on first use


def configure_logging(
        name: str = _DEFAULT_LOG_NAME,
        level: Union[int, str] = _DEFAULT_LOG_LEVEL,
        log_file: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Configure logging.

    Parameters:
    -----------
    name : str
        Name of the logger to configure (default "pamola_core")
    level : int or str, optional
        Logging level (default INFO). Can be string or numeric level
    log_file : str or Path, optional
        Path to log file or filename. If None, logs will only be output to console
    log_dir : str or Path, optional
        Directory where logs will be stored. Default is current working directory

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(_DEFAULT_LOG_FORMAT)

    # Determine logger name
    if not name:
        name = f"{_DEFAULT_LOG_NAME}"

    # Convert level to int if it's a string
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

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

        # Create log directory if needed
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_task_logging(
        *,  # Force keyword-only arguments for clarity
        task_id: str,
        name: Optional[str] = None,
        level: Union[int, str] = _DEFAULT_LOG_LEVEL,
        log_file: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Configure logging with a specific task.

    Parameters:
    -----------
    task_id : str
        Unique identifier for the task
    name : str, optional
        Name of the logger to configure (default "pamola_core.task.{task_id}")
    level : int or str, optional
        Logging level (default INFO). Can be string or numeric level
    log_file : str or Path, optional
        Path to log file or filename. If None, logs will only be output to console
    log_dir : str or Path, optional
        Directory where logs will be stored. Default is current working directory

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    try:
        # Determine logger name
        if not name:
            name = f"{_DEFAULT_LOG_TASK}"

        if task_id:
            name = f"{name}.{task_id}"

        # Convert level to int if it's a string
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        # Determine log file when only log_dir provided, use task_id for filename
        if log_dir and not log_file:
            log_file = f"{task_id}.log"

        return configure_logging(name=name, level=level, log_file=log_file, log_dir=log_dir)
    except Exception as e:
        # Return a basic logger instead of None in case of errors
        logger = logging.getLogger(f"{_DEFAULT_LOG_TASK}.{task_id}")
        logger.setLevel(level)
        logger.warning(f"Error configuring task logging: {str(e)}. Using fallback logger.")
        return logger


def get_logger(
        name: Optional[str]
) -> logging.Logger:
    """
    Get a basic logger.

    Parameters:
    -----------
    name : str, optional
        Name for logging

    Returns:
    --------
    logging.Logger
        Basic logger
    """
    return logging.getLogger(name)


def getLogger(
        name: Optional[str]
) -> logging.Logger:
    """
    Alias for get_logger to match Python's standard logging API.

    Parameters:
    -----------
    name : str, optional
        Name for logging

    Returns:
    --------
    logging.Logger
        Basic logger
    """
    return get_logger(name=name)
