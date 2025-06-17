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
from typing import Union, Optional


def configure_logging(log_file=None, level=logging.INFO, name="pamola_core", log_dir=None):
    """
    Configure logging for the project.

    Parameters:
    -----------
    log_file : str, optional
        Path to log file. If None, logs will only be output to console.
    level : int, optional
        Logging level (default INFO).
    name : str, optional
        Logger name (default "pamola_core").
    log_dir : str or Path, optional
        Directory for logs. Default is current working directory.

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Configure root logger
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
        log_level: Union[int, str] = logging.INFO,
        log_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[Union[str, Path]] = None,
        logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for a specific task.

    Parameters:
    -----------
    task_id : str
        Unique identifier for the task
    log_level : int or str, optional
        Logging level (default INFO). Can be string or numeric level.
    log_dir : Path or str, optional
        Directory where logs will be stored
    log_file : Path or str, optional
        Path to log file or filename
    logger_name : str, optional
        Name of the logger to configure (defaults to f"pamola_core.task.{task_id}")

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    try:
        # Convert level to int if it's a string
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Determine logger name
        if not logger_name:
            logger_name = f"pamola_core.task.{task_id}"

        # Determine log file path
        log_file_path = None
        if log_dir and log_file:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            # If log_file is just a filename, combine with log_dir
            log_file_path = Path(log_file)
            if not log_file_path.is_absolute() and str(log_file_path.parent) == ".":
                log_file_path = log_dir_path / log_file_path
        elif log_dir:
            # Only log_dir provided, use task_id for filename
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir_path / f"{task_id}.log"
        elif log_file:
            # Only log_file provided
            log_file_path = Path(log_file)

        # Configure the logger
        if log_file_path:
            return configure_logging(
                log_file=str(log_file_path),
                level=log_level,
                name=logger_name
            )
        else:
            # Console-only logger
            return configure_logging(
                level=log_level,
                name=logger_name
            )
    except Exception as e:
        # Return a basic logger instead of None in case of errors
        logger = logging.getLogger(f"task.{task_id}")
        logger.setLevel(log_level)
        logger.warning(f"Error configuring task logging: {str(e)}. Using fallback logger.")
        return logger


def get_logger(name):
    """
    Get a logger for the specified module.

    Parameters:
    -----------
    name : str
        Module/component name for logging

    Returns:
    --------
    logging.Logger
        Logger for the specified module
    """
    return logging.getLogger(name)


def getLogger(name):
    """
    Alias for get_logger to match Python's standard logging API.

    Parameters:
    -----------
    name : str
        Module/component name for logging

    Returns:
    --------
    logging.Logger
        Logger for the specified module
    """
    return get_logger(name)