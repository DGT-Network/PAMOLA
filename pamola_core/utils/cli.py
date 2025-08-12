"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Command Line Interface Utilities
Description: Standard CLI argument handling for PAMOLA/PAMOLA.CORE tasks
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides standardized command-line argument parsing for
various tasks in the PAMOLA ecosystem, ensuring consistent interface
and parameter handling across different scripts.

Key features:
- Standardized CLI arguments across tasks
- Support for common parameters (paths, I/O settings, performance options)
- Framework-specific options (encryption, checkpoints, parallelism)
- Integration with the PAMOLA task configuration system
"""

import argparse
from typing import Dict, Any


def add_common_task_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add standard arguments for PAMOLA tasks.

    Args:
        parser: ArgumentParser to add arguments to
    """
    # Project structure arguments
    project_group = parser.add_argument_group('Project structure')
    project_group.add_argument("--project-root", type=str,
                               help="Root directory of the project")
    project_group.add_argument("--data-repository", type=str,
                               help="Root directory of the data repository")

    # Task execution arguments
    execution_group = parser.add_argument_group('Task execution')
    execution_group.add_argument("--force", action="store_true",
                                 help="Force restart: delete all previous checkpoints")
    execution_group.add_argument("--enable-checkpoints", action="store_true",
                                 help="Enable loading from existing checkpoints")
    execution_group.add_argument("--continue-on-error", action="store_true",
                                 help="Continue execution when individual operations fail")

    # Input/Output arguments
    io_group = parser.add_argument_group('I/O settings')
    io_group.add_argument("--input-datasets", type=str, nargs='+',
                          help="List of input dataset paths (space-separated)")
    io_group.add_argument("--auxiliary-datasets", type=str, nargs='+',
                          help="List of auxiliary dataset paths (space-separated)")
    io_group.add_argument("--encoding", type=str,
                          help="Input/output file encoding (default from config)")
    io_group.add_argument("--separator", type=str,
                          help="CSV field delimiter (default from config)")
    io_group.add_argument("--text-qualifier", type=str,
                          help="CSV text qualifier (default from config)")

    # Logging arguments
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument("--log-level", type=str,
                               choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                               help="Logging level")
    logging_group.add_argument("--log-file", type=str,
                               help="Path to log file (defaults to project_root/logs/task_id.log)")

    # Performance arguments
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument("--use-vectorization", action="store_true",
                            help="Enable vectorized operations where supported")
    perf_group.add_argument("--parallel-processes", type=int,
                            help="Number of parallel processes for multi-processing")
    perf_group.add_argument("--use-dask", action="store_true",
                            help="Enable Dask for distributed computation (if supported)")

    # Security arguments
    security_group = parser.add_argument_group('Security')
    security_group.add_argument("--use-encryption", action="store_true",
                                help="Enable output encryption")
    security_group.add_argument("--encryption-mode", type=str,
                                choices=["simple", "age"],
                                help="Encryption mode to use")
    security_group.add_argument("--encryption-key", type=str,
                                help="Path to encryption key file")


def add_profiling_task_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments specific to profiling tasks.

    Args:
        parser: ArgumentParser to add arguments to
    """
    # Add common arguments first
    add_common_task_arguments(parser)

    # Profiling-specific arguments
    profiling_group = parser.add_argument_group('Profiling')
    profiling_group.add_argument("--threshold", type=float,
                                 help="Threshold value for profiling decisions")
    profiling_group.add_argument("--subsets", type=str, nargs='+',
                                 help="List of subsets to process (space-separated)")
    profiling_group.add_argument("--generate-visualizations", action="store_true",
                                 help="Generate visualization artifacts")
    profiling_group.add_argument("--text-length-threshold", type=int,
                                 help="Threshold for long text field handling")


def add_group_profiler_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments specific to the Group Profiler task (t_1P1).

    Args:
        parser: ArgumentParser to add arguments to
    """
    # Add profiling arguments first
    add_profiling_task_arguments(parser)

    # Group profiler specific arguments
    group_prof_group = parser.add_argument_group('Group Profiler')
    group_prof_group.add_argument("--variance-threshold", type=float,
                                  help="Threshold for group variance (for aggregation decisions)")
    group_prof_group.add_argument("--large-group-threshold", type=int,
                                  help="Size threshold for large group classification")
    group_prof_group.add_argument("--large-group-variance-threshold", type=float,
                                  help="Variance threshold specifically for large groups")
    group_prof_group.add_argument("--hash-algorithm", type=str,
                                  choices=["md5", "minhash"],
                                  help="Algorithm for text field comparison")


def parse_args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert parsed arguments to a dictionary for task configuration.

    Filters out None values and transforms arguments to the format
    expected by the PAMOLA task configuration system.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of arguments for task configuration
    """
    # Convert Namespace to dictionary
    args_dict = vars(args)

    # Filter out None values and transform keys
    result = {}
    for key, value in args_dict.items():
        if value is not None:
            # Transform snake_case to lowercase with underscores
            config_key = key.replace('-', '_')
            result[config_key] = value

    return result


def create_group_profiler_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the Group Profiler task (t_1P1).

    Returns:
        ArgumentParser configured for the Group Profiler task
    """
    parser = argparse.ArgumentParser(
        description="Group Profiler Task (t_1P1) - Analyze variability within grouped resume data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_group_profiler_arguments(parser)

    return parser


def parse_group_profiler_args() -> Dict[str, Any]:
    """
    Parse command line arguments for the Group Profiler task.

    Returns:
        Dictionary of arguments for task configuration
    """
    parser = create_group_profiler_parser()
    args = parser.parse_args()
    return parse_args_to_dict(args)


def get_task_args_dict(task_id: str) -> Dict[str, Any]:
    """
    Get command line arguments for a specific task.

    This function automatically selects the appropriate parser based on task_id
    and returns a dictionary of parsed arguments.

    Args:
        task_id: Task identifier (e.g., "t_1P1")

    Returns:
        Dictionary of arguments for task configuration

    Raises:
        ValueError: If the task_id is not supported
    """
    if task_id == "t_1P1":
        return parse_group_profiler_args()

    # Add more task-specific parsers here

    raise ValueError(f"No parser defined for task: {task_id}")