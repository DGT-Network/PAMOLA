"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Dependency Manager
Description: Dependency management for task execution
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for managing dependencies between tasks,
including locating dependency outputs, validating dependency completion,
and accessing dependency reports.

Key features:
- Resolution of dependency paths (absolute or task-relative)
- Validation of dependency completion status
- Security checks for dependency paths
- Support for various dependency types (tasks, absolute paths)
- Error handling with informative messages
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from pamola_core.utils.io import read_json
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError


class DependencyError(Exception):
    """Base exception for dependency-related errors."""
    pass


class DependencyMissingError(DependencyError):
    """Exception raised when a required dependency is missing."""
    pass


class DependencyFailedError(DependencyError):
    """Exception raised when a dependency task has failed."""
    pass


class TaskDependencyManager:
    """
    Manager for task dependencies.

    This class provides functionality for accessing and validating
    dependencies between tasks, including file paths, report status,
    and completion validation.

    Attributes:
        config: Task configuration containing dependency information
        logger: Logger for tracking dependency operations
    """

    def __init__(self, task_config, logger: logging.Logger):
        """
        Initialize dependency manager.

        Args:
            task_config: Task configuration containing dependency information
            logger: Logger for tracking dependency operations
        """
        self.config = task_config
        self.logger = logger

    def get_dependency_output(self, dependency_id: str, file_pattern: Optional[str] = None) -> Union[Path, List[Path]]:
        """
        Get the output directory or files from a dependency.

        Args:
            dependency_id: Dependency ID (task ID) or absolute path
            file_pattern: Optional file pattern to match within the dependency output dir

        Returns:
            Path to the dependency output directory or list of matching files

        Raises:
            PathSecurityError: If the path fails security validation
            DependencyMissingError: If the dependency output directory doesn't exist
        """
        # Check if dependency_id contains a path separator, treat as absolute path
        if self.is_absolute_dependency(dependency_id):
            path = Path(dependency_id)

            # Validate path security with allowed paths
            if not validate_path_security(
                    path,
                    allowed_paths=self.config.allowed_external_paths,
                    allow_external=self.config.allow_external
            ):
                self.logger.error(f"Absolute dependency path failed security validation: {path}")
                raise PathSecurityError(f"Absolute dependency path failed security validation: {path}")

            if not path.exists():
                error_message = f"Dependency path doesn't exist: {path}"
                if self.config.continue_on_error:
                    self.logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                    if file_pattern:
                        return []  # Return empty list for files
                    return path  # Return path even if it doesn't exist
                else:
                    raise DependencyMissingError(error_message)

            # If file pattern is specified, return matching files
            if file_pattern:
                matching_files = list(path.glob(file_pattern))
                if not matching_files:
                    self.logger.warning(f"No files matching pattern '{file_pattern}' in {path}")
                return matching_files

            return path

        # Treat as task ID
        try:
            # Use the project's standard directory structure logic to locate the task output directory
            output_dir = self.config.get_task_output_dir(dependency_id)

            if not output_dir.exists():
                error_message = f"Dependency output directory doesn't exist: {output_dir}"
                if self.config.continue_on_error:
                    self.logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                    if file_pattern:
                        return []  # Return empty list for files
                    return output_dir  # Return directory even if it doesn't exist
                else:
                    raise DependencyMissingError(error_message)

            # If file pattern is specified, return matching files
            if file_pattern:
                matching_files = list(output_dir.glob(file_pattern))
                if not matching_files:
                    self.logger.warning(f"No files matching pattern '{file_pattern}' in {output_dir}")
                return matching_files

            return output_dir
        except Exception as e:
            # Handle any errors during path resolution
            if not isinstance(e, DependencyMissingError):
                self.logger.error(f"Error resolving dependency output for {dependency_id}: {str(e)}")
            raise

    def get_dependency_report(self, dependency_id: str) -> Path:
        """
        Get the report file from a dependency.

        Args:
            dependency_id: Dependency ID (task ID)

        Returns:
            Path to the dependency report file

        Raises:
            DependencyMissingError: If the dependency report doesn't exist and continue_on_error is False
        """
        # Validate dependency_id for security
        if self.is_absolute_dependency(dependency_id):
            self.logger.error(f"Cannot get report for absolute dependency path: {dependency_id}")
            raise ValueError(f"Cannot get report for absolute dependency path: {dependency_id}")

        try:
            # Resolve report path using standard directory structure
            report_path = self.config.get_reports_dir() / f"{dependency_id}_report.json"

            if not report_path.exists():
                # Try alternate report formats
                alternate_paths = [
                    self.config.get_reports_dir() / f"{dependency_id}.json",
                    self.config.get_reports_dir() / dependency_id / "report.json"
                ]

                for alt_path in alternate_paths:
                    if alt_path.exists():
                        return alt_path

                # No report found
                error_message = f"Dependency report doesn't exist: {report_path}"
                if self.config.continue_on_error:
                    self.logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                    # IMPORTANT: Return an existing directory to avoid triggering FileNotFoundError
                    # when the caller tries to read this file
                    return self.config.get_reports_dir() / "dummy_report.json"
                else:
                    raise DependencyMissingError(error_message)

            return report_path
        except Exception as e:
            # Handle any errors during path resolution
            if not isinstance(e, DependencyMissingError):
                self.logger.error(f"Error resolving dependency report for {dependency_id}: {str(e)}")
            raise

    def assert_dependencies_completed(self) -> bool:
        """
        Check if all dependencies have completed successfully.

        Returns:
            True if all dependencies are complete, False otherwise

        Raises:
            DependencyMissingError: If a dependency report is missing or indicates failure
        """
        # Get dependencies from task configuration
        dependencies = getattr(self.config, "dependencies", [])

        if not dependencies:
            self.logger.info("No dependencies specified")
            return True

        self.logger.info(f"Checking {len(dependencies)} dependencies: {', '.join(dependencies)}")

        for dependency_id in dependencies:
            try:
                # Skip absolute paths in dependency checking
                if self.is_absolute_dependency(dependency_id):
                    self.logger.warning(f"Skipping completion check for absolute path dependency: {dependency_id}")
                    continue

                try:
                    # Get dependency report path
                    report_path = self.get_dependency_report(dependency_id)

                    # Safely try to load the report
                    try:
                        # Check if the report actually exists before trying to read it
                        if not report_path.exists():
                            if self.config.continue_on_error:
                                self.logger.warning(
                                    f"Report file {report_path} does not exist (continuing due to continue_on_error=True)")
                                continue
                            else:
                                raise FileNotFoundError(f"Dependency report not found: {report_path}")

                        # Now we can safely try to read the file
                        report_data = read_json(report_path)

                        if not report_data.get("success", False):
                            error_info = report_data.get('error_info', {})
                            error_message = f"Dependency {dependency_id} failed: "

                            if isinstance(error_info, dict) and 'message' in error_info:
                                error_message += error_info['message']
                            else:
                                error_message += 'Unknown error'

                            if self.config.continue_on_error:
                                self.logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                            else:
                                raise DependencyFailedError(error_message)

                        self.logger.info(f"Dependency {dependency_id} completed successfully")
                    except json.JSONDecodeError:
                        error_message = f"Invalid report format for dependency {dependency_id}"
                        if self.config.continue_on_error:
                            self.logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                        else:
                            raise DependencyMissingError(error_message)

                except DependencyMissingError as e:
                    # This is already handled properly based on continue_on_error
                    if not self.config.continue_on_error:
                        raise

            except FileNotFoundError:
                error_message = f"Missing report for dependency {dependency_id}"
                if self.config.continue_on_error:
                    self.logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                else:
                    raise DependencyMissingError(error_message)

        return True

    def is_absolute_dependency(self, dependency_id: str) -> bool:
        """
        Check if a dependency ID represents an absolute path.

        Args:
            dependency_id: Dependency ID to check

        Returns:
            True if dependency_id represents an absolute path
        """
        # Check if dependency_id contains a path separator or drive separator
        return (
                os.path.sep in dependency_id or  # Unix or Windows path separator
                ':' in dependency_id or  # Windows drive separator
                dependency_id.startswith('/')  # Absolute Unix path
        )

    def get_dependency_metrics(self, dependency_id: str, metric_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics from a dependency report.

        Args:
            dependency_id: Dependency ID (task ID)
            metric_path: Optional path within metrics to extract (dot notation)

        Returns:
            Dictionary of metrics from the dependency report

        Raises:
            DependencyMissingError: If the dependency report doesn't exist
            KeyError: If the specified metric path doesn't exist
        """
        try:
            # Get dependency report
            report_path = self.get_dependency_report(dependency_id)

            # Load report
            report_data = read_json(report_path)

            # Extract metrics
            metrics = report_data.get("metrics", {})

            # If a specific metric path is requested, extract it
            if metric_path:
                keys = metric_path.split('.')
                current = metrics
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        raise KeyError(f"Metric path '{metric_path}' not found in dependency {dependency_id}")

                # Return as a single-item dictionary with the leaf key
                leaf_key = keys[-1]
                return {leaf_key: current}

            return metrics
        except FileNotFoundError:
            raise DependencyMissingError(f"Dependency report not found for {dependency_id}")
        except json.JSONDecodeError:
            raise DependencyMissingError(f"Invalid report format for dependency {dependency_id}")

    def get_dependency_status(self, dependency_id: str) -> Dict[str, Any]:
        """
        Get status information about a dependency.

        Args:
            dependency_id: Dependency ID (task ID)

        Returns:
            Dictionary with status information

        Raises:
            DependencyMissingError: If the dependency report doesn't exist
        """
        try:
            # Get dependency report
            report_path = self.get_dependency_report(dependency_id)

            # Load report
            report_data = read_json(report_path)

            # Extract status information
            return {
                "task_id": dependency_id,
                "success": report_data.get("success", False),
                "execution_time": report_data.get("execution_time_seconds"),
                "completion_time": report_data.get("completion_time"),
                "error_info": report_data.get("error_info"),
                "report_path": str(report_path)
            }
        except FileNotFoundError:
            raise DependencyMissingError(f"Dependency report not found for {dependency_id}")
        except json.JSONDecodeError:
            raise DependencyMissingError(f"Invalid report format for dependency {dependency_id}")


class OptionalT1IDependencyManager(TaskDependencyManager):
    """
    Specialized dependency manager that treats t_1I dependency as optional.

    This manager extends the standard TaskDependencyManager to specifically
    handle t_1I dependency as an optional dependency, allowing execution to
    continue even if the t_1I_report.json is missing.
    """

    def assert_dependencies_completed(self) -> bool:
        """
        Check dependencies but skip errors for t_1I dependency.

        This method overrides the standard assert_dependencies_completed
        to provide special handling for t_1I dependency.

        Returns:
            bool: True if all dependencies are satisfied or t_1I is the only missing one

        Raises:
            DependencyMissingError: If non-t_1I dependencies are missing and continue_on_error is False
        """
        try:
            # Try standard dependency checking first
            return super().assert_dependencies_completed()
        except DependencyMissingError as e:
            # If the error is specifically about t_1I_report.json, skip it
            if "t_1I_report.json" in str(e):
                self.logger.warning(f"Optional dependency t_1I missing — skipping: {e}")
                return True

            # For other dependencies, re-raise the error
            raise