"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Reporting
Description: Task execution reporting and artifact tracking
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for creating and managing task reports,
including tracking operations, artifacts, and generating JSON reports.

Key features:
- Operation tracking during task execution
- Artifact registration and organization
- Metric collection and aggregation
- Standardized report generation
- Artifact grouping and categorization
- Progress tracking for report generation
"""

import os
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from pamola_core.utils.io import write_json, ensure_directory
from pamola_core.utils.tasks.task_config import validate_path_security


class ReportingError(Exception):
    """Exception raised for reporting errors."""
    pass


class ArtifactGroup:
    """
    Group of related artifacts in a task report.

    Allows organizing artifacts into logical groups for better structure.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize an artifact group.

        Args:
            name: Name of the group
            description: Description of the group
        """
        self.name = name
        self.description = description
        self.artifacts = []
        self.created_at = datetime.now().isoformat()

    def add_artifact(self, artifact: Dict[str, Any]):
        """
        Add an artifact to the group.

        Args:
            artifact: Artifact information dictionary
        """
        self.artifacts.append(artifact)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the group to a dictionary for serialization.

        Returns:
            Dictionary representation of the group
        """
        return {
            "name": self.name,
            "description": self.description,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "count": len(self.artifacts)
        }


class TaskReporter:
    """
    Task reporter for generating and managing task reports.

    Tracks operations, artifacts, and metrics, and generates
    JSON reports for task execution. Integrates with the progress manager
    for coordinated progress tracking during report generation.
    """

    def __init__(self, task_id: str, task_type: str, description: str, report_path: Union[str, Path],
                 progress_manager: Optional[Any] = None):
        """
        Initialize the task reporter.

        Args:
            task_id: ID of the task
            task_type: Type of the task
            description: Description of the task
            report_path: Path where the report will be saved
            progress_manager: Optional progress manager for tracking operations
        """
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.progress_manager = progress_manager

        # Validate path security
        if not validate_path_security(report_path):
            raise ReportingError(f"Insecure report path: {report_path}")

        self.report_path = Path(report_path)

        # Timestamp for report creation
        self.start_time = datetime.now()
        self.end_time = None

        # Lists for tracking operations and artifacts
        self.operations = []
        self.artifacts = []

        # Dictionary for artifact groups
        self.artifact_groups = {}

        # System information
        self.system_info = self._get_system_info()

        # Report status
        self.status = "running"
        self.success = None
        self.execution_time = None

        # Metrics collection
        self.metrics = {}

        # Error tracking
        self.errors = []
        self.warnings = []

        # Track memory usage
        self.peak_memory_usage = self._get_current_memory_usage()

    def _get_system_info(self) -> Dict[str, str]:
        """
        Get system information for the report.

        Returns:
            Dictionary with system information
        """
        return {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
            "machine": socket.gethostname(),
            "cpu_count": os.cpu_count() or 0,
            "pamola_version": os.environ.get("PAMOLA_VERSION", "unknown")
        }

    def _get_current_memory_usage(self) -> float:
        """
        Get current memory usage of the process.

        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except (ImportError, Exception):
            return 0.0

    def add_operation(self, name: str, status: str = "success", details: Dict[str, Any] = None):
        """
        Add an operation to the report.

        Args:
            name: Name of the operation
            status: Status of the operation (success, warning, error)
            details: Additional details about the operation
        """
        operation = {
            "operation": name,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details or {}
        }

        self.operations.append(operation)

        # Track errors and warnings
        if status == "error":
            self.errors.append({
                "operation": name,
                "timestamp": operation["timestamp"],
                "message": details.get("error_message", "Unknown error"),
                "details": details
            })
        elif status == "warning":
            self.warnings.append({
                "operation": name,
                "timestamp": operation["timestamp"],
                "message": details.get("warning_message", "Unknown warning"),
                "details": details
            })

        # Update memory usage
        current_memory = self._get_current_memory_usage()
        if current_memory > self.peak_memory_usage:
            self.peak_memory_usage = current_memory

        # Log operation addition via progress manager if available
        if self.progress_manager:
            self.progress_manager.log_info(f"Added operation: {name} (status: {status})")

    def add_artifact(self, artifact_type: str, path: Union[str, Path], description: str = "",
                     category: str = "output", tags: Optional[List[str]] = None,
                     group_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an artifact to the report.

        Args:
            artifact_type: Type of the artifact (e.g., "json", "csv", "png")
            path: Path to the artifact
            description: Description of the artifact
            category: Category of the artifact (e.g., "output", "metric", "visualization")
            tags: Tags for categorizing the artifact
            group_name: Name of the group to add this artifact to
            metadata: Additional metadata for the artifact
        """
        # Use progress manager for artifact addition if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                    name="add_artifact",
                    total=2,  # validate + add
                    description=f"Adding artifact: {artifact_type}",
                    leave=False
            ) as progress:
                try:
                    # Validate path security
                    if not validate_path_security(path):
                        progress.update(1, {"status": "validation_failed"})
                        raise ReportingError(f"Insecure artifact path: {path}")

                    progress.update(1, {"status": "validated"})

                    # Process artifact (use helper method to avoid code duplication)
                    self._process_artifact(
                        artifact_type, path, description, category,
                        tags or [], group_name, metadata or {}
                    )

                    progress.update(1, {"status": "added"})
                except Exception as e:
                    progress.update(1, {"status": "error", "error": str(e)})
                    raise
        else:
            # Validate path security
            if not validate_path_security(path):
                raise ReportingError(f"Insecure artifact path: {path}")

            # Process artifact without progress tracking
            self._process_artifact(
                artifact_type, path, description, category,
                tags or [], group_name, metadata or {}
            )

    def _process_artifact(self, artifact_type: str, path: Union[str, Path], description: str,
                          category: str, tags: List[str], group_name: Optional[str], metadata: Dict[str, Any]):
        """
        Process and add an artifact to the report (internal helper method).

        Args:
            artifact_type: Type of the artifact
            path: Path to the artifact
            description: Description of the artifact
            category: Category of the artifact
            tags: Tags for categorizing the artifact
            group_name: Name of the group to add this artifact to
            metadata: Additional metadata for the artifact
        """
        # Convert path to string and get filename
        path_obj = Path(path)
        path_str = str(path_obj)
        filename = path_obj.name

        # Get file metadata if it exists
        size_bytes = None
        timestamp = None
        if path_obj.exists():
            try:
                stat = path_obj.stat()
                size_bytes = stat.st_size
                timestamp = datetime.fromtimestamp(stat.st_mtime).isoformat()
            except (OSError, PermissionError):
                pass

        artifact = {
            "type": artifact_type,
            "path": path_str,
            "filename": filename,
            "description": description,
            "category": category,
            "tags": tags,
            "size_bytes": size_bytes,
            "timestamp": timestamp or datetime.now().isoformat(),
            "metadata": metadata
        }

        self.artifacts.append(artifact)

        # Add to group if specified
        if group_name:
            if group_name not in self.artifact_groups:
                self.add_artifact_group(group_name)
            self.artifact_groups[group_name].add_artifact(artifact)

    def add_artifact_group(self, name: str, description: str = "") -> ArtifactGroup:
        """
        Add or get an artifact group.

        Args:
            name: Name of the group
            description: Description of the group

        Returns:
            The artifact group
        """
        if name not in self.artifact_groups:
            self.artifact_groups[name] = ArtifactGroup(name, description)

            # Log group creation via progress manager if available
            if self.progress_manager:
                self.progress_manager.log_info(f"Created artifact group: {name}")

        return self.artifact_groups[name]

    def get_artifact_group(self, name: str) -> Optional[ArtifactGroup]:
        """
        Get an artifact group by name.

        Args:
            name: Name of the group

        Returns:
            The artifact group if it exists, None otherwise
        """
        return self.artifact_groups.get(name)

    def get_artifacts_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of artifacts with the specified tag
        """
        return [a for a in self.artifacts if tag in a.get("tags", [])]

    def get_artifacts_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts in a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of artifacts in the specified category
        """
        return [a for a in self.artifacts if a.get("category") == category]

    def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts of a specific type.

        Args:
            artifact_type: Type to filter by

        Returns:
            List of artifacts of the specified type
        """
        return [a for a in self.artifacts if a.get("type") == artifact_type]

    def add_metric(self, name: str, value: Any):
        """
        Add a metric to the report.

        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.metrics[name] = value

        # Log metric addition via progress manager if available
        if self.progress_manager:
            self.progress_manager.log_debug(f"Added metric: {name} = {value}")

    def add_nested_metric(self, category: str, name: str, value: Any):
        """
        Add a nested metric under a category.

        Args:
            category: Category for the metric
            name: Name of the metric
            value: Value of the metric
        """
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category][name] = value

        # Log nested metric addition via progress manager if available
        if self.progress_manager:
            self.progress_manager.log_debug(f"Added metric: {category}.{name} = {value}")

    def add_task_summary(self, success: bool, execution_time: float = None,
                         metrics: Dict[str, Any] = None, error_info: Dict[str, Any] = None,
                         encryption: Dict[str, Any] = None):
        """
        Add task summary to the report.

        Args:
            success: Whether the task executed successfully
            execution_time: Task execution time in seconds
            metrics: Additional metrics to include in the report
            error_info: Error information if the task failed
            encryption: Encryption information
        """
        # Use progress manager for task summary if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                    name="add_task_summary",
                    total=4,  # update state, add metrics, add errors, update memory
                    description="Adding task summary",
                    leave=False
            ) as progress:
                self._add_task_summary_internal(
                    success, execution_time, metrics, error_info, encryption, progress
                )
        else:
            # Process without progress tracking
            self._add_task_summary_internal(
                success, execution_time, metrics, error_info, encryption
            )

    def _add_task_summary_internal(self, success: bool, execution_time: float = None,
                                   metrics: Dict[str, Any] = None, error_info: Dict[str, Any] = None,
                                   encryption: Dict[str, Any] = None, progress=None):
        """
        Internal helper for adding task summary with progress tracking support.
        """
        self.end_time = datetime.now()
        self.status = "completed" if success else "failed"
        self.success = success

        if execution_time is not None:
            self.execution_time = execution_time
        else:
            # Calculate execution time from start and end times
            delta = self.end_time - self.start_time
            self.execution_time = delta.total_seconds()

        if progress:
            progress.update(1, {"status": "updated_state"})

        # Add metrics if provided
        if metrics:
            for category, values in metrics.items():
                if isinstance(values, dict):
                    # Nested metrics
                    for name, value in values.items():
                        self.add_nested_metric(category, name, value)
                else:
                    # Top-level metric
                    self.add_metric(category, values)

        if progress:
            progress.update(1, {"status": "added_metrics"})

        # Add error information if provided
        if error_info and not success:
            self.errors.append({
                "timestamp": datetime.now().isoformat(),
                "type": error_info.get("type", "unknown"),
                "message": error_info.get("message", "Unknown error"),
                "details": error_info
            })

        if progress:
            progress.update(1, {"status": "added_errors"})

        # Add encryption information if provided
        if encryption:
            self.add_metric("encryption", encryption)

        # Update peak memory usage one last time
        current_memory = self._get_current_memory_usage()
        if current_memory > self.peak_memory_usage:
            self.peak_memory_usage = current_memory

        if progress:
            progress.update(1, {"status": "updated_memory"})

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate the task report.

        Returns:
            Report as a dictionary
        """
        # Use progress manager for report generation if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                    name="generate_report",
                    total=1,
                    description="Generating task report"
            ) as progress:
                try:
                    report = self._generate_report_internal()
                    progress.update(1, {"status": "success"})
                    return report
                except Exception as e:
                    progress.update(1, {"status": "error", "error": str(e)})
                    raise
        else:
            # Generate report without progress tracking
            return self._generate_report_internal()

    def _generate_report_internal(self) -> Dict[str, Any]:
        """
        Internal helper for generating the report.
        """
        # If task is still running, set end_time to current time
        if self.end_time is None:
            self.end_time = datetime.now()

            # Calculate execution time
            delta = self.end_time - self.start_time
            self.execution_time = delta.total_seconds()

        report = {
            "task_id": self.task_id,
            "task_description": self.description,
            "task_type": self.task_type,
            "script_path": os.path.abspath(os.path.realpath(os.path.join(os.getcwd(), __file__))),
            "system_info": self.system_info,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_seconds": self.execution_time,
            "status": self.status,
            "operations": self.operations,
            "artifacts": self.artifacts,
            "artifact_groups": {name: group.to_dict() for name, group in self.artifact_groups.items()},
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "memory_usage_mb": round(self.peak_memory_usage, 2)
        }

        return report

    def save_report(self) -> Path:
        """
        Generate and save the report to disk.

        Returns:
            Path to the saved report

        Raises:
            ReportingError: If saving fails
        """
        # Use progress manager for report saving if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                    name="save_report",
                    total=3,  # generate report, ensure directory, write to disk
                    description="Saving task report"
            ) as progress:
                try:
                    # Generate report
                    report = self.generate_report()
                    progress.update(1, {"status": "generated"})

                    # Ensure directory exists
                    ensure_directory(self.report_path.parent)
                    progress.update(1, {"status": "directory_created"})

                    # Write to disk
                    try:
                        write_json(report, self.report_path)
                        progress.update(1, {"status": "written"})

                        self.progress_manager.log_info(f"Report saved to {self.report_path}")
                        return self.report_path
                    except Exception as e:
                        progress.update(1, {"status": "error", "error": str(e)})
                        raise ReportingError(f"Failed to save report: {str(e)}")
                except Exception as e:
                    if isinstance(e, ReportingError):
                        raise
                    raise ReportingError(f"Error generating or saving report: {str(e)}")
        else:
            # Generate and save report without progress tracking
            try:
                # Generate report
                report = self.generate_report()

                # Ensure directory exists
                ensure_directory(self.report_path.parent)

                # Save report
                try:
                    write_json(report, self.report_path)
                    return self.report_path
                except Exception as e:
                    raise ReportingError(f"Failed to save report: {str(e)}")
            except Exception as e:
                if isinstance(e, ReportingError):
                    raise
                raise ReportingError(f"Error generating or saving report: {str(e)}")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.

        Finalizes the report and saves it.
        """
        # Use progress manager for report finalization if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                    name="finalize_report",
                    total=2,  # add summary, save report
                    description="Finalizing task report"
            ) as progress:
                try:
                    # Check if an exception occurred
                    success = exc_type is None

                    # Add task summary
                    self.add_task_summary(success)
                    progress.update(1, {"status": "summary_added"})

                    # Add error information if exception occurred
                    if exc_type is not None:
                        import traceback
                        error_info = {
                            "type": exc_type.__name__,
                            "message": str(exc_val),
                            "traceback": traceback.format_exc()
                        }
                        self.add_operation(
                            name=f"Exception: {exc_type.__name__}",
                            status="error",
                            details=error_info
                        )

                    # Save report
                    try:
                        self.save_report()
                        progress.update(1, {"status": "report_saved"})
                    except Exception as e:
                        progress.update(1, {"status": "save_error", "error": str(e)})
                        self.progress_manager.log_error(f"Failed to save report: {str(e)}")
                except Exception as e:
                    if progress:
                        progress.update(1, {"status": "error", "error": str(e)})
                    self.progress_manager.log_error(f"Error during report finalization: {str(e)}")
        else:
            # Check if an exception occurred
            success = exc_type is None

            # Add task summary
            self.add_task_summary(success)

            # Add error information if exception occurred
            if exc_type is not None:
                import traceback
                error_info = {
                    "type": exc_type.__name__,
                    "message": str(exc_val),
                    "traceback": traceback.format_exc()
                }
                self.add_operation(
                    name=f"Exception: {exc_type.__name__}",
                    status="error",
                    details=error_info
                )

            # Save report
            try:
                self.save_report()
            except Exception as e:
                import logging
                logging.error(f"Failed to save report: {str(e)}")

        # Don't suppress exceptions
        return False