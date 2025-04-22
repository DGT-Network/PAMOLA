"""
Task Reporting Module for HHR project.

This module provides functionality for creating and managing task reports,
including tracking operations, artifacts, and generating JSON reports.
"""

import os
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from pamola_core.utils.io import write_json, ensure_directory


class ArtifactGroup:
    """
    Group of related artifacts in a task report.

    Allows organizing artifacts into logical groups for better structure.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize an artifact group.

        Parameters:
        -----------
        name : str
            Name of the group
        description : str
            Description of the group
        """
        self.name = name
        self.description = description
        self.artifacts = []
        self.created_at = datetime.now().isoformat()

    def add_artifact(self, artifact: Dict[str, Any]):
        """
        Add an artifact to the group.

        Parameters:
        -----------
        artifact : Dict[str, Any]
            Artifact information dictionary
        """
        self.artifacts.append(artifact)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the group to a dictionary for serialization.

        Returns:
        --------
        Dict[str, Any]
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
    JSON reports for task execution.
    """

    def __init__(self, task_id: str, task_type: str, description: str, report_path: Path):
        """
        Initialize the task reporter.

        Parameters:
        -----------
        task_id : str
            ID of the task
        task_type : str
            Type of the task
        description : str
            Description of the task
        report_path : Path
            Path where the report will be saved
        """
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
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

    def _get_system_info(self) -> Dict[str, str]:
        """
        Get system information for the report.

        Returns:
        --------
        Dict[str, str]
            Dictionary with system information
        """
        return {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
            "machine": socket.gethostname()
        }

    def add_operation(self, name: str, status: str = "success", details: Dict[str, Any] = None):
        """
        Add an operation to the report.

        Parameters:
        -----------
        name : str
            Name of the operation
        status : str
            Status of the operation (success, warning, error)
        details : Dict[str, Any], optional
            Additional details about the operation
        """
        operation = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details or {}
        }

        self.operations.append(operation)

    def add_artifact(self, artifact_type: str, path: Union[str, Path], description: str = "",
                     category: str = "output", tags: Optional[List[str]] = None,
                     group_name: Optional[str] = None):
        """
        Add an artifact to the report.

        Parameters:
        -----------
        artifact_type : str
            Type of the artifact (e.g., json, csv, png)
        path : Union[str, Path]
            Path to the artifact
        description : str
            Description of the artifact
        category : str
            Category of the artifact (e.g., output, metric, visualization)
        tags : List[str], optional
            Tags for categorizing the artifact
        group_name : str, optional
            Name of the group to add this artifact to
        """
        artifact = {
            "type": artifact_type,
            "path": str(path),
            "description": description,
            "category": category,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat()
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

        Parameters:
        -----------
        name : str
            Name of the group
        description : str
            Description of the group

        Returns:
        --------
        ArtifactGroup
            The artifact group
        """
        if name not in self.artifact_groups:
            self.artifact_groups[name] = ArtifactGroup(name, description)
        return self.artifact_groups[name]

    def get_artifact_group(self, name: str) -> Optional[ArtifactGroup]:
        """
        Get an artifact group by name.

        Parameters:
        -----------
        name : str
            Name of the group

        Returns:
        --------
        Optional[ArtifactGroup]
            The artifact group if it exists, None otherwise
        """
        return self.artifact_groups.get(name)

    def get_artifacts_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts with a specific tag.

        Parameters:
        -----------
        tag : str
            Tag to filter by

        Returns:
        --------
        List[Dict[str, Any]]
            List of artifacts with the specified tag
        """
        return [a for a in self.artifacts if tag in a.get("tags", [])]

    def get_artifacts_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts in a specific category.

        Parameters:
        -----------
        category : str
            Category to filter by

        Returns:
        --------
        List[Dict[str, Any]]
            List of artifacts in the specified category
        """
        return [a for a in self.artifacts if a.get("category") == category]

    def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts of a specific type.

        Parameters:
        -----------
        artifact_type : str
            Type to filter by

        Returns:
        --------
        List[Dict[str, Any]]
            List of artifacts of the specified type
        """
        return [a for a in self.artifacts if a.get("type") == artifact_type]

    def add_metric(self, name: str, value: Any):
        """
        Add a metric to the report.

        Parameters:
        -----------
        name : str
            Name of the metric
        value : Any
            Value of the metric
        """
        self.metrics[name] = value

    def add_nested_metric(self, category: str, name: str, value: Any):
        """
        Add a nested metric under a category.

        Parameters:
        -----------
        category : str
            Category for the metric
        name : str
            Name of the metric
        value : Any
            Value of the metric
        """
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category][name] = value

    def add_task_summary(self, success: bool, execution_time: float = None):
        """
        Add task summary to the report.

        Parameters:
        -----------
        success : bool
            Whether the task executed successfully
        execution_time : float, optional
            Task execution time in seconds
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

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate the task report.

        Returns:
        --------
        Dict[str, Any]
            Report as a dictionary
        """
        # If task is still running, set end_time to current time
        if self.end_time is None:
            self.end_time = datetime.now()

            # Calculate execution time
            delta = self.end_time - self.start_time
            self.execution_time = delta.total_seconds()

        report = {
            "task": {
                "id": self.task_id,
                "type": self.task_type,
                "description": self.description
            },
            "system": self.system_info,
            "execution": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": self.execution_time,
                "status": self.status
            },
            "operations": self.operations,
            "artifacts": self.artifacts,
            "artifact_groups": {name: group.to_dict() for name, group in self.artifact_groups.items()},
            "metrics": self.metrics
        }

        return report

    def save_report(self) -> Path:
        """
        Generate and save the report to disk.

        Returns:
        --------
        Path
            Path to the saved report
        """
        # Generate report
        report = self.generate_report()

        # Ensure directory exists
        ensure_directory(self.report_path.parent)

        # Save report
        write_json(report, self.report_path)

        return self.report_path

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.

        Finalizes the report and saves it.
        """
        # Check if an exception occurred
        success = exc_type is None

        # Add task summary
        self.add_task_summary(success)

        # Save report
        self.save_report()

        # Don't suppress exceptions
        return False