"""
Result classes for operations in the HHR project.

This module provides classes for representing and handling operation results,
including status codes, artifacts, and metrics.
"""

import enum
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict


class ValidationResult(TypedDict, total=False):
    exists: bool
    size_valid: bool
    type_valid: bool
    is_valid: bool
    checksum: Optional[str]

class OperationStatus(enum.Enum):
    """Status codes for operation results."""
    SUCCESS = "success"
    WARNING = "warning"  # Completed with some issues
    ERROR = "error"
    SKIPPED = "skipped"
    PARTIAL_SUCCESS = "partial_success"  # Completed but with some parts failed
    PENDING = "pending"  # Operation is still running or pending execution


class ArtifactGroup:
    """Represents a group of related artifacts."""

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

    def add_artifact(self, artifact):
        """Add an artifact to the group."""
        self.artifacts.append(artifact)

    def get_artifacts(self):
        """Get all artifacts in the group."""
        return self.artifacts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "artifacts": [a.to_dict() for a in self.artifacts]
        }


class OperationArtifact:
    """Represents a single artifact produced by an operation."""

    def __init__(self,
                 artifact_type: str,
                 path: Union[str, Path],
                 description: str = "",
                 category: str = "output",
                 tags: Optional[List[str]] = None):
        """
        Initialize an artifact.

        Parameters:
        -----------
        artifact_type : str
            Type of artifact (e.g., "json", "csv", "png")
        path : Union[str, Path]
            Path to the artifact
        description : str
            Description of the artifact
        category : str
            Category of the artifact (e.g., "output", "metric", "visualization")
        tags : List[str], optional
            Tags for categorizing the artifact
        """
        self.artifact_type = artifact_type
        self.path = Path(path) if isinstance(path, str) else path
        self.description = description
        self.category = category
        self.tags = tags or []

        # Additional metadata
        self.creation_time = datetime.now().isoformat()
        self.size = self._get_file_size()
        self.checksum = None  # Will be populated on demand

    def _get_file_size(self) -> Optional[int]:
        """Get the size of the artifact file."""
        try:
            if self.path.exists():
                return os.path.getsize(self.path)
        except Exception:
            pass
        return None

    def calculate_checksum(self, algorithm: str = 'sha256') -> Optional[str]:
        """
        Calculate a checksum for the artifact.

        Parameters:
        -----------
        algorithm : str
            Hash algorithm to use ('sha256', 'md5', 'sha1')

        Returns:
        --------
        str or None
            Checksum as a hexadecimal string, or None if file doesn't exist
        """
        if not self.path.exists():
            return None

        try:
            hash_func = getattr(hashlib, algorithm)()
            with open(self.path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)
            self.checksum = hash_func.hexdigest()
            return self.checksum
        except Exception:
            return None

    def exists(self) -> bool:
        """Check if the artifact file exists."""
        return self.path.exists()

    def validate(self) -> Dict[str, Any]:
        """
        Validate the artifact's existence and integrity.

        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        # Create results dict with explicit typing
        results: Dict[str, Any] = {
            "exists": self.exists(),
            "size_valid": False,
            "type_valid": False
        }

        if results["exists"]:
            # Check file size
            file_size = self._get_file_size()
            results["size_valid"] = file_size is not None and file_size > 0

            # Check file type based on extension
            expected_extension = f".{self.artifact_type}"
            actual_extension = self.path.suffix.lower()
            results["type_valid"] = actual_extension == expected_extension or (
                    self.artifact_type == 'json' and actual_extension == '.jsonl'
            )

            # Add checksum if needed
            if self.checksum is None:
                self.calculate_checksum()
            results["checksum"] = self.checksum

        results["is_valid"] = all([
            results["exists"],
            results["size_valid"],
            results["type_valid"]
        ])

        return results

    def add_tag(self, tag: str):
        """Add a tag to the artifact."""
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Update size before converting to dict
        self.size = self._get_file_size()

        return {
            "type": self.artifact_type,
            "path": str(self.path),
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "creation_time": self.creation_time,
            "size": self.size,
            "checksum": self.checksum
        }


class OperationResult:
    """
    Represents the result of an operation.

    This class encapsulates all information about the result of an operation,
    including status, artifacts, metrics, and error messages.
    """

    def __init__(self,
                 status: OperationStatus = OperationStatus.SUCCESS,
                 artifacts: List[OperationArtifact] = None,
                 metrics: Dict[str, Any] = None,
                 error_message: str = None,
                 execution_time: float = None):
        """
        Initialize an operation result.

        Parameters:
        -----------
        status : OperationStatus
            Status of the operation
        artifacts : List[OperationArtifact], optional
            List of artifacts produced by the operation
        metrics : Dict[str, Any], optional
            Metrics collected during the operation
        error_message : str, optional
            Error message if status is ERROR
        execution_time : float, optional
            Execution time in seconds
        """
        self.status = status
        self.artifacts = artifacts or []
        self.metrics = metrics or {}
        self.error_message = error_message
        self.execution_time = execution_time

        # Artifact groups
        self.artifact_groups = {}

    def add_artifact(self,
                     artifact_type: str,
                     path: Union[str, Path],
                     description: str = "",
                     category: str = "output",
                     tags: Optional[List[str]] = None,
                     group: Optional[str] = None) -> OperationArtifact:
        """
        Add an artifact to the result.

        Parameters:
        -----------
        artifact_type : str
            Type of artifact (e.g., "json", "csv", "png")
        path : Union[str, Path]
            Path to the artifact
        description : str
            Description of the artifact
        category : str
            Category of the artifact (e.g., "output", "metric", "visualization")
        tags : List[str], optional
            Tags for categorizing the artifact
        group : str, optional
            Name of the group to add this artifact to

        Returns:
        --------
        OperationArtifact
            The created artifact
        """
        artifact = OperationArtifact(artifact_type, path, description, category, tags)
        self.artifacts.append(artifact)

        # Add to group if specified
        if group:
            if group not in self.artifact_groups:
                self.artifact_groups[group] = ArtifactGroup(group)
            self.artifact_groups[group].add_artifact(artifact)

        return artifact

    def add_artifact_group(self, name: str, description: str = "") -> ArtifactGroup:
        """
        Add an artifact group.

        Parameters:
        -----------
        name : str
            Name of the group
        description : str
            Description of the group

        Returns:
        --------
        ArtifactGroup
            The created artifact group
        """
        if name not in self.artifact_groups:
            self.artifact_groups[name] = ArtifactGroup(name, description)
        return self.artifact_groups[name]

    def add_metric(self, name: str, value: Any):
        """
        Add a metric to the result.

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

    def get_artifacts_by_type(self, artifact_type: str) -> List[OperationArtifact]:
        """
        Get all artifacts of a specific type.

        Parameters:
        -----------
        artifact_type : str
            Type of artifacts to retrieve

        Returns:
        --------
        List[OperationArtifact]
            Artifacts of the specified type
        """
        return [a for a in self.artifacts if a.artifact_type == artifact_type]

    def get_artifacts_by_tag(self, tag: str) -> List[OperationArtifact]:
        """
        Get all artifacts with a specific tag.

        Parameters:
        -----------
        tag : str
            Tag to filter by

        Returns:
        --------
        List[OperationArtifact]
            Artifacts with the specified tag
        """
        return [a for a in self.artifacts if tag in a.tags]

    def get_artifacts_by_category(self, category: str) -> List[OperationArtifact]:
        """
        Get all artifacts in a specific category.

        Parameters:
        -----------
        category : str
            Category to filter by

        Returns:
        --------
        List[OperationArtifact]
            Artifacts in the specified category
        """
        return [a for a in self.artifacts if a.category == category]

    def get_artifact_group(self, group_name: str) -> Optional[ArtifactGroup]:
        """
        Get an artifact group by name.

        Parameters:
        -----------
        group_name : str
            Name of the group

        Returns:
        --------
        ArtifactGroup or None
            The artifact group, or None if not found
        """
        return self.artifact_groups.get(group_name)

    def validate_artifacts(self) -> Dict[str, Any]:
        """
        Validate all artifacts in the result.

        Returns:
        --------
        Dict[str, Any]
            Validation results for all artifacts
        """
        results = {}
        invalid_artifacts = []

        for artifact in self.artifacts:
            validation = artifact.validate()
            results[str(artifact.path)] = validation

            if not validation["is_valid"]:
                invalid_artifacts.append(str(artifact.path))

        results["all_valid"] = len(invalid_artifacts) == 0
        results["invalid_count"] = len(invalid_artifacts)
        results["invalid_artifacts"] = invalid_artifacts

        return results