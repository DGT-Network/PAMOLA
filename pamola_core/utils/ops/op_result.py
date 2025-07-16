"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Operation Results
Description: Classes for representing operation results and artifacts
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides classes for representing and handling operation results,
including status codes, artifacts, and metrics.

Key features:
- Standard result structure for all operations
- Status tracking for successful/failed operations
- Artifact registration and validation
- Metrics aggregation and reporting
- Support for grouping related artifacts
- Integration with DataWriter for consistent artifact generation

Satisfies:
- REQ-OPS-005: Operation results with artifact registration
- REQ-OPS-006: Artifact validation and metadata extraction

TODO:
- Delegate file operations to pamola_core.utils.io helpers
- Migrate filesystem operations to DataWriter for consistent handling
"""

import enum
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict

from pamola_core.utils.io import get_file_metadata, calculate_checksum


# Base error class for all operation-related errors
class OpsError(Exception):
    """Base class for all operation-related errors."""
    pass


class ValidationError(OpsError):
    """Error raised when artifact validation fails."""
    pass


class ValidationResult(TypedDict, total=False):
    """Typed dictionary for artifact validation results."""
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
        path : Path or str
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
        """
        Get the size of the artifact file.

        # TODO: delegate metadata extraction to pamola_core.utils.io.get_file_metadata()
        """
        try:
            metadata = get_file_metadata(self.path)
            return metadata.get("size_bytes")
        except Exception:
            return None

    def calculate_checksum(self, algorithm: str = 'sha256') -> Optional[str]:
        """
        Calculate a checksum for the artifact.

        # TODO: delegate metadata extraction to pamola_core.utils.io.calculate_checksum()

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
            self.checksum = calculate_checksum(self.path, algorithm)
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
            Validation results with detailed information about the artifact's validity
        """
        # Get full metadata
        metadata = get_file_metadata(self.path)

        # Create results dict with explicit typing
        results: Dict[str, Any] = {
            "exists": metadata.get("exists", False),
            "size_valid": False,
            "type_valid": False
        }

        if results["exists"]:
            # Check file size
            file_size = metadata.get("size_bytes")
            results["size_valid"] = file_size is not None and file_size > 0

            # Check file type based on extension
            expected_extension = f".{self.artifact_type}"
            actual_extension = metadata.get("extension", "")
            results["type_valid"] = actual_extension == expected_extension or (
                    self.artifact_type == 'json' and actual_extension == '.jsonl'
            )

            # Add additional metadata
            results["created_at"] = metadata.get("created_at")
            results["modified_at"] = metadata.get("modified_at")

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

    def add_tag(self, tag: str) -> None:
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

    def add_artifact(self, artifact: OperationArtifact) -> None:
        """Add an artifact to the group."""
        self.artifacts.append(artifact)

    def get_artifacts(self) -> List[OperationArtifact]:
        """Get all artifacts in the group."""
        return self.artifacts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "artifacts": [a.to_dict() for a in self.artifacts]
        }


class OperationResult:
    """
    Represents the result of an operation.

    This class encapsulates all information about the result of an operation,
    including status, artifacts, metrics, and error messages.

    Satisfies REQ-OPS-005: Provides result tracking and artifact management.
    """

    def __init__(
            self,
            status: OperationStatus = OperationStatus.SUCCESS,
            artifacts: Optional[List[OperationArtifact]] = None,
            metrics: Optional[Dict[str, Any]] = None,
            error_message: Optional[str] = None,
            execution_time: Optional[float] = None,
            error_trace: Optional[str] = None,
            exception: Optional[Exception] = None
    ):
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
        error_trace : str, optional
            Full error traceback for debugging
        """
        self.status = status
        self.artifacts = artifacts or []
        self.metrics = metrics or {}
        self.error_message = error_message
        self.execution_time = execution_time
        self.error_trace = error_trace
        self.exception = exception

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

        This method creates an OperationArtifact and adds it to the result.
        Artifacts represent files produced by the operation, such as output data,
        metrics, or visualizations.

        Parameters:
        -----------
        artifact_type : str
            Type of artifact (e.g., "json", "csv", "png"). This should match the file extension.
        path : Union[str, Path]
            Path to the artifact file. Can be a string or pathlib.Path object.
        description : str, optional
            Human-readable description of the artifact. Default is empty string.
        category : str, optional
            Category of the artifact for organization (e.g., "output", "metric", "visualization").
            Default is "output".
        tags : List[str], optional
            Tags for categorizing the artifact. Default is None.
        group : str, optional
            Name of the group to add this artifact to. If the group doesn't exist,
            it will be created. Default is None.

        Returns:
        --------
        OperationArtifact
            The created artifact object.

        Satisfies:
        ----------
        REQ-OPS-005: Registers artifacts in consistent, structured manner with metadata.

        Examples:
        ---------
        >>> result = OperationResult(status=OperationStatus.SUCCESS)
        >>> result.add_artifact("csv", "output/data.csv", "Anonymized data")
        >>> result.add_artifact("png", "visualizations/histogram.png", "Distribution visualization",
        ...                    category="visualization", tags=["histogram", "distribution"])
        """
        artifact = OperationArtifact(artifact_type, path, description, category, tags)
        self.artifacts.append(artifact)

        # Add to group if specified
        if group:
            if group not in self.artifact_groups:
                self.artifact_groups[group] = ArtifactGroup(group)
            self.artifact_groups[group].add_artifact(artifact)

        return artifact

    def register_artifact_via_writer(self,
                                     writer,  # DataWriter
                                     obj,  # DataFrame, dict, etc.
                                     subdir: str,
                                     name: str,
                                     artifact_type: str = None,
                                     description: str = "",
                                     category: str = "output",
                                     tags: Optional[List[str]] = None,
                                     group: Optional[str] = None) -> OperationArtifact:
        """
        Register an artifact using the DataWriter to write the file.

        This method delegates file writing to the DataWriter and then
        registers the resulting file as an operation artifact, providing
        a consistent interface for artifact generation.

        Parameters:
        -----------
        writer : DataWriter
            DataWriter instance to use for writing
        obj : Any
            Object to write (DataFrame, dict, etc.)
        subdir : str
            Subdirectory under task_dir to write to (e.g., "output", "dictionaries")
        name : str
            Base name for the artifact file (without extension)
        artifact_type : str, optional
            Type of artifact (e.g., "csv", "json"). If None, inferred from object type
        description : str, optional
            Description of the artifact
        category : str, optional
            Category of the artifact (e.g., "output", "metric", "visualization")
        tags : List[str], optional
            Tags for categorizing the artifact
        group : str, optional
            Name of the group to add this artifact to

        Returns:
        --------
        OperationArtifact
            The created artifact object

        Satisfies:
        ----------
        REQ-OPS-005: Uses DataWriter for consistent artifact generation.
        """
        # Determine artifact type if not provided
        if artifact_type is None:
            import pandas as pd
            import dask.dataframe as dd

            if isinstance(obj, (pd.DataFrame, dd.DataFrame)):
                artifact_type = "csv"  # Default for DataFrames
            elif isinstance(obj, dict):
                artifact_type = "json"  # Default for dictionaries
            elif str(type(obj).__module__).startswith(('matplotlib', 'plotly')):
                artifact_type = "png"  # Default for visualization
            else:
                artifact_type = "json"  # Default fallback

        try:
            # Use appropriate writer method based on object type
            import pandas as pd
            import dask.dataframe as dd

            if isinstance(obj, (pd.DataFrame, dd.DataFrame)):
                write_result = writer.write_dataframe(obj, name, format=artifact_type, subdir=subdir)
            elif isinstance(obj, dict) and artifact_type.lower() in ("json", "jsonl"):
                write_result = writer.write_json(obj, name, subdir=subdir)
            elif str(type(obj).__module__).startswith(('matplotlib', 'plotly')):
                write_result = writer.write_visualization(obj, name, subdir=subdir, format=artifact_type)
            else:
                raise ValidationError(
                    f"Unsupported combination of object type {type(obj)} and artifact_type {artifact_type}")

            # Create and register the artifact
            artifact = self.add_artifact(
                artifact_type=artifact_type,
                path=write_result.path,
                description=description,
                category=category,
                tags=tags,
                group=group
            )

            return artifact

        except Exception as e:
            # Wrap exceptions in ValidationError to maintain error hierarchy
            if not isinstance(e, OpsError):
                raise ValidationError(f"Failed to write artifact via DataWriter: {str(e)}") from e
            raise e

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

    def add_metric(self, name: str, value: Any) -> None:
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

    def add_nested_metric(self, category: str, name: str, value: Any) -> None:
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

        Satisfies:
        ----------
        REQ-OPS-006: Validates all artifacts exist and match expected formats.
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

    def to_reporter_details(self) -> Dict[str, Any]:
        """
        Create a dictionary of details for the reporter.

        Returns:
        --------
        Dict[str, Any]
            Dictionary with operation details for reporting

        Satisfies:
        ----------
        REQ-OPS-005: Provides structured reporting of operation results and metrics.
        """
        details: Dict[str, Any] = {
            "status": self.status.value,
            "execution_time": f"{self.execution_time:.2f} seconds"
            if self.execution_time else None,
        }

        if self.error_message:
            details["error_message"] = self.error_message

        if self.error_trace:
            details["error_trace"] = self.error_trace

        if self.metrics:
            # Add top-level metrics for quick reference
            for key, value in self.metrics.items():
                if isinstance(value, (bool, int, float, str)):
                    details[f"metric_{key}"] = value

        # Add artifact counts
        artifact_count = len(self.artifacts)
        if artifact_count > 0:
            details["artifacts_count"] = artifact_count

            # Group artifacts by type
            artifact_types = {}
            for artifact in self.artifacts:
                atype = artifact.artifact_type
                if atype not in artifact_types:
                    artifact_types[atype] = 0
                artifact_types[atype] += 1

            for atype, count in artifact_types.items():
                details[f"artifacts_{atype}_count"] = count

        return details


# Helper functions for external use
def get_file_size(path: Union[str, Path]) -> Optional[int]:
    """Get the size of a file in bytes."""
    from pamola_core.utils.io import get_file_size
    return get_file_size(path)