"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
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
from pamola_core.utils.artifact_policy import (
    ArtifactClass,
    check_class_and_content,
    class_for_category,
)
from pamola_core.errors.exceptions import ReportingError
from pamola_core.utils.io import write_json, ensure_directory
from pamola_core.utils.tasks.task_config import validate_path_security


def _normalized_path(path: Any) -> str:
    """Canonical key for an artifact file: symlinks resolved, case folded per platform."""
    p = os.path.abspath(str(path))
    try:
        p = os.path.realpath(p)
    except OSError:
        pass
    return os.path.normcase(p)


def _identity_key(path: Any):
    """File identity: (device, inode) for an existing file, else the normalized path.

    One ``stat`` per path; hard links, symlinks and case aliases of an existing file share
    the key. A path that does not exist yet is keyed by its normalized path and is
    reconciled once the file exists."""
    try:
        st = os.stat(str(path))
        if st.st_ino:
            return ("id", st.st_dev, st.st_ino)
    except OSError:
        pass
    return ("path", _normalized_path(path))


def _same_artifact_path(a: Any, b: Any) -> bool:
    """True when two path strings name the same file (identity key or normalized path)."""
    return _identity_key(a) == _identity_key(b) or _normalized_path(a) == _normalized_path(b)


#: Protection order for source-bearing artifacts and for explicit classes in general.
_PROTECTION_RANK = {
    ArtifactClass.PROTECTED_MAPPING.value: 4,
    ArtifactClass.DIAGNOSTIC.value: 3,
    ArtifactClass.DATA.value: 2,
    ArtifactClass.AGGREGATE_METRIC.value: 1,
    ArtifactClass.EVIDENCE.value: 0,
}
_SOURCE_BEARING = (
    ArtifactClass.PROTECTED_MAPPING.value,
    ArtifactClass.DIAGNOSTIC.value,
    ArtifactClass.DATA.value,
)


def _merge_artifact_entries(existing: Dict[str, Any], new: Dict[str, Any]) -> None:
    """Merge ``new`` into ``existing`` monotonically (in place).

    * ``contains_source_values`` is the OR of both;
    * an explicitly declared class is never overridden by one that was only inferred from
      the legacy category; between explicit classes the more protective one is kept
      (protected_mapping > diagnostic > data), and with source values never an
      aggregate_metric or evidence;
    * ``not_for_release`` is the OR of every restriction seen: it is never cleared because
      another class (a protected mapping, say) won;
    * ``task_product`` membership is the OR of both;
    * tags are united and missing metadata keys are added.
    """
    contains = bool(existing.get("contains_source_values")) or bool(
        new.get("contains_source_values")
    )
    candidates = [existing, new]
    explicit = [c for c in candidates if c.get("class_explicit")]
    if contains:
        pool = [c["artifact_class"] for c in explicit if c["artifact_class"] in _SOURCE_BEARING]
        if not pool and not explicit:
            pool = [
                c["artifact_class"] for c in candidates if c["artifact_class"] in _SOURCE_BEARING
            ]
        cls = max(pool, key=_PROTECTION_RANK.get) if pool else ArtifactClass.DIAGNOSTIC.value
    else:
        pool = [c["artifact_class"] for c in explicit] or [
            c["artifact_class"] for c in candidates
        ]
        cls = max(pool, key=_PROTECTION_RANK.get)
    restricted = bool(existing.get("not_for_release")) or bool(new.get("not_for_release"))
    existing["artifact_class"] = cls
    existing["contains_source_values"] = contains
    existing["class_explicit"] = bool(existing.get("class_explicit") or new.get("class_explicit"))
    existing["classified_explicitly"] = existing["class_explicit"] or bool(
        existing.get("classified_explicitly") or new.get("classified_explicitly")
    )
    existing["not_for_release"] = restricted or (
        contains
        and cls not in (ArtifactClass.DATA.value, ArtifactClass.PROTECTED_MAPPING.value)
    )
    existing["task_product"] = bool(existing.get("task_product") or new.get("task_product"))
    for tag in new.get("tags") or []:
        if tag not in existing.setdefault("tags", []):
            existing["tags"].append(tag)
    for k, v in (new.get("metadata") or {}).items():
        existing.setdefault("metadata", {}).setdefault(k, v)


class ArtifactGroup:
    """
    Group of related artifacts in a task report.

    Allows organizing artifacts into logical groups for better structure.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize an artifact group.

        Parameters
        ----------
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

        Parameters
        ----------
            artifact: Artifact information dictionary
        """
        self.artifacts.append(artifact)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the group to a dictionary for serialization.

        Returns
        -------
            Dictionary representation of the group
        """
        return {
            "name": self.name,
            "description": self.description,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "count": len(self.artifacts),
        }


class TaskReporter:
    """
    Task reporter for generating and managing task reports.

    Tracks operations, artifacts, and metrics, and generates
    JSON reports for task execution. Integrates with the progress manager
    for coordinated progress tracking during report generation.
    """

    def __init__(
        self,
        task_id: str,
        task_type: str,
        description: str,
        report_path: Union[str, Path],
        progress_manager: Optional[Any] = None,
    ):
        """
        Initialize the task reporter.

        Parameters
        ----------
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

        Returns
        -------
            Dictionary with system information
        """
        return {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
            "machine": socket.gethostname(),
            "cpu_count": os.cpu_count() or 0,
            "pamola_version": os.environ.get("PAMOLA_VERSION", "unknown"),
        }

    def _get_current_memory_usage(self) -> float:
        """
        Get current memory usage of the process.

        Returns
        -------
            Memory usage in MB
        """
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except (ImportError, Exception):
            return 0.0

    def add_operation(
        self, name: str, status: str = "success", details: Dict[str, Any] = None
    ):
        """
        Add an operation to the report.

        Parameters
        ----------
            name: Name of the operation
            status: Status of the operation (success, warning, error)
            details: Additional details about the operation
        """
        details = details or {}
        operation = {
            "operation": name,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details,
        }

        self.operations.append(operation)

        # Track errors and warnings
        if status == "error":
            self.errors.append(
                {
                    "operation": name,
                    "timestamp": operation["timestamp"],
                    "message": details.get("error_message", "Unknown error"),
                    "details": details,
                }
            )
        elif status == "warning":
            self.warnings.append(
                {
                    "operation": name,
                    "timestamp": operation["timestamp"],
                    "message": details.get("warning_message", "Unknown warning"),
                    "details": details,
                }
            )

        # Update memory usage
        current_memory = self._get_current_memory_usage()
        if current_memory > self.peak_memory_usage:
            self.peak_memory_usage = current_memory

        # Log operation addition via progress manager if available
        if self.progress_manager:
            self.progress_manager.log_info(
                f"Added operation: {name} (status: {status})"
            )

    def add_artifact(
        self,
        artifact_type: str,
        path: Union[str, Path],
        description: str = "",
        category: str = "output",
        tags: Optional[List[str]] = None,
        group_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_class: Optional[ArtifactClass] = None,
        contains_source_values: Optional[bool] = None,
        task_product: bool = False,
    ):
        """
        Add an artifact to the report.

        Parameters
        ----------
            artifact_type: Type of the artifact (e.g., "json", "csv", "png")
            path: Path to the artifact
            description: Description of the artifact
            category: Category of the artifact (e.g., "output", "metric", "visualization")
            tags: Tags for categorizing the artifact
            group_name: Name of the group to add this artifact to
            metadata: Additional metadata for the artifact
            artifact_class: Class of the artifact (data, protected_mapping,
                aggregate_metric, diagnostic, evidence); derived from ``category`` when omitted
        """
        # Use progress manager for artifact addition if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                name="add_artifact",
                total=2,  # validate + add
                description=f"Adding artifact: {artifact_type}",
                leave=False,
            ) as progress:
                try:
                    # Validate path security
                    if not validate_path_security(path):
                        progress.update(1, {"status": "validation_failed"})
                        raise ReportingError(f"Insecure artifact path: {path}")

                    progress.update(1, {"status": "validated"})

                    # Process artifact (use helper method to avoid code duplication)
                    self._process_artifact(
                        artifact_type,
                        path,
                        description,
                        category,
                        tags or [],
                        group_name,
                        metadata or {},
                        artifact_class,
                        contains_source_values,
                        task_product,
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
                artifact_type,
                path,
                description,
                category,
                tags or [],
                group_name,
                metadata or {},
                artifact_class,
                contains_source_values,
                task_product,
            )

    def _process_artifact(
        self,
        artifact_type: str,
        path: Union[str, Path],
        description: str,
        category: str,
        tags: List[str],
        group_name: Optional[str],
        metadata: Dict[str, Any],
        artifact_class: Optional[ArtifactClass] = None,
        contains_source_values: Optional[bool] = None,
        task_product: bool = False,
    ):
        """
        Process and add an artifact to the report (internal helper method).

        Parameters
        ----------
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
            **self._classification(category, artifact_class, contains_source_values),
            "task_product": bool(task_product),
            "tags": tags,
            "size_bytes": size_bytes,
            "timestamp": timestamp or datetime.now().isoformat(),
            "metadata": metadata,
        }

        # One entry per normalized path: a second registration merges, and never
        # replaces a classification that was already assigned.
        matches = self._find_matches(path_str)
        if matches:
            first = matches[0]
            for other in matches[1:]:  # earlier aliases that only now turn out to be one file
                _merge_artifact_entries(first, other)
                self._replace_entry(other, first)
            _merge_artifact_entries(first, artifact)
            if group_name:
                if group_name not in self.artifact_groups:
                    self.add_artifact_group(group_name)
                self._join_group(self.artifact_groups[group_name], first)
            self._build_index()
            return
        self.artifacts.append(artifact)
        self._index_entry(artifact)

        # Add to group if specified
        if group_name:
            if group_name not in self.artifact_groups:
                self.add_artifact_group(group_name)
            self.artifact_groups[group_name].add_artifact(artifact)

    @staticmethod
    def _classification(category, artifact_class, contains_source_values) -> Dict[str, Any]:
        """artifact_class / contains_source_values / not_for_release for one entry."""
        explicit = artifact_class is not None
        cls = ArtifactClass(artifact_class) if explicit else class_for_category(category)
        has_values = (
            bool(contains_source_values)
            if contains_source_values is not None
            else str(category).lower() == "dictionary"
        )
        cls = check_class_and_content(cls, has_values)
        return {
            "artifact_class": cls.value,
            "contains_source_values": has_values,
            "not_for_release": has_values
            and cls not in (ArtifactClass.DATA, ArtifactClass.PROTECTED_MAPPING),
            "class_explicit": explicit,
            "classified_explicitly": explicit or contains_source_values is not None,
        }

    # ---- alias index: O(1) per lookup, O(n) per rebuild --------------------------------
    def _index_entry(self, entry: Dict[str, Any]) -> None:
        idx = self.__dict__.setdefault("_alias_index", {})
        path = entry.get("path", "")
        idx[("path", _normalized_path(path))] = entry
        ident = _identity_key(path)
        idx[ident] = entry
        if ident[0] == "path":  # not on disk yet: its identity is unknown until it is
            self.__dict__.setdefault("_pending", []).append(entry)
        self._indexed_count = len(self.artifacts)

    def _refresh_pending(self) -> None:
        """Entries registered before their file existed get their file identity now; two
        entries that turn out to be one file are merged."""
        pending = self.__dict__.get("_pending")
        if not pending:
            return
        idx = self._alias_index
        still = []
        for entry in list(pending):
            if not any(entry is e for e in self.artifacts):
                continue
            ident = _identity_key(entry.get("path", ""))
            if ident[0] == "path":
                still.append(entry)
                continue
            other = idx.get(ident)
            if other is not None and other is not entry:
                _merge_artifact_entries(other, entry)
                self._replace_entry(entry, other)
                self._indexed_count = len(self.artifacts)
            else:
                idx[ident] = entry
        self._pending = still

    def _build_index(self) -> None:
        self._alias_index = {}
        self._pending = []
        for entry in self.artifacts:
            self._index_entry(entry)
        self._indexed_count = len(self.artifacts)

    def _find_matches(self, path: Any) -> list:
        """Entries naming the same file as ``path`` (usually zero or one)."""
        if getattr(self, "_indexed_count", -1) != len(self.artifacts):
            self.reconcile_aliases()  # entries were added or removed without the index
        self._refresh_pending()
        idx = self.__dict__.get("_alias_index", {})
        found = []
        for key in (("path", _normalized_path(path)), _identity_key(path)):
            entry = idx.get(key)
            if entry is not None and not any(entry is f for f in found):
                found.append(entry)
        return found

    @staticmethod
    def _join_group(group: "ArtifactGroup", entry: Dict[str, Any]) -> None:
        if not any(e is entry for e in group.artifacts):
            group.artifacts.append(entry)

    def _replace_entry(self, old: Dict[str, Any], keep: Dict[str, Any]) -> None:
        """Drop ``old`` and move every group membership it had to ``keep``."""
        self.artifacts[:] = [e for e in self.artifacts if e is not old]
        for group in self.artifact_groups.values():
            if any(e is old for e in group.artifacts):
                group.artifacts[:] = [e for e in group.artifacts if e is not old]
                self._join_group(group, keep)

    def reconcile_aliases(self) -> None:
        """Merge entries that name one file (files created after registration, hard links,
        case aliases on a case-insensitive filesystem). O(n): one ``stat`` per entry and a
        dictionary keyed by file identity. Runs before the report and the manifest are
        serialized; group memberships, tags and metadata are united."""
        owner: Dict[Any, Dict[str, Any]] = {}
        for entry in list(self.artifacts):
            keep = None
            for key in (("path", _normalized_path(entry.get("path", ""))), _identity_key(entry.get("path", ""))):
                found = owner.get(key)
                if found is not None and found is not entry:
                    keep = found
                    break
            if keep is None:
                owner[("path", _normalized_path(entry.get("path", "")))] = entry
                owner[_identity_key(entry.get("path", ""))] = entry
                continue
            _merge_artifact_entries(keep, entry)
            self._replace_entry(entry, keep)
            owner[("path", _normalized_path(entry.get("path", "")))] = keep
            owner[_identity_key(entry.get("path", ""))] = keep
        self._build_index()

    def add_artifact_group(self, name: str, description: str = "") -> ArtifactGroup:
        """
        Add or get an artifact group.

        Parameters
        ----------
            name: Name of the group
            description: Description of the group

        Returns
        -------
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

        Parameters
        ----------
            name: Name of the group

        Returns
        -------
            The artifact group if it exists, None otherwise
        """
        return self.artifact_groups.get(name)

    def get_artifacts_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts with a specific tag.

        Parameters
        ----------
            tag: Tag to filter by

        Returns
        -------
            List of artifacts with the specified tag
        """
        return [a for a in self.artifacts if tag in a.get("tags", [])]

    def get_artifacts_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts in a specific category.

        Parameters
        ----------
            category: Category to filter by

        Returns
        -------
            List of artifacts in the specified category
        """
        return [a for a in self.artifacts if a.get("category") == category]

    def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts of a specific type.

        Parameters
        ----------
            artifact_type: Type to filter by

        Returns
        -------
            List of artifacts of the specified type
        """
        return [a for a in self.artifacts if a.get("type") == artifact_type]

    def add_metric(self, name: str, value: Any):
        """
        Add a metric to the report.

        Parameters
        ----------
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

        Parameters
        ----------
            category: Category for the metric
            name: Name of the metric
            value: Value of the metric
        """
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category][name] = value

        # Log nested metric addition via progress manager if available
        if self.progress_manager:
            self.progress_manager.log_debug(
                f"Added metric: {category}.{name} = {value}"
            )

    def add_task_summary(
        self,
        success: bool,
        execution_time: float = None,
        metrics: Dict[str, Any] = None,
        error_info: Dict[str, Any] = None,
        encryption: Dict[str, Any] = None,
    ):
        """
        Add task summary to the report.

        Parameters
        ----------
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
                leave=False,
            ) as progress:
                self._add_task_summary_internal(
                    success, execution_time, metrics, error_info, encryption, progress
                )
        else:
            # Process without progress tracking
            self._add_task_summary_internal(
                success, execution_time, metrics, error_info, encryption
            )

    def _add_task_summary_internal(
        self,
        success: bool,
        execution_time: float = None,
        metrics: Dict[str, Any] = None,
        error_info: Dict[str, Any] = None,
        encryption: Dict[str, Any] = None,
        progress=None,
    ):
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
            self.errors.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": error_info.get("type", "unknown"),
                    "message": error_info.get("message", "Unknown error"),
                    "details": error_info,
                }
            )

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

        Returns
        -------
            Report as a dictionary
        """
        self.reconcile_aliases()
        # Use progress manager for report generation if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                name="generate_report", total=1, description="Generating task report"
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
            "script_path": os.path.abspath(os.path.realpath(__file__)),
            "system_info": self.system_info,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_seconds": self.execution_time,
            "status": self.status,
            "operations": self.operations,
            "artifacts": self.artifacts,
            "artifact_groups": {
                name: group.to_dict() for name, group in self.artifact_groups.items()
            },
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "memory_usage_mb": round(self.peak_memory_usage, 2),
        }

        return report

    def save_report(self) -> Path:
        """
        Generate and save the report to disk.

        Returns
        -------
            Path to the saved report

        Raises
        ------
            ReportingError: If saving fails
        """
        # Use progress manager for report saving if available
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                name="save_report",
                total=3,  # generate report, ensure directory, write to disk
                description="Saving task report",
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

                        self.progress_manager.log_info(
                            f"Report saved to {self.report_path}"
                        )
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
                description="Finalizing task report",
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
                            "traceback": traceback.format_exc(),
                        }
                        self.add_operation(
                            name=f"Exception: {exc_type.__name__}",
                            status="error",
                            details=error_info,
                        )

                    # Save report
                    try:
                        self.save_report()
                        progress.update(1, {"status": "report_saved"})
                    except Exception as e:
                        progress.update(1, {"status": "save_error", "error": str(e)})
                        self.progress_manager.log_error(
                            f"Failed to save report: {str(e)}"
                        )
                except Exception as e:
                    if progress:
                        progress.update(1, {"status": "error", "error": str(e)})
                    self.progress_manager.log_error(
                        f"Error during report finalization: {str(e)}"
                    )
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
                    "traceback": traceback.format_exc(),
                }
                self.add_operation(
                    name=f"Exception: {exc_type.__name__}",
                    status="error",
                    details=error_info,
                )

            # Save report
            try:
                self.save_report()
            except Exception as e:
                import logging

                logging.error(f"Failed to save report: {str(e)}")

        # Don't suppress exceptions
        return False