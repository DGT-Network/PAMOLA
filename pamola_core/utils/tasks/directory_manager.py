"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Directory Manager
Description: Task directory structure management and path resolution
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for managing task directory structures,
ensuring proper path validation, and generating standardized paths for
task artifacts.

Key features:
- Creation and validation of task directory structures
- Standardized artifact path generation
- Path security validation
- Temporary directory management
- Support for timestamped filenames
- Integration with progress tracking
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Protocol

from pamola_core.utils.io import ensure_directory
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError

# Default directory suffixes if not specified in configuration
DEFAULT_DIRECTORY_SUFFIXES = [
    "input", "output", "temp", "logs", "dictionaries", "visualizations", "metrics"
]


# Protocol for task configuration interface
class TaskConfigProtocol(Protocol):
    """Protocol defining the required interface for task configuration."""
    task_id: str
    project_root: Path

    def get_task_dir(self) -> Path:
        """Get the task directory path."""
        ...

    def get_reports_dir(self) -> Path:
        """Get the reports directory path."""
        ...


# Forward declaration for type hinting
class TaskProgressManager(Protocol):
    """Protocol defining the required interface for progress management."""

    def create_operation_context(self, name: str, total: int, description: Optional[str] = None,
                                 unit: str = "items", leave: bool = False) -> Any:
        """Create a context manager for tracking operation progress."""
        ...

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        ...

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        ...

    def log_error(self, message: str) -> None:
        """Log an error message."""
        ...

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        ...


class DirectoryManagerError(Exception):
    """Base exception for directory manager errors."""
    pass


class PathValidationError(DirectoryManagerError):
    """Exception raised when a path fails validation."""
    pass


class DirectoryCreationError(DirectoryManagerError):
    """Exception raised when directory creation fails."""
    pass


class TaskDirectoryManager:
    """
    Manager for task directory structures and path resolution.

    This class handles the creation and management of standard task directory
    structures, provides path resolution for artifacts, and ensures proper
    path security validation.
    """

    def __init__(self,
                 task_config: Any,
                 logger: Optional[logging.Logger] = None,
                 progress_manager: Optional[TaskProgressManager] = None):
        """
        Initialize the directory manager with task configuration.

        Args:
            task_config: Task configuration object containing directory information.
                         Must provide task_id, project_root, and get_task_dir() method.
            logger: Logger for directory operations (optional)
            progress_manager: Progress manager for tracking directory operations (optional)
        """
        self.config = task_config
        self.logger = logger or logging.getLogger(__name__)
        self.progress_manager = progress_manager

        # Store references to key directories
        self.task_id = getattr(task_config, 'task_id', 'unknown')
        self.project_root = getattr(task_config, 'project_root', Path.cwd())
        self.task_dir = self._resolve_task_dir()

        # Store reference to log_directory if available, for centralized logs/checkpoints/temp
        self.log_directory = getattr(task_config, 'log_directory', None)

        # Get directory suffixes from configuration or use defaults
        self.directory_suffixes = getattr(
            task_config, 'task_dir_suffixes', DEFAULT_DIRECTORY_SUFFIXES
        )

        # Initialize directories dictionary
        self.directories = {}

        # Track created directories for cleanup
        self._created_directories = set()
        self._initialized = False

    def _resolve_task_dir(self) -> Path:
        """
        Resolve the task directory path from configuration.

        Returns:
            Path to the task directory

        Raises:
            DirectoryManagerError: If task directory cannot be resolved
        """
        try:
            # Try to get task directory from configuration using API
            if hasattr(self.config, 'get_task_dir'):
                task_dir = self.config.get_task_dir()
                if self.progress_manager:
                    self.progress_manager.log_debug(f"Resolved task directory using API: {task_dir}")
                else:
                    self.logger.debug(f"Resolved task directory using API: {task_dir}")
                return task_dir

            # Fall back to directly accessing task_dir attribute
            if hasattr(self.config, 'task_dir'):
                task_dir = self.config.task_dir
                if self.progress_manager:
                    self.progress_manager.log_debug(f"Resolved task directory from attribute: {task_dir}")
                else:
                    self.logger.debug(f"Resolved task directory from attribute: {task_dir}")
                return task_dir

            # Last resort: construct from processed_data_path
            if hasattr(self.config, 'processed_data_path'):
                task_dir = self.config.processed_data_path / self.task_id
                if self.progress_manager:
                    self.progress_manager.log_debug(f"Resolved task directory from processed_data_path: {task_dir}")
                else:
                    self.logger.debug(f"Resolved task directory from processed_data_path: {task_dir}")
                return task_dir

            # If all else fails, construct from project root
            task_dir = self.project_root / "DATA" / "processed" / self.task_id

            if self.progress_manager:
                self.progress_manager.log_warning(
                    f"Could not resolve task directory from configuration, using default: {task_dir}"
                )
            else:
                self.logger.warning(
                    f"Could not resolve task directory from configuration, using default: {task_dir}"
                )
            return task_dir

        except Exception as e:
            error_msg = f"Error resolving task directory: {str(e)}"
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            else:
                self.logger.error(error_msg, exc_info=True)
            raise DirectoryManagerError(error_msg) from e

    def ensure_directories(self) -> Dict[str, Path]:
        """
        Create and validate all required task directories.

        Creates the task directory and all standard subdirectories based on
        the configured directory suffixes.

        Returns:
            Dictionary mapping directory types to their paths

        Raises:
            DirectoryCreationError: If directory creation fails
            PathValidationError: If a path fails security validation
        """
        # Use progress manager if available
        if self.progress_manager:
            return self._ensure_directories_with_progress()
        else:
            return self._ensure_directories_standard()

    def _ensure_directories_with_progress(self) -> Dict[str, Path]:
        """
        Create and validate directories with progress tracking.

        This method creates the task directory structure with visual progress
        tracking, ensuring all paths are secure and properly created.

        Returns:
            Dictionary mapping directory types to their paths

        Raises:
            DirectoryCreationError: If directory creation fails
            PathValidationError: If a path fails security validation
        """
        # Calculate total number of directories to create
        total_dirs = 1 + len(self.directory_suffixes)  # task dir + all suffixes
        # Add extra directories for centralized structures if log_directory is available
        if self.log_directory:
            total_dirs += 2  # Add centralized temp and logs directories

        # Create a progress context
        with self.progress_manager.create_operation_context(
                name="create_directories",
                total=total_dirs,
                description="Creating task directories",
                unit="directories",
                leave=False
        ) as progress:
            try:
                # Validate task directory path
                if not validate_path_security(self.task_dir):
                    error_msg = f"Task directory path failed security validation: {self.task_dir}"
                    self.progress_manager.log_error(error_msg)
                    raise PathValidationError(error_msg)

                # Create task directory if it doesn't exist
                ensure_directory(self.task_dir)
                self._created_directories.add(self.task_dir)
                self.progress_manager.log_debug(f"Created task directory: {self.task_dir}")

                # Update progress
                progress.update(1)

                # Create standard directories
                self.directories = {
                    "task": self.task_dir,
                }

                # Create directory for each suffix
                for suffix in self.directory_suffixes:
                    dir_path = self.task_dir / suffix

                    # Validate path security
                    if not validate_path_security(dir_path):
                        error_msg = f"Directory path failed security validation: {dir_path}"
                        self.progress_manager.log_error(error_msg)
                        raise PathValidationError(error_msg)

                    # Create directory
                    ensure_directory(dir_path)
                    self._created_directories.add(dir_path)
                    self.progress_manager.log_debug(f"Created {suffix} directory: {dir_path}")

                    # Add to directories dictionary
                    self.directories[suffix] = dir_path

                    # Update progress
                    progress.update(1)

                # Use centralized directories for temp, logs and checkpoints if log_directory is available
                if self.log_directory:
                    # Create centralized temp directory
                    central_temp_dir = self.log_directory / "temp" / self.task_id
                    ensure_directory(central_temp_dir)
                    self._created_directories.add(central_temp_dir)
                    self.directories["temp"] = central_temp_dir
                    self.progress_manager.log_debug(f"Using centralized temp directory: {central_temp_dir}")
                    progress.update(1)

                    # Create centralized logs directory
                    central_logs_dir = self.log_directory / "logs" / self.task_id
                    ensure_directory(central_logs_dir)
                    self._created_directories.add(central_logs_dir)
                    self.directories["logs"] = central_logs_dir
                    self.progress_manager.log_debug(f"Using centralized logs directory: {central_logs_dir}")
                    progress.update(1)

                # Add 'reports' directory at data repository level if available
                if hasattr(self.config, 'get_reports_dir'):
                    reports_dir = self.config.get_reports_dir()
                    ensure_directory(reports_dir)
                    self.directories["reports"] = reports_dir
                    self.progress_manager.log_debug(f"Using reports directory: {reports_dir}")

                # Mark as initialized
                self._initialized = True

                return self.directories

            except PathSecurityError as e:
                error_msg = f"Path security error: {str(e)}"
                self.progress_manager.log_error(error_msg)
                raise PathValidationError(error_msg) from e

            except Exception as e:
                error_msg = f"Error creating directories: {str(e)}"
                self.progress_manager.log_error(error_msg)
                raise DirectoryCreationError(error_msg) from e

    def _ensure_directories_standard(self) -> Dict[str, Path]:
        """
        Create and validate directories without progress tracking (standard version).

        This method creates the task directory structure without visual progress tracking,
        ensuring all paths are secure and properly created.

        Returns:
            Dictionary mapping directory types to their paths

        Raises:
            DirectoryCreationError: If directory creation fails
            PathValidationError: If a path fails security validation
        """
        try:
            # Validate task directory path
            if not validate_path_security(self.task_dir):
                error_msg = f"Task directory path failed security validation: {self.task_dir}"
                self.logger.error(error_msg)
                raise PathValidationError(error_msg)

            # Create task directory if it doesn't exist
            ensure_directory(self.task_dir)
            self._created_directories.add(self.task_dir)
            self.logger.debug(f"Created task directory: {self.task_dir}")

            # Create standard directories
            self.directories = {
                "task": self.task_dir,
            }

            # Create directory for each suffix
            for suffix in self.directory_suffixes:
                dir_path = self.task_dir / suffix

                # Validate path security
                if not validate_path_security(dir_path):
                    error_msg = f"Directory path failed security validation: {dir_path}"
                    self.logger.error(error_msg)
                    raise PathValidationError(error_msg)

                # Create directory
                ensure_directory(dir_path)
                self._created_directories.add(dir_path)
                self.logger.debug(f"Created {suffix} directory: {dir_path}")

                # Add to directories dictionary
                self.directories[suffix] = dir_path

            # Use centralized directories for temp, logs and checkpoints if log_directory is available
            if self.log_directory:
                # Create centralized temp directory
                central_temp_dir = self.log_directory / "temp" / self.task_id
                ensure_directory(central_temp_dir)
                self._created_directories.add(central_temp_dir)
                self.directories["temp"] = central_temp_dir
                self.logger.debug(f"Using centralized temp directory: {central_temp_dir}")

                # Create centralized logs directory
                central_logs_dir = self.log_directory / "logs" / self.task_id
                ensure_directory(central_logs_dir)
                self._created_directories.add(central_logs_dir)
                self.directories["logs"] = central_logs_dir
                self.logger.debug(f"Using centralized logs directory: {central_logs_dir}")

            # Add 'reports' directory at data repository level if available
            if hasattr(self.config, 'get_reports_dir'):
                reports_dir = self.config.get_reports_dir()
                ensure_directory(reports_dir)
                self.directories["reports"] = reports_dir
                self.logger.debug(f"Using reports directory: {reports_dir}")

            # Mark as initialized
            self._initialized = True

            return self.directories

        except PathSecurityError as e:
            error_msg = f"Path security error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise PathValidationError(error_msg) from e

        except Exception as e:
            error_msg = f"Error creating directories: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DirectoryCreationError(error_msg) from e

    def get_directory(self, dir_type: str) -> Path:
        """
        Get path to a specific directory type.

        Args:
            dir_type: Directory type (e.g., "input", "output", "temp")

        Returns:
            Path to the requested directory

        Raises:
            DirectoryManagerError: If directory type is unknown or not created
        """
        # Check if directories have been initialized
        if not self._initialized:
            self.ensure_directories()

        # Check if directory type exists
        if dir_type not in self.directories:
            error_msg = f"Unknown directory type: {dir_type}"
            if self.progress_manager:
                self.progress_manager.log_warning(error_msg)
            else:
                self.logger.warning(error_msg)
            raise DirectoryManagerError(error_msg)

        return self.directories[dir_type]

    def get_artifact_path(self,
                          artifact_name: str,
                          artifact_type: str = "json",
                          subdir: str = "output",
                          include_timestamp: bool = True) -> Path:
        """
        Generate standardized path for an artifact.

        Args:
            artifact_name: Name of the artifact
            artifact_type: Type/extension of the artifact (without dot)
            subdir: Subdirectory for the artifact (e.g., "output", "visualizations")
            include_timestamp: Whether to include a timestamp in the filename

        Returns:
            Path to the artifact

        Raises:
            PathValidationError: If artifact path fails validation
            DirectoryManagerError: If subdirectory does not exist
        """
        try:
            # Get subdirectory path
            artifact_dir = self.get_directory(subdir)
        except DirectoryManagerError:
            # Try creating the directory if it doesn't exist
            artifact_dir = self.task_dir / subdir
            ensure_directory(artifact_dir)
            self.directories[subdir] = artifact_dir

        # Ensure artifact type does not start with a dot
        artifact_type = artifact_type.lstrip('.')

        # Generate filename
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{artifact_name}_{timestamp}.{artifact_type}"
        else:
            filename = f"{artifact_name}.{artifact_type}"

        # Construct full path
        artifact_path = artifact_dir / filename

        # Validate full path for security
        if not validate_path_security(artifact_path):
            error_msg = f"Artifact path failed security validation: {artifact_path}"
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            else:
                self.logger.error(error_msg)
            raise PathValidationError(error_msg)

        if self.progress_manager:
            self.progress_manager.log_debug(f"Generated artifact path: {artifact_path}")
        else:
            self.logger.debug(f"Generated artifact path: {artifact_path}")
        return artifact_path

    def clean_temp_directory(self) -> bool:
        """
        Clean temporary files and directories.

        Removes all files and subdirectories in the task's temp directory.

        Returns:
            True if cleaning was successful or no cleanup needed,
            False if errors occurred during cleanup
        """
        try:
            # Get temp directory
            temp_dir = self.get_directory("temp")

            # Check if directory exists
            if not temp_dir.exists():
                if self.progress_manager:
                    self.progress_manager.log_warning(f"Temp directory does not exist: {temp_dir}")
                else:
                    self.logger.warning(f"Temp directory does not exist: {temp_dir}")
                return True

            # Use progress manager if available
            if self.progress_manager:
                return self._clean_temp_directory_with_progress(temp_dir)
            else:
                return self._clean_temp_directory_standard(temp_dir)

        except Exception as e:
            if self.progress_manager:
                self.progress_manager.log_error(f"Error accessing temp directory")
            else:
                self.logger.error(f"Error accessing temp directory", exc_info=True)
            return False

    def _clean_temp_directory_with_progress(self, temp_dir: Path) -> bool:
        """
        Clean temporary directory with progress tracking.

        Args:
            temp_dir: Path to the temporary directory

        Returns:
            True if successful, False if errors occurred
        """
        try:
            # Count items to clean
            items = list(temp_dir.iterdir())
            if not items:
                self.progress_manager.log_info(f"Temp directory is already empty: {temp_dir}")
                return True

            with self.progress_manager.create_operation_context(
                    name="clean_temp_directory",
                    total=len(items),
                    description="Cleaning temporary files",
                    unit="items",
                    leave=False
            ) as progress:
                items_removed = 0
                errors = 0

                for item in items:
                    try:
                        if item.is_file():
                            item.unlink()
                            items_removed += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            items_removed += 1

                        # Update progress
                        progress.update(1, {"status": "success"})

                    except Exception as e:
                        self.progress_manager.log_warning(f"Error removing item {item}: {str(e)}")
                        errors += 1
                        # Update progress with error status
                        progress.update(1, {"status": "error"})

                if errors > 0:
                    self.progress_manager.log_warning(
                        f"Cleaned {items_removed} items from temp directory with {errors} errors: {temp_dir}")
                    return False
                else:
                    self.progress_manager.log_info(
                        f"Successfully cleaned {items_removed} items from temp directory: {temp_dir}")
                    return True

        except Exception as e:
            self.progress_manager.log_error(f"Error cleaning temp directory: {str(e)}")
            return False

    def _clean_temp_directory_standard(self, temp_dir: Path) -> bool:
        """
        Clean temporary directory without progress tracking.

        Args:
            temp_dir: Path to the temporary directory

        Returns:
            True if successful, False if errors occurred
        """
        # Remove all files and subdirectories
        items_removed = 0
        errors = 0

        for item in temp_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    items_removed += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    items_removed += 1
            except Exception as e:
                self.logger.warning(f"Error removing item {item}: {str(e)}")
                errors += 1

        if errors > 0:
            self.logger.warning(
                f"Cleaned {items_removed} items from temp directory with {errors} errors: {temp_dir}")
            return False
        else:
            self.logger.info(f"Successfully cleaned {items_removed} items from temp directory: {temp_dir}")
            return True

    def get_timestamped_filename(self, base_name: str, extension: str = "json") -> str:
        """
        Generate a timestamped filename.

        Args:
            base_name: Base name for the file
            extension: File extension (without dot)

        Returns:
            Timestamped filename
        """
        # Ensure extension does not start with a dot
        extension = extension.lstrip('.')

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct filename
        return f"{base_name}_{timestamp}.{extension}"

    def validate_directory_structure(self) -> Dict[str, bool]:
        """
        Validate the task directory structure.

        Checks that all required directories exist and are accessible.

        Returns:
            Dictionary mapping directory types to validation results
        """
        validation_results = {}

        # Check task directory
        validation_results["task"] = self.task_dir.exists() and self.task_dir.is_dir()

        # Check each standard directory
        for suffix in self.directory_suffixes:
            dir_path = self.task_dir / suffix
            validation_results[suffix] = dir_path.exists() and dir_path.is_dir()

        return validation_results

    def list_artifacts(self, subdir: str = "output", pattern: str = "*") -> List[Path]:
        """
        List artifacts in a specific subdirectory.

        Args:
            subdir: Subdirectory to search (e.g., "output", "visualizations")
            pattern: Glob pattern for filtering files

        Returns:
            List of paths to matching artifacts

        Raises:
            DirectoryManagerError: If subdirectory does not exist
        """
        try:
            # Get subdirectory path
            artifact_dir = self.get_directory(subdir)

            # Check if directory exists
            if not artifact_dir.exists():
                return []

            # Find matching files
            return sorted(artifact_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        except Exception as e:
            if self.progress_manager:
                self.progress_manager.log_error(f"Error listing artifacts in {subdir}")
            else:
                self.logger.error(f"Error listing artifacts in {subdir}", exc_info=True)
            return []

    def import_external_file(self,
                             source_path: Union[str, Path],
                             subdir: str = "input",
                             new_name: Optional[str] = None) -> Path:
        """
        Import an external file into the task directory structure.

        Args:
            source_path: Path to the source file
            subdir: Target subdirectory (e.g., "input", "dictionaries")
            new_name: New name for the file (optional)

        Returns:
            Path to the imported file

        Raises:
            PathValidationError: If source path fails security validation
            DirectoryManagerError: If import fails
        """
        # Convert to Path object if string
        source_path = Path(source_path) if isinstance(source_path, str) else source_path

        # Validate source path security
        if not validate_path_security(source_path):
            error_msg = f"Source path failed security validation: {source_path}"
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            else:
                self.logger.error(error_msg)
            raise PathValidationError(error_msg)

        # Check if source file exists
        if not source_path.exists() or not source_path.is_file():
            error_msg = f"Source file does not exist or is not a file: {source_path}"
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            else:
                self.logger.error(error_msg)
            raise DirectoryManagerError(error_msg)

        try:
            # Get target directory
            target_dir = self.get_directory(subdir)

            # Determine target filename
            if new_name:
                target_name = new_name
            else:
                target_name = source_path.name

            # Construct target path
            target_path = target_dir / target_name

            # Copy file
            shutil.copy2(source_path, target_path)
            if self.progress_manager:
                self.progress_manager.log_info(f"Imported file from {source_path} to {target_path}")
            else:
                self.logger.info(f"Imported file from {source_path} to {target_path}")

            return target_path

        except Exception as e:
            error_msg = f"Error importing file {source_path}"
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            else:
                self.logger.error(error_msg, exc_info=True)
            raise DirectoryManagerError(error_msg) from e

    def normalize_and_validate_path(self, path: Union[str, Path]) -> Path:
        """
        Normalize a path and validate its security.

        Args:
            path: Path to normalize and validate

        Returns:
            Normalized Path object

        Raises:
            PathValidationError: If the path fails security validation
        """
        # Convert to Path object if string
        path_obj = Path(path) if isinstance(path, str) else path

        # If not exists and not absolute, resolve relative to task directory
        if not path_obj.exists() and not path_obj.is_absolute():
            path_obj = self.task_dir / path_obj

        # Validate the path
        if not validate_path_security(path_obj):
            error_msg = f"Path failed security validation: {path_obj}"
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            else:
                self.logger.error(error_msg)
            raise PathValidationError(error_msg)

        return path_obj

    def cleanup(self) -> bool:
        """
        Explicitly clean up resources.

        This method should be called when the directory manager is no longer needed,
        especially in cases where the context manager cannot be used.

        Returns:
            True if cleanup was successful, False otherwise
        """
        # Clean temp directory if enabled in configuration
        if hasattr(self.config, 'clean_temp_on_exit') and self.config.clean_temp_on_exit:
            return self.clean_temp_directory()
        return True

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.

        Performs cleanup operations when exiting the context.
        """
        if hasattr(self.config, 'clean_temp_on_exit') and self.config.clean_temp_on_exit:
            self.clean_temp_directory()
        return False  # Don't suppress exceptions


# Utility function to create a directory manager
def create_directory_manager(task_config: Any,
                             logger: Optional[logging.Logger] = None,
                             progress_manager: Optional[TaskProgressManager] = None,
                             initialize: bool = True) -> TaskDirectoryManager:
    """
    Create a directory manager for a task.

    Args:
        task_config: Task configuration object
        logger: Logger for directory operations (optional)
        progress_manager: Progress manager for tracking directory operations (optional)
        initialize: Whether to initialize directories immediately

    Returns:
        TaskDirectoryManager instance

    Raises:
        DirectoryManagerError: If directory manager creation fails
    """
    try:
        # Create directory manager
        manager = TaskDirectoryManager(task_config, logger, progress_manager)

        # Initialize directories if requested
        if initialize:
            manager.ensure_directories()

        return manager

    except Exception as e:
        if logger:
            logger.error(f"Error creating directory manager", exc_info=True)
        elif progress_manager:
            progress_manager.log_error(f"Error creating directory manager: {str(e)}")
        else:
            logging.error(f"Error creating directory manager", exc_info=True)
        raise DirectoryManagerError(f"Failed to create directory manager: {str(e)}") from e