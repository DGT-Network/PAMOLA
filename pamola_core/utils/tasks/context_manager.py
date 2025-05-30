"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Context Manager
Description: Task execution state management with checkpoint support
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for managing task execution state,
enabling checkpoint creation, state persistence, and resumable execution.

Key features:
- Task state serialization and restoration
- Automatic checkpointing between operations
- Resumable execution from checkpoints
- Execution state verification
- Integration with execution log for cross-run persistence

This module integrates with execution_log.py to persist checkpoint information
across task runs, enabling resumability even after process termination.
It also coordinates with progress_manager.py when available to provide
visual feedback during checkpoint operations.
"""

import json
import logging
import os
import tempfile
import hashlib
import shutil
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from contextlib import nullcontext
import numpy as np

import filelock

from pamola_core.utils.io import ensure_directory, read_json
from pamola_core.utils.tasks.execution_log import (
    update_execution_record, find_latest_execution
)
from pamola_core.utils.tasks.path_security import validate_path_security

# Lock timeout for concurrent access (seconds)
DEFAULT_LOCK_TIMEOUT = 10
# Default maximum number of checkpoints to keep
DEFAULT_MAX_CHECKPOINTS = 10
# Default maximum state size in bytes before pruning older checkpoints (10 MB)
DEFAULT_MAX_STATE_SIZE = 10 * 1024 * 1024


class ContextManagerError(Exception):
    """Base exception for context manager errors."""
    pass


class CheckpointError(ContextManagerError):
    """Exception raised for checkpoint-related errors."""
    pass


class StateSerializationError(ContextManagerError):
    """Exception raised when state serialization fails."""
    pass


class StateRestorationError(ContextManagerError):
    """Exception raised when state restoration fails."""
    pass


class NullProgressTracker:
    """
    A no-op progress tracker that implements the required interface.
    Used when a real progress tracker is unavailable to prevent null reference errors.
    """

    def update(self, steps=1, postfix=None):
        """No-op update method."""
        pass

    def set_description(self, description):
        """No-op set_description method."""
        pass

    def set_postfix(self, postfix):
        """No-op set_postfix method."""
        pass

    def close(self, failed=False):
        """No-op close method."""
        pass


class TaskContextManager:
    """
    Manager for task execution state with checkpoint support.

    This class provides functionality for managing task execution state,
    creating checkpoints, and enabling resumable execution after interruptions.
    It integrates with the execution log for consistent checkpoint tracking
    across task runs.
    """

    def __init__(self,
                 task_id: str,
                 task_dir: Path,
                 log_directory: Optional[Path] = None,
                 max_state_size: int = DEFAULT_MAX_STATE_SIZE,
                 progress_manager: Optional[Any] = None):
        """
        Initialize the context manager.

        Args:
            task_id: ID of the task
            task_dir: Path to the task directory
            log_directory: Optional directory for logs and checkpoints (defaults to task_dir)
            max_state_size: Maximum state size in bytes before pruning older checkpoints
            progress_manager: Optional progress manager for visual feedback during operations
        """
        self.task_id = task_id
        self.task_dir = task_dir
        self.max_state_size = max_state_size
        self.progress_manager = progress_manager

        # Create a consistent logger with standard naming convention
        self.logger = logging.getLogger(f"task.{task_id}.context")

        # Validate task directory
        if not validate_path_security(task_dir):
            error_msg = f"Insecure task directory path: {task_dir}"
            self.logger.error(error_msg)
            raise ContextManagerError(error_msg)

        # Initialize checkpoint directory - use log_directory if provided
        if log_directory:
            # Create a task-specific checkpoint directory under the logs
            lock_file = log_directory
            self.checkpoint_dir = log_directory / "checkpoints" / task_id
            self.logger.debug(f"Using centralized checkpoint location: {self.checkpoint_dir}")
        else:
            lock_file = task_dir
            self.checkpoint_dir = task_dir / "checkpoints"
            self.logger.debug(f"Using task-local checkpoint location: {self.checkpoint_dir}")

        try:
            ensure_directory(self.checkpoint_dir)
            self.logger.debug(f"Initialized checkpoint directory: {self.checkpoint_dir}")
        except Exception as e:
            error_msg = f"Failed to initialize checkpoint directory: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ContextManagerError(error_msg) from e

        # Create lock file path for atomic operations
        self.lock_file = lock_file / "checkpoints.lock"

        # Track current execution state
        self.current_state: Dict[str, Any] = {
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "operation_index": -1,  # Start before first operation
            "operations_completed": [],
            "operations_failed": [],
            "metrics": {},
            "artifacts": []
        }

        # Track checkpoint history with timestamps
        self.checkpoints: List[Tuple[str, datetime]] = []
        self._load_checkpoint_history()

    def _calculate_config_hash(self, config_dict: Optional[Dict[str, Any]] = None) -> str:
        """
        Calculate a hash of the configuration dictionary.

        Args:
            config_dict: Configuration dictionary to hash. If None, uses an empty dict.

        Returns:
            String representation of the configuration hash
        """
        if config_dict is None:
            config_dict = {}

        # Convert config to stable string representation
        try:
            # Sort keys for stable serialization
            config_str = json.dumps(config_dict, sort_keys=True)
            # Calculate hash
            return hashlib.md5(config_str.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.warning(f"Error calculating config hash: {str(e)}")
            return "unknown"

    def get_checkpoint_dir(self) -> Path:
        """
        Get the checkpoint directory path.

        Returns:
            Path to the checkpoint directory
        """
        return self.checkpoint_dir

    def clear_checkpoints(self) -> bool:
        """
        Remove all checkpoints for this task.

        Acquires a lock. Attempts to remove the checkpoint directory recursively.
        If successful, recreates the directory and clears internal state.
        If removal fails for any reason (e.g., locked files, permission issues),
        logs an error and returns False, indicating the directory may not be empty.

        Returns:
            bool: True if clearing was completely successful, False otherwise.
        """
        # Main try block to handle lock timeout and other unexpected errors
        try:
            # Use lock to prevent race conditions
            with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
                checkpoint_dir = self.get_checkpoint_dir()

                # Count existing checkpoints for informational logging (optional)
                existing_count = 0
                if checkpoint_dir.exists():
                    # Using glob may be slow for very large directories,
                    # but sufficient for counting
                    try:
                        existing_count = len(list(checkpoint_dir.glob("*.json")))
                    except Exception as e:
                        # Log if counting failed, but continue with deletion attempt
                        self.logger.warning(f"Could not count existing checkpoints in {checkpoint_dir}: {e}")

                # --- Deletion logic: Try to delete everything. If failed, return False. ---
                try:
                    if checkpoint_dir.exists():
                        # Try to delete the entire directory recursively.
                        # If any error occurs during this process, shutil.rmtree will raise an exception.
                        # We don't use onerror here because ANY error during deletion means
                        # we didn't achieve the desired state (completely empty directory).
                        shutil.rmtree(checkpoint_dir)
                    # If rmtree completed successfully (or directory didn't exist), consider target state achieved.

                except Exception as e:
                    # If rmtree raised an exception (e.g., OSError 145, PermissionError, etc.)
                    # it means the directory was not completely cleared.
                    error_msg = f"Failed to completely remove checkpoint directory {checkpoint_dir} (might contain files): {str(e)}"
                    self.logger.error(error_msg, exc_info=True)

                    # Log through progress manager if available
                    if self.progress_manager:
                        self.progress_manager.log_error(error_msg)

                    # Return False because the complete clearing operation failed.
                    # Internal state self.checkpoints is NOT cleared
                    # because the file system state doesn't match it.
                    return False

                # --- Success logic: If we got here, deletion was successful. ---

                # Recreate an empty directory (if it was deleted).
                # This is important so the path exists for future checkpoints.
                # ensure_directory should handle the case where the directory already exists.
                try:
                    ensure_directory(checkpoint_dir)
                except Exception as e:
                    # If we can't even create the directory after successful deletion - that's also a problem.
                    error_msg = f"Failed to recreate checkpoint directory {checkpoint_dir} after successful removal: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    if self.progress_manager:
                        self.progress_manager.log_error(error_msg)
                    return False  # Operation still considered unsuccessful overall

                # Clear internal checkpoint list ONLY if directory deletion and recreation were successful.
                self.checkpoints = []

                # Log successful completion
                self.logger.info(
                    f"Successfully cleared checkpoints from {checkpoint_dir}. Initially found {existing_count}.")

                # Log through progress manager if available
                if self.progress_manager:
                    # Can adjust message if count wasn't obtained
                    info_msg = f"Successfully cleared checkpoints."
                    if existing_count > 0:  # Only if count was obtained
                        info_msg += f" (Initially {existing_count})"
                    self.progress_manager.log_info(info_msg)

                return True  # Operation completely successful

        # --- Exception handling outside the main deletion flow ---

        except filelock.Timeout:
            # This block remains the same as lock timeout is a specific error
            error_msg = f"Timeout waiting for lock when attempting to clear checkpoints"
            self.logger.error(error_msg)
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            return False

        except Exception as e:
            # This block catches any other unexpected errors that might have occurred
            # outside the specific shutil.rmtree call (e.g., issues with get_checkpoint_dir,
            # filelock, or something completely unexpected).
            error_msg = f"An unexpected error occurred during checkpoint clearing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)
            return False

    def is_checkpoint_valid(self, checkpoint_name: str, config_hash: Optional[str] = None,
                            task_version: Optional[str] = None) -> bool:
        """
        Check if a checkpoint is still valid based on configuration and task version.

        Args:
            checkpoint_name: Name of the checkpoint to validate
            config_hash: Optional hash of current configuration to compare with checkpoint
            task_version: Optional task version to compare with checkpoint

        Returns:
            bool: True if checkpoint is valid, False otherwise
        """
        try:
            # Load checkpoint state
            try:
                state = self.restore_execution_state(checkpoint_name)
                if not state:
                    self.logger.warning(f"Checkpoint {checkpoint_name} not found or empty")
                    return False
            except Exception as e:
                self.logger.warning(f"Could not restore checkpoint {checkpoint_name}: {str(e)}")
                return False

            # Check config hash if provided
            if config_hash and state.get("config_hash"):
                if state.get("config_hash") != config_hash:
                    self.logger.info(f"Checkpoint config hash mismatch: {state.get('config_hash')} != {config_hash}")
                    return False

            # Check task version if provided
            if task_version and state.get("task_version"):
                if state.get("task_version") != task_version:
                    self.logger.info(f"Checkpoint task version mismatch: {state.get('task_version')} != {task_version}")
                    return False

            # All checks passed
            return True

        except Exception as e:
            self.logger.warning(f"Error validating checkpoint {checkpoint_name}: {str(e)}")
            return False

    def _load_checkpoint_history(self) -> None:
        """
        Load existing checkpoint history.

        This method scans the checkpoint directory for existing checkpoints
        and builds a history of available checkpoints with creation timestamps.
        """
        try:
            # Use lock to prevent race conditions during initialization
            with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
                # Get list of checkpoint files sorted by creation time
                if self.checkpoint_dir.exists():
                    checkpoint_files = sorted(
                        [f for f in self.checkpoint_dir.glob("*.json") if f.is_file()],
                        key=lambda f: f.stat().st_mtime,
                        reverse=True
                    )

                    # Extract checkpoint names and timestamps
                    self.checkpoints = [
                        (f.stem, datetime.fromtimestamp(f.stat().st_mtime))
                        for f in checkpoint_files
                    ]

                    if self.checkpoints:
                        self.logger.info(f"Found {len(self.checkpoints)} existing checkpoints")

                        # Log using progress_manager if available
                        if self.progress_manager:
                            self.progress_manager.log_info(
                                f"Found {len(self.checkpoints)} existing checkpoints for task {self.task_id}"
                            )
        except filelock.Timeout:
            self.logger.warning(f"Timeout waiting for lock when loading checkpoint history")
            # Continue without history rather than failing
        except Exception as e:
            self.logger.warning(f"Error loading checkpoint history: {str(e)}", exc_info=True)
            # Continue without history rather than failing

    def _atomic_write_json(self, data: Dict[str, Any], path: Path) -> None:
        """
        Write JSON data to a file atomically.

        Args:
            data: JSON data to write
            path: Target file path

        Raises:
            StateSerializationError: If writing fails
        """
        try:
            # Create a temporary file in the same directory
            fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp.json')

            try:
                # Write data to temporary file
                data_serialized = _serialize_data(data)
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data_serialized, f, indent=2, ensure_ascii=False)

                # Perform atomic replace
                os.replace(temp_path, path)

            except Exception as e:
                # Clean up temp file if it exists
                try:
                    os.unlink(temp_path)
                except (OSError, FileNotFoundError):
                    pass

                raise StateSerializationError(f"Failed to write checkpoint file: {str(e)}") from e

        except Exception as e:
            if not isinstance(e, StateSerializationError):
                raise StateSerializationError(f"Failed to create temporary file: {str(e)}") from e
            raise

    def save_execution_state(self, state: Dict[str, Any], checkpoint_name: Optional[str] = None) -> Path:
        """
        Save execution state to a checkpoint file.

        Args:
            state: Execution state to save
            checkpoint_name: Name for the checkpoint (optional)

        Returns:
            Path to the saved checkpoint file under .checkpoints directory

        Raises:
            StateSerializationError: If saving the state fails
        """
        # Create progress tracker if available
        if self.progress_manager:
            try:
                progress_context = self.progress_manager.create_operation_context(
                    name="save_checkpoint",
                    total=4,  # Lock, prepare, write, update log
                    description=f"Saving checkpoint {checkpoint_name or 'auto'}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to create progress context: {str(e)}")
                progress_context = nullcontext(NullProgressTracker())
        else:
            # Use null progress tracker
            progress_context = nullcontext(NullProgressTracker())

        with progress_context as progress:
            try:
                # Use lock to prevent race conditions
                with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
                    # Update progress
                    if progress is not None:
                        progress.update(1, {"status": "acquired_lock"})

                    # Update last modified timestamp
                    state_copy = state.copy()
                    state_copy["last_updated"] = datetime.now().isoformat()

                    # Check if state size exceeds maximum
                    data_serialized = _serialize_data(state_copy)                        
                    state_size = len(json.dumps(data_serialized))
                    if state_size > self.max_state_size:
                        self.logger.warning(
                            f"State size ({state_size} bytes) exceeds maximum ({self.max_state_size} bytes). "
                            f"Cleaning up old checkpoints to manage disk space."
                        )
                        self.cleanup_old_checkpoints()

                    # Generate checkpoint name if not provided
                    if checkpoint_name is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        operation_index = state_copy.get("operation_index", 0)
                        checkpoint_name = f"checkpoint_{self.task_id}_{operation_index}_{timestamp}"

                    # Ensure checkpoint name is safe
                    checkpoint_name = self._sanitize_checkpoint_name(checkpoint_name)

                    # Update progress
                    if progress is not None:
                        progress.update(1, {"status": "preparing", "checkpoint": checkpoint_name})

                    # Resolve checkpoint file path
                    checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"

                    # Validate path security
                    if not validate_path_security(checkpoint_path):
                        error_msg = f"Insecure checkpoint path: {checkpoint_path}"
                        self.logger.error(error_msg)
                        raise StateSerializationError(error_msg)

                    # Save state to file using atomic write
                    self._atomic_write_json(state_copy, checkpoint_path)

                    # Update progress
                    if progress is not None:
                        progress.update(1, {"status": "saved", "path": str(checkpoint_path)})

                    # Update checkpoint history - get timestamp from file
                    checkpoint_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)

                    # Check if checkpoint exists and update timestamp if it does
                    for i, (name, _) in enumerate(self.checkpoints):
                        if name == checkpoint_name:
                            self.checkpoints[i] = (checkpoint_name, checkpoint_time)
                            break
                    else:
                        # Add new checkpoint to the beginning of the list
                        self.checkpoints.insert(0, (checkpoint_name, checkpoint_time))

                # Update execution log for persistent tracking
                # (This is done outside the lock to avoid holding the lock during external API calls)
                try:
                    # Get the latest task execution ID
                    latest_execution = find_latest_execution(self.task_id, success_only=False)
                    if latest_execution and "task_run_id" in latest_execution:
                        task_run_id = latest_execution["task_run_id"]
                        # Update execution record with checkpoint info
                        checkpoint_updates = {
                            "checkpoint": {
                                "name": checkpoint_name,
                                "timestamp": datetime.now().isoformat(),
                                "path": str(checkpoint_path)
                            }
                        }
                        update_execution_record(task_run_id, checkpoint_updates)
                        self.logger.debug(f"Updated execution log with checkpoint: {checkpoint_name}")

                        # Update progress
                        if progress is not None:
                            progress.update(1, {"status": "updated_log"})
                except Exception as e:
                    self.logger.warning(f"Failed to update execution log with checkpoint: {str(e)}", exc_info=True)
                    # Continue even if execution log update fails
                    if progress is not None:
                        progress.update(1, {"status": "log_error", "error": str(e)})

                if self.progress_manager:
                    self.progress_manager.log_info(f"Saved checkpoint: {checkpoint_name}")
                else:
                    self.logger.info(f"Saved execution state to checkpoint: {checkpoint_path}")

                return checkpoint_path

            except filelock.Timeout:
                error_msg = f"Timeout waiting for lock when saving checkpoint"
                self.logger.error(error_msg)

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_error(error_msg)

                raise StateSerializationError(error_msg)
            except Exception as e:
                if isinstance(e, StateSerializationError):
                    raise

                error_msg = f"Failed to save execution state: {str(e)}"
                self.logger.error(error_msg, exc_info=True)

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_error(error_msg)

                raise StateSerializationError(error_msg) from e

    def restore_execution_state(self, checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Restore execution state from a checkpoint file.

        Args:
            checkpoint_name: Name of the checkpoint to restore (optional, uses latest if None)

        Returns:
            Restored execution state dictionary

        Raises:
            StateRestorationError: If restoring the state fails
        """
        # Create progress tracker if available
        if self.progress_manager:
            try:
                progress_context = self.progress_manager.create_operation_context(
                    name="restore_checkpoint",
                    total=3,  # Find checkpoint, load state, validate
                    description=f"Restoring from checkpoint {checkpoint_name or 'latest'}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to create progress context: {str(e)}")
                progress_context = nullcontext(NullProgressTracker())
        else:
            # Use null progress tracker
            progress_context = nullcontext(NullProgressTracker())

        with progress_context as progress:
            try:
                # Use lock to prevent race conditions
                with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
                    # If no checkpoint name is provided, use the latest
                    if checkpoint_name is None:
                        if not self.checkpoints:
                            # Try to find from execution log
                            latest_checkpoint = self.get_latest_checkpoint()
                            if latest_checkpoint:
                                checkpoint_name = latest_checkpoint
                            else:
                                error_msg = "No checkpoints available to restore"
                                self.logger.error(error_msg)

                                # Log through progress manager if available
                                if self.progress_manager:
                                    self.progress_manager.log_error(error_msg)

                                raise StateRestorationError(error_msg)
                        else:
                            # Get name from first tuple (newest checkpoint)
                            checkpoint_name = self.checkpoints[0][0]

                    # Update progress
                    if progress is not None:
                        progress.update(1, {"status": "found", "checkpoint": checkpoint_name})

                    # Try to get checkpoint from execution log first (for cross-run persistence)
                    checkpoint_path = None
                    try:
                        # Find checkpoint path from execution log
                        latest_execution = find_latest_execution(self.task_id, success_only=False)
                        if latest_execution and "checkpoint" in latest_execution:
                            checkpoint_info = latest_execution["checkpoint"]
                            if checkpoint_info.get("name") == checkpoint_name:
                                checkpoint_path = Path(checkpoint_info.get("path"))
                                if checkpoint_path.exists():
                                    stored_state = read_json(checkpoint_path)

                                    # Verify task ID consistency before modifying state
                                    if stored_state.get("task_id") != self.task_id:
                                        error_msg = f"Checkpoint task ID '{stored_state.get('task_id')}' does not match current task ID '{self.task_id}'"
                                        self.logger.error(error_msg)
                                        raise StateRestorationError(error_msg)

                                    # Keep original created_at from stored state
                                    original_created_at = stored_state.get("created_at")

                                    # Update current state
                                    self.current_state = stored_state

                                    # Ensure created_at is preserved but last_updated is refreshed
                                    if original_created_at:
                                        self.current_state["created_at"] = original_created_at
                                    self.current_state["last_updated"] = datetime.now().isoformat()

                                    # Update progress
                                    if progress is not None:
                                        progress.update(2, {"status": "restored_from_log"})

                                    if self.progress_manager:
                                        self.progress_manager.log_info(
                                            f"Restored from execution log checkpoint: {checkpoint_name}"
                                        )
                                    else:
                                        self.logger.info(
                                            f"Restored execution state from execution log checkpoint: {checkpoint_name}")

                                    return self.current_state
                    except Exception as e:
                        self.logger.debug(f"Couldn't load from execution log, trying local file: {str(e)}")
                        # Continue to local file fallback

                    # Ensure checkpoint name is safe
                    checkpoint_name = self._sanitize_checkpoint_name(checkpoint_name)

                    # Resolve checkpoint file path
                    checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"

                    # Validate path security
                    if not validate_path_security(checkpoint_path):
                        error_msg = f"Insecure checkpoint path: {checkpoint_path}"
                        self.logger.error(error_msg)
                        raise StateRestorationError(error_msg)

                    # Check if file exists
                    if not checkpoint_path.exists():
                        error_message = f"Checkpoint file not found: {checkpoint_path}"
                        self.logger.error(error_message)

                        # Log through progress manager if available
                        if self.progress_manager:
                            self.progress_manager.log_error(error_message)

                        raise StateRestorationError(error_message)

                    # Load state from file
                    stored_state = read_json(checkpoint_path)

                    # Update progress
                    if progress is not None:
                        progress.update(1, {"status": "loaded", "path": str(checkpoint_path)})

                    # Verify essential keys
                    required_keys = ["task_id", "operation_index", "operations_completed"]
                    missing_keys = [key for key in required_keys if key not in stored_state]
                    if missing_keys:
                        error_msg = f"Invalid checkpoint state, missing keys: {missing_keys}"
                        self.logger.error(error_msg)

                        # Log through progress manager if available
                        if self.progress_manager:
                            self.progress_manager.log_error(error_msg)

                        raise StateRestorationError(error_msg)

                    # Verify task ID consistency before modifying state
                    if stored_state["task_id"] != self.task_id:
                        error_msg = f"Checkpoint task ID '{stored_state['task_id']}' does not match current task ID '{self.task_id}'"
                        self.logger.error(error_msg)

                        # Log through progress manager if available
                        if self.progress_manager:
                            self.progress_manager.log_error(error_msg)

                        raise StateRestorationError(error_msg)

                    # Keep original created_at from stored state
                    original_created_at = stored_state.get("created_at")

                    # Update current state
                    self.current_state = stored_state

                    # Ensure created_at is preserved but last_updated is refreshed
                    if original_created_at:
                        self.current_state["created_at"] = original_created_at
                    self.current_state["last_updated"] = datetime.now().isoformat()

                    # Update progress
                    if progress is not None:
                        progress.update(1, {
                            "status": "validated",
                            "operation_index": self.current_state.get("operation_index", -1)
                        })

                    if self.progress_manager:
                        self.progress_manager.log_info(
                            f"Restored from checkpoint: {checkpoint_name} (Operation index: {self.current_state.get('operation_index', -1)})"
                        )
                    else:
                        self.logger.info(f"Restored execution state from checkpoint: {checkpoint_path}")

                    return self.current_state

            except filelock.Timeout:
                error_msg = f"Timeout waiting for lock when restoring checkpoint"
                self.logger.error(error_msg)

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_error(error_msg)

                raise StateRestorationError(error_msg)
            except Exception as e:
                if isinstance(e, StateRestorationError):
                    raise

                error_msg = f"Failed to restore execution state: {str(e)}"
                self.logger.error(error_msg, exc_info=True)

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_error(error_msg)

                raise StateRestorationError(error_msg) from e

    def create_automatic_checkpoint(self, operation_index: int, metrics: Dict = None) -> Path:
        """
        Create an automatic checkpoint at the current execution point.

        Args:
            operation_index: Index of the current operation
            metrics: Metrics collected up to this point

        Returns:
            Name of the created checkpoint

        Raises:
            CheckpointError: If creating the checkpoint fails
        """
        try:
            # Update current state
            self.current_state["operation_index"] = operation_index
            self.current_state["last_updated"] = datetime.now().isoformat()

            # Add config hash and task version to state for validation on restore
            config_dict = self._get_task_config()
            self.current_state["config_hash"] = self._calculate_config_hash(config_dict)
            self.current_state["task_version"] = self._get_task_version()

            # Update metrics
            if metrics:
                if "metrics" not in self.current_state:
                    self.current_state["metrics"] = {}

                # Merge metrics
                for key, value in metrics.items():
                    self.current_state["metrics"][key] = value

            # Generate checkpoint name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"auto_{self.task_id}_{operation_index}_{timestamp}"

            # Save checkpoint - with progress integration from save_execution_state
            checkpoint_path = self.save_execution_state(self.current_state, checkpoint_name)

            return checkpoint_path

        except Exception as e:
            error_msg = f"Failed to create automatic checkpoint: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)

            raise CheckpointError(error_msg) from e

    def _get_task_config(self) -> Dict[str, Any]:
        """
        Get the current task configuration if available through the progress manager.

        Returns:
            Dict containing configuration or empty dict if not available
        """
        # Try to get config from progress manager's task
        if self.progress_manager and hasattr(self.progress_manager, 'task'):
            task = self.progress_manager.task
            if hasattr(task, 'config') and hasattr(task.config, 'to_dict'):
                return task.config.to_dict()

        return {}

    def _get_task_version(self) -> str:
        """
        Get the current task version if available through the progress manager.

        Returns:
            Task version string or "unknown" if not available
        """
        # Try to get version from progress manager's task
        if self.progress_manager and hasattr(self.progress_manager, 'task'):
            task = self.progress_manager.task
            if hasattr(task, 'version'):
                return task.version

        return "unknown"

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update the current execution state.

        Args:
            updates: Dictionary of updates to apply to the current state

        Raises:
            ContextManagerError: If updating the state fails
        """
        try:
            # Update current state
            for key, value in updates.items():
                if key in self.current_state and isinstance(self.current_state[key], dict) and isinstance(value, dict):
                    # Merge nested dictionaries
                    self.current_state[key].update(value)
                else:
                    # Replace value
                    self.current_state[key] = value

            # Update timestamp
            self.current_state["last_updated"] = datetime.now().isoformat()

        except Exception as e:
            error_msg = f"Failed to update execution state: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)

            raise ContextManagerError(error_msg) from e

    def record_operation_completion(self, operation_index: int, operation_name: str,
                                    result_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the completion of an operation.

        Args:
            operation_index: Index of the completed operation
            operation_name: Name of the completed operation
            result_metrics: Metrics from the operation result

        Raises:
            ContextManagerError: If recording the operation fails
        """
        try:
            # Update operation index
            if operation_index > self.current_state["operation_index"]:
                self.current_state["operation_index"] = operation_index

            # Update completed operations
            if "operations_completed" not in self.current_state:
                self.current_state["operations_completed"] = []

            # Avoid duplicates
            operation_entry = {"index": operation_index, "name": operation_name,
                               "timestamp": datetime.now().isoformat()}
            if operation_entry not in self.current_state["operations_completed"]:
                self.current_state["operations_completed"].append(operation_entry)

            # Update metrics
            if result_metrics:
                if "metrics" not in self.current_state:
                    self.current_state["metrics"] = {}

                # Create operation-specific metrics entry
                self.current_state["metrics"][operation_name] = result_metrics

            # Update timestamp
            self.current_state["last_updated"] = datetime.now().isoformat()

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_debug(
                    f"Recorded completion of operation {operation_name} (index: {operation_index})"
                )

        except Exception as e:
            error_msg = f"Failed to record operation completion: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)

            raise ContextManagerError(error_msg) from e

    def record_operation_failure(self, operation_index: int, operation_name: str,
                                 error_info: Dict[str, Any]) -> None:
        """
        Record the failure of an operation.

        Args:
            operation_index: Index of the failed operation
            operation_name: Name of the failed operation
            error_info: Information about the error

        Raises:
            ContextManagerError: If recording the failure fails
        """
        try:
            # Update operation index
            if operation_index > self.current_state["operation_index"]:
                self.current_state["operation_index"] = operation_index

            # Update failed operations
            if "operations_failed" not in self.current_state:
                self.current_state["operations_failed"] = []

            # Add failure entry
            failure_entry = {
                "index": operation_index,
                "name": operation_name,
                "timestamp": datetime.now().isoformat(),
                "error": error_info
            }

            # Avoid duplicates - check if operation with same index and name exists
            existing = [op for op in self.current_state["operations_failed"]
                        if op.get("index") == operation_index and op.get("name") == operation_name]

            if existing:
                # Replace existing entry
                for i, entry in enumerate(self.current_state["operations_failed"]):
                    if entry.get("index") == operation_index and entry.get("name") == operation_name:
                        self.current_state["operations_failed"][i] = failure_entry
                        break
            else:
                # Add new entry
                self.current_state["operations_failed"].append(failure_entry)

            # Update timestamp
            self.current_state["last_updated"] = datetime.now().isoformat()

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_warning(
                    f"Recorded failure of operation {operation_name} (index: {operation_index}): {error_info.get('message', 'Unknown error')}"
                )

        except Exception as e:
            error_msg = f"Failed to record operation failure: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)

            raise ContextManagerError(error_msg) from e

    def record_artifact(self, artifact_path: Union[str, Path], artifact_type: str,
                        description: Optional[str] = None) -> None:
        """
        Record an artifact created during task execution.

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of the artifact
            description: Description of the artifact

        Raises:
            ContextManagerError: If recording the artifact fails
        """
        try:
            # Convert path to string
            path_str = str(artifact_path)

            # Initialize artifacts list if needed
            if "artifacts" not in self.current_state:
                self.current_state["artifacts"] = []

            # Create artifact entry
            artifact_entry = {
                "path": path_str,
                "type": artifact_type,
                "description": description or "",
                "timestamp": datetime.now().isoformat()
            }

            # Avoid duplicates - check if artifact with same path exists
            existing = [a for a in self.current_state["artifacts"] if a.get("path") == path_str]

            if existing:
                # Replace existing entry
                for i, entry in enumerate(self.current_state["artifacts"]):
                    if entry.get("path") == path_str:
                        self.current_state["artifacts"][i] = artifact_entry
                        break
            else:
                # Add new entry
                self.current_state["artifacts"].append(artifact_entry)

            # Update timestamp
            self.current_state["last_updated"] = datetime.now().isoformat()

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_debug(
                    f"Recorded artifact: {artifact_type} at {path_str}"
                )

        except Exception as e:
            error_msg = f"Failed to record artifact: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_error(error_msg)

            raise ContextManagerError(error_msg) from e

    def can_resume_execution(self) -> Tuple[bool, Optional[str]]:
        """
        Check if task execution can be resumed from a checkpoint.

        Returns:
            Tuple containing:
                - Boolean indicating if execution can be resumed
                - Name of the checkpoint to resume from (or None)

        Raises:
            ContextManagerError: If checking for resumable execution fails
        """
        # Create progress tracker if available
        if self.progress_manager:
            try:
                progress_context = self.progress_manager.create_operation_context(
                    name="check_resumable",
                    total=2,  # Check local, check execution log
                    description="Checking for resumable execution"
                )
            except Exception as e:
                self.logger.warning(f"Failed to create progress context: {str(e)}")
                progress_context = nullcontext(NullProgressTracker())
        else:
            # Use null progress tracker
            progress_context = nullcontext(NullProgressTracker())

        with progress_context as progress:
            try:
                # Use lock to prevent race conditions
                with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
                    # Check if we have local checkpoints
                    if self.checkpoints:
                        # Get name from first tuple (newest checkpoint)
                        latest_checkpoint = self.checkpoints[0][0]

                        # Update progress
                        if progress is not None:
                            progress.update(2, {"status": "found_local", "checkpoint": latest_checkpoint})

                        if self.progress_manager:
                            self.progress_manager.log_info(
                                f"Found local checkpoint: {latest_checkpoint}"
                            )

                        return True, latest_checkpoint

                # Update progress
                if progress is not None:
                    progress.update(1, {"status": "checking_log"})

                # Check execution log for checkpoints (outside lock)
                latest_checkpoint = self.get_latest_checkpoint()
                if latest_checkpoint:
                    # Update progress
                    if progress is not None:
                        progress.update(1, {"status": "found_log", "checkpoint": latest_checkpoint})

                    if self.progress_manager:
                        self.progress_manager.log_info(
                            f"Found checkpoint in execution log: {latest_checkpoint}"
                        )

                    return True, latest_checkpoint

                # Update progress for no checkpoints found
                if progress is not None:
                    progress.update(1, {"status": "no_checkpoints"})

                # No checkpoints available
                return False, None

            except filelock.Timeout:
                self.logger.warning(f"Timeout waiting for lock when checking resumable execution")

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_warning(
                        "Timeout waiting for lock when checking resumable execution"
                    )

                # Try with execution log anyway
                latest_checkpoint = self.get_latest_checkpoint()
                if latest_checkpoint:
                    return True, latest_checkpoint
                return False, None
            except Exception as e:
                error_msg = f"Failed to check for resumable execution: {str(e)}"
                self.logger.error(error_msg, exc_info=True)

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_error(error_msg)

                raise ContextManagerError(error_msg) from e

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the name of the latest checkpoint from the execution log.

        Returns:
            Name of the latest checkpoint or None if not found
        """
        try:
            # Find latest execution of this task
            latest_execution = find_latest_execution(self.task_id, success_only=False)

            # Check if it has checkpoint information
            if latest_execution and "checkpoint" in latest_execution:
                checkpoint_info = latest_execution["checkpoint"]
                return checkpoint_info.get("name")

            return None
        except Exception as e:
            self.logger.warning(f"Error getting latest checkpoint from execution log: {str(e)}", exc_info=True)

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_warning(
                    f"Error getting latest checkpoint from execution log: {str(e)}"
                )

            return None

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current execution state.

        Returns:
            Current execution state dictionary
        """
        return self.current_state.copy()

    def get_checkpoints(self) -> List[Tuple[str, datetime]]:
        """
        Get a list of available checkpoints with timestamps.

        Returns:
            List of tuples containing checkpoint names and timestamps, sorted by creation time (newest first)
        """
        try:
            # Use lock to prevent race conditions
            with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
                return self.checkpoints.copy()
        except filelock.Timeout:
            self.logger.warning(f"Timeout waiting for lock when getting checkpoints")

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_warning(
                    "Timeout waiting for lock when getting checkpoints"
                )

            return []  # Return empty list on timeout
        except Exception as e:
            self.logger.warning(f"Error getting checkpoints: {str(e)}", exc_info=True)

            # Log through progress manager if available
            if self.progress_manager:
                self.progress_manager.log_warning(
                    f"Error getting checkpoints: {str(e)}"
                )

            return []  # Return empty list on error

    def _sanitize_checkpoint_name(self, name: str) -> str:
        """
        Sanitize a checkpoint name to ensure it's safe to use as a filename.

        Args:
            name: Checkpoint name to sanitize

        Returns:
            Sanitized checkpoint name
        """
        # Replace unsafe characters
        sanitized = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        sanitized = sanitized.replace("*", "_").replace("?", "_").replace("\"", "_")
        sanitized = sanitized.replace("<", "_").replace(">", "_").replace("|", "_")
        sanitized = sanitized.replace(" ", "_")

        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]

        return sanitized

    def cleanup_old_checkpoints(self, max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS) -> int:
        """
        Remove old checkpoints to manage disk space.

        Args:
            max_checkpoints: Maximum number of checkpoints to keep

        Returns:
            Number of checkpoints removed

        Raises:
            ContextManagerError: If cleaning up checkpoints fails
        """
        # Create progress tracker if available
        if self.progress_manager:
            try:
                progress_context = self.progress_manager.create_operation_context(
                    name="cleanup_checkpoints",
                    total=2,  # Identify, remove
                    description=f"Cleaning up old checkpoints (keeping {max_checkpoints})"
                )
            except Exception as e:
                self.logger.warning(f"Failed to create progress context: {str(e)}")
                progress_context = nullcontext(NullProgressTracker())
        else:
            # Use null progress tracker
            progress_context = nullcontext(NullProgressTracker())

        with progress_context as progress:
            try:
                # Use lock to prevent race conditions
                with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
                    # If we have fewer checkpoints than the maximum, nothing to do
                    if len(self.checkpoints) <= max_checkpoints:
                        # Update progress
                        if progress is not None:
                            progress.update(2, {"status": "no_cleanup_needed"})

                        return 0

                    # Determine checkpoints to remove
                    checkpoints_to_remove = self.checkpoints[max_checkpoints:]

                    # Update progress
                    if progress is not None:
                        progress.update(1, {
                            "status": "identified",
                            "to_remove": len(checkpoints_to_remove)
                        })

                    # Remove checkpoints
                    removed_count = 0
                    for checkpoint_name, _ in checkpoints_to_remove:
                        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"

                        try:
                            # Validate path security
                            if not validate_path_security(checkpoint_path):
                                self.logger.warning(f"Skipping removal of insecure checkpoint path: {checkpoint_path}")
                                continue

                            # Remove file
                            if checkpoint_path.exists():
                                checkpoint_path.unlink()
                                removed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Error removing checkpoint {checkpoint_path}: {str(e)}", exc_info=True)

                    # Update checkpoint list
                    self.checkpoints = self.checkpoints[:max_checkpoints]

                    # Update progress
                    if progress is not None:
                        progress.update(1, {
                            "status": "removed",
                            "removed_count": removed_count
                        })

                    # Log cleaning result
                    cleanup_msg = (
                        f"Removed {removed_count} old checkpoints, kept {len(self.checkpoints)}. "
                        f"Note: execution log may still contain references to removed checkpoints."
                    )

                    if self.progress_manager:
                        self.progress_manager.log_info(cleanup_msg)
                    else:
                        self.logger.warning(cleanup_msg)

                    return removed_count

            except filelock.Timeout:
                error_msg = f"Timeout waiting for lock when cleaning up checkpoints"
                self.logger.error(error_msg)

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_error(error_msg)

                raise ContextManagerError(error_msg)
            except Exception as e:
                error_msg = f"Failed to clean up old checkpoints: {str(e)}"
                self.logger.error(error_msg, exc_info=True)

                # Log through progress manager if available
                if self.progress_manager:
                    self.progress_manager.log_error(error_msg)

                raise ContextManagerError(error_msg) from e

    def __enter__(self) -> "TaskContextManager":
        """
        Enter the context manager.

        Returns:
            Self for use in with-statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context manager.

        Creates a final checkpoint if an exception occurred, and cleans up old checkpoints.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise
            exc_val: Exception value if an exception occurred, None otherwise
            exc_tb: Exception traceback if an exception occurred, None otherwise

        Returns:
            False to allow exception propagation (do not suppress exceptions)
        """
        try:
            if exc_type is not None:
                # An exception occurred, create a final checkpoint with error information
                error_info = {
                    "type": exc_type.__name__ if exc_type else "unknown",
                    "message": str(exc_val) if exc_val else "No message",
                    "timestamp": datetime.now().isoformat(),
                }
                try:
                    # Try to create a final error checkpoint
                    checkpoint_name = f"error_{self.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.update_state({"error_info": error_info})
                    self.save_execution_state(self.current_state, checkpoint_name)

                    if self.progress_manager:
                        self.progress_manager.log_info(f"Created error checkpoint: {checkpoint_name}")
                    else:
                        self.logger.info(f"Created error checkpoint: {checkpoint_name}")

                except Exception as e:
                    if self.progress_manager:
                        self.progress_manager.log_warning(f"Failed to create error checkpoint: {str(e)}")
                    else:
                        self.logger.warning(f"Failed to create error checkpoint: {str(e)}", exc_info=True)
            else:
                # Clean up old checkpoints if execution completed successfully
                try:
                    self.cleanup_old_checkpoints()
                except Exception as e:
                    if self.progress_manager:
                        self.progress_manager.log_warning(f"Failed to clean up old checkpoints: {str(e)}")
                    else:
                        self.logger.warning(f"Failed to clean up old checkpoints: {str(e)}", exc_info=True)
        except Exception as e:
            # Make sure exceptions in __exit__ don't prevent normal exception handling
            if self.progress_manager:
                self.progress_manager.log_error(f"Error in context manager exit: {str(e)}")
            else:
                self.logger.error(f"Error in context manager exit: {str(e)}", exc_info=True)

        # Don't suppress the original exception
        return False

    def cleanup(self) -> None:
        """
        Explicitly clean up resources.

        This method is called when the manager is no longer needed,
        especially in cases where the context manager cannot be used.
        """
        try:
            # Clean up old checkpoints
            self.cleanup_old_checkpoints()
        except Exception as e:
            if self.progress_manager:
                self.progress_manager.log_warning(f"Error during cleanup: {str(e)}")
            else:
                self.logger.warning(f"Error during cleanup: {str(e)}", exc_info=True)


# Helper function to create a context manager
def create_task_context_manager(
    task_id: str,
    task_dir: Path,
    log_directory: Optional[Path] = None,
    max_state_size: int = DEFAULT_MAX_STATE_SIZE,
    progress_manager: Optional[Any] = None
) -> TaskContextManager:
    """
    Create a context manager for a task.

    Args:
        task_id: ID of the task
        task_dir: Path to the task directory
        log_directory: Optional path to store logs and checkpoints centrally
        max_state_size: Maximum state size in bytes before pruning older checkpoints
        progress_manager: Optional progress manager for visual feedback during operations

    Returns:
        TaskContextManager instance

    Raises:
        ContextManagerError: If context manager creation fails
    """
    try:
        # Create context manager with log directory
        context_manager = TaskContextManager(
            task_id=task_id,
            task_dir=task_dir,
            log_directory=log_directory,
            max_state_size=max_state_size,
            progress_manager=progress_manager
        )

        return context_manager

    except Exception as e:
        # Use standard logger for initial errors before context manager is created
        logger = logging.getLogger(f"task.{task_id}.context")
        logger.error(f"Error creating context manager", exc_info=True)

        # Log through progress manager if available
        if progress_manager:
            progress_manager.log_error(f"Failed to create context manager: {str(e)}")

        raise ContextManagerError(f"Failed to create context manager: {str(e)}") from e
    

def _serialize_data(data):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if data is None:
        return None

    data_type = type(data)
    
    # Ultra-fast path for Python built-ins (using 'is' instead of isinstance)
    if data_type is dict:
        return {k: _serialize_data(v) for k, v in data.items()}
    if data_type is list:
        return [_serialize_data(item) for item in data]
    
    # Fast lookup for common NumPy types
    _NP_HANDLERS = {
        np.ndarray: lambda x: x.tolist(),
    }
    handler = _NP_HANDLERS.get(data_type)
    if handler is not None:
        return handler(data)
    
    # Safe NumPy scalar handling with minimal checks
    try:
        if issubclass(data_type, np.integer):  # Covers all integer types (int32, int64, etc.)
            return int(data)
        if issubclass(data_type, np.floating):  # Covers all float types (float32, float64, etc.)
            return float(data)
        if issubclass(data_type, np.bool_):
            return bool(data)
    except TypeError:  # Not a NumPy type
        pass
    
    return data  # Return unchanged for other types