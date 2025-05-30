# PAMOLA Core Task Framework API Reference

This document provides a comprehensive reference of all public classes and functions in the PAMOLA Core Task Framework.

## 1. BaseTask (base_task.py)

`BaseTask` is the primary class for task implementation, defining the standard task lifecycle and serving as a facade for all other framework components.

### Classes

#### `class TaskError(Exception)`
Base exception for task-related errors.

#### `class TaskInitializationError(TaskError)`
Exception raised when task initialization fails.

#### `class TaskDependencyError(TaskError)`
Exception raised when task dependencies are not satisfied.

#### `class BaseTask`
Base class for all tasks in the PAMOLA ecosystem.

##### Constructor

```python
def __init__(self, 
             task_id: str, 
             task_type: str, 
             description: str, 
             input_datasets: Optional[Dict[str, str]] = None,
             auxiliary_datasets: Optional[Dict[str, str]] = None,
             version: str = "1.0.0")
```

##### Pamola Core Lifecycle Methods

```python
def initialize(self, args: Optional[Dict[str, Any]] = None) -> bool
```
Initialize the task by loading configuration, creating directories, setting up logging, and checking dependencies.

```python
def configure_operations(self) -> None
```
Configure operations to be executed by the task (must be implemented by subclasses).

```python
def execute(self) -> bool
```
Execute the task by running operations sequentially, collecting results, and generating metrics.

```python
def finalize(self, success: bool) -> bool
```
Finalize the task by releasing resources, closing files, and registering the execution result.

```python
def run(self, args: Optional[Dict[str, Any]] = None) -> bool
```
Run the complete task lifecycle: initialize, configure, execute, finalize.

##### Operation Management

```python
def add_operation(self, operation_class: Union[str, Type[BaseOperation]], **kwargs) -> bool
```
Add an operation to the task's execution queue.

##### Result Access Methods

```python
def get_results(self) -> Dict[str, OperationResult]
```
Get the results of all operations executed by the task.

```python
def get_artifacts(self) -> List[Any]
```
Get all artifacts produced by the task.

```python
def get_metrics(self) -> Dict[str, Any]
```
Get all metrics collected by the task.

```python
def get_execution_status(self) -> Tuple[str, Optional[Dict[str, Any]]]
```
Get the execution status and error information.

```python
def get_encryption_info(self) -> Dict[str, Any]
```
Get information about the task's encryption settings.

```python
def get_checkpoint_status(self) -> Dict[str, Any]
```
Get information about the checkpoint status of the task.

##### Helper Methods

```python
def _setup_logging(self) -> None
```
Set up dual logging to both project-level and task-specific logs.

```python
def _initialize_data_source(self) -> None
```
Initialize the data source with input and auxiliary datasets.

```python
def _initialize_data_writer(self) -> None
```
Initialize the data writer for task outputs.

```python
def _run_operations(self, start_idx: int = 0) -> bool
```
Run operations starting from the specified index.

```python
def _prepare_operation_parameters(self, operation: BaseOperation) -> Dict[str, Any]
```
Prepare parameters for an operation, including system parameters and operation-specific parameters.

```python
def _get_operation_supported_parameters(self, operation: Union[BaseOperation, Type[BaseOperation]]) -> set
```
Get the set of parameters that an operation's constructor accepts.

---

## 2. TaskConfig (task_config.py)

`TaskConfig` manages configuration loading, path resolution, and environment variable handling.

### Enums

#### `class EncryptionMode(Enum)`
Encryption modes supported by the task framework.

```python
NONE = "none"
SIMPLE = "simple"
AGE = "age"

@classmethod
def from_string(cls, value: str) -> 'EncryptionMode'
```

### Exceptions

#### `class ConfigurationError(Exception)`
Exception raised for configuration-related errors.

#### `class DependencyMissingError(Exception)`
Exception raised when required dependency is missing.

### Classes

#### `class TaskConfig`
Task configuration container and manager.

##### Constructor

```python
def __init__(self,
             config_dict: Dict[str, Any],
             task_id: str,
             task_type: str,
             env_override: bool = True,
             progress_manager: Optional[TaskProgressManager] = None)
```

##### Path API Methods

```python
def get_project_root(self) -> Path
```
Get the project root directory.

```python
def get_data_repository(self) -> Path
```
Get the data repository path.

```python
def get_raw_dir(self) -> Path
```
Get the raw data directory.

```python
def get_processed_dir(self) -> Path
```
Get the processed data directory.

```python
def get_reports_dir(self) -> Path
```
Get the reports directory.

```python
def get_task_dir(self, task_id: Optional[str] = None) -> Path
```
Get the task directory.

```python
def get_task_input_dir(self, task_id: Optional[str] = None) -> Path
```
Get the task input directory.

```python
def get_task_output_dir(self, task_id: Optional[str] = None) -> Path
```
Get the task output directory.

```python
def get_task_temp_dir(self, task_id: Optional[str] = None) -> Path
```
Get the task temporary directory.

```python
def get_task_dict_dir(self, task_id: Optional[str] = None) -> Path
```
Get the task dictionaries directory.

```python
def get_task_logs_dir(self, task_id: Optional[str] = None) -> Path
```
Get the task logs directory.

```python
def processed_subdir(self, task_id: Optional[str] = None, *parts) -> Path
```
Get a subdirectory within the processed directory for a specific task.

##### Dependency Management

```python
def get_dependency_output(self, dependency_id: str, file_pattern: Optional[str] = None) -> Union[Path, List[Path]]
```
Get the output directory or files from a dependency.

```python
def get_dependency_report(self, dependency_id: str) -> Path
```
Get the report file for a dependency.

```python
def assert_dependencies_completed(self) -> bool
```
Check if all dependencies have completed successfully.

##### Scope Methods

```python
def get_scope_fields(self) -> List[str]
```
Get fields defined in the scope.

```python
def get_scope_datasets(self) -> List[str]
```
Get datasets defined in the scope.

```python
def get_scope_field_groups(self) -> Dict[str, List[str]]
```
Get field groups defined in the scope.

##### Configuration Management

```python
def override_with_args(self, args: Dict[str, Any]) -> None
```
Override configuration with command line arguments.

```python
def validate(self) -> Tuple[bool, List[str]]
```
Validate the configuration.

```python
def to_dict(self) -> Dict[str, Any]
```
Convert configuration to dictionary.

```python
def save(self, path: Optional[Path] = None, format: str = "json") -> Path
```
Save configuration to file.

```python
def resolve_legacy_path(self, path: Union[str, Path]) -> Path
```
Resolve a path using legacy format during transition period.

### Functions

```python
def load_task_config(
    task_id: str,
    task_type: str,
    args: Optional[Dict[str, Any]] = None,
    progress_manager: Optional[TaskProgressManager] = None
) -> TaskConfig
```
Load task configuration from project configuration file and override with command line arguments.

---

## 3. TaskReporter (task_reporting.py)

`TaskReporter` handles operation tracking, artifact registration, and report generation.

### Exceptions

#### `class ReportingError(Exception)`
Exception raised for reporting errors.

### Classes

#### `class ArtifactGroup`
Group of related artifacts in a task report.

##### Constructor

```python
def __init__(self, name: str, description: str = "")
```

##### Methods

```python
def add_artifact(self, artifact: Dict[str, Any])
```
Add an artifact to the group.

```python
def to_dict(self) -> Dict[str, Any]
```
Convert the group to a dictionary for serialization.

#### `class TaskReporter`
Task reporter for generating and managing task reports.

##### Constructor

```python
def __init__(self, task_id: str, task_type: str, description: str, report_path: Union[str, Path],
             progress_manager: Optional[Any] = None)
```

##### Methods

```python
def add_operation(self, name: str, status: str = "success", details: Dict[str, Any] = None)
```
Add an operation to the report.

```python
def add_artifact(self, artifact_type: str, path: Union[str, Path], description: str = "",
                 category: str = "output", tags: Optional[List[str]] = None,
                 group_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None)
```
Add an artifact to the report.

```python
def add_artifact_group(self, name: str, description: str = "") -> ArtifactGroup
```
Add or get an artifact group.

```python
def get_artifact_group(self, name: str) -> Optional[ArtifactGroup]
```
Get an artifact group by name.

```python
def get_artifacts_by_tag(self, tag: str) -> List[Dict[str, Any]]
```
Get all artifacts with a specific tag.

```python
def get_artifacts_by_category(self, category: str) -> List[Dict[str, Any]]
```
Get all artifacts in a specific category.

```python
def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]
```
Get all artifacts of a specific type.

```python
def add_metric(self, name: str, value: Any)
```
Add a metric to the report.

```python
def add_nested_metric(self, category: str, name: str, value: Any)
```
Add a nested metric under a category.

```python
def add_task_summary(self, success: bool, execution_time: float = None,
                     metrics: Dict[str, Any] = None, error_info: Dict[str, Any] = None,
                     encryption: Dict[str, Any] = None)
```
Add task summary to the report.

```python
def generate_report(self) -> Dict[str, Any]
```
Generate the task report.

```python
def save_report(self) -> Path
```
Generate and save the report to disk.

---

## 4. DirectoryManger (directory_manager.py)

`TaskDirectoryManager` handles directory structure creation and path resolution.

### Exceptions

#### `class DirectoryManagerError(Exception)`
Base exception for directory manager errors.

#### `class PathValidationError(DirectoryManagerError)`
Exception raised when a path fails validation.

#### `class DirectoryCreationError(DirectoryManagerError)`
Exception raised when directory creation fails.

### Classes

#### `class TaskDirectoryManager`
Manager for task directory structures and path resolution.

##### Constructor

```python
def __init__(self,
             task_config: Any,
             logger: Optional[logging.Logger] = None,
             progress_manager: Optional[TaskProgressManager] = None)
```

##### Methods

```python
def ensure_directories(self) -> Dict[str, Path]
```
Create and validate all required task directories.

```python
def get_directory(self, dir_type: str) -> Path
```
Get path to a specific directory type.

```python
def get_artifact_path(self,
                      artifact_name: str,
                      artifact_type: str = "json",
                      subdir: str = "output",
                      include_timestamp: bool = True) -> Path
```
Generate standardized path for an artifact.

```python
def clean_temp_directory(self) -> bool
```
Clean temporary files and directories.

```python
def get_timestamped_filename(self, base_name: str, extension: str = "json") -> str
```
Generate a timestamped filename.

```python
def validate_directory_structure(self) -> Dict[str, bool]
```
Validate the task directory structure.

```python
def list_artifacts(self, subdir: str = "output", pattern: str = "*") -> List[Path]
```
List artifacts in a specific subdirectory.

```python
def import_external_file(self,
                         source_path: Union[str, Path],
                         subdir: str = "input",
                         new_name: Optional[str] = None) -> Path
```
Import an external file into the task directory structure.

```python
def normalize_and_validate_path(self, path: Union[str, Path]) -> Path
```
Normalize a path and validate its security.

```python
def cleanup(self) -> bool
```
Clean up resources.

### Functions

```python
def create_directory_manager(task_config: Any,
                             logger: Optional[logging.Logger] = None,
                             progress_manager: Optional[TaskProgressManager] = None,
                             initialize: bool = True) -> TaskDirectoryManager
```
Create a directory manager for a task.

---

## 5. EncryptionManager (encryption_manager.py)

`TaskEncryptionManager` handles secure encryption key management and sensitive data handling.

### Classes

#### `class EncryptionContext`
Secure encryption context that provides encryption capabilities without exposing the raw encryption key.

##### Constructor

```python
def __init__(self, mode: EncryptionMode, key_fingerprint: str)
```

##### Properties

```python
@property
def can_encrypt(self) -> bool
```
Check if this context can perform encryption operations.

##### Methods

```python
def to_dict(self) -> Dict[str, Any]
```
Convert context to dictionary for serialization.

#### `class MemoryProtectedKey`
Memory-protected encryption key container.

##### Constructor

```python
def __init__(self, key_material: bytes, key_id: Optional[str] = None)
```

##### Properties

```python
@property
def fingerprint(self) -> str
```
Get the key fingerprint (safe to expose).

```python
@property
def key_id(self) -> str
```
Get the key ID (safe to expose).

```python
@property
def has_been_used(self) -> bool
```
Check if this key has been used.

#### `class TaskEncryptionManager`
Encryption manager for secure handling of encryption keys and sensitive data.

##### Constructor

```python
def __init__(self,
             task_config: Any,
             logger: Optional[logging.Logger] = None,
             progress_manager: Optional['TaskProgressManager'] = None)
```

##### Methods

```python
def initialize(self) -> bool
```
Initialize encryption based on configuration.

```python
def get_encryption_context(self) -> EncryptionContext
```
Get secure encryption context for operations.

```python
def encrypt_data(self, data: bytes) -> bytes
```
Encrypt binary data using the configured encryption method.

```python
def decrypt_data(self, encrypted_data: bytes) -> bytes
```
Decrypt binary data using the configured encryption method.

```python
def add_sensitive_param_names(self, param_names: Union[str, List[str]]) -> None
```
Add parameter names that should be treated as sensitive.

```python
def is_sensitive_param(self, param_name: str) -> bool
```
Check if a parameter name should be treated as sensitive.

```python
def redact_sensitive_data(self, data: Any, redact_keys: bool = True) -> Any
```
Redact sensitive information from data structures.

```python
def redact_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]
```
Redact sensitive information from a configuration dictionary.

```python
def get_encryption_info(self) -> Dict[str, Any]
```
Get information about the encryption configuration.

```python
def check_dataset_encryption(self, data_source: Any) -> bool
```
Check if datasets in the data source are encrypted.

```python
def is_file_encrypted(self, file_path: Union[str, Path]) -> bool
```
Check if a file appears to be encrypted.

```python
def supports_encryption_mode(self, mode: Union[str, EncryptionMode]) -> bool
```
Check if the requested encryption mode is supported.

```python
def cleanup(self) -> None
```
Clean up resources.

### Exceptions

#### `class EncryptionError(Exception)`
Base exception for encryption-related errors.

#### `class EncryptionInitializationError(EncryptionError)`
Exception raised when encryption initialization fails.

#### `class KeyGenerationError(EncryptionError)`
Exception raised when key generation fails.

#### `class KeyLoadingError(EncryptionError)`
Exception raised when key loading fails.

#### `class DataRedactionError(EncryptionError)`
Exception raised when data redaction fails.

---

## 6. TaskContextManager (context_manager.py)

`TaskContextManager` manages task execution state, enabling checkpoints and resumable execution.

### Exceptions

#### `class ContextManagerError(Exception)`
Base exception for context manager errors.

#### `class CheckpointError(ContextManagerError)`
Exception raised for checkpoint-related errors.

#### `class StateSerializationError(ContextManagerError)`
Exception raised when state serialization fails.

#### `class StateRestorationError(ContextManagerError)`
Exception raised when state restoration fails.

### Classes

#### `class NullProgressTracker`
A no-op progress tracker for quiet mode.

#### `class TaskContextManager`
Manager for task execution state with checkpoint support.

##### Constructor

```python
def __init__(self,
             task_id: str,
             task_dir: Path,
             max_state_size: int = DEFAULT_MAX_STATE_SIZE,
             progress_manager: Optional[Any] = None)
```

##### Methods

```python
def save_execution_state(self, state: Dict[str, Any], checkpoint_name: Optional[str] = None) -> Path
```
Save execution state to a checkpoint file.

```python
def restore_execution_state(self, checkpoint_name: Optional[str] = None) -> Dict[str, Any]
```
Restore execution state from a checkpoint file.

```python
def create_automatic_checkpoint(self, operation_index: int, metrics: Dict[str, Any]) -> str
```
Create an automatic checkpoint at the current execution point.

```python
def update_state(self, updates: Dict[str, Any]) -> None
```
Update the current execution state.

```python
def record_operation_completion(self, operation_index: int, operation_name: str,
                                result_metrics: Optional[Dict[str, Any]] = None) -> None
```
Record the completion of an operation.

```python
def record_operation_failure(self, operation_index: int, operation_name: str,
                             error_info: Dict[str, Any]) -> None
```
Record the failure of an operation.

```python
def record_artifact(self, artifact_path: Union[str, Path], artifact_type: str,
                    description: Optional[str] = None) -> None
```
Record an artifact created during task execution.

```python
def can_resume_execution(self) -> Tuple[bool, Optional[str]]
```
Check if task execution can be resumed from a checkpoint.

```python
def get_latest_checkpoint(self) -> Optional[str]
```
Get the name of the latest checkpoint from the execution log.

```python
def get_current_state(self) -> Dict[str, Any]
```
Get the current execution state.

```python
def get_checkpoints(self) -> List[Tuple[str, datetime]]
```
Get a list of available checkpoints with timestamps.

```python
def cleanup_old_checkpoints(self, max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS) -> int
```
Remove old checkpoints to manage disk space.

```python
def cleanup(self) -> None
```
Clean up resources.

### Functions

```python
def create_task_context_manager(task_id: str,
                                task_dir: Path,
                                max_state_size: int = DEFAULT_MAX_STATE_SIZE,
                                progress_manager: Optional[Any] = None) -> TaskContextManager
```
Create a context manager for a task.

---

## 7. TaskDependencyManager (dependency_manager.py)

`TaskDependencyManager` handles dependency validation and resolution.

### Exceptions

#### `class DependencyError(Exception)`
Base exception for dependency-related errors.

#### `class DependencyMissingError(DependencyError)`
Exception raised when a required dependency is missing.

#### `class DependencyFailedError(DependencyError)`
Exception raised when a dependency task has failed.

### Classes

#### `class TaskDependencyManager`
Manager for task dependencies.

##### Constructor

```python
def __init__(self, task_config, logger: logging.Logger)
```

##### Methods

```python
def get_dependency_output(self, dependency_id: str, file_pattern: Optional[str] = None) -> Union[Path, List[Path]]
```
Get the output directory or files from a dependency.

```python
def get_dependency_report(self, dependency_id: str) -> Path
```
Get the report file from a dependency.

```python
def assert_dependencies_completed(self) -> bool
```
Check if all dependencies have completed successfully.

```python
def is_absolute_dependency(self, dependency_id: str) -> bool
```
Check if a dependency ID represents an absolute path.

```python
def get_dependency_metrics(self, dependency_id: str, metric_path: Optional[str] = None) -> Dict[str, Any]
```
Get metrics from a dependency report.

```python
def get_dependency_status(self, dependency_id: str) -> Dict[str, Any]
```
Get status information about a dependency.

---

## 8. TaskOperationExecutor (operation_executor.py)

`TaskOperationExecutor` handles operation execution with retry capabilities.

### Exceptions

#### `class ExecutionError(Exception)`
Base exception for operation execution errors.

#### `class MaxRetriesExceededError(ExecutionError)`
Exception raised when maximum retry attempts are reached.

#### `class NonRetriableError(ExecutionError)`
Exception raised for errors that should not be retried.

### Classes

#### `class TaskOperationExecutor`
Executor for task operations with retry capabilities.

##### Constructor

```python
def __init__(self,
             task_config: Any,
             logger: logging.Logger,
             reporter: Optional[TaskReporter] = None,
             default_max_retries: int = 3,
             default_backoff_factor: float = 2.0,
             default_initial_wait: float = 1.0,
             default_max_wait: float = 60.0,
             default_jitter: bool = True)
```

##### Methods

```python
def add_retriable_exception(self, exception_type: ExceptionType) -> None
```
Add an exception type to the set of retriable exceptions.

```python
def remove_retriable_exception(self, exception_type: ExceptionType) -> None
```
Remove an exception type from the set of retriable exceptions.

```python
def is_retriable_error(self, exception: Exception) -> bool
```
Determine if an exception should trigger a retry.

```python
def execute_operation(self,
                      operation: BaseOperation,
                      params: Dict[str, Any],
                      progress_tracker: Optional[ProgressTracker] = None) -> OperationResult
```
Execute a single operation without retry logic.

```python
def execute_with_retry(self,
                       operation: BaseOperation,
                       params: Dict[str, Any],
                       max_retries: Optional[int] = None,
                       backoff_factor: Optional[float] = None,
                       initial_wait: Optional[float] = None,
                       max_wait: Optional[float] = None,
                       jitter: Optional[bool] = None,
                       progress_tracker: Optional[ProgressTracker] = None,
                       on_retry: Optional[Callable[[Exception, int, float], None]] = None) -> OperationResult
```
Execute an operation with retry logic.

```python
def execute_operations(self,
                       operations: List[BaseOperation],
                       common_params: Dict[str, Any],
                       progress_tracker: Optional[ProgressTracker] = None,
                       continue_on_error: Optional[bool] = None) -> Dict[str, OperationResult]
```
Execute a list of operations sequentially.

```python
def execute_operations_parallel(self,
                                operations: List[BaseOperation],
                                common_params: Dict[str, Any],
                                max_workers: Optional[int] = None,
                                progress_tracker: Optional[ProgressTracker] = None,
                                continue_on_error: Optional[bool] = None) -> Dict[str, OperationResult]
```
Execute operations in parallel using multiple processes.

```python
def get_execution_stats(self) -> Dict[str, Any]
```
Get execution statistics.

### Functions

```python
def create_operation_executor(task_config: Any,
                              logger: logging.Logger,
                              reporter: Optional[TaskReporter] = None,
                              **kwargs) -> TaskOperationExecutor
```
Create an operation executor for a task.

---

## 9. TaskProgressManager (progress_manager.py)

`TaskProgressManager` provides centralized progress tracking and logging coordination.

### Classes

#### `class ProgressTrackerProtocol(Protocol)`
Protocol defining the interface for progress trackers.

#### `class NoOpProgressTracker`
No-operation progress tracker for quiet mode.

#### `class ProgressTracker`
Progress tracker for individual operations with fixed positioning.

##### Constructor

```python
def __init__(self,
             total: int,
             description: str,
             unit: str = "items",
             position: int = 0,
             leave: bool = True,
             parent: Optional['ProgressTracker'] = None,
             color: Optional[str] = None,
             disable: bool = False)
```

##### Methods

```python
def update(self, steps: int = 1, postfix: Optional[Dict[str, Any]] = None) -> None
```
Update progress by specified number of steps.

```python
def set_description(self, description: str) -> None
```
Update the description of the progress bar.

```python
def set_postfix(self, postfix: Dict[str, Any]) -> None
```
Set the postfix metrics display.

```python
def close(self, failed: bool = False) -> None
```
Close the progress bar and compute final metrics.

```python
def clear(self) -> None
```
Clear the progress bar from display.

```python
def refresh(self) -> None
```
Redraw the progress bar.

#### `class TaskProgressManager`
Centralized manager for task progress and logging coordination.

##### Constructor

```python
def __init__(self,
             task_id: str,
             task_type: str,
             logger: logging.Logger,
             reporter: Optional[Any] = None,
             total_operations: int = 0,
             quiet: bool = False)
```

##### Methods

```python
def start_operation(self,
                    name: str,
                    total: int,
                    description: Optional[str] = None,
                    unit: str = "items",
                    leave: bool = False) -> Union[ProgressTracker, NoOpProgressTracker]
```
Start tracking a new operation.

```python
def update_operation(self,
                     name: str,
                     steps: int = 1,
                     postfix: Optional[Dict[str, Any]] = None) -> None
```
Update progress of an operation.

```python
def complete_operation(self,
                       name: str,
                       success: bool = True,
                       metrics: Optional[Dict[str, Any]] = None) -> None
```
Mark an operation as completed.

```python
def log_message(self,
                level: int,
                message: str,
                preserve_progress: bool = True) -> None
```
Log a message without breaking progress bars.

```python
def log_info(self, message: str) -> None
```
Convenience method for logging info messages.

```python
def log_warning(self, message: str) -> None
```
Convenience method for logging warning messages.

```python
def log_error(self, message: str) -> None
```
Convenience method for logging error messages.

```python
def log_debug(self, message: str) -> None
```
Convenience method for logging debug messages.

```python
def log_critical(self, message: str, preserve_progress: bool = False) -> None
```
Convenience method for logging critical messages.

```python
def create_operation_context(self,
                             name: str,
                             total: int,
                             description: Optional[str] = None,
                             unit: str = "items",
                             leave: bool = False) -> 'ProgressContext'
```
Create a context manager for an operation.

```python
def set_total_operations(self, total: int) -> None
```
Set the total number of operations.

```python
def operation_completed(self, name: str, success: bool = True, metrics: Optional[Dict[str, Any]] = None) -> None
```
Notify that an operation has completed.

```python
def increment_total_operations(self) -> None
```
Increment the total number of operations.

```python
def get_metrics(self) -> Dict[str, Any]
```
Get overall metrics for the task.

```python
def close(self) -> None
```
Close all progress bars and release resources.

#### `class ProgressContext`
Context manager for operation execution with progress tracking.

##### Constructor

```python
def __init__(self,
             progress_manager: TaskProgressManager,
             operation_name: str,
             total: int,
             description: Optional[str] = None,
             unit: str = "items",
             leave: bool = False)
```

### Functions

```python
def create_task_progress_manager(task_id: str,
                                 task_type: str,
                                 logger: logging.Logger,
                                 reporter: Optional[Any] = None,
                                 total_operations: int = 0,
                                 quiet: Optional[bool] = None) -> TaskProgressManager
```
Create a task progress manager with auto-detection of quiet mode.

---

## 10. Execution Log (execution_log.py)

The execution log module manages persistent task execution history at the project level.

### Exceptions

#### `class ExecutionLogError(Exception)`
Exception raised for execution log errors.

### Functions

```python
def initialize_execution_log(project_path: Optional[Path] = None,
                             progress_manager: Optional[ProgressManagerProtocol] = None) -> Path
```
Initialize the execution log file.

```python
def record_task_execution(task_id: str,
                          task_type: str,
                          success: bool,
                          execution_time: float,
                          report_path: Path,
                          input_datasets: Optional[Dict[str, str]] = None,
                          output_artifacts: Optional[List[Any]] = None,
                          progress_manager: Optional[ProgressManagerProtocol] = None) -> Optional[str]
```
Record a task execution in the execution log.

```python
def get_task_execution_history(task_id: Optional[str] = None,
                               limit: int = 10,
                               success_only: bool = False,
                               progress_manager: Optional[ProgressManagerProtocol] = None) -> List[Dict[str, Any]]
```
Get execution history for a specific task or all tasks.

```python
def find_latest_execution(task_id: str, success_only: bool = True,
                          progress_manager: Optional[ProgressManagerProtocol] = None) -> Optional[Dict[str, Any]]
```
Find the most recent execution of a task.

```python
def find_task_by_output(file_path: Union[str, Path],
                        progress_manager: Optional[ProgressManagerProtocol] = None) -> Optional[Dict[str, Any]]
```
Find the task that produced a specific output file.

```python
def track_input_files(task_id: str, file_paths: List[Union[str, Path]],
                      progress_manager: Optional[ProgressManagerProtocol] = None) -> bool
```
Register input files for a task.

```python
def track_output_files(task_id: str, file_paths: List[Union[str, Path]],
                       progress_manager: Optional[ProgressManagerProtocol] = None) -> bool
```
Register output files from a task.

```python
def update_execution_record(task_run_id: str, updates: Dict[str, Any],
                            progress_manager: Optional[ProgressManagerProtocol] = None) -> bool
```
Update an existing execution record.

```python
def remove_execution_record(task_run_id: str,
                            progress_manager: Optional[ProgressManagerProtocol] = None) -> bool
```
Remove an execution record from the log.

```python
def cleanup_old_executions(max_age_days: int = 30,
                           max_per_task: int = 10,
                           dry_run: bool = False,
                           progress_manager: Optional[ProgressManagerProtocol] = None) -> Tuple[int, List[str]]
```
Clean up old execution records.

```python
def validate_execution_log(progress_manager: Optional[ProgressManagerProtocol] = None) -> Tuple[bool, List[str]]
```
Validate the execution log.

```python
def export_execution_log(output_path: Optional[Path] = None,
                         format: str = "json",
                         progress_manager: Optional[ProgressManagerProtocol] = None) -> Path
```
Export the execution log to a file.

```python
def import_execution_log(input_path: Path, merge: bool = False,
                       progress_manager: Optional[ProgressManagerProtocol] = None) -> bool
```
Import an execution log from a file.

---

## 11. TaskRegistry (task_registry.py)

The task registry module handles task class registration, discovery, and instantiation.

### Exceptions

#### `class TaskRegistryError(Exception)`
Exception raised for task registry errors.

### Functions

```python
def register_task_class(task_id: str, task_class: Type) -> bool
```
Register a task class by ID.

```python
def get_task_class(task_id: str) -> Optional[Type]
```
Get a task class by ID.

```python
def list_registered_tasks() -> Dict[str, Dict[str, Any]]
```
List all registered task types with their metadata.

```python
def create_task_instance(task_id: str, **kwargs) -> Optional[Any]
```
Create a task instance by ID.

```python
def discover_task_classes(package_paths: Optional[List[str]] = None,
                          recursive: bool = True) -> Dict[str, Type]
```
Discover task classes in specified packages.

```python
def register_discovered_tasks(package_paths: Optional[List[str]] = None,
                              recursive: bool = True) -> int
```
Discover and register task classes in specified packages.

```python
def get_task_metadata(task_class: Type) -> Dict[str, Any]
```
Extract metadata from a task class.

```python
def check_task_dependencies(task_id: str, task_type: str, dependencies: List[str]) -> bool
```
Check if dependencies for a task are satisfied.

---

## 12. PathSecurity (path_security.py)

The path security module provides utilities for validating path security.

### Exceptions

#### `class PathSecurityError(Exception)`
Exception raised for path security violations.

### Functions

```python
def validate_path_security(path: Union[str, Path],
                           allowed_paths: Optional[List[Union[str, Path]]] = None,
                           allow_external: bool = False,
                           strict_mode: bool = True) -> bool
```
Validate that a path is safe to use.

```python
def is_within_allowed_paths(path: Path,
                           allowed_paths: List[Union[str, Path]]) -> bool
```
Check if a path is within any of the allowed paths.

```python
def get_system_specific_dangerous_paths() -> List[str]
```
Get a list of system-specific paths that should be protected.

```python
def validate_paths(paths: List[Union[str, Path]],
                   allowed_paths: Optional[List[Union[str, Path]]] = None,
                   allow_external: bool = False) -> Tuple[bool, List[str]]
```
Validate multiple paths at once.

```python
def is_potentially_dangerous_path(path: Union[str, Path]) -> bool
```
Check if a path might be potentially dangerous without raising exceptions.

```python
def normalize_and_validate_path(path: Union[str, Path],
                                base_dir: Optional[Path] = None,
                                allowed_paths: Optional[List[Union[str, Path]]] = None,
                                allow_external: bool = False) -> Path
```
Normalize a path and validate its security.

---

## 13. TaskUtils (task_utils.py)

The task utilities module provides utility functions for working with tasks.

### Functions

```python
def create_task_directories(task_dir: Path) -> Dict[str, Path]
```
Create standard directories for a task.

```python
def prepare_data_source_from_paths(file_paths: Dict[str, str],
                                   show_progress: bool = True) -> DataSource
```
Prepare a data source from file paths.

```python
def format_execution_time(seconds: float) -> str
```
Format execution time in seconds to a human-readable string.

```python
def get_artifact_path(task_dir: Path,
                      artifact_name: str,
                      artifact_type: str = "json",
                      sub_dir: str = "output",
                      include_timestamp: bool = True) -> Path
```
Get a standardized path for a task artifact.

```python
def find_previous_output(task_id: str,
                         data_repository: Optional[Path] = None,
                         project_root: Optional[Path] = None,
                         file_pattern: Optional[str] = None) -> List[Path]
```
Find output files from a previous task.

```python
def find_task_report(task_id: str,
                     data_repository: Optional[Path] = None,
                     project_root: Optional[Path] = None) -> Optional[Path]
```
Find the report file from a previous task.

```python
def get_temp_dir(task_dir: Path) -> Path
```
Get a temporary directory for the task.

```python
def clean_temp_dir(task_dir: Path) -> bool
```
Clean the temporary directory for the task.

```python
def format_error_for_report(error: Exception) -> Dict[str, Any]
```
Format an exception for inclusion in a task report.

```python
def ensure_secure_directory(path: Union[str, Path]) -> Path
```
Create a directory with secure permissions.

```python
def is_master_key_exposed() -> bool
```
Check if the master encryption key has insecure permissions.

```python
def extract_previous_output_info(task_id: str,
                                 data_repository: Optional[Path] = None) -> Dict[str, Any]
```
Extract information about outputs from a previous task.

---

## 14. Project Config Loader (project_config_loader.py)

The project config loader module provides functionality for loading project-level configurations.

### Functions

```python
def find_project_root() -> Path
```
Locate the project root directory.

```python
def substitute_variables(config_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]
```
Perform variable substitution in configuration values using Jinja2.

```python
def load_project_config(project_root: Optional[Path] = None,
                        config_filename: Optional[str] = None,
                        use_cache: bool = True) -> Dict[str, Any]
```
Load the project configuration from a YAML file with JSON fallback.

```python
def apply_default_values(config_data: Dict[str, Any]) -> Dict[str, Any]
```
Apply default values to the configuration where values are missing.

```python
def clear_config_cache() -> None
```
Clear the configuration cache.

```python
def get_project_paths(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Path]
```
Get standard project paths from configuration.

```python
def save_project_config(config_data: Dict[str, Any],
                        project_root: Optional[Path] = None,
                        format: str = "yaml") -> Path
```
Save the project configuration to a file.

```python
def is_valid_project_root(path: Path) -> bool
```
Check if a path is a valid project root.

```python
def create_default_project_structure(root_path: Path, data_path: Optional[Path] = None) -> Dict[str, Path]
```
Create a default project structure at the specified location.

```python
def get_recursive_variables(config_data: Dict[str, Any]) -> Dict[str, Any]
```
Extract all variables from configuration data that can be used in substitution.