# BaseTask Module Documentation

## Overview

The `base_task.py` module provides the foundational class for all task implementations in the PAMOLA Core framework. It serves as a facade that orchestrates the entire task lifecycle while delegating specialized responsibilities to dedicated components. The BaseTask class defines a standardized interface and workflow for privacy-enhancing data processing tasks, ensuring consistent behavior, error handling, and resource management.

## Key Features

- **Facade Design Pattern**: Delegates specialized responsibilities to dedicated components
- **Standardized Task Lifecycle**: Well-defined initialization, execution, and finalization phases
- **Configuration Management**: Handles configuration loading and cascading priorities
- **Component Orchestration**: Coordinates multiple specialized components for different responsibilities
- **Operation Management**: Simplified operation registration and execution
- **Checkpoint Support**: Provides task resumability from points of failure
- **Progress Tracking**: Integrated progress visualization
- **Encryption Integration**: Secure handling of sensitive data
- **Directory Management**: Standardized directory structure handling
- **Error Handling**: Comprehensive error detection, reporting, and recovery
- **Context Manager Support**: Safe resource handling with `with` statement

## Dependencies

The `BaseTask` class relies on several specialized components:

- `task_config`: Configuration loading and management
- `directory_manager`: Directory structure management
- `encryption_manager`: Encryption and sensitive data handling
- `operation_executor`: Operation execution with retry capabilities
- `context_manager`: Task state and checkpoint management
- `progress_manager`: Progress tracking and visualization
- `dependency_manager`: Task dependency validation
- `task_reporting`: Reporting and artifact tracking
- `execution_log`: Task execution history management

## Task Lifecycle

Tasks in the PAMOLA Core framework follow a standard four-phase lifecycle:

1. **Initialization Phase**:
   - Load configuration
   - Set up logging
   - Create directory structure
   - Initialize component managers
   - Check dependencies
   - Prepare data sources

2. **Configuration Phase**:
   - Define operations to execute
   - Set operation parameters
   - Establish execution sequence

3. **Execution Phase**:
   - Execute operations in sequence
   - Track progress
   - Collect results and artifacts
   - Create checkpoints

4. **Finalization Phase**:
   - Generate reports
   - Record execution history
   - Clean up resources
   - Handle final status

## Exception Classes

### TaskError

Base exception for all task-related errors.

### TaskInitializationError

Exception raised when task initialization fails due to:
- Missing or invalid configuration
- Failed directory creation
- Logging setup failure
- Data source initialization problems
- Path security violations

### TaskDependencyError

Exception raised when task dependencies are not satisfied, when previous tasks:
- Have not been executed
- Failed during execution
- Did not produce required outputs

## BaseTask Class

### Constructor

```python
def __init__(
    self,
    task_id: str,
    task_type: str,
    description: str,
    input_datasets: Optional[Dict[str, str]] = None,
    auxiliary_datasets: Optional[Dict[str, str]] = None,
    version: str = "1.0.0"
)
```

**Parameters:**
- `task_id`: Unique identifier for the task
- `task_type`: Type of task (e.g., profiling, anonymization)
- `description`: Human-readable description of the task's purpose
- `input_datasets`: Dictionary mapping dataset names to file paths (primary inputs)
- `auxiliary_datasets`: Dictionary mapping auxiliary dataset names to file paths (secondary inputs)
- `version`: Version of the task implementation

### Key Attributes

- `task_id`, `task_type`, `description`, `version`: Task metadata
- `input_datasets`, `auxiliary_datasets`: Dataset references
- `config`: Task configuration
- `logger`: Task-specific logger
- `reporter`: Task reporter for generating reports
- `operations`: List of operations to execute
- `results`: Dictionary mapping operation names to results
- `artifacts`: List of artifacts produced by operations
- `metrics`: Dictionary of metrics collected during execution
- `status`: Current task status
- `error_info`: Detailed error information if an error occurred

### Component Managers

- `directory_manager`: Manages task directory structure
- `context_manager`: Manages task state and checkpoints
- `encryption_manager`: Handles encryption and sensitive data
- `dependency_manager`: Validates task dependencies
- `operation_executor`: Executes operations with retry capability
- `progress_manager`: Tracks and displays execution progress
- `data_source`: Manages input data access
- `data_writer`: Handles output data writing

### Key Methods

#### initialize

```python
def initialize(self, args: Optional[Dict[str, Any]] = None) -> bool
```

Initialize the task by loading configuration, creating directories, setting up logging, and checking dependencies.

**Parameters:**
- `args`: Command line arguments to override configuration

**Returns:**
- `True` if initialization is successful, `False` otherwise

**Raises:**
- `TaskDependencyError`: If task dependencies are not satisfied

#### configure_operations

```python
def configure_operations(self) -> None
```

Configure operations to be executed by the task. This method should be overridden in subclasses to define the specific operations that the task will execute.

**Raises:**
- `NotImplementedError`: If not overridden in a subclass

#### execute

```python
def execute(self) -> bool
```

Execute the task by running operations sequentially, collecting results, and generating metrics.

**Returns:**
- `True` if execution is successful, `False` otherwise

#### finalize

```python
def finalize(self, success: bool) -> bool
```

Finalize the task by releasing resources, closing files, and registering the execution result.

**Parameters:**
- `success`: Whether the task executed successfully

**Returns:**
- `True` if finalization is successful, `False` otherwise

#### run

```python
def run(self, args: Optional[Dict[str, Any]] = None) -> bool
```

Run the complete task lifecycle: initialize, configure, execute, finalize. This is the main method that orchestrates the entire task.

**Parameters:**
- `args`: Command line arguments to override configuration

**Returns:**
- `True` if the task executed successfully, `False` otherwise

#### add_operation

```python
def add_operation(self, operation_class: Union[str, Type[BaseOperation]], **kwargs) -> bool
```

Add an operation to the task's execution queue.

**Parameters:**
- `operation_class`: Name of the operation class or the class itself
- `**kwargs`: Parameters for the operation constructor

**Returns:**
- `True` if operation was added successfully, `False` otherwise

#### get_results

```python
def get_results(self) -> Dict[str, OperationResult]
```

Get the results of all operations executed by the task.

**Returns:**
- Dictionary mapping operation names to their results

#### get_artifacts

```python
def get_artifacts(self) -> List[Any]
```

Get all artifacts produced by the task.

**Returns:**
- List of artifact objects with metadata

#### get_metrics

```python
def get_metrics(self) -> Dict[str, Any]
```

Get all metrics collected by the task.

**Returns:**
- Dictionary containing metrics organized by category

#### get_execution_status

```python
def get_execution_status(self) -> Tuple[str, Optional[Dict[str, Any]]]
```

Get the execution status and error information.

**Returns:**
- Tuple containing:
  - Status string (e.g., "pending", "success", "error")
  - Error information dictionary (or None if no error)

#### get_encryption_info

```python
def get_encryption_info(self) -> Dict[str, Any]
```

Get information about the task's encryption settings.

**Returns:**
- Dictionary with encryption information including mode and status

#### get_checkpoint_status

```python
def get_checkpoint_status(self) -> Dict[str, Any]
```

Get information about the checkpoint status of the task.

**Returns:**
- Dictionary with checkpoint information including restoration status

### Context Manager Support

The class implements the context manager protocol (`__enter__` and `__exit__`), allowing it to be used with the `with` statement:

```python
with BaseTask(task_id="t_1P1", task_type="profiling", description="Profile data") as task:
    # Task will be properly finalized even if an exception occurs
    task.run()
```

## Internal Methods

### _setup_logging

```python
def _setup_logging(self) -> None
```

Set up dual logging to both project-level and task-specific logs.

### _initialize_data_source

```python
def _initialize_data_source(self) -> None
```

Initialize the data source with input and auxiliary datasets.

### _initialize_data_writer

```python
def _initialize_data_writer(self) -> None
```

Initialize the data writer for task outputs.

### _run_operations

```python
def _run_operations(self, start_idx: int = 0) -> bool
```

Run operations starting from the specified index.

**Parameters:**
- `start_idx`: Index of the first operation to execute

**Returns:**
- `True` if all operations executed successfully, `False` otherwise

### _prepare_operation_parameters

```python
def _prepare_operation_parameters(self, operation: BaseOperation) -> Dict[str, Any]
```

Prepare parameters for an operation, including system parameters and operation-specific parameters.

**Parameters:**
- `operation`: The operation to prepare parameters for

**Returns:**
- Dictionary of parameters for the operation

### _get_operation_supported_parameters

```python
def _get_operation_supported_parameters(self, operation: Union[BaseOperation, Type[BaseOperation]]) -> set
```

Get the set of parameters that an operation's constructor accepts. This method uses efficient caching to avoid repeated inspections of the same operation classes.

**Parameters:**
- `operation`: Operation instance or class

**Returns:**
- Set of parameter names that the operation accepts

## Reserved Operation Parameters

The following parameter names are reserved for the framework and should not be included in operation configuration:

- `data_source`: Data source for operation input
- `task_dir`: Directory for operation artifacts
- `reporter`: Reporter for logging operation progress
- `progress_tracker`: Tracker for operation progress
- `use_encryption`, `encryption_key`, `encryption_context`, `encryption_mode`: Encryption parameters
- `parallel_processes`, `use_vectorization`: Performance parameters

## Task Status Values

The task's `status` attribute can have the following values:

- `"pending"`: Initial state before execution
- `"success"`: Task completed successfully
- `"dependency_error"`: Task dependencies not satisfied
- `"initialization_error"`: Error during initialization
- `"configuration_error"`: Error configuring operations
- `"operation_error"`: Error during operation execution
- `"context_exception"`: Exception in context manager
- `"unhandled_exception"`: Unexpected error during execution
- `"report_error"`: Error generating report
- `"log_error"`: Error recording execution log
- `"finalization_error"`: Error during finalization

## Usage Examples

### Basic Task Implementation

```python
from pamola_core.utils.tasks.base_task import BaseTask

class SimpleProfileTask(BaseTask):
    """Simple profiling task implementation."""
    
    def __init__(self, task_id: str, input_file: str):
        """Initialize the task."""
        super().__init__(
            task_id=task_id,
            task_type="profiling",
            description="Simple data profiling task",
            input_datasets={"main_data": input_file}
        )
    
    def configure_operations(self):
        """Configure operations to execute."""
        # Add data loading operation
        self.add_operation(
            "LoadCsvOperation",
            input_dataset="main_data",
            output_name="data"
        )
        
        # Add profiling operation
        self.add_operation(
            "ProfileDataOperation",
            dataset_name="data",
            output_prefix="profile"
        )
        
        # Add report generation operation
        self.add_operation(
            "GenerateReportOperation",
            input_profile="profile",
            report_type="html"
        )

# Usage
task = SimpleProfileTask("t_profile", "data/customers.csv")
success = task.run()

if success:
    print("Task completed successfully!")
    artifacts = task.get_artifacts()
    print(f"Generated {len(artifacts)} artifacts")
else:
    status, error_info = task.get_execution_status()
    print(f"Task failed with status: {status}")
    print(f"Error: {error_info.get('message', 'Unknown error')}")
```

### Using Component Managers Directly

```python
from pamola_core.utils.tasks.base_task import BaseTask

class AdvancedTask(BaseTask):
    """Advanced task with direct component access."""
    
    def configure_operations(self):
        """Configure operations with advanced options."""
        # Use directory manager to get artifact paths
        output_path = self.directory_manager.get_artifact_path(
            artifact_name="results",
            artifact_type="json",
            subdir="output"
        )
        
        # Add operation with custom path
        self.add_operation(
            "CustomOperation",
            output_path=output_path
        )
    
    def execute(self):
        """Custom execution with progress tracking."""
        # Use progress manager for custom operation
        with self.progress_manager.create_operation_context(
            name="custom_process",
            total=100,
            description="Processing data"
        ) as progress:
            # Perform custom processing
            for i in range(100):
                # Do work
                progress.update(1)
            
            # Log a message that won't break progress display
            self.progress_manager.log_info("Custom processing completed")
        
        # Continue with standard execution
        return super().execute()
    
    def create_encrypted_artifact(self, data: bytes, name: str):
        """Create an encrypted artifact using encryption manager."""
        if self.use_encryption:
            # Encrypt data
            encrypted_data = self.encryption_manager.encrypt_data(data)
            
            # Get output path
            path = self.directory_manager.get_artifact_path(
                artifact_name=name,
                artifact_type="bin",
                subdir="output"
            )
            
            # Write encrypted data
            with open(path, "wb") as f:
                f.write(encrypted_data)
            
            # Register artifact
            self.reporter.add_artifact(
                artifact_type="encrypted_data",
                path=path,
                description=f"Encrypted {name}",
                metadata={"encrypted": True}
            )
            
            return path
        else:
            self.logger.warning("Encryption not enabled")
            return None
```

### Task with Dependency Management

```python
from pamola_core.utils.tasks.base_task import BaseTask

class DependentTask(BaseTask):
    """Task that depends on other tasks."""
    
    def __init__(self, task_id: str, description: str, dependencies: List[str]):
        """Initialize task with dependencies."""
        super().__init__(
            task_id=task_id,
            task_type="dependent",
            description=description
        )
        # Store dependencies for configuration
        self.task_dependencies = dependencies
    
    def initialize(self, args=None):
        """Initialize and verify dependencies."""
        # Basic initialization
        if not super().initialize(args):
            return False
        
        # Check if we can access dependency outputs
        for dep_id in self.task_dependencies:
            try:
                # Get dependency output path
                dep_output = self.dependency_manager.get_dependency_output(dep_id)
                self.logger.info(f"Found dependency output for {dep_id}: {dep_output}")
                
                # Get dependency report
                dep_report = self.dependency_manager.get_dependency_report(dep_id)
                self.logger.info(f"Found dependency report for {dep_id}: {dep_report}")
            except Exception as e:
                self.logger.error(f"Error accessing dependency {dep_id}: {e}")
                return False
        
        return True
    
    def configure_operations(self):
        """Configure operations using dependency outputs."""
        # Use dependency outputs in operations
        for dep_id in self.task_dependencies:
            dep_output = self.dependency_manager.get_dependency_output(dep_id)
            
            # Add operation to process dependency output
            self.add_operation(
                "ProcessDependencyOperation",
                dependency_id=dep_id,
                input_path=dep_output
            )
```

### Using Checkpoints for Resumable Tasks

```python
from pamola_core.utils.tasks.base_task import BaseTask

class ResumableTask(BaseTask):
    """Task that supports resuming from checkpoint."""
    
    def execute(self):
        """Execute with checkpoint awareness."""
        # Check if we're resuming from checkpoint
        checkpoint_status = self.get_checkpoint_status()
        
        if checkpoint_status["resuming_from_checkpoint"]:
            self.logger.info(f"Resuming from checkpoint: {checkpoint_status['checkpoint_name']}")
            self.logger.info(f"Last completed operation index: {checkpoint_status['operation_index']}")
            
            # Could perform custom restoration here if needed
        
        # Standard execution (will automatically skip completed operations)
        return super().execute()
    
    def _run_operations(self, start_idx=0):
        """Custom operation execution with manual checkpointing."""
        for i in range(start_idx, len(self.operations)):
            operation = self.operations[i]
            operation_name = getattr(operation, 'name', f"Operation {i + 1}")
            
            # Execute operation
            try:
                # Standard execution
                result = self.operation_executor.execute_with_retry(
                    operation=operation,
                    params=self._prepare_operation_parameters(operation)
                )
                
                # Store result
                self.results[operation_name] = result
                
                # Create manual checkpoint with custom data
                self.context_manager.create_automatic_checkpoint(
                    operation_index=i,
                    metrics=self.metrics,
                    custom_data={
                        "last_operation": operation_name,
                        "timestamp": time.time()
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Error in operation {operation_name}: {e}")
                
                if not self.config.continue_on_error:
                    self.status = "operation_error"
                    return False
        
        self.status = "success"
        return True
```

## Integration with Component Managers

The `BaseTask` class integrates with several specialized component managers:

### Directory Manager

```python
# In BaseTask.initialize()
self.directory_manager = create_directory_manager(
    task_config=self.config,
    logger=self.logger
)

# Ensure directories are created
self.directories = self.directory_manager.ensure_directories()
self.task_dir = self.directory_manager.get_directory("task")
```

### Encryption Manager

```python
# In BaseTask.initialize()
self.encryption_manager = TaskEncryptionManager(
    task_config=self.config,
    logger=self.logger
)

# Initialize encryption
self.encryption_manager.initialize()

# Update encryption settings for backward compatibility
encryption_info = self.encryption_manager.get_encryption_info()
self.use_encryption = encryption_info["enabled"]
self.encryption_mode = EncryptionMode.from_string(encryption_info["mode"])
```

### Context Manager

```python
# In BaseTask.initialize()
self.context_manager = create_task_context_manager(
    task_id=self.task_id,
    task_dir=self.task_dir
)

# Check if task can be resumed from checkpoint
self._resuming_from_checkpoint, self._restored_checkpoint_name = self.context_manager.can_resume_execution()
if self._resuming_from_checkpoint:
    # Restore execution state
    self._restored_state = self.context_manager.restore_execution_state(self._restored_checkpoint_name)
```

### Progress Manager

```python
# In BaseTask.initialize()
self.progress_manager = create_task_progress_manager(
    task_id=self.task_id,
    task_type=self.task_type,
    logger=self.logger,
    reporter=self.reporter,
    total_operations=0  # Will be updated after configure_operations()
)

# In BaseTask.configure_operations() or after
if self.progress_manager:
    self.progress_manager.set_total_operations(len(self.operations))
```

### Operation Executor

```python
# In BaseTask.initialize()
self.operation_executor = create_operation_executor(
    task_config=self.config,
    logger=self.logger,
    reporter=self.reporter,
    progress_manager=self.progress_manager
)

# In BaseTask._run_operations()
result = self.operation_executor.execute_with_retry(
    operation=operation,
    params=operation_params
)
```

### Dependency Manager

```python
# In BaseTask.initialize()
self.dependency_manager = TaskDependencyManager(
    task_config=self.config,
    logger=self.logger
)

# Check dependencies using the dependency manager
if not self.dependency_manager.assert_dependencies_completed():
    raise TaskDependencyError(f"Task dependencies not satisfied for {self.task_id}")
```

## Task Execution Flow

The complete task execution flow is as follows:

1. **Initialization**:
   - Load configuration
   - Set up logging
   - Initialize directory manager
   - Create directories
   - Create reporter
   - Initialize dependency manager
   - Check dependencies
   - Initialize encryption manager
   - Initialize context manager
   - Check for checkpoints
   - Initialize progress manager
   - Initialize operation executor
   - Initialize data source
   - Initialize data writer

2. **Configuration**:
   - Call `configure_operations()` (user-defined)
   - Add operations to execution queue
   - Update progress manager with total operations

3. **Execution**:
   - Check if operations are configured
   - Check for checkpoint resumption
   - For each operation:
     - Prepare operation parameters
     - Execute operation with retry
     - Store result
     - Collect artifacts and metrics
     - Create checkpoint
     - Handle errors based on configuration

4. **Finalization**:
   - Calculate execution time
   - Add final status to reporter
   - Register artifacts
   - Generate and save report
   - Record execution in execution log
   - Clean up resources
   - Close components

## Error Handling Strategy

The `BaseTask` class implements a comprehensive error handling strategy:

1. **Error Classification**:
   - Initialization errors
   - Dependency errors
   - Configuration errors
   - Operation errors
   - Unhandled exceptions
   - Finalization errors

2. **Error Reporting**:
   - Detailed error information in `error_info` dictionary
   - Log messages with stack traces
   - Error recording in task report

3. **Recovery Options**:
   - `continue_on_error` configuration for resilience
   - Checkpointing for resumability
   - Context manager for resource cleanup

4. **User Feedback**:
   - Clear status messages
   - Detailed error information
   - Execution metrics even on failure

## Best Practices

1. **Implement `configure_operations()`**: Always override this method in subclasses to define task-specific operations.

2. **Use `add_operation()`**: Use this method to add operations to the task rather than manipulating `self.operations` directly.

3. **Respect Component Roles**: Use the appropriate component manager for each responsibility.

4. **Error Handling**: Use try-except blocks for operations that might fail and set appropriate error information.

5. **Progress Reporting**: Use the progress manager for tracking to provide a good user experience.

6. **Clean Up Resources**: Ensure resources are cleaned up in `finalize()` or by using the task as a context manager.

7. **Checkpoint Management**: Create checkpoints at appropriate points to support resumable execution.

8. **Parameter Inspection**: Use `_get_operation_supported_parameters()` to check which parameters an operation supports.

9. **Component Initialization Order**: Initialize components in the correct order to respect dependencies.

10. **Configuration Management**: Use the task's configuration for operation parameters rather than hardcoding values.

## Limitations and Considerations

1. **Checkpoint Size**: Checkpoints can become large if storing extensive state.

2. **Memory Usage**: Be aware of memory usage when processing large datasets.

3. **Threading Issues**: The task is not inherently thread-safe; synchronization needed for concurrent access.

4. **Operation Compatibility**: Ensure operations are compatible with the task's data and configuration.

5. **Error Propagation**: Errors in components can propagate to the task level.

6. **Configuration Validation**: Validate configuration before use to avoid runtime errors.

7. **Path Management**: Use directory manager for path handling to ensure security.

8. **Encryption Overhead**: Be aware of performance impact when encryption is enabled.

9. **Logging Volume**: Configure appropriate logging levels to avoid excessive output.

10. **Backward Compatibility**: Consider backward compatibility when extending the task framework.

## Task Development Example

Here's a complete example of developing a custom task:

```python
from pamola_core.utils.tasks.base_task import BaseTask
from typing import Dict, Any, Optional

class DataProfilerTask(BaseTask):
    """
    Task for profiling data sources.
    
    This task analyzes input datasets to generate statistical profiles,
    data quality metrics, and visualizations.
    """
    
    def __init__(
        self,
        task_id: str,
        input_datasets: Dict[str, str],
        output_formats: Optional[Dict[str, bool]] = None,
        version: str = "1.0.0"
    ):
        """
        Initialize the data profiler task.
        
        Args:
            task_id: Unique identifier for the task
            input_datasets: Dictionary mapping dataset names to file paths
            output_formats: Dictionary of output format flags (json, html, csv)
            version: Version of the task implementation
        """
        super().__init__(
            task_id=task_id,
            task_type="profiling",
            description="Generate profile of input datasets",
            input_datasets=input_datasets,
            version=version
        )
        
        # Default output formats
        self.output_formats = output_formats or {
            "json": True,
            "html": True,
            "csv": False
        }
    
    def configure_operations(self) -> None:
        """Configure profiling operations for each input dataset."""
        # For each input dataset
        for dataset_name, _ in self.input_datasets.items():
            # Add data loading operation
            self.add_operation(
                "LoadDataOperation",
                dataset_name=dataset_name,
                sample_size=self.config.get("sample_size", 100000)
            )
            
            # Add basic profiling operation
            self.add_operation(
                "BasicProfileOperation",
                dataset_name=dataset_name,
                output_prefix=f"{dataset_name}_basic",
                include_histograms=True
            )
            
            # Add correlation analysis if enabled
            if self.config.get("include_correlations", True):
                self.add_operation(
                    "CorrelationAnalysisOperation",
                    dataset_name=dataset_name,
                    output_prefix=f"{dataset_name}_correlation",
                    correlation_method=self.config.get("correlation_method", "pearson")
                )
            
            # Add data quality assessment
            self.add_operation(
                "DataQualityOperation",
                dataset_name=dataset_name,
                output_prefix=f"{dataset_name}_quality"
            )
            
            # Add report generation based on output formats
            if self.output_formats.get("json", True):
                self.add_operation(
                    "GenerateReportOperation",
                    dataset_name=dataset_name,
                    input_profiles=[
                        f"{dataset_name}_basic",
                        f"{dataset_name}_correlation",
                        f"{dataset_name}_quality"
                    ],
                    format="json"
                )
            
            if self.output_formats.get("html", True):
                self.add_operation(
                    "GenerateReportOperation",
                    dataset_name=dataset_name,
                    input_profiles=[
                        f"{dataset_name}_basic",
                        f"{dataset_name}_correlation",
                        f"{dataset_name}_quality"
                    ],
                    format="html",
                    include_visualizations=True
                )
            
            if self.output_formats.get("csv", False):
                self.add_operation(
                    "GenerateReportOperation",
                    dataset_name=dataset_name,
                    input_profiles=[
                        f"{dataset_name}_basic",
                        f"{dataset_name}_quality"
                    ],
                    format="csv"
                )
    
    def finalize(self, success: bool) -> bool:
        """
        Finalize the task with additional summary information.
        
        Args:
            success: Whether the task executed successfully
        
        Returns:
            True if finalization is successful, False otherwise
        """
        # Add task-specific metrics before standard finalization
        if success:
            # Count profiles and reports by type
            profile_counts = {
                "basic_profiles": 0,
                "correlation_profiles": 0,
                "quality_profiles": 0,
                "json_reports": 0,
                "html_reports": 0,
                "csv_reports": 0
            }
            
            # Count artifacts by type
            for artifact in self.artifacts:
                artifact_type = getattr(artifact, 'artifact_type', '')
                if 'basic_profile' in artifact_type:
                    profile_counts["basic_profiles"] += 1
                elif 'correlation' in artifact_type:
                    profile_counts["correlation_profiles"] += 1
                elif 'quality' in artifact_type:
                    profile_counts["quality_profiles"] += 1
                elif 'report' in artifact_type:
                    if 'json' in artifact_type:
                        profile_counts["json_reports"] += 1
                    elif 'html' in artifact_type:
                        profile_counts["html_reports"] += 1
                    elif 'csv' in artifact_type:
                        profile_counts["csv_reports"] += 1
            
            # Add to metrics
            self.metrics["profile_summary"] = profile_counts
        
        # Call standard finalization
        return super().finalize(success)
```

## Related Modules

The `BaseTask` module works closely with these related modules:

- `task_config.py`: Configuration loading and management
- `progress_manager.py`: Progress tracking and visualization
- `directory_manager.py`: Directory structure management
- `encryption_manager.py`: Encryption and sensitive data handling
- `operation_executor.py`: Operation execution with retry capabilities
- `context_manager.py`: Task state and checkpoint management
- `dependency_manager.py`: Task dependency validation
- `task_reporting.py`: Reporting and artifact tracking
- `execution_log.py`: Task execution history management