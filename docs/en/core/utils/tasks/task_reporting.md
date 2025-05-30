# TaskReporting Module Documentation

## Overview

The `task_reporting.py` module provides functionality for creating and managing task reports in the PAMOLA Core framework. It enables comprehensive tracking of operations, artifacts, and metrics during task execution, and generates standardized JSON reports that document the entire task lifecycle. This module is essential for maintaining execution history, facilitating result analysis, and ensuring accountability in privacy-enhancing workflows.

## Key Features

- **Comprehensive Task Reporting**: Captures all aspects of task execution
- **Operation Lifecycle Tracking**: Records execution flow with timestamps
- **Artifact Registration and Organization**: Tracks produced files and resources
- **Artifact Grouping**: Organizes artifacts into logical collections
- **Metric Collection and Aggregation**: Records both task and operation metrics
- **System Environment Capture**: Documents execution context for reproducibility
- **Memory Usage Tracking**: Monitors peak memory consumption
- **Error and Warning Monitoring**: Records issues during execution
- **JSON Report Generation**: Produces structured, machine-readable reports
- **Context Manager Support**: Enables automatic report finalization
- **Progress Manager Integration**: Coordinates with progress tracking for consistent UX

## Dependencies

- `os`, `platform`, `socket`: System information collection
- `datetime`: Timestamp management
- `pathlib.Path`: Path handling
- `typing`: Type annotations
- `pamola_core.utils.io`: File I/O utilities for report saving
- `pamola_core.utils.tasks.task_config`: Path security validation

## Exception Class

### ReportingError

Exception raised for reporting-related errors.

## Classes

### ArtifactGroup

#### Description

Represents a group of related artifacts in a task report, allowing for logical organization of artifacts produced by operations.

#### Constructor

```python
def __init__(self, name: str, description: str = "")
```

**Parameters:**
- `name`: Name of the group
- `description`: Description of the group purpose

#### Methods

##### add_artifact

```python
def add_artifact(self, artifact: Dict[str, Any])
```

Adds an artifact to the group.

**Parameters:**
- `artifact`: Artifact information dictionary

##### to_dict

```python
def to_dict(self) -> Dict[str, Any]
```

Converts the group to a dictionary for serialization.

**Returns:**
- Dictionary representation of the group

### TaskReporter

#### Description

Main class for generating and managing task reports. Tracks operations, artifacts, metrics, errors, and warnings, and generates comprehensive JSON reports.

#### Constructor

```python
def __init__(self, task_id: str, task_type: str, description: str, report_path: Union[str, Path], 
             progress_manager: Optional[Any] = None)
```

**Parameters:**
- `task_id`: ID of the task
- `task_type`: Type of the task
- `description`: Description of the task's purpose
- `report_path`: Path where the report will be saved
- `progress_manager`: Optional progress manager for tracking operations

**Raises:**
- `ReportingError`: If report path fails security validation

#### Key Attributes

- `task_id`: ID of the task
- `task_type`: Type of the task
- `description`: Description of the task
- `report_path`: Path where the report will be saved
- `start_time`: Timestamp when the reporter was created
- `end_time`: Timestamp when the task completed
- `operations`: List of operation entries with status
- `artifacts`: List of artifact entries
- `artifact_groups`: Dictionary of artifact groups
- `system_info`: System environment information
- `status`: Current status of the task
- `success`: Whether the task executed successfully
- `execution_time`: Total execution time
- `metrics`: Dictionary of metrics
- `errors`: List of errors that occurred during execution
- `warnings`: List of warnings that occurred during execution
- `peak_memory_usage`: Maximum memory usage during execution
- `progress_manager`: Optional progress manager for coordinated tracking

#### Key Methods

##### add_operation

```python
def add_operation(self, name: str, status: str = "success", details: Dict[str, Any] = None)
```

Adds an operation to the report.

**Parameters:**
- `name`: Name of the operation
- `status`: Status of the operation (success, warning, error)
- `details`: Additional details about the operation

##### add_artifact

```python
def add_artifact(self, artifact_type: str, path: Union[str, Path], description: str = "",
                 category: str = "output", tags: Optional[List[str]] = None,
                 group_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None)
```

Adds an artifact to the report.

**Parameters:**
- `artifact_type`: Type of the artifact (e.g., "json", "csv", "png")
- `path`: Path to the artifact
- `description`: Description of the artifact
- `category`: Category of the artifact (e.g., "output", "metric", "visualization")
- `tags`: Tags for categorizing the artifact
- `group_name`: Name of the group to add this artifact to
- `metadata`: Additional metadata for the artifact

**Raises:**
- `ReportingError`: If artifact path fails security validation

##### add_artifact_group

```python
def add_artifact_group(self, name: str, description: str = "") -> ArtifactGroup
```

Adds or gets an artifact group.

**Parameters:**
- `name`: Name of the group
- `description`: Description of the group

**Returns:**
- The artifact group

##### get_artifact_group

```python
def get_artifact_group(self, name: str) -> Optional[ArtifactGroup]
```

Gets an artifact group by name.

**Parameters:**
- `name`: Name of the group

**Returns:**
- The artifact group if it exists, None otherwise

##### get_artifacts_by_tag

```python
def get_artifacts_by_tag(self, tag: str) -> List[Dict[str, Any]]
```

Gets all artifacts with a specific tag.

**Parameters:**
- `tag`: Tag to filter by

**Returns:**
- List of artifacts with the specified tag

##### get_artifacts_by_category

```python
def get_artifacts_by_category(self, category: str) -> List[Dict[str, Any]]
```

Gets all artifacts in a specific category.

**Parameters:**
- `category`: Category to filter by

**Returns:**
- List of artifacts in the specified category

##### get_artifacts_by_type

```python
def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]
```

Gets all artifacts of a specific type.

**Parameters:**
- `artifact_type`: Type to filter by

**Returns:**
- List of artifacts of the specified type

##### add_metric

```python
def add_metric(self, name: str, value: Any)
```

Adds a metric to the report.

**Parameters:**
- `name`: Name of the metric
- `value`: Value of the metric

##### add_nested_metric

```python
def add_nested_metric(self, category: str, name: str, value: Any)
```

Adds a nested metric under a category.

**Parameters:**
- `category`: Category for the metric
- `name`: Name of the metric
- `value`: Value of the metric

##### add_task_summary

```python
def add_task_summary(self, success: bool, execution_time: float = None,
                     metrics: Dict[str, Any] = None, error_info: Dict[str, Any] = None,
                     encryption: Dict[str, Any] = None)
```

Adds task summary to the report.

**Parameters:**
- `success`: Whether the task executed successfully
- `execution_time`: Task execution time in seconds
- `metrics`: Additional metrics to include in the report
- `error_info`: Error information if the task failed
- `encryption`: Encryption information

##### generate_report

```python
def generate_report(self) -> Dict[str, Any]
```

Generates the task report.

**Returns:**
- Report as a dictionary

##### save_report

```python
def save_report(self) -> Path
```

Generates and saves the report to disk.

**Returns:**
- Path to the saved report

**Raises:**
- `ReportingError`: If generating or saving the report fails

##### Context Manager Methods

The TaskReporter class supports being used as a context manager:

```python
def __enter__(self)
```

Context manager entry point.

```python
def __exit__(self, exc_type, exc_val, exc_tb)
```

Context manager exit point that finalizes the report and saves it.

## Report Structure

The complete task report generated by TaskReporter includes:

```json
{
  "task_id": "t_1A_profiling",
  "task_description": "Profile customer data for privacy analysis",
  "task_type": "profiling",
  "script_path": "/path/to/executed/script.py",
  "system_info": {
    "os": "Linux-5.4.0-42-generic-x86_64-with-glibc2.29",
    "python_version": "3.8.10",
    "user": "pamola_user",
    "machine": "analysis-server",
    "cpu_count": 8,
    "pamola_version": "1.0.0"
  },
  "start_time": "2025-05-08 10:15:30",
  "end_time": "2025-05-08 10:20:45",
  "execution_time_seconds": 315.5,
  "status": "completed",
  "operations": [
    {
      "operation": "Load Data",
      "timestamp": "2025-05-08T10:15:35",
      "status": "success",
      "details": {
        "rows": 1000000,
        "columns": 25,
        "execution_time": 12.3
      }
    },
    {
      "operation": "Field Profiling",
      "timestamp": "2025-05-08T10:17:20",
      "status": "success",
      "details": {
        "fields_profiled": 25,
        "execution_time": 145.7
      }
    }
  ],
  "artifacts": [
    {
      "type": "json",
      "path": "/path/to/output/field_statistics.json",
      "filename": "field_statistics.json",
      "description": "Detailed statistics for each field",
      "category": "metrics",
      "tags": ["statistics", "profiling"],
      "size_bytes": 256789,
      "timestamp": "2025-05-08T10:17:45",
      "metadata": {
        "encrypted": false,
        "encryption_mode": "none"
      }
    }
  ],
  "artifact_groups": {
    "field_statistics": {
      "name": "field_statistics",
      "description": "Statistical analysis of data fields",
      "artifacts": [...],
      "created_at": "2025-05-08T10:17:00",
      "count": 15
    }
  },
  "metrics": {
    "field_count": 25,
    "row_count": 1000000,
    "uniqueness": {
      "average": 0.35,
      "max": 0.98,
      "min": 0.01
    }
  },
  "errors": [],
  "warnings": [
    {
      "operation": "Risk Analysis",
      "timestamp": "2025-05-08T10:19:55",
      "message": "Some fields have high uniqueness",
      "details": {
        "fields": ["email", "phone"]
      }
    }
  ],
  "memory_usage_mb": 1256.45
}
```

## Integration with Progress Manager

The TaskReporter class can be integrated with the `TaskProgressManager` from the `progress_manager.py` module to provide coordinated progress tracking and reporting:

```python
# Create a progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1A_profiling",
    task_type="profiling",
    logger=logger,
    total_operations=3
)

# Create a reporter with progress manager
reporter = TaskReporter(
    task_id="t_1A_profiling",
    task_type="profiling",
    description="Profile customer data for privacy analysis",
    report_path=Path("DATA/reports/t_1A_profiling_report.json"),
    progress_manager=progress_manager
)

# Operations will now use progress tracking
with progress_manager.create_operation_context(
    name="load_data",
    total=100,
    description="Loading data"
) as progress:
    # Do work and update progress
    progress.update(50, {"status": "reading"})
    # ...
    progress.update(50, {"status": "complete"})
    
    # Add to report (uses progress manager for coordinated output)
    reporter.add_operation(
        name="Load Data",
        status="success",
        details={"rows": 1000000, "columns": 25}
    )
```

## Usage Examples

### Basic Reporting

```python
from pamola_core.utils.tasks.task_reporting import TaskReporter
from pathlib import Path

# Create reporter
reporter = TaskReporter(
    task_id="t_1A_profiling",
    task_type="profiling",
    description="Profile customer data for privacy analysis",
    report_path=Path("DATA/reports/t_1A_profiling_report.json")
)

# Record operation start
reporter.add_operation(
    name="Load Data",
    status="running",
    details={"source": "customer_data.csv"}
)

# Record operation completion
reporter.add_operation(
    name="Load Data Completed",
    status="success",
    details={
        "rows": 1000000,
        "columns": 25,
        "execution_time": 12.3
    }
)

# Add artifacts
reporter.add_artifact(
    artifact_type="json",
    path="DATA/processed/t_1A_profiling/output/field_statistics.json",
    description="Detailed statistics for each field",
    category="metrics",
    tags=["statistics", "profiling"]
)

# Add metrics
reporter.add_metric("field_count", 25)
reporter.add_metric("row_count", 1000000)
reporter.add_nested_metric("uniqueness", "average", 0.35)

# Add task summary
reporter.add_task_summary(
    success=True,
    execution_time=315.5,
    metrics={"quality_score": 0.95},
    encryption={"enabled": True, "mode": "simple"}
)

# Save report
report_path = reporter.save_report()
print(f"Report saved to: {report_path}")
```

### Using Artifact Groups

```python
from pamola_core.utils.tasks.task_reporting import TaskReporter
from pathlib import Path

# Create reporter
reporter = TaskReporter(
    task_id="t_1P_anonymization",
    task_type="anonymization",
    description="Anonymize customer data using k-anonymity",
    report_path=Path("DATA/reports/t_1P_anonymization_report.json")
)

# Create artifact groups
statistics_group = reporter.add_artifact_group(
    name="statistics",
    description="Statistical analysis before and after anonymization"
)

outputs_group = reporter.add_artifact_group(
    name="outputs",
    description="Anonymized datasets"
)

# Add artifacts to groups
reporter.add_artifact(
    artifact_type="json",
    path="DATA/processed/t_1P_anonymization/output/before_stats.json",
    description="Statistics before anonymization",
    group_name="statistics"
)

reporter.add_artifact(
    artifact_type="csv",
    path="DATA/processed/t_1P_anonymization/output/anonymized_data.csv",
    description="Anonymized dataset",
    group_name="outputs",
    metadata={"encrypted": True, "encryption_mode": "simple"}
)

# Save report
report_path = reporter.save_report()
```

### Using as a Context Manager

```python
from pamola_core.utils.tasks.task_reporting import TaskReporter
from pathlib import Path

# Use reporter as context manager for automatic finalization
with TaskReporter(
    task_id="t_1S_synthesis",
    task_type="synthesis",
    description="Generate synthetic data",
    report_path=Path("DATA/reports/t_1S_synthesis_report.json")
) as reporter:
    
    # Record operations
    reporter.add_operation("Training Data Analysis", status="success")
    reporter.add_operation("Model Training", status="success")
    reporter.add_operation("Synthetic Data Generation", status="success")
    
    # Add artifacts
    reporter.add_artifact(
        artifact_type="csv",
        path="DATA/processed/t_1S_synthesis/output/synthetic_data.csv",
        description="Generated synthetic dataset",
        metadata={"encrypted": True}
    )
    
    # Add metrics
    reporter.add_nested_metric("quality", "fidelity_score", 0.89)
    reporter.add_nested_metric("quality", "privacy_score", 0.95)
    
    # Report is automatically saved when context exits
```

### Progress Manager Integration

```python
from pamola_core.utils.tasks.task_reporting import TaskReporter
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager
from pathlib import Path

# Set up logger
logger = logging.getLogger("task.t_1A")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1A_profiling",
    task_type="profiling",
    logger=logger,
    total_operations=3
)

# Create reporter with progress manager
reporter = TaskReporter(
    task_id="t_1A_profiling",
    task_type="profiling",
    description="Profile customer data for privacy analysis",
    report_path=Path("DATA/reports/t_1A_profiling_report.json"),
    progress_manager=progress_manager
)

# Use context managers for operations with progress tracking
with progress_manager.create_operation_context(
    name="load_data",
    total=100,
    description="Loading data"
) as progress:
    # Simulate work with progress updates
    for i in range(10):
        time.sleep(0.1)
        progress.update(10, {"stage": f"chunk {i+1}/10"})
    
    # Record in report
    reporter.add_operation(
        name="Load Data",
        status="success",
        details={
            "rows": 1000000,
            "columns": 25,
            "execution_time": 1.0
        }
    )

# Save report
report_path = reporter.save_report()
```

## Best Practices

1. **Structure Your Operations**: Record both the start and completion of operations to provide a clear timeline.

2. **Group Related Artifacts**: Use artifact groups to organize related outputs, making reports more navigable.

3. **Include Detailed Metadata**: Add thorough descriptions and metadata to artifacts for better traceability.

4. **Use Consistent Tagging**: Establish a consistent tagging system for artifacts to enable effective filtering.

5. **Track Memory Usage**: Monitor memory usage during execution to identify potential performance issues.

6. **Categorize Metrics**: Use nested metrics to organize related measurements under logical categories.

7. **Document Encryption Status**: Always record whether artifacts are encrypted and what encryption mode was used.

8. **Use Context Managers**: Use the reporter as a context manager to ensure reports are saved even if exceptions occur.

9. **Include Execution Context**: Capture detailed system information to ensure reproducibility.

10. **Report Warnings and Errors**: Record both warnings and errors for complete execution transparency.

11. **Integrate with Progress Manager**: Use the progress manager integration for coordinated progress tracking and reporting.

12. **Validate Paths**: Ensure all paths recorded in the report are valid and accessible by using the built-in path validation.