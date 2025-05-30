# TaskRegistry Module Documentation

## Overview

The `task_registry.py` module provides functionality for registering task classes, discovering available tasks, and checking task dependencies in the PAMOLA Core framework. It serves as an in-memory registry for task types and enables type-based instantiation, while working in conjunction with the `execution_log.py` module to track task execution history and dependencies.

## Key Features

- **Task Class Registration**: In-memory registration of task classes by ID
- **Task Class Discovery**: Auto-discovery of task classes from specified packages
- **Task Metadata Extraction**: Extraction of metadata from task classes
- **Task Dependency Validation**: Verification of task dependencies
- **Task Instantiation**: Creation of task instances by ID
- **Task Class Validation**: Validation of task class implementation

## Dependencies

- `importlib`: Module import utilities
- `inspect`: Introspection capabilities
- `logging`: Logging functionality
- `pkgutil`: Package traversal utilities
- `typing`: Type annotations
- `pamola_core.utils.tasks.execution_log`: Task execution history tracking

## Exception Class

### TaskRegistryError

Exception raised for task registry errors.

## Global Variables

- `_task_classes`: Dictionary mapping task IDs to task classes (private)

## Functions

### register_task_class

```python
def register_task_class(task_id: str, task_class: Type) -> bool
```

Registers a task class by ID.

**Parameters:**
- `task_id`: ID of the task
- `task_class`: Task class to register

**Returns:**
- `True` if registration was successful, `False` otherwise

**Raises:**
- `TaskRegistryError`: If registration fails

### get_task_class

```python
def get_task_class(task_id: str) -> Optional[Type]
```

Gets a task class by ID.

**Parameters:**
- `task_id`: ID of the task

**Returns:**
- Task class or None if not found

**Raises:**
- `TaskRegistryError`: If lookup fails

### list_registered_tasks

```python
def list_registered_tasks() -> Dict[str, Dict[str, Any]]
```

Lists all registered task types with their metadata.

**Returns:**
- Dictionary mapping task IDs to task metadata

**Raises:**
- `TaskRegistryError`: If listing fails

### create_task_instance

```python
def create_task_instance(task_id: str, **kwargs) -> Optional[Any]
```

Creates a task instance by ID.

**Parameters:**
- `task_id`: ID of the task
- `**kwargs`: Arguments to pass to the task constructor

**Returns:**
- Task instance or None if task class not found

**Raises:**
- `TaskRegistryError`: If instantiation fails

### discover_task_classes

```python
def discover_task_classes(package_paths: Optional[List[str]] = None,
                          recursive: bool = True) -> Dict[str, Type]
```

Discovers task classes in specified packages.

**Parameters:**
- `package_paths`: List of package paths to scan (e.g., ["mypackage.tasks"])
- `recursive`: Whether to scan subpackages recursively

**Returns:**
- Dictionary mapping task IDs to task classes

**Raises:**
- `TaskRegistryError`: If discovery fails

### register_discovered_tasks

```python
def register_discovered_tasks(package_paths: Optional[List[str]] = None,
                              recursive: bool = True) -> int
```

Discovers and registers task classes in specified packages.

**Parameters:**
- `package_paths`: List of package paths to scan
- `recursive`: Whether to scan subpackages recursively

**Returns:**
- Number of tasks registered

**Raises:**
- `TaskRegistryError`: If registration fails

### get_task_metadata

```python
def get_task_metadata(task_class: Type) -> Dict[str, Any]
```

Extracts metadata from a task class.

**Parameters:**
- `task_class`: Task class to extract metadata from

**Returns:**
- Dictionary with task metadata

**Raises:**
- `TaskRegistryError`: If metadata extraction fails

### check_task_dependencies

```python
def check_task_dependencies(task_id: str, task_type: str, dependencies: List[str]) -> bool
```

Checks if dependencies for a task are satisfied.

**Parameters:**
- `task_id`: ID of the task
- `task_type`: Type of the task
- `dependencies`: List of task IDs that this task depends on

**Returns:**
- `True` if all dependencies are satisfied, `False` otherwise

**Raises:**
- `TaskRegistryError`: If dependency check fails

## Private Functions

### _is_task_class

```python
def _is_task_class(cls: Type) -> bool
```

Checks if a class is a task class.

**Parameters:**
- `cls`: Class to check

**Returns:**
- `True` if the class is a task class, `False` otherwise

### _get_task_id

```python
def _get_task_id(cls: Type) -> Optional[str]
```

Gets the task ID from a task class.

**Parameters:**
- `cls`: Task class

**Returns:**
- Task ID or None if not found

### _validate_task_class

```python
def _validate_task_class(cls: Type) -> bool
```

Validates that a class meets the requirements for a task class.

**Parameters:**
- `cls`: Class to validate

**Returns:**
- `True` if the class is valid, `False` otherwise

## Task Class Requirements

For a class to be considered a valid task class, it must:

1. Have a `task_id` attribute or method
2. Inherit from a class named 'BaseTask'
3. Have required constructor parameters: `task_id`, `task_type`, `description`
4. Implement required methods: `configure_operations`, `run`

If the class doesn't directly inherit from `BaseTask` but has a compatible interface, it must also implement:
- `initialize`
- `execute`
- `finalize`

## Task Metadata

The metadata extracted from task classes includes:

- `task_id`: ID of the task
- `task_type`: Type of the task
- `description`: Description of the task
- `version`: Version of the task implementation
- `class_name`: Name of the task class
- `module`: Module where the task class is defined
- `dependencies`: List of task IDs that this task depends on
- `author`: Author of the task implementation

Additional metadata can be extracted from class docstrings using tags (e.g., `@author`, `@version`).

## Task Dependency Tracking

Task dependencies are checked using the `check_task_dependencies` function, which:

1. Looks up each dependency task ID in the execution log
2. Verifies that the dependency task has been executed successfully
3. Returns `True` only if all dependencies are satisfied

## Usage Examples

### Registering a Task Class

```python
from pamola_core.utils.tasks.base_task import BaseTask
from pamola_core.utils.tasks.task_registry import register_task_class

class MyProfilingTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="t_1P_profiling",
            task_type="profiling",
            description="Profiling task for customer data"
        )
    
    def configure_operations(self):
        # Configure operations
        pass

# Register the task class
register_task_class("t_1P_profiling", MyProfilingTask)
```

### Creating a Task Instance by ID

```python
from pamola_core.utils.tasks.task_registry import create_task_instance

# Create a task instance by ID
task = create_task_instance("t_1P_profiling")

# Run the task
if task:
    success = task.run()
    print(f"Task execution {'succeeded' if success else 'failed'}")
else:
    print("Task class not found")
```

### Discovering and Registering Task Classes

```python
from pamola_core.utils.tasks.task_registry import discover_task_classes, register_discovered_tasks

# Discover task classes in a specific package
task_classes = discover_task_classes(["scripts.tasks"], recursive=True)
print(f"Discovered {len(task_classes)} task classes")

# Register all discovered tasks
registered_count = register_discovered_tasks(["scripts.tasks"])
print(f"Registered {registered_count} task classes")
```

### Checking Task Dependencies

```python
from pamola_core.utils.tasks.task_registry import check_task_dependencies

# Define task dependencies
dependencies = ["t_1I_import", "t_1C_clean"]

# Check if dependencies are satisfied
dependencies_satisfied = check_task_dependencies(
    task_id="t_1P_profiling",
    task_type="profiling",
    dependencies=dependencies
)

if dependencies_satisfied:
    print("Dependencies satisfied, task can be executed")
else:
    print("Dependencies not satisfied, task cannot be executed")
```

### Listing Registered Tasks

```python
from pamola_core.utils.tasks.task_registry import list_registered_tasks

# Get all registered tasks with metadata
registered_tasks = list_registered_tasks()

# Print task information
for task_id, metadata in registered_tasks.items():
    print(f"Task ID: {task_id}")
    print(f"  Type: {metadata['task_type']}")
    print(f"  Description: {metadata['description']}")
    print(f"  Version: {metadata['version']}")
    print(f"  Dependencies: {', '.join(metadata['dependencies'])}")
    print()
```

### Extracting Task Metadata

```python
from pamola_core.utils.tasks.task_registry import get_task_metadata
from scripts.tasks.my_task import MyTask

# Extract metadata from a task class
metadata = get_task_metadata(MyTask)

# Print metadata
print(f"Task ID: {metadata['task_id']}")
print(f"Task Type: {metadata['task_type']}")
print(f"Description: {metadata['description']}")
print(f"Version: {metadata['version']}")
print(f"Author: {metadata['author']}")
print(f"Dependencies: {metadata['dependencies']}")
```

## Task Discovery Algorithm

The task discovery algorithm works by:

1. Importing specified packages
2. Walking through all modules in the packages and their subpackages (if recursive)
3. Scanning all classes in each module to find those that are task classes
4. Extracting the task ID from each task class
5. Building a dictionary mapping task IDs to task classes

## Best Practices

1. **Consistent Task IDs**: Use a consistent naming convention for task IDs (e.g., `t_phase_description`)
2. **Clear Dependencies**: Define task dependencies explicitly in the task class
3. **Metadata in Docstrings**: Include metadata like author and version in class docstrings
4. **Proper Task Class Structure**: Ensure all task classes inherit from `BaseTask` and implement required methods
5. **Automatic Discovery**: Use task discovery to find task classes instead of manual registration
6. **Dependency Checking**: Always check task dependencies before execution
7. **Version Tracking**: Include version information in task metadata
8. **Error Handling**: Handle registry errors gracefully
9. **Package Organization**: Organize task classes in logical package structures
10. **Instantiation Parameters**: Use keyword arguments when creating task instances

## Advanced Usage

### Creating a Task Pipeline

```python
from pamola_core.utils.tasks.task_registry import list_registered_tasks, create_task_instance, check_task_dependencies
import networkx as nx
import matplotlib.pyplot as plt

# Create a dependency graph
graph = nx.DiGraph()

# Get all registered tasks
registered_tasks = list_registered_tasks()

# Add nodes and edges to the graph
for task_id, metadata in registered_tasks.items():
    graph.add_node(task_id, **metadata)
    for dep in metadata.get('dependencies', []):
        graph.add_edge(dep, task_id)

# Find execution order (topological sort)
try:
    execution_order = list(nx.topological_sort(graph))
    print(f"Execution order: {execution_order}")
    
    # Execute tasks in order
    for task_id in execution_order:
        # Check dependencies
        task_data = registered_tasks[task_id]
        dependencies = task_data.get('dependencies', [])
        
        if check_task_dependencies(task_id, task_data['task_type'], dependencies):
            # Create and run task
            task = create_task_instance(task_id)
            if task:
                print(f"Running task: {task_id}")
                task.run()
            else:
                print(f"Task not found: {task_id}")
        else:
            print(f"Dependencies not satisfied for: {task_id}")
    
    # Visualize the pipeline
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=1500, arrows=True)
    plt.title("Task Dependency Graph")
    plt.show()
    
except nx.NetworkXUnfeasible:
    print("Circular dependencies detected!")
```

### Creating Custom Task Base Classes

```python
from pamola_core.utils.tasks.base_task import BaseTask
from pamola_core.utils.tasks.task_registry import register_task_class

# Custom base task with additional functionality
class CustomBaseTask(BaseTask):
    def __init__(self, task_id, task_type, description, **kwargs):
        super().__init__(task_id, task_type, description)
        self.additional_config = kwargs.get('additional_config', {})
    
    def pre_operation_hook(self, operation_name):
        """Called before each operation is executed"""
        print(f"Preparing to execute operation: {operation_name}")
    
    def post_operation_hook(self, operation_name, result):
        """Called after each operation is executed"""
        print(f"Completed execution of operation: {operation_name}")
        print(f"Result status: {result.status}")
    
    def execute(self):
        """Override execute to add pre/post hooks"""
        if not self.operations:
            self.logger.error("No operations configured for this task")
            return False
        
        for i, operation in enumerate(self.operations):
            operation_name = operation.name if hasattr(operation, 'name') else f"Operation {i + 1}"
            
            self.pre_operation_hook(operation_name)
            result = super().execute_operation(operation)
            self.post_operation_hook(operation_name, result)
            
            if result.status == "ERROR" and not self.config.continue_on_error:
                return False
        
        return True

# Task implementation using custom base class
class CustomTask(CustomBaseTask):
    def __init__(self):
        super().__init__(
            task_id="custom_task",
            task_type="custom",
            description="Task using custom base class",
            additional_config={"custom_setting": True}
        )
    
    def configure_operations(self):
        # Configure operations
        pass

# Register the custom task
register_task_class("custom_task", CustomTask)
```