# PAMOLA Task Templates

This package provides templates and utilities for creating standardized PAMOLA tasks. It's designed to help you quickly get started with creating new tasks that follow the best practices and architectural patterns of the PAMOLA framework.

## Overview

The templates package includes:

1. **Simple Task Template** - A standard task template for basic data processing
2. **Complex Processing Task Template** - A template for tasks with multiple stages and complex data flows
3. **Secure Task Template** - A task template with encryption enabled for sensitive data
4. **Parallel Task Template** - A template for tasks that leverage parallel processing for improved performance
5. **Task with Checkpoint Template** - A template for long-running tasks with checkpoint support for resilience

## Using the Templates

### Method 1: Command Line Utility

You can use the included `task_template_generator.py` utility to create a new task:

```bash
python -m pamola_core.utils.tasks.templates.task_template_generator \
  --template [simple|complex|secure|parallel|checkpoint] \
  --task_id t_1X_my_task \
  --task_type profiling \
  --description "My custom profiling task" \
  --create_directories
```

### Method 2: Programmatic Creation

You can also create tasks programmatically:

```python
from pamola_core.utils.tasks.templates import create_task

task_file = create_task(
    template="simple",  # Choose from: simple, complex, secure, parallel, checkpoint
    task_id="t_1X_my_task",
    task_type="profiling",
    description="My custom profiling task",
    create_directories=True
)

print(f"Created task file: {task_file}")
```

### Method 3: Manual Copy

You can manually copy and modify the template files:

1. Copy one of the template files that best matches your requirements
2. Rename it to match your task ID (e.g., `t_1X_my_task.py`)
3. Update the task_id, task_type, and description in the file
4. Update the operations in the `configure_operations` method

## Template Options

### Simple Task Template (`simple_task_template.py`)

The simple task template provides a basic structure for creating a standard PAMOLA task. It includes:

- Basic task initialization
- Operation configuration
- Result processing
- Command line argument parsing

### Complex Processing Task Template (`complex_processing_task_template.py`)

The complex processing template is designed for tasks with multiple stages and complex data flows:

- Multi-stage processing pipeline
- Dependent operations
- Advanced configuration
- Multiple input/output datasets
- Comprehensive result aggregation

### Secure Task Template (`secure_task_template.py`)

The secure task template extends the simple template with additional security features:

- Encryption configuration
- Secure directory handling
- Encryption key management
- Sensitive data filtering
- Enhanced security checks

### Parallel Task Template (`parallel_task_template.py`)

The parallel task template demonstrates how to implement parallel processing for improved performance:

- Parallel execution of operations
- Data partitioning strategies
- Worker pool configuration
- Optimal resource utilization
- Result aggregation from parallel operations

### Task with Checkpoint Template (`task_with_checkpoint.py`)

The checkpoint task template provides resilience for long-running tasks:

- Checkpoint creation during execution
- State persistence between restarts
- Resumable operations
- Error recovery mechanisms
- Progress tracking and state management

## Directory Structure

When you create a task with `create_directories=True`, the following directory structure will be created:

```
PROJECT_ROOT/
├── configs/
│   ├── prj_config.json        # Project configuration
│   └── t_1X_my_task.json      # Task-specific configuration
├── logs/                      # Log files
└── DATA/
    ├── raw/                   # Raw input data
    ├── processed/
    │   └── t_1X_my_task/      # Task-specific directory
    │       ├── output/        # Task outputs
    │       ├── dictionaries/  # Task dictionaries
    │       ├── temp/          # Temporary files
    │       ├── logs/          # Task-specific logs
    │       └── checkpoints/   # Checkpoint files (for resumable tasks)
    └── reports/               # Task reports
```

## Customizing Templates

After creating a task from a template, you should:

1. Update the input/auxiliary datasets with your actual data files
2. Configure the operations specific to your task
3. Customize the `process_results` method to handle your specific results
4. Add any additional methods or functionality needed for your task

## Template Selection Guidelines

Choose the appropriate template based on your task requirements:

- **Simple Task**: When you need a basic, linear workflow with minimal complexity
- **Complex Processing**: When your task involves multiple dependent stages and complex data flows
- **Secure Task**: When working with sensitive or confidential data that requires encryption
- **Parallel Task**: When processing large datasets or performance-critical operations
- **Task with Checkpoint**: When working with long-running processes that benefit from resumability

## Best Practices

- Use descriptive task IDs that follow the convention `t_#X_description` where:
  - `#` is the phase number
  - `X` is a letter indicating the step within the phase
  - `description` briefly describes the task's purpose
- Provide comprehensive descriptions for your tasks
- Keep operations organized and well-documented
- Include error handling for operations that may fail
- Process and aggregate results for better reporting
- Use appropriate logging levels for different types of messages
- Leverage task configurations for parameterization rather than hardcoding values