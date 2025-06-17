# PAMOLA.CORE Operations Framework Documentation

## Overview

The PAMOLA.CORE Operations Framework provides a standardized architecture for implementing privacy-enhancing operations on data. It establishes a consistent pattern for operations to:

1. Receive parameters from user scripts and tasks
2. Process data through a uniform interface
3. Generate structured metrics, visualizations, and transformed outputs
4. Return standardized results through a common API

This framework enables modular, reproducible, and well-orchestrated data privacy workflows.

## Pamola Core Architecture

The Operations Framework is organized around a central concept: **operations are atomic units of computation** that:

- Focus on specific privacy-enhancing tasks (profiling, anonymization, transformation)
- Follow a predictable execution lifecycle
- Produce standardized artifacts and metrics
- Can be composed into larger workflows

### Package Structure

```
pamola_core/utils/ops/
├── __init__.py                  # Package initialization, OpsError base class
├── op_base.py                   # Base operation classes and execution lifecycle
├── op_config.py                 # Configuration validation and management
├── op_config_errors.py          # Configuration-related error classes
├── op_cache.py                  # Caching system for operation results
├── op_cache_errors.py           # Cache-related error classes
├── op_data_source.py            # Unified data source abstraction
├── op_data_source_helpers.py    # Helper functions for data sources
├── op_data_reader.py            # Reading data from various sources
├── op_data_writer.py            # Writing operation outputs consistently
├── op_data_writer_errors.py     # Data writer error classes
├── op_registry.py               # Operation discovery and registration
├── op_registry_errors.py        # Registry-related error classes
├── op_result.py                 # Operation result and artifact management
├── op_result_errors.py          # Result validation error classes
├── op_test_helpers.py           # Testing utilities for operations
└── templates/                   # Templates for new operations
    ├── operation_skeleton.py    # Base skeleton for new operations
    └── config_example.json      # Example configuration schema
```

### Pamola Core Components and Their Responsibilities

#### 1. Base Operation Classes (`op_base.py`)

The foundation of the Operations Framework, providing abstract base classes with standardized interfaces.

- **BaseOperation**: Abstract base class for all operations
  - Manages execution lifecycle via `run()`
  - Defines abstract `execute()` method for subclasses
  - Handles configuration saving, directory preparation, progress tracking
  - Provides error handling and reporting

- **FieldOperation**: For operations on specific fields/columns
  - Inherits from BaseOperation
  - Adds field-specific parameters (field name, output mode, etc.)
  - Handles field validation and output field name generation

- **DataFrameOperation**: For operations on entire DataFrames
  - Inherits from BaseOperation
  - Adds DataFrame-specific parameters (chunking, parallelization)
  - Support for large-scale data processing

#### 2. Configuration Management (`op_config.py`)

Validates and manages operation parameters through JSON Schema.

- **OperationConfig**: Generic configuration class
  - Validates parameters against JSON schema
  - Provides serialization/deserialization to/from JSON
  - Offers dict-like parameter access

- **OperationConfigRegistry**: Registry for operation-specific configs
  - Maps operation types to configuration classes
  - Provides factory methods for configuration creation

#### 3. Data Access (`op_data_source.py`, `op_data_reader.py`)

Abstracts data access regardless of source (memory, file, database).

- **DataSource**: Unified interface to data
  - Manages in-memory DataFrames and file paths
  - Provides access to data in a consistent way
  - Supports chunked processing for large datasets

- **DataReader**: Handles actual reading from files/sources
  - Supports various formats (CSV, Parquet, JSON)
  - Handles encryption when needed
  - Manages progress reporting for long-running reads

#### 4. Data Output (`op_data_writer.py`)

Standardizes writing operation outputs in a consistent structure.

- **DataWriter**: Unified interface for writing artifacts
  - Ensures consistent directory structure
  - Handles various output formats (CSV, Parquet, JSON, PNG)
  - Supports encryption of sensitive outputs
  - Integrates with progress tracking
  - Returns detailed metadata about written files

#### 5. Result Management (`op_result.py`)

Encapsulates operation results with standardized status and artifacts.

- **OperationResult**: Complete result container
  - Tracks operation status (success, error, etc.)
  - Manages artifacts (files produced by operation)
  - Collects metrics for monitoring and reporting
  - Validates artifact integrity

- **OperationArtifact**: Individual artifact metadata
  - Links to physical files
  - Tracks metadata (type, creation time, size)
  - Supports validation and integrity checking

- **OperationStatus**: Enumeration of possible execution states
  - SUCCESS, WARNING, ERROR, SKIPPED, PARTIAL_SUCCESS, PENDING

#### 6. Operation Registration (`op_registry.py`)

Provides discovery and lookup of operation classes.

- **Registry Functions**: Manage operation registration
  - `register_operation()`: Register operation classes
  - `get_operation_class()`: Look up by name
  - `list_operations()`: Discover available operations
  - `create_operation_instance()`: Factory for operation creation

#### 7. Caching (`op_cache.py`)

Manages caching of operation results for performance optimization.

- **OperationCache**: Cache manager for results
  - Stores and retrieves cached results
  - Generates deterministic cache keys
  - Manages cache size and expiration
  - Provides async API for non-blocking access

#### 8. Testing Support (`op_test_helpers.py`)

Utilities for testing operations without complex setup.

- **MockDataSource**: Test double for DataSource
  - Works with in-memory DataFrames
  - Simulates real DataSource behavior

- **StubDataWriter**: Test double for DataWriter
  - Records calls for verification
  - Writes to temporary directories

- **Testing Functions**: Helper utilities
  - `create_test_operation_env()`: Set up test environment
  - `assert_artifact_exists()`: Validate artifact creation
  - `assert_metrics_content()`: Verify metrics content

## Operation Lifecycle

An operation in PAMOLA.CORE follows a standard lifecycle:

1. **Initialization**
   - Operation instance created with parameters
   - Configuration validated against schema

2. **Execution Start**
   - `run()` called with data source, task directory, and reporter
   - Configuration saved to task directory
   - Progress tracking initialized
   - Writer and result objects created

3. **Data Access**
   - Data loaded from source
   - Input validation performed

4. **Processing**
   - Operation-specific logic executed
   - Progress updated during processing

5. **Metrics Collection**
   - Operation-specific metrics gathered
   - Metrics saved to task directory

6. **Result Generation**
   - Output data written to task directory
   - Artifacts created and registered

7. **Completion**
   - Final status set (SUCCESS, ERROR, etc.)
   - Result returned to caller

## Directory Structure

Operations follow a standardized directory structure for artifacts:

```
{task_dir}/
├── config.json                # Operation configuration
├── operation_metrics.json     # Operation metrics
├── output/                    # Transformed data outputs
│   └── processed_data.csv
├── dictionaries/              # Extracted dictionaries and mappings
│   └── field_mapping.json
└── logs/                      # Operation logs
    └── operation.log
```

## Error Handling

The framework provides a complete error hierarchy:

- **OpsError**: Base error class for all operations
  - **ConfigError**: Configuration validation errors
    - **ConfigSaveError**: Error saving configuration
  - **DataWriteError**: Error writing operation outputs
  - **RegistryError**: Error in operation registration
  - **ValidationError**: Error validating artifacts
  - **CacheError**: Error in caching operations

## Code Style and Conventions

The PAMOLA.CORE Operations Framework follows strict conventions:

- **Docstring Style**: Google style
- **Type Hints**: Required for all public interfaces
- **Exception Handling**: All operations must handle exceptions
- **Naming**: PascalCase for classes, snake_case for methods
- **Imports**: Organized by standard library, third-party, framework, project
- **Logging**: Operation-specific loggers for contextualized logs

## Best Practices

### 1. Operation Implementation

- Inherit from the appropriate base class (BaseOperation, FieldOperation, DataFrameOperation)
- Override `execute()` method, not `run()`
- Always call `self.save_config(task_dir)` at the start of execution
- Use `DataWriter` for all file operations
- Register artifacts with `OperationResult`
- Handle all exceptions and return appropriate status

### 2. Data Handling

- Always use `DataSource` for data access
- Support chunked processing for large datasets
- Make no assumptions about data types without validation
- Preserve original data when possible
- Report progress for long-running operations

### 3. Result and Artifact Management

- Use proper artifact types and categories
- Include detailed metadata with results
- Register all artifacts with the result
- Validate artifacts before returning

### 4. Testing

- Use `op_test_helpers` to simplify test setup
- Test both success and error paths
- Verify artifact creation and metrics
- Use `MockDataSource` and `StubDataWriter` for isolation

## Integration with Other Components

Operations integrate with other pamola core utilities:

- **Logging**: Configure operation-specific loggers
- **Progress Tracking**: Update progress during execution
- **I/O**: Use standardized I/O through DataSource and DataWriter
- **Visualization**: Generate visualizations through standard utilities
- **Reporters**: Report progress and artifacts to reporters

## Advanced Features

### 1. Caching for Performance

```python
def execute(self, data_source, task_dir, reporter, **kwargs):
    # Generate cache key based on parameters
    cache_key = operation_cache.generate_cache_key(
        operation_name=self.name,
        parameters={
            "field_name": self.field_name,
            "bins": self.bins
        }
    )
    
    # Try to get cached result
    cached_result = operation_cache.get_cache(cache_key)
    if cached_result:
        # Create result from cached data
        result = OperationResult(status=OperationStatus.SUCCESS)
        # Populate from cache...
        return result
    
    # Normal execution if not cached
    # ...
    
    # Cache the result
    operation_cache.save_cache(result_data, cache_key)
    return result
```

### 2. Chunked Processing for Large Datasets

```python
def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
    # Process in chunks to handle large datasets
    results = []
    for i, chunk in enumerate(data_source.get_dataframe_chunks("main", chunk_size=10000)):
        # Process chunk
        chunk_result = self._process_chunk(chunk)
        results.append(chunk_result)
        
        # Update progress
        if progress_tracker:
            progress_tracker.update(i+1, {"chunk": i+1})
    
    # Combine chunk results
    combined_result = self._combine_results(results)
    return combined_result
```

### 3. Artifact Group Management

```python
def execute(self, data_source, task_dir, reporter, **kwargs):
    # Create result
    result = OperationResult(status=OperationStatus.SUCCESS)
    
    # Create artifact groups
    viz_group = result.add_artifact_group(
        name="visualizations",
        description="Data visualizations"
    )
    
    # Add artifacts to groups
    result.add_artifact(
        artifact_type="png",
        path=histogram_path,
        description="Distribution histogram",
        group="visualizations"
    )
    
    return result
```

## Conclusion

The PAMOLA.CORE Operations Framework provides a robust foundation for building privacy-enhancing operations with consistent interfaces, error handling, and result management. By leveraging this framework, developers can create modular, testable, and easily composable operations that integrate seamlessly into larger workflows.