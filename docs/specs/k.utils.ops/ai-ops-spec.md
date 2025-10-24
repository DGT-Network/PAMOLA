# PAMOLA.CORE Operations Framework AI-Friendly Specification

## 1. Framework Overview

```yaml
framework:
  name: "PAMOLA.CORE Operations Framework"
  description: "Standardized framework for privacy-enhancing operations in PAMOLA"
  version: "1.0.0"
  pythonVersion: ">=3.8"
  requiresFutureAnnotations: true
```

The PAMOLA.CORE Operations Framework provides a standardized architecture for implementing privacy-enhancing operations on data. Operations follow a consistent pattern, receiving parameters from user scripts/tasks, processing data, generating metrics and visualizations, and returning results through a uniform interface.

## 2. Package Layout & Structure

```yaml
packageLayout:
  root: "pamola_core/utils/ops"
  modules:
    - name: "op_base.py"
      description: "Base operation classes and execution lifecycle"
    - name: "op_config.py"
      description: "Configuration schema and validation"
    - name: "op_cache.py"
      description: "Caching system for operation results"
    - name: "op_data_reader.py"
      description: "Reading data from various sources"
    - name: "op_data_source.py"
      description: "Unified data source abstraction"
    - name: "op_data_source_helpers.py"
      description: "Helper functions for data sources"
    - name: "op_data_writer.py"
      description: "Writing operation outputs consistently"
    - name: "op_registry.py"
      description: "Operation discovery and registration"
    - name: "op_result.py"
      description: "Operation result management"
    - name: "op_test_helpers.py"
      description: "Testing utilities for operations"
  templates: "pamola_core/utils/ops/templates/"
  tests: "tests/utils/ops/"
  dependencies:
    - "pamola_core/utils/logging.py"
    - "pamola_core/utils/progress.py"
    - "pamola_core/utils/io.py"
    - "pamola_core/utils/visualization.py"
```

## 3. Base Classes & Class Variants

### 3.1 BaseOperation

```yaml
class:
  name: "BaseOperation"
  description: "Abstract base class for all operations"
  moduleLocation: "pamola_core.utils.ops.op_base"
  importStyle: "from pamola_core.utils.ops.op_base import BaseOperation"
  inheritanceRelationship: "Abstract base class"
  init:
    signature: "__init__(self, name: str = 'unnamed', description: str = '', scope: Optional[OperationScope] = None, config: Optional[OperationConfig] = None, use_encryption: bool = False, encryption_key: Optional[Union[str, Path]] = None, use_vectorization: bool = False) -> None"
    required:
      - "name"
    defaults:
      name: "'unnamed'"
      description: "''"
      scope: "None"
      config: "None"
      use_encryption: "False"
      encryption_key: "None"
      use_vectorization: "False"
    superCall: "None - this is the base class"
  abstractMethods:
    - name: "execute"
      signature: "execute(self, data_source: DataSource, task_dir: Path, reporter: Any, progress_tracker: Optional[ProgressTracker] = None, **kwargs) -> OperationResult"
      description: "Implementation-specific logic for the operation"
  concreteClassMethods:
    - name: "run"
      signature: "run(self, data_source: DataSource, task_dir: Path, reporter: Any, track_progress: bool = True, parallel_processes: int = 1, **kwargs) -> OperationResult"
      description: "Public method that wraps execute with lifecycle management"
      requirementId: "REQ-OPS-001"
    - name: "save_config"
      signature: "save_config(self, task_dir: Path) -> None"
      description: "Serialize operation configuration to JSON"
      requirementId: "REQ-OPS-004"
    - name: "_prepare_directories"
      signature: "_prepare_directories(self, task_dir: Path) -> Dict[str, Path]"
      description: "Create standard directory structure for operation artifacts"
  instanceVariables:
    - name: "name"
      type: "str"
      description: "Name of the operation"
    - name: "description" 
      type: "str"
      description: "Human-readable description"
    - name: "version"
      type: "str"
      description: "Semantic version (MAJOR.MINOR.PATCH) of the operation"
      example: "'1.0.0'"
    - name: "config"
      type: "Optional[OperationConfig]"
      description: "Configuration parameters for the operation"
    - name: "logger"
      type: "logging.Logger"
      description: "Logger for the operation"
      example: "self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')"
```

### 3.2 FieldOperation

```yaml
class:
  name: "FieldOperation"
  description: "Base class for operations on specific fields"
  moduleLocation: "pamola_core.utils.ops.op_base"
  importStyle: "from pamola_core.utils.ops.op_base import FieldOperation"
  inheritanceRelationship: "Extends BaseOperation"
  init:
    signature: "__init__(self, name: str = 'field_operation', description: str = '', field_name: str = None, field_prefix: str = '', field_suffix: str = '_processed', mode: str = 'ENRICH', output_field_name: Optional[str] = None, config: Optional[OperationConfig] = None, use_encryption: bool = False, encryption_key: Optional[Union[str, Path]] = None) -> None"
    required:
      - "name"
      - "field_name"
    superCall: "super().__init__(name=name, description=description, config=config, use_encryption=use_encryption, encryption_key=encryption_key)"
  instanceVariables:
    - name: "field_name"
      type: "str"
      description: "Name of the field to process"
    - name: "field_prefix"
      type: "str"
      description: "Prefix for output field name"
      default: "''"
    - name: "field_suffix"
      type: "str"
      description: "Suffix for output field name"
      default: "'_processed'"
    - name: "mode"
      type: "str"
      description: "ENRICH (create new field) or REPLACE (modify existing)"
      default: "'ENRICH'"
      allowedValues: ["'ENRICH'", "'REPLACE'"]
    - name: "output_field_name"
      type: "Optional[str]"
      description: "Explicit output field name (overrides prefix/suffix)"
      default: "None"
```

### 3.3 DataFrameOperation

```yaml
class:
  name: "DataFrameOperation"
  description: "Base class for operations on entire DataFrames"
  moduleLocation: "pamola_core.utils.ops.op_base"
  importStyle: "from pamola_core.utils.ops.op_base import DataFrameOperation"
  inheritanceRelationship: "Extends BaseOperation"
  init:
    signature: "__init__(self, name: str = 'dataframe_operation', description: str = '', config: Optional[OperationConfig] = None, use_encryption: bool = False, encryption_key: Optional[Union[str, Path]] = None, use_parallel: bool = False, chunked_processing: bool = False, chunk_size: int = 10000) -> None"
    required:
      - "name"
    superCall: "super().__init__(name=name, description=description, config=config, use_encryption=use_encryption, encryption_key=encryption_key)"
  instanceVariables:
    - name: "use_parallel"
      type: "bool"
      description: "Whether to use parallel processing"
      default: "False"
    - name: "chunked_processing"
      type: "bool"
      description: "Whether to process data in chunks"
      default: "False"
    - name: "chunk_size"
      type: "int"
      description: "Size of chunks for processing large datasets"
      default: "10000"
```

## 4. Utility Classes

### 4.1 OperationConfig

```yaml
class:
  name: "OperationConfig"
  description: "Configuration for operations with schema validation"
  moduleLocation: "pamola_core.utils.ops.op_config"
  importStyle: "from pamola_core.utils.ops.op_config import OperationConfig"
  classVariables:
    - name: "schema"
      type: "Dict[str, Any]"
      description: "JSON Schema for validating configuration"
      example: |
        schema = {
            "type": "object",
            "properties": {
                "field_name": {"type": "string"},
                "threshold": {"type": "number", "minimum": 0, "maximum": 1.0}
            },
            "required": ["field_name"]
        }
  init:
    signature: "__init__(self, **kwargs) -> None"
    description: "Initialize with validated configuration parameters"
    requirementId: "REQ-OPS-002"
  methods:
    - name: "get"
      signature: "get(self, key: str, default: Any = None) -> Any"
      description: "Get configuration parameter with default fallback"
    - name: "to_dict"
      signature: "to_dict(self) -> Dict[str, Any]"
      description: "Convert configuration to dictionary"
    - name: "save"
      signature: "save(self, path: Union[str, Path]) -> None"
      description: "Save configuration to a JSON file"
    - name: "load"
      signature: "load(cls: Type[T], path: Union[str, Path]) -> T"
      description: "Load configuration from a JSON file"
      classMethod: true
```

### 4.2 DataWriter

```yaml
class:
  name: "DataWriter"
  description: "Utility for writing operation outputs consistently"
  moduleLocation: "pamola_core.utils.ops.op_data_writer"
  importStyle: "from pamola_core.utils.ops.op_data_writer import DataWriter"
  init:
    signature: "__init__(self, task_dir: Path, logger: Optional[logging.Logger] = None, progress_tracker: Optional[ProgressTracker] = None) -> None"
    required:
      - "task_dir"
  methods:
    - name: "write_dataframe"
      signature: "write_dataframe(self, df: Union[pd.DataFrame, dd.DataFrame], name: str, format: str = 'csv', subdir: str = 'output', timestamp_in_name: bool = False, encryption_key: Optional[Union[str, Path]] = None, **kwargs) -> WriterResult"
      description: "Write DataFrame to a file"
      requirementId: "REQ-OPS-001"
    - name: "write_json"
      signature: "write_json(self, data: Any, name: str, subdir: Optional[str] = None, timestamp_in_name: bool = True, encryption_key: Optional[Union[str, Path]] = None, pretty: bool = True, **kwargs) -> WriterResult"
      description: "Write JSON data to a file"
    - name: "write_metrics"
      signature: "write_metrics(self, metrics: Dict[str, Any], name: str, timestamp_in_name: bool = True, encryption_key: Optional[Union[str, Path]] = None, **kwargs) -> WriterResult"
      description: "Write metrics to a JSON file"
    - name: "write_dictionary"
      signature: "write_dictionary(self, data: Dict[str, Any], name: str, format: str = 'json', timestamp_in_name: bool = True, encryption_key: Optional[Union[str, Path]] = None, **kwargs) -> WriterResult"
      description: "Write a dictionary to dictionaries/ subdirectory"
    - name: "write_visualization"
      signature: "write_visualization(self, figure: Any, name: str, format: str = 'png', subdir: Optional[str] = None, timestamp_in_name: bool = True, **kwargs) -> WriterResult"
      description: "Save visualization to a file"
```

### 4.3 OperationResult

```yaml
class:
  name: "OperationResult"
  description: "Result of an operation, containing status, artifacts, and metrics"
  moduleLocation: "pamola_core.utils.ops.op_result"
  importStyle: "from pamola_core.utils.ops.op_result import OperationResult, OperationStatus"
  init:
    signature: "__init__(self, status: OperationStatus = OperationStatus.PENDING, artifacts: Optional[List[OperationArtifact]] = None, metrics: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None, execution_time: Optional[float] = None) -> None"
    required:
      - "status"
  methods:
    - name: "add_artifact"
      signature: "add_artifact(self, artifact_type: str, path: Union[str, Path], description: str = '', category: str = 'output', tags: Optional[List[str]] = None, group: Optional[str] = None) -> OperationArtifact"
      description: "Add an artifact to the result"
      requirementId: "REQ-OPS-005"
    - name: "register_artifact_via_writer"
      signature: "register_artifact_via_writer(self, writer: DataWriter, obj: Any, subdir: str, name: str, artifact_type: Optional[str] = None, description: str = '', category: str = 'output', tags: Optional[List[str]] = None, group: Optional[str] = None) -> OperationArtifact"
      description: "Register an artifact using DataWriter"
      requirementId: "REQ-OPS-005"
    - name: "add_metric"
      signature: "add_metric(self, name: str, value: Any) -> None"
      description: "Add a metric to the result"
    - name: "add_nested_metric"
      signature: "add_nested_metric(self, category: str, name: str, value: Any) -> None"
      description: "Add a metric under a category"
    - name: "validate_artifacts"
      signature: "validate_artifacts(self) -> Dict[str, Any]"
      description: "Validate all artifacts"
      requirementId: "REQ-OPS-006"
  instanceVariables:
    - name: "status"
      type: "OperationStatus"
      description: "Status of the operation"
    - name: "artifacts"
      type: "List[OperationArtifact]"
      description: "Artifacts produced by the operation"
    - name: "metrics"
      type: "Dict[str, Any]"
      description: "Metrics collected during the operation"
    - name: "error_message"
      type: "Optional[str]"
      description: "Error message if operation failed"
    - name: "execution_time"
      type: "Optional[float]"
      description: "Execution time in seconds"
```

### 4.4 OperationStatus Enum

```yaml
enum:
  name: "OperationStatus"
  description: "Status codes for operation results"
  moduleLocation: "pamola_core.utils.ops.op_result"
  importStyle: "from pamola_core.utils.ops.op_result import OperationStatus"
  values:
    - name: "SUCCESS"
      description: "Operation completed successfully"
    - name: "WARNING"
      description: "Operation completed with some issues"
    - name: "ERROR"
      description: "Operation failed"
    - name: "SKIPPED"
      description: "Operation was skipped"
    - name: "PARTIAL_SUCCESS"
      description: "Operation completed but with some parts failed"
    - name: "PENDING"
      description: "Operation is still running or pending execution"
```

## 5. Operation Registration

```yaml
operationRegistration:
  description: "Register operations for discovery and usage"
  options:
    - type: "Function call (preferred)"
      syntax: "register_operation(MyOperation)"
      location: "End of module, after class definition"
    - type: "Decorator (alternative)"
      syntax: "@register_operation"
      location: "Before class definition"
  moduleLocation: "pamola_core.utils.ops.op_registry"
  importStyle: "from pamola_core.utils.ops.op_registry import register_operation"
  example: |
    from pamola_core.utils.ops.op_registry import register_operation
    
    class MyOperation(BaseOperation):
        # Class implementation...
    
    # Register the operation
    register_operation(MyOperation)
```

## 6. Directory Structure

```yaml
directoryStructure:
  name: "task_dir"
  description: "Root directory for operation artifacts"
  creation: "Automatically created by operation framework or provided by user script"
  subdirectories:
    - path: "output/"
      description: "Directory for operation output files"
      usage: "Operation-processed data files (CSV, Parquet, etc.)"
    - path: "dictionaries/"
      description: "Directory for extracted dictionaries and mappings"
      usage: "Lookup tables, mappings, and extracted data"
    - path: "logs/"
      description: "Directory for operation logs"
      usage: "Log files generated during operation execution"
  files:
    - path: "config.json"
      description: "Operation configuration"
      creation: "Via self.save_config(task_dir)"
    - path: "{metrics_name}.json"
      description: "Operation metrics"
      creation: "Via DataWriter.write_metrics()"
```

## 7. Exceptions Hierarchy

```yaml
exceptions:
  baseClass:
    name: "OpsError"
    moduleLocation: "pamola_core.utils.ops"
    description: "Base exception for all operations framework errors"
  subclasses:
    - name: "ConfigError"
      moduleLocation: "pamola_core.utils.ops.op_config"
      description: "Configuration validation errors"
      whenToRaise: "When configuration parameters fail schema validation"
    - name: "ConfigSaveError"
      moduleLocation: "pamola_core.utils.ops.op_base"
      description: "Error saving configuration"
      whenToRaise: "When saving configuration to JSON fails"
    - name: "DataWriteError"
      moduleLocation: "pamola_core.utils.ops.op_data_writer"
      description: "Error writing data"
      whenToRaise: "When writing data to disk fails"
    - name: "RegistryError"
      moduleLocation: "pamola_core.utils.ops.op_registry"
      description: "Operation registry errors"
      whenToRaise: "When registering or looking up operations fails"
    - name: "ValidationError"
      moduleLocation: "pamola_core.utils.ops.op_result"
      description: "Artifact validation errors"
      whenToRaise: "When validating artifacts fails"
    - name: "CacheError"
      moduleLocation: "pamola_core.utils.ops.op_cache"
      description: "Operation cache errors"
      whenToRaise: "When caching or retrieving cached results fails"
  style:
    derivation: "Always derive from OpsError or its subclasses"
    messaging: "Always include descriptive error message"
    example: |
      class ConfigError(OpsError):
          """Error in configuration parameters."""
          pass
```

## 8. Test Helpers

```yaml
testHelpers:
  moduleLocation: "pamola_core.utils.ops.op_test_helpers"
  importStyle: "from pamola_core.utils.ops.op_test_helpers import create_test_operation_env, MockDataSource, StubDataWriter, assert_artifact_exists, assert_metrics_content"
  classes:
    - name: "MockDataSource"
      description: "Test double for DataSource that works with in-memory DataFrames"
      init:
        signature: "__init__(self, dataframes: Optional[Dict[str, pd.DataFrame]] = None) -> None"
        description: "Initialize with optional dictionary of DataFrames"
      factoryMethods:
        - name: "from_dataframe"
          signature: "from_dataframe(cls, df: pd.DataFrame, name: str = 'main') -> MockDataSource"
          description: "Create MockDataSource from a single DataFrame"
      methods:
        - name: "add_dataframe"
          signature: "add_dataframe(self, name: str, df: pd.DataFrame) -> None"
          description: "Add a DataFrame to the data source"
        - name: "get_dataframe"
          signature: "get_dataframe(self, name: str, **kwargs) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]"
          description: "Get a DataFrame by name with error information"
        - name: "get_schema"
          signature: "get_schema(self, name: str) -> Optional[Dict]"
          description: "Get schema information for a DataFrame"
    
    - name: "StubDataWriter"
      description: "Test double for DataWriter that records calls and writes to temporary directory"
      init:
        signature: "__init__(self, task_dir: Path, logger: Optional[logging.Logger] = None) -> None"
        description: "Initialize with task directory and optional logger"
      methods:
        - name: "write_dataframe"
          signature: "write_dataframe(self, df: pd.DataFrame, name: str, format: str = 'csv', subdir: str = 'output', timestamp_in_name: bool = False, encryption_key: Optional[Union[str, Path]] = None, **kwargs) -> WriterResult"
          description: "Stub for writing DataFrame"
        - name: "write_json"
          signature: "write_json(self, data: Any, name: str, subdir: Optional[str] = None, timestamp_in_name: bool = True, encryption_key: Optional[Union[str, Path]] = None, pretty: bool = True, **kwargs) -> WriterResult"
          description: "Stub for writing JSON"
        - name: "get_calls"
          signature: "get_calls(self, method_name: Optional[str] = None) -> List[CallRecord]"
          description: "Get recorded calls, optionally filtered by method name"
        - name: "clear_calls"
          signature: "clear_calls(self) -> None"
          description: "Clear recorded calls"
  
  functions:
    - name: "create_test_operation_env"
      signature: "create_test_operation_env(tmp_path: Path, config_overrides: Optional[Dict[str, Any]] = None) -> Tuple[Path, Dict[str, Any]]"
      description: "Create test environment with task directory and config"
    
    - name: "assert_artifact_exists"
      signature: "assert_artifact_exists(task_dir: Path, subdir: Optional[str], filename_pattern: str) -> Path"
      description: "Assert that an artifact exists matching the pattern"
    
    - name: "assert_metrics_content"
      signature: "assert_metrics_content(task_dir: Path, expected_metrics: Dict[str, Any]) -> Dict[str, Any]"
      description: "Assert that metrics file contains expected metrics"
  
  usage: |
    def test_my_operation(tmp_path):
        # Create test data
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        
        # Create test environment
        data_source = MockDataSource.from_dataframe(df)
        task_dir, config = create_test_operation_env(tmp_path)
        writer = StubDataWriter(task_dir)
        
        # Create and execute operation
        op = MyOperation(column_name="value")
        result = op.execute(data_source, task_dir, None, writer=writer)
        
        # Assertions
        assert result.status == OperationStatus.SUCCESS
        output_file = assert_artifact_exists(task_dir, "output", r"processed_.*\.csv")
        metrics = assert_metrics_content(task_dir, {"processed_count": 3})
        
        # Check writer calls
        df_calls = writer.get_calls("write_dataframe")
        assert len(df_calls) == 1
```

## 9. Code Style & Conventions

```yaml
codeStyle:
  docstringStyle: "Google"
  lineLength: 100
  typeHints: 
    required: true
    future: "from __future__ import annotations"
  namingConventions:
    classes: "PascalCase"
    methods: "snake_case"
    variables: "snake_case"
    constants: "UPPER_SNAKE_CASE"
  importOrder:
    - "Standard library imports (sorted alphabetically)"
    - "Third-party imports (sorted alphabetically)"
    - "Core framework imports (sorted alphabetically)"
    - "Project-specific imports (sorted alphabetically)"
  importStyle: "Absolute imports preferred (from pamola_core.utils.ops...)"
  indentation: "4 spaces"
  todoStyle: 
    format: "# TODO: {description}"
    location: "End of function, beginning of file, or after relevant line"
  deprecatedStyle:
    format: "# DEPRECATED: {reason}"
    warning: "warnings.warn('Function X is deprecated: {reason}', DeprecationWarning, stacklevel=2)"
  loggingConventions:
    naming: "self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')"
    levels:
      debug: "Detailed development information"
      info: "Normal successful operations"
      warning: "Non-critical issues"
      error: "Operation failures"
      critical: "System-level failures"
    messages: "Should be specific, include parameter values, avoid PII"
```

## 10. Lifecycle Sequence

```yaml
lifecycleSequence:
  stages:
    - name: "Initialization"
      description: "Operation creation and setup"
      steps:
        - operation: "Create operation instance with parameters"
          code: "op = MyOperation(field_name='value')"
    
    - name: "Execution Start"
      description: "Beginning of operation execution"
      steps:
        - operation: "Call run or execute method"
          code: "result = op.run(data_source, task_dir, reporter)"
        - operation: "Save configuration to task directory"
          code: "self.save_config(task_dir)"
          mustCall: true
        - operation: "Set up progress tracking"
          code: "if progress_tracker: progress_tracker.update(0, {'status': 'starting'})"
    
    - name: "Data Access"
      description: "Retrieving input data"
      steps:
        - operation: "Get data from source"
          code: "df, error = data_source.get_dataframe('main')"
          mustCall: true
        - operation: "Check for data loading errors"
          code: "if df is None: return OperationResult(status=OperationStatus.ERROR, error_message=error['message'])"
    
    - name: "Input Validation"
      description: "Validating input data and parameters"
      steps:
        - operation: "Check required fields exist"
          code: "if self.field_name not in df.columns: raise ValueError(f'Field {self.field_name} not found')"
        - operation: "Validate data types"
          code: "if not pd.api.types.is_numeric_dtype(df[self.field_name]): logger.warning('Field is not numeric')"
    
    - name: "Processing"
      description: "Main operation logic"
      steps:
        - operation: "Process data - operation-specific logic"
          code: "# Implementation varies by operation type"
        - operation: "Update progress"
          code: "if progress_tracker: progress_tracker.update(step, {'status': 'processing'})"
    
    - name: "Metrics Collection"
      description: "Gathering operation metrics"
      steps:
        - operation: "Calculate metrics"
          code: "metrics = self._collect_metrics(original_data, processed_data)"
        - operation: "Write metrics to file"
          code: "writer.write_metrics(metrics, 'operation_metrics')"
    
    - name: "Result Generation"
      description: "Creating output files and artifacts"
      steps:
        - operation: "Write output data"
          code: "writer.write_dataframe(df, 'processed_data', subdir='output')"
        - operation: "Generate visualizations"
          code: "self._generate_visualizations(original_data, processed_data, task_dir, result, reporter)"
    
    - name: "Completion"
      description: "Finalizing operation"
      steps:
        - operation: "Create result object with status"
          code: "result = OperationResult(status=OperationStatus.SUCCESS)"
        - operation: "Add metrics to result"
          code: "for key, value in metrics.items(): result.add_metric(key, value)"
        - operation: "Register artifacts"
          code: "result.add_artifact('csv', output_path, 'Processed data')"
        - operation: "Update final progress"
          code: "if progress_tracker: progress_tracker.update(total_steps, {'status': 'complete'})"
```

## 11. CI Validation & Checks

```yaml
ciChecks:
  schemaLocation: "pamola_core/utils/ops/ci/operation_schema.json"
  codeQuality:
    pylint:
      minScore: 8.0
      disabledRules: ["C0103", "C0111"]
    mypy:
      strictMode: true
      disallow_untyped_defs: true
  testRequirements:
    minCoverage: 85
    requiredTests:
      - "test_{operation_name}_success"
      - "test_{operation_name}_error_handling"
    pythonVersions: ["3.8", "3.9", "3.10"]
  operationChecks:
    validateJsonSchema: |
      {
        "type": "object",
        "properties": {
          "name": {"type": "string", "minLength": 3},
          "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
          "execute": {"type": "method", "required": true}
        },
        "required": ["name", "version", "execute"]
      }
    validateMethodPresence:
      - "__init__"
      - "execute"
    validateFile:
      - "Must import BaseOperation or subclass"
      - "Must call register_operation"
      - "Must have class docstring"
```

## 12. Working Examples

### 12.1 Minimal Example (BaseOperation)

```python
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: SimpleOperation
Description: Simple example operation
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This operation multiplies a numeric column by a specified factor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker


class SimpleOperationConfig(OperationConfig):
    """Configuration for SimpleOperation."""
    
    schema = {
        "type": "object",
        "properties": {
            "operation_name": {"type": "string"},
            "version": {"type": "string"},
            "column_name": {"type": "string"},
            "multiplier": {"type": "number"}
        },
        "required": ["operation_name", "version", "column_name"]
    }


class SimpleOperation(BaseOperation):
    """Simple operation that multiplies a column by a value."""
    
    def __init__(
        self,
        name: str = "simple_operation",
        description: str = "Multiply column by value",
        column_name: Optional[str] = None,
        multiplier: float = 2.0,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the operation.
        
        Args:
            name: Name of the operation
            description: Description of what the operation does
            column_name: Name of the column to process
            multiplier: Factor to multiply the column by
            use_encryption: Whether to encrypt output files
            encryption_key: The encryption key or path to a key file
        """
        # Create configuration
        config = SimpleOperationConfig(
            column_name=column_name,
            multiplier=multiplier
        )
        
        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )
        
        # Store operation-specific parameters
        self.column_name = column_name
        self.multiplier = multiplier
        self.version = "1.0.0"
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[ProgressTracker] = None,
        **kwargs
    ) -> OperationResult:
        """
        Execute the operation.
        
        Args:
            data_source: Source of data for the operation
            task_dir: Directory where task artifacts should be saved
            reporter: Reporter object for tracking progress and artifacts
            progress_tracker: Progress tracker for the operation
            **kwargs: Additional parameters for the operation
            
        Returns:
            Results of the operation
        """
        self.logger.info(f"Starting {self.name} operation")
        
        # Initialize result and writer
        result = OperationResult(status=OperationStatus.PENDING)
        writer = DataWriter(task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker)
        
        # Save configuration
        self.save_config(task_dir)
        
        try:
            # Set up progress tracking
            total_steps = 4
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "starting"})
            
            # Get input data
            self.logger.info("Loading input data")
            df, error_info = data_source.get_dataframe("main")
            
            if df is None:
                error_message = f"Failed to load input data: {error_info['message'] if error_info else 'Unknown error'}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message
                )
            
            if progress_tracker:
                progress_tracker.update(1, {"status": "data_loaded", "rows": len(df)})
            
            # Validate inputs
            if self.column_name not in df.columns:
                error_message = f"Column '{self.column_name}' not found in input data"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message
                )
            
            # Process data
            self.logger.info(f"Processing column: {self.column_name}")
            
            # Make a copy of the original data for metrics calculation
            original_data = df[self.column_name].copy()
            
            # Create output column name
            output_column = f"{self.column_name}_multiplied"
            
            # Multiply values
            df[output_column] = df[self.column_name].apply(
                lambda x: x * self.multiplier if isinstance(x, (int, float)) else x
            )
            
            if progress_tracker:
                progress_tracker.update(2, {"status": "processing_complete"})
            
            # Calculate metrics
            metrics = {
                "row_count": len(df),
                "column_name": self.column_name,
                "output_column": output_column,
                "multiplier_used": self.multiplier,
                "null_values_count": df[self.column_name].isna().sum(),
                "processed_values_count": (~df[output_column].isna()).sum()
            }
            
            # Write metrics file
            metrics_result = writer.write_metrics(
                metrics,
                "operation_metrics", 
                timestamp_in_name=True
            )
            
            if progress_tracker:
                progress_tracker.update(3, {"status": "metrics_calculated"})
            
            # Write output data
            output_result = writer.write_dataframe(
                df,
                name="processed_data",
                format="csv",
                subdir="output",
                encryption_key=self.encryption_key if self.use_encryption else None
            )
            
            if progress_tracker:
                progress_tracker.update(4, {"status": "complete"})
            
            # Add metrics to result
            for key, value in metrics.items():
                result.add_metric(key, value)
            
            # Register artifacts
            result.add_artifact(
                artifact_type="csv",
                path=output_result.path,
                description=f"Processed data with {self.column_name} transformation",
                category="output"
            )
            
            result.add_artifact(
                artifact_type="json",
                path=metrics_result.path,
                description="Operation metrics",
                category="metrics"
            )
            
            # Set success status
            result.status = OperationStatus.SUCCESS
            self.logger.info(f"Operation {self.name} completed successfully")
            
            return result
            
        except Exception as e:
            # Handle any errors
            error_message = f"Error in {self.name} operation: {str(e)}"
            self.logger.exception(error_message)
            
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_message
            )


# Register the operation so it's discoverable
register_operation(SimpleOperation)
```

### 12.2 FieldOperation Example

```python
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: FieldMaskingOperation
Description: Field-level masking operation
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This operation implements field-level masking of sensitive data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker


class FieldMaskingConfig(OperationConfig):
    """Configuration for FieldMaskingOperation."""
    
    schema = {
        "type": "object",
        "properties": {
            "operation_name": {"type": "string"},
            "version": {"type": "string"},
            "mask_char": {"type": "string", "minLength": 1, "maxLength": 1},
            "preserve_start": {"type": "integer", "minimum": 0},
            "preserve_end": {"type": "integer", "minimum": 0}
        },
        "required": ["operation_name", "version"]
    }


class FieldMaskingOperation(FieldOperation):
    """
    Operation for masking field values with a specified character.
    
    This operation masks text field values, optionally preserving 
    some characters at the start and/or end.
    """
    
    def __init__(
        self,
        name: str = "field_masking",
        description: str = "Mask field with specified character",
        field_name: Optional[str] = None,
        mask_char: str = "*",
        preserve_start: int = 0,
        preserve_end: int = 0,
        mode: str = "ENRICH",
        output_field_name: Optional[str] = None,
        field_prefix: str = "",
        field_suffix: str = "_masked",
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize field masking operation.
        
        Args:
            name: Name of the operation
            description: Description of what the operation does
            field_name: Field to mask
            mask_char: Character to use for masking
            preserve_start: Number of characters to preserve at start
            preserve_end: Number of characters to preserve at end
            mode: ENRICH (create new field) or REPLACE (modify existing)
            output_field_name: Explicit output field name
            field_prefix: Prefix for output field name
            field_suffix: Suffix for output field name
            use_encryption: Whether to encrypt output files
            encryption_key: The encryption key or path to a key file
        """
        # Validate mask_char
        if len(mask_char) != 1:
            raise ValueError("mask_char must be a single character")
        
        # Create configuration
        config = FieldMaskingConfig(
            mask_char=mask_char,
            preserve_start=preserve_start,
            preserve_end=preserve_end
        )
        
        # Initialize base class (FieldOperation)
        super().__init__(
            name=name,
            description=description,
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            field_prefix=field_prefix,
            field_suffix=field_suffix,
            config=config,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )
        
        # Store operation-specific parameters
        self.mask_char = mask_char
        self.preserve_start = preserve_start
        self.preserve_end = preserve_end
        self.version = "1.0.0"
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def mask_value(self, value: Any) -> Any:
        """
        Mask a single value.
        
        Args:
            value: Value to mask
            
        Returns:
            Masked value
        """
        # Skip non-string values
        if not isinstance(value, str) or pd.isna(value):
            return value
        
        # Calculate how many characters to mask
        total_length = len(value)
        masked_length = max(0, total_length - self.preserve_start - self.preserve_end)
        
        if masked_length <= 0:
            return value
        
        # Create masked value
        start = value[:self.preserve_start] if self.preserve_start > 0 else ""
        end = value[-self.preserve_end:] if self.preserve_end > 0 else ""
        masked = self.mask_char * masked_length
        
        return start + masked + end
    
    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[ProgressTracker] = None,
        **kwargs
    ) -> OperationResult:
        """
        Execute the operation.
        
        Args:
            data_source: Source of data for the operation
            task_dir: Directory where task artifacts should be saved
            reporter: Reporter object for tracking progress and artifacts
            progress_tracker: Progress tracker for the operation
            **kwargs: Additional parameters for the operation
            
        Returns:
            Results of the operation
        """
        self.logger.info(f"Starting {self.name} operation on field {self.field_name}")
        
        # Initialize result and writer
        result = OperationResult(status=OperationStatus.PENDING)
        writer = DataWriter(task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker)
        
        # Save configuration
        self.save_config(task_dir)
        
        try:
            # Set up progress tracking
            total_steps = 4
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "starting"})
            
            # Get input data
            self.logger.info("Loading input data")
            df, error_info = data_source.get_dataframe("main")
            
            if df is None:
                error_message = f"Failed to load input data: {error_info['message'] if error_info else 'Unknown error'}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message
                )
            
            if progress_tracker:
                progress_tracker.update(1, {"status": "data_loaded", "rows": len(df)})
            
            # Validate inputs
            if self.field_name not in df.columns:
                error_message = f"Field '{self.field_name}' not found in input data"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message
                )
            
            # Make a copy of the original data for metrics calculation
            original_data = df[self.field_name].copy()
            
            # Determine output field name
            if self.mode == "REPLACE":
                output_field = self.field_name
            else:  # ENRICH mode
                if self.output_field_name:
                    output_field = self.output_field_name
                else:
                    output_field = f"{self.field_prefix}{self.field_name}{self.field_suffix}"
            
            # Process data - apply masking
            self.logger.info(f"Masking field: {self.field_name}")
            df[output_field] = df[self.field_name].apply(self.mask_value)
            
            if progress_tracker:
                progress_tracker.update(2, {"status": "processing_complete"})
            
            # Calculate metrics
            metrics = {
                "field_name": self.field_name,
                "output_field": output_field,
                "total_values": len(df),
                "null_values": df[self.field_name].isna().sum(),
                "masked_values": (df[self.field_name].notna() & df[self.field_name].astype(str).str.len() > 0).sum(),
                "mask_character": self.mask_char,
                "preserve_start": self.preserve_start,
                "preserve_end": self.preserve_end,
                "mode": self.mode
            }
            
            # Write metrics file
            metrics_result = writer.write_metrics(
                metrics,
                "operation_metrics", 
                timestamp_in_name=True
            )
            
            if progress_tracker:
                progress_tracker.update(3, {"status": "metrics_calculated"})
            
            # Write output data
            output_result = writer.write_dataframe(
                df,
                name="processed_data",
                format="csv",
                subdir="output",
                encryption_key=self.encryption_key if self.use_encryption else None
            )
            
            if progress_tracker:
                progress_tracker.update(4, {"status": "complete"})
            
            # Add metrics to result
            for key, value in metrics.items():
                result.add_metric(key, value)
            
            # Register artifacts
            result.add_artifact(
                artifact_type="csv",
                path=output_result.path,
                description=f"Data with masked {self.field_name} field",
                category="output"
            )
            
            result.add_artifact(
                artifact_type="json",
                path=metrics_result.path,
                description="Operation metrics",
                category="metrics"
            )
            
            # Set success status
            result.status = OperationStatus.SUCCESS
            self.logger.info(f"Operation {self.name} completed successfully")
            
            return result
            
        except Exception as e:
            # Handle any errors
            error_message = f"Error in {self.name} operation: {str(e)}"
            self.logger.exception(error_message)
            
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_message
            )


# Register the operation
register_operation(FieldMaskingOperation)
```

### 12.3 Unit Testing Example

```python
"""
Unit tests for SimpleOperation.
"""

import pandas as pd
import pytest
from pathlib import Path

from pamola_core.utils.ops.op_result import OperationStatus
from pamola_core.utils.ops.op_test_helpers import (
    MockDataSource,
    StubDataWriter,
    assert_artifact_exists,
    assert_metrics_content,
    create_test_operation_env
)
from your_package.operations.simple_operation import SimpleOperation


def test_simple_operation_success(tmp_path):
    """Test successful execution of SimpleOperation."""
    # Create test data
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10, 20, 30, 40, 50]
    })
    
    # Create test environment
    data_source = MockDataSource.from_dataframe(df, name="main")
    task_dir, config = create_test_operation_env(
        tmp_path,
        {"column_name": "value", "multiplier": 2.0}
    )
    
    # Create operation
    operation = SimpleOperation(
        column_name="value",
        multiplier=2.0
    )
    
    # Execute operation
    result = operation.execute(
        data_source=data_source,
        task_dir=task_dir,
        reporter=None
    )
    
    # Check status
    assert result.status == OperationStatus.SUCCESS
    
    # Check metrics
    assert result.metrics["row_count"] == 5
    assert result.metrics["multiplier_used"] == 2.0
    
    # Check artifacts
    output_file = assert_artifact_exists(task_dir, "output", r"processed_data\.csv")
    metrics_file = assert_metrics_content(
        task_dir,
        {"row_count": 5, "column_name": "value"}
    )
    
    # Verify processed data has the right values
    processed_df = pd.read_csv(output_file)
    assert "value_multiplied" in processed_df.columns
    
    # Check multiplied values
    for i, row in processed_df.iterrows():
        assert row["value_multiplied"] == row["value"] * 2.0


def test_simple_operation_missing_field(tmp_path):
    """Test error handling when field is missing."""
    # Create test data without the required field
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        # "value" field is missing
    })
    
    # Create test environment
    data_source = MockDataSource.from_dataframe(df, name="main")
    task_dir, _ = create_test_operation_env(tmp_path)
    
    # Create operation
    operation = SimpleOperation(column_name="value")
    
    # Execute operation - should fail
    result = operation.execute(
        data_source=data_source,
        task_dir=task_dir,
        reporter=None
    )
    
    # Check error status
    assert result.status == OperationStatus.ERROR
    assert "not found in input data" in result.error_message


def test_simple_operation_with_writer(tmp_path):
    """Test operation with StubDataWriter."""
    # Create test data
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10, 20, 30, 40, 50]
    })
    
    # Create test environment
    data_source = MockDataSource.from_dataframe(df, name="main")
    task_dir, _ = create_test_operation_env(tmp_path)
    writer = StubDataWriter(task_dir)
    
    # Create operation
    operation = SimpleOperation(column_name="value")
    
    # Execute operation
    result = operation.execute(
        data_source=data_source,
        task_dir=task_dir,
        reporter=None,
        writer=writer
    )
    
    # Check status
    assert result.status == OperationStatus.SUCCESS
    
    # Check writer calls
    df_calls = writer.get_calls("write_dataframe")
    assert len(df_calls) >= 1
    
    metrics_calls = writer.get_calls("write_metrics")
    assert len(metrics_calls) >= 1
    
    # Verify writer params
    assert df_calls[0].params["name"] == "processed_data"
    assert df_calls[0].params["format"] == "csv"
```

## 13. Validation Schema for Operations

```yaml
validationSchema:
  type: "object"
  title: "PAMOLA Operation Validator"
  description: "Schema for validating PAMOLA operations"
  required: ["name", "description", "version", "logger"]
  properties:
    name:
      type: "string"
      description: "Name of the operation"
      minLength: 3
    description:
      type: "string"
      description: "Description of the operation"
    version:
      type: "string"
      description: "Semantic version"
      pattern: "^\\d+\\.\\d+\\.\\d+$"
    logger:
      type: "object"
      description: "Logger instance"
    execute:
      type: "function"
      description: "The execute method"
      required: true
    run:
      type: "function"
      description: "The run method"
      required: false
  patternProperties:
    "^[a-z][a-z0-9_]*$":
      description: "Instance variables should use snake_case"
  dependencies:
    use_encryption: ["encryption_key"]
    mode:
      oneOf:
        - properties:
            mode: { enum: ["ENRICH"] }
          required: ["field_prefix", "field_suffix"]
        - properties:
            mode: { enum: ["REPLACE"] }
```