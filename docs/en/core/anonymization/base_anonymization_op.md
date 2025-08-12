# PAMOLA.CORE Base Anonymization Operation Module Documentation

**Module:** `pamola_core.anonymization.base_anonymization_op`  
**Version:** 1.0.0  
**Last Updated:** 2025-07-29

---

> **Canonical Location:**
> This document is the **unique, canonical reference** for the abstract base class of all anonymization operations in PAMOLA.CORE. All module-level operation documentation (e.g., masking, generalization, suppression) must reference this file for base class details and must not duplicate its content. For implementation-specific details, see the respective module operation docs.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation & Dependencies](#installation--dependencies)
5. [Core Classes](#core-classes)
6. [Usage Guide](#usage-guide)
7. [Examples](#examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Inheritance in Module Operations](#inheritance-in-module-operations)
11. [References & Cross-Module Usage](#references--cross-module-usage)
12. [Unit Test Coverage](#unit-test-coverage)
13. [Summary Analysis](#8-summary-analysis)
14. [Technical Summary](#technical-summary)

## Overview

The `base_anonymization_op.py` module provides the foundational base class for all anonymization operations in the PAMOLA.CORE framework. It implements a standardized interface for privacy-preserving data transformations while handling common functionality such as data loading, chunking, progress tracking, metrics collection, and visualization generation.

### Purpose

This module serves as the abstract base class that:
- Standardizes the interface for all anonymization operations
- Provides common functionality to avoid code duplication
- Ensures consistent behavior across different anonymization techniques
- Integrates with the PAMOLA.CORE operations framework
- Handles large dataset processing efficiently
- Supports conditional anonymization based on profiling results

## Key Features

### 1. **Framework Integration**
- Full integration with PAMOLA.CORE operations framework
- Uses `DataSource` for input abstraction
- Returns `OperationResult` with metrics and artifacts
- Integrates with `DataWriter` for standardized output

### 2. **Processing Modes**
- **REPLACE**: Modifies the original field in-place
- **ENRICH**: Creates a new field with anonymized values

### 3. **Conditional Processing**
- Single-field conditions with multiple operators
- Multi-field conditions with AND/OR logic
- K-anonymity risk-based processing
- Integration with profiling results

### 4. **Memory Management**
- Automatic DataFrame dtype optimization
- Adaptive batch sizing based on available memory
- Chunked processing for large datasets
- Forced garbage collection after processing

### 5. **Null Value Strategies**
- **PRESERVE**: Keep null values unchanged
- **EXCLUDE**: Remove null values before processing
- **ERROR**: Raise error if nulls are encountered
- **ANONYMIZE**: Apply anonymization to null values

### 6. **Comprehensive Metrics**
- Anonymization effectiveness metrics
- Performance metrics (time, throughput)
- Memory usage tracking
- Conditional processing statistics
- Privacy risk metrics

### 7. **Visualization Generation**
- Before/after comparison charts
- Distribution visualizations
- Automatic artifact registration
- Consistent naming conventions

## Architecture

### Class Hierarchy

```
pamola_core.utils.ops.op_base.BaseOperation
    └── pamola_core.anonymization.base_anonymization_op.AnonymizationOperation
            └── [Your Custom Anonymization Operation]
```

### Component Integration

```python
┌─────────────────────────────────────────────────────────────────┐
│                     AnonymizationOperation                      │
├─────────────────────────────────────────────────────────────────┤
│ Framework Integration:                                          │
│ - pamola_core.utils.ops.op_data_processing (memory optimization)      │
│ - pamola_core.utils.ops.op_field_utils (field management)             │
│ - pamola_core.utils.ops.op_data_source (data loading)                 │
│ - pamola_core.utils.ops.op_data_writer (output handling)              │
│ - pamola_core.utils.ops.op_result (result management)                 │
│ - pamola_core.utils.progress (progress tracking)                      │
├─────────────────────────────────────────────────────────────────┤
│ Privacy-Specific Utilities:                                     │
│ - pamola_core.anonymization.commons.data_utils                        │
│ - pamola_core.anonymization.commons.metric_utils                      │
│ - pamola_core.anonymization.commons.visualization_utils               │
└─────────────────────────────────────────────────────────────────┘
```

## Installation & Dependencies

### Required Framework Modules
```python
# Operations framework
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# Data processing utilities
from pamola_core.utils.ops.op_data_processing import (
    optimize_dataframe_dtypes,
    get_memory_usage,
    force_garbage_collection,
    get_dataframe_chunks
)

# Field utilities
from pamola_core.utils.ops.op_field_utils import (
    generate_output_field_name,
    create_field_mask,
    create_multi_field_mask
)

# Progress tracking
from pamola_core.utils.progress import ProgressTracker
```

### Privacy-Specific Dependencies
```python
# Anonymization utilities
from pamola_core.anonymization.commons.data_utils import handle_vulnerable_records
from pamola_core.anonymization.commons.metric_utils import calculate_anonymization_effectiveness
from pamola_core.anonymization.commons.visualization_utils import (
    register_visualization_artifact,
    create_metric_visualization,
    create_comparison_visualization
)
```

## Core Classes

### AnonymizationOperation (Abstract Base Class)

The main base class for all anonymization operations. **All concrete anonymization modules (e.g., masking, generalization, suppression) must inherit from this class.**

#### Constructor Signature

```python
def __init__(self,
             field_name: str,
             mode: str = "REPLACE",
             output_field_name: Optional[str] = None,
             column_prefix: str = "_",
             null_strategy: str = "PRESERVE",
             batch_size: int = 10000,
             use_cache: bool = True,
             description: str = "",
             use_encryption: bool = False,
             encryption_key: Optional[Union[str, Path]] = None,
             # Conditional processing parameters
             condition_field: Optional[str] = None,
             condition_values: Optional[List] = None,
             condition_operator: str = "in",
             # Multi-field conditions
             multi_conditions: Optional[List[Dict[str, Any]]] = None,
             multi_condition_logic: str = "AND",
             # K-anonymity integration
             ka_risk_field: Optional[str] = None,
             risk_threshold: float = 5.0,
             vulnerable_record_strategy: str = "suppress",
             # Memory optimization
             optimize_memory: bool = True,
             adaptive_batch_size: bool = True)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | str | Required | Field to anonymize |
| `mode` | str | "REPLACE" | Processing mode: "REPLACE" or "ENRICH" |
| `output_field_name` | Optional[str] | None | Custom name for output field (ENRICH mode) |
| `column_prefix` | str | "_" | Prefix for generated field names |
| `null_strategy` | str | "PRESERVE" | How to handle nulls: "PRESERVE", "EXCLUDE", "ERROR" |
| `batch_size` | int | 10000 | Records per processing batch |
| `use_cache` | bool | True | Enable operation result caching |
| `description` | str | "" | Operation description |
| `use_encryption` | bool | False | Encrypt output files |
| `encryption_key` | Optional[Union[str, Path]] | None | Encryption key or path |
| `condition_field` | Optional[str] | None | Field for conditional processing |
| `condition_values` | Optional[List] | None | Values for condition matching |
| `condition_operator` | str | "in" | Condition operator: "in", "not_in", "gt", "lt", "eq", "range" |
| `multi_conditions` | Optional[List[Dict]] | None | Multiple condition definitions |
| `multi_condition_logic` | str | "AND" | Logic for multiple conditions: "AND", "OR" |
| `ka_risk_field` | Optional[str] | None | Field with k-anonymity risk scores |
| `risk_threshold` | float | 5.0 | K-anonymity threshold for vulnerable records |
| `vulnerable_record_strategy` | str | "suppress" | Strategy for vulnerable records |
| `optimize_memory` | bool | True | Enable memory optimization |
| `adaptive_batch_size` | bool | True | Adjust batch size based on memory |

### Abstract Methods (Must Override)

#### 1. process_batch
```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """
    Process a batch of data. Must be implemented by subclasses.
    
    Parameters:
    -----------
    batch : pd.DataFrame
        DataFrame batch to process
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame batch
    """
```

#### 2. process_value
```python
def process_value(self, value, **params):
    """
    Process a single value. Must be implemented by subclasses.
    
    Parameters:
    -----------
    value : Any
        Value to process
    **params : dict
        Additional parameters
        
    Returns:
    --------
    Any
        Processed value
    """
```

#### 3. _collect_specific_metrics
```python
def _collect_specific_metrics(self,
                              original_data: pd.Series,
                              anonymized_data: pd.Series) -> Dict[str, Any]:
    """
    Collect operation-specific metrics. Should be overridden by subclasses.
    
    Parameters:
    -----------
    original_data : pd.Series
        Original data before anonymization
    anonymized_data : pd.Series
        Data after anonymization
        
    Returns:
    --------
    Dict[str, Any]
        Operation-specific metrics
    """
```

### Key Methods (Inherited)

#### execute
```python
def execute(self,
            data_source: DataSource,
            task_dir: Path,
            reporter: Any,
            progress_tracker: Optional[ProgressTracker] = None,
            **kwargs) -> OperationResult:
    """
    Execute the anonymization operation.
    
    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operation
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress
    progress_tracker : Optional[ProgressTracker]
        Progress tracker for the operation
    **kwargs : dict
        Additional parameters including profiling_results
        
    Returns:
    --------
    OperationResult
        Results of the operation with metrics and artifacts
    """
```

## Usage Guide

### Partial Masking Example
```python
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation

# Partial masking: preserve first 2 and last 4 characters
partial_op = PartialMaskingOperation(mask_char='*', unmasked_prefix=2, unmasked_suffix=4)
masked = partial_op.mask_partial_value('SensitiveData', {
    'mask_char': '*',
    'unmasked_prefix': 2,
    'unmasked_suffix': 4
})
# masked -> 'Se****iveData'
```

### Full Masking Example
```python
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation

# Full masking: mask all characters
full_op = FullMaskingOperation(mask_char='*')
masked_full = full_op.mask_full_value('SensitiveData', {'mask_char': '*'})
# masked_full -> '*************'
```

### Basic Usage Pattern

```python
from pathlib import Path
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.anonymization.your_module import YourAnonymizationOperation

# Create data source
data_source = DataSource.from_file_path("data.csv", name="main")

# Create operation instance
operation = YourAnonymizationOperation(
    field_name="salary",
    mode="REPLACE",
    null_strategy="PRESERVE"
)

# Execute operation
result = operation.execute(
    data_source=data_source,
    task_dir=Path("output/task_001"),
    reporter=None  # Optional reporter
)

# Check results
if result.status == OperationStatus.SUCCESS:
    print(f"Operation completed successfully")
    print(f"Metrics: {result.metrics}")
    print(f"Artifacts: {result.artifacts}")
else:
    print(f"Operation failed: {result.error_message}")
```

### Conditional Processing

```python
# Single field condition
operation = YourAnonymizationOperation(
    field_name="income",
    condition_field="age",
    condition_values=[18, 65],
    condition_operator="range"  # Only process ages 18-65
)

# Multiple conditions
operation = YourAnonymizationOperation(
    field_name="address",
    multi_conditions=[
        {"field": "country", "values": ["US", "UK"], "operator": "in"},
        {"field": "consent", "values": [True], "operator": "eq"}
    ],
    multi_condition_logic="AND"  # Both conditions must be true
)
```

### K-Anonymity Integration

```python
# Process only vulnerable records
operation = YourAnonymizationOperation(
    field_name="zip_code",
    ka_risk_field="k_anonymity_score",
    risk_threshold=5.0,  # Process records with k < 5
    vulnerable_record_strategy="generalize"
)

# Pass profiling results
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    profiling_results={
        "k_anonymity": {...},
        "vulnerability_assessment": {...}
    }
)
```

## Examples

### Example 1: Partial Masking Operation
```python
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
op = PartialMaskingOperation(mask_char='*', unmasked_prefix=2, unmasked_suffix=4)
masked = op.mask_partial_value('SensitiveData', {'mask_char': '*', 'unmasked_prefix': 2, 'unmasked_suffix': 4})
# masked -> 'Se****iveData'
```

### Example 2: Full Masking Operation
```python
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
op = FullMaskingOperation(mask_char='*')
masked = op.mask_full_value('SensitiveData', {'mask_char': '*'})
# masked -> '*************'
```

### Example 3: Simple Masking Operation
```python
class SimpleMaskingOp(AnonymizationOperation):
    def process_value(self, value, **params):
        if value is None:
            return value
        return "***MASKED***"
```

### Example 4: Conditional Generalization
```python
class AgeBinningOp(AnonymizationOperation):
    def process_value(self, value, **params):
        if value is None:
            return value
        if value < 18:
            return "<18"
        elif value < 65:
            return "18-64"
        else:
            return "65+"
```

---

## Best Practices

### 1. Memory Management

```python
# Always use chunked processing for large datasets
class MemoryEfficientOperation(AnonymizationOperation):
    def __init__(self, field_name: str, **kwargs):
        # Enable memory optimization
        super().__init__(
            field_name=field_name,
            optimize_memory=True,
            adaptive_batch_size=True,
            batch_size=5000,  # Smaller batches for memory-intensive ops
            **kwargs
        )
```

### 2. Error Handling

```python
def process_value(self, value, **params):
    try:
        # Your processing logic
        if not self._is_valid_value(value):
            self.logger.warning(f"Invalid value encountered: {value}")
            return self._get_default_value()
        
        return self._transform_value(value)
        
    except Exception as e:
        self.logger.error(f"Error processing value: {e}")
        if self.null_strategy == "ERROR":
            raise
        return value  # Return original on error
```

### 3. Metrics Collection

```python
def _collect_specific_metrics(self, original_data, anonymized_data):
    """Collect comprehensive metrics."""
    metrics = {}
    
    # Basic statistics
    metrics["original_unique"] = len(original_data.unique())
    metrics["anonymized_unique"] = len(anonymized_data.unique())
    metrics["reduction_ratio"] = 1 - (metrics["anonymized_unique"] / metrics["original_unique"])
    
    # Distribution metrics
    if pd.api.types.is_numeric_dtype(original_data):
        metrics["original_mean"] = original_data.mean()
        metrics["anonymized_mean"] = anonymized_data.mean()
        metrics["mean_difference"] = abs(metrics["original_mean"] - metrics["anonymized_mean"])
    
    # Privacy metrics
    metrics["information_loss"] = self._calculate_information_loss(
        original_data, anonymized_data
    )
    
    return metrics
```

### 4. Visualization

```python
def _generate_custom_visualizations(self, original_data, anonymized_data, task_dir):
    """Generate custom visualizations."""
    # Use the base class method for standard visualizations
    vis_paths = super()._generate_visualizations(
        original_data, anonymized_data, task_dir
    )
    
    # Add custom visualizations if needed
    try:
        # Your custom visualization logic
        custom_vis_path = self._create_custom_chart(
            original_data, anonymized_data, task_dir
        )
        vis_paths["custom"] = custom_vis_path
    except Exception as e:
        self.logger.warning(f"Failed to create custom visualization: {e}")
    
    return vis_paths
```

## Troubleshooting

### Common Issues

1. **Memory Errors with Large Datasets**
   ```python
   # Solution: Reduce batch size or enable adaptive sizing
   operation = YourOperation(
       field_name="large_field",
       batch_size=1000,  # Smaller batches
       adaptive_batch_size=True,
       optimize_memory=True
   )
   ```

2. **Slow Processing**
   ```python
   # Solution: Enable caching for repeated operations
   operation = YourOperation(
       field_name="field",
       use_cache=True
   )
   ```

3. **Null Value Errors**
   ```python
   # Solution: Choose appropriate null strategy
   operation = YourOperation(
       field_name="field",
       null_strategy="PRESERVE"  # or "EXCLUDE"
   )
   ```

4. **Conditional Processing Not Working**
   ```python
   # Solution: Verify condition field exists and values are correct type
   # Check your data
   print(df["condition_field"].unique())
   print(df["condition_field"].dtype)
   
   # Ensure correct operator
   operation = YourOperation(
       field_name="field",
       condition_field="age",
       condition_values=[18, 65],
       condition_operator="range"  # not "in" for ranges
   )
   ```

### Debugging Tips

1. **Enable Debug Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Operation Parameters**
   ```python
   # After creating operation
   print(f"Cache parameters: {operation._get_cache_parameters()}")
   print(f"Batch size: {operation.batch_size}")
   ```

3. **Inspect Results**
   ```python
   result = operation.execute(...)
   
   # Check metrics
   for key, value in result.metrics.items():
       print(f"{key}: {value}")
   
   # Check artifacts
   for artifact in result.artifacts:
       print(f"Type: {artifact.artifact_type}, Path: {artifact.path}")
   ```

## Unit Test Coverage

6 tests are skipped due to codebase implementation issues (e.g., chunk size mismatch, ValueError for empty condition_values, NotImplementedError not raised, KeyError: None, etc.). All skips are documented in the test file and summary report. Remaining tests pass and coverage is otherwise complete. File cannot be marked as 'Full' until codebase is fixed and all tests pass.

## 8. Summary Analysis

The `AnonymizationOperation` base class is the cornerstone of the PAMOLA anonymization framework. It:

- Defines the unified interface, contract, and extensibility model for all anonymization operations
- Serves as the single source of truth for all module-level operation implementations
- Enforces standardized method signatures for process_batch, process_value, and metrics collection
- Provides built-in memory management and chunked processing for large datasets
- Supports conditional processing and k-anonymity integration
- Ensures robust, testable, and fully integrated derived operations (masking, generalization, suppression, etc.)
- Enables maintainable, scalable, and compliant privacy solutions across diverse data domains

---

## Technical Summary

The `AnonymizationOperation` base class is the foundation of the PAMOLA anonymization framework. It provides:
- A unified, extensible interface for all anonymization operations
- Centralized memory management, batch processing, and metrics collection
- Support for conditional processing, k-anonymity, and advanced privacy controls
- A robust, testable, and scalable base for implementing custom privacy-preserving techniques

By following the extension and onboarding guidance in this documentation, developers can rapidly build, test, and maintain new anonymization modules that are fully integrated with the PAMOLA.CORE platform.

---

> **Documentation Policy for Module-Level Operations**
>
> - **All module-level operation documentation (e.g., masking, generalization, suppression) must include a reference section pointing to this file as the canonical parent class documentation.**
> - **Do not duplicate the base class interface, constructor, or parameter documentation in module operation docs.**
> - For module-specific logic, parameters, and examples, document only the unique aspects in the module operation doc and refer to this file for all shared base class details.
> - Example reference section for module docs:
>
>   > **Reference:** This operation inherits from the abstract base class documented in [base_anonymization_op.md](../base_anonymization_op.md). See that file for all shared parameters, methods, and usage patterns.

For additional examples and advanced usage patterns, refer to the existing operation implementations in the `pamola_core.anonymization` package.

---