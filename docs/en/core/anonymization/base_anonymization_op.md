# PAMOLA.CORE Base Anonymization Operation Module Documentation

**Module:** `pamola_core.anonymization.base_anonymization_op`  
**Version:** 2.0.0  
**Status:** Stable  
**Last Updated:** December 16, 2025

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation & Dependencies](#installation--dependencies)
5. [Core Classes](#core-classes)
6. [Usage Guide](#usage-guide)
7. [Creating Custom Operations](#creating-custom-operations)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

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

### AnonymizationOperation

The main base class for all anonymization operations.

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

## Creating Custom Operations

### Step 1: Create Your Operation Class

```python
from typing import Dict, Any
import pandas as pd
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

class CustomAnonymizationOperation(AnonymizationOperation):
    """
    Custom anonymization operation for demonstration.
    """
    
    def __init__(self,
                 field_name: str,
                 custom_param: str = "default",
                 **kwargs):
        """
        Initialize custom operation.
        
        Parameters:
        -----------
        field_name : str
            Field to anonymize
        custom_param : str
            Custom parameter specific to this operation
        **kwargs : dict
            Additional parameters passed to base class
        """
        # Initialize base class
        super().__init__(
            field_name=field_name,
            description=f"Custom anonymization for {field_name}",
            **kwargs
        )
        
        # Store custom parameters
        self.custom_param = custom_param
```

### Step 2: Implement Required Methods

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """
    Process a batch of data.
    
    Parameters:
    -----------
    batch : pd.DataFrame
        Batch to process
        
    Returns:
    --------
    pd.DataFrame
        Processed batch
    """
    # Get the field data
    field_data = batch[self.field_name].copy()
    
    # Apply your anonymization logic
    anonymized_data = field_data.apply(
        lambda x: self.process_value(x, param=self.custom_param)
    )
    
    # Update the batch based on mode
    if self.mode == "REPLACE":
        batch[self.field_name] = anonymized_data
    else:  # ENRICH mode
        output_field = generate_output_field_name(
            self.field_name, self.mode, self.output_field_name,
            operation_suffix="custom", column_prefix=self.column_prefix
        )
        batch[output_field] = anonymized_data
    
    return batch

def process_value(self, value, **params):
    """
    Process a single value.
    
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
    if pd.isna(value):
        # Handle null based on strategy
        if self.null_strategy == "PRESERVE":
            return value
        elif self.null_strategy == "ERROR":
            raise ValueError(f"Null value found in {self.field_name}")
        # Add your null handling logic
    
    # Apply your anonymization logic here
    custom_param = params.get('param', self.custom_param)
    
    # Example: simple transformation
    if isinstance(value, (int, float)):
        return value * 2  # Your logic here
    else:
        return f"{value}_{custom_param}"  # Your logic here

def _collect_specific_metrics(self,
                              original_data: pd.Series,
                              anonymized_data: pd.Series) -> Dict[str, Any]:
    """
    Collect operation-specific metrics.
    
    Parameters:
    -----------
    original_data : pd.Series
        Original data
    anonymized_data : pd.Series
        Anonymized data
        
    Returns:
    --------
    Dict[str, Any]
        Custom metrics
    """
    metrics = {}
    
    # Add your custom metrics
    metrics["custom_metric"] = len(anonymized_data.unique())
    metrics["transformation_rate"] = (
        (original_data != anonymized_data).sum() / len(original_data)
    )
    
    return metrics
```

### Step 3: Add Cache Parameters (Optional)

```python
def _get_cache_parameters(self) -> Dict[str, Any]:
    """
    Get operation-specific parameters for cache key generation.
    
    Returns:
    --------
    Dict[str, Any]
        Parameters for cache key
    """
    # Get base parameters
    params = super()._get_cache_parameters()
    
    # Add your custom parameters
    params["custom_param"] = self.custom_param
    
    return params
```

## Examples

### Example 1: Simple Masking Operation

```python
class SimpleMaskingOperation(AnonymizationOperation):
    """Mask values with a fixed character."""
    
    def __init__(self, field_name: str, mask_char: str = "*", **kwargs):
        super().__init__(
            field_name=field_name,
            description=f"Mask {field_name} with '{mask_char}'",
            **kwargs
        )
        self.mask_char = mask_char
    
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "REPLACE":
            batch[self.field_name] = batch[self.field_name].apply(
                lambda x: self.process_value(x)
            )
        else:
            output_field = generate_output_field_name(
                self.field_name, self.mode, self.output_field_name,
                operation_suffix="masked", column_prefix=self.column_prefix
            )
            batch[output_field] = batch[self.field_name].apply(
                lambda x: self.process_value(x)
            )
        return batch
    
    def process_value(self, value, **params):
        if pd.isna(value) and self.null_strategy == "PRESERVE":
            return value
        return self.mask_char * len(str(value))
    
    def _collect_specific_metrics(self, original_data, anonymized_data):
        return {
            "masked_count": (anonymized_data != original_data).sum(),
            "mask_character": self.mask_char
        }
```

### Example 2: Conditional Generalization

```python
class ConditionalGeneralizationOperation(AnonymizationOperation):
    """Generalize values based on conditions."""
    
    def __init__(self, field_name: str, bin_size: int = 10, **kwargs):
        super().__init__(
            field_name=field_name,
            description=f"Conditionally generalize {field_name}",
            **kwargs
        )
        self.bin_size = bin_size
    
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Process only if record meets conditions
        for idx, row in batch.iterrows():
            if self._should_process_record(row):
                value = row[self.field_name]
                new_value = self.process_value(value)
                
                if self.mode == "REPLACE":
                    batch.at[idx, self.field_name] = new_value
                else:
                    output_field = generate_output_field_name(
                        self.field_name, self.mode, self.output_field_name,
                        operation_suffix="generalized", column_prefix=self.column_prefix
                    )
                    batch.at[idx, output_field] = new_value
        
        return batch
    
    def process_value(self, value, **params):
        if pd.isna(value) and self.null_strategy == "PRESERVE":
            return value
        
        if isinstance(value, (int, float)):
            # Round to nearest bin
            return int(value // self.bin_size) * self.bin_size
        
        return value
    
    def _collect_specific_metrics(self, original_data, anonymized_data):
        return {
            "bin_size": self.bin_size,
            "generalized_values": len(anonymized_data.unique())
        }
```

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

## Summary

The `AnonymizationOperation` base class provides a robust foundation for implementing privacy-preserving data transformations. By inheriting from this class and implementing the required methods, you can create custom anonymization operations that:

- Integrate seamlessly with the PAMOLA.CORE framework
- Handle large datasets efficiently
- Support conditional processing
- Generate comprehensive metrics and visualizations
- Provide consistent error handling and logging

For additional examples and advanced usage patterns, refer to the existing operation implementations in the `pamola_core.anonymization` package.