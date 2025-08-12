Looking at the `metric_utils.md` documentation, I'll update it to reflect the new version 2.1.0 with the added categorical functions. Here's the updated documentation:

```markdown
# PAMOLA.CORE Anonymization Metric Utilities Documentation

## Module Overview

**Module:** `pamola_core.anonymization.commons.metric_utils`  
**Version:** 2.1.0  
**Status:** Stable  
**License:** BSD 3-Clause

### Purpose

The `metric_utils.py` module provides lightweight metric utilities specifically designed for anonymization operations within the PAMOLA.CORE framework. It focuses on process-oriented metrics for monitoring and guiding anonymization operations in real-time, particularly for generalization, masking, and suppression techniques.

### Key Features

- **Process-Oriented Metrics**: Fast calculations suitable for batch processing
- **Anonymization Effectiveness**: Measures how well anonymization reduces uniqueness
- **Strategy-Specific Metrics**: Tailored metrics for different anonymization techniques
- **Categorical Metrics**: Information loss and generalization height for categorical data
- **Performance Tracking**: Simple timing and throughput measurements
- **Lightweight Design**: No complex statistics or heavy computations
- **Integration Ready**: Works seamlessly with DataWriter and ProgressTracker

### Design Principles

- **Speed**: Fast calculations suitable for real-time batch processing
- **Simplicity**: Basic metrics only, no complex statistical analysis
- **Focus**: Process monitoring, not final quality assessment
- **Integration**: Direct support for PAMOLA.CORE framework components

## Architecture Integration

The module sits within the anonymization commons layer and integrates with:

```
pamola_core/
├── anonymization/
│   ├── commons/
│   │   ├── metric_utils.py          # This module
│   │   ├── privacy_metric_utils.py  # Privacy-specific metrics
│   │   └── visualization_utils.py   # Visualization helpers
│   └── [operations]/
└── utils/
    ├── io.py                        # File I/O utilities
    └── ops/
        └── op_data_writer.py        # Artifact writing
```

### Module Dependencies

- `numpy` - Basic numerical operations
- `pandas` - Data manipulation
- `pamola_core.utils.io` - File writing operations
- `pamola_core.utils.ops.op_data_writer` - Standardized artifact management

## Core Functions

### 1. calculate_anonymization_effectiveness

```python
def calculate_anonymization_effectiveness(
    original_series: pd.Series,
    anonymized_series: pd.Series
) -> Dict[str, float]
```

Calculate basic effectiveness metrics for anonymization without heavy statistical analysis.

**Parameters:**
- `original_series` (pd.Series): Original data before anonymization
- `anonymized_series` (pd.Series): Data after anonymization

**Returns:**
- Dict[str, float]: Dictionary containing:
  - `total_records`: Total number of records
  - `original_unique`: Unique values in original data
  - `anonymized_unique`: Unique values in anonymized data
  - `effectiveness_ratio`: Reduction in uniqueness (0.0 to 1.0)
  - `null_increase`: Increase in null values ratio

**Example:**
```python
import pandas as pd
from pamola_core.anonymization.commons.metric_utils import calculate_anonymization_effectiveness

original = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
anonymized = pd.Series([1, 1, 3, 3, 5, 5, 7, 7, 9, 9])

metrics = calculate_anonymization_effectiveness(original, anonymized)
print(f"Effectiveness: {metrics['effectiveness_ratio']:.2%}")
# Output: Effectiveness: 55.56%
```

### 2. calculate_generalization_metrics

```python
def calculate_generalization_metrics(
    original_series: pd.Series,
    anonymized_series: pd.Series,
    strategy: str,
    strategy_params: Dict[str, Any]
) -> Dict[str, Any]
```

Calculate metrics specific to generalization strategies.

**Parameters:**
- `original_series` (pd.Series): Original data
- `anonymized_series` (pd.Series): Generalized data
- `strategy` (str): Generalization strategy used ("binning", "rounding", "range")
- `strategy_params` (Dict[str, Any]): Parameters used for the strategy

**Returns:**
- Dict[str, Any]: Strategy-specific metrics including:
  - `strategy`: Strategy name
  - `parameters`: Strategy parameters
  - `reduction_ratio`: Effectiveness ratio
  - Strategy-specific metrics (e.g., `bin_count`, `avg_bin_size`, `bin_utilization`)

**Example:**
```python
from pamola_core.anonymization.commons.metric_utils import calculate_generalization_metrics

original = pd.Series(range(100))
anonymized = pd.Series([0]*25 + [25]*25 + [50]*25 + [75]*25)  # 4 bins

metrics = calculate_generalization_metrics(
    original, 
    anonymized,
    strategy="binning",
    strategy_params={"bin_count": 4, "method": "equal_width"}
)

print(f"Bin utilization: {metrics['bin_utilization']:.2%}")
# Output: Bin utilization: 100.00%
```

### 3. calculate_categorical_information_loss

```python
def calculate_categorical_information_loss(
    original_series: pd.Series,
    anonymized_series: pd.Series,
    category_mapping: Optional[Dict[str, str]] = None,
    hierarchy_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]
```

Calculate information loss metrics for categorical generalization. This function measures how much information is lost when categorical values are generalized, providing quick feedback on the trade-off between privacy and utility.

**Parameters:**
- `original_series` (pd.Series): Original categorical data
- `anonymized_series` (pd.Series): Generalized categorical data
- `category_mapping` (Optional[Dict[str, str]]): Mapping from original to generalized categories
- `hierarchy_info` (Optional[Dict[str, Any]]): Information about hierarchy (levels, structure) if available

**Returns:**
- Dict[str, float]: Information loss metrics:
  - `precision_loss`: Loss of granularity (0-1)
  - `entropy_loss`: Normalized entropy reduction (0-1)
  - `distribution_shift`: Change in value distribution (0-1)
  - `category_reduction_ratio`: Ratio of unique values reduced
  - `average_group_size`: Average size of generalized groups

**Example:**
```python
from pamola_core.anonymization.commons.metric_utils import calculate_categorical_information_loss

original = pd.Series(['NYC', 'Boston', 'LA', 'Chicago', 'Miami', 'Seattle'])
anonymized = pd.Series(['East', 'East', 'West', 'Central', 'East', 'West'])

mapping = {
    'NYC': 'East', 'Boston': 'East', 'Miami': 'East',
    'LA': 'West', 'Seattle': 'West',
    'Chicago': 'Central'
}

metrics = calculate_categorical_information_loss(original, anonymized, mapping)
print(f"Precision loss: {metrics['precision_loss']:.2%}")
print(f"Entropy loss: {metrics['entropy_loss']:.2%}")
# Output: Precision loss: 50.00%
# Output: Entropy loss: 48.46%
```

### 4. calculate_generalization_height

```python
def calculate_generalization_height(
    original_values: Union[pd.Series, List[str]],
    generalized_values: Union[pd.Series, List[str]],
    hierarchy_dict: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]
```

Calculate the generalization height in a hierarchy. This function measures how many levels up in the hierarchy the values have been generalized.

**Parameters:**
- `original_values` (Union[pd.Series, List[str]]): Original values before generalization
- `generalized_values` (Union[pd.Series, List[str]]): Values after generalization
- `hierarchy_dict` (Optional[Dict[str, Dict[str, Any]]]): Hierarchy information with structure:
  ```python
  {
      "value": {
          "parent": "parent_value",
          "level": 0,  # Leaf level
          "root": "root_value"
      }
  }
  ```

**Returns:**
- Dict[str, Any]: Generalization height metrics:
  - `min_height`: Minimum generalization height
  - `max_height`: Maximum generalization height
  - `avg_height`: Average generalization height
  - `height_distribution`: Distribution of heights
  - `uniform_generalization`: Whether all values generalized equally
  - `total_generalized`: Number of values that were generalized
  - `total_unchanged`: Number of values that remained unchanged

**Example:**
```python
from pamola_core.anonymization.commons.metric_utils import calculate_generalization_height

original = ['NYC', 'Boston', 'Brooklyn', 'Queens']
generalized = ['New York', 'Massachusetts', 'New York', 'New York']

hierarchy = {
    'NYC': {'parent': 'New York', 'level': 0},
    'Brooklyn': {'parent': 'New York', 'level': 0},
    'Queens': {'parent': 'New York', 'level': 0},
    'Boston': {'parent': 'Massachusetts', 'level': 0},
    'New York': {'parent': 'USA', 'level': 1},
    'Massachusetts': {'parent': 'USA', 'level': 1}
}

metrics = calculate_generalization_height(original, generalized, hierarchy)
print(f"Average height: {metrics['avg_height']}")
print(f"Uniform generalization: {metrics['uniform_generalization']}")
# Output: Average height: 1.0
# Output: Uniform generalization: True
```

### 5. calculate_masking_metrics

```python
def calculate_masking_metrics(
    original_series: pd.Series,
    masked_series: pd.Series,
    mask_char: str = "*"
) -> Dict[str, float]
```

Calculate metrics for masking operations.

**Parameters:**
- `original_series` (pd.Series): Original data
- `masked_series` (pd.Series): Masked data
- `mask_char` (str): Character used for masking (default: "*")

**Returns:**
- Dict[str, float]: Dictionary containing:
  - `total_records`: Total number of records
  - `masked_records`: Number of masked records
  - `masking_ratio`: Proportion of masked records
  - `null_count`: Number of null values

**Example:**
```python
from pamola_core.anonymization.commons.metric_utils import calculate_masking_metrics

original = pd.Series(['john@email.com', 'jane@email.com', 'bob@email.com'])
masked = pd.Series(['j***@email.com', 'j***@email.com', 'b**@email.com'])

metrics = calculate_masking_metrics(original, masked)
print(f"Masking ratio: {metrics['masking_ratio']:.2%}")
# Output: Masking ratio: 100.00%
```

### 6. calculate_suppression_metrics

```python
def calculate_suppression_metrics(
    original_series: pd.Series,
    suppressed_series: pd.Series
) -> Dict[str, float]
```

Calculate metrics for suppression operations.

**Parameters:**
- `original_series` (pd.Series): Original data
- `suppressed_series` (pd.Series): Data after suppression

**Returns:**
- Dict[str, float]: Dictionary containing:
  - `total_records`: Total number of records
  - `suppressed_count`: Number of suppressed records
  - `suppression_ratio`: Proportion of suppressed records
  - `null_suppressions`: New null values
  - `marker_suppressions`: Values replaced with markers

**Example:**
```python
from pamola_core.anonymization.commons.metric_utils import calculate_suppression_metrics

original = pd.Series(['Alice', 'Bob', 'Charlie', 'David'])
suppressed = pd.Series(['Alice', None, 'SUPPRESSED', 'David'])

metrics = calculate_suppression_metrics(original, suppressed)
print(f"Suppression ratio: {metrics['suppression_ratio']:.2%}")
# Output: Suppression ratio: 50.00%
```

### 7. calculate_process_performance

```python
def calculate_process_performance(
    start_time: float,
    end_time: float,
    records_processed: int,
    batch_count: Optional[int] = None
) -> Dict[str, float]
```

Calculate performance metrics for the anonymization process.

**Parameters:**
- `start_time` (float): Process start time (from time.time())
- `end_time` (float): Process end time
- `records_processed` (int): Number of records processed
- `batch_count` (Optional[int]): Number of batches processed

**Returns:**
- Dict[str, float]: Performance metrics including:
  - `duration_seconds`: Total processing time
  - `records_processed`: Number of records
  - `records_per_second`: Processing throughput
  - `batch_count`: Number of batches (if provided)
  - `avg_batch_size`: Average batch size (if batch_count provided)

**Example:**
```python
import time
from pamola_core.anonymization.commons.metric_utils import calculate_process_performance

start = time.time()
# ... processing ...
time.sleep(1)  # Simulate processing
end = time.time()

metrics = calculate_process_performance(start, end, 10000, batch_count=10)
print(f"Speed: {metrics['records_per_second']:.0f} records/second")
# Output: Speed: 10000 records/second
```

### 8. get_value_distribution_summary

```python
def get_value_distribution_summary(
    series: pd.Series,
    max_categories: int = 10
) -> Dict[str, Any]
```

Get a lightweight summary of value distribution for process monitoring.

**Parameters:**
- `series` (pd.Series): Data to summarize
- `max_categories` (int): Maximum categories to include in top values

**Returns:**
- Dict[str, Any]: Distribution summary including:
  - `total_count`: Total number of values
  - `null_count`: Number of null values
  - `unique_count`: Number of unique values
  - `uniqueness_ratio`: Ratio of unique to non-null values
  - `top_values`: Dictionary of top values and their counts (if applicable)

**Example:**
```python
from pamola_core.anonymization.commons.metric_utils import get_value_distribution_summary

data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', None, 'D'])
summary = get_value_distribution_summary(data)
print(f"Unique values: {summary['unique_count']}")
print(f"Top values: {summary['top_values']}")
# Output: Unique values: 4
# Output: Top values: {'A': 3, 'B': 2, 'C': 1, 'D': 1}
```

### 9. collect_operation_metrics

```python
def collect_operation_metrics(
    operation_type: str,
    original_data: pd.Series,
    processed_data: pd.Series,
    operation_params: Dict[str, Any],
    timing_info: Optional[Dict[str, float]] = None
) -> Dict[str, Any]
```

Central function to gather process metrics based on operation type. Now includes automatic collection of categorical metrics when appropriate.

**Parameters:**
- `operation_type` (str): Type of operation ("generalization", "masking", "suppression")
- `original_data` (pd.Series): Original data
- `processed_data` (pd.Series): Processed data
- `operation_params` (Dict[str, Any]): Operation parameters
- `timing_info` (Optional[Dict[str, float]]): Timing information with keys:
  - `start_time`: Operation start time
  - `end_time`: Operation end time
  - `batch_count`: Number of batches (optional)

**Returns:**
- Dict[str, Any]: Comprehensive metrics including:
  - `operation_type`: Type of operation
  - `timestamp`: ISO format timestamp
  - `field_info`: Before/after distribution summaries
  - Operation-specific metrics
  - `categorical_info_loss`: For categorical strategies
  - `generalization_height`: When hierarchy is available
  - Performance metrics (if timing provided)

**Example:**
```python
import time
from pamola_core.anonymization.commons.metric_utils import collect_operation_metrics

original = pd.Series(['NYC', 'Boston', 'LA', 'Chicago'])
processed = pd.Series(['East', 'East', 'West', 'Central'])

hierarchy_dict = {
    'NYC': {'parent': 'East', 'level': 0},
    'Boston': {'parent': 'East', 'level': 0},
    'LA': {'parent': 'West', 'level': 0},
    'Chicago': {'parent': 'Central', 'level': 0}
}

start_time = time.time()
# ... processing ...
end_time = time.time()

metrics = collect_operation_metrics(
    operation_type="generalization",
    original_data=original,
    processed_data=processed,
    operation_params={
        "strategy": "hierarchy",
        "hierarchy_dict": hierarchy_dict
    },
    timing_info={
        "start_time": start_time,
        "end_time": end_time,
        "batch_count": 1
    }
)

print(f"Operation: {metrics['operation_type']}")
print(f"Info loss: {metrics['categorical_info_loss']['precision_loss']:.2%}")
print(f"Avg height: {metrics['generalization_height']['avg_height']}")
```

### 10. save_process_metrics

```python
def save_process_metrics(
    metrics: Dict[str, Any],
    task_dir: Path,
    operation_name: str,
    field_name: str,
    writer: Optional[DataWriter] = None
) -> Optional[Path]
```

Save process metrics to file using PAMOLA.CORE conventions.

**Parameters:**
- `metrics` (Dict[str, Any]): Metrics to save
- `task_dir` (Path): Task directory for output
- `operation_name` (str): Name of the operation
- `field_name` (str): Field being processed
- `writer` (Optional[DataWriter]): DataWriter instance for standardized output

**Returns:**
- Optional[Path]: Path to saved metrics file, or None if failed

**Example:**
```python
from pathlib import Path
from pamola_core.anonymization.commons.metric_utils import save_process_metrics
from pamola_core.utils.ops.op_data_writer import DataWriter

task_dir = Path("/path/to/task")
writer = DataWriter(task_dir)

metrics = {
    "operation_type": "generalization",
    "effectiveness": {"reduction_ratio": 0.75},
    "categorical_info_loss": {"precision_loss": 0.5}
}

saved_path = save_process_metrics(
    metrics=metrics,
    task_dir=task_dir,
    operation_name="categorical_generalization",
    field_name="location",
    writer=writer
)

if saved_path:
    print(f"Metrics saved to: {saved_path}")
```

### 11. get_process_summary_message

```python
def get_process_summary_message(
    metrics: Dict[str, Any]
) -> str
```

Generate a human-readable summary of process metrics for logging. Now includes categorical metrics in the summary.

**Parameters:**
- `metrics` (Dict[str, Any]): Process metrics dictionary

**Returns:**
- str: Formatted summary message

**Example:**
```python
from pamola_core.anonymization.commons.metric_utils import get_process_summary_message

metrics = {
    "operation_type": "generalization",
    "generalization": {
        "strategy": "hierarchy",
        "reduction_ratio": 0.85
    },
    "categorical_info_loss": {
        "precision_loss": 0.45
    },
    "generalization_height": {
        "avg_height": 1.5
    },
    "performance": {
        "records_per_second": 50000
    }
}

summary = get_process_summary_message(metrics)
print(summary)
# Output: Generalization process completed | reduction: 85.0% | precision loss: 45.0% | avg height: 1.5 | speed: 50000 rec/s
```

## Usage in Categorical Anonymization Operations

### Integration Example with Categorical Generalization

```python
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
    calculate_categorical_information_loss,
    calculate_generalization_height,
    save_process_metrics,
    get_process_summary_message
)
import time

class CategoricalGeneralizationOperation(AnonymizationOperation):
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        
        # Get original data
        original_data = batch[self.field_name].copy()
        
        # Apply categorical generalization
        processed_data, mapping = self._apply_categorical_generalization(original_data)
        
        # Update the batch
        batch[self.field_name] = processed_data
        
        # Collect comprehensive metrics
        operation_params = {
            "strategy": self.strategy,
            "category_mapping": mapping,
            "hierarchy_dict": self.hierarchy_dict if hasattr(self, 'hierarchy_dict') else None
        }
        
        metrics = collect_operation_metrics(
            operation_type="generalization",
            original_data=original_data,
            processed_data=processed_data,
            operation_params=operation_params,
            timing_info={
                "start_time": start_time,
                "end_time": time.time(),
                "batch_count": 1
            }
        )
        
        # Calculate additional categorical metrics if not automatically included
        if "categorical_info_loss" not in metrics:
            metrics["categorical_info_loss"] = calculate_categorical_information_loss(
                original_data, processed_data, mapping
            )
        
        # Save metrics
        if self.writer:
            save_process_metrics(
                metrics=metrics,
                task_dir=self.task_dir,
                operation_name=self.name,
                field_name=self.field_name,
                writer=self.writer
            )
        
        # Log summary
        summary = get_process_summary_message(metrics)
        self.logger.info(summary)
        
        return batch
```

## Best Practices

### 1. Metric Collection for Categorical Data

- Always provide category mappings when available for accurate information loss calculation
- Include hierarchy information for proper height calculation
- Consider the trade-off between precision loss and privacy gain

### 2. Performance Considerations

```python
# Good: Use sampling for large categorical datasets
if len(data) > 100000:
    sample_indices = data.sample(n=10000, random_state=42).index
    sample_original = original_data.loc[sample_indices]
    sample_processed = processed_data.loc[sample_indices]
    
    info_loss = calculate_categorical_information_loss(
        sample_original, sample_processed, category_mapping
    )
else:
    info_loss = calculate_categorical_information_loss(
        original_data, processed_data, category_mapping
    )
```

### 3. Hierarchy Integration

```python
# Good: Build hierarchy dict once and reuse
if self.hierarchy_file:
    self.hierarchy_dict = self.load_hierarchy(self.hierarchy_file)
    
# Then use for height calculation
height_metrics = calculate_generalization_height(
    original_values,
    generalized_values,
    self.hierarchy_dict
)
```

### 4. Error Handling for Categorical Metrics

```python
try:
    # Calculate categorical-specific metrics
    info_loss = calculate_categorical_information_loss(
        original, processed, mapping, hierarchy_info
    )
    
    # Validate results
    if info_loss['precision_loss'] > 0.9:
        self.logger.warning("Very high precision loss detected")
        
except Exception as e:
    self.logger.error(f"Failed to calculate categorical metrics: {e}")
    info_loss = {
        "precision_loss": 0.0,
        "entropy_loss": 0.0,
        "error": str(e)
    }
```

## Summary

The `metric_utils.py` module (version 2.1.0) provides essential lightweight metrics for monitoring anonymization processes in real-time. Key enhancements include:

- **Categorical Support**: New functions for measuring information loss and generalization height in categorical data
- **Hierarchy Integration**: Support for hierarchy-based metrics and height calculations
- **Enhanced Collection**: Automatic inclusion of categorical metrics in the main collection function
- **Comprehensive Summaries**: Updated summary messages include categorical-specific metrics

The module continues to focus on:

- **Process Monitoring**: Quick feedback during anonymization
- **Efficiency Metrics**: Measure effectiveness without heavy computation
- **Framework Integration**: Seamless integration with PAMOLA.CORE components
- **Flexibility**: Support for various anonymization strategies including categorical

The module is designed to be fast, simple, and focused on process metrics rather than final quality assessment, making it ideal for use during batch processing of large datasets, including categorical data.
