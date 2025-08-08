# PAMOLA.CORE Privacy Process Metrics Documentation

## Module Overview

**Module:** `pamola_core.anonymization.commons.privacy_metric_utils.py`  
**Package:** `pamola_core.anonymization.commons`  
**Version:** 2.0.0  
**Status:** Stable  
**License:** BSD 3-Clause

## Purpose

The Privacy Process Metrics module provides lightweight privacy metrics for monitoring and controlling anonymization processes in real-time. It focuses on fast, simple indicators that help track the progress and effectiveness of anonymization operations without heavy computational overhead.

This module serves as a real-time monitoring tool for anonymization operations, providing quick feedback on privacy levels, coverage, and basic risk indicators during the anonymization process. It is NOT intended for final quality assessment or detailed risk analysis.

## Key Features

- **Fast Coverage and Suppression Metrics** - Real-time tracking of anonymization coverage
- **Quick K-Anonymity Checks** - Minimum k-value calculation only
- **Simple Generalization Level Indicators** - Basic metrics for generalization effectiveness
- **Basic Group Size Distribution** - Quick analysis of equivalence class sizes
- **Threshold Monitoring** - Real-time process control based on thresholds
- **Lightweight Risk Indicators** - Simple disclosure risk calculation

## Design Principles

1. **Performance** - Each function completes in <100ms on 100K records
2. **Simplicity** - No complex statistics or ML algorithms
3. **Focus** - Process monitoring only, not final assessment
4. **Independence** - Minimal dependencies on other modules

## Architecture Integration

The module sits within the anonymization commons layer, providing real-time metrics for operations:

```
pamola_core/
├── anonymization/
│   ├── commons/
│   │   ├── privacy_metric_utils.py  # This module
│   │   ├── metric_utils.py          # General metrics
│   │   ├── validation_utils.py      # Validation
│   │   └── visualization_utils.py   # Visualization
│   └── operations/
│       └── [Various anonymization operations use these metrics]
```

## Core Functions

### Coverage Metrics

#### `calculate_anonymization_coverage(original, anonymized)`

Calculates the coverage of the anonymization process.

**Parameters:**
- `original` (pd.Series): Original data before anonymization
- `anonymized` (pd.Series): Data after anonymization

**Returns:**
- `Dict[str, float]`: Coverage metrics including:
  - `total_coverage`: Percentage of non-null anonymized values
  - `changed_ratio`: Percentage of values that were modified
  - `suppressed_ratio`: Percentage of values set to null
  - `unchanged_ratio`: Percentage of values left unchanged

**Example:**
```python
import pandas as pd
from pamola_core.anonymization.commons.privacy_metric_utils import calculate_anonymization_coverage

original = pd.Series([100, 200, 300, 400, 500])
anonymized = pd.Series([100, None, 300, 450, 500])

coverage = calculate_anonymization_coverage(original, anonymized)
print(f"Coverage: {coverage['total_coverage']:.2%}")
print(f"Changed: {coverage['changed_ratio']:.2%}")
print(f"Suppressed: {coverage['suppressed_ratio']:.2%}")
```

#### `calculate_suppression_rate(series, original_nulls=None)`

Calculates the suppression rate in anonymized data.

**Parameters:**
- `series` (pd.Series): Anonymized data series
- `original_nulls` (Optional[int]): Number of nulls in original data

**Returns:**
- `float`: Suppression rate [0.0, 1.0]

### Group Metrics

#### `get_group_size_distribution(df, quasi_identifiers, max_groups=100)`

Gets quick distribution of group sizes for quasi-identifiers.

**Parameters:**
- `df` (pd.DataFrame): Data to analyze
- `quasi_identifiers` (List[str]): List of quasi-identifier columns
- `max_groups` (int): Maximum number of groups to analyze (default: 100)

**Returns:**
- `Dict[str, Any]`: Distribution info including:
  - `total_groups`: Total number of equivalence classes
  - `size_distribution`: Dictionary of group sizes and counts
  - `min_size`: Minimum group size
  - `max_size`: Maximum group size
  - `mean_size`: Average group size

**Example:**
```python
df = pd.DataFrame({
    'age_range': ['20-30', '20-30', '30-40', '30-40', '40-50'],
    'city': ['NYC', 'NYC', 'LA', 'LA', 'Chicago'],
    'salary': [50000, 52000, 60000, 62000, 55000]
})

dist = get_group_size_distribution(df, ['age_range', 'city'])
print(f"Total groups: {dist['total_groups']}")
print(f"Min group size: {dist['min_size']}")
```

#### `calculate_min_group_size(df, quasi_identifiers, sample_size=10000)`

Calculates minimum group size (k) for quasi-identifiers.

**Parameters:**
- `df` (pd.DataFrame): Data to analyze
- `quasi_identifiers` (List[str]): List of quasi-identifier columns
- `sample_size` (Optional[int]): Sample size for large datasets

**Returns:**
- `int`: Minimum group size (k value)

### Risk Metrics

#### `calculate_vulnerable_records_ratio(df, quasi_identifiers, k_threshold=5, sample_size=10000)`

Calculates ratio of vulnerable records (k < threshold).

**Parameters:**
- `df` (pd.DataFrame): Data to analyze
- `quasi_identifiers` (List[str]): List of quasi-identifier columns
- `k_threshold` (int): Minimum acceptable group size (default: 5)
- `sample_size` (Optional[int]): Sample size for large datasets

**Returns:**
- `float`: Ratio of vulnerable records [0.0, 1.0]

**Example:**
```python
vulnerable_ratio = calculate_vulnerable_records_ratio(
    df, 
    ['age_range', 'city'], 
    k_threshold=3
)
print(f"Vulnerable records: {vulnerable_ratio:.2%}")
```

### Generalization Metrics

#### `calculate_generalization_level(original, generalized)`

Calculates the level of generalization applied.

**Parameters:**
- `original` (pd.Series): Original data
- `generalized` (pd.Series): Generalized data

**Returns:**
- `float`: Generalization level [0.0, 1.0] where 1.0 = maximum generalization

#### `calculate_value_reduction_ratio(original, anonymized)`

Calculates the ratio of unique value reduction.

**Parameters:**
- `original` (pd.Series): Original data
- `anonymized` (pd.Series): Anonymized data

**Returns:**
- `float`: Value reduction ratio [0.0, 1.0]

### Process Control

#### `check_anonymization_thresholds(metrics, thresholds=None)`

Checks if anonymization metrics meet specified thresholds.

**Parameters:**
- `metrics` (Dict[str, float]): Current process metrics
- `thresholds` (Optional[Dict[str, float]]): Target thresholds

**Default Thresholds:**
- `min_k`: 5 (minimum k-anonymity)
- `max_suppression`: 0.2 (maximum 20% suppression)
- `min_coverage`: 0.95 (minimum 95% coverage)
- `max_vulnerable_ratio`: 0.05 (maximum 5% vulnerable)

**Returns:**
- `Dict[str, bool]`: Pass/fail status for each threshold

**Example:**
```python
metrics = {
    'min_k': 4,
    'suppression_rate': 0.15,
    'total_coverage': 0.92,
    'vulnerable_ratio': 0.08
}

results = check_anonymization_thresholds(metrics)
print(f"All thresholds met: {results['all_thresholds_met']}")
```

### Batch Processing

#### `calculate_batch_metrics(original_batch, anonymized_batch, field_name, quasi_identifiers=None)`

Calculates all process metrics for a batch in one call.

**Parameters:**
- `original_batch` (pd.DataFrame): Original data batch
- `anonymized_batch` (pd.DataFrame): Anonymized data batch
- `field_name` (str): Primary field being anonymized
- `quasi_identifiers` (Optional[List[str]]): List of quasi-identifier columns

**Returns:**
- `Dict[str, Any]`: Complete set of process metrics

**Example:**
```python
# During batch processing in an anonymization operation
original_batch = df[['salary', 'age', 'city']].copy()
anonymized_batch = anonymize_batch(original_batch)

metrics = calculate_batch_metrics(
    original_batch,
    anonymized_batch,
    field_name='salary',
    quasi_identifiers=['age', 'city']
)

# Get human-readable summary
summary = get_process_summary(metrics)
print(summary['status'])
```

## Usage in Anonymization Operations

The module is designed to be used within anonymization operations for real-time monitoring:

```python
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_batch_metrics,
    check_anonymization_thresholds,
    get_process_summary
)

class MyAnonymizationOperation(AnonymizationOperation):
    def process_batch(self, batch):
        # Store original for metrics
        original_batch = batch.copy()
        
        # Apply anonymization
        anonymized_batch = self._apply_anonymization(batch)
        
        # Calculate metrics
        metrics = calculate_batch_metrics(
            original_batch,
            anonymized_batch,
            self.field_name,
            self.quasi_identifiers
        )
        
        # Check thresholds
        threshold_check = check_anonymization_thresholds(metrics)
        if not threshold_check['all_thresholds_met']:
            self.logger.warning(get_process_summary(metrics)['status'])
            
        return anonymized_batch
```

## Constants

The module defines several constants for process monitoring:

- `DEFAULT_K_THRESHOLD = 5`: Default k-anonymity threshold
- `DEFAULT_SUPPRESSION_WARNING = 0.2`: Warning threshold for suppression (20%)
- `DEFAULT_COVERAGE_TARGET = 0.95`: Target coverage (95%)
- `EPSILON = 1e-10`: Small constant to avoid division by zero

## Performance Considerations

1. **Sampling**: Large datasets are automatically sampled for performance
2. **Limited Groups**: Group analysis is limited to top groups only
3. **Simple Calculations**: All metrics use basic arithmetic operations
4. **No ML Dependencies**: No machine learning or complex statistics

## Integration with Other Modules

- **metric_utils.py**: For more comprehensive post-processing metrics
- **validation_utils.py**: For input validation before processing
- **visualization_utils.py**: For visualizing the calculated metrics
- **base_anonymization_op.py**: Used by all anonymization operations

## Best Practices

1. **Real-time Monitoring**: Use during batch processing, not for final assessment
2. **Threshold Setting**: Adjust thresholds based on privacy requirements
3. **Sampling**: Use appropriate sample sizes for large datasets
4. **Error Handling**: All functions handle errors gracefully and log issues
5. **Performance**: Functions are optimized for speed over accuracy

## Version History

- **2.0.0** (Current): Complete rewrite focusing on lightweight process metrics
- **1.0.0**: Initial implementation (deprecated - too heavy)