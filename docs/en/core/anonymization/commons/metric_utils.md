# PAMOLA.CORE Anonymization Metrics Utilities

## Overview

The `metric_utils.py` module provides common metric utilities for anonymization operations, particularly for assessing the effectiveness and utility of various generalization techniques and other privacy-enhancing transformations.

This module is part of the `pamola_core.anonymization.commons` package and serves as a foundation for measuring the impact of anonymization on data quality, statistical properties, and privacy protection levels.

## Key Features

- **Statistical Comparison**: Calculate comprehensive metrics comparing original and anonymized data
- **Strategy-Specific Metrics**: Generate tailored metrics for different generalization strategies (binning, rounding, range)
- **Performance Assessment**: Track execution time and throughput for operation efficiency
- **Distribution Analysis**: Analyze and compare data distributions before and after anonymization
- **Visualization Support**: Generate visualizations for clearer understanding of anonymization effects
- **Standardized Output**: Produce consistent JSON metric outputs with metadata
- **Secure Storage**: Support for encrypted metric persistence
- **Error Resilience**: Robust error handling for fault tolerance

## Architecture

The `metric_utils.py` module sits within the commons package of the anonymization framework:

```
pamola_core/anonymization/
├── commons/                      # Common utilities 
│   ├── metric_utils.py           # This module - metrics calculations
│   ├── processing_utils.py       # Data processing functions
│   ├── validation_utils.py       # Parameter validation
│   └── visualization_utils.py    # Visualization helpers
├── base_anonymization_op.py      # Base class using these metrics
└── ... (other anonymization operations)
```

The module integrates with:
- `base_anonymization_op.py` which calls metric functions during operation execution
- Pamola Core I/O utilities for saving metric results
- Visualization utilities for rendering metrics as plots and charts
- Progress tracking systems for monitoring long-running operations

## Functions

### Basic Metric Calculations

#### `calculate_basic_numeric_metrics`

Calculates fundamental statistical metrics comparing original and anonymized numeric data.

```python
def calculate_basic_numeric_metrics(original_series: pd.Series,
                                    anonymized_series: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic metrics comparing original and anonymized numeric data.

    Parameters:
    -----------
    original_series : pd.Series
        The original numeric data
    anonymized_series : pd.Series
        The anonymized numeric data

    Returns:
    --------
    Dict[str, Any]
        Dictionary with basic metrics
    """
```

**Returns a dictionary containing:**
- `total_records`: Total number of records processed
- `null_count_original`: Number of null values in original data
- `null_count_anonymized`: Number of null values in anonymized data
- `unique_values_original`: Count of unique values in original data
- `unique_values_anonymized`: Count of unique values in anonymized data
- When data remains numeric after anonymization, additional statistics:
  - `mean_original`, `mean_anonymized`: Mean values before and after
  - `std_original`, `std_anonymized`: Standard deviations
  - `min_original`, `min_anonymized`: Minimum values
  - `max_original`, `max_anonymized`: Maximum values
  - `median_original`, `median_anonymized`: Median values
  - `mean_absolute_difference`: Average absolute difference (when possible)
- `generalization_ratio`: Measure of information reduction (1 - unique_after/unique_before)

#### `calculate_generalization_metrics`

Calculates strategy-specific metrics based on the generalization technique used.

```python
def calculate_generalization_metrics(original_series: pd.Series,
                                     anonymized_series: pd.Series,
                                     strategy: str,
                                     strategy_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate generalization-specific metrics based on strategy.

    Parameters:
    -----------
    original_series : pd.Series
        The original data
    anonymized_series : pd.Series
        The anonymized data
    strategy : str
        The generalization strategy used
    strategy_params : Dict[str, Any]
        Parameters used for the strategy

    Returns:
    --------
    Dict[str, Any]
        Dictionary with generalization metrics
    """
```

**Strategy-specific metrics include:**

For **binning** strategy:
- `bin_count`: Number of bins used
- `average_bin_size`: Average number of records per bin
- `bin_distribution`: Distribution of records across bins

For **rounding** strategy:
- `rounding_precision`: Number of decimal places retained (can be negative)
- `estimated_information_loss`: Estimated information reduction based on precision

For **range** strategy:
- `range_min`, `range_max`: Range limits
- `range_size`: Size of the range
- `range_distribution`: Distribution of values by range category

#### `calculate_performance_metrics`

Calculates performance metrics for the anonymization operation.

```python
def calculate_performance_metrics(start_time: float, end_time: float, records_processed: int) -> Dict[str, Any]:
    """
    Calculate performance metrics for the operation.

    Parameters:
    -----------
    start_time : float
        Start time of the operation (from time.time())
    end_time : float
        End time of the operation (from time.time())
    records_processed : int
        Number of records processed

    Returns:
    --------
    Dict[str, Any]
        Dictionary with performance metrics
    """
```

**Performance metrics include:**
- `execution_time_seconds`: Total operation time
- `records_processed`: Number of records processed
- `records_per_second`: Processing throughput rate

### Data Distribution Analysis

#### `get_distribution_data`

Analyzes data distribution for visualization and statistical understanding.

```python
def get_distribution_data(series: pd.Series, bins: int = 10) -> Dict[str, Any]:
    """
    Get distribution data for visualization.

    Parameters:
    -----------
    series : pd.Series
        Data series to analyze
    bins : int, optional
        Number of bins for histogram (default: 10)

    Returns:
    --------
    Dict[str, Any]
        Distribution data for visualization
    """
```

**Returns distribution analysis based on data type:**
- For numeric data: histogram counts, bin edges, and basic statistics
- For non-numeric data: categorical value counts

#### `get_categorical_distribution`

Specialized function for analyzing categorical or string data distributions.

```python
def get_categorical_distribution(series: pd.Series, max_categories: int = 20) -> Dict[str, Any]:
    """
    Get distribution data for categorical or string data.

    Parameters:
    -----------
    series : pd.Series
        Data series to analyze
    max_categories : int, optional
        Maximum number of categories to include (default: 20)

    Returns:
    --------
    Dict[str, Any]
        Distribution data for visualization
    """
```

**Returns categorical distribution data including:**
- `categories`: Dictionary of category values and counts (limited to max_categories)
- `count`: Total number of records
- `unique_count`: Number of unique categories
- `null_count`: Number of null values

### Result Persistence and Visualization

#### `save_metrics_json`

Saves metrics to a JSON file, either using DataWriter or direct file write.

```python
def save_metrics_json(metrics: Dict[str, Any],
                      task_dir: Path,
                      operation_name: str,
                      field_name: str,
                      writer: Optional[DataWriter] = None,
                      progress_tracker: Optional[ProgressTracker] = None,
                      encrypt_output: bool = False) -> Path:
    """
    Save metrics to a JSON file using DataWriter if available, otherwise direct file write.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics to save
    task_dir : Path
        Task directory
    operation_name : str
        Name of the operation
    field_name : str
        Name of the field
    writer : Optional[DataWriter]
        DataWriter instance to use for saving
    progress_tracker : Optional[ProgressTracker]
        Progress tracker for the operation
    encrypt_output : bool, optional
        Whether to encrypt output file (default: False)

    Returns:
    --------
    Path
        Path to the saved metrics file
    """
```

**Features:**
- Adds timestamp and metadata to metrics
- Uses DataWriter if available (preferred method)
- Falls back to direct file write if DataWriter is unavailable
- Supports optional encryption of output
- Returns the path to the saved metrics file

#### `create_distribution_visualization`

Creates visualizations comparing original and anonymized data distributions.

```python
def create_distribution_visualization(original_data: pd.Series,
                                      anonymized_data: pd.Series,
                                      task_dir: Path,
                                      field_name: str,
                                      operation_name: str,
                                      writer: Optional[DataWriter] = None,
                                      progress_tracker: Optional[ProgressTracker] = None) -> Union[Dict[str, Path], None]:
    """
    Create distribution visualizations comparing original and anonymized data.

    Parameters:
    -----------
    original_data : pd.Series
        Original data before anonymization
    anonymized_data : pd.Series
        Anonymized data after processing
    task_dir : Path
        Task directory for saving visualizations
    field_name : str
        Name of the field being analyzed
    operation_name : str
        Name of the anonymization operation
    writer : Optional[DataWriter]
        DataWriter instance to use for saving
    progress_tracker : Optional[ProgressTracker]
        Progress tracker for monitoring the operation

    Returns:
    --------
    Union[Dict[str, Path], None]
        Dictionary mapping visualization types to file paths, or None if no visualizations were created
    """
```

**Visualization types:**
- For numeric data: Histogram comparison with KDE
- For categorical data: Bar chart comparison of value distributions

### Utilities

#### `generate_metrics_hash`

Generates a hash of metrics for caching and comparison purposes.

```python
def generate_metrics_hash(metrics: Dict[str, Any]) -> str:
    """
    Generate a hash of the metrics for caching and comparison.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics to hash

    Returns:
    --------
    str
        Hash string of the metrics
    """
```

**Features:**
- Filters out non-serializable values and timestamps
- Provides stable, deterministic hashing
- Used for caching operation results

## Usage Examples

### Basic Metrics Calculation

```python
import pandas as pd
from pamola_core.anonymization.commons.metric_utils import calculate_basic_numeric_metrics

# Original data (precise ages)
original_ages = pd.Series([25, 32, 41, 53, 29, 36, 47, 19, 62, 55])

# Anonymized data (age groups)
anonymized_ages = pd.Series(['20-29', '30-39', '40-49', '50-59', '20-29', 
                             '30-39', '40-49', '10-19', '60-69', '50-59'])

# Calculate metrics
metrics = calculate_basic_numeric_metrics(original_ages, anonymized_ages)

# Print key metrics
print(f"Records: {metrics['total_records']}")
print(f"Unique values (original): {metrics['unique_values_original']}")
print(f"Unique values (anonymized): {metrics['unique_values_anonymized']}")
print(f"Generalization ratio: {metrics['generalization_ratio']:.2f}")
```

### Strategy-Specific Metrics for Binning

```python
from pamola_core.anonymization.commons.metric_utils import calculate_generalization_metrics

# Calculate binning-specific metrics
bin_metrics = calculate_generalization_metrics(
    original_ages, 
    anonymized_ages,
    strategy="binning",
    strategy_params={"bin_count": 6}
)

# Print bin distribution
print("Distribution by bin:")
for bin_range, count in bin_metrics["bin_distribution"].items():
    print(f"{bin_range}: {count} records")
```

### Creating Visualizations

```python
from pathlib import Path
from pamola_core.anonymization.commons.metric_utils import create_distribution_visualization

# Create visualization comparing original and anonymized data
viz_paths = create_distribution_visualization(
    original_data=original_ages,
    anonymized_data=anonymized_ages,
    task_dir=Path("/path/to/task_dir"),
    field_name="age",
    operation_name="binning"
)

if viz_paths:
    print(f"Visualization created at: {viz_paths['category_comparison']}")
```

### Saving Metrics to JSON

```python
from pathlib import Path
from pamola_core.anonymization.commons.metric_utils import save_metrics_json

# Combined metrics from different calculations
all_metrics = {**metrics, **bin_metrics}

# Save to file
metrics_path = save_metrics_json(
    metrics=all_metrics,
    task_dir=Path("/path/to/task_dir"),
    operation_name="age_binning",
    field_name="age"
)

print(f"Metrics saved to: {metrics_path}")
```

## Considerations and Limitations

### Data Type Handling

- For numeric-to-numeric transformations, all statistical metrics are calculated
- For numeric-to-categorical transformations (e.g., binning), statistical comparisons are limited
- For categorical data, distribution analysis focuses on value counts rather than statistical properties
- When anonymized data changes type (e.g., numeric to string bins), the module detects this and adjusts metrics accordingly

### Error Handling

- All functions include comprehensive error handling to prevent operation failure
- If specific calculations fail, functions return minimal metrics with error information
- Errors are logged but typically don't halt operation execution (fault tolerance)

### Performance Considerations

- Distribution analysis for very large datasets may be memory-intensive
- For extremely large datasets, consider:
  - Using a smaller sample for visualization
  - Adjusting bin count for appropriate memory usage
  - Using chunked processing before visualization

### Visualization Dependencies

- Visualization functions rely on the pamola core visualization utilities (`pamola_core.utils.visualization`)
- Requires matplotlib and seaborn for proper rendering
- PNG format is the standard output for all visualizations

## Future Enhancements

- Support for privacy model guarantee metrics (k-anonymity, l-diversity verification)
- Integration with utility metrics based on machine learning outcomes
- Enhanced visualization options with interactive capabilities
- Support for multivariate/multi-field metric analysis
- Implementation of standardized privacy risk scoring
- Distribution comparison using advanced statistical tests