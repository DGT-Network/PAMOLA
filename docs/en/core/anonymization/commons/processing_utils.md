# PAMOLA.CORE Anonymization Processing Utilities

## Overview

The `processing_utils.py` module provides essential data processing utilities for anonymization operations within the PAMOLA.CORE framework. It focuses on efficient handling of large datasets through chunking, parallel processing, and specialized transformations for different types of data generalization.

This module is part of the `pamola_core.anonymization.commons` package and serves as a foundation for implementing privacy-enhancing transformations on tabular data while maintaining computational efficiency.

## Key Features

- **Chunked Data Processing**: Efficiently process large datasets by breaking them into manageable chunks
- **Parallel Processing**: Utilize multiple CPU cores for faster data processing using joblib
- **Generalization Functions**: Implement common data generalization techniques (binning, rounding, range mapping)
- **Null Value Handling**: Multiple strategies for handling missing values in datasets
- **Progress Tracking**: Integrated progress monitoring for long-running operations
- **Error Resilience**: Robust error handling for fault tolerance during processing
- **Output Field Management**: Flexible handling of output field names based on operation mode

## Architecture

The `processing_utils.py` module sits within the commons package of the anonymization framework:

```
pamola_core/anonymization/
├── commons/                      # Common utilities 
│   ├── processing_utils.py       # This module - data processing functions
│   ├── metric_utils.py           # Metrics calculations
│   ├── validation_utils.py       # Parameter validation
│   └── visualization_utils.py    # Visualization helpers
├── base_anonymization_op.py      # Base class using these utilities
└── ... (other anonymization operations)
```

The module integrates with:
- `base_anonymization_op.py` which calls processing functions during operation execution
- Data source and output abstractions
- Progress tracking systems for monitoring long-running operations
- Other commons utilities for a complete anonymization framework

## Functions

### Data Chunking and Processing

#### `process_in_chunks`

Processes a DataFrame in chunks to handle large datasets efficiently.

```python
def process_in_chunks(df: pd.DataFrame,
                      process_function: Callable,
                      batch_size: int = 10000,
                      progress_tracker: Optional[ProgressTracker] = None,
                      **kwargs) -> Union[pd.DataFrame, None, Any]:
    """
    Process a DataFrame in chunks to handle large datasets efficiently.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process
    process_function : Callable
        Function to apply to each chunk, should take a DataFrame chunk as the first argument
    batch_size : int, optional
        Number of rows to process in each chunk (default: 10000)
    progress_tracker : Optional[ProgressTracker]
        Progress tracker for monitoring the operation
    **kwargs : dict
        Additional arguments to pass to the process_function

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame
    """
```

**Key Features:**
- Breaks large datasets into manageable chunks
- Applies the provided function to each chunk
- Combines results back into a single DataFrame
- Tracks progress during processing
- Provides detailed logging of processing steps
- Handles processing errors without failing the entire operation

#### `get_dataframe_chunks`

Generates chunks of a DataFrame for efficient processing of large datasets.

```python
def get_dataframe_chunks(df: pd.DataFrame, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """
    Generate chunks of a DataFrame for efficient processing of large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to chunk
    chunk_size : int, optional
        Number of rows in each chunk (default: 10000)

    Yields:
    -------
    pd.DataFrame
        Chunk of the original DataFrame
    """
```

**Key Features:**
- Creates a generator that yields chunks of the DataFrame
- Efficiently handles memory for very large datasets
- Provides detailed logging about chunk sizes and counts

#### `process_dataframe_parallel`

Processes a DataFrame in parallel using joblib for large datasets.

```python
def process_dataframe_parallel(df: pd.DataFrame,
                               process_function: Callable,
                               n_jobs: int = -1,
                               batch_size: int = 10000,
                               progress_tracker: Optional[ProgressTracker] = None,
                               **kwargs) -> pd.DataFrame:
    """
    Process a DataFrame in parallel using joblib for large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process
    process_function : Callable
        Function to apply to each chunk
    n_jobs : int, optional
        Number of jobs to run in parallel (-1 to use all processors) (default: -1)
    batch_size : int, optional
        Number of rows in each chunk (default: 10000)
    progress_tracker : Optional[ProgressTracker]
        Progress tracker for monitoring the operation
    **kwargs : dict
        Additional arguments to pass to the process_function

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame
    """
```

**Key Features:**
- Utilizes multiple CPU cores for parallel processing
- Significantly improves performance for large datasets
- Falls back to sequential processing if parallel processing fails
- Handles progress tracking across parallel jobs
- Combines results from parallel workers

### Generalization Functions

#### `numeric_generalization_binning`

Generalizes numeric values by binning them into intervals.

```python
def numeric_generalization_binning(series: pd.Series,
                                   bin_count: int,
                                   labels: Optional[List[str]] = None,
                                   handle_nulls: bool = True) -> pd.Series:
    """
    Generalize numeric values by binning them into intervals.

    Parameters:
    -----------
    series : pd.Series
        The numeric series to generalize
    bin_count : int
        Number of bins to create
    labels : List[str], optional
        Custom labels for the bins (default: None, will use interval notation)
    handle_nulls : bool, optional
        Whether to handle null values specially (default: True)

    Returns:
    --------
    pd.Series
        The generalized series
    """
```

**Key Features:**
- Creates bins of equal width across the data range
- Supports custom labels for bins
- Preserves null values in the output if desired
- Handles edge cases like single-valued series
- Returns a categorical series with bin labels

#### `numeric_generalization_rounding`

Generalizes numeric values by rounding to a specified precision.

```python
def numeric_generalization_rounding(series: pd.Series,
                                    precision: int,
                                    handle_nulls: bool = True) -> pd.Series:
    """
    Generalize numeric values by rounding to a specified precision.

    Parameters:
    -----------
    series : pd.Series
        The numeric series to generalize
    precision : int
        Number of decimal places to round to (can be negative for rounding to 10s, 100s, etc.)
    handle_nulls : bool, optional
        Whether to handle null values specially (default: True)

    Returns:
    --------
    pd.Series
        The generalized series
    """
```

**Key Features:**
- Supports both decimal place rounding (positive precision)
- Supports rounding to 10s, 100s, etc. (negative precision)
- Preserves null values in the output if desired
- Maintains the original series index

#### `numeric_generalization_range`

Generalizes numeric values by mapping to custom range intervals.

```python
def numeric_generalization_range(series: pd.Series,
                                 range_limits: Tuple[float, float],
                                 handle_nulls: bool = True) -> pd.Series:
    """
    Generalize numeric values by mapping to custom range intervals.

    Parameters:
    -----------
    series : pd.Series
        The numeric series to generalize
    range_limits : Tuple[float, float]
        The (min, max) range limits
    handle_nulls : bool, optional
        Whether to handle null values specially (default: True)

    Returns:
    --------
    pd.Series
        The generalized series with range labels
    """
```

**Key Features:**
- Maps values to ranges (e.g., "<10", "10-20", ">20")
- Creates three categories: below range, within range, above range
- Preserves null values in the output if desired
- Returns series with text labels for the ranges

### Utility Functions

#### `process_nulls`

Processes null values according to the specified strategy.

```python
def process_nulls(series: pd.Series, null_strategy: str) -> pd.Series:
    """
    Process null values according to the specified strategy.

    Parameters:
    -----------
    series : pd.Series
        The series to process
    null_strategy : str
        Strategy for handling nulls: "PRESERVE", "EXCLUDE", or "ERROR"

    Returns:
    --------
    pd.Series
        The processed series

    Raises:
    -------
    ValueError
        If null_strategy is "ERROR" and nulls are found
    """
```

**Key Features:**
- Supports three strategies for null handling:
  - "PRESERVE": Keep nulls as they are
  - "EXCLUDE": Remove null values
  - "ERROR": Raise an error if nulls are found
- Provides detailed logging of null handling

#### `generate_output_field_name`

Generates the appropriate output field name based on mode and parameters.

```python
def generate_output_field_name(field_name: str, mode: str, output_field_name: Optional[str], column_prefix: str) -> str:
    """
    Generate the appropriate output field name based on mode and parameters.

    Parameters:
    -----------
    field_name : str
        Original field name
    mode : str
        Operation mode ("REPLACE" or "ENRICH")
    output_field_name : str, optional
        User-specified output field name (can be None)
    column_prefix : str
        Prefix to use when creating new field name in ENRICH mode

    Returns:
    --------
    str
        The output field name to use
    """
```

**Key Features:**
- Handles two operation modes:
  - "REPLACE": Use the original field name
  - "ENRICH": Create a new field name
- Supports custom output field names
- Applies a prefix when creating new field names

#### `prepare_output_directory`

Prepares an output directory within the task directory.

```python
def prepare_output_directory(task_dir: Path, subdirectory: str) -> Path:
    """
    Prepare an output directory within the task directory.

    Parameters:
    -----------
    task_dir : Path
        The task directory
    subdirectory : str
        Name of the subdirectory to create

    Returns:
    --------
    Path
        Path to the created subdirectory
    """
```

**Key Features:**
- Creates the specified subdirectory if it doesn't exist
- Ensures parent directories are created as needed
- Returns the path to the created directory

## Usage Examples

### Processing Large DataFrames in Chunks

```python
import pandas as pd
from pamola_core.anonymization.commons.processing_utils import process_in_chunks

# Create a large example DataFrame
large_df = pd.DataFrame({
    'id': range(1, 100001),
    'value': range(1, 100001)
})

# Define a processing function
def double_values(chunk_df):
    result = chunk_df.copy()
    result['value'] = result['value'] * 2
    return result

# Process in chunks of 10,000 rows
processed_df = process_in_chunks(
    df=large_df,
    process_function=double_values,
    batch_size=10000
)

print(f"Processed {len(processed_df)} rows")
print(f"Sample result: {processed_df.iloc[0:5]}")
```

### Parallel Processing for Performance

```python
from pamola_core.anonymization.commons.processing_utils import process_dataframe_parallel

# Process a large DataFrame in parallel
processed_df = process_dataframe_parallel(
    df=large_df,
    process_function=double_values,
    n_jobs=4,  # Use 4 parallel workers
    batch_size=10000
)

print(f"Processed {len(processed_df)} rows in parallel")
```

### Applying Generalization to Numeric Data

```python
import pandas as pd
from pamola_core.anonymization.commons.processing_utils import numeric_generalization_binning

# Create sample age data
ages = pd.Series([18, 25, 32, 41, 53, 29, 36, 47, 19, 62, 55, None])

# Apply binning generalization (into 5 bins)
binned_ages = numeric_generalization_binning(
    series=ages,
    bin_count=5,
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)

print("Original ages:")
print(ages)
print("\nBinned ages:")
print(binned_ages)
```

### Handling Different Null Strategies

```python
from pamola_core.anonymization.commons.processing_utils import process_nulls

# Sample data with nulls
data_with_nulls = pd.Series([1, 2, None, 4, None, 6])

# Try different null strategies
preserved = process_nulls(data_with_nulls, "PRESERVE")
excluded = process_nulls(data_with_nulls, "EXCLUDE")

print("Original data:")
print(data_with_nulls)
print("\nPreserved nulls:")
print(preserved)
print("\nExcluded nulls:")
print(excluded)

# This would raise an error:
# process_nulls(data_with_nulls, "ERROR")
```

### Creating Output Field Names

```python
from pamola_core.anonymization.commons.processing_utils import generate_output_field_name

# Original field name
field = "customer_income"

# Create output field name based on mode
replace_mode = generate_output_field_name(
    field_name=field,
    mode="REPLACE",
    output_field_name=None,
    column_prefix="anon_"
)

enrich_mode = generate_output_field_name(
    field_name=field,
    mode="ENRICH",
    output_field_name=None,
    column_prefix="anon_"
)

custom_name = generate_output_field_name(
    field_name=field,
    mode="ENRICH",
    output_field_name="income_anonymized",
    column_prefix="anon_"
)

print(f"Replace mode: {replace_mode}")
print(f"Enrich mode: {enrich_mode}")
print(f"Custom name: {custom_name}")
```

## Considerations and Limitations

### Memory Management

- For very large datasets (>10M rows), adjust batch size to balance memory usage and performance
- When using parallel processing, monitor system memory usage as multiple workers consume more RAM
- Consider using Dask for datasets that exceed available memory

### Performance Considerations

- `process_dataframe_parallel` provides best performance for CPU-bound operations
- Optimal batch size depends on data characteristics and available memory
- Small batch sizes increase overhead but use less memory
- Large batch sizes improve performance but require more memory

### Error Handling

- All functions include comprehensive error handling to prevent operation failure
- Chunk processing continues even if some chunks fail (fault tolerance)
- Errors are logged but typically don't halt operation execution
- Partial results are returned if some chunks were successfully processed

### Generalization Limitations

- Binning works best with well-distributed data; skewed data may produce uneven bins
- Rounding with high precision may not provide sufficient anonymization
- Range generalization only supports a single range; more complex ranges require custom processing

## Future Enhancements

- Support for more advanced generalization techniques (e.g., k-anonymity-aware binning)
- Integration with differential privacy mechanisms
- Adaptive chunk size based on available system resources
- Support for distributed processing across multiple nodes
- Specialized handling for common data types (e.g., geospatial, categorical)
- Memory-mapped file support for extremely large datasets