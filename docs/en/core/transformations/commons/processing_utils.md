# PAMOLA Core: `processing_utils.py` Module

---

## Overview

The `processing_utils.py` module is a core component of the PAMOLA Core framework, providing robust, scalable, and privacy-preserving utilities for processing pandas DataFrames. It is designed to support both single-machine and distributed workflows, enabling efficient data transformation, aggregation, and merging operations for large-scale AI and data privacy pipelines.

This module is intended for use by developers building data processing pipelines, privacy-preserving analytics, and custom transformation tasks within the PAMOLA ecosystem. It is also suitable for advanced users who require fine-grained control over DataFrame operations, chunking, parallelization, and progress tracking.

---

## Key Features

- **Parallel DataFrame Processing**: Efficiently process large DataFrames in parallel using `joblib`.
- **Chunk-wise Processing**: Handle memory constraints by processing data in manageable chunks.
- **Progress Tracking**: Integrate with hierarchical progress trackers for monitoring long-running operations.
- **Flexible Merging**: Merge DataFrames with robust error handling and optional Dask support for distributed execution.
- **Advanced Aggregation**: Group and aggregate data with support for custom and Dask-compatible functions.
- **Field-based Splitting**: Split DataFrames into logical groups based on field definitions.
- **Fault Tolerance**: Continue processing even if some chunks fail, with detailed logging.

---

## Dependencies

### Standard Library
- `logging`
- `time`
- `typing`

### Third-Party Libraries
- `pandas`
- `joblib`
- `dask` (optional, for distributed processing)

### Internal Modules
- `pamola_core.transformations.commons.aggregation_utils`
- `pamola_core.transformations.commons.validation_utils`
- `pamola_core.utils.progress`

---

## Exception Classes

> **Note:** This module does not define custom exception classes directly. Instead, it raises standard exceptions (e.g., `ValueError`, `ImportError`, and generic `Exception`) with detailed logging. Exception handling is designed to be robust and informative for pipeline developers.

### Example: Handling Merge Errors

```python
try:
    merged_df = merge_dataframes(left_df, right_df, left_key="id")
except Exception as e:
    # Handle merge failure (e.g., log, fallback, alert)
    logger.error(f"Merge failed: {e}")
```

#### When Exceptions Are Raised
- **ValueError**: Raised when invalid fields are provided for splitting or aggregation.
- **ImportError**: Raised if Dask is requested but not installed.
- **Exception**: Raised for general errors during merging, aggregation, or chunk processing.

---

## Main Functions and Classes

### 1. `process_in_chunks`

```python
def process_in_chunks(
    df: pd.DataFrame,
    process_function: Callable,
    batch_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> Union[pd.DataFrame, None, Any]:
```

**Parameters:**
- `df`: DataFrame to process
- `process_function`: Function to apply to each chunk
- `batch_size`: Number of rows per chunk
- `progress_tracker`: Optional progress tracker
- `**kwargs`: Additional arguments for the process function

**Returns:** Processed DataFrame (or partial result on error)

**Raises:** Logs errors for failed chunks, continues processing

---

### 2. `_get_dataframe_chunks`

```python
def _get_dataframe_chunks(
    df: pd.DataFrame, chunk_size: int = 10000
) -> Iterator[pd.DataFrame]:
```

**Parameters:**
- `df`: DataFrame to chunk
- `chunk_size`: Number of rows per chunk

**Yields:** DataFrame chunks

---

### 3. `process_dataframe_parallel`

```python
def process_dataframe_parallel(
    df: pd.DataFrame,
    process_function: Callable,
    n_jobs: int = -1,
    batch_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> pd.DataFrame:
```

**Parameters:**
- `df`: DataFrame to process
- `process_function`: Function to apply to each chunk
- `n_jobs`: Number of parallel jobs
- `batch_size`: Rows per chunk
- `progress_tracker`: Optional progress tracker
- `**kwargs`: Additional arguments

**Returns:** Processed DataFrame

**Raises:** Logs errors, falls back to sequential processing on failure

---

### 4. `split_dataframe`

```python
def split_dataframe(
    df: pd.DataFrame,
    field_groups: Dict[str, List[str]],
    id_field: str,
    include_id_field: bool = True,
) -> Dict[str, pd.DataFrame]:
```

**Parameters:**
- `df`: Source DataFrame
- `field_groups`: Mapping of group names to field lists
- `id_field`: Identifier field
- `include_id_field`: Whether to include the ID in each group

**Returns:** Dictionary of group name to DataFrame

**Raises:**
- `ValueError`: If fields are invalid or missing

---

### 5. `merge_dataframes`

```python
def merge_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: Optional[str] = None,
    join_type: str = "left",
    suffixes: Tuple[str, str] = ("_x", "_y"),
    left_index: bool = False,
    right_index: bool = False,
    use_dask: bool = False,
    npartitions: Optional[int] = None,
) -> pd.DataFrame:
```

**Parameters:**
- `left_df`: Left DataFrame
- `right_df`: Right DataFrame
- `left_key`: Join key for left DataFrame
- `right_key`: Join key for right DataFrame (optional)
- `join_type`: Type of join ("left", "right", "outer", "inner")
- `suffixes`: Suffixes for overlapping columns
- `left_index`: Use left index as join key
- `right_index`: Use right index as join key
- `use_dask`: Use Dask for distributed merge
- `npartitions`: Number of Dask partitions

**Returns:** Merged DataFrame

**Raises:**
- `ImportError`: If Dask is not installed when requested
- `Exception`: For merge failures

---

### 6. `aggregate_dataframe`

```python
def aggregate_dataframe(
    df: pd.DataFrame,
    group_by_fields: List[str],
    aggregations: Optional[Dict[str, List[str]]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
    use_dask: bool = False,
    npartitions: Optional[int] = None,
) -> pd.DataFrame:
```

**Parameters:**
- `df`: Input DataFrame
- `group_by_fields`: Fields to group by
- `aggregations`: Standard aggregation functions
- `custom_aggregations`: Custom aggregation functions
- `use_dask`: Use Dask for distributed aggregation
- `npartitions`: Number of Dask partitions

**Returns:** Aggregated DataFrame

**Raises:**
- `ImportError`: If Dask is not installed when requested
- `Exception`: For aggregation failures

---

### 7. `_determine_partitions`

```python
def _determine_partitions(df: pd.DataFrame, npartitions: Optional[int] = None) -> int:
```

**Parameters:**
- `df`: Input DataFrame
- `npartitions`: Desired number of partitions

**Returns:** Number of partitions

---

## Dependency Resolution and Completion Validation

- **Validation**: All functions validate input DataFrames and parameters before processing. For example, `validate_dataframe` ensures required columns exist, and `validate_join_type` checks join type validity.
- **Completion**: Progress trackers can be used to monitor and validate the completion of long-running operations, especially in chunked or parallel processing.

---

## Usage Examples

### Processing a DataFrame in Chunks

```python
# Define a processing function for each chunk
def clean_chunk(chunk):
    # Example: Remove rows with missing values
    return chunk.dropna()

# Process a large DataFrame in chunks
processed_df = process_in_chunks(df, clean_chunk, batch_size=5000)
```

### Parallel Processing with Progress Tracking

```python
from pamola_core.utils.progress import HierarchicalProgressTracker

progress = HierarchicalProgressTracker("Parallel Processing")
processed_df = process_dataframe_parallel(
    df,
    process_function=clean_chunk,
    n_jobs=4,
    batch_size=10000,
    progress_tracker=progress
)
```

### Merging DataFrames with Dask

```python
try:
    merged = merge_dataframes(
        left_df, right_df, left_key="user_id", use_dask=True, npartitions=8
    )
except ImportError:
    # Fallback to pandas merge if Dask is unavailable
    merged = merge_dataframes(left_df, right_df, left_key="user_id")
```

### Aggregating with Custom Functions

```python
def custom_sum(series):
    return series.sum() + 1  # Example custom logic

agg_df = aggregate_dataframe(
    df,
    group_by_fields=["group_id"],
    aggregations={"value": ["mean", "max"]},
    custom_aggregations={"value": custom_sum}
)
```

### Splitting a DataFrame by Field Groups

```python
field_groups = {
    "demographics": ["age", "gender"],
    "metrics": ["score", "rank"]
}
split_dfs = split_dataframe(df, field_groups, id_field="user_id")
```

---

## Integration Notes

- Designed for seamless integration with PAMOLA Core's `BaseTask` and pipeline components.
- Progress tracking is compatible with hierarchical trackers for nested operations.
- All functions are stateless and can be used independently or as part of larger workflows.

---

## Error Handling and Exception Hierarchy

- **ValueError**: Raised for invalid field or parameter configurations.
- **ImportError**: Raised if Dask is required but not installed.
- **Exception**: Used for general errors in merging, aggregation, or chunk processing.
- All errors are logged with detailed context for debugging and monitoring.

---

## Configuration Requirements

- DataFrames must contain required columns for operations (e.g., join keys, group-by fields).
- For Dask operations, ensure Dask is installed and available in the environment.
- Progress trackers (if used) should be instantiated and passed to processing functions.

---

## Security Considerations and Best Practices

- **Data Privacy**: Ensure that chunked and parallel processing does not leak sensitive data through logs or error messages.
- **Path Security**: Avoid using absolute paths for dependencies unless necessary. Prefer logical task IDs for internal data flows.
- **Dask Usage**: When using Dask, be aware of distributed execution risks (e.g., data serialization, network exposure).

### Example: Security Failure and Handling

```python
# BAD: Using absolute paths for sensitive data
merged = merge_dataframes(
    left_df, right_df, left_key="id", use_dask=True, npartitions=4
)
# If right_df is loaded from an untrusted source, this may expose data

# GOOD: Use validated, internal DataFrames and logical keys
merged = merge_dataframes(
    left_df, right_df, left_key="id", join_type="inner"
)
```

**Risks of Disabling Path Security:**
- May allow access to external or untrusted data sources
- Increases risk of data leakage or pipeline compromise
- Always validate sources and restrict access to trusted locations

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical task IDs and DataFrames produced within the pipeline.
- **External Dependencies**: Use absolute paths only for data not generated by the pipeline, and validate all external sources.

---

## Best Practices

1. **Use Chunking for Large DataFrames**: Prevent memory issues by processing in chunks.
2. **Leverage Parallelism**: Use `process_dataframe_parallel` for compute-intensive tasks.
3. **Validate Inputs**: Always validate DataFrames and parameters before processing.
4. **Monitor Progress**: Integrate progress trackers for long-running operations.
5. **Handle Errors Gracefully**: Log errors and continue processing where possible.
6. **Prefer Logical Keys**: Use task IDs and logical keys for internal data flows.
7. **Restrict Absolute Paths**: Only use absolute paths for external, trusted data.
8. **Secure Dask Usage**: Ensure Dask clusters are secure and data is not exposed.
