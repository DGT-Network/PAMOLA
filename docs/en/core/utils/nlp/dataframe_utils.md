# PAMOLA.CORE NLP DataFrame Utilities Documentation

## Module: `pamola_core.utils.nlp.dataframe_utils`

### Overview

The DataFrame Utilities module provides specialized pandas DataFrame manipulation functions for NLP text processing pipelines within the PAMOLA.CORE framework. It focuses on marker-based processing workflows, enabling efficient tracking and management of processed/unprocessed records in large-scale text analysis tasks.

### Key Features

- **Marker-based Processing**: Track processed records using configurable markers
- **Column Management**: Safe column backup and restoration
- **Legacy Data Support**: Detect and handle previously processed data
- **Vectorized Operations**: Optimized performance for large DataFrames
- **Type Safety**: TypedDict definitions for structured return values
- **Validation Framework**: Comprehensive input validation
- **Memory Efficiency**: Chunked processing support for large datasets

### Architecture

The module follows PAMOLA.CORE design principles:

```
┌─────────────────────────────────────────────┐
│          DataFrame Utilities                 │
├─────────────────────────────────────────────┤
│  Validation Layer                           │
│  ├── validate_marker()                      │
│  └── validate_dataframe_for_processing()   │
├─────────────────────────────────────────────┤
│  Core Operations                            │
│  ├── prepare_marked_column()                │
│  ├── get_marker_statistics()               │
│  ├── create_column_backup()                 │
│  └── identify_processed_rows()              │
├─────────────────────────────────────────────┤
│  Analysis Functions                         │
│  ├── get_unprocessed_indices()             │
│  ├── compare_columns()                      │
│  └── split_by_processing_status()          │
└─────────────────────────────────────────────┘
```

### Type Definitions

```python
class MarkerStatistics(TypedDict):
    """Statistics about marked/unmarked records."""
    total: int                    # Total number of records
    processed: int                # Records with marker
    unprocessed: int             # Records without marker
    percentage_complete: float    # Completion percentage
    non_empty: int               # Non-empty values
    empty: int                   # Empty/NA values
    column_exists: bool          # Whether column exists

class ProcessingIndicesStats(TypedDict):
    """Statistics for processing indices."""
    total_in_range: int          # Records in ID range
    already_processed: int       # Already processed count
    to_process: int             # To be processed count
    limited_by_max: bool        # If limited by max_records
```

### Core Functions

#### `prepare_marked_column()`

Prepares a DataFrame for marker-based processing by ensuring the target column exists and properly handles existing data.

```python
def prepare_marked_column(
    df: pd.DataFrame,
    source: str,
    target: str,
    marker: str,
    clear_target: bool = False
) -> pd.DataFrame
```

**Parameters:**
- `df`: Input DataFrame
- `source`: Source column name
- `target`: Target column name for processed data
- `marker`: Processing marker (e.g., "~", "PROCESSED:")
- `clear_target`: Whether to clear existing target column

**Returns:**
- Modified DataFrame with prepared target column

**Example:**
```python
# Prepare DataFrame for entity extraction
df = prepare_marked_column(
    df=data,
    source="text",
    target="entities",
    marker="~",
    clear_target=False
)
```

#### `get_marker_statistics()`

Retrieves comprehensive statistics about processing status in a column.

```python
def get_marker_statistics(
    df: pd.DataFrame,
    column: str,
    marker: str
) -> MarkerStatistics
```

**Example:**
```python
stats = get_marker_statistics(df, "entities", "~")
print(f"Progress: {stats['percentage_complete']:.1f}%")
print(f"Processed: {stats['processed']}/{stats['total']}")
```

#### `create_column_backup()`

Creates a backup of a column before in-place modifications.

```python
def create_column_backup(
    df: pd.DataFrame,
    source_column: str,
    backup_suffix: str = "_original",
    force: bool = False
) -> Optional[str]
```

**Example:**
```python
# Backup original text before normalization
backup_col = create_column_backup(
    df=data,
    source_column="text",
    backup_suffix="_raw"
)
```

#### `identify_processed_rows()`

Identifies which rows have been processed based on markers or value changes.

```python
def identify_processed_rows(
    df: pd.DataFrame,
    target_column: str,
    marker: str,
    source_column: Optional[str] = None
) -> pd.Series
```

**Returns:**
- Boolean Series where True indicates processed rows

**Example:**
```python
processed_mask = identify_processed_rows(
    df=data,
    target_column="normalized_text",
    marker="~",
    source_column="text"
)
processed_df = data[processed_mask]
```

#### `get_unprocessed_indices()`

Retrieves indices of unprocessed records with optional filtering.

```python
def get_unprocessed_indices(
    df: pd.DataFrame,
    target_column: str,
    marker: str,
    source_column: Optional[str] = None,
    id_column: Optional[str] = None,
    start_id: Optional[Union[int, str]] = None,
    end_id: Optional[Union[int, str]] = None,
    max_records: Optional[int] = None
) -> Tuple[pd.Index, ProcessingIndicesStats]
```

**Example:**
```python
# Get next batch of unprocessed records
indices, stats = get_unprocessed_indices(
    df=data,
    target_column="entities",
    marker="~",
    id_column="record_id",
    start_id=1000,
    max_records=100
)
print(f"Processing {stats['to_process']} records")
```

#### `compare_columns()`

Compares two columns for differences with normalization options.

```python
def compare_columns(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    ignore_whitespace: bool = True,
    ignore_case: bool = False,
    chunksize: Optional[int] = None
) -> pd.Series
```

**Example:**
```python
# Find records where normalization changed the text
changes = compare_columns(
    df=data,
    column1="text",
    column2="normalized_text",
    ignore_whitespace=True,
    ignore_case=True
)
modified_df = data[changes]
```

#### `split_by_processing_status()`

Splits DataFrame into processed and unprocessed subsets.

```python
def split_by_processing_status(
    df: pd.DataFrame,
    target_column: str,
    marker: str,
    source_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

**Example:**
```python
processed_df, unprocessed_df = split_by_processing_status(
    df=data,
    target_column="entities",
    marker="~"
)
```

### Usage Patterns

#### Pattern 1: Basic Processing Workflow

```python
import pandas as pd
from pamola_core.utils.nlp.dataframe_utils import (
    prepare_marked_column,
    get_marker_statistics,
    get_unprocessed_indices
)

# Load data
df = pd.read_csv("texts.csv")

# Prepare for processing
df = prepare_marked_column(
    df=df,
    source="text",
    target="processed_text",
    marker="DONE:"
)

# Check initial status
stats = get_marker_statistics(df, "processed_text", "DONE:")
print(f"Starting with {stats['unprocessed']} unprocessed records")

# Process in batches
while stats['unprocessed'] > 0:
    # Get next batch
    indices, _ = get_unprocessed_indices(
        df=df,
        target_column="processed_text",
        marker="DONE:",
        max_records=1000
    )
    
    # Process batch
    for idx in indices:
        text = df.loc[idx, 'text']
        processed = process_text(text)  # Your processing function
        df.loc[idx, 'processed_text'] = f"DONE:{processed}"
    
    # Update stats
    stats = get_marker_statistics(df, "processed_text", "DONE:")
    print(f"Progress: {stats['percentage_complete']:.1f}%")
```

#### Pattern 2: Legacy Data Handling

```python
# Handle mixed processed/unprocessed data
df = prepare_marked_column(
    df=legacy_df,
    source="original_text",
    target="clean_text",
    marker="~",
    clear_target=False  # Preserve existing processed data
)

# Identify legacy processed rows (different but no marker)
processed_mask = identify_processed_rows(
    df=legacy_df,
    target_column="clean_text",
    marker="~",
    source_column="original_text"  # Compare to detect legacy
)

print(f"Found {processed_mask.sum()} previously processed records")
```

#### Pattern 3: Safe In-Place Operations

```python
# Create backup before modification
backup_col = create_column_backup(
    df=df,
    source_column="text",
    backup_suffix="_original"
)

# Perform in-place normalization
df['text'] = df['text'].str.lower().str.strip()

# Compare changes
changes = compare_columns(
    df=df,
    column1=backup_col,
    column2="text",
    ignore_case=False
)

print(f"Modified {changes.sum()} records")
```

### Best Practices

1. **Marker Selection**
   - Use unique, non-conflicting markers (e.g., "~", "PROCESSED:", "##")
   - Avoid regex special characters unless necessary
   - Be consistent across your pipeline

2. **Performance Optimization**
   - Use `chunksize` parameter for large DataFrames in `compare_columns()`
   - Process in batches using `max_records` in `get_unprocessed_indices()`
   - Leverage vectorized operations wherever possible

3. **Error Handling**
   ```python
   from pamola_core.utils.nlp.dataframe_utils import (
       ColumnNotFoundError,
       MarkerValidationError
   )
   
   try:
       df = prepare_marked_column(df, "text", "processed", "~")
   except ColumnNotFoundError as e:
       logger.error(f"Missing column: {e}")
   except MarkerValidationError as e:
       logger.error(f"Invalid marker: {e}")
   ```

4. **Memory Management**
   - Use `get_unprocessed_indices()` with `max_records` for large datasets
   - Consider chunked processing for memory-intensive operations
   - Clean up backup columns when no longer needed

### Integration Examples

#### With Entity Extraction

```python
from pamola_core.utils.nlp.entity_extraction import extract_entities
from pamola_core.utils.nlp.dataframe_utils import prepare_marked_column

# Prepare DataFrame
df = prepare_marked_column(
    df=data,
    source="text",
    target="entities",
    marker="~"
)

# Process unprocessed records
indices, _ = get_unprocessed_indices(df, "entities", "~")
for idx in indices:
    text = df.loc[idx, 'text']
    entities = extract_entities(text)
    df.loc[idx, 'entities'] = f"~{json.dumps(entities)}"
```

#### With Text Transformation

```python
from pamola_core.utils.nlp.text_transformer import TextTransformer
from pamola_core.utils.nlp.dataframe_utils import create_column_backup

# Backup original
backup = create_column_backup(df, "text")

# Apply transformations
transformer = TextTransformer()
df['text'] = df['text'].apply(transformer.transform)

# Validate changes
changes = compare_columns(df, backup, "text")
```

### Common Issues and Solutions

1. **Type Mismatches in ID Filtering**
   - The module handles numeric/string ID mismatches automatically
   - Converts types as needed for comparison

2. **Legacy Data Detection**
   - Rows without markers but different from source are considered processed
   - Use `source_column` parameter for accurate legacy detection

3. **Concurrent Processing**
   - Use unique markers for different processing stages
   - Consider file locking for parallel workflows

### Version History

- **1.1.1** - Current version with improved type handling and validation
- **1.1.0** - Added TypedDict support and performance optimizations
- **1.0.0** - Initial implementation

### See Also

- [`pamola_core.utils.nlp.text_utils`](./text_utils.md) - Text processing utilities
- [`pamola_core.utils.nlp.entity_extraction`](./entity_extraction.md) - Entity extraction
- [`pamola_core.utils.nlp.text_transformer`](./text_transformer.md) - Text transformation
- [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) - Pandas documentation