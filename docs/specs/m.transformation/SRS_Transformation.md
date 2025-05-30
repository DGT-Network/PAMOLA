# PAMOLA.CORE Transformation Package Software Requirements Specification

**Document Version:** 1.0.0  
**Last Updated:** May 5, 2025  
**Status:** Draft  

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document defines the requirements for the PAMOLA.CORE transformation package, which implements data preparation, cleaning, splitting, and restructuring operations to support privacy-enhancing workflows.

### 1.2 Scope

The transformation package provides a set of operations for modifying datasets to prepare them for anonymization, synthetic generation, and privacy analysis. It is implemented according to the PAMOLA.CORE Operations Framework, providing standardized interfaces for executing transformation operations, collecting metrics, generating visualizations, managing configuration, and processing large datasets efficiently.

### 1.3 Document Conventions

- **REQ-TRAN-XXX**: General transformation package requirements
- **REQ-BASE-XXX**: Base transformation operation requirements
- **REQ-SPLIT-XXX**: Splitting operation requirements
- **REQ-CLEAN-XXX**: Data cleaning operation requirements
- **REQ-IMPUTE-XXX**: Missing value imputation requirements
- **REQ-GROUP-XXX**: Grouping and aggregation requirements
- **REQ-MERGE-XXX**: Dataset merging requirements
- **REQ-FIELD-XXX**: Field manipulation requirements
- **REQ-COMMON-XXX**: Common utilities requirements

Priority levels:
- **MUST**: Essential requirement (Mandatory)
- **SHOULD**: Important but not essential requirement (Recommended)
- **MAY**: Optional requirement (Optional)

## 2. Overall Description

### 2.1 Key Principles

**REQ-TRAN-001 [MUST]** All transformation operations must follow the operation-based architecture of the PAMOLA.CORE framework.

**REQ-TRAN-002 [MUST]** All operations must use the `DataSource` abstraction for input and `OperationResult` for output.

**REQ-TRAN-003 [MUST]** All operations must organize outputs in a consistent structure under the `task_dir`.

**REQ-TRAN-004 [MUST]** All operations must generate relevant metrics and visualizations.

**REQ-TRAN-005 [MUST]** All operations must support chunked processing and progress tracking for large datasets.

**REQ-TRAN-006 [MUST]** All operations must implement efficient caching for result reuse.

**REQ-TRAN-007 [MUST]** All operations must follow standardized error handling practices.

**REQ-TRAN-008 [MUST]** All operations must provide consistent visualization and reporting.

### 2.2 Package Structure

**REQ-TRAN-009 [MUST]** The transformation package must adhere to the following structure:

```
pamola_core/transformation/
├── __init__.py
├── base_transformation_op.py      # Base class for all transformation operations
├── commons/                       # Common utilities for all transformation methods
│   ├── __init__.py
│   ├── metric_utils.py            # Common metrics utilities
│   ├── validation_utils.py        # Parameter validation utilities
│   ├── processing_utils.py        # Data processing utilities
│   └── visualization_utils.py     # Visualization helper utilities
├── splitting/
│   ├── __init__.py
│   ├── split_fields_op.py         # Vertical splitting by fields
│   └── split_by_id_values_op.py   # Vertical splitting by ID values
├── cleaning/
│   ├── __init__.py
│   └── clean_invalid_values_op.py # Clean values outside bounds or violating constraints
├── imputation/
│   ├── __init__.py
│   └── impute_missing_values_op.py # Replace missing values with statistical functions
├── grouping/
│   ├── __init__.py
│   └── aggregate_records_op.py    # Group by and aggregate operations
├── merging/
│   ├── __init__.py
│   └── merge_datasets_op.py       # Join operations between datasets
└── field_ops/
    ├── __init__.py
    ├── remove_fields_op.py        # Remove columns from dataset
    └── add_modify_fields_op.py    # Add or modify fields based on conditions
```

### 2.3 Interface Inheritance

**REQ-TRAN-010 [MUST]** All transformation operations must inherit from the base classes in the following inheritance chain:

```
BaseOperation (from pamola_core.utils.ops.op_base)
└── TransformationOperation (from pamola_core.transformation.base_transformation_op)
    └── [Specific Transformation Operations]
```

### 2.4 Task Directory Structure

**REQ-TRAN-011 [MUST]** All transformation operations must follow this standardized directory structure for artifacts:

```
{task_dir}/
├── config.json                   # Operation configuration (via self.save_config)
├── {operation}_metrics_{timestamp}.json  # Operation metrics
├── {operation}_viz_{timestamp}.png       # Operation visualization 
│     
├── output/                       # Transformed data outputs (if needed)
│   └── {dataset}_{timestamp}.{format}    # Transformed data
├── dictionaries/                 # Extracted mappings and lookups
│   └── {field}_mapping_{timestamp}.json  # Field mappings
└── logs/                         # Operation logs
    └── operation.log             # Logging output
```

**REQ-TRAN-012 [SHOULD]** All output files should include timestamps by default (configurable) to prevent overwriting previous runs.

## 3. Base Transformation Operation Requirements

### 3.1 Base Class Interface

**REQ-BASE-001 [MUST]** The `TransformationOperation` class must inherit from `BaseOperation` and provide the following constructor interface:

```python
def __init__(self, 
             name: str = "transformation_operation",
             description: str = "",
             batch_size: int = 10000,
             use_cache: bool = True,
             use_dask: bool = False,
             use_encryption: bool = False,
             encryption_key: Optional[Union[str, Path]] = None):
    """Initialize the transformation operation."""
```

**REQ-BASE-002 [MUST]** The `TransformationOperation` class must provide methods for:
1. Executing operation lifecycle via `execute()` method
2. Handling chunked processing of large datasets
3. Managing caching of operation results
4. Generating metrics and visualizations
5. Providing standardized error handling
6. Handling operations with single or multiple input datasets

### 3.2 Required Methods

**REQ-BASE-003 [MUST]** All transformation operations must implement the following methods:

```python
def execute(self, data_source: DataSource, task_dir: Path, reporter: Any, progress_tracker: Optional[ProgressTracker] = None, **kwargs) -> OperationResult:
    """Execute the transformation operation."""
    
def _process_data(self, df: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Process data according to operation-specific logic."""
    
def _collect_metrics(self, input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """Collect operation-specific metrics."""
    
def _generate_visualizations(self, input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], task_dir: Path, result: OperationResult) -> None:
    """Generate operation-specific visualizations."""
```

### 3.3 Standard Parameters

**REQ-BASE-004 [MUST]** All transformation operations must support the following standard parameters:

| Parameter | Type | Description | Default |
|---|---|---|---|
| name | str | Operation name | "transformation_operation" |
| description | str | Operation description | "" |
| batch_size | int | Batch size for processing large datasets | 10000 |
| use_cache | bool | Whether to use operation caching | True |
| use_dask | bool | Whether to use Dask for distributed processing | False |
| use_encryption | bool | Whether to encrypt output files | False |
| encryption_key | Optional[Union[str, Path]] | The encryption key or path to a key file | None |

**REQ-BASE-005 [MUST]** The `execute()` method must support the following parameters:

| Parameter | Type | Description | Default |
|---|---|---|---|
| data_source | DataSource | Source of data (DataFrame or file path) | - |
| task_dir | Path | Directory for task artifacts | - |
| reporter | Any | Reporter object for tracking progress and artifacts | - |
| progress_tracker | Optional[ProgressTracker] | Progress tracker for the operation | None |
| **kwargs | dict | Additional parameters | |

**REQ-BASE-006 [MUST]** The `execute()` method must handle the following additional parameters through **kwargs:

| Parameter | Type | Description | Default |
|---|---|---|---|
| force_recalculation | bool | Force operation even if cached results exist | False |
| generate_visualization | bool | Create visualizations | True |
| include_timestamp | bool | Include timestamp in filenames | True |
| save_output | bool | Save processed data to output directory | True |
| parallel_processes | int | Number of parallel processes to use | 1 |

### 3.4 Execution Lifecycle

**REQ-BASE-007 [MUST]** The execution lifecycle of a transformation operation must follow these steps:

1. **Initialization**: Create and initialize operation with parameters
2. **Execution Start**: Call the `execute()` method 
3. **Cache Check**: Check if a cached result exists (if enabled)
4. **Data Loading**: Get DataFrame(s) from the DataSource
5. **Validation**: Verify input data meets operation requirements
6. **Processing**: Transform the data according to the operation type
7. **Metrics Collection**: Calculate metrics on the original and transformed data
8. **Visualization Generation**: Create visualizations comparing before/after
9. **Output Generation**: Save transformed data to output directory
10. **Cache Update**: Save result to cache (if enabled)
11. **Result Return**: Return the OperationResult with status, metrics, and artifacts

### 3.5 Caching Requirements

**REQ-BASE-008 [MUST]** All transformation operations must implement proper caching with the following requirements:

1. Override `_get_cache_parameters()` to provide operation-specific parameters
2. Use provided caching methods from the base class:
   - `_check_cache()` - Check if a cached result exists
   - `_save_to_cache()` - Save operation results to cache
   - `_generate_cache_key()` - Generate a deterministic cache key
3. Include version information in the cache key to invalidate cache when code changes
4. Cache keys must incorporate all operation parameters that affect the result

### 3.6 Error Handling

**REQ-BASE-009 [MUST]** All operations must implement proper error handling:

1. Validate all inputs before starting processing
2. Use specific exception types for different error categories
3. Track and report errors without crashing the entire operation
4. Add appropriate context to error messages
5. Return `OperationResult` with `OperationStatus.ERROR` when errors occur
6. Log all errors with appropriate context via the logger

### 3.7 Metrics and Visualization

**REQ-BASE-010 [MUST]** All operations must generate and save metrics:

1. Generate basic metrics (record count, execution time, etc.)
2. Generate operation-specific metrics
3. Save metrics in a JSON file under the task directory
4. Include metrics in the OperationResult

**REQ-BASE-011 [MUST]** All operations must generate and save visualizations:

1. Generate appropriate visualizations for the operation type
2. Save visualizations in PNG format under the task directory
3. Add visualization artifacts to the OperationResult

### 3.8 Documentation

**REQ-BASE-012 [MUST]** All operations must be properly documented:

1. Include module docstring with purpose, key features, and usage examples
2. Document all methods with parameters, return values, and exceptions
3. Include inline comments for complex logic
4. Follow a consistent documentation style

## 4. Splitting Operations Requirements

### 4.1 Split Fields Operation

**REQ-SPLIT-001 [MUST]** The `SplitFieldsOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Split dataset vertically into multiple subsets by columns
- Duplicate a specified ID field into each subset

**REQ-SPLIT-002 [MUST]** The `SplitFieldsOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "split_fields_operation",
             description: str = "Split dataset by fields",
             id_field: str = None,
             field_groups: Optional[Dict[str, List[str]]] = None,
             include_id_field: bool = True,
             output_format: str = "csv",
             **kwargs):
    """Initialize split fields operation."""
```

**REQ-SPLIT-003 [MUST]** The operation must generate these visualizations:
- Bar chart showing number of fields in each subset
- Network diagram showing field distribution across subsets
- Schema visualization of original vs. split datasets

**REQ-SPLIT-004 [MUST]** The operation must calculate these metrics:
- Number of generated subsets
- Number of fields per subset
- Percentage of original dataset size for each subset
- Memory usage comparison

### 4.2 Split By ID Values Operation

**REQ-SPLIT-005 [MUST]** The `SplitByIDValuesOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Split dataset horizontally based on values in an ID field
- Support both explicit value lists and automatic partitioning

**REQ-SPLIT-006 [MUST]** The `SplitByIDValuesOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "split_by_id_values_operation",
             description: str = "Split dataset by ID values",
             id_field: str = None,
             value_groups: Optional[Dict[str, List[Any]]] = None,
             number_of_partitions: int = 0,
             partition_method: str = "equal_size",  # "equal_size", "random", "modulo"
             output_format: str = "csv",
             **kwargs):
    """Initialize split by ID values operation."""
```

**REQ-SPLIT-007 [MUST]** The operation must generate these visualizations:
- Bar chart showing record count in each subset
- Pie chart showing distribution of records across subsets
- Distribution visualization of ID values across partitions

**REQ-SPLIT-008 [MUST]** The operation must calculate these metrics:
- Number of records per partition
- Distribution statistics (min, max, mean records per partition)
- Partition size as percentage of original dataset
- Number of unique ID values per partition

## 5. Cleaning Operations Requirements

### 5.1 Clean Invalid Values Operation

**REQ-CLEAN-001 [MUST]** The `CleanInvalidValuesOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Nullify values that violate defined constraints
- Support different constraint types for different data types
- Support whitelist/blacklist validation from external files

**REQ-CLEAN-002 [MUST]** The `CleanInvalidValuesOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "clean_invalid_values_operation",
             description: str = "Clean values violating constraints",
             field_constraints: Dict[str, Dict[str, Any]] = None,
             whitelist_path: Optional[Dict[str, Path]] = None,
             blacklist_path: Optional[Dict[str, Path]] = None,
             null_replacement: Optional[Union[str, Dict[str, Any]]] = None,
             output_format: str = "csv",
             **kwargs):
    """Initialize clean invalid values operation."""
```

**REQ-CLEAN-003 [MUST]** The operation must support different constraint types:

| Data Type | Constraint Types |
|-----------|------------------|
| Numeric | min_value, max_value, valid_range, custom_function |
| Categorical | allowed_values, disallowed_values, whitelist_file, blacklist_file, pattern |
| Date | min_date, max_date, valid_format, date_range |
| Text | min_length, max_length, valid_pattern, valid_format |

**REQ-CLEAN-004 [MUST]** The operation must generate these visualizations:
- Bar chart showing number of invalid values per field
- Heatmap showing distribution of invalid values
- Before/after histograms for numeric fields
- Before/after bar charts for categorical fields

**REQ-CLEAN-005 [MUST]** The operation must calculate these metrics:
- Number of invalid values per field
- Percentage of invalid values per field
- Distribution of constraint violations by type
- List of top constraint violations

## 6. Imputation Operations Requirements

### 6.1 Impute Missing Values Operation

**REQ-IMPUTE-001 [MUST]** The `ImputeMissingValuesOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Replace missing or invalid values using statistical functions
- Support different imputation strategies per field
- Handle both NULL values and user-defined "invalid" values

**REQ-IMPUTE-002 [MUST]** The `ImputeMissingValuesOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "impute_missing_values_operation",
             description: str = "Impute missing or invalid values",
             field_strategies: Dict[str, Dict[str, Any]] = None,
             invalid_values: Optional[Dict[str, List[Any]]] = None,
             output_format: str = "csv",
             **kwargs):
    """Initialize impute missing values operation."""
```

**REQ-IMPUTE-003 [MUST]** The operation must support different imputation strategies:

| Data Type | Imputation Strategies |
|-----------|------------------------|
| Numeric | mean, median, mode, constant, min, max, interpolation |
| Categorical | mode, constant, most_frequent, random_sample |
| Date | mean_date, median_date, mode_date, constant_date, previous_date, next_date |
| Text | constant, most_frequent, random_sample |

**REQ-IMPUTE-004 [MUST]** The operation must generate these visualizations:
- Bar chart showing number of imputed values per field
- Histograms comparing distributions before/after imputation
- Box plots comparing statistical properties before/after

**REQ-IMPUTE-005 [MUST]** The operation must calculate these metrics:
- Number of imputed values per field
- Percentage of imputed values per field
- Statistical comparison before/after (mean, median, mode)
- Imputation impact on distribution measures

## 7. Grouping Operations Requirements

### 7.1 Aggregate Records Operation

**REQ-GROUP-001 [MUST]** The `AggregateRecordsOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Group records based on one or more fields
- Calculate aggregations on specified fields
- Support multiple aggregation functions

**REQ-GROUP-002 [MUST]** The `AggregateRecordsOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "aggregate_records_operation",
             description: str = "Group and aggregate records",
             group_by_fields: List[str] = None,
             aggregations: Dict[str, List[str]] = None,
             custom_aggregations: Optional[Dict[str, Callable]] = None,
             output_format: str = "csv",
             **kwargs):
    """Initialize aggregate records operation."""
```

**REQ-GROUP-003 [MUST]** The operation must support standard aggregation functions:
- count
- sum
- mean
- median
- min
- max
- std (standard deviation)
- var (variance)
- first
- last
- nunique (number of unique values)

**REQ-GROUP-004 [MUST]** The operation must generate these visualizations:
- Bar chart showing record count per group
- Aggregation comparison across groups
- Distribution of group sizes

**REQ-GROUP-005 [MUST]** The operation must calculate these metrics:
- Number of groups
- Statistics on group sizes (min, max, mean, median)
- Reduction ratio (output rows / input rows)
- Statistical properties of aggregated fields

## 8. Merging Operations Requirements

### 8.1 Merge Datasets Operation

**REQ-MERGE-001 [MUST]** The `MergeDatasetsOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Merge two datasets based on a key field
- Support different join types
- Handle one-to-one and one-to-many relationships

**REQ-MERGE-002 [MUST]** The `MergeDatasetsOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "merge_datasets_operation",
             description: str = "Merge datasets by key field",
             left_dataset_name: str = "main",
             right_dataset_name: str = None,
             right_dataset_path: Optional[Path] = None,
             left_key: str = None,
             right_key: Optional[str] = None,
             join_type: str = "left",  # "inner", "left", "right", "outer"
             suffixes: Tuple[str, str] = ("_x", "_y"),
             output_format: str = "csv",
             **kwargs):
    """Initialize merge datasets operation."""
```

**REQ-MERGE-003 [MUST]** The operation must generate these visualizations:
- Venn diagram showing record overlap
- Bar chart comparing dataset sizes before/after
- Field overlap visualization
- Join type visualization

**REQ-MERGE-004 [MUST]** The operation must calculate these metrics:
- Number of matching records
- Number of records only in left dataset
- Number of records only in right dataset
- Match percentage (matched / total)
- Number of duplicate keys in each dataset
- Number of fields before and after

## 9. Field Operations Requirements

### 9.1 Remove Fields Operation

**REQ-FIELD-001 [MUST]** The `RemoveFieldsOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Remove one or more specified fields from a dataset
- Support both explicit field lists and pattern-based selection

**REQ-FIELD-002 [MUST]** The `RemoveFieldsOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "remove_fields_operation",
             description: str = "Remove fields from dataset",
             fields_to_remove: List[str] = None,
             pattern: Optional[str] = None,
             output_format: str = "csv",
             **kwargs):
    """Initialize remove fields operation."""
```

**REQ-FIELD-003 [MUST]** The operation must generate these visualizations:
- Bar chart comparing field count before/after
- Memory usage comparison before/after
- Field removal impact visualization

**REQ-FIELD-004 [MUST]** The operation must calculate these metrics:
- Number of fields removed
- Percentage of fields removed
- Memory usage before and after
- Data shape before and after

### 9.2 Add or Modify Fields Operation

**REQ-FIELD-005 [MUST]** The `AddOrModifyFieldsOperation` class must inherit from `TransformationOperation` and support the following functionality:
- Add new fields based on lookups or conditions
- Modify existing fields based on conditions
- Support both explicit lookup tables and conditional logic

**REQ-FIELD-006 [MUST]** The `AddOrModifyFieldsOperation` constructor must support these parameters:

```python
def __init__(self, 
             name: str = "add_modify_fields_operation",
             description: str = "Add or modify fields",
             field_operations: Dict[str, Dict[str, Any]] = None,
             lookup_tables: Optional[Dict[str, Union[Path, Dict[Any, Any]]]] = None,
             output_format: str = "csv",
             **kwargs):
    """Initialize add or modify fields operation."""
```

**REQ-FIELD-007 [MUST]** The operation must support different field operation types:

| Operation Type | Description |
|----------------|-------------|
| add_constant | Add field with constant value |
| add_from_lookup | Add field with values from lookup table |
| add_conditional | Add field with values based on conditions |
| modify_constant | Replace field with constant value |
| modify_from_lookup | Replace field with values from lookup table |
| modify_conditional | Replace field with values based on conditions |
| modify_expression | Replace field with values from expression |

**REQ-FIELD-008 [MUST]** The operation must generate these visualizations:
- Bar chart showing field count before/after
- Distribution of new/modified field values
- Correlation between original and new fields

**REQ-FIELD-009 [MUST]** The operation must calculate these metrics:
- Number of fields added/modified
- Distribution statistics for new/modified fields
- Correlation between original and modified fields
- Number of null values in new fields

## 10. Common Utilities Requirements

### 10.1 Metric Utilities

**REQ-COMMON-001 [MUST]** The `metric_utils.py` module must provide functions for:
- Calculating dataset comparison metrics
- Calculating field statistics 
- Calculating transformation impact metrics
- Calculating performance metrics

**REQ-COMMON-002 [MUST]** The module must provide the following key functions:

```python
def calculate_dataset_comparison(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate metrics comparing two datasets."""

def calculate_field_statistics(
    df: pd.DataFrame,
    fields: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Calculate statistical metrics for specified fields."""

def calculate_transformation_impact(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate metrics showing impact of transformation."""

def calculate_performance_metrics(
    start_time: float,
    end_time: float,
    input_rows: int,
    output_rows: int
) -> Dict[str, Any]:
    """Calculate performance metrics for the operation."""
```

### 10.2 Validation Utilities

**REQ-COMMON-003 [MUST]** The `validation_utils.py` module must provide functions for:
- Validating field existence
- Validating field types
- Validating operation parameters
- Verifying data constraints

**REQ-COMMON-004 [MUST]** The module must provide the following key functions:

```python
def validate_fields_exist(
    df: pd.DataFrame,
    required_fields: List[str]
) -> Tuple[bool, Optional[List[str]]]:
    """Validate that required fields exist in the DataFrame."""

def validate_field_types(
    df: pd.DataFrame,
    field_types: Dict[str, str]
) -> Tuple[bool, Optional[Dict[str, str]]]:
    """Validate that fields have the expected types."""

def validate_parameters(
    parameters: Dict[str, Any],
    required_params: List[str],
    param_types: Dict[str, Type]
) -> Tuple[bool, Optional[List[str]]]:
    """Validate operation parameters."""

def validate_constraints(
    df: pd.DataFrame,
    constraints: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Validate data against defined constraints."""
```

### 10.3 Processing Utilities

**REQ-COMMON-005 [MUST]** The `processing_utils.py` module must provide functions for:
- Processing data in chunks
- Efficient dataframe operations
- Common transformation functions

**REQ-COMMON-006 [MUST]** The module must provide the following key functions:

```python
def process_in_chunks(
    df: pd.DataFrame,
    processing_function: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 10000,
    progress_tracker: Optional[ProgressTracker] = None
) -> pd.DataFrame:
    """Process large DataFrame in chunks."""

def split_dataframe(
    df: pd.DataFrame,
    field_groups: Dict[str, List[str]],
    id_field: str,
    include_id_field: bool = True
) -> Dict[str, pd.DataFrame]:
    """Split DataFrame into multiple DataFrames by field groups."""

def merge_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: Optional[str] = None,
    join_type: str = "left",
    suffixes: Tuple[str, str] = ("_x", "_y")
) -> pd.DataFrame:
    """Merge two DataFrames with proper error handling."""

def aggregate_dataframe(
    df: pd.DataFrame,
    group_by_fields: List[str],
    aggregations: Dict[str, List[str]],
    custom_aggregations: Optional[Dict[str, Callable]] = None
) -> pd.DataFrame:
    """Aggregate DataFrame by grouping fields."""
```

### 10.4 Visualization Utilities

**REQ-COMMON-007 [MUST]** The `visualization_utils.py` module must provide functions for:
- Creating standard transformation visualizations
- Comparing datasets before and after transformation
- Visualizing transformation impact

**REQ-COMMON-008 [MUST]** The module must provide the following key functions:

```python
def create_field_count_comparison(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    operation_name: str,
    output_path: Path
) -> Path:
    """Create visualization comparing field counts before/after."""

def create_record_count_comparison(
    original_df: pd.DataFrame,
    transformed_dfs: Dict[str, pd.DataFrame],
    operation_name: str,
    output_path: Path
) -> Path:
    """Create visualization comparing record counts before/after."""

def create_data_distribution_comparison(
    original_series: pd.Series,
    transformed_series: pd.Series,
    field_name: str,
    operation_name: str,
    output_path: Path
) -> Path:
    """Create visualization comparing data distributions before/after."""

def create_dataset_overview(
    df: pd.DataFrame,
    title: str,
    output_path: Path
) -> Path:
    """Create overview visualization of dataset properties."""
```

## 11. Standard Metrics and Visualizations

### 11.1 Universal Metrics

**REQ-TRAN-013 [MUST]** All transformation operations must include these base metrics:

| Metric | Description |
|---|---|
| total_input_records | Total number of input records |
| total_output_records | Total number of output records |
| total_input_fields | Total number of input fields |
| total_output_fields | Total number of output fields |
| execution_time_seconds | Total execution time in seconds |
| records_per_second | Processing throughput rate |
| transformation_type | Type of transformation performed |

### 11.2 JSON Metrics Format

**REQ-TRAN-014 [MUST]** All operations must save metrics in a consistent JSON format to `{task_dir}/{operation}_metrics_{timestamp}.json` with the following structure:

```json
{
  "operation_type": "SplitFieldsOperation",
  "input_dataset": "customers.csv",
  "total_input_records": 10000,
  "total_input_fields": 15,
  "total_output_records": 10000,
  "total_output_fields": 8,
  "id_field": "customer_id",
  "number_of_splits": 2,
  "split_info": {
    "customer_data": {
      "field_count": 5,
      "included_fields": ["customer_id", "name", "email", "phone", "address"]
    },
    "purchase_data": {
      "field_count": 4,
      "included_fields": ["customer_id", "purchase_date", "amount", "product_id"]
    }
  },
  "execution_time_seconds": 3.45,
  "processing_date": "2025-05-05T14:32:10"
}
```

**REQ-TRAN-015 [MUST]** The metrics JSON must:
1. Have a consistent structure across operations
2. Include all base metrics plus operation-specific metrics
3. Use standardized naming conventions
4. Include metadata about the operation
5. Be written using the `DataWriter.write_metrics()` method to ensure consistency

### 11.3 Visualization Standards

**REQ-TRAN-016 [MUST]** All visualizations must:

1. Use the PAMOLA.CORE visualization utilities (never direct plotting)
2. Follow consistent size and styling (800x600 pixels, PNG format)
3. Have clear titles and labels 
4. Focus on showing "before and after" effects of transformation
5. Be saved with consistent naming: `{operation}_viz_{timestamp}.png`
6. Be created through the `_generate_visualizations()` method
7. Be registered with both the result object and the reporter

**REQ-TRAN-017 [MUST]** Operations must use these primary visualization types based on transformation type:

- **Splitting operations** → Bar charts, network diagrams, Venn diagrams
- **Cleaning operations** → Bar charts, heatmaps, histograms
- **Imputation operations** → Histograms, box plots, missing value matrices
- **Grouping operations** → Bar charts, group size distributions
- **Merging operations** → Venn diagrams, field overlap visualizations
- **Field operations** → Field count comparisons, distribution charts

## 12. Performance and Scalability Requirements

### 12.1 Large Data Processing

**REQ-TRAN-018 [MUST]** All operations must support processing large datasets:

1. Process data in chunks using the `batch_size` parameter
2. Support Dask DataFrames for distributed processing when `use_dask=True`
3. Report progress during long-running operations
4. Implement memory-efficient algorithms

### 12.2 Performance Metrics

**REQ-TRAN-019 [MUST]** All operations must track and report performance metrics:

1. Execution time in seconds
2. Records processed per second
3. Memory usage (peak memory usage if available)
4. Processing stages timing breakdown

### 12.3 Memory Management

**REQ-TRAN-020 [MUST]** All operations must implement proper memory management:

1. Release temporary data structures after use
2. Process data in chunks to limit memory usage
3. Use efficient algorithms for large datasets
4. Avoid unnecessary copies of large DataFrames

## 13. Security and Privacy Requirements

### 13.1 Encryption

**REQ-TRAN-021 [MUST]** All operations must support output encryption:

1. Enable encryption when `use_encryption=True` is specified
2. Use the provided `encryption_key` for output file encryption
3. Encrypt sensitive outputs (data, metrics, etc.)
4. Report encryption status in the operation result

### 13.2 Sensitive Data Handling

**REQ-TRAN-022 [MUST]** All operations must handle sensitive data securely:

1. Never log sensitive data
2. Clear sensitive data from memory after use
3. Apply proper access controls to output files
4. Validate data transformation preserves privacy properties

## 14. Testing Requirements

### 14.1 Test Case Coverage

**REQ-TRAN-023 [MUST]** All operations must have these test categories:

- **Basic functionality**: Test with simple datasets
- **Edge cases**: Test with empty datasets, extreme values, missing fields
- **Validation**: Test parameter validation and error conditions
- **Large data**: Test with larger datasets to verify chunking
- **Compatibility**: Test with various input formats and data types

### 14.2 Testing Checklist

**REQ-TRAN-024 [MUST]** Tests must verify:

- Field validation works correctly
- Operation parameters are properly validated
- Data is correctly transformed
- Metrics are calculated correctly
- Visualizations are generated properly
- Errors are reported clearly
- Progress tracking is updated appropriately
- Output files have the correct format and content

## 15. Implementation Sequence

The recommended implementation order for transformation operations:

1. Common utilities and base transformation operation class
2. Splitting operations (field splitting, ID value splitting)
3. Field operations (remove fields, add/modify fields)
4. Cleaning and imputation operations
5. Grouping and aggregation operations
6. Merging operations

## 16. Conclusion

This specification provides a comprehensive guide for implementing transformation operations within the PAMOLA.CORE framework. By following these guidelines, developers can create consistent, robust, and well-integrated operations that provide:

1. Standardized interfaces for executing data transformation operations
2. Comprehensive metrics for evaluating transformation effectiveness
3. Clear visualizations for understanding data transformations
4. Efficient processing of large datasets
5. Robust error handling and reporting
6. Effective caching for improved performance
7. Thorough documentation for maintainability

The transformation package is a central component of PAMOLA.CORE's data preparation capabilities, providing tools for real-world data transformation with measurable impact on subsequent privacy-enhancing operations.