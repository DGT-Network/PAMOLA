# PAMOLA.CORE Anonymization Package Software Requirements Specification

**Document Version:** 1.1.0  
**Last Updated:** May 4, 2025  
**Status:** Draft  

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document defines the requirements for the PAMOLA.CORE anonymization package, which implements privacy-enhancing operations through various techniques such as generalization, masking, noise addition, pseudonymization, and suppression.

### 1.2 Scope

The anonymization package provides a set of operations for anonymizing data while preserving analytical utility. It is implemented according to the PAMOLA.CORE Operations Framework, providing standardized interfaces for executing privacy operations, collecting metrics, generating visualizations, managing configuration, and processing large datasets efficiently.

### 1.3 Document Conventions

- **REQ-ANON-XXX**: General anonymization package requirements
- **REQ-BASE-XXX**: Base anonymization operation requirements
- **REQ-GEN-XXX**: Generalization operation requirements
- **REQ-MASK-XXX**: Masking operation requirements
- **REQ-NOISE-XXX**: Noise addition operation requirements
- **REQ-PSEUDO-XXX**: Pseudonymization operation requirements
- **REQ-SUPP-XXX**: Suppression operation requirements
- **REQ-COMMON-XXX**: Common utilities requirements

Priority levels:
- **MUST**: Essential requirement (Mandatory)
- **SHOULD**: Important but not essential requirement (Recommended)
- **MAY**: Optional requirement (Optional)

## 2. Overall Description

### 2.1 Key Principles

**REQ-ANON-001 [MUST]** All anonymization operations must follow the operation-based architecture of the PAMOLA.CORE framework.

**REQ-ANON-002 [MUST]** All operations must use the `DataSource` abstraction for input and `OperationResult` for output.

**REQ-ANON-003 [MUST]** All operations must organize outputs in a consistent structure under the `task_dir`.

**REQ-ANON-004 [MUST]** All operations must generate relevant metrics and visualizations.

**REQ-ANON-005 [MUST]** All operations must support chunked processing and progress tracking for large datasets.

**REQ-ANON-006 [MUST]** All operations must implement efficient caching for result reuse.

**REQ-ANON-007 [MUST]** All operations must follow standardized error handling practices.

**REQ-ANON-008 [MUST]** All operations must provide consistent visualization and reporting.

### 2.2 Package Structure

**REQ-ANON-009 [MUST]** The anonymization package must adhere to the following structure:

```
pamola_core/anonymization/
├── __init__.py
├── base_anonymization_op.py      # Base class for all anonymization operations
├── commons/                      # Common utilities for all anonymization methods
│   ├── __init__.py
│   ├── metric_utils.py           # Common metrics utilities
│   ├── validation_utils.py       # Parameter validation utilities
│   ├── processing_utils.py       # Data processing utilities
│   └── visualization_utils.py    # Visualization helper utilities
├── generalization/
│   ├── __init__.py
│   ├── categorical_op.py         # Categorical generalization operation
│   ├── numeric_op.py             # Numeric generalization operation
│   └── datetime_op.py            # Datetime generalization operation
├── masking/
│   ├── __init__.py
│   ├── full_masking_op.py        # Full masking operation
│   └── partial_masking_op.py     # Partial masking operation
├── noise_addition/
│   ├── __init__.py
│   ├── gaussian_op.py            # Gaussian noise addition operation
│   ├── laplace_op.py             # Laplace noise addition operation 
│   └── uniform_op.py             # Uniform noise addition operation
├── pseudonymization/
│   ├── __init__.py
│   ├── base.py                   # Base pseudonymization functionality
│   ├── hashing_op.py             # Hash-based pseudonymization operation
│   ├── replacement_op.py         # Dictionary-based replacement operation
│   └── tokenization_op.py        # Tokenization operation
└── suppression/
    ├── __init__.py
    ├── attribute_suppression_op.py  # Attribute-level suppression operation
    ├── record_suppression_op.py     # Record-level suppression operation
    └── cell_suppression_op.py       # Cell-level suppression operation
```

### 2.3 Interface Inheritance

**REQ-ANON-010 [MUST]** All anonymization operations must inherit from the base classes in the following inheritance chain:

```
BaseOperation (from pamola_core.utils.ops.op_base)
└── AnonymizationOperation (from pamola_core.anonymization.base_anonymization_op)
    └── [Specific Anonymization Operations]
```

### 2.4 Task Directory Structure

**REQ-ANON-011 [MUST]** All anonymization operations must follow this standardized directory structure for artifacts:

```
{task_dir}/
├── config.json                   # Operation configuration (via self.save_config)
├── {field}_{operation}_{timestamp}.json        # Operation / Data metrics
├── {field}_{operation_visType}_{timestamp}.png # Visualization 
│     
├── output/                       # Transformed data outputs (if needed)
│   └── {dataset}_{timestamp}.{format} # Anonymized data (could be encrypted)
├── dictionaries/                 # Extracted dictionaries and mappings
│   ├── {field}_mapping_{timestamp}.csv         # Frequency/ Value mappings
│   └── {field}_hierarchy_{timestamp}.json      # Generalization hierarchies
└── logs/                         # Operation logs
    └── operation.log             # Logging output
```

**REQ-ANON-012 [SHOULD]** All output files should include timestamps by default (configurable) to prevent overwriting previous runs.

## 3. Base Anonymization Operation Requirements

### 3.1 Base Class Interface

**REQ-BASE-001 [MUST]** The `AnonymizationOperation` class must inherit from `BaseOperation` and provide the following constructor interface:

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
             encryption_key: Optional[Union[str, Path]] = None):
    """Initialize the anonymization operation."""
```

**REQ-BASE-002 [MUST]** The `AnonymizationOperation` class must provide methods for:
1. Executing operation lifecycle via `execute()` method
2. Handling chunked processing of large datasets
3. Managing caching of operation results
4. Generating metrics and visualizations
5. Providing standardized error handling
6. Supporting both in-place (REPLACE) and new field creation (ENRICH) modes
7. Handling different null value strategies (PRESERVE, EXCLUDE, ERROR)

### 3.2 Required Methods

**REQ-BASE-003 [MUST]** All anonymization operations must implement the following methods:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process a batch of data."""
    
def process_value(self, value, **params):
    """Process a single value."""
    
def _collect_specific_metrics(self, original_data: pd.Series, anonymized_data: pd.Series) -> Dict[str, Any]:
    """Collect operation-specific metrics."""
    
def _get_cache_parameters(self) -> Dict[str, Any]:
    """Get operation-specific parameters for cache key generation."""
```

### 3.3 Standard Parameters

**REQ-BASE-004 [MUST]** All anonymization operations must support the following standard parameters:

| Parameter | Type | Description | Default |
|---|---|---|---|
| field_name | str | Field(s) to anonymize | - |
| mode | str | "REPLACE" or "ENRICH" | "REPLACE" |
| output_field_name | Optional[str] | Name for output field if mode is "ENRICH" | None |
| column_prefix | str | Prefix for new column if mode is "ENRICH" | "_" |
| null_strategy | str | How to handle NULL values: "PRESERVE", "EXCLUDE", "ERROR" | "PRESERVE" |
| batch_size | int | Batch size for processing large datasets | 10000 |
| use_cache | bool | Whether to use operation caching | True |
| description | str | Operation description | "" |
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
| use_dask | bool | Whether to use parallel processing | False |
| force_recalculation | bool | Force operation even if cached results exist | False |
| encrypt_output | bool | Override encryption setting for outputs | False |
| generate_visualization | bool | Create visualizations | True |
| include_timestamp | bool | Include timestamp in filenames | True |
| save_output | bool | Save processed data to output directory | True |
| parallel_processes | int | Number of parallel processes to use | 1 |

### 3.4 Execution Lifecycle

**REQ-BASE-007 [MUST]** The execution lifecycle of an anonymization operation must follow these steps:

1. **Initialization**: Create and initialize operation with parameters
2. **Execution Start**: Call the `execute()` method
3. **Cache Check**: Check if a cached result exists (if enabled)
4. **Data Loading**: Get DataFrame from the DataSource
5. **Validation**: Verify field exists and has the correct type
6. **Processing**: Process the data according to the operation type
7. **Metrics Collection**: Calculate metrics on the original and processed data
8. **Visualization Generation**: Create visualizations comparing before/after
9. **Output Generation**: Save processed data to output directory
10. **Cache Update**: Save result to cache (if enabled)
11. **Memory Cleanup**: Release temporary data structures
12. **Result Return**: Return the OperationResult with status, metrics, and artifacts

### 3.5 Caching Requirements

**REQ-BASE-008 [MUST]** All anonymization operations must implement proper caching with the following requirements:

1. Override `_get_cache_parameters()` to provide operation-specific parameters
2. Use provided caching methods from the base class:
   - `_check_cache()` - Check if a cached result exists
   - `_save_to_cache()` - Save operation results to cache
   - `_generate_cache_key()` - Generate a deterministic cache key
3. Include version information in the cache key to invalidate cache when code changes
4. Cache keys must incorporate all operation parameters that affect the result
5. Cache invalidation must happen automatically when operation code is updated

**REQ-BASE-009 [SHOULD]** Operations should respect the cache control parameters:
   - `max_age_days` (default: 7.0) - Maximum age of cache files in days
   - `max_size_mb` (default: 500.0) - Maximum size of cache directory in MB

### 3.6 Error Handling

**REQ-BASE-010 [MUST]** All operations must implement proper error handling:

1. Validate all inputs before starting processing
2. Use specific exception types for different error categories
3. Handle null values according to the null_strategy parameter
4. Track and report errors without crashing the entire operation
5. Add appropriate context to error messages
6. Return `OperationResult` with `OperationStatus.ERROR` when errors occur
7. Log all errors with appropriate context via the logger

### 3.7 Metrics and Visualization

**REQ-BASE-011 [MUST]** All operations must generate and save metrics:

1. Generate basic metrics (record count, execution time, etc.)
2. Generate operation-specific metrics
3. Save metrics in a JSON file under the task directory
4. Include metrics in the OperationResult
5. Follow the standard metrics format

**REQ-BASE-012 [MUST]** All operations must generate and save visualizations:

1. Generate appropriate visualizations for the operation type
2. Save visualizations in PNG format under the task directory
3. Add visualization artifacts to the OperationResult
4. Follow the standard visualization format

### 3.8 Memory Management

**REQ-BASE-013 [MUST]** All operations must implement proper memory management:

1. Process data in chunks for large datasets
2. Clean up temporary data structures after processing
3. Use Dask for very large datasets when enabled
4. Implement a `_cleanup_memory()` method to release resources

### 3.9 Documentation

**REQ-BASE-014 [MUST]** All operations must be properly documented:

1. Include module docstring with purpose, key features, and usage examples
2. Document all methods with parameters, return values, and exceptions
3. Include inline comments for complex logic
4. Follow a consistent documentation style

## 4. Generalization Operations Requirements

### 4.1 Categorical Generalization Operation

**REQ-GEN-001 [MUST]** The `CategoricalGeneralizationOperation` class must inherit from `AnonymizationOperation` and support the following strategies:
- `merge_low_freq`: Merge low-frequency categories
- `hierarchy`: Use a predefined hierarchy for generalization
- `semantic`: Group categories based on semantic similarity

**REQ-GEN-002 [MUST]** The `CategoricalGeneralizationOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             strategy: str = "merge_low_freq",  # "merge_low_freq", "hierarchy", "semantic"
             min_group_size: int = 5,
             hierarchy: Optional[Dict[str, str]] = None,
             **kwargs):
    """Initialize categorical generalization operation."""
```

**REQ-GEN-003 [MUST]** The operation must generate these visualizations:
- Bar plot comparing original and generalized value distributions
- Pie chart showing reduction in unique categories
- Heatmap for hierarchical generalizations (if using hierarchy strategy)

**REQ-GEN-004 [MUST]** The operation must calculate these metrics:
- Category reduction ratio
- Information loss estimate
- Grouping consistency metrics
- Minimum group size verification

### 4.2 Numeric Generalization Operation

**REQ-GEN-005 [MUST]** The `NumericGeneralizationOperation` class must inherit from `AnonymizationOperation` and support the following strategies:
- `binning`: Group values into bins
- `rounding`: Round values to a specified precision
- `range`: Replace values with range expressions

**REQ-GEN-006 [MUST]** The `NumericGeneralizationOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             strategy: str = "binning",  # "binning", "rounding", "range"
             bin_count: int = 10,
             precision: int = 0,  # For rounding strategy
             range_limits: Optional[Tuple[float, float]] = None,  # For range strategy
             **kwargs):
    """Initialize numeric generalization operation."""
```

**REQ-GEN-007 [MUST]** The operation must generate these visualizations:
- Histogram comparing original and generalized distributions
- Box plot showing statistical changes
- Bin distribution visualization (for binning strategy)

**REQ-GEN-008 [MUST]** The operation must calculate these metrics:
- Generalization ratio (reduction in unique values)
- Mean absolute difference
- Statistical property preservation metrics (mean, median, std deviation)
- Strategy-specific metrics (bin counts, rounding precision)

**REQ-GEN-009 [MUST]** The operation must implement the following strategy-specific processing logic:
- For binning: Create optimal bin edges and assign values to bins
- For rounding: Round values to the specified precision
- For range: Replace values with range expressions based on the specified limits

**REQ-GEN-010 [MUST]** The operation must handle non-numeric fields gracefully:
- If a field is not numeric, log a warning
- If in ENRICH mode, copy values to the output field
- If in REPLACE mode, leave values unchanged
- Return with SUCCESS status and appropriate metrics

**REQ-GEN-011 [MUST]** The operation must handle null values according to the `null_strategy` parameter:
- `PRESERVE`: Keep null values as null
- `EXCLUDE`: Skip null values during processing
- `ERROR`: Raise an error if null values are found

**REQ-GEN-012 [MUST]** The operation must produce well-defined output values:
- For binning: Range expressions (e.g., "10.0-20.0")
- For rounding: Rounded numeric values
- For range: Range expressions or boundary indicators (e.g., "<10.0", "10.0-20.0", ">20.0")

### 4.3 Datetime Generalization Operation

**REQ-GEN-013 [MUST]** The `DatetimeGeneralizationOperation` class must inherit from `AnonymizationOperation` and support the following granularities:
- `year`: Truncate to year
- `month`: Truncate to month
- `day`: Truncate to day
- `hour`: Truncate to hour

**REQ-GEN-014 [MUST]** The `DatetimeGeneralizationOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             granularity: str = "month",  # "year", "month", "day", "hour"
             format_string: Optional[str] = None,
             **kwargs):
    """Initialize datetime generalization operation."""
```

**REQ-GEN-015 [MUST]** The operation must generate these visualizations:
- Time-series histogram before/after
- Calendar heatmap (if appropriate)
- Distribution by granularity level

**REQ-GEN-016 [MUST]** The operation must calculate these metrics:
- Datetime precision reduction
- Temporal distance metrics
- Seasonality preservation metrics

## 5. Masking Operations Requirements

### 5.1 Full Masking Operation

**REQ-MASK-001 [MUST]** The `FullMaskingOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Replace all characters in the field with a mask character
- Optionally preserve the length of the original value

**REQ-MASK-002 [MUST]** The `FullMaskingOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             mask_char: str = "*",
             preserve_length: bool = True,
             **kwargs):
    """Initialize full masking operation."""
```

**REQ-MASK-003 [MUST]** The operation must generate these visualizations:
- Information removal bar chart
- Character type preservation visualization

**REQ-MASK-004 [MUST]** The operation must calculate these metrics:
- Masking coverage percentage
- Information loss metrics
- Type preservation metrics

### 5.2 Partial Masking Operation

**REQ-MASK-005 [MUST]** The `PartialMaskingOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Mask portions of the field while preserving some characters
- Support different masking patterns based on the field type
- Support custom pattern-based masking

**REQ-MASK-006 [MUST]** The `PartialMaskingOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             mask_char: str = "*",
             mask_pattern: Optional[str] = None,  # Regex pattern for what to mask
             unmasked_prefix: int = 0,  # Number of characters to leave unmasked at start
             unmasked_suffix: int = 0,  # Number of characters to leave unmasked at end
             special_field_type: Optional[str] = None,  # "email", "phone", "credit_card", etc.
             **kwargs):
    """Initialize partial masking operation."""
```

**REQ-MASK-007 [MUST]** The operation must generate these visualizations:
- Information retention graph
- Pattern-matching visualization if applicable
- Field type-specific visualizations

**REQ-MASK-008 [MUST]** The operation must calculate these metrics:
- Information retention ratio
- Re-identification risk estimate
- Field format preservation metrics

## 6. Noise Addition Operations Requirements

### 6.1 Gaussian Noise Operation

**REQ-NOISE-001 [MUST]** The `GaussianNoiseOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Add Gaussian noise to numeric fields
- Support differential privacy parameters
- Support bounds on the noise range

**REQ-NOISE-002 [MUST]** The `GaussianNoiseOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             mean: float = 0.0,
             std_dev: float = 1.0,
             min_val: Optional[float] = None,
             max_val: Optional[float] = None,
             epsilon: Optional[float] = None,  # For differential privacy
             delta: Optional[float] = None,  # For differential privacy
             noise_budget_percentage: float = 10.0,  # Maximum percentage of value to add as noise
             **kwargs):
    """Initialize Gaussian noise operation."""
```

**REQ-NOISE-003 [MUST]** The operation must generate these visualizations:
- Overlaid histograms before/after
- Noise distribution visualization
- Privacy vs. utility trade-off graph

**REQ-NOISE-004 [MUST]** The operation must calculate these metrics:
- Statistical property changes (mean, std)
- Privacy guarantee metrics (if ε-DP)
- Re-identification risk reduction

### 6.2 Laplace Noise Operation

**REQ-NOISE-005 [MUST]** The `LaplaceNoiseOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Add Laplace noise to numeric fields
- Support differential privacy parameters
- Support bounds on the noise range

**REQ-NOISE-006 [MUST]** The `LaplaceNoiseOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             scale: float = 1.0,
             min_val: Optional[float] = None,
             max_val: Optional[float] = None,
             epsilon: Optional[float] = None,  # For differential privacy
             noise_budget_percentage: float = 10.0,  # Maximum percentage of value to add as noise
             **kwargs):
    """Initialize Laplace noise operation."""
```

**REQ-NOISE-007 [MUST]** The operation must generate these visualizations:
- Overlaid histograms before/after
- Noise distribution visualization
- Privacy guarantee metrics (if ε-DP)

**REQ-NOISE-008 [MUST]** The operation must calculate these metrics:
- Statistical property changes
- Privacy guarantee metrics
- Utility preservation metrics

### 6.3 Uniform Noise Operation

**REQ-NOISE-009 [MUST]** The `UniformNoiseOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Add uniform noise to numeric fields
- Support bounds on the noise range
- Support percentage-based noise budgets

**REQ-NOISE-010 [MUST]** The `UniformNoiseOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             min_range: float = -1.0,
             max_range: float = 1.0,
             noise_budget_percentage: float = 10.0,  # Maximum percentage of value to add as noise
             **kwargs):
    """Initialize uniform noise operation."""
```

**REQ-NOISE-011 [MUST]** The operation must generate these visualizations:
- Overlaid histograms before/after
- Range modification visualization
- Error distribution graph

**REQ-NOISE-012 [MUST]** The operation must calculate these metrics:
- Data range expansion metrics
- Statistical property changes
- Re-identification risk reduction

## 7. Pseudonymization Operations Requirements

### 7.1 Hashing Operation

**REQ-PSEUDO-001 [MUST]** The `HashingOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Hash field values using cryptographic hash functions
- Support salting for improved security
- Check for and report hash collisions

**REQ-PSEUDO-002 [MUST]** The `HashingOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             hash_function: str = "SHA256",  # "SHA256", "MD5", etc.
             salt: str = "",  # Salt for hashing (recommended for security)
             check_collisions: bool = True,
             collision_threshold: float = 0.01,  # Warn if collision rate exceeds this percentage
             **kwargs):
    """Initialize hashing operation."""
```

**REQ-PSEUDO-003 [MUST]** The operation must generate these visualizations:
- Hash distribution visualization
- Collision detection graph (if enabled)
- Security strength indicator

**REQ-PSEUDO-004 [MUST]** The operation must calculate these metrics:
- Collision rate metrics
- Uniqueness preservation
- Format changes (e.g., length, character distribution)

### 7.2 Replacement Operation

**REQ-PSEUDO-005 [MUST]** The `ReplacementOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Replace field values using a dictionary
- Support different dictionary types (custom, names, organizations, locations)
- Support consistent mapping between original and replacement values

**REQ-PSEUDO-006 [MUST]** The `ReplacementOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             replacement_dict: Optional[Dict[str, str]] = None,
             replacement_file: Optional[str] = None,
             dictionary_type: str = "custom",  # "custom", "names", "organizations", "locations"
             save_mapping: bool = False,
             mapping_file: Optional[str] = None,
             consistent_mapping: bool = True,
             **kwargs):
    """Initialize replacement operation."""
```

**REQ-PSEUDO-007 [MUST]** The operation must generate these visualizations:
- Value distribution before/after
- Dictionary coverage visualization
- Mapping completeness graph

**REQ-PSEUDO-008 [MUST]** The operation must calculate these metrics:
- Replacement coverage percentage
- Dictionary efficiency metrics
- Format consistency metrics

### 7.3 Tokenization Operation

**REQ-PSEUDO-009 [MUST]** The `TokenizationOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Replace field values with tokens
- Support different token generation strategies
- Support consistent mapping between original values and tokens

**REQ-PSEUDO-010 [MUST]** The `TokenizationOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             token_length: int = 10,
             preserve_original_length: bool = False,
             token_prefix: str = "",
             token_dictionary: Optional[List[str]] = None,  # Custom token source dictionary
             save_mapping: bool = False,
             mapping_file: Optional[str] = None,
             **kwargs):
    """Initialize tokenization operation."""
```

**REQ-PSEUDO-011 [MUST]** The operation must generate these visualizations:
- Token length distribution
- Format changes visualization
- Token uniqueness visualization

**REQ-PSEUDO-012 [MUST]** The operation must calculate these metrics:
- Tokenization consistency metrics
- Format properties metrics
- One-to-one mapping verification

## 8. Suppression Operations Requirements

### 8.1 Attribute Suppression Operation

**REQ-SUPP-001 [MUST]** The `AttributeSuppressionOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Remove or mask entire columns from the dataset
- Support multiple field suppression in a single operation

**REQ-SUPP-002 [MUST]** The `AttributeSuppressionOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str,  # Can be a list of fields
             suppression_strategy: str = "remove",  # "remove", "mask"
             replacement_value: Optional[Any] = None,
             **kwargs):
    """Initialize attribute suppression operation."""
```

**REQ-SUPP-003 [MUST]** The operation must generate these visualizations:
- Before/after DataFrame structure visualization
- Impact on data completeness

**REQ-SUPP-004 [MUST]** The operation must calculate these metrics:
- Information loss metrics
- Impact on overall dataset utility
- Dimensionality reduction metrics

### 8.2 Cell Suppression Operation

**REQ-SUPP-005 [MUST]** The `CellSuppressionOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Suppress individual cells based on specified criteria
- Support different suppression criteria (rare values, outliers, custom)
- Replace suppressed values with a specified replacement

**REQ-SUPP-006 [MUST]** The `CellSuppressionOperation` constructor must support these parameters:

```python
def __init__(self, 
             field_name: str, 
             suppression_criteria: str = "rare_values",  # "rare_values", "outliers", "custom"
             threshold: float = 5,  # For rare_values: count, for outliers: std deviations
             custom_filter: Optional[Callable] = None,  # For custom criteria
             replacement_value: Optional[Any] = None,
             **kwargs):
    """Initialize cell suppression operation."""
```

**REQ-SUPP-007 [MUST]** The operation must generate these visualizations:
- Suppressed cells visualization (heatmap or similar)
- Impact on distribution visualization
- Criteria thresholds visualization

**REQ-SUPP-008 [MUST]** The operation must calculate these metrics:
- Suppression rate metrics
- Value rarity/uniqueness metrics
- Outlier detection metrics

### 8.3 Record Suppression Operation

**REQ-SUPP-009 [MUST]** The `RecordSuppressionOperation` class must inherit from `AnonymizationOperation` and support the following functionality:
- Suppress entire records based on specified criteria
- Support different suppression criteria (rare values, outliers, custom)
- Evaluate criteria based on a specified field

**REQ-SUPP-010 [MUST]** The `RecordSuppressionOperation` constructor must support these parameters:

```python
def __init__(self, 
             criteria_field: str,  # Field to evaluate for suppression criteria
             criteria: str = "rare_values",  # "rare_values", "outliers", "custom"
             threshold: float = 5,  # For rare_values: count, for outliers: std deviations
             custom_filter: Optional[Callable] = None,  # For custom criteria
             **kwargs):
    """Initialize record suppression operation."""
```

**REQ-SUPP-011 [MUST]** The operation must generate these visualizations:
- Suppressed records distribution
- Impact on overall dataset visualization
- Criteria thresholds visualization

**REQ-SUPP-012 [MUST]** The operation must calculate these metrics:
- Suppression rate metrics
- Dataset size reduction metrics
- Statistical property changes

## 9. Common Utilities Requirements

### 9.1 Metric Utilities

**REQ-COMMON-001 [MUST]** The `metric_utils.py` module must provide functions for:
- Calculating basic numeric metrics (mean, median, std, etc.)
- Calculating generalization metrics (information loss, generalization ratio, etc.)
- Calculating performance metrics (execution time, records per second, etc.)
- Calculating privacy metrics (re-identification risk, etc.)

**REQ-COMMON-002 [MUST]** The metric utilities must:
- Handle null values gracefully
- Support both numeric and categorical data
- Process large datasets efficiently
- Return well-structured metric dictionaries

### 9.2 Validation Utilities

**REQ-COMMON-003 [MUST]** The `validation_utils.py` module must provide functions for:
- Validating field existence in DataFrames
- Validating field types (numeric, categorical, date, etc.)
- Validating operation parameters
- Handling null values according to the null_strategy parameter

**REQ-COMMON-004 [MUST]** The validation utilities must:
- Raise specific exceptions with clear error messages
- Handle edge cases gracefully
- Support both single-field and multi-field validation
- Validate field types appropriately

### 9.3 Processing Utilities

**REQ-COMMON-005 [MUST]** The `processing_utils.py` module must provide functions for:
- Processing data in chunks for large datasets
- Processing data in parallel when enabled
- Generating output field names based on the operation mode
- Processing null values according to the null_strategy parameter
- Implementing specific processing strategies (binning, rounding, range, etc.)

**REQ-COMMON-006 [MUST]** The processing utilities must:
- Support both pandas and Dask DataFrames
- Handle large datasets efficiently
- Report progress during long-running operations
- Handle errors gracefully

### 9.4 Visualization Utilities

**REQ-COMMON-007 [MUST]** The `visualization_utils.py` module must provide functions for:
- Preparing data for visualization
- Generating visualization filenames
- Creating visualization directories
- Registering visualization artifacts
- Calculating optimal visualization parameters (bins, etc.)
- Sampling large datasets for visualization

**REQ-COMMON-008 [MUST]** The visualization utilities must:
- Support different visualization types (histograms, bar plots, etc.)
- Handle large datasets efficiently
- Generate consistent visualizations across operations
- Save visualizations in standard formats (PNG, etc.)

## 10. Standard Metrics and Visualizations

### 10.1 Universal Metrics

**REQ-ANON-013 [MUST]** All anonymization operations must include these base metrics:

| Metric | Description |
|---|---|
| total_records_processed | Total number of records processed |
| execution_time_seconds | Total execution time in seconds |
| records_per_second | Processing throughput rate |
| unique_values_before | Number of unique values before anonymization |
| unique_values_after | Number of unique values after anonymization |
| generalization_ratio | Reduction in unique values (1 - unique_after/unique_before) |
| null_count | Number of null values in the original data |

### 10.2 JSON Metrics Format

**REQ-ANON-014 [MUST]** All operations must save metrics in a consistent JSON format to `{task_dir}/metrics/{field}_{operation}_{timestamp}.json` with the following structure:

```json
{
  "operation_type": "NumericGeneralizationOperation",
  "field_name": "income",
  "mode": "REPLACE",
  "null_strategy": "PRESERVE",
  "total_records": 10000,
  "null_count": 123,
  "unique_values_before": 4567,
  "unique_values_after": 10,
  "generalization_ratio": 0.9978,
  "mean_original": 45678.90,
  "mean_anonymized": 45700.00,
  "std_original": 12345.67,
  "std_anonymized": 12000.00,
  "strategy": "binning",
  "bin_count": 10,
  "average_records_per_bin": 1000,
  "execution_time_seconds": 2.34,
  "processing_date": "2025-05-04T14:32:10"
}
```

**REQ-ANON-015 [MUST]** The metrics JSON must:
1. Have a consistent structure across operations
2. Include all base metrics plus operation-specific metrics
3. Use standardized naming conventions
4. Include metadata about the operation
5. Be written using the `DataWriter.write_metrics()` method to ensure consistency

### 10.3 Visualization Standards

**REQ-ANON-016 [MUST]** All visualizations must:

1. Use the PAMOLA.CORE visualization utilities (never direct plotting)
2. Follow consistent size and styling (800x600 pixels, PNG format)
3. Have clear titles and labels 
4. Focus on showing "before and after" effects of anonymization
5. Be saved with consistent naming: `{field}_{visualization_type}_{operation_core}_{timestamp}.png`
6. Be created through the `_generate_visualizations()` method
7. Be registered with both the result object and the reporter

**REQ-ANON-017 [MUST]** Operations must use these primary visualization types based on field type:

- **Numeric fields** → Histograms, box plots
- **Categorical fields** → Bar charts, pie charts
- **Suppression** → Heatmaps, bar charts
- **Masking** → Information indicators
- **Noise addition** → Overlay density plots
- **Pseudonymization** → Distribution graphs

## 11. Performance and Scalability Requirements

### 11.1 Large Data Processing

**REQ-ANON-018 [MUST]** All operations must support processing large datasets:

1. Process data in chunks using the `batch_size` parameter
2. Use `process_in_chunks` from `commons.processing_utils` for datasets that exceed memory limits
3. Support parallel processing when `parallel_processes > 1` is specified
4. Report progress during long-running operations

**REQ-ANON-019 [SHOULD]** Operations should support Dask integration for very large datasets:

1. Process Dask DataFrames when provided
2. Use Dask-aware functions for distributed processing
3. Support lazy evaluation for memory-efficient processing

### 11.2 Performance Metrics

**REQ-ANON-020 [MUST]** All operations must track and report performance metrics:

1. Execution time in seconds
2. Records processed per second
3. Memory usage (peak memory usage if available)
4. Processing stages timing breakdown

### 11.3 Memory Management

**REQ-ANON-021 [MUST]** All operations must implement proper memory management:

1. Release temporary data structures after use
2. Implement a `_cleanup_memory()` method to free resources
3. Process data in chunks to limit memory usage
4. Use generators for memory-efficient processing

## 12. Security and Privacy Requirements

### 12.1 Encryption

**REQ-ANON-022 [MUST]** All operations must support output encryption:

1. Enable encryption when `use_encryption=True` is specified
2. Use the provided `encryption_key` for output file encryption
3. Encrypt sensitive outputs (data, metrics, etc.)
4. Report encryption status in the operation result

### 12.2 Privacy Metrics

**REQ-ANON-023 [SHOULD]** Operations should calculate privacy metrics where applicable:

1. Re-identification risk metrics
2. Information loss metrics
3. Privacy guarantee metrics (k-anonymity, ε-differential privacy, etc.)
4. Utility-privacy trade-off metrics

### 12.3 Sensitive Data Handling

**REQ-ANON-024 [MUST]** All operations must handle sensitive data securely:

1. Never log sensitive data
2. Clear sensitive data from memory after use
3. Use secure random number generation for privacy-preserving operations
4. Validate privacy guarantees when applicable

## 13. Testing Requirements

### 13.1 Test Case Coverage

**REQ-ANON-025 [MUST]** All operations must have these test categories:

- **Basic functionality**: Test with simple datasets
- **Edge cases**: Test with empty datasets, all null values, extreme values
- **Validation**: Test parameter validation and error conditions
- **Large data**: Test with larger datasets to verify chunking
- **Caching**: Test both cache hit and miss scenarios

### 13.2 Testing Checklist

**REQ-ANON-026 [MUST]** Tests must verify:

- Field validation works correctly (field exists, type checking)
- Null handling strategies (PRESERVE, EXCLUDE, ERROR) work as expected
- Both operation modes (REPLACE, ENRICH) function properly
- All metrics are calculated correctly
- Visualizations are generated properly
- Errors are reported clearly
- Progress tracking is updated appropriately
- Cache mechanism works as expected

## 14. Future Extensions

While not required in the initial implementation, the anonymization framework should be designed to support these future extensions:

### 14.1 Workflow Visualization

**REQ-ANON-027 [MAY]** The framework may support workflow visualization:

1. Generate a Mermaid diagram of the operation workflow
2. Visualize dependencies between operations
3. Integrate with the broader operation framework's workflow visualization

### 14.2 Distributed Processing

**REQ-ANON-028 [MAY]** The framework may support distributed processing:

1. Support distributed processing of large datasets
2. Integrate with cloud-based processing frameworks
3. Implement distributed caching mechanisms

### 14.3 Privacy Model Integration

**REQ-ANON-029 [MAY]** The framework may support privacy model integration:

1. Integrate with privacy models (k-anonymity, differential privacy)
2. Automatically tune parameters based on privacy requirements
3. Validate privacy guarantees

## 15. Implementation Sequence

The recommended implementation order for anonymization operations:

1. Common utilities in the commons/ package
2. Base `AnonymizationOperation` class
3. Generalization operations
4. Masking operations
5. Noise addition operations
6. Pseudonymization operations
7. Suppression operations

## 16. Conclusion

This specification provides a comprehensive guide for implementing anonymization operations within the PAMOLA.CORE framework. By following these guidelines, developers can create consistent, robust, and well-integrated operations that provide:

1. Standardized interfaces for executing privacy operations
2. Comprehensive metrics for evaluating anonymization effectiveness
3. Clear visualizations for understanding data transformations
4. Efficient processing of large datasets
5. Robust error handling and reporting
6. Effective caching for improved performance
7. Thorough documentation for maintainability

The anonymization package is a central component of PAMOLA.CORE's privacy-enhancing capabilities, providing tools for real-world data anonymization with measurable privacy and utility characteristics.