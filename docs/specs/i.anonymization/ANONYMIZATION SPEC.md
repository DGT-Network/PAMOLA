# PAMOLA.CORE Anonymization Package Software Requirements Specification

**Document Version:** 4.1.0  
**Last Updated:** 2025-06-15  
**Status:** Release Candidate

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document defines the requirements for the PAMOLA.CORE anonymization package. The package implements privacy-enhancing operations as atomic, self-contained processes that transform data while preserving analytical utility.

### 1.2 Scope

The anonymization package provides operations organized into five core categories:
- **Generalization**: Reducing data precision while maintaining utility
- **Noise Addition**: Adding statistical noise for differential privacy
- **Suppression**: Removing sensitive values or records
- **Masking**: Hiding parts of data values
- **Pseudonymization**: Replacing identifiers with consistent substitutes

Each operation is implemented as an independent, atomic process following the PAMOLA.CORE Operations Framework.

### 1.3 Document Conventions

- **REQ-ANON-XXX**: General anonymization requirements
- **REQ-GEN-XXX**: Generalization operation requirements
- **REQ-NOISE-XXX**: Noise addition requirements
- **REQ-SUPP-XXX**: Suppression requirements
- **REQ-MASK-XXX**: Masking requirements
- **REQ-PSEUDO-XXX**: Pseudonymization requirements
- **REQ-COMMONS-XXX**: Commons sub-framework requirements

Priority levels:
- **MUST**: Essential requirement (MVP)
- **SHOULD**: Important but not essential
- **MAY**: Optional enhancement

## 2. Core Architecture Principles

### 2.1 Operation Contract

**REQ-ANON-001 [MUST]** Every anonymization operation is an atomic process that:
- Inherits from `AnonymizationOperation` (which inherits from `BaseOperation`)
- Implements only the transformation logic (no quality assessment, no final reporting)
- Uses only framework-provided services for I/O, progress tracking, and result reporting
- Returns process metrics only (effectiveness, performance, basic statistics)
- Supports both pandas and Dask for large-scale processing

**REQ-ANON-002 [MUST]** Operations SHALL NOT:
- Implement their own file I/O (must use `DataWriter`)
- Create custom progress tracking (must use `ProgressTracker` or `HierarchicalProgressTracker`)
- Define custom result structures (must use `OperationResult`)
- Perform final quality assessment or risk evaluation
- Integrate directly with profiling or other packages
- Store state between executions
- Override `run()` method from base class

### 2.2 Framework Integration Requirements

**REQ-ANON-003 [MUST]** All operations must use these framework components:

| Component | Purpose | Required Methods/Usage |
|-----------|---------|----------------------|
| `DataWriter` | All file output | `write_dataframe()`, `write_metrics()`, `write_json()` |
| `DataSource` | All data input | `get_dataframe()`, `get_file_path()` |
| `OperationResult` | Result reporting | `add_metric()`, `add_artifact()`, `add_nested_metric()` |
| `ProgressTracker` | Progress updates | `update()`, `create_subtask()` |
| `HierarchicalProgressTracker` | Complex progress | For operations with multiple phases |
| `OperationConfig` | Configuration | Schema validation, `to_dict()`, `save()` |

**REQ-ANON-004 [MUST]** All operations must use utilities from:
- `pamola_core.utils.ops.op_field_utils` for field naming and conditions
- `pamola_core.utils.ops.op_data_processing` for memory optimization and chunking
- `pamola_core.anonymization.commons.metric_utils` for standardized metrics
- `pamola_core.anonymization.commons.privacy_metric_utils` for privacy indicators
- `pamola_core.anonymization.commons.validation_utils` for input validation
- `pamola_core.utils.logging` for logging (module-specific loggers)

### 2.3 Package Structure

**REQ-ANON-005 [MUST]** The package structure SHALL be:

```
pamola_core/anonymization/
├── __init__.py
├── base_anonymization_op.py    # Base class for all operations
├── commons/                    # Shared utilities sub-framework
│   ├── __init__.py
│   ├── validation/            # Modular validation subsystem
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── decorators.py
│   │   ├── exceptions.py
│   │   ├── field_validators.py
│   │   ├── file_validators.py
│   │   ├── strategy_validators.py
│   │   └── type_validators.py
│   ├── categorical_config.py   # Category-specific configuration
│   ├── categorical_strategies.py # Category processing strategies
│   ├── category_utils.py       # Category analysis utilities
│   ├── data_utils.py          # Privacy-aware data processing
│   ├── hierarchy_dictionary.py # Hierarchical generalization support
│   ├── metric_utils.py        # Lightweight metrics
│   ├── privacy_metric_utils.py # Privacy process metrics
│   ├── text_processing_utils.py # Text normalization
│   ├── validation_utils.py    # Legacy validation facade
│   └── visualization_utils.py  # Chart generation wrappers
├── generalization/            
│   ├── __init__.py
│   ├── numeric_op.py          # Numeric generalization
│   ├── categorical_op.py      # Categorical generalization
│   └── datetime_op.py         # DateTime generalization
├── noise/
│   ├── __init__.py
│   ├── gaussian_op.py         # Gaussian noise
│   ├── laplace_op.py          # Laplace noise
│   └── uniform_op.py          # Uniform noise
├── suppression/
│   ├── __init__.py
│   ├── cell_op.py             # Cell-level suppression
│   ├── record_op.py           # Record-level suppression
│   └── attribute_op.py        # Attribute suppression
├── masking/
│   ├── __init__.py
│   ├── full_op.py             # Full field masking
│   └── partial_op.py          # Partial masking
└── pseudonymization/
    ├── __init__.py
    ├── hash_op.py             # Hash-based pseudonymization
    └── mapping_op.py          # Dictionary mapping
```

## 3. Base Operation Requirements

### 3.1 AnonymizationOperation Base Class

**REQ-ANON-006 [MUST]** All operations SHALL inherit from `AnonymizationOperation` which provides:
- Lifecycle management (`execute()` method with standardized phases)
- Memory optimization and adaptive batch sizing
- Risk-based processing with k-anonymity integration
- Automatic Dask switching for large datasets
- Progress tracking integration
- Metric collection and visualization generation
- Error handling and recovery

### 3.2 Constructor Interface

**REQ-ANON-007 [MUST]** All operations SHALL have this constructor signature:

```python
def __init__(self,
             field_name: str,                    # Field to process
             mode: str = "REPLACE",              # REPLACE or ENRICH
             output_field_name: Optional[str] = None,
             column_prefix: str = "_",
             null_strategy: str = "PRESERVE",    # PRESERVE, EXCLUDE, ERROR, ANONYMIZE
             batch_size: int = 10000,
             use_cache: bool = True,
             use_encryption: bool = False,
             encryption_key: Optional[Union[str, Path]] = None,
             condition_field: Optional[str] = None,
             condition_values: Optional[List] = None,
             condition_operator: str = "in",
             multi_conditions: Optional[List[Dict[str, Any]]] = None,
             multi_condition_logic: str = "AND",
             ka_risk_field: Optional[str] = None,
             risk_threshold: float = 5.0,
             vulnerable_record_strategy: str = "suppress",
             optimize_memory: bool = True,
             adaptive_batch_size: bool = True,
             engine: str = "auto",              # pandas, dask, or auto
             max_rows_in_memory: int = 1000000,
             dask_chunk_size: str = "100MB",
             dask_npartitions: Optional[int] = None,
             **operation_specific_params):       # Strategy-specific parameters
```

### 3.3 Required Method Implementations

**REQ-ANON-008 [MUST]** All operations must implement:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process a batch of data - core transformation logic."""
    
def _get_cache_parameters(self) -> Dict[str, Any]:
    """Return operation-specific parameters for cache key."""
    
def _collect_specific_metrics(self, 
                             original_data: pd.Series,
                             anonymized_data: pd.Series) -> Dict[str, Any]:
    """Collect operation-specific metrics (optional override)."""
```

**REQ-ANON-009 [SHOULD]** Operations supporting Dask should implement:

```python
def _process_batch_dask(self, ddf: dd.DataFrame) -> dd.DataFrame:
    """Process Dask DataFrame for distributed computing."""
```

**REQ-ANON-010 [MUST]** Operations may override `execute()` ONLY when:
- Advanced Dask integration is required beyond base class capabilities
- Custom metric collection or visualization is needed
- Complex multi-phase processing requires different lifecycle

**When overriding `execute()`, operations MUST:**
1. Call `super().execute()` at the beginning or implement ALL base class phases
2. Maintain the exact same method signature
3. Return the same `OperationResult` structure
4. Use DataWriter for ALL file operations
5. Use ProgressTracker for ALL progress updates
6. Document why override is necessary
7. Preserve all base class functionality

**Example of valid execute() override:**
```python
def execute(self, data_source: DataSource, task_dir: Path, 
            reporter: Optional[Any] = None) -> OperationResult:
    """
    Override to add custom Dask optimization for large hierarchies.
    
    Justification: Base class Dask handling doesn't support 
    hierarchical dictionary partitioning needed for this operation.
    """
    # Option 1: Extend base functionality
    result = super().execute(data_source, task_dir, reporter)
    # Add custom processing here
    return result
    
    # Option 2: Reimplement with all required phases
    # Must include ALL phases from base class
```

**Operations SHALL NOT override:**
- `run()` - Use framework implementation
- File I/O methods - Use DataWriter
- `_load_and_optimize_data()` - Use base class implementation (unless documented reason)
- `_apply_conditional_filtering()` - Use base class implementation (unless documented reason)

### 3.4 Execution Lifecycle

**REQ-ANON-011 [MUST]** The base class `execute()` method provides these phases:

1. **Configuration Saving**: Save operation config to task_dir
2. **Data Loading & Optimization**: Load data and optimize memory usage
3. **Engine Selection**: Determine pandas vs Dask based on data size
4. **Output Field Preparation**: Generate output field name
5. **Conditional Filtering**: Apply conditions and risk-based filtering
6. **Data Processing**: Call `process_batch()` with chunking
7. **Vulnerable Record Handling**: Process high-risk records
8. **Metrics Collection**: Gather effectiveness and privacy metrics
9. **Visualization Generation**: Create comparison charts
10. **Output Writing**: Save transformed data via DataWriter
11. **Memory Cleanup**: Force garbage collection for large datasets

### 3.5 Metric Requirements

**REQ-ANON-012 [MUST]** All operations must collect these standard metrics:

| Metric Category | Required Metrics |
|----------------|------------------|
| Process | `records_processed`, `execution_time`, `batch_count`, `processing_engine` |
| Effectiveness | `unique_values_before`, `unique_values_after`, `information_loss` |
| Performance | `memory_usage_mb`, `processing_rate_records_per_sec` |
| Errors | `error_count`, `null_count`, `invalid_values` |
| Privacy | `suppression_rate`, `generalization_level`, `disclosure_risk` (if QIs provided) |

## 4. Commons Sub-Framework Requirements

### 4.1 Purpose and Design

**REQ-COMMONS-001 [MUST]** The commons sub-framework provides:
- Privacy-specific utilities extending general framework capabilities
- Lightweight, fast operations suitable for real-time processing
- Standardized validation, metrics, and visualization
- Consistent error handling across all anonymization operations

### 4.2 Validation Requirements

**REQ-COMMONS-002 [MUST]** Operations must use the modular validation system:

```python
from pamola_core.anonymization.commons.validation import (
    FieldExistsValidator,
    NumericFieldValidator,
    CategoricalFieldValidator,
    ValidationResult
)

# Example usage
validator = NumericFieldValidator(min_value=0, max_value=100)
result = validator.validate(df[field_name])
if not result.is_valid:
    raise FieldValidationError(result.errors[0])
```

**REQ-COMMONS-003 [SHOULD]** Use validation decorators for method validation:

```python
from pamola_core.anonymization.commons.validation import (
    validation_handler,
    requires_field,
    validate_types
)

@validation_handler()
@requires_field('field_name')
@validate_types(strategy=str, bin_count=int)
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    # Implementation
```

### 4.3 Metric Collection Requirements

**REQ-COMMONS-004 [MUST]** Use standardized metric collection:

```python
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
    calculate_anonymization_effectiveness,
    calculate_categorical_information_loss
)

# Collect all metrics in one call
metrics = collect_operation_metrics(
    operation_type="suppression",
    original_data=original_series,
    processed_data=processed_series,
    operation_params=self.strategy_params,
    timing_info=timing_info
)
```

**REQ-COMMONS-005 [MUST]** Include privacy metrics when quasi-identifiers are available:

```python
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_batch_metrics,
    check_anonymization_thresholds,
    get_process_summary
)

if self.quasi_identifiers:
    privacy_metrics = calculate_batch_metrics(
        original_batch=original_df,
        anonymized_batch=anonymized_df,
        field_name=self.field_name,
        quasi_identifiers=self.quasi_identifiers
    )
```

### 4.4 Data Processing Requirements

**REQ-COMMONS-006 [MUST]** Use commons data utilities for privacy-aware processing:

```python
from pamola_core.anonymization.commons.data_utils import (
    process_nulls,
    filter_records_conditionally,
    handle_vulnerable_records,
    create_risk_based_processor
)

# Handle nulls with privacy strategy
processed = process_nulls(series, strategy="ANONYMIZE", anonymize_value="SUPPRESSED")
```

### 4.5 Visualization Requirements

**REQ-COMMONS-007 [MUST]** Use visualization utilities, never direct plotting:

```python
from pamola_core.anonymization.commons.visualization_utils import (
    create_comparison_visualization,
    create_category_distribution_comparison,
    register_visualization_artifact
)

# Create visualization
vis_path = create_comparison_visualization(
    original_data, anonymized_data, task_dir, 
    field_name, operation_name
)

# Register with result
register_visualization_artifact(
    result, reporter, vis_path, field_name,
    "comparison", "Before/after comparison"
)
```

## 5. Operation Categories

### 5.1 Generalization Operations

**REQ-GEN-001 [MUST]** Generalization operations reduce data precision:

#### 5.1.1 Numeric Generalization

```python
class NumericGeneralizationOperation(AnonymizationOperation):
    """
    Strategies: binning, rounding, range
    Parameters:
    - bin_count: int (for binning)
    - binning_method: str (equal_width, equal_frequency, quantile)
    - precision: int (for rounding)
    - range_limits: List[Tuple[float, float]] (for range)
    """
```

Required metrics:
- `bin_distribution` or `range_distribution`
- `precision_loss`
- `outliers_handled`

#### 5.1.2 Categorical Generalization

```python
class CategoricalGeneralizationOperation(AnonymizationOperation):
    """
    Strategies: merge_low_freq, hierarchy, frequency_based
    Parameters:
    - min_group_size: int
    - hierarchy_file: Optional[str] (external dictionary)
    - merge_threshold: float
    - text_normalization: str (none, basic, advanced, aggressive)
    - fuzzy_matching: bool
    """
```

**REQ-GEN-002 [MUST]** Support external hierarchy files (JSON/CSV format)
**REQ-GEN-003 [SHOULD]** Use categorical_strategies module for implementation

#### 5.1.3 DateTime Generalization

```python
class DateTimeGeneralizationOperation(AnonymizationOperation):
    """
    Strategies: truncate, shift, generalize
    Parameters:
    - granularity: str (year, month, day, hour)
    - shift_days: int (for shifting)
    - timezone_handling: str
    """
```

### 5.2 Noise Addition Operations

**REQ-NOISE-001 [MUST]** Noise operations add calibrated random noise:

#### 5.2.1 Gaussian Noise

```python
class GaussianNoiseOperation(AnonymizationOperation):
    """
    Parameters:
    - noise_level: float (standard deviation)
    - bounds: Optional[Tuple[float, float]]
    - clip_values: bool
    - seed: Optional[int] (for reproducibility)
    """
```

Required metrics:
- `noise_statistics` (mean, std of added noise)
- `clipped_values_count`
- `signal_to_noise_ratio`

#### 5.2.2 Laplace Noise

```python
class LaplaceNoiseOperation(AnonymizationOperation):
    """
    Parameters:
    - epsilon: float (privacy parameter)
    - sensitivity: float
    - bounds: Optional[Tuple[float, float]]
    - mechanism: str (global, local)
    """
```

### 5.3 Suppression Operations

**REQ-SUPP-001 [MUST]** Suppression operations remove or replace values:

#### 5.3.1 Cell Suppression

```python
class CellSuppressionOperation(AnonymizationOperation):
    """
    Parameters:
    - suppression_value: Any (default: None)
    - condition_threshold: Optional[float]
    - suppression_marker: str (default: "SUPPRESSED")
    - preserve_type: bool (maintain data type after suppression)
    """
```

Required metrics:
- `cells_suppressed`
- `suppression_rate`
- `condition_matches`

#### 5.3.2 Record Suppression

```python
class RecordSuppressionOperation(AnonymizationOperation):
    """
    Parameters:
    - risk_threshold: float
    - multi_field_conditions: List[Dict]
    - keep_suppressed_records: bool (for analysis)
    - suppression_log_path: Optional[str]
    """
```

#### 5.3.3 Attribute Suppression

```python
class AttributeSuppressionOperation(AnonymizationOperation):
    """
    Parameters:
    - attributes_to_suppress: List[str]
    - conditional_suppression: Dict[str, Any]
    - replacement_strategy: str (null, default, calculated)
    """
```

### 5.4 Masking Operations

**REQ-MASK-001 [MUST]** Masking operations hide parts of values:

#### 5.4.1 Full Masking

```python
class FullMaskingOperation(AnonymizationOperation):
    """
    Parameters:
    - mask_char: str (default: "*")
    - preserve_length: bool
    - mask_pattern: str (full, random)
    - consistent_masking: bool (same value → same mask)
    """
```

#### 5.4.2 Partial Masking

```python
class PartialMaskingOperation(AnonymizationOperation):
    """
    Parameters:
    - mask_char: str (default: "*")
    - mask_pattern: Optional[str] (regex)
    - preserve_prefix: int
    - preserve_suffix: int
    - preserve_format: bool (maintain separators)
    - format_patterns: Dict[str, str] (e.g., phone, email)
    """
```

### 5.5 Pseudonymization Operations

**REQ-PSEUDO-001 [MUST]** Pseudonymization replaces identifiers consistently:

#### 5.5.1 Hash Pseudonymization

```python
class HashPseudonymizationOperation(AnonymizationOperation):
    """
    Parameters:
    - hash_algorithm: str (sha256, blake2b, sha3_256)
    - salt: Optional[str]
    - truncate_length: Optional[int]
    - use_hmac: bool
    - encoding: str (hex, base64, base32)
    """
```

Required artifacts:
- No mapping dictionary (one-way function)
- Salt configuration (if used)

#### 5.5.2 Mapping Pseudonymization

```python
class MappingPseudonymizationOperation(AnonymizationOperation):
    """
    Parameters:
    - mapping_strategy: str (sequential, random, custom, uuid)
    - preserve_uniqueness: bool
    - external_mapping: Optional[str] (file path)
    - format_template: Optional[str] (e.g., "USER_{:06d}")
    - collision_handling: str (error, regenerate, append)
    """
```

Required artifacts:
- Mapping dictionary saved to `{task_dir}/mappings/`
- Collision report if any occurred

## 6. Integration Requirements

### 6.1 DataWriter Integration

**REQ-ANON-013 [MUST]** All file outputs use DataWriter:

```python
# In execute() method (handled by base class):
writer = DataWriter(task_dir=task_dir, logger=self.logger)

# Write transformed data
output_result = writer.write_dataframe(
    df=result_df,
    name="anonymized_data",
    format=self.output_format,
    subdir="output",
    timestamp_in_name=True,
    encryption_key=self.encryption_key if self.use_encryption else None
)

# Write metrics
metrics_result = writer.write_metrics(
    metrics=operation_metrics,
    name=f"{self.field_name}_{self.__class__.__name__}",
    timestamp_in_name=True
)

# Write JSON artifacts
mapping_result = writer.write_json(
    data=mapping_dict,
    name="pseudonym_mapping",
    subdir="mappings",
    timestamp_in_name=True
)
```

### 6.2 Progress Tracking

**REQ-ANON-014 [MUST]** Use HierarchicalProgressTracker:

```python
# Base class provides progress tracker
# Operations can create subtasks
batch_progress = progress_tracker.create_subtask(
    total=total_batches,
    description="Processing batches",
    unit="batches"
)

# Update within processing loop
batch_progress.update(1, {
    "batch": current_batch,
    "records_processed": processed_count,
    "percentage": (processed_count / total_records) * 100
})
```

### 6.3 Field Utilities Usage

**REQ-ANON-015 [MUST]** Use field utilities for:

```python
from pamola_core.utils.ops.op_field_utils import (
    generate_output_field_name,
    apply_condition_operator,
    create_field_mask,
    create_multi_field_mask
)

# Generate output field name (handled by base class)
output_field = generate_output_field_name(
    self.field_name, 
    self.mode, 
    self.output_field_name,
    operation_suffix="anonymized"
)

# Apply conditions (handled by base class)
mask = apply_condition_operator(
    df[self.condition_field],
    self.condition_values,
    self.condition_operator
)
```

### 6.4 Error Handling

**REQ-ANON-016 [MUST]** Standardized error handling:

```python
# In process_batch():
try:
    # Validate input
    if not validate_field_exists(batch, self.field_name):
        raise FieldNotFoundError(self.field_name, list(batch.columns))
    
    # Process batch
    result = self._apply_transformation(batch)
    
except ValueError as e:
    self.logger.error(f"Invalid value in batch: {e}")
    if self.error_handling == "fail":
        raise
    elif self.error_handling == "skip":
        return batch  # Return unmodified
    else:  # log
        self._error_batches.append({
            "batch_id": batch_id,
            "error": str(e),
            "indices": batch.index.tolist()[:100]
        })
```

## 7. Performance and Scalability

### 7.1 Memory Management

**REQ-ANON-017 [MUST]** Operations must support:
- Adaptive batch sizing based on available memory
- Automatic DataFrame dtype optimization
- Explicit garbage collection for large datasets
- Memory usage tracking and reporting

### 7.2 Dask Integration

**REQ-ANON-018 [SHOULD]** Large data support:
- Automatic switching to Dask when rows > max_rows_in_memory
- Configurable chunk size and partitions
- Lazy evaluation for memory efficiency
- Progress tracking across partitions

### 7.3 Caching

**REQ-ANON-019 [SHOULD]** Cache support includes:
- Operation result caching based on parameters
- Dictionary/mapping caching for repeated operations
- Cache key generation via `_get_cache_parameters()`
- LRU eviction for memory management

## 8. Testing Requirements

**REQ-ANON-020 [MUST]** Each operation must have:

1. **Unit tests** covering:
   - All parameter combinations
   - Null handling strategies (PRESERVE, EXCLUDE, ERROR, ANONYMIZE)
   - Error conditions and edge cases
   - Mode switching (REPLACE/ENRICH)
   - Small and large data scenarios

2. **Integration tests** verifying:
   - DataWriter integration
   - Progress tracking across phases
   - Metric collection completeness
   - Visualization generation
   - Cache functionality

3. **Performance tests** checking:
   - Memory usage under load
   - Processing rate benchmarks
   - Batch size optimization
   - Dask switching thresholds

4. **Privacy tests** validating:
   - Effectiveness metrics accuracy
   - Privacy metric calculations
   - Risk-based processing logic
   - Vulnerable record handling

## 9. State Management

**REQ-ANON-021 [MUST]** Operations must be stateless between executions:
- No persistent state between `execute()` calls
- Thread-safe for concurrent execution
- State reset after each execution
- Clean up all temporary resources

**REQ-ANON-022 [SHOULD]** Operations should implement:
```python
def reset_state(self):
    """Reset operation state after execution."""
    # Clear any cached data
    # Reset batch metrics
    # Clear error logs
```

## 10. Artifact Generation

**REQ-ANON-023 [MUST]** Standard artifacts include:

| Artifact Type | Location | Description |
|--------------|----------|-------------|
| Transformed Data | `{task_dir}/output/` | Main output dataset |
| Process Metrics | `{task_dir}/metrics/` | JSON metrics file |
| Visualizations | `{task_dir}/visualizations/` | PNG comparison charts |
| Mappings | `{task_dir}/mappings/` | Pseudonymization mappings |
| Error Reports | `{task_dir}/errors/` | Batch processing errors |

## 11. Configuration Schema

**REQ-ANON-024 [MUST]** Each operation must define configuration schema:

```python
class OperationNameConfig(OperationConfig):
    """Configuration for specific operation."""
    
    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "strategy": {"type": "string", "enum": ["strategy1", "strategy2"]},
            # Strategy-specific parameters
        },
        "required": ["field_name", "strategy"],
        "allOf": [
            # Conditional requirements based on strategy
        ]
    }
```

## 12. Registration and Discovery

**REQ-ANON-025 [MUST]** Operations must be registered:

```python
from pamola_core.utils.ops.op_registry import register_operation

# At module level
register_operation(MyAnonymizationOperation)

# Factory function
def create_my_anonymization_operation(field_name: str, **kwargs):
    """Create operation instance with defaults."""
    return MyAnonymizationOperation(field_name=field_name, **kwargs)
```

## 13. Implementation Checklist

### 13.1 New Operation Checklist

**REQ-ANON-026 [MUST]** Every new operation must:

- [ ] Inherit from `AnonymizationOperation`
- [ ] Use DataWriter for ALL file operations
- [ ] Use base class lifecycle management (or document override reason)
- [ ] Implement `process_batch()` method
- [ ] Implement `_get_cache_parameters()` method
- [ ] Use commons validation utilities
- [ ] Use commons metric utilities
- [ ] Return standard metrics via base class
- [ ] Support REPLACE and ENRICH modes
- [ ] Handle null values according to strategy
- [ ] Support conditional processing
- [ ] Include comprehensive docstrings
- [ ] Add configuration schema class
- [ ] Register with operation registry
- [ ] Provide factory function
- [ ] Add unit tests with >90% coverage
- [ ] Support Dask for large datasets
- [ ] Implement state reset

### 13.2 Anti-Patterns to Avoid

**REQ-ANON-027 [MUST NOT]** Operations must NOT:

- [ ] Open files directly (use DataWriter)
- [ ] Print to console (use logger)
- [ ] Create custom progress bars
- [ ] Implement quality assessment
- [ ] Call profiling operations
- [ ] Store state between calls
- [ ] Override execute() without justification
- [ ] Implement custom caching logic
- [ ] Use matplotlib/plotly directly
- [ ] Calculate metrics manually
- [ ] Handle vulnerable records directly

## 14. Migration Guide

### 14.1 From Version 3.x to 4.x

Major changes:
1. **Commons validation**: Migrate from `validation_utils` functions to validation classes
2. **Metric collection**: Use `collect_operation_metrics()` instead of manual calculation
3. **Visualization**: Use commons visualization utilities exclusively
4. **Configuration**: Implement proper configuration schema classes
5. **Dask support**: Implement `_process_batch_dask()` for large data

### 14.2 Example Migration

```python
# Old approach (3.x)
from pamola_core.anonymization.commons.validation_utils import validate_field_exists
if not validate_field_exists(df, field_name):
    raise ValueError("Field not found")

# New approach (4.x)
from pamola_core.anonymization.commons.validation import check_field_exists
check_field_exists(df, field_name)  # Raises FieldNotFoundError
```

## 15. Commons Module Signatures

### 15.1 categorical_config.py

```python
class CategoricalGeneralizationConfig(OperationConfig):
    """Configuration for categorical generalization operations."""
    
    def __init__(self, field_name: str, strategy: str = "hierarchy", **kwargs):
        """Initialize configuration with validation."""
    
    def validate(self) -> bool:
        """Validate configuration against schema and business rules."""
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get parameters specific to the selected strategy."""
    
    def should_use_dask(self, row_count: int) -> bool:
        """Determine if Dask should be used based on data size."""
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CategoricalGeneralizationConfig':
        """Create configuration from dictionary."""
```

### 15.2 categorical_strategies.py

```python
def apply_hierarchy(series: pd.Series, config: Dict[str, Any], 
                   context: Dict[str, Any], logger: Optional[logging.Logger] = None) -> pd.Series:
    """Apply hierarchical generalization using external dictionary."""

def apply_merge_low_freq(series: pd.Series, config: Dict[str, Any], 
                        context: Dict[str, Any], logger: Optional[logging.Logger] = None) -> pd.Series:
    """Apply merging of low frequency categories."""

def apply_frequency_based(series: pd.Series, config: Dict[str, Any], 
                         context: Dict[str, Any], logger: Optional[logging.Logger] = None) -> pd.Series:
    """Apply frequency-based generalization."""

def apply_null_and_unknown_strategy(series: pd.Series, null_strategy: Union[str, NullStrategy],
                                   unknown_value: str = "OTHER", 
                                   rare_value_template: Optional[str] = None,
                                   context: Optional[Dict[str, Any]] = None,
                                   logger: Optional[logging.Logger] = None) -> pd.Series:
    """Apply NULL and unknown value handling strategy."""
```

### 15.3 category_mapping.py

```python
class CategoryMappingEngine:
    """Thread-safe engine for applying category mappings."""
    
    def __init__(self, unknown_value: str = "OTHER", unknown_template: Optional[str] = None,
                 cache_size: int = 10000):
        """Initialize the category mapping engine."""
    
    def add_mapping(self, original: str, replacement: str) -> None:
        """Add a simple value-to-category mapping."""
    
    def add_conditional_mapping(self, original: str, replacement: str,
                               condition: Dict[str, Any], priority: int = 0) -> None:
        """Add a conditional mapping rule."""
    
    def apply_to_series(self, series: pd.Series, 
                       context_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """Apply mappings to a pandas Series."""
    
    def get_mapping_dict(self) -> Dict[str, str]:
        """Get a copy of simple mappings dictionary."""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mapping engine statistics."""
```

### 15.4 category_utils.py

```python
def analyze_category_distribution(series: pd.Series, top_n: int = 20,
                                 min_frequency: int = 1, calculate_entropy: bool = True,
                                 calculate_gini: bool = True, calculate_concentration: bool = True,
                                 value_counts: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Comprehensive analysis of category distribution for anonymization planning."""

def identify_rare_categories(series: pd.Series, count_threshold: int = 10,
                            percent_threshold: float = 0.01, combined_criteria: bool = True,
                            value_counts: Optional[pd.Series] = None) -> Tuple[Set[str], Dict[str, Any]]:
    """Identify rare categories based on privacy risk criteria."""

def group_rare_categories(series: pd.Series, grouping_strategy: str = "single_other",
                         threshold: Union[int, float] = 10, max_groups: int = 100,
                         group_prefix: str = "GROUP_", preserve_top_n: Optional[int] = None,
                         other_label: str = "OTHER", 
                         value_counts: Optional[pd.Series] = None) -> Tuple[pd.Series, Dict[str, Any]]:
    """Group rare categories using privacy-preserving strategies."""

def validate_category_mapping(original: pd.Series, mapped: pd.Series,
                             mapping: Optional[Dict[str, str]] = None,
                             coverage_threshold: float = 0.95) -> Dict[str, Any]:
    """Validate category mapping for anonymization correctness."""
```

### 15.5 data_utils.py

```python
def process_nulls(series: pd.Series, strategy: str = "PRESERVE",
                 anonymize_value: str = "SUPPRESSED") -> pd.Series:
    """Process null values with privacy-aware strategies."""

def filter_records_conditionally(df: pd.DataFrame, risk_field: Optional[str] = None,
                               risk_threshold: float = 5.0, operator: str = "ge",
                               condition_field: Optional[str] = None,
                               condition_values: Optional[List] = None,
                               condition_operator: str = "in") -> Tuple[pd.DataFrame, pd.Series]:
    """Filter DataFrame records based on risk scores and optional conditions."""

def handle_vulnerable_records(df: pd.DataFrame, field_name: str,
                            vulnerability_mask: pd.Series, strategy: str = "suppress",
                            replacement_value: Optional[Any] = None) -> pd.DataFrame:
    """Handle vulnerable records identified by risk assessment."""

def create_risk_based_processor(strategy: str = "adaptive",
                              risk_threshold: float = 5.0) -> Callable:
    """Factory for creating risk-based processing functions."""

def create_privacy_level_processor(privacy_level: str = "MEDIUM") -> Dict[str, Any]:
    """Create a configuration for processing based on privacy level."""
```

### 15.6 hierarchy_dictionary.py

```python
class HierarchyDictionary:
    """Manages hierarchical mappings for categorical generalization."""
    
    def __init__(self):
        """Initialize empty hierarchy dictionary."""
    
    def load_from_file(self, filepath: Union[str, Path], format_type: str = 'auto',
                      encryption_key: Optional[str] = None) -> None:
        """Load hierarchy dictionary from file."""
    
    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load hierarchy from a pandas DataFrame."""
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load hierarchy from a dictionary."""
    
    @lru_cache(maxsize=10000)
    def get_hierarchy(self, value: str, level: int = 1, normalize: bool = True) -> Optional[str]:
        """Get generalized value at specified hierarchy level."""
    
    def validate_structure(self) -> Tuple[bool, List[str]]:
        """Validate dictionary structure and consistency."""
    
    def get_coverage(self, values: List[str], normalize: bool = True) -> Dict[str, Any]:
        """Calculate dictionary coverage for a set of values."""
    
    def get_all_values_at_level(self, level: int) -> Set[str]:
        """Get all unique values at a specific hierarchy level."""
```

### 15.7 metric_utils.py

```python
def calculate_anonymization_effectiveness(original_series: pd.Series,
                                        anonymized_series: pd.Series) -> Dict[str, float]:
    """Calculate basic effectiveness metrics for anonymization."""

def calculate_generalization_metrics(original_series: pd.Series, anonymized_series: pd.Series,
                                   strategy: str, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate metrics specific to generalization strategies."""

def calculate_categorical_information_loss(original_series: pd.Series, anonymized_series: pd.Series,
                                         category_mapping: Optional[Dict[str, str]] = None,
                                         hierarchy_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Calculate information loss metrics for categorical generalization."""

def calculate_generalization_height(original_values: Union[pd.Series, List[str]],
                                  generalized_values: Union[pd.Series, List[str]],
                                  hierarchy_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Calculate the generalization height in a hierarchy."""

def collect_operation_metrics(operation_type: str, original_data: pd.Series,
                            processed_data: pd.Series, operation_params: Dict[str, Any],
                            timing_info: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Collect all relevant metrics for an anonymization operation."""

def save_process_metrics(metrics: Dict[str, Any], task_dir: Path, operation_name: str,
                       field_name: str, writer: Optional[DataWriter] = None) -> Optional[Path]:
    """Save process metrics to file."""
```

### 15.8 privacy_metric_utils.py

```python
def calculate_anonymization_coverage(original: pd.Series, anonymized: pd.Series) -> Dict[str, float]:
    """Calculate the coverage of anonymization process."""

def calculate_suppression_rate(series: pd.Series, original_nulls: Optional[int] = None) -> float:
    """Calculate the suppression rate in anonymized data."""

def get_group_size_distribution(df: pd.DataFrame, quasi_identifiers: List[str],
                              max_groups: int = 100) -> Dict[str, Any]:
    """Get quick distribution of group sizes for quasi-identifiers."""

def calculate_min_group_size(df: pd.DataFrame, quasi_identifiers: List[str],
                           sample_size: Optional[int] = 10000) -> int:
    """Calculate minimum group size (k) for quasi-identifiers."""

def calculate_vulnerable_records_ratio(df: pd.DataFrame, quasi_identifiers: List[str],
                                     k_threshold: int = 5, sample_size: Optional[int] = 10000) -> float:
    """Calculate ratio of vulnerable records (k < threshold)."""

def check_anonymization_thresholds(metrics: Dict[str, float],
                                 thresholds: Optional[Dict[str, float]] = None) -> Dict[str, bool]:
    """Check if anonymization metrics meet specified thresholds."""

def calculate_batch_metrics(original_batch: pd.DataFrame, anonymized_batch: pd.DataFrame,
                          field_name: str, quasi_identifiers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate all process metrics for a batch."""
```

### 15.9 text_processing_utils.py

```python
def normalize_text(text: str, level: str = "basic", preserve_case: bool = False) -> str:
    """Normalize text for anonymization purposes."""

def clean_category_name(name: str, max_length: int = 40,
                       invalid_chars: str = r'[^a-zA-Z0-9\s\-_]',
                       separator: str = "_") -> str:
    """Clean category names for safe anonymization field naming."""

def find_closest_category(value: str, categories: List[str], threshold: float = 0.8,
                         method: str = "ratio", normalize_value: bool = True,
                         fallback: str = "OTHER") -> str:
    """Find the best matching category with anonymization-specific handling."""

def prepare_value_for_matching(value: str, remove_common_prefixes: bool = True,
                              remove_common_suffixes: bool = True) -> str:
    """Prepare a value for category matching by removing common prefixes/suffixes."""
```

### 15.10 validation_utils.py

```python
# Facade module - re-exports from validation subpackage
from .validation import (
    ValidationResult,
    BaseValidator,
    NumericFieldValidator,
    CategoricalFieldValidator,
    DateTimeFieldValidator,
    create_field_validator,
    validation_handler,
    requires_field,
    validate_types,
    check_field_exists,
    check_multiple_fields_exist,
    # ... plus all other validators and utilities
)

# Legacy support functions
def validate_field_exists(df: pd.DataFrame, field_name: str,
                         logger_instance: Optional[logging.Logger] = None) -> bool:
    """DEPRECATED: Use check_field_exists() instead."""

def validate_numeric_field(df: pd.DataFrame, field_name: str, allow_null: bool = True,
                          min_value: Optional[float] = None, max_value: Optional[float] = None,
                          logger_instance: Optional[logging.Logger] = None) -> bool:
    """DEPRECATED: Use NumericFieldValidator instead."""
```

### 15.11 visualization_utils.py

```python
def create_comparison_visualization(original_data: pd.Series, anonymized_data: pd.Series,
                                  task_dir: Path, field_name: str, operation_name: str,
                                  timestamp: Optional[str] = None) -> Optional[Path]:
    """Create a before/after comparison visualization."""

def create_metric_visualization(metric_name: str, metric_data: Union[Dict[str, Any], pd.Series, List],
                              task_dir: Path, field_name: str, operation_name: str,
                              timestamp: Optional[str] = None) -> Optional[Path]:
    """Create a visualization for a specific metric using appropriate chart type."""

def create_category_distribution_comparison(original_data: pd.Series, anonymized_data: pd.Series,
                                          task_dir: Path, field_name: str, operation_name: str,
                                          max_categories: int = 15, show_percentages: bool = True,
                                          timestamp: Optional[str] = None) -> Optional[Path]:
    """Create a specialized comparison visualization for categorical distributions."""

def create_hierarchy_sunburst(hierarchy_data: Dict[str, Any], task_dir: Path,
                            field_name: str, operation_name: str, max_depth: int = 3,
                            max_categories: int = 50, timestamp: Optional[str] = None) -> Optional[Path]:
    """Create a sunburst visualization for hierarchical category structure."""

def register_visualization_artifact(result: Any, reporter: Any, path: Path,
                                  field_name: str, visualization_type: str,
                                  description: Optional[str] = None) -> None:
    """Register a visualization artifact with the result and reporter."""
```

## 16. Summary

This specification defines a comprehensive framework for anonymization operations that:
- Are atomic, self-contained, and stateless
- Use framework services exclusively
- Focus on transformation, not assessment
- Support both small and large-scale data processing
- Follow strict architectural constraints
- Integrate seamlessly with the PAMOLA.CORE ecosystem
- Allow controlled execute() overrides for advanced use cases

Each operation is a building block that can be composed into complex anonymization workflows while maintaining simplicity, testability, and privacy guarantees.