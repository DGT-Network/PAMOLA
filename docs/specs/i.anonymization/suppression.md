# PAMOLA.CORE Suppression Operations Software Requirements Sub-Specification

**Document Version:** 1.0.0
**Parent Document:** PAMOLA.CORE Anonymization Package SRS v4.1.0
**Last Updated:** 2025-06-15
**Status:** Draft

## 1. Introduction

### 1.1 Purpose

This Software Requirements Sub-Specification (Sub-SRS) defines the detailed requirements for suppression operations within the PAMOLA.CORE anonymization package. Suppression operations remove or replace sensitive data at various granularities: attribute (column), record (row), and cell level.

### 1.2 Scope

This document covers three suppression operations:
- **Attribute Suppression**: Removes entire columns from the dataset
- **Record Suppression**: Removes entire rows based on conditions
- **Cell Suppression**: Replaces individual cell values with NULL or calculated values

All operations follow the base anonymization framework defined in the parent SRS.

### 1.3 Document Conventions

- **REQ-SUPP-XXX**: General suppression requirements
- **REQ-ATTR-XXX**: Attribute suppression specific requirements
- **REQ-REC-XXX**: Record suppression specific requirements
- **REQ-CELL-XXX**: Cell suppression specific requirements

## 2. Common Suppression Requirements

### 2.1 Base Class Inheritance

**REQ-SUPP-001 [MUST]** All suppression operations SHALL inherit from `AnonymizationOperation` and follow the standard operation contract defined in the parent SRS (REQ-ANON-001).

### 2.2 Simplicity Principle

**REQ-SUPP-002 [MUST]** Suppression operations SHALL be simple, focused transformations that:
- Do not perform complex calculations beyond basic statistics (mean, mode)
- Do not require external dictionaries or mappings
- Complete processing in a single pass when possible
- Maintain clear audit trails of what was suppressed

### 2.3 Output Requirements

**REQ-SUPP-003 [MUST]** All suppression operations SHALL:
- Use `DataWriter` to save output files
- Return either a transformed DataFrame or a file reference
- Generate standard metrics about suppression rates
- Create simple before/after visualizations when applicable

## 3. Attribute Suppression Operation

### 3.1 Overview

**REQ-ATTR-001 [MUST]** The `AttributeSuppressionOperation` removes one or more columns from the dataset entirely.

### 3.2 Constructor Interface

**REQ-ATTR-002 [MUST]** Constructor signature:

```python
class AttributeSuppressionOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Primary field to suppress
                 additional_fields: Optional[List[str]] = None,  # Additional fields
                 mode: str = "REMOVE",               # REMOVE only (REPLACE not applicable)
                 save_suppressed_schema: bool = True, # Save removed column info
                 # Standard parameters from base class
                 batch_size: int = 10000,
                 use_cache: bool = False,            # Caching not useful for removal
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 3.3 Processing Logic

**REQ-ATTR-003 [MUST]** The `process_batch()` method SHALL:
1. Validate that all fields to suppress exist in the DataFrame
2. Remove specified columns using `df.drop(columns=[...])`
3. Not modify the original DataFrame
4. Handle the operation in a single pass

**REQ-ATTR-004 [SHOULD]** The operation SHOULD save metadata about suppressed columns:
```python
suppressed_info = {
    "columns_removed": ["field1", "field2"],
    "original_dtypes": {"field1": "int64", "field2": "object"},
    "null_counts": {"field1": 10, "field2": 5},
    "unique_counts": {"field1": 100, "field2": 50}
}
```

### 3.4 Metrics

**REQ-ATTR-005 [MUST]** Collect these specific metrics:
- `columns_suppressed`: Number of columns removed
- `data_width_reduction`: Percentage reduction in columns
- `suppressed_column_names`: List of removed column names

### 3.5 Example Usage

```python
# Suppress PII columns
op = AttributeSuppressionOperation(
    field_name="ssn",
    additional_fields=["phone", "email"],
    save_suppressed_schema=True
)

result = op.execute(data_source, task_dir)
# Output: DataFrame without ssn, phone, email columns
```

## 4. Record Suppression Operation

### 4.1 Overview

**REQ-REC-001 [MUST]** The `RecordSuppressionOperation` removes entire rows from the dataset based on conditions.

### 4.2 Constructor Interface

**REQ-REC-002 [MUST]** Constructor signature:

```python
class RecordSuppressionOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Field to check for suppression
                 suppression_condition: str = "null", # null, value, range, custom
                 suppression_values: Optional[List[Any]] = None,  # For value condition
                 suppression_range: Optional[Tuple[Any, Any]] = None,  # For range
                 mode: str = "REMOVE",               # REMOVE only
                 save_suppressed_records: bool = False,  # Save removed records separately
                 suppression_reason_field: str = "_suppression_reason",
                 # Multi-field conditions
                 multi_field_conditions: Optional[List[Dict[str, Any]]] = None,
                 condition_logic: str = "OR",        # OR, AND
                 # Standard parameters
                 ka_risk_field: Optional[str] = None,  # For risk-based suppression
                 risk_threshold: float = 5.0,
                 batch_size: int = 10000,
                 use_cache: bool = False,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 4.3 Processing Logic

**REQ-REC-003 [MUST]** Support these suppression conditions:
1. **null**: Remove records where field is NULL
2. **value**: Remove records where field matches specific values
3. **range**: Remove records where field is within range
4. **risk**: Remove records based on k-anonymity risk score
5. **custom**: Remove based on multi-field conditions

**REQ-REC-004 [MUST]** The `process_batch()` method SHALL:
```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    # Build suppression mask
    if self.suppression_condition == "null":
        mask = batch[self.field_name].isna()
    elif self.suppression_condition == "value":
        mask = batch[self.field_name].isin(self.suppression_values)
    elif self.suppression_condition == "range":
        mask = batch[self.field_name].between(*self.suppression_range)
    elif self.suppression_condition == "risk":
        if self.ka_risk_field:
            mask = batch[self.ka_risk_field] < self.risk_threshold

    # Apply multi-field conditions if specified
    if self.multi_field_conditions:
        # Build complex mask based on condition_logic

    # Save suppressed records if requested
    if self.save_suppressed_records:
        suppressed = batch[mask].copy()
        suppressed[self.suppression_reason_field] = self._get_suppression_reason()
        # Save using DataWriter

    # Return records that are NOT suppressed
    return batch[~mask].copy()
```

### 4.4 Metrics

**REQ-REC-005 [MUST]** Collect these specific metrics:
- `records_suppressed`: Number of rows removed
- `suppression_rate`: Percentage of records removed
- `suppression_by_condition`: Breakdown by condition type
- `remaining_records`: Number of records after suppression

### 4.5 Example Usage

```python
# Remove high-risk records
op = RecordSuppressionOperation(
    field_name="k_anonymity_score",
    suppression_condition="risk",
    ka_risk_field="k_anonymity_score",
    risk_threshold=5.0,
    save_suppressed_records=True
)

# Remove records with specific values
op2 = RecordSuppressionOperation(
    field_name="country",
    suppression_condition="value",
    suppression_values=["Unknown", "Invalid", "N/A"]
)
```

## 5. Cell Suppression Operation

### 5.1 Overview

**REQ-CELL-001 [MUST]** The `CellSuppressionOperation` replaces individual cell values while preserving the record structure.

### 5.2 Constructor Interface

**REQ-CELL-002 [MUST]** Constructor signature:

```python
class CellSuppressionOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Field containing cells to suppress
                 suppression_strategy: str = "null", # null, mean, mode, constant
                 suppression_value: Optional[Any] = None,  # For constant strategy
                 mode: str = "REPLACE",              # Standard mode
                 output_field_name: Optional[str] = None,
                 # Conditional suppression
                 condition_field: Optional[str] = None,
                 condition_values: Optional[List] = None,
                 condition_operator: str = "in",
                 # Group-based suppression (for mean/mode)
                 group_by_field: Optional[str] = None,
                 min_group_size: int = 5,           # Minimum size for group statistics
                 # Value-based conditions
                 suppress_if: Optional[str] = None,  # "outlier", "rare", "null"
                 outlier_method: str = "iqr",       # iqr, zscore
                 outlier_threshold: float = 1.5,     # IQR multiplier or z-score
                 rare_threshold: int = 10,          # For rare value detection
                 # Standard parameters
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 5.3 Processing Logic

**REQ-CELL-003 [MUST]** Support these suppression strategies:
1. **null**: Replace with NULL/NaN
2. **mean**: Replace with column mean (numeric only)
3. **mode**: Replace with column mode (any type)
4. **constant**: Replace with specified value
5. **group_mean**: Replace with group mean
6. **group_mode**: Replace with group mode

**REQ-CELL-004 [MUST]** The `process_batch()` method SHALL:
```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    result = batch.copy()

    # Determine which cells to suppress
    if self.suppress_if == "outlier":
        suppress_mask = self._detect_outliers(batch[self.field_name])
    elif self.suppress_if == "rare":
        value_counts = batch[self.field_name].value_counts()
        rare_values = value_counts[value_counts < self.rare_threshold].index
        suppress_mask = batch[self.field_name].isin(rare_values)
    elif self.suppress_if == "null":
        suppress_mask = batch[self.field_name].isna()
    else:
        # Use conditional suppression
        suppress_mask = self._build_condition_mask(batch)

    # Apply suppression strategy
    if self.suppression_strategy == "null":
        result.loc[suppress_mask, self.field_name] = None

    elif self.suppression_strategy == "mean":
        if pd.api.types.is_numeric_dtype(batch[self.field_name]):
            if self.group_by_field:
                # Calculate group means
                group_means = batch.groupby(self.group_by_field)[self.field_name].mean()
                for group, mean_val in group_means.items():
                    group_mask = (batch[self.group_by_field] == group) & suppress_mask
                    result.loc[group_mask, self.field_name] = mean_val
            else:
                # Global mean
                mean_val = batch[self.field_name].mean()
                result.loc[suppress_mask, self.field_name] = mean_val

    elif self.suppression_strategy == "mode":
        if self.group_by_field:
            # Calculate group modes
            group_modes = batch.groupby(self.group_by_field)[self.field_name].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else None
            )
            for group, mode_val in group_modes.items():
                group_mask = (batch[self.group_by_field] == group) & suppress_mask
                result.loc[group_mask, self.field_name] = mode_val
        else:
            # Global mode
            mode_val = batch[self.field_name].mode()
            if len(mode_val) > 0:
                result.loc[suppress_mask, self.field_name] = mode_val[0]

    elif self.suppression_strategy == "constant":
        result.loc[suppress_mask, self.field_name] = self.suppression_value

    return result
```

### 5.4 Outlier Detection

**REQ-CELL-005 [SHOULD]** Implement outlier detection methods:
```python
def _detect_outliers(self, series: pd.Series) -> pd.Series:
    if self.outlier_method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif self.outlier_method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > self.outlier_threshold
```

### 5.5 Metrics

**REQ-CELL-006 [MUST]** Collect these specific metrics:
- `cells_suppressed`: Number of individual cells suppressed
- `suppression_rate`: Percentage of non-null cells suppressed
- `suppression_by_strategy`: Breakdown by strategy used
- `group_statistics`: If group-based, statistics per group
- `outliers_detected`: If outlier suppression, count of outliers

### 5.6 Example Usage

```python
# Suppress outliers with mean
op = CellSuppressionOperation(
    field_name="salary",
    suppression_strategy="mean",
    suppress_if="outlier",
    outlier_method="iqr",
    outlier_threshold=1.5
)

# Suppress rare categories with mode by group
op2 = CellSuppressionOperation(
    field_name="job_title",
    suppression_strategy="group_mode",
    suppress_if="rare",
    rare_threshold=5,
    group_by_field="department"
)

# Conditional suppression with constant
op3 = CellSuppressionOperation(
    field_name="age",
    suppression_strategy="constant",
    suppression_value=99,
    condition_field="consent",
    condition_values=[False],
    condition_operator="eq"
)
```

## 6. Common Implementation Details

### 6.1 Batch Processing

**REQ-SUPP-004 [MUST]** All suppression operations SHALL support batch processing:
- Attribute suppression: Process entire DataFrame at once (no batching needed)
- Record suppression: Process in batches but maintain record counts
- Cell suppression: Process in batches with statistics calculated per batch or globally

### 6.2 Visualization Requirements

**REQ-SUPP-005 [SHOULD]** Generate simple visualizations:
- **Attribute**: Bar chart showing number of columns before/after
- **Record**: Bar chart showing record count before/after
- **Cell**: Histogram or bar chart showing distribution changes

### 6.3 Validation Requirements

**REQ-SUPP-006 [MUST]** Use commons validation utilities:
```python
from pamola_core.anonymization.commons.validation import (
    check_field_exists,
    check_multiple_fields_exist,
    validate_numeric_field
)

# In process_batch for cell suppression with mean
if self.suppression_strategy == "mean":
    validate_numeric_field(batch, self.field_name)
```

### 6.4 Metrics Collection

**REQ-SUPP-007 [MUST]** Use standardized metrics collection:
```python
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
    calculate_suppression_metrics
)

# Collect standard metrics
metrics = collect_operation_metrics(
    operation_type="suppression",
    original_data=original_series,
    processed_data=processed_series,
    operation_params={
        "strategy": self.suppression_strategy,
        "condition": self.suppress_if
    },
    timing_info=timing_info
)

# Add suppression-specific metrics
suppression_metrics = calculate_suppression_metrics(
    original_series=original_series,
    suppressed_series=processed_series
)
metrics.update(suppression_metrics)
```

## 7. Error Handling

### 7.1 Common Errors

**REQ-SUPP-008 [MUST]** Handle these common error conditions:

1. **Field not found**: Use `FieldNotFoundError`
2. **Invalid strategy**: Use `InvalidStrategyError`
3. **Type mismatch**: e.g., mean on non-numeric field
4. **Empty result**: Warning when all records are suppressed
5. **Group too small**: When group size < min_group_size

### 7.2 Error Responses

**REQ-SUPP-009 [MUST]** Follow standard error handling:
```python
try:
    # Validation
    if self.suppression_strategy == "mean":
        if not pd.api.types.is_numeric_dtype(batch[self.field_name]):
            raise FieldTypeError(
                self.field_name,
                expected_type="numeric",
                actual_type=str(batch[self.field_name].dtype)
            )
except FieldTypeError as e:
    self.logger.error(f"Type error: {e}")
    if self.error_handling == "fail":
        raise
    elif self.error_handling == "skip":
        return batch  # Return unmodified
    else:  # fallback
        # Fall back to mode strategy
        self.suppression_strategy = "mode"
        self.logger.warning("Falling back to mode strategy")
```

## 8. Performance Considerations

### 8.1 Optimization Guidelines

**REQ-SUPP-010 [SHOULD]** Optimize for performance:

1. **Attribute Suppression**:
   - Single operation, no optimization needed
   - Use `inplace=False` to preserve original

2. **Record Suppression**:
   - Use boolean indexing for efficiency
   - Avoid iterating over rows
   - Pre-calculate complex conditions

3. **Cell Suppression**:
   - Calculate statistics once if possible
   - Use vectorized operations
   - Cache group statistics for reuse

### 8.2 Memory Management

**REQ-SUPP-011 [MUST]** Manage memory efficiently:
```python
# For record suppression with saved records
if self.save_suppressed_records and len(suppressed) > 100000:
    # Write in chunks to avoid memory issues
    writer = DataWriter(self.task_dir)
    for chunk_start in range(0, len(suppressed), 50000):
        chunk_end = min(chunk_start + 50000, len(suppressed))
        chunk = suppressed.iloc[chunk_start:chunk_end]
        writer.write_dataframe(
            df=chunk,
            name=f"suppressed_records_chunk_{chunk_start}",
            format="parquet",  # Efficient format
            subdir="suppressed"
        )
```

## 9. Testing Requirements

### 9.1 Unit Tests

**REQ-SUPP-012 [MUST]** Test coverage must include:

1. **Attribute Suppression**:
   - Single column removal
   - Multiple column removal
   - Non-existent column handling
   - Empty DataFrame handling

2. **Record Suppression**:
   - Each condition type (null, value, range, risk)
   - Multi-field conditions with AND/OR logic
   - Edge cases (all records suppressed, none suppressed)
   - Saved suppressed records validation

3. **Cell Suppression**:
   - Each strategy (null, mean, mode, constant)
   - Group-based suppression
   - Outlier detection methods
   - Type compatibility (mean on categorical, etc.)

### 9.2 Integration Tests

**REQ-SUPP-013 [MUST]** Test framework integration:
- DataWriter output verification
- Metric collection completeness
- Visualization generation
- Progress tracking accuracy

## 10. Example Implementations

### 10.1 Simple Attribute Suppression

```python
class AttributeSuppressionOperation(AnonymizationOperation):
    """Remove sensitive columns from dataset."""

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Remove specified columns."""
        # Collect fields to suppress
        fields_to_drop = [self.field_name]
        if self.additional_fields:
            fields_to_drop.extend(self.additional_fields)

        # Validate all fields exist
        check_multiple_fields_exist(batch, fields_to_drop)

        # Save metadata if requested
        if self.save_suppressed_schema:
            schema_info = {
                col: {
                    "dtype": str(batch[col].dtype),
                    "null_count": int(batch[col].isna().sum()),
                    "unique_count": int(batch[col].nunique())
                }
                for col in fields_to_drop
            }
            self._suppressed_schema = schema_info

        # Drop columns
        return batch.drop(columns=fields_to_drop)

    def _collect_specific_metrics(self, original_data: pd.Series,
                                 anonymized_data: pd.Series) -> Dict[str, Any]:
        """Collect attribute suppression metrics."""
        return {
            "columns_suppressed": len(self._suppressed_schema),
            "suppressed_columns": list(self._suppressed_schema.keys())
        }
```

### 10.2 Risk-Based Record Suppression

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Suppress records based on risk scores."""
    # Build suppression mask
    if self.ka_risk_field and self.ka_risk_field in batch.columns:
        # Risk-based suppression
        mask = batch[self.ka_risk_field] < self.risk_threshold
        self._suppression_reasons["risk"] = mask.sum()
    else:
        # Fallback to value-based
        mask = self._build_value_mask(batch)

    # Save suppressed records if needed
    if self.save_suppressed_records and mask.any():
        suppressed = batch[mask].copy()
        suppressed["_suppression_timestamp"] = pd.Timestamp.now()
        suppressed["_suppression_reason"] = self.suppression_condition

        # Write suppressed records
        writer = DataWriter(self.task_dir)
        writer.write_dataframe(
            df=suppressed,
            name="suppressed_records",
            format="parquet",
            subdir="audit"
        )

    # Return non-suppressed records
    return batch[~mask].copy()
```

## 11. Summary

The suppression operations provide simple, effective privacy protection through:
- **Attribute Suppression**: Complete column removal
- **Record Suppression**: Conditional row removal
- **Cell Suppression**: Targeted value replacement

Each operation follows the PAMOLA.CORE framework principles while maintaining simplicity and focusing on its specific suppression task. The operations integrate seamlessly with the commons utilities and framework services for validation, metrics, and output handling.