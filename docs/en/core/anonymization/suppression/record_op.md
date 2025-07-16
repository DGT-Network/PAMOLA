# PAMOLA Record Suppression Operation Documentation

**Module:** `pamola_core.anonymization.suppression.record_op`  
**Version:** 1.1.0  
**Last Updated:** 2025-06-15  
**Status:** Stable  
**License:** Apache License 2.0

## 1. Overview

The Record Suppression Operation module provides a comprehensive solution for removing entire rows (records) from datasets based on various conditions. This operation is a critical component of the PAMOLA privacy protection framework, enabling selective record removal to meet privacy requirements while maintaining data utility.

### 1.1 Purpose

Record suppression is used when:
- Certain records contain sensitive information that cannot be adequately anonymized
- Records fail to meet k-anonymity thresholds
- Specific values or conditions indicate high re-identification risk
- Compliance requirements mandate removal of certain data categories
- Data quality issues require record exclusion

### 1.2 Key Features

- **Multiple Suppression Conditions**: Support for null, value, range, risk-based, and custom multi-field conditions
- **Flexible Logic**: AND/OR logic for combining multiple conditions
- **Audit Trail**: Optional saving of suppressed records with reasons and timestamps
- **Memory Efficient**: Batch-wise processing with immediate disk writes for large datasets
- **Distributed Processing**: Full Dask support for handling datasets exceeding memory limits
- **Comprehensive Metrics**: Detailed tracking of suppression rates and reasons
- **Visual Analytics**: Automatic generation of suppression visualizations

## 2. Module Structure

### 2.1 Class Hierarchy

```
AnonymizationOperation (base)
    └── RecordSuppressionOperation
```

### 2.2 Key Components

```python
RecordSuppressionOperation
    ├── __init__()              # Configuration and validation
    ├── process_batch()         # Core batch processing logic
    ├── _build_suppression_mask()    # Condition evaluation
    ├── _save_suppressed_batch()     # Audit trail management
    ├── _process_batch_dask()        # Distributed processing
    ├── _collect_specific_metrics()  # Metrics collection
    ├── _generate_visualization()    # Visualization creation
    └── execute()               # Operation orchestration
```

### 2.3 Dependencies

#### Internal Dependencies
- `pamola_core.anonymization.base_anonymization_op.AnonymizationOperation`
- `pamola_core.anonymization.commons.validation_utils`
- `pamola_core.anonymization.commons.visualization_utils`
- `pamola_core.utils.ops.op_data_source`
- `pamola_core.utils.ops.op_data_writer`
- `pamola_core.utils.ops.op_result`
- `pamola_core.utils.ops.op_field_utils`
- `pamola_core.utils.progress`

#### External Dependencies
- `pandas`: DataFrame operations
- `numpy`: Numerical operations
- `dask` (optional): Distributed processing
- `pathlib`: File path handling

## 3. API Reference

### 3.1 Constructor Signature

```python
RecordSuppressionOperation(
    field_name: str,
    suppression_condition: str = "null",
    suppression_values: Optional[List[Any]] = None,
    suppression_range: Optional[Tuple[Any, Any]] = None,
    mode: str = "REMOVE",
    save_suppressed_records: bool = False,
    suppression_reason_field: str = "_suppression_reason",
    # Multi-field conditions
    multi_field_conditions: Optional[List[Dict[str, Any]]] = None,
    condition_logic: str = "OR",
    # Risk-based suppression
    ka_risk_field: Optional[str] = None,
    risk_threshold: float = 5.0,
    # Standard parameters
    output_field_name: Optional[str] = None,
    null_strategy: str = "PRESERVE",
    batch_size: int = 10000,
    use_cache: bool = False,
    use_encryption: bool = False,
    encryption_key: Optional[Path] = None,
    engine: str = "auto",
    max_rows_in_memory: int = 1000000,
    **kwargs
)
```

### 3.2 Parameters

#### Primary Parameters
- **field_name** (str): Primary field to check for suppression conditions
- **suppression_condition** (str): Type of condition - "null", "value", "range", "risk", "custom"
- **suppression_values** (List[Any]): Values to match for "value" condition
- **suppression_range** (Tuple[Any, Any]): (min, max) bounds for "range" condition
- **save_suppressed_records** (bool): Whether to save removed records for audit

#### Multi-field Parameters
- **multi_field_conditions** (List[Dict]): List of field conditions with structure:
  ```python
  {
      "field": "field_name",
      "operator": "eq|ne|gt|lt|gte|lte|in|null|not_null",
      "value": Any
  }
  ```
- **condition_logic** (str): "AND" or "OR" for combining conditions

#### Risk-based Parameters
- **ka_risk_field** (str): Field containing k-anonymity risk scores
- **risk_threshold** (float): Threshold below which records are suppressed

#### Processing Parameters
- **batch_size** (int): Number of records to process per batch
- **engine** (str): "pandas", "dask", or "auto"
- **max_rows_in_memory** (int): Threshold for automatic Dask switching

### 3.3 Methods

#### process_batch(batch: pd.DataFrame) -> pd.DataFrame
Processes a single batch of data, applying suppression conditions.

**Returns:** DataFrame with suppressed records removed

#### execute(data_source: DataSource, task_dir: Path, ...) -> OperationResult
Main execution method that orchestrates the entire suppression operation.

**Returns:** OperationResult with metrics, artifacts, and status

## 4. Suppression Conditions

### 4.1 Null Condition
Removes records where the specified field contains null values.

```python
op = RecordSuppressionOperation(
    field_name="email",
    suppression_condition="null"
)
```

### 4.2 Value Condition
Removes records matching specific values.

```python
op = RecordSuppressionOperation(
    field_name="status",
    suppression_condition="value",
    suppression_values=["Invalid", "Error", "Unknown"]
)
```

### 4.3 Range Condition
Removes records within a numeric range.

```python
op = RecordSuppressionOperation(
    field_name="age",
    suppression_condition="range",
    suppression_range=(0, 17)  # Remove minors
)
```

### 4.4 Risk Condition
Removes records based on k-anonymity risk scores.

```python
op = RecordSuppressionOperation(
    field_name="user_id",  # Any field (required by base class)
    suppression_condition="risk",
    ka_risk_field="k_anonymity_score",
    risk_threshold=5.0
)
```

### 4.5 Custom Multi-field Condition
Removes records based on complex conditions across multiple fields.

```python
op = RecordSuppressionOperation(
    field_name="status",
    suppression_condition="custom",
    multi_field_conditions=[
        {"field": "age", "operator": "lt", "value": 18},
        {"field": "consent", "operator": "eq", "value": False},
        {"field": "risk_score", "operator": "gt", "value": 0.8}
    ],
    condition_logic="OR"  # Remove if ANY condition is true
)
```

## 5. Usage Examples

### 5.1 Basic Usage

```python
from pathlib import Path
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation

# Create data source
data_source = DataSource("path/to/data.csv")

# Create operation
op = RecordSuppressionOperation(
    field_name="ssn",
    suppression_condition="null",
    save_suppressed_records=True
)

# Execute
result = op.execute(
    data_source=data_source,
    task_dir=Path("output/"),
    reporter=reporter,
    progress_tracker=progress_tracker
)

# Check results
print(f"Records suppressed: {result.metrics['records_suppressed']}")
print(f"Suppression rate: {result.metrics['suppression_rate']}%")
```

### 5.2 Complex Multi-condition Example

```python
# Remove high-risk records with multiple criteria
op = RecordSuppressionOperation(
    field_name="user_id",
    suppression_condition="custom",
    multi_field_conditions=[
        # Remove if k < 3
        {"field": "k_anonymity", "operator": "lt", "value": 3},
        # OR if in high-risk category
        {"field": "risk_category", "operator": "in", "value": ["HIGH", "CRITICAL"]},
        # OR if missing critical fields
        {"field": "postal_code", "operator": "null", "value": None},
        {"field": "birth_date", "operator": "null", "value": None}
    ],
    condition_logic="OR",
    save_suppressed_records=True,
    use_encryption=True,  # Encrypt audit trail
    encryption_key=Path("keys/audit.key")
)
```

### 5.3 Large Dataset with Dask

```python
# Process large dataset with automatic Dask switching
op = RecordSuppressionOperation(
    field_name="transaction_id",
    suppression_condition="value",
    suppression_values=["CANCELLED", "FAILED", "REJECTED"],
    engine="auto",  # Automatically use Dask if needed
    max_rows_in_memory=1_000_000,  # Switch to Dask above 1M rows
    batch_size=50000  # Larger batches for efficiency
)
```

### 5.4 Risk-based Suppression Pipeline

```python
# First run k-anonymity profiling
from pamola_core.profiling import KAnonymityProfiler

profiler = KAnonymityProfiler(quasi_identifiers=["age", "zipcode", "gender"])
profiling_result = profiler.execute(data_source, task_dir)

# Then suppress high-risk records
op = RecordSuppressionOperation(
    field_name="record_id",
    suppression_condition="risk",
    ka_risk_field="k_anonymity_score",
    risk_threshold=3.0,  # Remove records with k < 3
    save_suppressed_records=True
)

# Pass profiling results to suppression
result = op.execute(
    data_source=data_source,
    task_dir=task_dir,
    profiling_results=profiling_result.metrics
)
```

## 6. Output Structure

### 6.1 Metrics
```json
{
    "operation_type": "record_suppression",
    "suppression_condition": "risk",
    "records_suppressed": 1523,
    "suppression_rate": 2.4,
    "remaining_records": 61977,
    "suppression_by_condition": {
        "risk": 1523
    },
    "risk_threshold": 5.0,
    "ka_risk_field": "k_anonymity_score"
}
```

### 6.2 Artifacts

#### Suppressed Records (Parquet)
```
output/
├── audit/
│   ├── suppressed_records_consolidated_20250615_143022.parquet
│   └── suppression_summary.json
└── visualizations/
    ├── record_suppression_summary.png
    └── suppression_reasons_breakdown.png
```

#### Suppression Summary (JSON)
```json
{
    "total_suppressed": 1523,
    "suppression_rate": 2.4,
    "suppression_by_condition": {
        "risk": 1523
    },
    "suppression_condition": "risk",
    "original_records": 63500,
    "remaining_records": 61977,
    "timestamp": "2025-06-15T14:30:22.123456"
}
```

### 6.3 Suppressed Records Schema
Each suppressed record includes:
- All original fields
- `_suppression_reason`: Human-readable reason
- `_suppression_reason_timestamp`: When suppressed
- `_suppression_reason_batch`: Batch number

## 7. Performance Considerations

### 7.1 Memory Management
- Suppressed records are written to disk batch-by-batch
- No accumulation of DataFrames in memory
- Temporary files consolidated at operation end
- Automatic cleanup of intermediate files

### 7.2 Processing Optimization
- Vectorized boolean masking for efficiency
- Minimal data copying
- Batch size auto-adjustment based on memory
- Dask integration for datasets > 1M rows

### 7.3 Benchmarks
| Dataset Size | Processing Time | Memory Usage |
|-------------|-----------------|--------------|
| 100K rows   | ~2 seconds      | 150 MB       |
| 1M rows     | ~15 seconds     | 800 MB       |
| 10M rows    | ~2 minutes      | 1.5 GB (Dask)|
| 100M rows   | ~18 minutes     | 2 GB (Dask)  |

## 8. Security Considerations

### 8.1 Audit Trail Protection
- Optional encryption of suppressed records
- Secure storage of sensitive audit data
- No sensitive data in log files
- Configurable retention policies

### 8.2 Privacy Compliance
- Complete removal from output dataset
- Audit trail for compliance verification
- Reason tracking for accountability
- Timestamp recording for temporal analysis

## 9. Integration Guidelines

### 9.1 Pipeline Integration
```python
# Typical anonymization pipeline
pipeline = [
    DataValidationOperation(),
    KAnonymityProfiler(),
    RecordSuppressionOperation(
        suppression_condition="risk",
        ka_risk_field="k_score",
        risk_threshold=5.0
    ),
    GeneralizationOperation(),
    MaskingOperation()
]
```

### 9.2 Error Handling
```python
try:
    result = op.execute(data_source, task_dir)
    if result.status == OperationStatus.SUCCESS:
        print(f"Successfully suppressed {result.metrics['records_suppressed']} records")
    else:
        print(f"Operation failed: {result.error_message}")
except FieldNotFoundError as e:
    print(f"Field validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 10. Best Practices

### 10.1 Condition Selection
- Use specific conditions over broad ones
- Combine conditions logically for precision
- Validate fields exist before execution
- Test conditions on sample data first

### 10.2 Audit Management
- Always save suppressed records for regulated data
- Use encryption for sensitive audit trails
- Implement retention policies
- Monitor suppression rates

### 10.3 Performance Tuning
- Adjust batch size based on record width
- Use Dask for datasets > 1M rows
- Pre-filter data when possible
- Index on suppression fields

## 11. Troubleshooting

### 11.1 Common Issues

**Issue:** High suppression rates
```python
# Solution: Review and adjust conditions
# Check distribution first
value_counts = df[field_name].value_counts()
print(f"Values that would be suppressed: {value_counts[suppression_values].sum()}")
```

**Issue:** Memory errors with large datasets
```python
# Solution: Force Dask usage
op = RecordSuppressionOperation(
    field_name="id",
    suppression_condition="null",
    engine="dask",  # Force Dask
    batch_size=25000  # Smaller batches
)
```

**Issue:** Slow processing
```python
# Solution: Optimize conditions
# Instead of multiple value checks:
# BAD: suppression_values = [val1, val2, ..., val1000]
# GOOD: Use range or custom condition with efficient logic
```

### 11.2 Debug Mode
```python
import logging

# Enable debug logging
logging.getLogger("pamola_core.anonymization.suppression").setLevel(logging.DEBUG)

# Operation will now log detailed progress
op = RecordSuppressionOperation(
    field_name="ssn",
    suppression_condition="null"
)
```

## 12. Future Enhancements

### 12.1 Planned Features
- Probabilistic suppression (random sampling)
- Time-based suppression conditions
- Geospatial suppression regions
- Machine learning-based risk assessment
- Real-time suppression streaming

### 12.2 API Stability
The current API is stable and backward compatible. Future versions will maintain compatibility while adding new optional parameters.

---

**Document Version:** 1.0.0  
**Last Reviewed:** 2025-06-15  
**Next Review:** 2025-09-15  
**Maintainers:** PAMOLA Core Team