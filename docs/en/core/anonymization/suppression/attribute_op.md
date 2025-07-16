# PAMOLA Attribute Suppression Operation Documentation

**Module:** `pamola_core.anonymization.suppression.attribute_op`  
**Version:** 1.4.0  
**Last Updated:** 2025-06-15  
**Status:** Stable  
**License:** Apache License 2.0

## 1. Overview

The Attribute Suppression Operation module provides a straightforward yet powerful mechanism for removing entire columns (attributes) from datasets. This operation is fundamental to privacy protection when certain fields contain information that cannot be adequately anonymized through other means or must be completely removed for compliance reasons.

### 1.1 Purpose

Attribute suppression is used when:
- Columns contain direct identifiers that must be removed (SSN, phone numbers, email addresses)
- Fields have high re-identification risk that cannot be mitigated through generalization
- Compliance requirements mandate complete removal of specific data types
- Data minimization principles require reducing data collection scope
- Certain attributes are irrelevant for the intended analysis

### 1.2 Key Features

- **Single or Multiple Column Removal**: Remove one or many columns in a single operation
- **Metadata Preservation**: Save information about suppressed columns for documentation
- **Schema Tracking**: Maintain audit trail of data transformations
- **Simple Operation**: No complex calculations, just clean column removal
- **Memory Efficient**: Process entire DataFrame without chunking overhead
- **Comprehensive Metrics**: Track data width reduction and schema changes
- **Visual Analytics**: Automatic generation of before/after visualizations

## 2. Module Structure

### 2.1 Class Hierarchy

```
AnonymizationOperation (base)
    └── AttributeSuppressionOperation
```

### 2.2 Key Components

```python
AttributeSuppressionOperation
    ├── __init__()                    # Configuration and validation
    ├── process_batch()               # Core column removal logic
    ├── _collect_suppressed_metadata() # Schema information gathering
    ├── _process_batch_dask()         # Distributed processing support
    ├── _collect_specific_metrics()   # Metrics collection
    ├── _generate_visualization()     # Visualization creation
    └── execute()                     # Operation orchestration with schema saving
```

### 2.3 Dependencies

#### Internal Dependencies
- `pamola_core.anonymization.base_anonymization_op.AnonymizationOperation`
- `pamola_core.anonymization.commons.validation`
- `pamola_core.anonymization.commons.visualization_utils`
- `pamola_core.utils.ops.op_data_source`
- `pamola_core.utils.ops.op_data_writer`
- `pamola_core.utils.ops.op_result`
- `pamola_core.utils.progress`

#### External Dependencies
- `pandas`: DataFrame operations
- `numpy`: Numerical operations
- `dask` (optional): Distributed processing
- `pathlib`: File path handling

## 3. API Reference

### 3.1 Constructor Signature

```python
AttributeSuppressionOperation(
    field_name: str,
    additional_fields: Optional[List[str]] = None,
    mode: str = "REMOVE",
    save_suppressed_schema: bool = True,
    # Standard parameters from base class
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
- **field_name** (str): Primary field to suppress (required)
- **additional_fields** (List[str]): Additional fields to suppress in the same operation
- **mode** (str): Must be "REMOVE" (only valid mode for attribute suppression)
- **save_suppressed_schema** (bool): Whether to save metadata about removed columns

#### Processing Parameters
- **batch_size** (int): Not used for attribute suppression (processes entire DataFrame)
- **use_cache** (bool): Whether to cache results (limited utility for column removal)
- **use_encryption** (bool): Whether to encrypt output files
- **engine** (str): "pandas" or "dask" (Dask provides limited benefits for this operation)

### 3.3 Methods

#### process_batch(batch: pd.DataFrame) -> pd.DataFrame
Removes specified columns from the DataFrame.

**Returns:** DataFrame with specified columns removed

#### execute(data_source: DataSource, task_dir: Path, ...) -> OperationResult
Main execution method that orchestrates the suppression operation and saves schema metadata.

**Returns:** OperationResult with metrics, artifacts, and status

## 4. Suppression Metadata

### 4.1 Schema Information
When `save_suppressed_schema=True`, the operation collects:

```python
{
    "column_name": {
        "dtype": "int64",              # Data type
        "null_count": 150,             # Number of null values
        "non_null_count": 9850,        # Number of non-null values
        "unique_count": 1234,          # Number of unique values
        "memory_usage": 80000,         # Memory usage in bytes
        # For numeric columns:
        "min": 0.0,                    # Minimum value
        "max": 100.0,                  # Maximum value
        "mean": 45.3,                  # Mean value
        "std": 12.7                    # Standard deviation
    }
}
```

### 4.2 Schema File Output
```json
{
    "ssn": {
        "dtype": "object",
        "null_count": 12,
        "non_null_count": 9988,
        "unique_count": 9988,
        "memory_usage": 319616
    },
    "phone": {
        "dtype": "object",
        "null_count": 523,
        "non_null_count": 9477,
        "unique_count": 9400,
        "memory_usage": 319616
    }
}
```

## 5. Usage Examples

### 5.1 Basic Single Column Removal

```python
from pathlib import Path
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.anonymization.suppression.attribute_op import AttributeSuppressionOperation

# Create data source
data_source = DataSource("path/to/data.csv")

# Remove SSN column
op = AttributeSuppressionOperation(
    field_name="ssn",
    save_suppressed_schema=True
)

# Execute
result = op.execute(
    data_source=data_source,
    task_dir=Path("output/"),
    reporter=reporter
)

print(f"Columns suppressed: {result.metrics['columns_suppressed']}")
```

### 5.2 Multiple Column Removal

```python
# Remove all PII columns at once
op = AttributeSuppressionOperation(
    field_name="ssn",
    additional_fields=["phone", "email", "full_name", "address"],
    save_suppressed_schema=True
)

result = op.execute(data_source, task_dir)

# Check data width reduction
print(f"Data width reduced by: {result.metrics['data_width_reduction']:.1f}%")
```

### 5.3 Compliance-Driven Suppression

```python
# GDPR compliance - remove all direct identifiers
direct_identifiers = [
    "national_id", "passport_number", "driver_license",
    "credit_card", "bank_account", "tax_id"
]

op = AttributeSuppressionOperation(
    field_name=direct_identifiers[0],
    additional_fields=direct_identifiers[1:],
    save_suppressed_schema=True,
    use_encryption=True,  # Encrypt schema information
    encryption_key=Path("keys/schema.key")
)
```

### 5.4 Integration with Analysis Pipeline

```python
# Remove columns before analysis
columns_to_remove = ["internal_id", "created_by", "modified_by", "system_flags"]

# First, suppress internal columns
suppress_op = AttributeSuppressionOperation(
    field_name=columns_to_remove[0],
    additional_fields=columns_to_remove[1:]
)

# Then proceed with analysis
from pamola_core.analysis import DataProfiler

pipeline = [
    suppress_op,              # Remove internal columns
    DataProfiler(),          # Profile cleaned data
    GeneralizationOperation() # Apply generalizations
]
```

### 5.5 Conditional Column Removal

```python
# Analyze data first to determine which columns to remove
df = data_source.get_dataframe()[0]

# Find columns with too many nulls
high_null_columns = []
for col in df.columns:
    null_ratio = df[col].isna().sum() / len(df)
    if null_ratio > 0.9:  # More than 90% null
        high_null_columns.append(col)

if high_null_columns:
    # Remove high-null columns
    op = AttributeSuppressionOperation(
        field_name=high_null_columns[0],
        additional_fields=high_null_columns[1:] if len(high_null_columns) > 1 else None,
        save_suppressed_schema=True
    )
    result = op.execute(data_source, task_dir)
```

## 6. Output Structure

### 6.1 Metrics
```json
{
    "operation_type": "attribute_suppression",
    "columns_suppressed": 5,
    "suppressed_column_names": ["ssn", "phone", "email", "address", "full_name"],
    "data_width_reduction": 15.6
}
```

### 6.2 Artifacts

#### Directory Structure
```
output/
├── audit/
│   └── suppressed_columns_schema.json
├── output/
│   └── ssn_anonymized_20250615_143022.csv
└── visualizations/
    ├── attribute_suppression_summary.png
    └── suppressed_columns_dtype_distribution.png
```

#### Suppressed Schema (JSON)
```json
{
    "ssn": {
        "dtype": "object",
        "null_count": 12,
        "non_null_count": 9988,
        "unique_count": 9988,
        "memory_usage": 319616
    },
    "salary": {
        "dtype": "float64",
        "null_count": 45,
        "non_null_count": 9955,
        "unique_count": 8234,
        "memory_usage": 80000,
        "min": 25000.0,
        "max": 250000.0,
        "mean": 65432.1,
        "std": 28901.5
    }
}
```

### 6.3 Visualizations

1. **Attribute Suppression Summary**: Bar chart showing column count before/after
2. **Data Type Distribution**: Distribution of suppressed columns by data type

## 7. Performance Considerations

### 7.1 Operation Characteristics
- **Time Complexity**: O(1) - Column removal is a metadata operation
- **Space Complexity**: O(n×m') where m' is the number of remaining columns
- **Memory Usage**: Minimal overhead, creates new DataFrame view

### 7.2 Processing Optimization
- Single-pass operation (no batching needed)
- Efficient pandas column dropping
- Metadata collection adds minimal overhead
- No data copying for Dask DataFrames

### 7.3 Benchmarks
| Dataset Size | Columns | Suppressed | Time    | Memory Delta |
|-------------|---------|------------|---------|--------------|
| 100K × 50   | 50      | 10         | 0.05s   | -20%         |
| 1M × 100    | 100     | 25         | 0.2s    | -25%         |
| 10M × 200   | 200     | 50         | 0.8s    | -25%         |
| 100M × 500  | 500     | 100        | 2.5s    | -20%         |

## 8. Best Practices

### 8.1 Column Selection
- Group related columns for removal (e.g., all PII fields)
- Document reasons for suppression
- Validate column existence before execution
- Consider impact on downstream analysis

### 8.2 Schema Management
```python
# Always save schema for audit trails
op = AttributeSuppressionOperation(
    field_name="sensitive_field",
    save_suppressed_schema=True  # Default, but be explicit
)

# For temporary analysis, skip schema
op_temp = AttributeSuppressionOperation(
    field_name="temp_column",
    save_suppressed_schema=False
)
```

### 8.3 Integration Patterns
```python
# Pattern 1: Remove identifiers first
pipeline = [
    AttributeSuppressionOperation(field_name="id", additional_fields=["uuid"]),
    KAnonymityProfiler(),
    GeneralizationOperation()
]

# Pattern 2: Remove after analysis
pipeline = [
    DataProfiler(),  # Analyze all columns
    AttributeSuppressionOperation(  # Then remove unnecessary ones
        field_name="internal_field",
        additional_fields=identified_removals
    )
]
```

## 9. Error Handling

### 9.1 Common Errors

**FieldNotFoundError**
```python
# Error: Field doesn't exist
try:
    op = AttributeSuppressionOperation(field_name="nonexistent_column")
    result = op.execute(data_source, task_dir)
except FieldNotFoundError as e:
    print(f"Column not found: {e.missing_field}")
    print(f"Available columns: {e.existing_fields}")
```

**ValueError for Invalid Mode**
```python
# Error: Wrong mode
try:
    op = AttributeSuppressionOperation(
        field_name="ssn",
        mode="MASK"  # Invalid - only "REMOVE" allowed
    )
except ValueError as e:
    print(f"Invalid mode: {e}")
```

### 9.2 Validation Helpers
```python
# Pre-validate columns exist
from pamola_core.anonymization.commons.validation import check_multiple_fields_exist

df = data_source.get_dataframe()[0]
columns_to_remove = ["ssn", "phone", "email"]

valid, missing = check_multiple_fields_exist(df, columns_to_remove)
if not valid:
    print(f"Missing columns: {missing}")
else:
    op = AttributeSuppressionOperation(
        field_name=columns_to_remove[0],
        additional_fields=columns_to_remove[1:]
    )
```

## 10. Dask Support

### 10.1 Dask Behavior
While attribute suppression is inherently simple for Dask, the module provides full support:

```python
# For large datasets
op = AttributeSuppressionOperation(
    field_name="large_text_field",
    additional_fields=["another_large_field"],
    engine="dask"
)
```

### 10.2 Limitations with Dask
- Schema metadata collection is limited (no value statistics)
- Only column names are preserved in metadata
- No memory usage statistics available

## 11. Security Considerations

### 11.1 Schema Information Security
- Schema may reveal sensitive information about data structure
- Use encryption for schema files when needed
- Limit access to audit directory
- Consider schema sanitization for external sharing

### 11.2 Compliance Tracking
```python
# Track GDPR Article 17 compliance
op = AttributeSuppressionOperation(
    field_name="personal_data",
    additional_fields=gdpr_fields,
    save_suppressed_schema=True
)

# Schema provides evidence of data removal
result = op.execute(data_source, task_dir)

# Log for compliance
compliance_log.info(
    f"Removed {result.metrics['columns_suppressed']} columns "
    f"per GDPR Article 17 request. Schema saved at: "
    f"{result.artifacts[0].path}"
)
```

## 12. Comparison with Other Suppression Types

| Aspect | Attribute Suppression | Record Suppression | Cell Suppression |
|--------|----------------------|-------------------|------------------|
| **Scope** | Entire columns | Entire rows | Individual values |
| **Use Case** | Remove identifiers | Remove outliers | Hide specific values |
| **Performance** | Fastest | Moderate | Slowest |
| **Data Loss** | High (entire feature) | Moderate | Low |
| **Reversibility** | No | No | Partial |

## 13. Troubleshooting

### 13.1 Common Issues

**Issue:** Duplicate columns in removal list
```python
# Solution: Operation handles duplicates automatically
op = AttributeSuppressionOperation(
    field_name="ssn",
    additional_fields=["phone", "ssn", "email"]  # "ssn" duplicated
)
# Will remove "ssn" only once with warning
```

**Issue:** All columns removed
```python
# Solution: Validate removal list
total_columns = len(df.columns)
columns_to_remove = len(removal_list)

if columns_to_remove >= total_columns:
    raise ValueError("Cannot remove all columns from DataFrame")
```

**Issue:** Schema file too large
```python
# Solution: Disable detailed statistics
# Currently requires code modification, planned for future parameter
```

### 13.2 Debug Mode
```python
import logging

# Enable debug logging
logger = logging.getLogger("pamola_core.anonymization.suppression")
logger.setLevel(logging.DEBUG)

# Operation will log:
# - Columns to be removed
# - Duplicate detection
# - Schema collection progress
# - File writing operations
```

## 14. Future Enhancements

### 14.1 Planned Features
- Conditional column removal based on statistics
- Pattern-based column selection (regex)
- Column dependency detection
- Reversible suppression with secure storage
- Streaming support for very wide datasets

### 14.2 API Stability
The current API is stable. Future versions will maintain backward compatibility while adding optional parameters for new features.

## 15. Performance Tips

### 15.1 Optimization Strategies
1. **Batch Multiple Removals**: Remove all columns in one operation
2. **Order in Pipeline**: Place early to reduce data size for subsequent operations
3. **Skip Schema for Temporary**: Disable schema collection for intermediate steps
4. **Use Column Patterns**: Plan removal patterns to minimize operations

### 15.2 Memory Management
```python
# For very wide datasets (1000+ columns)
# Remove in groups to manage metadata collection
chunk_size = 100
columns = columns_to_remove

for i in range(0, len(columns), chunk_size):
    chunk = columns[i:i + chunk_size]
    op = AttributeSuppressionOperation(
        field_name=chunk[0],
        additional_fields=chunk[1:] if len(chunk) > 1 else None
    )
    data_source = DataSource(result.artifacts[0].path)  # Chain operations
```

---

**Document Version:** 1.0.0  
**Last Reviewed:** 2025-06-15  
**Next Review:** 2025-09-15  
**Maintainers:** PAMOLA Core Team