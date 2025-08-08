# Consistent Mapping Pseudonymization Operation Documentation

## Overview

The `ConsistentMappingPseudonymizationOperation` class implements reversible pseudonymization using encrypted mapping storage. It maintains a bidirectional mapping between original values and generated pseudonyms, enabling data re-identification when authorized. This is essential for scenarios requiring data linkage across systems or time periods while maintaining privacy.

**Module:** `pamola_core.anonymization.pseudonymization.mapping_op`  
**Version:** 1.0.2  
**Status:** Stable  
**License:** BSD 3-Clause

## Table of Contents

1. [Features](#features)
2. [Installation & Dependencies](#installation--dependencies)
3. [Basic Usage](#basic-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Mapping Storage](#mapping-storage)
6. [Security Considerations](#security-considerations)
7. [Performance Optimization](#performance-optimization)
8. [Examples](#examples)
9. [API Reference](#api-reference)
10. [Metrics & Monitoring](#metrics--monitoring)
11. [Troubleshooting](#troubleshooting)

## Features

### Core Features
- **Reversible Transformation**: Encrypted mapping storage enables authorized re-identification
- **Multiple Pseudonym Types**: UUID, sequential numbers, random strings
- **Encrypted Storage**: All mappings encrypted with AES-256-GCM
- **Thread-Safe**: Concurrent access support with proper locking
- **Atomic Operations**: Prevents mapping corruption during updates
- **Compound Identifiers**: Support for multi-field pseudonymization
- **Batch Processing**: Efficient handling of large datasets
- **Automatic Persistence**: Configurable save frequency

### Framework Integration
- Full PAMOLA.CORE framework compatibility
- Comprehensive metrics collection
- Automatic visualization generation
- Progress tracking and reporting
- Conditional processing support
- Risk-based processing integration

## Installation & Dependencies

### Required Dependencies
```python
# Core dependencies
pandas >= 1.3.0
numpy >= 1.20.0
threading  # Built-in
uuid       # Built-in
pathlib    # Built-in
```

### Framework Dependencies
The operation requires these PAMOLA.CORE modules:
- `pamola_core.anonymization.base_anonymization_op`
- `pamola_core.anonymization.commons.mapping_storage`
- `pamola_core.anonymization.commons.pseudonymization_utils`
- `pamola_core.utils.crypto_helpers.pseudonymization`
- `pamola_core.utils.ops.*`

## Basic Usage

### Simple Example
```python
from pamola_core.anonymization.pseudonymization.mapping_op import ConsistentMappingPseudonymizationOperation

# Create operation with UUID pseudonyms
operation = ConsistentMappingPseudonymizationOperation(
    field_name="customer_id",
    encryption_key="0123456789abcdef" * 4,  # 256-bit key (32 bytes as hex)
    pseudonym_type="uuid"
)

# Execute operation
result = operation.execute(
    data_source=data_source,
    task_dir=Path("./output"),
    progress_tracker=tracker
)

# Reverse a pseudonym (when authorized)
original_value = operation.get_reverse_mapping("550e8400-e29b-41d4-a716-446655440000")
```

### With Custom Configuration
```python
# Sequential pseudonyms with prefix
operation = ConsistentMappingPseudonymizationOperation(
    field_name="patient_id",
    encryption_key=encryption_key_bytes,  # 32 bytes
    pseudonym_type="sequential",
    pseudonym_prefix="PAT-",
    pseudonym_suffix="-2025",
    persist_frequency=500,  # Save every 500 new mappings
    backup_on_update=True,  # Keep backup before updates
    mapping_format="json"   # Use JSON format
)
```

## Advanced Configuration

### Pseudonym Types

#### 1. UUID (Default)
```python
pseudonym_type="uuid"
# Output: "550e8400-e29b-41d4-a716-446655440000"
```

#### 2. Sequential
```python
pseudonym_type="sequential",
pseudonym_prefix="ID-",
pseudonym_suffix=""
# Output: "ID-000001", "ID-000002", ...
```

Sequential counter is preserved across sessions in mapping metadata.

#### 3. Random String
```python
pseudonym_type="random_string",
pseudonym_length=12,  # Total length including prefix/suffix
pseudonym_prefix="USR_"
# Output: "USR_a3F9kL2m"
```

**Note:** Effective random part must be at least 4 characters after prefix/suffix.

### Processing Modes

#### REPLACE Mode (Default)
Replaces original values with pseudonyms:
```python
mode="REPLACE"
# Before: {"patient_id": "12345"}
# After:  {"patient_id": "PAT-000001"}
```

#### ENRICH Mode
Adds pseudonyms in a new field:
```python
mode="ENRICH",
output_field_name="patient_pseudo"
# Before: {"patient_id": "12345"}
# After:  {"patient_id": "12345", "patient_pseudo": "PAT-000001"}
```

### Compound Identifiers

Combine multiple fields into a single pseudonym:
```python
operation = ConsistentMappingPseudonymizationOperation(
    field_name="user_id",
    additional_fields=["department", "location"],
    compound_mode=True,
    compound_separator="|",
    compound_null_handling="skip",  # skip, empty, or null
    encryption_key=your_key
)
# Combines: "john_doe|sales|NYC" → "USR_x7K9mP3n"
```

### Conditional Processing

Process only records meeting specific conditions:
```python
operation = ConsistentMappingPseudonymizationOperation(
    field_name="employee_id",
    encryption_key=your_key,
    condition_field="department",
    condition_values=["HR", "Finance"],
    condition_operator="in"  # Default when condition_field is set
)
```

## Mapping Storage

### File Organization
```
{task_dir}/
├── maps/                           # Mapping storage directory
│   ├── customer_id_mapping.csv.enc # Encrypted mapping file
│   └── customer_id_mapping.csv.bak # Backup (if enabled)
├── output/                         # Processed data output
│   └── pseudonymized_data.csv
└── metrics/                        # Operation metrics
```

### Mapping File Format

Mappings are stored encrypted with optional metadata:
```json
{
  "mappings": {
    "original_value1": "pseudonym1",
    "original_value2": "pseudonym2"
  },
  "_metadata": {
    "last_sequential": 42,
    "total_mappings": 1000,
    "last_updated": "2025-06-15T10:30:00",
    "pseudonym_type": "sequential",
    "version": "1.0.2"
  }
}
```

### Persistence Strategy

Mappings are persisted:
1. Automatically after N new mappings (configurable via `persist_frequency`)
2. At the end of operation execution
3. Manually via `_persist_mappings()`

## Security Considerations

### 1. Encryption Key Management
- **Required**: 256-bit (32 bytes) encryption key
- **Format**: Hex string or raw bytes
- **Validation**: Key size is validated on initialization
- **Storage**: Never store keys in code or logs

```python
# Good: Load from secure storage
with open('/secure/path/key.txt', 'r') as f:
    encryption_key = f.read().strip()

# Bad: Hardcoded key
encryption_key = "0123456789abcdef" * 4  # Don't do this in production!
```

### 2. Mapping File Security
- All mappings encrypted with AES-256-GCM
- Atomic file operations prevent corruption
- Optional backups before updates
- File permissions should be restricted (600)

### 3. Thread Safety
- All mapping operations are thread-safe
- Uses `threading.RLock()` for recursive locking
- Safe for concurrent batch processing

### 4. Memory Security
- No plaintext mappings in logs
- Secure error messages (no key details exposed)
- Automatic cleanup on operation completion

## Performance Optimization

### 1. Batch Processing
```python
batch_size=50000  # Larger batches for better performance
```

Considerations:
- Larger batches reduce overhead
- Memory usage increases with batch size
- Automatic cleanup for batches > 50k records

### 2. Persistence Frequency
```python
persist_frequency=1000  # Save every 1000 new mappings
```

Trade-offs:
- Higher frequency = more I/O, better durability
- Lower frequency = better performance, risk of data loss

### 3. Mapping as Cache
- No separate caching needed - mappings serve as cache
- O(1) lookup time for existing mappings
- Hit rate tracked in metrics

### 4. Memory Optimization
```python
# Automatic DataFrame optimization before output
df, info = optimize_dataframe_dtypes(df)

# Force garbage collection for large datasets
if len(batch) > 50000:
    force_garbage_collection()
```

## Examples

### Example 1: Basic Customer ID Pseudonymization
```python
# Pseudonymize customer IDs with UUIDs
operation = ConsistentMappingPseudonymizationOperation(
    field_name="customer_id",
    encryption_key="your-256-bit-key-here",
    pseudonym_type="uuid",
    mapping_file="customer_mappings.json.enc",
    mapping_format="json"
)

result = operation.execute(data_source, task_dir)
print(f"Created {result.metrics['new_mappings_created']} new mappings")
print(f"Total mappings: {result.metrics['total_mappings']}")
```

### Example 2: Sequential Patient IDs
```python
# Generate sequential patient IDs
operation = ConsistentMappingPseudonymizationOperation(
    field_name="patient_mrn",
    encryption_key=load_encryption_key(),
    pseudonym_type="sequential",
    pseudonym_prefix="PAT",
    pseudonym_suffix=f"-{datetime.now().year}",
    persist_frequency=100,  # Save frequently for medical data
    backup_on_update=True
)

# Sequential numbers continue from last saved value
# Output: PAT000001-2025, PAT000002-2025, ...
```

### Example 3: Multi-field Employee IDs
```python
# Create compound employee pseudonyms
operation = ConsistentMappingPseudonymizationOperation(
    field_name="employee_id",
    additional_fields=["department", "hire_year"],
    compound_mode=True,
    compound_separator="-",
    encryption_key=company_key,
    pseudonym_type="random_string",
    pseudonym_length=10,
    pseudonym_prefix="EMP_"
)

# Combines fields before pseudonymization
# "12345-IT-2020" → "EMP_x7K9mP"
```

### Example 4: Risk-Based Processing
```python
# Only pseudonymize high-risk records
operation = ConsistentMappingPseudonymizationOperation(
    field_name="user_id",
    encryption_key=security_key,
    ka_risk_field="k_anonymity_score",
    risk_threshold=5.0,  # Process if k < 5
    vulnerable_record_strategy="pseudonymize",
    mode="ENRICH",
    output_field_name="secure_id"
)
```

### Example 5: Data Recovery
```python
# Export mappings for backup
operation.export_mappings(
    output_path=Path("backup/mappings_2025.enc"),
    include_metadata=True
)

# Later: Reverse pseudonyms
for pseudonym in pseudonyms_to_reverse:
    original = operation.get_reverse_mapping(pseudonym)
    if original:
        print(f"{pseudonym} → {original}")
```

## API Reference

### Class: ConsistentMappingPseudonymizationOperation

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | str | Required | Primary field to pseudonymize |
| `encryption_key` | Union[str, bytes] | Required | 256-bit encryption key |
| `additional_fields` | List[str] | None | Additional fields for compound mode |
| `mapping_file` | Union[str, Path] | None | Custom mapping file name |
| `mapping_format` | str | "csv" | Storage format: "csv" or "json" |
| `pseudonym_type` | str | "uuid" | Type: "uuid", "sequential", "random_string" |
| `pseudonym_prefix` | str | None | Prefix for pseudonyms |
| `pseudonym_suffix` | str | None | Suffix for pseudonyms |
| `pseudonym_length` | int | 36 | Total length for random_string type |
| `create_if_not_exists` | bool | True | Create mapping file if missing |
| `backup_on_update` | bool | True | Backup before updates |
| `persist_frequency` | int | 1000 | Save after N new mappings |
| `mode` | str | "REPLACE" | Processing mode: "REPLACE" or "ENRICH" |
| `output_field_name` | str | None | Output field for ENRICH mode |
| `null_strategy` | str | "PRESERVE" | Null handling strategy |
| `batch_size` | int | 10000 | Batch size for processing |
| `condition_field` | str | None | Field for conditional processing |
| `condition_values` | List | None | Values for condition |
| `condition_operator` | str | None | Operator (default: "in" if condition_field set) |

#### Methods

##### execute()
```python
def execute(self, data_source, task_dir, reporter=None, 
            progress_tracker=None, **kwargs) -> OperationResult
```
Execute the pseudonymization operation.

**Parameters:**
- `data_source`: Data source containing the DataFrame
- `task_dir`: Directory for output and artifacts
- `reporter`: Optional reporter for notifications
- `progress_tracker`: Optional progress tracker
- `**kwargs`: Additional options

**Returns:** `OperationResult` with status, metrics, and artifacts

##### get_reverse_mapping()
```python
def get_reverse_mapping(self, pseudonym: str) -> Optional[str]
```
Get original value for a pseudonym (requires authorization).

**Parameters:**
- `pseudonym`: Pseudonym to reverse

**Returns:** Original value if found, None otherwise

##### export_mappings()
```python
def export_mappings(self, output_path: Path, 
                   include_metadata: bool = True) -> None
```
Export encrypted mappings for backup or transfer.

**Parameters:**
- `output_path`: Export file path
- `include_metadata`: Include metadata (recommended)

## Metrics & Monitoring

### Collected Metrics

The operation automatically collects comprehensive metrics:

#### Performance Metrics
- `execution_time`: Total execution time
- `records_processed`: Number of records processed
- `records_per_second`: Processing throughput
- `batch_count`: Number of batches processed

#### Mapping Metrics
- `total_mappings`: Current total mappings
- `new_mappings_created`: New mappings in this execution
- `mapping_hit_rate`: Cache hit rate (0.0 to 1.0)
- `percent_new_mappings`: Percentage of lookups creating new mappings
- `lookup_time_avg`: Average lookup time in milliseconds
- `collision_count`: Collisions for random_string type
- `mapping_file_size`: Size of encrypted mapping file
- `mapping_file_path`: Full path to mapping file

#### Privacy Metrics
- `k_anonymity`: K-anonymity score (if quasi-identifiers provided)
- `l_diversity`: L-diversity score
- `disclosure_risk`: Simple disclosure risk
- `unique_before`: Unique values before pseudonymization
- `unique_after`: Unique values after pseudonymization

### Accessing Metrics
```python
result = operation.execute(data_source, task_dir)

# Access metrics
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Hit rate: {result.metrics['mapping_hit_rate']:.2%}")
print(f"New mappings: {result.metrics['new_mappings_created']}")
print(f"Avg lookup time: {result.metrics['lookup_time_avg']:.2f}ms")
```

### Generated Artifacts

1. **Encrypted Mapping File**: In `task_dir/maps/`
2. **Metrics JSON**: Detailed metrics in `task_dir/`
3. **Comparison Visualization**: Before/after uniqueness comparison
4. **Mapping Growth Chart**: Visual of mapping statistics
5. **Pseudonymized Dataset**: Output data in `task_dir/output/`

## Troubleshooting

### Common Issues

#### 1. Invalid Encryption Key
```python
# Error: Invalid encryption key size
# Solution: Ensure key is exactly 32 bytes (256 bits)
key = "0123456789abcdef" * 4  # 64 hex chars = 32 bytes
```

#### 2. Sequential Counter Reset
```python
# Issue: Sequential numbers restart after mapping file deletion
# Solution: The counter is now preserved in metadata
# Check _metadata.last_sequential in mapping file
```

#### 3. Insufficient Pseudonym Length
```python
# Error: Pseudonym length (8) minus prefix/suffix (6) must be at least 4 characters
# Solution: Increase pseudonym_length or reduce prefix/suffix
pseudonym_length=12,  # Allows for longer prefix/suffix
```

#### 4. High Collision Rate
```python
# Warning: High collision rate detected for random_string generation
# Solutions:
1. Increase pseudonym_length
2. Use UUID type for guaranteed uniqueness
3. Check collision_count in metrics
```

#### 5. Mapping File Corruption
```python
# Enable backups to prevent data loss
backup_on_update=True

# Recovery from backup:
# Rename {mapping_file}.bak to {mapping_file}
```

### Performance Issues

1. **Slow Processing**
   - Increase `batch_size` if memory allows
   - Reduce `persist_frequency` for fewer I/O operations
   - Check `lookup_time_avg` metric

2. **High Memory Usage**
   - Reduce `batch_size`
   - Enable DataFrame optimization
   - Consider splitting large datasets

3. **Frequent Saves**
   - Increase `persist_frequency`
   - Balance between performance and durability

### Debug Mode
```python
# Enable detailed logging
import logging
logging.getLogger("pamola_core.anonymization").setLevel(logging.DEBUG)

# Monitor mapping operations
logging.getLogger("pamola_core.anonymization.commons.mapping_storage").setLevel(logging.DEBUG)
```

## Best Practices

1. **Key Management**
   - Store encryption keys securely (HSM, key vault, etc.)
   - Rotate keys periodically with proper migration
   - Never log or hardcode keys

2. **Mapping Maintenance**
   - Regular backups of mapping files
   - Monitor mapping file growth
   - Archive old mappings if needed

3. **Performance**
   - Set `persist_frequency` based on data criticality
   - Use appropriate `batch_size` for your system
   - Monitor metrics for optimization opportunities

4. **Security**
   - Restrict file permissions on mapping files
   - Use strong encryption keys (randomly generated)
   - Implement access controls for reversal operations

5. **Testing**
   - Test reversal functionality before production
   - Verify mapping persistence across sessions
   - Check sequential counter continuity

## Version History

- **1.0.0** (2025-01-20): Initial implementation
- **1.0.1** (2025-06-15): Updated validation framework imports
- **1.0.2** (2025-06-15): Multiple fixes including condition normalization, accurate metrics, and sequential counter persistence

## References

- [AES-256-GCM Encryption](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf)
- [GDPR Pseudonymization Requirements](https://ec.europa.eu/info/law/law-topic/data-protection)
- [PAMOLA Framework Documentation](../../../README.md)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in `task_dir/logs/`
3. Verify mapping file integrity
4. Consult PAMOLA.CORE documentation
5. Contact the PAMOLA Core Team