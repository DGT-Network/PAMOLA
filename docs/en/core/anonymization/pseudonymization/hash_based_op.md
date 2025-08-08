# Hash-Based Pseudonymization Operation Documentation

## Overview

The `HashBasedPseudonymizationOperation` class implements irreversible pseudonymization using cryptographic hash functions. It transforms sensitive identifiers into pseudonyms that cannot be reversed to obtain the original values, providing strong privacy protection while maintaining referential integrity.

**Module:** `pamola_core.anonymization.pseudonymization.hash_based_op`  
**Version:** 1.0.1  
**Status:** Stable  
**License:** BSD 3-Clause

## Table of Contents

1. [Features](#features)
2. [Installation & Dependencies](#installation--dependencies)
3. [Basic Usage](#basic-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Security Considerations](#security-considerations)
6. [Performance Optimization](#performance-optimization)
7. [Examples](#examples)
8. [API Reference](#api-reference)
9. [Metrics & Monitoring](#metrics--monitoring)
10. [Troubleshooting](#troubleshooting)

## Features

### Core Features
- **Irreversible Transformation**: Uses SHA3-256/512 cryptographic hash functions
- **Salt & Pepper**: Configurable salt (persistent) and pepper (session-based) for enhanced security
- **Multiple Output Formats**: hex, base64, base32, base58, UUID
- **Compound Identifiers**: Support for multi-field pseudonymization
- **Batch Processing**: Efficient processing of large datasets
- **Caching**: LRU cache for performance optimization
- **Risk-Based Processing**: Integration with k-anonymity risk scores

### Framework Integration
- Full PAMOLA.CORE framework compatibility
- Comprehensive metrics collection
- Automatic visualization generation
- Progress tracking and reporting
- Conditional processing support

## Installation & Dependencies

### Required Dependencies
```python
# Core dependencies
pandas >= 1.3.0
numpy >= 1.20.0
hashlib  # Built-in
base64   # Built-in
uuid     # Built-in

# Optional dependencies
base58  # For base58 output format (optional)
```

### Framework Dependencies
The operation requires these PAMOLA.CORE modules:
- `pamola_core.anonymization.base_anonymization_op`
- `pamola_core.anonymization.commons.*`
- `pamola_core.utils.crypto_helpers.pseudonymization`
- `pamola_core.utils.ops.*`

## Basic Usage

### Simple Example
```python
from pamola_core.anonymization.pseudonymization.hash_based_op import HashBasedPseudonymizationOperation

# Create operation with default settings
operation = HashBasedPseudonymizationOperation(
    field_name="customer_id",
    algorithm="sha3_256",
    salt_config={
        "source": "parameter",
        "value": "0123456789abcdef" * 4  # 32-byte salt as hex
    }
)

# Execute operation
result = operation.execute(
    data_source=data_source,
    task_dir=Path("./output"),
    progress_tracker=tracker
)
```

### With Custom Configuration
```python
# Advanced configuration
operation = HashBasedPseudonymizationOperation(
    field_name="email",
    algorithm="sha3_512",  # Stronger hash
    salt_config={
        "source": "file",
        "field_name": "email"
    },
    salt_file=Path("config/salts.json"),
    use_pepper=True,
    pepper_length=32,
    output_format="base64",
    output_length=16,  # Truncate to 16 characters
    output_prefix="USR_",
    mode="ENRICH",  # Keep original field
    output_field_name="email_pseudo",
    batch_size=50000,
    use_cache=True,
    cache_size=200000
)
```

## Advanced Configuration

### Salt Configuration

Salt provides protection against rainbow table attacks. Two sources are supported:

#### 1. Parameter-based Salt
```python
salt_config = {
    "source": "parameter",
    "value": "0123456789abcdef" * 4  # 32-byte hex string
}
```

#### 2. File-based Salt
```python
# salts.json
{
    "email": "abcdef0123456789" * 4,
    "phone": "fedcba9876543210" * 4,
    "ssn": "1234567890abcdef" * 4
}

# Configuration
salt_config = {
    "source": "file",
    "field_name": "email"  # Which salt to use from file
}
```

### Pepper Configuration

Pepper adds session-specific security:
```python
use_pepper=True,        # Enable pepper
pepper_length=32        # Length in bytes (default: 32)
```

### Output Formats

| Format | Description | Example |
|--------|-------------|---------|
| `hex` | Hexadecimal (default) | `a3f5b8c9d2e1...` |
| `base64` | Base64 URL-safe | `o_W4ydLh...` |
| `base32` | Base32 encoding | `UH5XQZ4K...` |
| `base58` | Base58 (Bitcoin-style) | `9kHfWc8L...` |
| `uuid` | UUID format | `a3f5b8c9-d2e1-4567-89ab-cdef01234567` |

### Processing Modes

#### REPLACE Mode (Default)
Replaces the original field with pseudonyms:
```python
mode="REPLACE"
# Before: {"email": "user@example.com"}
# After:  {"email": "a3f5b8c9d2e1..."}
```

#### ENRICH Mode
Adds pseudonyms in a new field:
```python
mode="ENRICH",
output_field_name="email_pseudo"
# Before: {"email": "user@example.com"}
# After:  {"email": "user@example.com", "email_pseudo": "a3f5b8c9d2e1..."}
```

### Compound Identifiers

Combine multiple fields into a single pseudonym:
```python
operation = HashBasedPseudonymizationOperation(
    field_name="customer_id",
    additional_fields=["region", "account_type"],
    compound_mode=True,
    compound_separator="|",
    compound_null_handling="skip"  # skip, empty, or null
)
```

## Security Considerations

### 1. Salt Management
- **Never share salt values** in logs or error messages
- Store salt files with restricted permissions (600)
- Use different salts for different fields when possible
- Rotate salts periodically (requires re-pseudonymization)

### 2. Pepper Security
- Pepper is generated per session using cryptographically secure random
- Automatically cleared from memory after operation
- Different pepper for each execution ensures additional randomization

### 3. Hash Algorithm Selection
- **SHA3-256**: Default, suitable for most use cases
- **SHA3-512**: Higher security, larger output, slightly slower

### 4. Collision Handling
```python
# Monitor for collisions
check_collisions=True,
collision_strategy="log"  # or "fail" to stop on collision
```

### 5. Memory Security
- Sensitive data (salt, pepper) stored in `SecureBytes` objects
- Automatic memory cleanup on operation completion
- No plaintext values in logs or debug output

## Performance Optimization

### 1. Caching
```python
use_cache=True,         # Enable caching
cache_size=100000       # Maximum entries (default: 100k)
```

Cache metrics are automatically collected:
- Hit rate
- Size
- Performance impact

### 2. Batch Processing
```python
batch_size=50000        # Larger batches for better performance
```

Considerations:
- Larger batches = better performance but more memory
- Automatic memory cleanup for batches > 50k records

### 3. Memory Optimization
```python
# Automatic DataFrame optimization
df, info = optimize_dataframe_dtypes(df)

# Force garbage collection for large datasets
if len(batch) > 50000:
    force_garbage_collection()
```

### 4. Conditional Processing
Process only specific records:
```python
condition_field="country",
condition_values=["US", "CA", "UK"],
condition_operator="in"
```

## Examples

### Example 1: Basic Email Pseudonymization
```python
# Pseudonymize email addresses
operation = HashBasedPseudonymizationOperation(
    field_name="email",
    algorithm="sha3_256",
    salt_config={
        "source": "parameter",
        "value": "your-secret-salt-value" * 2
    },
    output_format="hex",
    output_length=16  # First 16 characters only
)

result = operation.execute(data_source, task_dir)
print(f"Processed {result.metrics['records_processed']} records")
```

### Example 2: Multi-field Customer ID
```python
# Create compound customer pseudonym
operation = HashBasedPseudonymizationOperation(
    field_name="customer_id",
    additional_fields=["store_id", "signup_date"],
    compound_mode=True,
    compound_separator="-",
    algorithm="sha3_512",
    salt_config={
        "source": "file",
        "field_name": "customer_compound"
    },
    salt_file=Path("config/compound_salts.json"),
    output_format="uuid"
)
```

### Example 3: Risk-Based Pseudonymization
```python
# Only pseudonymize high-risk records
operation = HashBasedPseudonymizationOperation(
    field_name="patient_id",
    ka_risk_field="k_anonymity_score",
    risk_threshold=5.0,  # k < 5 is vulnerable
    vulnerable_record_strategy="pseudonymize",
    algorithm="sha3_512",  # Stronger for sensitive data
    use_pepper=True,
    pepper_length=64  # Extra security
)
```

### Example 4: Formatted Output
```python
# Create user-friendly pseudonyms
operation = HashBasedPseudonymizationOperation(
    field_name="account_number",
    output_format="base58",
    output_length=12,
    output_prefix="ACC-",
    output_suffix="-2025"
)
# Output example: "ACC-9kHfWc8LmN3p-2025"
```

## API Reference

### Class: HashBasedPseudonymizationOperation

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | str | Required | Primary field to pseudonymize |
| `additional_fields` | List[str] | None | Additional fields for compound mode |
| `algorithm` | str | "sha3_256" | Hash algorithm: "sha3_256" or "sha3_512" |
| `salt_config` | Dict | See below | Salt configuration |
| `salt_file` | Path | None | Path to salt file (for file source) |
| `use_pepper` | bool | True | Enable session pepper |
| `pepper_length` | int | 32 | Pepper length in bytes |
| `output_format` | str | "hex" | Output format (see formats table) |
| `output_length` | int | None | Truncate output to N characters |
| `output_prefix` | str | None | Prefix for pseudonyms |
| `output_suffix` | str | None | Suffix for pseudonyms |
| `mode` | str | "REPLACE" | Processing mode: "REPLACE" or "ENRICH" |
| `output_field_name` | str | None | Output field for ENRICH mode |
| `null_strategy` | str | "PRESERVE" | Null handling strategy |
| `batch_size` | int | 10000 | Batch size for processing |
| `use_cache` | bool | True | Enable caching |
| `cache_size` | int | 100000 | Maximum cache entries |

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

##### process_batch()
```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame
```
Process a single batch of data.

**Parameters:**
- `batch`: DataFrame batch to process

**Returns:** Processed DataFrame with pseudonyms

## Metrics & Monitoring

### Collected Metrics

The operation automatically collects comprehensive metrics:

#### Performance Metrics
- `execution_time`: Total execution time
- `records_processed`: Number of records processed
- `records_per_second`: Processing throughput
- `batch_count`: Number of batches processed

#### Pseudonymization Metrics
- `algorithm`: Hash algorithm used
- `output_format`: Output format used
- `collision_probability`: Estimated collision probability
- `collision_count`: Actual collisions detected
- `unique_pseudonyms`: Number of unique pseudonyms generated
- `hash_computation_time`: Time spent in hash computation

#### Cache Metrics (if enabled)
- `cache_size`: Current cache size
- `cache_hit_rate`: Cache hit percentage
- `hits`: Number of cache hits
- `misses`: Number of cache misses

#### Privacy Metrics
- `k_anonymity`: K-anonymity score
- `l_diversity`: L-diversity score
- `disclosure_risk`: Simple disclosure risk
- `suppression_rate`: Percentage of suppressed values

### Accessing Metrics
```python
result = operation.execute(data_source, task_dir)

# Access metrics
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Cache hit rate: {result.metrics['cache']['hit_rate']:.2%}")
print(f"Collision probability: {result.metrics['collision_probability']:.2e}")
```

### Generated Artifacts

1. **Metrics JSON**: Detailed metrics in `task_dir/metrics/`
2. **Comparison Visualization**: Before/after uniqueness comparison
3. **Cache Performance Chart**: Cache hit/miss visualization
4. **Pseudonymized Dataset**: Output data file

## Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Error: Cannot find reference 'validate_field_type'
# Solution: Update to use validation_utils facade
from pamola_core.anonymization.commons.validation_utils import check_field_exists
```

#### 2. Hash Collisions
```python
# Warning: Hash collision detected
# Solutions:
# 1. Use sha3_512 for larger hash space
# 2. Check for duplicate input values
# 3. Enable collision monitoring
collision_strategy="fail"  # Stop on collision
```

#### 3. Memory Issues
```python
# For large datasets:
batch_size=5000,  # Smaller batches
use_cache=False   # Disable cache if memory constrained
```

#### 4. Performance Issues
```python
# Optimization strategies:
1. Increase batch_size (if memory allows)
2. Enable caching for repeated values
3. Use conditional processing to limit scope
4. Consider parallel processing with Dask
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.getLogger("pamola_core.anonymization").setLevel(logging.DEBUG)

# Monitor progress
progress_tracker = HierarchicalProgressTracker(
    total=100,
    description="Pseudonymization Progress"
)
```

### Error Handling
```python
try:
    result = operation.execute(data_source, task_dir)
    if result.status == OperationStatus.SUCCESS:
        print("Success!")
    else:
        print(f"Error: {result.error_message}")
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # Check logs for detailed traceback
```

## Best Practices

1. **Salt Management**
   - Use strong, random salts (minimum 32 bytes)
   - Store salts securely, separate from data
   - Document salt rotation procedures

2. **Performance**
   - Enable caching for datasets with repeated values
   - Adjust batch size based on available memory
   - Monitor metrics to identify bottlenecks

3. **Security**
   - Always use pepper for additional security
   - Choose SHA3-512 for highly sensitive data
   - Regularly review collision reports

4. **Integration**
   - Use progress trackers for long operations
   - Enable metrics collection for monitoring
   - Implement proper error handling

5. **Testing**
   - Test with sample data first
   - Verify referential integrity is maintained
   - Check output format meets requirements

## Version History

- **1.0.0** (2025-01-20): Initial implementation
- **1.0.1** (2025-06-15): Fixed validation framework imports

## References

- [SHA-3 Standard (FIPS 202)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf)
- [GDPR Pseudonymization Guidelines](https://ec.europa.eu/info/law/law-topic/data-protection)
- [PAMOLA Framework Documentation](../../../README.md)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in `task_dir/logs/`
3. Consult PAMOLA.CORE documentation
4. Contact the PAMOLA Core Team