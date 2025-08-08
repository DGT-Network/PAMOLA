# PAMOLA Categorical Generalization Configuration Documentation

## Module Information
- **Name**: `categorical_config.py`
- **Package**: `pamola_core.anonymization.generalization`
- **Version**: 1.1.0
- **Status**: Stable
- **License**: BSD 3-Clause

## Purpose

The Categorical Generalization Configuration module provides comprehensive configuration management for categorical generalization operations within the PAMOLA framework. It ensures type-safe, validated configuration for all categorical anonymization strategies while supporting advanced features like:

- **Multiple Generalization Strategies**: Hierarchy-based, frequency-based, and low-frequency merging
- **Flexible NULL Handling**: Preserve, exclude, anonymize, or error strategies
- **Large-Scale Processing**: Automatic Dask integration for big data
- **Template-Based Anonymization**: Configurable patterns for rare values
- **Privacy Threshold Management**: Built-in k-anonymity and disclosure risk controls

## Architecture

### Class Hierarchy

```
OperationConfig (base)
    └── CategoricalGeneralizationConfig
         ├── Configuration Fields (40+ parameters)
         ├── JSON Schema Validation
         ├── Strategy-Specific Logic
         └── Business Rule Validation
```

### Key Components

1. **Configuration Dataclass**: Type-safe parameter storage with defaults
2. **JSON Schema**: Comprehensive validation rules with Draft-07 compliance
3. **Enum Types**: Validated choices for strategies and options
4. **Validation System**: Multi-level validation (schema + business rules)
5. **Parameter Groups**: Logical organization of related settings

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *Required* | Field to generalize |
| `strategy` | `str` | `"hierarchy"` | Generalization strategy |
| `mode` | `str` | `"REPLACE"` | Operation mode (REPLACE/ENRICH) |
| `output_field_name` | `Optional[str]` | `None` | Output field for ENRICH mode |

### Strategy-Specific Parameters

#### Hierarchy Strategy
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `external_dictionary_path` | `Optional[str]` | `None` | Path to hierarchy dictionary |
| `dictionary_format` | `str` | `"auto"` | Format (auto/json/csv) |
| `hierarchy_level` | `int` | `1` | Generalization level (1-5) |
| `fuzzy_matching` | `bool` | `False` | Enable fuzzy matching |
| `similarity_threshold` | `float` | `0.85` | Fuzzy match threshold |

#### Frequency-Based Strategies
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_group_size` | `int` | `10` | Minimum group size |
| `freq_threshold` | `float` | `0.01` | Frequency threshold (0-1) |
| `max_categories` | `int` | `1000000` | Maximum categories to keep |
| `group_rare_as` | `str` | `"OTHER"` | Rare grouping strategy |

### Unknown Value Handling
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_unknown` | `bool` | `True` | Allow unknown values |
| `unknown_value` | `str` | `"OTHER"` | Default unknown label |
| `rare_value_template` | `str` | `"OTHER_{n}"` | Template for numbered groups |
| `null_strategy` | `str` | `"PRESERVE"` | NULL handling strategy |

### Performance & Engine
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `str` | `"auto"` | Processing engine (pandas/dask/auto) |
| `max_rows_in_memory` | `int` | `1000000` | Rows before Dask switch |
| `dask_chunk_size` | `int` | `50000` | Dask processing chunk size |
| `batch_size` | `int` | `10000` | Processing batch size |
| `adaptive_batch_size` | `bool` | `True` | Dynamic batch sizing |

### Privacy Controls
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `privacy_check_enabled` | `bool` | `True` | Enable privacy validation |
| `min_acceptable_k` | `int` | `5` | Minimum k-anonymity |
| `max_acceptable_disclosure_risk` | `float` | `0.2` | Max disclosure risk (0-1) |
| `quasi_identifiers` | `Optional[List[str]]` | `None` | QI fields for privacy checks |

## Enumerations

### GeneralizationStrategy
```python
HIERARCHY = "hierarchy"          # Use external hierarchy dictionary
MERGE_LOW_FREQ = "merge_low_freq"  # Merge low frequency categories
FREQUENCY_BASED = "frequency_based" # Group by frequency bands
```

### NullStrategy
```python
PRESERVE = "PRESERVE"    # Keep NULL values as-is
EXCLUDE = "EXCLUDE"      # Remove NULL values
ANONYMIZE = "ANONYMIZE"  # Replace with unknown_value
ERROR = "ERROR"          # Raise error on NULL
```

### OperationMode
```python
REPLACE = "REPLACE"  # Replace original field
ENRICH = "ENRICH"    # Add new field with result
```

### Engine
```python
PANDAS = "pandas"  # Force pandas processing
DASK = "dask"      # Force Dask processing
AUTO = "auto"      # Auto-select based on data size
```

## Usage Examples

### Example 1: Basic Hierarchy Configuration

```python
from pamola_core.anonymization.generalization.categorical_config import (
    CategoricalGeneralizationConfig
)

# Configure hierarchy-based generalization
config = CategoricalGeneralizationConfig(
    field_name="city",
    strategy="hierarchy",
    external_dictionary_path="/data/geo_hierarchy.json",
    hierarchy_level=2,
    fuzzy_matching=True,
    similarity_threshold=0.9
)

# Validate and use
if config.validate():
    params = config.get_strategy_params()
    print(f"Configuration: {config.get_summary()}")
```

### Example 2: Frequency-Based with Templates

```python
# Configure frequency-based grouping
config = CategoricalGeneralizationConfig(
    field_name="product_category",
    strategy="merge_low_freq",
    min_group_size=20,
    rare_value_template="RARE_PRODUCT_{n}",
    max_categories=100,
    null_strategy="ANONYMIZE"
)

# Format rare values
for i in range(5):
    print(config.format_rare_value(i))
# Output: RARE_PRODUCT_1, RARE_PRODUCT_2, etc.
```

### Example 3: ENRICH Mode Configuration

```python
# Add anonymized field without replacing original
config = CategoricalGeneralizationConfig(
    field_name="department",
    strategy="frequency_based",
    mode="ENRICH",
    output_field_name="department_generalized",
    max_categories=10,
    privacy_check_enabled=True,
    min_acceptable_k=10
)
```

### Example 4: Large Data with Auto-Dask

```python
# Configure for large datasets
config = CategoricalGeneralizationConfig(
    field_name="customer_id",
    strategy="merge_low_freq",
    engine="auto",
    max_rows_in_memory=500000,
    dask_chunk_size=25000,
    batch_size=50000,
    adaptive_batch_size=True
)

# Check if Dask will be used
row_count = 1_000_000
use_dask = config.should_use_dask(row_count)
print(f"Use Dask for {row_count} rows: {use_dask}")
```

### Example 5: Export/Import Configuration

```python
# Export configuration
config = CategoricalGeneralizationConfig(
    field_name="diagnosis_code",
    strategy="hierarchy",
    external_dictionary_path="medical_hierarchy.csv",
    privacy_check_enabled=True,
    min_acceptable_k=20
)
config.export_json("diagnosis_config.json")

# Import configuration
loaded_config = CategoricalGeneralizationConfig.import_json(
    "diagnosis_config.json"
)
print(loaded_config.get_summary())
```

## Validation System

### Multi-Level Validation

The configuration uses a three-tier validation system:

1. **JSON Schema Validation**: Structural and type validation
2. **Strategy-Specific Validation**: Parameter consistency checks
3. **Business Rule Validation**: Cross-parameter logic validation

### Validation Examples

```python
# Schema validation catches type errors
try:
    config = CategoricalGeneralizationConfig(
        field_name="test",
        hierarchy_level="invalid"  # Should be int
    )
except ValueError as e:
    print(f"Schema error: {e}")

# Strategy validation ensures required parameters
try:
    config = CategoricalGeneralizationConfig(
        field_name="test",
        strategy="hierarchy"
        # Missing external_dictionary_path
    )
except ValueError as e:
    print(f"Strategy error: {e}")

# Business rule validation
try:
    config = CategoricalGeneralizationConfig(
        field_name="test",
        mode="ENRICH"
        # Missing output_field_name
    )
except ValueError as e:
    print(f"Business rule error: {e}")
```

## Advanced Features

### 1. Conditional Processing

```python
config = CategoricalGeneralizationConfig(
    field_name="salary",
    strategy="merge_low_freq",
    condition_field="department",
    condition_values=["IT", "Finance"],
    condition_operator="in"
)
```

### 2. Risk-Based Processing

```python
config = CategoricalGeneralizationConfig(
    field_name="zip_code",
    strategy="hierarchy",
    ka_risk_field="risk_score",
    risk_threshold=0.1,
    vulnerable_record_strategy="generalize"
)
```

### 3. Custom Privacy Thresholds

```python
config = CategoricalGeneralizationConfig(
    field_name="medical_condition",
    strategy="hierarchy",
    privacy_check_enabled=True,
    min_acceptable_k=50,  # Strict for medical data
    max_acceptable_disclosure_risk=0.05
)
```

### 4. Text Normalization Options

```python
config = CategoricalGeneralizationConfig(
    field_name="company_name",
    strategy="merge_low_freq",
    text_normalization="aggressive",
    case_sensitive=False,
    fuzzy_matching=True
)
```

## Best Practices

### 1. Strategy Selection

| Data Characteristic | Recommended Strategy | Key Parameters |
|-------------------|---------------------|----------------|
| Natural hierarchy exists | `hierarchy` | `hierarchy_level`, `fuzzy_matching` |
| Many rare categories | `merge_low_freq` | `min_group_size`, `rare_value_template` |
| Need specific category count | `frequency_based` | `max_categories` |

### 2. NULL Handling Guidelines

```python
# Healthcare/Financial (strict)
config.null_strategy = "ERROR"  # No NULLs allowed

# Analytics (flexible)
config.null_strategy = "PRESERVE"  # Keep NULLs

# Reporting (clean)
config.null_strategy = "ANONYMIZE"  # Replace with "UNKNOWN"
```

### 3. Performance Optimization

```python
def optimize_config_for_data_size(row_count: int) -> Dict[str, Any]:
    """Optimize configuration based on data size."""
    
    if row_count < 100_000:
        return {
            "engine": "pandas",
            "batch_size": 10_000,
            "adaptive_batch_size": False
        }
    elif row_count < 1_000_000:
        return {
            "engine": "auto",
            "batch_size": 50_000,
            "adaptive_batch_size": True
        }
    else:
        return {
            "engine": "dask",
            "dask_chunk_size": 100_000,
            "batch_size": 100_000
        }
```

### 4. Privacy-Utility Balance

```python
# High privacy, low utility
strict_config = CategoricalGeneralizationConfig(
    field_name="diagnosis",
    strategy="hierarchy",
    hierarchy_level=3,  # High generalization
    min_acceptable_k=20,
    max_acceptable_disclosure_risk=0.01
)

# Balanced privacy-utility
balanced_config = CategoricalGeneralizationConfig(
    field_name="diagnosis",
    strategy="hierarchy",
    hierarchy_level=2,  # Moderate generalization
    min_acceptable_k=10,
    max_acceptable_disclosure_risk=0.1
)
```

## Integration with PAMOLA Operations

### 1. With CategoricalGeneralizationOperation

```python
from pamola_core.anonymization.generalization.categorical_op import (
    CategoricalGeneralizationOperation
)

# Create configuration
config = CategoricalGeneralizationConfig(
    field_name="location",
    strategy="hierarchy",
    external_dictionary_path="geo_hierarchy.json"
)

# Initialize operation with config
operation = CategoricalGeneralizationOperation(**config.to_dict())

# Process data
result = operation.execute(data_source)
```

### 2. Configuration Validation in Operations

```python
class CustomCategoricalOperation(CategoricalGeneralizationOperation):
    def __init__(self, **kwargs):
        # Create and validate config
        self.config = CategoricalGeneralizationConfig(**kwargs)
        
        # Additional validation
        if self.config.strategy == "hierarchy":
            self._validate_hierarchy_exists()
        
        # Initialize parent
        super().__init__(**self.config.to_dict())
```

### 3. Dynamic Configuration Updates

```python
# Start with base config
base_config = CategoricalGeneralizationConfig(
    field_name="category",
    strategy="merge_low_freq"
)

# Update based on data analysis
if data_cardinality > 1000:
    base_config.max_categories = 100
    base_config.min_group_size = 50

# Update based on privacy requirements
if sensitive_data:
    base_config.min_acceptable_k = 20
    base_config.privacy_check_enabled = True
```

## Error Handling

### Common Validation Errors

```python
# 1. Missing required parameters
try:
    config = CategoricalGeneralizationConfig(
        strategy="hierarchy"  # Missing field_name
    )
except ValueError as e:
    # "Configuration validation failed: 'field_name' is a required property"

# 2. Invalid enum values
try:
    config = CategoricalGeneralizationConfig(
        field_name="test",
        strategy="invalid_strategy"
    )
except ValueError as e:
    # "Configuration validation failed: 'invalid_strategy' is not one of..."

# 3. Business rule violations
try:
    config = CategoricalGeneralizationConfig(
        field_name="test",
        mode="ENRICH",
        output_field_name="test"  # Same as input
    )
except ValueError as e:
    # "output_field_name must be different from field_name in ENRICH mode"
```

### Validation Helper Functions

```python
from pamola_core.anonymization.generalization.categorical_config import (
    validate_strategy_parameters,
    validate_null_strategy
)

# Validate strategy parameters
params = {
    "external_dictionary_path": "hierarchy.json",
    "hierarchy_level": 2
}
result = validate_strategy_parameters("hierarchy", params, VALID_STRATEGIES)
if not result["is_valid"]:
    print(f"Errors: {result['errors']}")

# Validate NULL strategy
null_result = validate_null_strategy(
    null_strategy="ANONYMIZE",
    allow_unknown=True,
    unknown_value=""
)
if null_result["warnings"]:
    print(f"Warnings: {null_result['warnings']}")
```

## Performance Considerations

### 1. Validator Caching

The module uses a cached validator to improve performance:
- First validation: ~2-3ms (creates validator)
- Subsequent validations: ~0.1ms (uses cache)

### 2. Memory Efficiency

- Configuration objects are lightweight (~1KB each)
- JSON schema is shared across all instances (ClassVar)
- Validator is cached at class level

### 3. Thread Safety

- Configuration instances are immutable after creation
- Validator cache is thread-safe
- Safe for concurrent use in multi-threaded operations

## Version History

### v1.1.0 (Current)
- Fixed schema conflict with base class
- Added ENRICH mode validation
- Enhanced NULL strategy validation
- Added template validation
- Improved error messages
- Added privacy threshold validation

### v1.0.0
- Initial implementation
- Complete parameter set
- JSON schema validation
- Strategy-specific validation

## Summary

The Categorical Generalization Configuration module provides a robust, type-safe configuration system for categorical anonymization operations. Key benefits include:

- **Comprehensive Validation**: Multi-level validation ensures configuration correctness
- **Type Safety**: Full type hints and enum validation
- **Flexibility**: Supports multiple strategies and advanced features
- **Performance**: Optimized for large-scale processing with Dask support
- **Privacy-First**: Built-in privacy controls and thresholds
- **Easy Integration**: Seamless use with PAMOLA operations

By following the patterns and best practices outlined in this documentation, developers can create reliable, privacy-preserving categorical anonymization configurations that balance data utility with privacy protection.