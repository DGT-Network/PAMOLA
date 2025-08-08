# PAMOLA Category Mapping Engine Documentation

## Module Information
- **Name**: `category_mapping.py`
- **Package**: `pamola_core.anonymization.commons`
- **Version**: 2.0.0
- **Status**: Stable
- **License**: BSD 3-Clause

## Purpose

The Category Mapping Engine is a universal, thread-safe component designed for managing and applying value transformations across various privacy-preserving operations in PAMOLA. While primarily used for categorical anonymization (generalization), it also serves critical roles in:

- **Data Masking**: Replacing sensitive values with safe alternatives
- **Tokenization**: Mapping original values to tokens
- **Pseudonymization**: Consistent value replacement
- **Generalization**: Hierarchical category mapping
- **Suppression**: Mapping values to suppression tokens

## Description

This module provides a high-performance, thread-safe engine for managing complex mapping rules with conditional logic, caching, and batch processing capabilities. It centralizes all value transformation logic to ensure consistency across different privacy operations while maintaining optimal performance through LRU caching and vectorized operations.

## Key Features

- **Thread-Safety**: All operations protected with RLock for concurrent access
- **High Performance**: O(1) lookups with thread-safe LRU caching
- **Conditional Mappings**: Support for context-aware transformations
- **Batch Processing**: Optimized pandas Series operations with vectorization
- **Unknown Handling**: Configurable templates for unmapped values
- **Memory Efficient**: Configurable cache size with automatic eviction
- **Statistics Tracking**: Comprehensive performance metrics
- **Import/Export**: Full serialization support for persistence
- **Vectorized Operations**: Automatic optimization for large datasets

## Architecture

### Core Components

1. **CategoryMappingEngine**: Main engine class
2. **ConditionalMapping**: Rule representation for conditional transformations
3. **Thread-Safe Cache**: LRU cache with concurrent access protection
4. **Statistics Tracker**: Performance monitoring system

### Storage Structure

```
CategoryMappingEngine
├── _mappings: Dict[str, str]          # Simple value mappings
├── _conditional_mappings: List[ConditionalMapping]  # Conditional rules
├── _cache: LRUCache                   # Thread-safe lookup cache
├── _lock: threading.RLock             # Concurrency control
├── _stats: Dict[str, int]             # Performance metrics
└── _unknown_mapping_cache: Dict[str, str]  # Generated unknowns
```

## API Reference

### CategoryMappingEngine

#### Constructor

```python
CategoryMappingEngine(
    unknown_value: str = "OTHER",
    unknown_template: Optional[str] = "OTHER_{n}",
    cache_size: int = 10000
)
```

**Parameters:**
- `unknown_value`: Default value for unmapped entries
- `unknown_template`: Template for numbered unknown values (must contain `{n}`)
- `cache_size`: Maximum size of the LRU cache

#### Core Methods

##### add_mapping
```python
add_mapping(original: str, replacement: str) -> None
```
Add a simple direct mapping rule.

**Parameters:**
- `original`: Original value to map
- `replacement`: Target replacement value

**Example:**
```python
engine.add_mapping("New York", "NY")
engine.add_mapping("Los Angeles", "CA")
```

##### add_conditional_mapping
```python
add_conditional_mapping(
    original: str,
    replacement: str,
    condition: Dict[str, Any],
    priority: int = 0
) -> None
```
Add a mapping rule with conditions.

**Parameters:**
- `original`: Original value to map
- `replacement`: Target replacement value
- `condition`: Dictionary of conditions to match
- `priority`: Rule priority (higher = higher precedence)

**Example:**
```python
engine.add_conditional_mapping(
    original="Manager",
    replacement="Executive",
    condition={"department": "Sales", "experience": {"op": "gt", "value": 5}},
    priority=10
)
```

##### apply_to_series
```python
apply_to_series(
    series: pd.Series,
    context_df: Optional[pd.DataFrame] = None
) -> pd.Series
```
Apply mappings to a pandas Series with optional context.

**Parameters:**
- `series`: Input data series
- `context_df`: Context DataFrame for conditional rules

**Returns:**
- Transformed pandas Series

**Example:**
```python
# Simple application
result = engine.apply_to_series(df['city'])

# With context for conditional rules
result = engine.apply_to_series(df['position'], context_df=df[['department', 'experience']])
```

##### apply_to_value
```python
apply_to_value(
    value: str,
    context: Optional[Dict[str, Any]] = None
) -> str
```
Apply mapping to a single value with caching.

**Parameters:**
- `value`: Value to transform
- `context`: Optional context dictionary

**Returns:**
- Mapped value or unknown_value

##### get_statistics
```python
get_statistics() -> Dict[str, Any]
```
Retrieve performance statistics.

**Returns:**
Dictionary containing:
- `total_lookups`: Total transformation requests
- `cache_hits`: Number of cache hits
- `cache_hit_rate`: Cache efficiency percentage
- `unknown_count`: Number of unmapped values
- `conditional_matches`: Conditional rule matches
- `vectorized_ops`: Vectorized operation count
- `mapping_count`: Total simple mappings
- `cache_size`: Current cache size

##### get_coverage
```python
get_coverage(values: pd.Series) -> Dict[str, Any]
```
Analyze mapping coverage for a dataset.

**Parameters:**
- `values`: Series to analyze

**Returns:**
Dictionary containing:
- `total_unique`: Unique value count
- `mapped`: Successfully mapped count
- `unmapped`: Unmapped value count
- `coverage_percent`: Coverage percentage
- `unmapped_values`: List of unmapped values (max 100)

##### import_mappings
```python
import_mappings(
    mappings: Dict[str, str],
    check_duplicates: bool = True
) -> None
```
Bulk import simple mappings.

**Parameters:**
- `mappings`: Dictionary of mappings
- `check_duplicates`: Whether to check for overwrites

##### export_to_dict / import_from_dict
```python
export_to_dict() -> Dict[str, Any]
import_from_dict(data: Dict[str, Any]) -> None
```
Serialize/deserialize complete engine state.

### Utility Functions

##### create_mapping_from_hierarchy
```python
create_mapping_from_hierarchy(
    hierarchy_dict: Dict[str, str],
    level: int = 1,
    unknown_template: Optional[str] = None,
    hierarchy_version: Optional[str] = None
) -> CategoryMappingEngine
```
Create engine from hierarchy dictionary.

##### merge_mapping_engines
```python
merge_mapping_engines(
    engines: List[CategoryMappingEngine],
    unknown_value: str = "OTHER",
    unknown_template: Optional[str] = None
) -> CategoryMappingEngine
```
Merge multiple engines into one.

## Usage Examples

### Basic Anonymization
```python
# Create engine for location anonymization
engine = CategoryMappingEngine(unknown_value="UNKNOWN_LOCATION")

# Add city to state mappings
engine.add_mapping("New York", "NY")
engine.add_mapping("Los Angeles", "CA")
engine.add_mapping("Chicago", "IL")

# Apply to data
df['state'] = engine.apply_to_series(df['city'])
```

### Data Masking with Templates
```python
# Create engine for ID masking
engine = CategoryMappingEngine(
    unknown_value="MASKED",
    unknown_template="USER_{n}"
)

# Mask sensitive IDs
df['masked_id'] = engine.apply_to_series(df['user_id'])
```

### Conditional Anonymization
```python
# Create engine with conditional rules
engine = CategoryMappingEngine()

# Add conditional mapping for salary ranges
engine.add_conditional_mapping(
    original="High",
    replacement="Above Average",
    condition={"department": "Engineering"},
    priority=10
)

# Apply with context
df['salary_category'] = engine.apply_to_series(
    df['salary_range'],
    context_df=df[['department']]
)
```

### Performance Monitoring
```python
# Process large dataset
result = engine.apply_to_series(large_series)

# Check performance
stats = engine.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Vectorized operations: {stats['vectorized_ops']}")

# Analyze coverage
coverage = engine.get_coverage(large_series)
print(f"Mapping coverage: {coverage['coverage_percent']:.2%}")
```

### Persistence
```python
# Export configuration
config = engine.export_to_dict()
with open('mappings.json', 'w') as f:
    json.dump(config, f)

# Import configuration
new_engine = CategoryMappingEngine()
with open('mappings.json', 'r') as f:
    config = json.load(f)
new_engine.import_from_dict(config)
```

## Performance Considerations

1. **Cache Size**: Adjust based on unique value count
2. **Vectorization**: Automatically enabled for Series > 1000 rows
3. **Conditional Rules**: Order by priority for optimal performance
4. **Thread Safety**: No performance penalty for single-threaded use

## Integration with PAMOLA Operations

The Category Mapping Engine integrates seamlessly with:
- **K-Anonymity**: Generalization hierarchies
- **L-Diversity**: Category grouping
- **Differential Privacy**: Value perturbation
- **Data Masking**: Consistent pseudonymization