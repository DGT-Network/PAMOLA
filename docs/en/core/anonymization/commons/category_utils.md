# PAMOLA Category Utilities Module Documentation v2.0

## Module Information
- **Name**: `category_utils.py`
- **Package**: `pamola_core.anonymization.commons`
- **Version**: 2.0.0
- **Status**: Stable
- **License**: BSD 3-Clause

## Purpose

The Category Utilities module provides comprehensive statistical analysis, distribution metrics, and grouping strategies for categorical data within the PAMOLA privacy-preserving framework. It serves as a foundational component for categorical anonymization operations, supporting:

- **Privacy Risk Assessment**: Identifying rare categories that pose disclosure risks
- **Distribution Analysis**: Understanding data characteristics for optimal anonymization
- **Grouping Strategies**: Implementing various approaches to handle rare categories
- **Validation**: Ensuring correctness of category transformations
- **Integration**: Seamless operation with CategoryMappingEngine and HierarchyDictionary

## Key Updates in v2.0

- **Enhanced Integration**: Direct support for CategoryMappingEngine
- **Thread-Safe Operations**: All functions are thread-safe for concurrent processing
- **Performance Optimization**: Vectorized operations for large datasets
- **Extended Metrics**: New concentration and diversity metrics
- **Improved Grouping**: Multiple strategies with template support
- **Better Validation**: Comprehensive mapping validation with detailed diagnostics

## Architecture

### Module Structure

```
category_utils.py
├── Distribution Analysis
│   ├── analyze_category_distribution()
│   └── calculate_category_entropy()
├── Rare Category Detection
│   └── identify_rare_categories()
├── Grouping Strategies
│   ├── group_rare_categories()
│   └── build_category_mapping()
├── Validation
│   └── validate_category_mapping()
├── Diversity Analysis
│   └── calculate_semantic_diversity_safe()
└── Helper Functions
    ├── _empty_distribution_result()
    ├── _empty_grouping_result()
    └── _format_rare_value()
```

### Integration Points

- **CategoryMappingEngine**: Direct creation and configuration
- **HierarchyDictionary**: Coverage analysis and validation
- **Statistical Metrics**: Gini, entropy, concentration calculations
- **NLP Metrics**: Semantic diversity analysis
- **Text Processing**: Category normalization and cleaning

## API Reference

### Core Functions

#### analyze_category_distribution

```python
analyze_category_distribution(
    series: pd.Series,
    top_n: int = 20,
    min_frequency: int = 1,
    calculate_entropy: bool = True,
    calculate_gini: bool = True,
    calculate_concentration: bool = True
) -> Dict[str, Any]
```

Performs comprehensive analysis of categorical distribution for anonymization planning.

**Parameters:**
- `series`: Categorical data to analyze
- `top_n`: Number of top categories to detail (default: 20)
- `min_frequency`: Minimum frequency for detailed analysis (default: 1)
- `calculate_entropy`: Whether to calculate Shannon entropy
- `calculate_gini`: Whether to calculate Gini coefficient
- `calculate_concentration`: Whether to calculate concentration metrics

**Returns:**
Dictionary containing:
- `total_categories`: Total unique categories
- `total_records`: Total number of records
- `frequency_counts`: Category frequencies (pd.Series)
- `percentage_distribution`: Category percentages (pd.Series)
- `top_n_categories`: List of tuples (category, count, percentage)
- `rare_categories`: List of categories below min_frequency
- `null_count`: Number of null values
- `null_percentage`: Percentage of nulls
- `entropy`: Shannon entropy (if calculated)
- `normalized_entropy`: Normalized entropy [0, 1]
- `gini_coefficient`: Inequality measure (if calculated)
- `concentration_metrics`: Dict with CR-5, CR-10, HHI
- `coverage_90_percentile`: Categories needed for 90% coverage
- `distribution_type`: Classification (uniform/skewed/highly_skewed)

**Example:**
```python
dist = analyze_category_distribution(
    df['city'],
    top_n=10,
    calculate_concentration=True
)
print(f"Distribution type: {dist['distribution_type']}")
print(f"90% coverage needs {dist['coverage_90_percentile']} categories")
```

#### identify_rare_categories

```python
identify_rare_categories(
    series: pd.Series,
    count_threshold: int = 10,
    percent_threshold: float = 0.01,
    combined_criteria: bool = True
) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]
```

Identifies rare categories based on privacy risk criteria.

**Parameters:**
- `series`: Categorical data to analyze
- `count_threshold`: Minimum count threshold
- `percent_threshold`: Minimum percentage threshold (0-1)
- `combined_criteria`: If True, both criteria must be met (AND); if False, either (OR)

**Returns:**
Tuple containing:
- `rare_categories`: Set of category names identified as rare
- `detailed_info`: Dictionary with details for each rare category
  - `count`: Frequency count
  - `percentage`: Percentage of total
  - `rank`: Rank by frequency
  - `cumulative_percentage`: Cumulative percentage up to this category

**Example:**
```python
rare_cats, details = identify_rare_categories(
    df['job_title'],
    count_threshold=5,
    percent_threshold=0.001  # 0.1%
)
for cat, info in details.items():
    print(f"{cat}: {info['count']} occurrences ({info['percentage']:.2%})")
```

#### group_rare_categories

```python
group_rare_categories(
    series: pd.Series,
    grouping_strategy: str = "single_other",
    threshold: Union[int, float] = 10,
    max_groups: int = 100,
    group_prefix: str = "GROUP_",
    preserve_top_n: Optional[int] = None,
    other_label: str = "OTHER",
    rare_value_template: Optional[str] = None
) -> Tuple[pd.Series, Dict[str, Any]]
```

Groups rare categories using privacy-preserving strategies.

**Parameters:**
- `series`: Categorical data to process
- `grouping_strategy`: Strategy to use:
  - `"single_other"`: All rare → single group
  - `"numbered"`: Rare → numbered groups (GROUP_1, GROUP_2, etc.)
  - `"frequency_bands"`: Group by frequency ranges
- `threshold`: Count threshold (int) or percentage (float < 1.0)
- `max_groups`: Maximum number of groups to create
- `group_prefix`: Prefix for numbered groups
- `preserve_top_n`: Number of top categories to never group
- `other_label`: Label for single "other" group
- `rare_value_template`: Template with `{n}` placeholder (e.g., "OTHER_{n}")

**Returns:**
Tuple containing:
- `grouped_series`: Series with rare categories grouped
- `grouping_info`: Dictionary with:
  - `strategy_used`: Applied strategy
  - `groups_created`: Number of groups created
  - `categories_grouped`: Number of categories grouped
  - `group_mapping`: Dict of original → group mappings
  - `group_sizes`: Dict of group → size
  - `reduction_ratio`: Reduction in unique values
  - `preserved_categories`: List of preserved top categories

**Example:**
```python
# Strategy 1: Single other group
grouped, info = group_rare_categories(
    df['department'],
    grouping_strategy="single_other",
    threshold=10,
    preserve_top_n=5
)

# Strategy 2: Numbered groups with template
grouped, info = group_rare_categories(
    df['product_category'],
    grouping_strategy="numbered",
    threshold=0.001,  # 0.1%
    max_groups=10,
    rare_value_template="RARE_CATEGORY_{n}"
)

# Strategy 3: Frequency bands
grouped, info = group_rare_categories(
    df['location'],
    grouping_strategy="frequency_bands",
    threshold=5,
    max_groups=5
)
```

#### build_category_mapping

```python
build_category_mapping(
    series: pd.Series,
    strategy: str = "frequency",
    target_categories: Optional[int] = None,
    min_group_size: int = 5,
    preserve_top_n: Optional[int] = None,
    unknown_value: str = "OTHER",
    unknown_template: Optional[str] = None
) -> CategoryMappingEngine
```

Builds a CategoryMappingEngine with mapping rules based on the specified strategy.

**Parameters:**
- `series`: Source categorical data
- `strategy`: Mapping strategy:
  - `"frequency"`: Group by frequency
  - `"alphabetical"`: Group alphabetically
  - `"semantic"`: Group by semantic similarity (if available)
- `target_categories`: Target number of categories after mapping
- `min_group_size`: Minimum size for each group
- `preserve_top_n`: Number of top categories to preserve
- `unknown_value`: Default value for unmapped entries
- `unknown_template`: Template for numbered unknowns

**Returns:**
Configured CategoryMappingEngine instance

**Example:**
```python
# Build mapping engine
engine = build_category_mapping(
    df['city'],
    strategy="frequency",
    target_categories=50,
    preserve_top_n=20,
    unknown_template="OTHER_CITY_{n}"
)

# Apply to data
df['city_grouped'] = engine.apply_to_series(df['city'])

# Check statistics
stats = engine.get_statistics()
print(f"Mapping reduced {stats['mapping_count']} cities to {target_categories}")
```

#### calculate_category_entropy

```python
calculate_category_entropy(
    series: pd.Series,
    base: float = 2.0,
    normalize: bool = True
) -> float
```

Calculates Shannon entropy of categorical distribution.

**Parameters:**
- `series`: Categorical data
- `base`: Logarithm base (2 for bits, e for nats)
- `normalize`: Whether to normalize by maximum entropy

**Returns:**
- Entropy value (normalized to [0, 1] if normalize=True)

**Example:**
```python
entropy = calculate_category_entropy(df['category'])
print(f"Normalized entropy: {entropy:.3f}")
# 0 = all same category, 1 = uniform distribution
```

#### validate_category_mapping

```python
validate_category_mapping(
    original: pd.Series,
    mapped: pd.Series,
    mapping: Optional[Dict[str, str]] = None,
    check_consistency: bool = True,
    check_completeness: bool = True
) -> Dict[str, Any]
```

Validates category mapping for correctness and consistency.

**Parameters:**
- `original`: Original categorical data
- `mapped`: Data after category mapping
- `mapping`: Expected mapping dictionary (original → generalized)
- `check_consistency`: Whether to check mapping consistency
- `check_completeness`: Whether to check mapping completeness

**Returns:**
Dictionary containing:
- `is_valid`: Overall validation status
- `length_match`: Same number of records
- `null_preservation`: Nulls handled correctly
- `unmapped_categories`: Categories without mapping
- `inconsistent_mappings`: Mapping inconsistencies found
- `coverage`: Percentage of values successfully mapped
- `reduction_ratio`: Reduction in unique values
- `original_categories`: Number of original categories
- `mapped_categories`: Number of categories after mapping
- `errors`: List of validation errors
- `warnings`: List of validation warnings

**Example:**
```python
# Validate transformation
validation = validate_category_mapping(
    original=df['original_category'],
    mapped=df['anonymized_category'],
    mapping=engine.get_mapping_dict()
)

if not validation['is_valid']:
    print(f"Validation failed: {validation['errors']}")
else:
    print(f"Mapping coverage: {validation['coverage']:.1%}")
    print(f"Reduction ratio: {validation['reduction_ratio']:.1%}")
```

#### calculate_semantic_diversity_safe

```python
calculate_semantic_diversity_safe(
    categories: List[str],
    method: str = "character",
    sample_size: Optional[int] = 1000
) -> float
```

Safely calculates semantic diversity of categories.

**Parameters:**
- `categories`: List of category names
- `method`: Diversity calculation method
- `sample_size`: Maximum categories to analyze (for performance)

**Returns:**
- Diversity score [0, 1]

### Helper Functions

#### _format_rare_value

```python
_format_rare_value(
    template: str,
    counter: int,
    default: str = "OTHER"
) -> str
```

Formats a rare value using template with counter.

**Parameters:**
- `template`: Template string containing `{n}`
- `counter`: Counter value to insert
- `default`: Default if template is invalid

**Returns:**
- Formatted string

## Usage Patterns

### Pattern 1: Privacy-Aware Analysis Pipeline

```python
def analyze_for_k_anonymity(
    series: pd.Series,
    k_threshold: int = 5,
    strategy: str = "auto"
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Complete analysis pipeline for k-anonymity."""
    
    # Step 1: Analyze distribution
    dist = analyze_category_distribution(series)
    logger.info(
        f"Distribution: {dist['total_categories']} categories, "
        f"entropy={dist['normalized_entropy']:.2f}, "
        f"type={dist['distribution_type']}"
    )
    
    # Step 2: Identify privacy risks
    rare_cats, rare_details = identify_rare_categories(
        series,
        count_threshold=k_threshold
    )
    
    if not rare_cats:
        logger.info("No privacy risks detected")
        return series, {'status': 'no_action_needed'}
    
    logger.warning(
        f"Found {len(rare_cats)} categories with k < {k_threshold}"
    )
    
    # Step 3: Choose strategy
    if strategy == "auto":
        if dist['distribution_type'] == 'highly_skewed':
            strategy = "frequency_bands"
        elif len(rare_cats) > 50:
            strategy = "numbered"
        else:
            strategy = "single_other"
    
    # Step 4: Apply grouping
    grouped, info = group_rare_categories(
        series,
        grouping_strategy=strategy,
        threshold=k_threshold,
        preserve_top_n=10,
        rare_value_template=f"ANON_{strategy.upper()}_{{n}}"
    )
    
    # Step 5: Validate
    validation = validate_category_mapping(series, grouped)
    info['validation'] = validation
    
    logger.info(
        f"Anonymization complete: {info['reduction_ratio']:.1%} reduction, "
        f"coverage={validation['coverage']:.1%}"
    )
    
    return grouped, info
```

### Pattern 2: Integration with CategoryMappingEngine

```python
def create_hierarchical_mapping(
    series: pd.Series,
    hierarchy_dict: Dict[str, str],
    min_group_size: int = 5
) -> CategoryMappingEngine:
    """Create mapping engine with hierarchy and rare handling."""
    
    # Analyze distribution
    dist = analyze_category_distribution(series)
    
    # Build base mapping from hierarchy
    engine = CategoryMappingEngine(
        unknown_template="UNKNOWN_CATEGORY_{n}"
    )
    
    # Add hierarchy mappings
    for value, category in hierarchy_dict.items():
        engine.add_mapping(value, category)
    
    # Handle rare categories not in hierarchy
    coverage = engine.get_coverage(series)
    if coverage['unmapped']:
        # Group unmapped rare categories
        unmapped_series = series[series.isin(coverage['unmapped_values'])]
        grouped, info = group_rare_categories(
            unmapped_series,
            threshold=min_group_size,
            grouping_strategy="numbered",
            max_groups=10
        )
        
        # Add rare groupings to engine
        for orig, group in info['group_mapping'].items():
            engine.add_mapping(orig, group)
    
    return engine
```

### Pattern 3: Multi-Level Generalization

```python
def create_multi_level_generalization(
    series: pd.Series,
    levels: List[int] = [100, 50, 20, 10, 5]
) -> Dict[int, pd.Series]:
    """Create multiple generalization levels."""
    
    results = {}
    
    for level in levels:
        # Group at this level
        grouped, info = group_rare_categories(
            series,
            threshold=level,
            grouping_strategy="frequency_bands",
            max_groups=20,
            preserve_top_n=10,
            rare_value_template=f"LEVEL_{level}_GROUP_{{n}}"
        )
        
        # Store result and metrics
        results[level] = {
            'data': grouped,
            'info': info,
            'entropy': calculate_category_entropy(grouped),
            'categories': grouped.nunique()
        }
        
        # Log level statistics
        logger.info(
            f"Level {level}: {results[level]['categories']} categories, "
            f"entropy={results[level]['entropy']:.3f}"
        )
    
    return results
```

### Pattern 4: Semantic-Aware Grouping

```python
def group_with_semantic_similarity(
    series: pd.Series,
    min_group_size: int = 5,
    similarity_threshold: float = 0.7
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Group categories considering semantic similarity."""
    
    # Initial grouping by frequency
    grouped, info = group_rare_categories(
        series,
        threshold=min_group_size,
        grouping_strategy="numbered"
    )
    
    # Calculate semantic diversity before and after
    original_diversity = calculate_semantic_diversity_safe(
        list(series.dropna().unique())
    )
    grouped_diversity = calculate_semantic_diversity_safe(
        list(grouped.dropna().unique())
    )
    
    info['semantic_metrics'] = {
        'original_diversity': original_diversity,
        'grouped_diversity': grouped_diversity,
        'diversity_loss': original_diversity - grouped_diversity
    }
    
    # If diversity loss is too high, try different strategy
    if info['semantic_metrics']['diversity_loss'] > 0.3:
        logger.warning(
            "High semantic diversity loss, trying frequency bands"
        )
        grouped, info = group_rare_categories(
            series,
            threshold=min_group_size,
            grouping_strategy="frequency_bands",
            max_groups=30
        )
    
    return grouped, info
```

## Best Practices

### 1. **Threshold Selection Guidelines**

```python
# Healthcare/Financial data (strict privacy)
STRICT_PRIVACY = {
    'count_threshold': 20,
    'percent_threshold': 0.01,  # 1%
    'combined_criteria': True,   # Both conditions
    'preserve_top_n': 5
}

# General business data (balanced)
BALANCED_PRIVACY = {
    'count_threshold': 10,
    'percent_threshold': 0.005,  # 0.5%
    'combined_criteria': False,  # Either condition
    'preserve_top_n': 10
}

# Research/Analytics (utility-focused)
UTILITY_FOCUSED = {
    'count_threshold': 5,
    'percent_threshold': 0.001,  # 0.1%
    'combined_criteria': False,
    'preserve_top_n': 20
}
```

### 2. **Strategy Selection Matrix**

| Data Characteristic | Recommended Strategy | Parameters |
|-------------------|---------------------|------------|
| Few categories (<20) | `single_other` | preserve_top_n=10 |
| Many rare categories | `numbered` | max_groups=20 |
| Highly skewed | `frequency_bands` | max_groups=10 |
| Hierarchical data | Use with hierarchy | fuzzy_matching=True |
| High cardinality | `frequency_bands` | aggressive threshold |

### 3. **Performance Optimization**

```python
def optimize_for_large_datasets(series: pd.Series) -> Dict[str, Any]:
    """Optimize analysis for large categorical datasets."""
    
    # Sample for initial analysis if very large
    if len(series) > 1_000_000:
        sample = series.sample(
            n=100_000,
            random_state=42
        )
        dist = analyze_category_distribution(
            sample,
            calculate_concentration=False  # Skip expensive metrics
        )
    else:
        dist = analyze_category_distribution(series)
    
    # Use results to guide full processing
    if dist['total_categories'] > 10_000:
        # High cardinality - aggressive grouping
        return {
            'strategy': 'frequency_bands',
            'threshold': 0.001,  # 0.1%
            'max_groups': 100
        }
    else:
        # Normal processing
        return {
            'strategy': 'auto',
            'threshold': 10,
            'preserve_top_n': 20
        }
```

### 4. **Error Handling**

```python
def safe_category_analysis(
    series: pd.Series,
    operation: str = "analyze"
) -> Dict[str, Any]:
    """Safely perform category analysis with error handling."""
    
    try:
        # Check input
        if series is None or len(series) == 0:
            logger.warning("Empty series provided")
            return _empty_distribution_result()
        
        # Check data type
        if not pd.api.types.is_categorical_dtype(series) and \
           not pd.api.types.is_object_dtype(series):
            logger.warning(f"Non-categorical dtype: {series.dtype}")
            series = series.astype(str)
        
        # Perform operation
        if operation == "analyze":
            return analyze_category_distribution(series)
        elif operation == "identify_rare":
            return identify_rare_categories(series)
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    except Exception as e:
        logger.error(f"Error in category analysis: {e}")
        return {
            'error': str(e),
            'status': 'failed',
            'series_info': {
                'length': len(series) if series is not None else 0,
                'dtype': str(series.dtype) if series is not None else 'unknown'
            }
        }
```

## Integration Guidelines

### With Categorical Generalization

```python
from pamola_core.anonymization.generalization.categorical_op import (
    CategoricalGeneralizationOperation
)
from pamola_core.anonymization.commons.category_utils import (
    analyze_category_distribution,
    group_rare_categories,
    build_category_mapping
)

class EnhancedCategoricalOperation(CategoricalGeneralizationOperation):
    def prepare_data(self, series: pd.Series) -> pd.Series:
        """Prepare data using category utilities."""
        
        # Analyze distribution
        dist = analyze_category_distribution(series)
        self.logger.info(
            f"Processing {dist['total_categories']} categories, "
            f"distribution: {dist['distribution_type']}"
        )
        
        # Build appropriate mapping
        if self.strategy == "merge_low_freq":
            # Use category utilities for grouping
            grouped, info = group_rare_categories(
                series,
                threshold=self.min_group_size,
                grouping_strategy="single_other",
                preserve_top_n=self.preserve_top_n,
                rare_value_template=self.rare_value_template
            )
            
            # Store mapping info
            self.category_mapping = info['group_mapping']
            self.grouping_info = info
            
            return grouped
        
        return series
```

### With Privacy Metrics

```python
def calculate_category_privacy_metrics(
    original: pd.Series,
    anonymized: pd.Series,
    k_threshold: int = 5
) -> Dict[str, Any]:
    """Calculate privacy metrics for categorical data."""
    
    # Basic distribution comparison
    orig_dist = analyze_category_distribution(original)
    anon_dist = analyze_category_distribution(anonymized)
    
    # Privacy improvement
    orig_rare, _ = identify_rare_categories(
        original,
        count_threshold=k_threshold
    )
    anon_rare, _ = identify_rare_categories(
        anonymized,
        count_threshold=k_threshold
    )
    
    # Information loss metrics
    metrics = {
        'category_reduction': 1 - (anon_dist['total_categories'] / 
                                  orig_dist['total_categories']),
        'entropy_loss': orig_dist['entropy'] - anon_dist['entropy'],
        'rare_eliminated': len(orig_rare) - len(anon_rare),
        'k_anonymity_achieved': len(anon_rare) == 0,
        'distribution_shift': abs(orig_dist['gini_coefficient'] - 
                                anon_dist['gini_coefficient'])
    }
    
    # Semantic diversity if applicable
    if orig_dist['total_categories'] <= 1000:
        metrics['semantic_diversity_loss'] = (
            calculate_semantic_diversity_safe(
                list(original.dropna().unique())
            ) -
            calculate_semantic_diversity_safe(
                list(anonymized.dropna().unique())
            )
        )
    
    return metrics
```

## Performance Considerations

### Memory Optimization

- **Large Series**: Functions handle large datasets efficiently using vectorized pandas operations
- **Sampling**: For very large datasets (>1M records), consider sampling for initial analysis
- **Chunking**: Process in chunks if memory is constrained
- **Category Limits**: Functions have built-in limits for extremely high cardinality data

### Speed Optimization

- **Vectorization**: All operations use pandas vectorized methods where possible
- **Early Termination**: Functions return early for edge cases (empty data, single category)
- **Selective Computation**: Expensive metrics (Gini, concentration) are optional
- **Caching**: Results can be cached when used with CategoryMappingEngine

### Thread Safety

All functions in this module are thread-safe and can be used in parallel processing:
- No global state modification
- Input data is not modified in-place
- All returned objects are new instances

## Error Handling

Functions handle various error conditions gracefully:
- Empty or null series return valid empty results
- Invalid parameters fall back to defaults with warnings
- Type mismatches are coerced when possible
- All exceptions are caught and logged

## Version History

### v2.0.0 (Current)
- Added integration with CategoryMappingEngine
- Enhanced grouping strategies with templates
- Improved validation with detailed diagnostics
- Added concentration and diversity metrics
- Performance optimizations for large datasets
- Thread-safe implementation

### v1.0.0
- Initial implementation
- Basic distribution analysis
- Simple rare category identification
- Single grouping strategy