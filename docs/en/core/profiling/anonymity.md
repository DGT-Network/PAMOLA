# K-Anonymity Profiling Module Documentation

## Overview

The K-Anonymity Profiling Module provides tools for analyzing data privacy and re-identification risks in datasets. It implements preliminary k-anonymity profiling to identify quasi-identifiers and assess their potential for re-identification attacks.

This module is part of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) project's data anonymization framework, focusing on the initial assessment phase before applying anonymization techniques.

## Features

- **KA Index Generation**: Creates human-readable identifiers for field combinations
- **Automated Field Combination Analysis**: Assesses various combinations of quasi-identifiers
- **K-Anonymity Metrics**: Calculates minimum, maximum, mean, and median k values
- **Shannon Entropy Calculation**: Evaluates information content and distribution
- **Vulnerability Detection**: Identifies records at risk of re-identification
- **Comprehensive Visualizations**: Generates graphs for k-distributions, thresholds, and field comparisons
- **Spider/Radar Charts**: Presents multi-metric comparisons across different field combinations

## Pamola Core Components

The module consists of two main components:

1. **`anonymity_utils.py`**: Utility functions for k-anonymity calculations and metrics
2. **`anonymity.py`**: Operation class implementing the k-anonymity profiling workflow

## Functions and Classes

### anonymity_utils.py

#### `generate_ka_index(fields, prefix_length=2, max_prefix_length=4, existing_indices=None)`

Generates a readable identifier for a combination of fields, handling potential collisions.

**Parameters**:
- `fields`: List of field names
- `prefix_length`: Initial number of characters from field names
- `max_prefix_length`: Maximum characters when resolving collisions
- `existing_indices`: Set of existing indices to avoid duplicates

**Returns**: String identifier in format "KA_field1_field2"

#### `get_field_combinations(fields, min_size=2, max_size=4, excluded_combinations=None)`

Generates all possible combinations of fields within specified size constraints.

**Parameters**:
- `fields`: List of field names to generate combinations from
- `min_size`: Minimum number of fields per combination
- `max_size`: Maximum number of fields per combination
- `excluded_combinations`: List of specific combinations to exclude

**Returns**: List of field combinations

#### `calculate_k_anonymity(df, fields, progress_tracker=None)`

Calculates k-anonymity metrics for a specific combination of fields.

**Parameters**:
- `df`: DataFrame containing the data
- `fields`: List of field names (quasi-identifiers)
- `progress_tracker`: Optional progress tracker object

**Returns**: Dictionary of metrics including min_k, max_k, mean_k, uniqueness percentages, and distributions

#### `calculate_shannon_entropy(df, fields)`

Calculates Shannon entropy for a combination of fields.

**Parameters**:
- `df`: DataFrame containing the data
- `fields`: List of field names

**Returns**: Entropy value in bits

#### `normalize_entropy(entropy, unique_values_count)`

Normalizes entropy to a [0,1] range for easier comparison.

**Parameters**:
- `entropy`: Raw entropy value
- `unique_values_count`: Number of unique value combinations

**Returns**: Normalized entropy value

#### Other Utility Functions

- `create_ka_index_map`: Maps field combinations to their KA indices
- `find_vulnerable_records`: Identifies records vulnerable to re-identification
- `prepare_metrics_for_spider_chart`: Prepares metrics for visualization
- `save_ka_metrics`: Exports metrics to CSV format
- `save_vulnerable_records`: Exports vulnerable records information to JSON

### anonymity.py

#### `KAnonymityProfilerOperation`

Main operation class that implements the k-anonymity profiling workflow.

**Key Methods**:
- `execute`: Performs the full profiling operation
- `_create_visualizations`: Generates visualizations of the profiling results
- `_prepare_directories`: Sets up directory structure for artifacts

**Parameters**:
- `min_combination_size`: Minimum fields per combination
- `max_combination_size`: Maximum fields per combination
- `treshold_k`: Threshold for considering records vulnerable

## Usage Examples

### Basic Usage

```python
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import create_operation_instance
from pathlib import Path

# Create data source
data_source = DataSource.from_file_path("resume_data.csv", load=True)

# Create operation
operation = create_operation_instance(
    "KAnonymityProfilerOperation",
    min_combination_size=2,
    max_combination_size=3,
    treshold_k=5
)

# Define output directory
task_dir = Path("output/anonymity_analysis")
task_dir.mkdir(parents=True, exist_ok=True)

# Execute operation
reporter = {}  # Replace with actual reporter implementation
result = operation.run(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    track_progress=True,
    id_fields=["resume_id", "ID"]
)

# Access results
print(f"Operation status: {result.status}")
print(f"Artifacts generated: {len(result.artifacts)}")
```

### Specifying Field Combinations

```python
# Define specific field combinations to analyze
quasi_identifiers = [
    ["first_name", "last_name", "birth_day"],
    ["gender", "birth_day", "area_name"],
    ["education_level", "salary_range", "area_name"]
]

# Execute with specific combinations
result = operation.run(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    track_progress=True,
    fields_combinations=quasi_identifiers,
    id_fields=["resume_id"]
)
```

### Using Utility Functions Directly

```python
import pandas as pd
from pamola_core.profiling.commons.anonymity_utils import (
    calculate_k_anonymity,
    calculate_shannon_entropy,
    find_vulnerable_records
)

# Load data
df = pd.read_csv("resume_data.csv")

# Analyze specific combination
fields = ["gender", "birth_day", "area_name"]
metrics = calculate_k_anonymity(df, fields)

# Calculate entropy
entropy = calculate_shannon_entropy(df, fields)
print(f"Shannon entropy: {entropy:.4f} bits")

# Find vulnerable records
vulnerable = find_vulnerable_records(df, fields, k_threshold=5, id_field="resume_id")
print(f"Vulnerable records: {vulnerable['vulnerable_count']} ({vulnerable['vulnerable_percentage']:.2f}%)")
```

## Generated Artifacts

The module produces several artifacts:

1. **Metrics Files**:
   - `ka_metrics.csv`: Comprehensive metrics for each KA combination
   - `ka_vulnerable_records.json`: Details on vulnerable records
   - `field_uniqueness.json`: Uniqueness statistics for individual fields

2. **Visualizations**:
   - `ka_k_distribution.png`: Distribution of records across k-value ranges
   - `ka_vulnerable_curve.png`: Percentage of records meeting k-anonymity thresholds
   - `ka_comparison_spider.png`: Spider chart comparing metrics across combinations
   - `ka_field_uniqueness.png`: Uniqueness analysis of individual fields

3. **Dictionary Files**:
   - `ka_index_map.csv`: Mapping between KA indices and field combinations

## Integration

This module integrates with the broader PAMOLA CORE framework through:

- The `DataSource` abstraction for data access
- `OperationResult` for standardized result reporting
- `ProgressTracker` for operation progress monitoring
- `TaskReporter` for logging operations and artifacts
- Visualization utility functions for creating standardized charts

## Performance Considerations

- For large datasets (>100,000 records), consider using chunking or Dask
- Analysis of many field combinations can be computationally intensive
- Memory usage increases with the number of unique value combinations
- Consider limiting the number of field combinations for initial analysis

## Best Practices

1. Start with a small set of potential quasi-identifiers
2. Focus on fields with high uniqueness percentages
3. Pay special attention to combinations with low minimum k values
4. Use the spider chart to identify the most problematic combinations
5. Export vulnerable record IDs for further investigation
6. Consider field combinations with high entropy as candidates for anonymization

The K-Anonymity Profiling Module provides a foundation for understanding re-identification risks in your dataset before applying more complex anonymization techniques like generalization, suppression, or noise addition.