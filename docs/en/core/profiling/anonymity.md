# K-Anonymity Profiling Module Documentation

## Overview

The K-Anonymity Profiling Module provides tools for analyzing data privacy and re-identification risks in datasets. The `KAnonymityProfilerOperation` performs k-anonymity analysis to identify quasi-identifiers and assess their re-identification potential.

This operation can work in three analysis modes: ANALYZE (metrics only), ENRICH (add k-value column), or BOTH.

## Features

- **Multi-mode Analysis**: ANALYZE, ENRICH, or BOTH modes
- **K-Anonymity Metrics**: Calculates k-value distributions and vulnerability thresholds
- **Automatic QI Detection**: Identifies quasi-identifier combinations automatically
- **Threshold-based Risk Scoring**: Identifies vulnerable records below k-threshold
- **Metrics Export**: Exports detailed metrics to CSV/JSON
- **Visualization Support**: Generates distribution and comparison charts
- **Chunked Processing**: Efficient memory management for large datasets
- **Caching**: Optional result caching for repeated operations

## Class Reference

### `KAnonymityProfilerOperation`

Main operation class for k-anonymity profiling and enrichment.

**Inheritance**: Extends `BaseOperation`

**Constructor Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "KAnonymityProfiler" | Operation name |
| `quasi_identifiers` | List[str] | None | List of QI fields to analyze |
| `analysis_mode` | str | "ANALYZE" | Mode: "ANALYZE", "ENRICH", or "BOTH" |
| `threshold_k` | int | 5 | K-value threshold (records with k < threshold are vulnerable) |
| `export_metrics` | bool | True | Export metrics to CSV/JSON |
| `max_combinations` | int | 50 | Maximum number of QI combinations to analyze |
| `output_field_suffix` | str | "k_anon" | Suffix for k-value column in ENRICH mode |
| `quasi_identifier_sets` | Optional[List[List[str]]] | None | Pre-defined QI sets (alternative to auto-detection) |
| `id_fields` | Optional[List[str]] | None | ID columns for record tracking |
| `**kwargs` | dict | - | Additional BaseOperation parameters (chunk_size, etc.) |

**Key Methods**:

- `execute(data_source, task_dir, reporter, progress_tracker, **kwargs) -> OperationResult`
  - Main execution method for k-anonymity profiling
  - Returns OperationResult with metrics, artifacts, and optionally enriched data

**Operation Modes**:

- **ANALYZE**: Generates k-anonymity metrics and reports without modifying data
- **ENRICH**: Adds a k-anonymity column to the dataset
- **BOTH**: Performs both analysis and enrichment

## Usage Examples

### Basic ANALYZE Mode

```python
from pamola_core.profiling.analyzers.anonymity import KAnonymityProfilerOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pathlib import Path

# Create data source
data_source = DataSource.from_file_path("resume_data.csv", load=True)

# Create operation in ANALYZE mode
operation = KAnonymityProfilerOperation(
    quasi_identifiers=["first_name", "last_name", "gender", "birth_date"],
    analysis_mode="ANALYZE",
    threshold_k=5,
    max_combinations=30,
    export_metrics=True
)

# Define output directory
task_dir = Path("output/anonymity_analysis")
task_dir.mkdir(parents=True, exist_ok=True)

# Execute operation
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=None  # Optional task reporter
)

# Access results
print(f"Operation status: {result.status}")
print(f"Metrics generated: {result.metrics}")
```

### ENRICH Mode: Add K-Value Column

```python
# Create operation in ENRICH mode
operation = KAnonymityProfilerOperation(
    quasi_identifiers=["gender", "birth_date", "zipcode"],
    analysis_mode="ENRICH",
    output_field_suffix="k_anon",
    id_fields=["person_id"]
)

# Execute - returns enriched DataFrame
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=None
)

# Access enriched data
enriched_df = result.data
print(enriched_df.columns)  # Contains new 'k_anon' column
```

### BOTH Mode: Analysis + Enrichment

```python
# Create operation in BOTH mode
operation = KAnonymityProfilerOperation(
    quasi_identifiers=["first_name", "last_name", "gender"],
    analysis_mode="BOTH",
    threshold_k=3,
    quasi_identifier_sets=[
        ["first_name", "last_name"],
        ["gender", "birth_date"],
        ["first_name", "gender", "birth_date"]
    ],
    id_fields=["id"]
)

# Execute - performs analysis AND adds k_anon column
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=None
)

# Both metrics and enriched data are available
print(f"Metrics: {result.metrics}")
print(f"Enriched data shape: {result.data.shape}")
```

## Generated Artifacts

In ANALYZE or BOTH mode, the operation produces:

1. **Metrics Files**:
   - `ka_metrics.json`: K-anonymity metrics by QI combination
   - `ka_vulnerable_records.json`: Records below threshold_k

2. **Visualizations**:
   - `ka_k_distribution.png`: K-value distribution histogram
   - `ka_threshold_compliance.png`: Threshold compliance curves
   - `ka_comparison.png`: Cross-combination comparison charts

3. **Output Data** (in ENRICH or BOTH mode):
   - DataFrame with new k-anonymity column (default suffix: `k_anon`)

## Integration with PAMOLA Pipeline

`KAnonymityProfilerOperation` integrates with:

- `DataSource` abstraction for flexible input handling
- `TaskRunner` for orchestrated workflows
- `OperationResult` for standardized output and metrics
- PAMOLA's attack and metrics suite for risk assessment

## Performance Notes

- **Memory**: Chunking enabled for large datasets; respects chunk_size parameter
- **Combinations**: max_combinations limits analysis scope (default: 50)
- **Caching**: Results cached with configurable cache parameters
- **Large Datasets**: Chunked k-value calculation available for >100K rows

## Best Practices

1. Start with 3-5 fields as candidate QIs
2. Use ANALYZE mode first to explore k-value distributions
3. Focus on combinations with low minimum k values
4. Switch to ENRICH mode once QI set is finalized
5. Use BOTH mode for comprehensive assessment + data preparation
6. Compare metrics across combinations to identify highest-risk QI sets
7. Use threshold_k based on acceptable re-identification risk level

The K-Anonymity Profiler is essential for initial privacy risk assessment and prerequisite for selecting anonymization strategies.