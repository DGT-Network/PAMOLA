# Group Analysis Module Documentation

## Overview

The `group.py` module is a pamola core component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) profiling system, designed for analyzing groups of records with the same identifier. It calculates variability within groups, identifies patterns across different groups, and enables record collapsing for anonymization algorithms like k-anonymity. This module is essential for understanding data structure, identifying redundancies, and preparing data for secure anonymization.

## Architecture

The module follows a clean separation of concerns with analytical logic isolated in `group_utils.py` and operational components in `group.py`:

1. **GroupAnalyzer**: Contains analysis methods for evaluating groups, calculating variations, and generating visualizations.
2. **GroupOperation**: Orchestrates the group analysis workflow, including data acquisition, artifact generation, and result reporting.

This architecture adheres to the operation framework of the PAMOLA.CORE system:

```
┌─────────────────┐      ┌───────────────────┐      ┌────────────────────┐
│   DataSource    │──────▶  GroupOperation   │─────▶   OperationResult   │
└─────────────────┘      └────────┬──────────┘      └────────────────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │   GroupAnalyzer   │─────▶ Utilizes functions from
                         └───────────────────┘      group_utils.py
```

The module interacts with several supporting components:
- `group_utils.py`: Contains pure analytical functions for calculating variations and analyzing groups
- `io.py`: Handles saving analysis results and other artifacts
- `visualization.py`: Creates visualizations for group distributions and heatmaps
- `op_result.py`: Structures operation results and artifacts
- `progress.py`: Provides progress tracking for long-running operations

## Key Capabilities

The module performs the following types of analysis:

1. **Weighted Variation Analysis**: Calculates how variable fields are within groups, with customizable field weights
2. **Change Frequency Analysis**: Tracks how often fields change within groups
3. **Collapsibility Analysis**: Determines potential for collapsing similar records within groups
4. **Cross-Group Identifier Analysis**: Detects when secondary identifiers span across primary groups
5. **Multiple Field Set Analysis**: Supports analyzing multiple sets of fields with different weights
6. **Visualization Generation**: Creates various visualizations of group variations and distributions

## Generated Artifacts

The module produces the following artifacts:

1. **JSON Analysis Results**:
   - `group_variation.json` or `group_variation_set{N}.json`: Detailed group variation analysis
   - `cross_group_identifiers.json`: Analysis of identifiers spanning multiple groups

2. **CSV Dictionaries**:
   - `group_variation_details.csv` or `group_variation_details_set{N}.csv`: Detailed variation information for each group
   - `cross_group_mapping.csv`: Mapping of relationships between primary and secondary identifiers

3. **Visualizations**:
   - `group_variation_distribution.png`: Distribution of variation values across groups
   - `group_size_distribution.png`: Distribution of group sizes
   - `field_variation_heatmap.png`: Heatmap showing variation of different fields across groups

## Usage Example

Here's an example of how to use `GroupOperation` to analyze groups in a dataset:

```python
from pathlib import Path
from pamola_core.profiling.analyzers.group import GroupOperation
from pamola_core.utils.ops.op_data_source import DataSource
from your_reporting_module import Reporter  # Your report system

# Create a reporter instance
reporter = Reporter()

# Prepare data source (DataFrame with group identifier field)
data_source = DataSource.from_dataframe(df, "main")

# Create output directory
task_dir = Path("output/analysis")

# Define field weights (importance of each field)
fields_weights = {
    "post": 0.3,
    "education_level": 0.2,
    "area_name": 0.2,
    "salary": 0.3
}

# Create and run the operation
operation = GroupOperation(
    group_field="resume_id",
    fields_weights=fields_weights,
    min_group_size=2,
    set_name="main"
)

# Execute the operation
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    analyze_collapsibility=True,
    collapsibility_threshold=0.2,
    analyze_changes=True,
    generate_visualization=True
)

# Check the result
if result.status.name == "SUCCESS":
    # Access the metrics
    metrics = result.metrics
    print(f"Total groups: {metrics['total_groups']}")
    print(f"Analyzed groups: {metrics['analyzed_groups']}")
    print(f"Mean variation: {metrics['mean_variation']}")

    # List the generated artifacts
    for artifact in result.artifacts:
        print(f"Artifact: {artifact.artifact_type}, Path: {artifact.path}")
```

For convenience, the module also provides a utility function for direct analysis:

```python
from pamola_core.profiling.analyzers.group import analyze_resume_group_variation

# Simple analysis with automatic field selection
result = analyze_resume_group_variation(
    df=df,
    reporter=reporter,
    group_field="resume_id",
    min_group_size=2
)

# Analysis with specific fields and weights
result = analyze_resume_group_variation(
    df=df,
    reporter=reporter,
    group_field="resume_id",
    fields_weights=fields_weights,
    analyze_collapsibility=True
)
```

## Parameters

### GroupOperation Class

**Constructor Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `group_field` | str | Yes | - | Field to group by (e.g., 'resume_id') |
| `fields_weights` | Dict[str, float] | Yes | - | Dictionary of fields and their weights |
| `min_group_size` | int | No | 2 | Minimum size of groups to analyze |
| `set_name` | str | No | "" | Optional name for the field set |
| `description` | str | No | "" | Custom description of the operation |

**Execute Method Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data_source` | DataSource | Yes | - | Source of data for the operation |
| `task_dir` | Path | Yes | - | Directory for saving artifacts |
| `reporter` | Any | Yes | - | Reporter object for tracking artifacts |
| `progress_tracker` | ProgressTracker | No | None | Progress tracking object |
| `title_prefix` | str | No | "Group" | Prefix for plot titles |
| `generate_visualization` | bool | No | True | Whether to create visualizations |
| `save_details` | bool | No | True | Whether to save detailed group information |
| `handle_nulls` | str | No | "as_value" | How to handle nulls ('as_value', 'exclude') |
| `analyze_collapsibility` | bool | No | True | Whether to analyze group collapsibility |
| `collapsibility_threshold` | float | No | 0.2 | Threshold for collapsibility analysis |
| `analyze_changes` | bool | No | False | Whether to analyze change patterns |
| `metadata_fields` | List[str] | No | [] | List of fields to extract as group metadata |
| `analyze_cross_groups` | bool | No | False | Whether to analyze relationships between identifiers |
| `secondary_identifier_fields` | List[str] | No | [] | Fields forming secondary identifiers |
| `include_timestamp` | bool | No | True | Whether to include timestamps in filenames |

## Return Value

The `execute` method returns an `OperationResult` object with the following properties:

| Property | Type | Description |
|----------|------|-------------|
| `status` | OperationStatus | SUCCESS or ERROR indicating operation outcome |
| `artifacts` | List[OperationArtifact] | List of generated files (JSONs, CSVs, PNGs) |
| `metrics` | Dict[str, Any] | Key metrics extracted from the analysis |
| `error_message` | str | Error details if the operation failed |
| `execution_time` | float | Time taken for the operation in seconds |

## Metrics and Analysis Results

The module calculates the following metrics and results:

### Group Statistics
| Metric | Description |
|--------|-------------|
| `total_groups` | Total number of groups in the dataset |
| `analyzed_groups` | Number of groups meeting the minimum size criteria |
| `min_group_size` | Minimum size threshold for group analysis |
| `fields_analyzed` | List of fields included in the analysis |
| `fields_weights` | Dictionary of field weights used for analysis |

### Variation Analysis
| Metric | Description |
|--------|-------------|
| `mean_variation` | Average variation across all analyzed groups |
| `median_variation` | Median of variation values |
| `min_variation` | Minimum variation found in any group |
| `max_variation` | Maximum variation found in any group |
| `variation_distribution` | Distribution of variation values across bins |

### Collapsibility Analysis (when enabled)
| Metric | Description |
|--------|-------------|
| `collapsible_groups_count` | Number of groups below the collapsibility threshold |
| `collapsible_groups_percentage` | Percentage of groups eligible for collapsing |
| `collapsible_records_count` | Number of records in collapsible groups |
| `collapsible_records_percentage` | Percentage of records in collapsible groups |
| `collapsible_groups` | List of groups eligible for collapsing with their details |

### Change Pattern Analysis (when enabled)
| Metric | Description |
|--------|-------------|
| `field_statistics` | Statistics about how each field varies within groups |
| `field_correlations` | Correlations between field changes |

### Cross-Group Analysis (when enabled)
| Metric | Description |
|--------|-------------|
| `cross_group_count` | Number of secondary identifiers spanning multiple groups |
| `cross_group_percentage` | Percentage of secondary IDs spanning groups |
| `cross_group_details` | Details about each cross-group relationship |
| `valid_secondary_ids` | Number of valid secondary identifiers |

## Handling Large Datasets

For large datasets, the module:
1. Automatically detects dataset size and switches to chunk-based processing when needed
2. Processes groups in manageable chunks to optimize memory usage
3. Uses progressive memory cleanup to handle very large datasets
4. Provides detailed progress tracking for long-running operations
5. Estimates resource requirements before processing

## Visualization Features

The module creates the following visualizations:

1. **Group Variation Distribution**: Shows the distribution of variation values across all groups, helping identify how homogeneous groups are overall.

2. **Group Size Distribution**: Displays the distribution of group sizes across categories (1, 2-5, 6-10, 11-20, 21-50, 51-100, 101+), providing insights into the structure of the dataset.

3. **Field Variation Heatmap**: Creates a heatmap showing how different fields vary across groups, highlighting which fields contribute most to variation within groups.

## Integration Points

The module integrates with:

1. **Data Source System**: Gets data through the `DataSource` abstraction
2. **Operation Framework**: Follows the standard operation workflow
3. **I/O System**: Uses `io.py` for saving results and artifacts
4. **Visualization System**: Uses `visualization.py` for creating plots
5. **Reporting System**: Reports progress and results to the reporter object
6. **Progress Tracking**: Integrates with `progress.py` for tracking long operations

## Multiple Field Set Support

The module supports analyzing multiple sets of fields with different weights:

1. Each set can have its own weights, reflecting different importance for fields
2. Results are saved with set-specific naming (e.g., `group_variation_set1.json`)
3. Visualizations include set information in titles and filenames

This allows for comparing different field groupings and their impact on record variation.

## Conclusion

The `group.py` module provides comprehensive analysis of record groups with robust variation calculations, visualization capabilities, and integration with anonymization workflows. It follows a clean architecture that separates analytical logic from operational concerns, making it both powerful and maintainable.

By delivering insights about group structures, field variations, and cross-group relationships, it helps data scientists and analysts better understand data patterns and prepare optimal anonymization strategies that balance privacy protection with data utility.