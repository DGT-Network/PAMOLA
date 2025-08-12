# Date Analysis Module Documentation

## Overview

The `date.py` module is a critical component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) profiling system, designed for comprehensive analysis of date fields within datasets. It provides validation, distribution analysis, anomaly detection, and visualization capabilities for date data, making it essential for profiling tasks that involve temporal information such as birth dates, creation dates, and other time-related fields.

## Architecture

The module follows a clean separation of concerns with two primary classes:

1. **DateAnalyzer**: Contains pure analytical logic for analyzing dates and detecting anomalies.
2. **DateOperation**: Orchestrates the profiling workflow, including data acquisition, analysis execution, artifact generation, and result reporting.

This architecture adheres to the operation framework of the PAMOLA.CORE system:

```
┌─────────────────┐      ┌───────────────────┐      ┌────────────────────┐
│   DataSource    │──────▶   DateOperation   │─────▶   OperationResult   │
└─────────────────┘      └────────┬──────────┘      └────────────────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │    DateAnalyzer   │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │    date_utils.py  │
                         └───────────────────┘
```

The module also interacts with several supporting components:
- `date_utils.py`: Contains pamola core date analysis functions
- `io.py`: Handles saving analysis results and other artifacts
- `visualization.py`: Creates date distribution visualizations
- `op_result.py`: Structures operation results and artifacts

## Key Capabilities

The module performs the following types of analysis:

1. **Date Validation**: Verifies date formats and identifies invalid dates
2. **Date Range Analysis**: Determines minimum and maximum dates in the dataset
3. **Distribution Analysis**: Generates distributions by year, decade, month, and day of week
4. **Anomaly Detection**: Identifies dates outside valid ranges or with invalid formats
5. **Group Analysis**: Detects date inconsistencies within groups (e.g., same resume ID)
6. **UID Analysis**: Identifies date inconsistencies across records with the same UID

## Generated Artifacts

The module produces the following artifacts:

1. **JSON Statistics Report** (`{field_name}_stats.json`):
   - Basic statistics (valid count, invalid count, min/max dates)
   - Null/missing value analysis
   - Distribution metrics by year, decade, month, and day of week
   - Anomaly information
   - Group and UID inconsistency reports

2. **Year Distribution Visualization** (`{field_name}_year_distribution.png`):
   - Distribution of dates by year
   - Shows the overall temporal pattern of the data

3. **Month Distribution Visualization** (`{field_name}_month_distribution.png`):
   - Distribution of dates by month
   - Reveals seasonal patterns in the data

4. **Day of Week Distribution** (`{field_name}_dow_distribution.png`):
   - Distribution of dates by day of week
   - Reveals weekly patterns in the data

5. **Anomalies CSV** (`{field_name}_anomalies.csv`):
   - List of detected date anomalies
   - Includes index, value, anomaly type, and year (when applicable)

## Usage Example

Here's an example of how to use `DateOperation` to analyze a date field:

```python
from pathlib import Path
from pamola_core.profiling.analyzers.date import DateOperation
from pamola_core.utils.ops.op_data_source import DataSource
from your_reporting_module import Reporter  # Your report system

# Create a reporter instance
reporter = Reporter()

# Prepare data source (DataFrame with date field)
data_source = DataSource.from_dataframe(df, "main")

# Create output directory
task_dir = Path("output/analysis")

# Create and run the operation
operation = DateOperation(
    field_name="birth_date",
    min_year=1940,
    max_year=2005,
    id_column="resume_id",
    uid_column="UID"
)

# Execute the operation
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    generate_visualization=True
)

# Check the result
if result.status.name == "SUCCESS":
    # Access the statistics
    stats = result.metrics
    print(f"Valid dates: {stats['valid_count']}")
    print(f"Date range: {stats['min_date']} to {stats['max_date']}")

    # List the generated artifacts
    for artifact in result.artifacts:
        print(f"Artifact: {artifact.artifact_type}, Path: {artifact.path}")
```

## Parameters

### DateOperation Class

**Constructor Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `field_name` | str | Yes | - | Name of the date field to analyze |
| `min_year` | int | No | 1940 | Minimum year considered valid (earlier dates flagged as anomalies) |
| `max_year` | int | No | 2005 | Maximum year considered valid (later dates flagged as anomalies) |
| `id_column` | str | No | None | Column to use for group analysis (e.g., resume_id) |
| `uid_column` | str | No | None | Column to use for UID analysis (person identifier) |
| `description` | str | No | "" | Custom description of the operation |

**Execute Method Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data_source` | DataSource | Yes | - | Source of data for the operation |
| `task_dir` | Path | Yes | - | Directory for saving artifacts |
| `reporter` | Any | Yes | - | Reporter object for tracking artifacts |
| `progress_tracker` | ProgressTracker | No | None | Progress tracking object |
| `generate_visualization` | bool | No | True | Whether to create visualizations |
| `include_timestamp` | bool | No | True | Whether to include timestamps in filenames |
| `profile_type` | str | No | "date" | Type of profiling for organizing artifacts |

## Return Value

The `execute` method returns an `OperationResult` object with the following properties:

| Property | Type | Description |
|----------|------|-------------|
| `status` | OperationStatus | SUCCESS or ERROR indicating operation outcome |
| `artifacts` | List[OperationArtifact] | List of generated files (JSONs, PNGs, CSVs) |
| `metrics` | Dict[str, Any] | Key metrics extracted from the analysis |
| `error_message` | str | Error details if the operation failed |
| `execution_time` | float | Time taken for the operation in seconds |

## Metrics and Statistical Calculations

The module calculates the following metrics for date fields:

### Basic Statistics
| Metric | Description |
|--------|-------------|
| `min_date` | Earliest date in the dataset (after validation) |
| `max_date` | Latest date in the dataset (after validation) |
| `year_distribution` | Distribution of dates by year |
| `decade_distribution` | Distribution of dates by decade |
| `month_distribution` | Distribution of dates by month |
| `day_of_week_distribution` | Distribution of dates by day of week |

### Data Quality Metrics
| Metric | Description |
|--------|-------------|
| `total_records` | Total number of records in the dataset |
| `null_count` | Number of missing (NULL) date values |
| `non_null_count` | Number of non-missing date values |
| `valid_count` | Number of valid dates after parsing |
| `invalid_count` | Number of non-null values that couldn't be parsed as dates |
| `fill_rate` | Percentage of non-null values (completeness) |
| `valid_rate` | Percentage of valid dates among non-null values |

### Anomaly Detection
| Metric | Description |
|--------|-------------|
| `anomalies` | Counts of different anomaly types |
| `invalid_format` | Examples of dates with invalid formats |
| `too_old` | Examples of dates before the minimum year |
| `future_dates` | Examples of dates in the future |
| `too_young` | Examples of dates after the maximum year but not in the future |
| `negative_years` | Examples of dates with negative years |

### Group and UID Analysis
| Metric | Description |
|--------|-------------|
| `date_changes_within_group` | Statistics on inconsistent dates within the same group |
| `groups_with_changes` | Count of groups with varying date values |
| `date_inconsistencies_by_uid` | Statistics on inconsistent dates for the same UID |
| `uids_with_inconsistencies` | Count of UIDs with varying date values |

## Handling of Special Cases

The module effectively handles a variety of special cases:

1. **Missing Values**: Properly counts null/NaN values and calculates fill rates
2. **Invalid Date Formats**: Identifies values that can't be parsed as valid dates
3. **Dates outside Valid Range**: Flags dates that are too old, in the future, or otherwise suspicious
4. **Negative Years**: Detects and reports negative years as a distinct anomaly type
5. **String Date Formats**: Accepts dates in various string formats, converting them using pandas' to_datetime
6. **Group Inconsistencies**: Identifies when the same entity has different date values across records

## Using `analyze_date_fields` for Multiple Fields

The module provides a utility function `analyze_date_fields` that can analyze multiple date fields in a single operation:

```python
from pamola_core.profiling.analyzers.date import analyze_date_fields

# Analyze multiple date fields
results = analyze_date_fields(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    date_fields=["birth_date", "creation_date", "modification_date"],
    id_column="resume_id",
    uid_column="UID",
    generate_visualization=True
)

# Process results
for field_name, result in results.items():
    if result.status.name == "SUCCESS":
        print(f"Field {field_name} analyzed successfully")
        # Process specific field results
```

The function has the following capabilities:

1. Automatically detecting date fields if none are explicitly specified
2. Validating the existence of ID and UID columns
3. Tracking overall progress across multiple fields
4. Providing a summary of successful and failed analyses

## Integration Points

The module integrates with:

1. **Data Source System**: Gets data through the `DataSource` abstraction
2. **Operation Framework**: Follows the standard operation workflow
3. **I/O System**: Uses `io.py` for saving results and artifacts
4. **Visualization System**: Uses `visualization.py` for creating distribution plots
5. **Reporting System**: Reports progress and results to the reporter object
6. **Registry System**: Registers the operation with `@register` decorator

## Conclusion

The `date.py` module provides comprehensive analysis of date fields with robust validation, distribution analysis, and anomaly detection capabilities. It follows a clean architecture that separates analytical logic from infrastructure concerns, making it both powerful and maintainable.

By delivering insights about date distributions, identifying anomalies, and detecting inconsistencies, it helps data scientists and analysts better understand the temporal characteristics of their data, which is essential for data quality assessment, demographic analysis, and time-based feature engineering tasks.