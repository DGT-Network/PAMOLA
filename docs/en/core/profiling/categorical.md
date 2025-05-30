# PAMOLA.CORE Categorical Analysis Module Documentation

## Overview

The PAMOLA.CORE Categorical Analysis module provides functionality for analyzing categorical fields in datasets. It is part of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) project's profiling system and follows the new operation architecture. The module is designed to identify patterns, distributions, and anomalies in categorical data, supporting the profiling and anonymization processes.

The module consists of two main components:
1. `categorical_utils.py` - Pamola Core analytical functions
2. `categorical.py` - Operation implementation integrating with the PAMOLA.CORE system

## Features

- **Distribution analysis** of categorical values with metrics like entropy and cardinality ratio
- **Frequency dictionary** creation for domain values
- **Anomaly detection** for identifying potential typos, unusual values, and inconsistencies
- **Visualization generation** of value distributions
- **Multi-field analysis** for processing multiple categorical fields in one operation
- **Progress tracking** for long-running operations
- **Seamless integration** with PAMOLA.CORE's IO, visualization, and logging systems

## Architecture

The module follows a clear separation of concerns:

```
┌────────────────────────┐     ┌───────────────────────┐
│  categorical.py        │     │  categorical_utils.py │
│                        │     │                       │
│ ┌───────────────────┐  │     │ ┌─────────────────┐   │
│ │CategoricalAnalyzer│──┼─────┼─►  analyze_field  │   │
│ └───────────────────┘  │     │ └─────────────────┘   │
│                        │     │                       │
│ ┌────────────────────┐ │     │ ┌─────────────────┐   │
│ │CategoricalOperation│ │     │ │create_dictionary│   │
│ └────────────────────┘ │     │ └─────────────────┘   │
│                        │     │                       │
│ ┌──────────────────┐   │     │ ┌────────────────┐    │
│ │analyze_fields    │   │     │ │detect_anomalies│    │
│ └──────────────────┘   │     │ └────────────────┘    │
└────────────────────────┘     └───────────────────────┘
        │   │     │
        ▼   ▼     ▼
┌─────────┐ ┌────────┐ ┌────────────────┐
│ io.py   │ │progress│ │visualization.py│
└─────────┘ └────────┘ └────────────────┘
```

This architecture ensures:
- Pure analytical logic is separated from operation management
- Reusable components can be utilized across the system
- Integration with other PAMOLA.CORE components is clean and standardized

## Key Components

### CategoricalAnalyzer

Static methods for analyzing categorical fields and estimating resource requirements.

```python
from pamola_core.profiling.analyzers.categorical import CategoricalAnalyzer

# Analyze a field directly
result = CategoricalAnalyzer.analyze(
    df=dataframe,
    field_name="education_level",
    top_n=10
)

# Check resource requirements
resources = CategoricalAnalyzer.estimate_resources(
    df=dataframe,
    field_name="education_level"
)
```

### CategoricalOperation

Implementation of the operation interface for the PAMOLA.CORE system, handling task execution, artifact generation, and integration.

```python
from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create a data source
data_source = DataSource(dataframes={"main": dataframe})

# Create and execute operation
operation = CategoricalOperation(
    field_name="education_level",
    top_n=15,
    min_frequency=2
)
result = operation.execute(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter
)
```

### analyze_categorical_fields

Helper function for analyzing multiple fields in one operation.

```python
from pamola_core.profiling.analyzers.categorical import analyze_categorical_fields

# Analyze multiple fields
results = analyze_categorical_fields(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter,
    cat_fields=["education_level", "gender", "job_title"]
)
```

## Function Reference

### CategoricalAnalyzer.analyze

```python
@staticmethod
def analyze(df: pd.DataFrame,
            field_name: str,
            top_n: int = 15,
            min_frequency: int = 1,
            **kwargs) -> Dict[str, Any]:
```

**Parameters:**
- `df` (required): DataFrame containing the data to analyze
- `field_name` (required): Name of the field to analyze
- `top_n` (default: 15): Number of top values to include in results
- `min_frequency` (default: 1): Minimum frequency for inclusion in dictionary
- `**kwargs`: Additional parameters:
  - `detect_anomalies` (default: True): Whether to detect anomalies
  - `anomaly_threshold` (default: 1): Minimum frequency for anomaly detection
  - `analyze_distribution` (default: True): Whether to analyze distribution characteristics

**Returns:**
- Dictionary with analysis results including:
  - Basic statistics (nulls, unique values)
  - Top values with frequencies
  - Entropy and cardinality metrics
  - Distribution characteristics
  - Detected anomalies (if any)
  - Value dictionary data

### CategoricalOperation.execute

```python
def execute(self,
            data_source: DataSource,
            task_dir: Path,
            reporter: Any,
            progress_tracker: Optional[ProgressTracker] = None,
            **kwargs) -> OperationResult:
```

**Parameters:**
- `data_source` (required): Source of data for the operation
- `task_dir` (required): Directory where task artifacts should be saved
- `reporter` (required): Reporter object for tracking progress and artifacts
- `progress_tracker` (optional): Progress tracker for the operation
- `**kwargs`: Additional parameters:
  - `generate_plots` (default: True): Whether to generate visualizations
  - `include_timestamp` (default: True): Whether to include timestamps in filenames
  - `profile_type` (default: 'categorical'): Type of profiling for organizing artifacts
  - `analyze_anomalies` (default: True): Whether to analyze anomalies

**Returns:**
- `OperationResult` object containing:
  - Status of operation
  - List of generated artifacts
  - Metrics and statistics
  - Error information (if any)

### analyze_categorical_fields

```python
def analyze_categorical_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        cat_fields: List[str] = None,
        **kwargs) -> Dict[str, OperationResult]:
```

**Parameters:**
- `data_source` (required): Source of data for the operations
- `task_dir` (required): Directory where task artifacts should be saved
- `reporter` (required): Reporter object for tracking progress and artifacts
- `cat_fields` (optional): List of categorical fields to analyze. If None, tries to find categorical fields automatically
- `**kwargs`: Additional parameters passed to each operation:
  - `top_n` (default: 15): Number of top values to include in results
  - `min_frequency` (default: 1): Minimum frequency for dictionary
  - `generate_plots` (default: True): Whether to generate plots
  - `include_timestamp` (default: True): Whether to include timestamps in filenames
  - `track_progress` (default: True): Whether to track overall progress

**Returns:**
- Dictionary mapping field names to their operation results

## Generated Artifacts

The module generates the following artifacts:

1. **JSON Statistics** (output directory)
   - `{field_name}_stats.json`: Statistical analysis of the categorical field
   - Contains counts, unique values, null percentages, entropy, top values, etc.

2. **CSV Dictionaries** (dictionaries directory)
   - `{field_name}_dictionary.csv`: Frequency dictionary of field values
   - Contains all values meeting the minimum frequency threshold
   - Includes frequency and percentage for each value

3. **Anomaly Reports** (dictionaries directory, if anomalies detected)
   - `{field_name}_anomalies.csv`: List of detected anomalies
   - Types include potential typos, single character values, numeric-like strings

4. **Visualizations** (visualizations directory)
   - `{field_name}_distribution.png`: Bar chart of top values distribution
   - Shows the most frequent values and their counts

## Usage Examples

### Basic Field Analysis

```python
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult

# Load data
df = pd.read_csv("resume_data.csv")

# Create data source
data_source = DataSource(dataframes={"main": df})

# Set up task directory
task_dir = Path("./analysis_results")


# Create a simple reporter
class SimpleReporter:
    def add_operation(self, *args, **kwargs):
        print(f"Operation: {args}")

    def add_artifact(self, *args, **kwargs):
        print(f"Artifact: {args}")


reporter = SimpleReporter()

# Execute analysis on education_level field
operation = CategoricalOperation(
    field_name="education_level",
    top_n=10,
    min_frequency=5
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    generate_plots=True
)

# Check results
if result.status == "success":
    print("Success! Artifacts:")
    for artifact in result.artifacts:
        print(f"- {artifact.artifact_type}: {artifact.path}")
else:
    print(f"Error: {result.error_message}")
```

### Multiple Field Analysis

```python
from pathlib import Path
from pamola_core.profiling.analyzers.categorical import analyze_categorical_fields
from pamola_core.utils.ops.op_data_source import DataSource

# Load data (assuming df is already defined)
data_source = DataSource(dataframes={"main": df})

# Set up task directory
task_dir = Path("./multi_field_analysis")

# Define reporter (assuming reporter is already defined)

# Analyze multiple fields
results = analyze_categorical_fields(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    cat_fields=["education_level", "job_category", "employment_type"],
    top_n=15,
    generate_plots=True
)

# Process results
for field_name, result in results.items():
    if result.status == "success":
        print(f"{field_name}: Analysis successful")
        print(f"  Unique values: {result.metrics.get('unique_values', 'N/A')}")
        print(f"  Null percentage: {result.metrics.get('null_percent', 'N/A')}%")
    else:
        print(f"{field_name}: Analysis failed - {result.error_message}")
```

### Direct Use of Analyzer for Quick Insights

```python
import pandas as pd
from pamola_core.profiling.analyzers.categorical import CategoricalAnalyzer

# Load data
df = pd.read_csv("survey_responses.csv")

# Get quick insights for a categorical field
insights = CategoricalAnalyzer.analyze(
    df=df,
    field_name="response_category",
    top_n=5
)

# Print key metrics
print(f"Total records: {insights['total_records']}")
print(f"Unique values: {insights['unique_values']}")
print(f"Null percentage: {insights['null_percent']}%")
print(f"Entropy (diversity): {insights['entropy']:.2f}")

# Display top values
print("\nTop values:")
for value, count in insights['top_values'].items():
    print(f"  {value}: {count}")

# Check for anomalies
if 'anomalies' in insights:
    print("\nPotential anomalies detected:")
    for anomaly_type, details in insights['anomalies'].items():
        print(f"  {anomaly_type}: {len(details)} instances")
```

## Integration with Other PAMOLA.CORE Components

The categorical analysis module integrates with:

1. **IO System** (`pamola_core.utils.io`)
   - Uses `write_json`, `write_dataframe_to_csv` for saving artifacts
   - Uses `ensure_directory` for directory management
   - Uses `get_timestamped_filename` for consistent file naming

2. **Visualization System** (`pamola_core.utils.visualization`)
   - Uses `plot_value_distribution` for creating standardized visualizations

3. **Progress Tracking** (`pamola_core.utils.progress`)
   - Uses `ProgressTracker` for monitoring operation progress

4. **Logging System** (`pamola_core.utils.logging`)
   - Uses standardized logging throughout the module

5. **Task System**
   - Implements the operation interface
   - Supports task-level reporting and artifact management

## Best Practices

1. **Field Selection**
   - For best results, analyze true categorical fields (limited set of values)
   - Free-text fields may generate extremely large dictionaries
   
2. **Performance Considerations**
   - Use appropriate `top_n` values (lower for fields with many unique values)
   - For very large datasets, consider using a sample for initial profiling

3. **Anomaly Handling**
   - Review detected anomalies for potential data quality issues
   - Consider using detected typos to clean data before anonymization

4. **Artifact Management**
   - Use consistent `task_dir` structures for organized outputs
   - Include timestamps in filenames for versioned artifacts

5. **Integration Guidelines**
   - Use `DataSource` to provide flexible data access
   - Always provide a reporter for tracking operation progress
   - Handle the returned `OperationResult` appropriately

## Conclusion

The PAMOLA.CORE Categorical Analysis module provides comprehensive tools for analyzing categorical fields in datasets. By leveraging its functionality, users can gain insights into data distributions, detect anomalies, and generate useful artifacts for further processing. The module's integration with the broader PAMOLA.CORE system ensures consistent outputs and seamless operation within the profiling workflow.