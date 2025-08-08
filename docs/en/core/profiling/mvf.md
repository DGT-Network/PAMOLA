# PAMOLA.CORE Multi-Valued Field (MVF) Analysis Module Documentation

## Overview

The Multi-Valued Field (MVF) Analysis module is a specialized component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) profiling system, designed to analyze fields containing multiple values per record. This module effectively processes and analyzes various formats of multi-valued data, such as JSON arrays, string representations of arrays, or comma-separated values, which are common in resume databases for fields like skill sets, preferred work schedules, employment types, or driver's license categories.

The module consists of two main components:
1. `mvf_utils.py` - Pamola Core analytical functions for parsing and analyzing MVF fields
2. `mvf.py` - Operation implementation integrating with the PAMOLA.CORE system

## Features

- **Multiple format support** for MVF fields (JSON arrays, string arrays, comma-separated values)
- **Automatic format detection** to handle mixed format datasets
- **Individual value analysis** with frequency distributions
- **Combinations analysis** to understand value co-occurrence patterns
- **Value count distribution** analysis (how many values each record contains)
- **Empty array detection** differentiated from NULL values
- **Dictionary generation** for both individual values and combinations
- **Visualization generation** for values, combinations, and count distributions
- **Standardization functions** to convert between different MVF formats
- **Robust error handling** with configurable error limits
- **Performance estimation** for large datasets
- **Progress tracking** for long-running operations
- **Seamless integration** with PAMOLA.CORE's IO, visualization, and logging systems

## Architecture

The module follows a clear separation of concerns:

```
┌──────────────────┐     ┌───────────────────┐
│  mvf.py          │     │  mvf_utils.py     │
│                  │     │                   │
│ ┌──────────────┐ │     │ ┌───────────────┐ │
│ │ MVFAnalyzer  │─┼─────┼─► parse_mvf     │ │
│ └──────────────┘ │     │ └───────────────┘ │
│                  │     │                   │
│ ┌──────────────┐ │     │ ┌───────────────┐ │
│ │ MVFOperation │ │     │ │analyze_mvf_field│
│ └──────────────┘ │     │ └───────────────┘ │
│                  │     │                   │
│ ┌───────────────┐│     │ ┌────────────────┐│
│ │analyze_mvf_fields│    │ │create_dictionary│
│ └───────────────┘│     │ └────────────────┘│
└──────────────────┘     └───────────────────┘
        │   │     │
        ▼   ▼     ▼
┌─────────┐ ┌────────┐ ┌────────────────┐
│ io.py   │ │progress│ │visualization.py│
└─────────┘ └────────┘ └────────────────┘
```

This architecture ensures:
- Pure analytical logic is separated from operation management
- MVF parsing logic is encapsulated and reusable
- Different analysis types are properly organized
- Integration with other PAMOLA.CORE components is clean and standardized

## Key Components

### MVFAnalyzer

Static methods for analyzing MVF fields, parsing formats, and creating dictionaries.

```python
from pamola_core.profiling.analyzers.mvf import MVFAnalyzer

# Analyze an MVF field
result = MVFAnalyzer.analyze(
    df=dataframe,
    field_name="work_schedules",
    top_n=15,
    min_frequency=2
)

# Parse an MVF field and add a new column with parsed values
parsed_df = MVFAnalyzer.parse_field(
    df=dataframe,
    field_name="work_schedules",
    format_type="json"  # Optional - auto-detected if not specified
)

# Create a dictionary of values
values_dict = MVFAnalyzer.create_value_dictionary(
    df=dataframe,
    field_name="work_schedules",
    min_frequency=5
)

# Create a dictionary of value combinations
combinations_dict = MVFAnalyzer.create_combinations_dictionary(
    df=dataframe,
    field_name="work_schedules",
    min_frequency=2
)
```

### MVFOperation

Implementation of the operation interface for the PAMOLA.CORE system, handling task execution, artifact generation, and integration.

```python
from pamola_core.profiling.analyzers.mvf import MVFOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create a data source
data_source = DataSource(dataframes={"main": dataframe})

# Create and execute operation
operation = MVFOperation(
    field_name="work_schedules",
    top_n=15,
    min_frequency=2
)
result = operation.execute(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter,
    format_type="json"  # Optional format hint
)
```

### analyze_mvf_fields

Helper function for analyzing multiple MVF fields in one operation.

```python
from pamola_core.profiling.analyzers.mvf import analyze_mvf_fields

# Analyze multiple MVF fields
results = analyze_mvf_fields(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter,
    mvf_fields=["work_schedules", "employments", "driver_license_types"]
)
```

## MVF Format Support

The module supports several MVF formats, automatically detecting and parsing each format:

| Format Type | Example | Description |
|-------------|---------|-------------|
| JSON Array | `["Value1", "Value2"]` | Standard JSON array format |
| Array String | `"['Value1', 'Value2']"` | String representation of an array (Python-like) |
| CSV | `"Value1, Value2"` | Simple comma-separated values |
| Empty Array | `"[]"` or `[]` | Empty array representation |
| NULL | `null` or `NaN` | Missing values |

## Function Reference

### MVFAnalyzer.analyze

```python
@staticmethod
def analyze(df: pd.DataFrame,
            field_name: str,
            top_n: int = 20,
            min_frequency: int = 1,
            **kwargs) -> Dict[str, Any]:
```

**Parameters:**
- `df` (required): DataFrame containing the data to analyze
- `field_name` (required): Name of the field to analyze
- `top_n` (default: 20): Number of top values to include in results
- `min_frequency` (default: 1): Minimum frequency for inclusion in dictionary
- `**kwargs`: Additional parameters:
  - `format_type`: Format hint for parsing (auto-detected if not specified)
  - Other parameters passed to `parse_mvf`

**Returns:**
- Dictionary with analysis results including:
  - Basic statistics (total records, nulls, empty arrays)
  - Value analysis (unique values, frequency distribution)
  - Combinations analysis (unique combinations, frequency distribution)
  - Value count distribution (how many values per record)
  - Error statistics (if any errors occurred during parsing)

### MVFOperation.execute

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
  - `generate_visualization` (default: True): Whether to generate visualizations
  - `include_timestamp` (default: True): Whether to include timestamps in filenames
  - `profile_type` (default: 'mvf'): Type of profiling for organizing artifacts
  - `format_type` (default: None): Format type hint for parsing
  - `parse_kwargs` (default: {}): Additional parameters for MVF parsing

**Returns:**
- `OperationResult` object containing:
  - Status of operation
  - List of generated artifacts
  - Metrics and statistics
  - Error information (if any)

### analyze_mvf_fields

```python
def analyze_mvf_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        mvf_fields: List[str],
        **kwargs) -> Dict[str, OperationResult]:
```

**Parameters:**
- `data_source` (required): Source of data for the operations
- `task_dir` (required): Directory where task artifacts should be saved
- `reporter` (required): Reporter object for tracking progress and artifacts
- `mvf_fields` (required): List of MVF fields to analyze
- `**kwargs`: Additional parameters passed to each operation:
  - `top_n` (default: 20): Number of top values to include in results
  - `min_frequency` (default: 1): Minimum frequency for inclusion in dictionary
  - `generate_visualization` (default: True): Whether to generate visualization
  - `include_timestamp` (default: True): Whether to include timestamps in filenames
  - `format_type` (default: None): Format type hint for parsing
  - `parse_kwargs` (default: {}): Additional parameters for MVF parsing

**Returns:**
- Dictionary mapping field names to their operation results

### parse_mvf

```python
def parse_mvf(value: Any, 
              format_type: Optional[str] = None,
              separator: str = ',',
              quote_char: str = '"',
              array_markers: Tuple[str, str] = ('[', ']'),
              handle_json: bool = True) -> List[str]:
```

**Parameters:**
- `value` (required): The MVF value to parse
- `format_type` (optional): Format type hint: 'json', 'array_string', 'csv', or None (auto-detect)
- `separator` (default: ','): Character used to separate values
- `quote_char` (default: '"'): Character used for quoting values
- `array_markers` (default: ('[', ']')): Start and end markers for array representation
- `handle_json` (default: True): Whether to attempt parsing as JSON

**Returns:**
- List of individual values extracted from the MVF field

## Generated Artifacts

The module generates the following artifacts:

1. **JSON Statistics** (output directory)
   - `{field_name}_stats.json`: Statistical analysis of the MVF field
   - Contains counts, unique values, null percentages, empty arrays, etc.

2. **CSV Dictionaries** (dictionaries directory)
   - `{field_name}_values_dictionary.csv`: Frequency dictionary of individual values
   - `{field_name}_combinations_dictionary.csv`: Frequency dictionary of value combinations
   - Both include frequency and percentage columns

3. **Visualizations** (visualizations directory)
   - `{field_name}_values_distribution.png`: Bar chart of individual value distribution
   - `{field_name}_combinations_distribution.png`: Bar chart of combinations distribution
   - `{field_name}_value_counts_distribution.png`: Bar chart showing how many values each record contains

## Usage Examples

### Basic MVF Field Analysis

```python
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.mvf import MVFOperation
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

# Execute analysis on work_schedules field
operation = MVFOperation(
    field_name="work_schedules",
    top_n=15,
    min_frequency=2
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    generate_visualization=True
)

# Check results
if result.status == OperationStatus.SUCCESS:
    print("Success! Artifacts:")
    for artifact in result.artifacts:
        print(f"- {artifact.artifact_type}: {artifact.path}")

    # Print some key metrics
    print(f"\nUnique values: {result.metrics['unique_values']}")
    print(f"Unique combinations: {result.metrics['unique_combinations']}")
    print(f"Average values per record: {result.metrics['avg_values_per_record']}")
else:
    print(f"Error: {result.error_message}")
```

### Analysis with Custom Format Handling

```python
from pathlib import Path
from pamola_core.profiling.analyzers.mvf import MVFOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Custom parsing parameters for non-standard format
parse_kwargs = {
    "separator": ";",  # Use semicolon as separator
    "quote_char": "'",  # Use single quotes
    "array_markers": ("<", ">")  # Use angle brackets instead of square brackets
}

# Create and execute operation with custom parsing
operation = MVFOperation(
    field_name="skills",
    top_n=20,
    min_frequency=1
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    parse_kwargs=parse_kwargs
)
```

### Multiple Field Analysis

```python
from pamola_core.profiling.analyzers.mvf import analyze_mvf_fields

# Define MVF fields to analyze
mvf_fields = ["work_schedules", "employments", "driver_license_types", "skills"]

# Analyze multiple fields
results = analyze_mvf_fields(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    mvf_fields=mvf_fields,
    top_n=10,
    min_frequency=2
)

# Process results
for field_name, result in results.items():
    if result.status == OperationStatus.SUCCESS:
        print(f"{field_name}: Analysis successful")
        print(f"  Unique values: {result.metrics.get('unique_values', 'N/A')}")
        print(f"  Average values per record: {result.metrics.get('avg_values_per_record', 'N/A')}")
    else:
        print(f"{field_name}: Analysis failed - {result.error_message}")
```

### Direct Use of MVFAnalyzer for Quick Insights

```python
import pandas as pd
from pamola_core.profiling.analyzers.mvf import MVFAnalyzer

# Load data
df = pd.read_csv("employee_data.csv")

# Get quick insights for an MVF field
insights = MVFAnalyzer.analyze(
    df=df,
    field_name="language_skills",
    top_n=5
)

# Print key metrics
print(f"Total records: {insights['total_records']}")
print(f"Null records: {insights['null_count']} ({insights['null_percentage']}%)")
print(f"Empty arrays: {insights['empty_arrays_count']} ({insights['empty_arrays_percentage']}%)")
print(f"Unique values: {insights['unique_values']}")
print(f"Unique combinations: {insights['unique_combinations']}")
print(f"Average values per record: {insights['avg_values_per_record']}")

# Display top values
print("\nTop values:")
for value, count in insights['values_analysis'].items():
    print(f"  {value}: {count}")

# Display top combinations
print("\nTop combinations:")
for combo, count in insights['combinations_analysis'].items():
    print(f"  {combo}: {count}")

# Show value count distribution
print("\nValue count distribution:")
for count, frequency in insights['value_counts_distribution'].items():
    print(f"  {count} values: {frequency} records")
```

## Error Handling

The module includes robust error handling for parsing MVF fields:

1. **Graceful Error Recovery**: The parser attempts multiple parsing strategies before giving up
2. **Error Logging**: First 10 parsing errors are logged with details
3. **Error Limit**: If more than 1000 parsing errors occur, the operation stops to prevent wasting resources
4. **Error Statistics**: The results include details about parsing errors (count, percentage)

Example of handling parsing errors:

```python
# Define a potentially problematic field
parse_result = MVFAnalyzer.analyze(
    df=df,
    field_name="problematic_field"
)

# Check if there were parsing errors
if 'error_count' in parse_result:
    print(f"Encountered {parse_result['error_count']} parsing errors")
    print(f"Error percentage: {parse_result['error_percentage']}%")
    
    # If operation failed entirely due to too many errors
    if 'error' in parse_result:
        print(f"Analysis failed: {parse_result['error']}")
```

## MVF Format Standardization

The module provides functions to standardize MVF formats:

```python
from pamola_core.profiling.commons.mvf_utils import standardize_mvf_format

# Convert MVF values to a consistent format
csv_value = "Value1, Value2, Value3"

# Convert to JSON array format
json_array = standardize_mvf_format(csv_value, target_format='json')
# Result: ["Value1", "Value2", "Value3"]

# Convert to array string format
array_string = standardize_mvf_format(csv_value, target_format='array_string')
# Result: "['Value1', 'Value2', 'Value3']"

# Convert to list (Python object)
value_list = standardize_mvf_format(csv_value, target_format='list')
# Result: ['Value1', 'Value2', 'Value3']
```

## Performance Considerations

For large datasets, the module provides resource estimation:

```python
from pamola_core.profiling.analyzers.mvf import MVFAnalyzer

# Estimate resources needed for analysis
resources = MVFAnalyzer.estimate_resources(
    df=large_dataframe,
    field_name="work_schedules"
)

print(f"Estimated memory: {resources['estimated_memory_mb']} MB")
print(f"Estimated time: {resources['estimated_time_seconds']} seconds")
print(f"Dask recommended: {resources['dask_recommended']}")
```

Key performance techniques used by the module:
1. **Memory-efficient parsing**: Values are processed one at a time
2. **Automatic format detection**: No need for redundant parsing attempts
3. **Error limits**: Prevents wasting resources on malformed data
4. **Progress tracking**: Monitors performance and provides feedback

## Integration with Other PAMOLA.CORE Components

The MVF analysis module integrates with:

1. **IO System** (`pamola_core.utils.io`)
   - Uses `write_json` for saving analysis results
   - Uses `ensure_directory` for directory management
   - Uses `get_timestamped_filename` for consistent file naming

2. **Visualization System** (`pamola_core.utils.visualization`)
   - Uses `plot_value_distribution` for value and combination distributions
   - Uses `create_bar_plot` for value count distributions

3. **Progress Tracking** (`pamola_core.utils.progress`)
   - Uses `ProgressTracker` for monitoring operation progress

4. **Logging System** (`pamola_core.utils.logging`)
   - Uses standardized logging throughout the module

5. **Task System**
   - Implements the operation interface (`FieldOperation`)
   - Supports task-level reporting and artifact management
   - Returns `OperationResult` objects for consistent handling

## Best Practices

1. **Field Selection**
   - Choose fields that are truly multi-valued (multiple values per record)
   - Common examples: skill sets, work schedules, employments, languages

2. **Format Handling**
   - Let the system auto-detect formats when possible
   - Provide format hints only when necessary (e.g., for non-standard formats)
   - Use custom parse parameters for special cases

3. **Performance Considerations**
   - For large datasets, check resource estimates before running full analysis
   - Monitor error rates to identify data quality issues
   - Consider analyzing a sample first for very large datasets

4. **Artifact Usage**
   - Use value dictionaries to understand the domain of each MVF field
   - Use combination dictionaries to understand value co-occurrence patterns
   - Use value count distributions to understand data completeness

5. **Integration Recommendations**
   - Use `DataSource` to provide flexible data access
   - Always provide a reporter for tracking operation progress
   - Handle the returned `OperationResult` objects appropriately

## Conclusion

The PAMOLA.CORE Multi-Valued Field Analysis module provides comprehensive tools for understanding and analyzing fields containing multiple values per record. It supports various formats, generates useful artifacts, and integrates seamlessly with the PAMOLA.CORE profiling system.

By leveraging this module, data professionals can extract valuable insights from multi-valued fields, which is particularly important for resume data where many attributes (skills, preferences, qualifications) are inherently multi-valued.