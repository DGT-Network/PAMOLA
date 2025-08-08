# PAMOLA CORE Correlation Analysis Module Documentation

## Overview

The Correlation Analysis module is a key component of the PAMOLA CORE profiling system, designed to analyze relationships between fields in datasets. It supports various correlation methods for different data types, including numeric-numeric, categorical-categorical, and mixed-type field relationships. This module is essential for understanding dependencies between data fields, identifying patterns, and making informed decisions during data anonymization.

The module consists of two main components:
1. `correlation_utils.py` - Core analytical functions for calculating correlations
2. `correlation.py` - Operation implementation integrating with the PAMOLA CORE system

## Features

- **Multiple correlation methods** based on data types (Pearson, Spearman, Cramer's V, correlation ratio)
- **Automatic method selection** based on field data types
- **Correlation matrix generation** for analyzing multiple field relationships simultaneously
- **Interpretation of correlation coefficients** in human-readable terms
- **Visualization generation** appropriate to data types (scatter plots, heatmaps, boxplots)
- **Multi-valued field (MVF) support** with custom parsing capabilities
- **Null value handling** with multiple strategies (drop, fill, pairwise)
- **Progress tracking** for long-running correlation analyses
- **Resource estimation** for optimizing performance with large datasets
- **Seamless integration** with PAMOLA CORE's IO, visualization, and logging systems

## Architecture

The module follows a clear separation of concerns:

```
┌────────────────────────┐     ┌───────────────────────┐
│  correlation.py        │     │  correlation_utils.py │
│                        │     │                       │
│ ┌───────────────────┐  │     │ ┌─────────────────┐   │
│ │CorrelationAnalyzer│──┼─────┼─►analyze_correlation│ │
│ └───────────────────┘  │     │ └─────────────────┘   │
│                        │     │                       │
│ ┌────────────────────┐ │     │ ┌─────────────────┐   │
│ │CorrelationOperation│ │     │ │calculate_cramers_v│ │
│ └────────────────────┘ │     │ └─────────────────┘   │
│                        │     │                       │
│ ┌──────────────────────┐     │ ┌──────────────────┐  │
│ │CorrelationMatrixOp   │     │ │correlation_ratio │  │
│ └──────────────────────┘     │ └──────────────────┘  │
│                        │     │                       │
│ ┌──────────────────┐   │     │ ┌────────────────┐    │
│ │analyze_correlations│ │     │ │interpret_correlation│
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
- Different correlation methods are properly encapsulated
- Reusable components can be utilized across the system
- Integration with other PAMOLA CORE components is clean and standardized

## Key Components

### CorrelationAnalyzer

Static methods for analyzing correlations between fields and creating correlation matrices.

```python
from pamola_core.profiling.analyzers.correlation import CorrelationAnalyzer

# Analyze correlation between two fields
result = CorrelationAnalyzer.analyze(
    df=dataframe,
    field1="salary",
    field2="education_level"
)

# Create a correlation matrix for multiple fields
matrix_result = CorrelationAnalyzer.analyze_matrix(
    df=dataframe,
    fields=["salary", "education_level", "area_name", "experience_years"]
)

# Estimate resource requirements
resources = CorrelationAnalyzer.estimate_resources(
    df=dataframe,
    field1="salary",
    field2="education_level"
)
```

### CorrelationOperation

Implementation of the operation interface for the PAMOLA CORE system, handling analysis of correlation between two fields.

```python
from pamola_core.profiling.analyzers.correlation import CorrelationOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create a data source
data_source = DataSource(dataframes={"main": dataframe})

# Create and execute operation
operation = CorrelationOperation(
    field1="salary",
    field2="education_level",
    method="correlation_ratio"  # Optional - auto-selected if not specified
)
result = operation.execute(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter
)
```

### CorrelationMatrixOperation

Implementation for creating correlation matrices across multiple fields.

```python
from pamola_core.profiling.analyzers.correlation import CorrelationMatrixOperation

# Create and execute matrix operation
matrix_operation = CorrelationMatrixOperation(
    fields=["salary", "education_level", "area_name", "experience_years"]
)
matrix_result = matrix_operation.execute(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter
)
```

### analyze_correlations

Helper function for analyzing multiple field pairs in one operation.

```python
from pamola_core.profiling.analyzers.correlation import analyze_correlations

# Define pairs to analyze
field_pairs = [
    ("salary", "education_level"),
    ("salary", "experience_years"),
    ("age", "salary")
]

# Analyze multiple pairs
results = analyze_correlations(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter,
    pairs=field_pairs
)
```

## Correlation Methods

The module supports several correlation methods, automatically selecting the appropriate one based on data types:

| Method | Data Types | Description | Range |
|--------|------------|-------------|-------|
| `pearson` | numeric-numeric | Linear correlation coefficient | -1 to 1 |
| `spearman` | numeric-numeric | Rank correlation coefficient | -1 to 1 |
| `cramers_v` | categorical-categorical | Based on chi-squared statistic | 0 to 1 |
| `correlation_ratio` | categorical-numeric | Measures relationship between categorical and numeric | 0 to 1 |
| `point_biserial` | binary-numeric | Special case for binary categorical variables | -1 to 1 |

## Function Reference

### CorrelationAnalyzer.analyze

```python
@staticmethod
def analyze(df: pd.DataFrame,
            field1: str,
            field2: str,
            method: Optional[str] = None,
            **kwargs) -> Dict[str, Any]:
```

**Parameters:**
- `df` (required): DataFrame containing the data
- `field1` (required): Name of the first field
- `field2` (required): Name of the second field
- `method` (optional): Correlation method to use (auto-selected if None)
- `**kwargs`: Additional parameters:
  - `mvf_parser`: Callable function to parse multi-valued fields
  - `null_handling`: Method for handling nulls ('drop', 'fill', 'pairwise')
  - `include_plots`: Whether to include plot data in results

**Returns:**
- Dictionary with analysis results including:
  - Field information (names, types)
  - Correlation method used
  - Correlation coefficient
  - p-value (if applicable)
  - Interpretation of correlation strength
  - Null value statistics
  - Plot data for visualization (if requested)

### CorrelationOperation.execute

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
  - `profile_type` (default: 'correlation'): Type of profiling for organizing artifacts
  - `null_handling` (default: 'drop'): Method for handling nulls

**Returns:**
- `OperationResult` object containing:
  - Status of operation
  - List of generated artifacts
  - Metrics and statistics
  - Error information (if any)

### CorrelationMatrixOperation.execute

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
  - `include_timestamp` (default: True): Whether to include timestamps
  - `profile_type` (default: 'correlation'): Type of profiling for artifacts
  - `null_handling` (default: 'drop'): Method for handling nulls
  - `min_threshold` (default: 0.3): Minimum correlation threshold for significant correlations

**Returns:**
- `OperationResult` object containing matrix analysis results and artifacts

### analyze_correlations

```python
def analyze_correlations(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        pairs: List[Tuple[str, str]],
        **kwargs) -> Dict[str, OperationResult]:
```

**Parameters:**
- `data_source` (required): Source of data for the operations
- `task_dir` (required): Directory where task artifacts should be saved
- `reporter` (required): Reporter object for tracking progress and artifacts
- `pairs` (required): List of field pairs to analyze as tuples (field1, field2)
- `**kwargs`: Additional parameters:
  - `methods`: Dictionary mapping field pairs to correlation methods
  - `null_handling` (default: 'drop'): Method for handling nulls
  - `generate_visualization` (default: True): Whether to generate visualization
  - `track_progress` (default: True): Whether to track overall progress

**Returns:**
- Dictionary mapping pair names to their operation results

## Generated Artifacts

The module generates the following artifacts:

1. **JSON Analysis Results** (output directory)
   - `{field1}_{field2}_correlation.json`: Analysis results for a field pair
   - Contains correlation coefficient, method, interpretation, p-value, etc.
   - `correlation_matrix.json`: Results for correlation matrix analysis
   - Contains full matrix, methods used, significant correlations, etc.

2. **Visualizations** (visualizations directory)
   - For numeric-numeric correlations:
     - `{field1}_{field2}_correlation_plot.png`: Scatter plot with trend line
   - For categorical-numeric correlations:
     - `{field1}_{field2}_correlation_plot.png`: Box plot showing distributions
   - For categorical-categorical correlations:
     - `{field1}_{field2}_correlation_plot.png`: Heatmap of contingency table
   - For correlation matrices:
     - `correlation_matrix_heatmap.png`: Heatmap visualization of correlation matrix

## Usage Examples

### Basic Correlation Analysis

```python
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.correlation import CorrelationOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult

# Load data
df = pd.read_csv("employee_data.csv")

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

# Execute correlation analysis between salary and experience
operation = CorrelationOperation(
    field1="salary",
    field2="experience_years"
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    generate_visualization=True
)

# Check results
if result.status == OperationStatus.SUCCESS:
    print("Success! Correlation details:")
    print(f"Method: {result.metrics.get('correlation_method')}")
    print(f"Coefficient: {result.metrics.get('correlation_coefficient')}")
    if 'p_value' in result.metrics:
        print(f"P-value: {result.metrics.get('p_value')}")
    print("\nArtifacts:")
    for artifact in result.artifacts:
        print(f"- {artifact.artifact_type}: {artifact.path}")
else:
    print(f"Error: {result.error_message}")
```

### Correlation Matrix Analysis

```python
from pathlib import Path
from pamola_core.profiling.analyzers.correlation import CorrelationMatrixOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Load data (assuming df is already defined)
data_source = DataSource(dataframes={"main": df})

# Set up task directory
task_dir = Path("./correlation_matrix_analysis")

# Define fields to analyze
fields = ["salary", "experience_years", "education_level", "performance_score", "department"]

# Create and execute matrix operation
matrix_operation = CorrelationMatrixOperation(
    fields=fields,
    methods={
        # Specify custom methods for particular pairs (optional)
        "salary_department": "correlation_ratio",
        "education_level_department": "cramers_v"
    }
)

result = matrix_operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    min_threshold=0.2  # Lower threshold to capture more correlations
)

# Process results
if result.status == OperationStatus.SUCCESS:
    print(f"Successfully analyzed {result.metrics.get('fields_analyzed')} fields")
    print(f"Found {result.metrics.get('significant_correlations')} significant correlations")
    print("\nArtifacts:")
    for artifact in result.artifacts:
        print(f"- {artifact.artifact_type}: {artifact.path}")
else:
    print(f"Error: {result.error_message}")
```

### Multiple Correlation Analyses

```python
from pamola_core.profiling.analyzers.correlation import analyze_correlations

# Define pairs to analyze
field_pairs = [
    ("salary", "experience_years"),
    ("salary", "education_level"),
    ("experience_years", "performance_score"),
    ("department", "performance_score")
]

# Analyze multiple pairs
results = analyze_correlations(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    pairs=field_pairs,
    methods={
        # Specify custom methods for particular pairs (optional)
        "salary_education_level": "correlation_ratio"
    }
)

# Process results
for pair_name, result in results.items():
    if result.status == OperationStatus.SUCCESS:
        coefficient = result.metrics.get('correlation_coefficient', 'N/A')
        method = result.metrics.get('correlation_method', 'unknown')
        print(f"{pair_name}: {coefficient:.4f} ({method})")
    else:
        print(f"{pair_name}: Analysis failed - {result.error_message}")
```

### Direct Use of Analyzer for Quick Insights

```python
import pandas as pd
from pamola_core.profiling.analyzers.correlation import CorrelationAnalyzer

# Load data
df = pd.read_csv("customer_data.csv")

# Get quick correlation insights
correlation = CorrelationAnalyzer.analyze(
    df=df,
    field1="customer_age",
    field2="purchase_amount"
)

# Print key metrics
print(f"Correlation method: {correlation['method']}")
print(f"Correlation coefficient: {correlation['correlation_coefficient']:.4f}")
if 'p_value' in correlation:
    print(f"P-value: {correlation['p_value']:.6f}")
print(f"Interpretation: {correlation['interpretation']}")
print(f"Sample size: {correlation['sample_size']}")

# Check null statistics
null_stats = correlation['null_stats']
print(f"\nNull handling summary:")
print(f"Total rows: {null_stats['total_rows']}")
print(f"Rows with nulls: {null_stats['null_rows']} ({null_stats['null_percentage']}%)")
print(f"Rows used in analysis: {null_stats['rows_after_handling']}")
```

## Correlation Coefficient Interpretation

The module provides automatic interpretation of correlation coefficients based on their magnitude and direction:

| Correlation Method | Range | Interpretation Examples |
|--------------------|-------|-------------------------|
| pearson, spearman, point_biserial | -1 to 1 | "Strong negative correlation", "Moderate positive correlation" |
| cramers_v, correlation_ratio | 0 to 1 | "Weak association", "Strong association" |

For directional correlations (pearson, spearman), the interpretation includes both strength and direction. For non-directional measures (cramers_v, correlation_ratio), only the strength is indicated.

## Null Value Handling

The module provides three strategies for handling null values:

1. **drop** (default): Remove rows with any null values in the analyzed fields
2. **fill**: Replace nulls with default values (0 for numeric, empty string for categorical)
3. **pairwise**: Retain rows with at least one non-null value (for correlation matrices)

Each strategy affects the sample size and potentially the correlation results, especially for fields with many null values.

## Multi-valued Field Support

For fields containing multiple values (like arrays or comma-separated lists), the module provides support through a custom parser function:

```python
def parse_mvf(value):
    """Parse a multi-valued field into a list of values."""
    if pd.isna(value):
        return []
    
    # Handle various formats: string lists, JSON arrays, etc.
    try:
        if isinstance(value, str):
            if value.startswith('[') and value.endswith(']'):
                # Handle JSON-like strings
                import ast
                return ast.literal_eval(value)
            elif ',' in value:
                # Handle comma-separated values
                return [v.strip() for v in value.split(',')]
        return [value]
    except:
        return [value]

# Use the parser in the operation
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    mvf_parser=parse_mvf
)
```

## Integration with Other PAMOLA CORE Components

The correlation analysis module integrates with:

1. **IO System** (`pamola_core.utils.io`)
   - Uses `write_json` for saving analysis results
   - Uses `ensure_directory` for directory management
   - Uses `get_timestamped_filename` for consistent file naming

2. **Visualization System** (`pamola_core.utils.visualization`)
   - Uses `create_scatter_plot`, `create_boxplot`, `create_heatmap` for field pair visualizations
   - Uses `create_correlation_matrix` for matrix visualization

3. **Progress Tracking** (`pamola_core.utils.progress`)
   - Uses `ProgressTracker` for monitoring operation progress

4. **Logging System** (`pamola_core.utils.logging`)
   - Uses standardized logging throughout the module

5. **Task System**
   - Implements the operation interfaces (`BaseOperation`, `FieldOperation`)
   - Supports task-level reporting and artifact management
   - Returns `OperationResult` objects for consistent handling

## Best Practices

1. **Field Selection**
   - Choose fields that might have meaningful relationships
   - Ensure fields have sufficient non-null values for reliable correlation analysis
   
2. **Method Selection**
   - Let the system automatically select methods when possible
   - Override methods only when you have specific requirements
   
3. **Performance Considerations**
   - For large datasets, analyze important field pairs individually before creating full matrices
   - Use resource estimation to anticipate computation time and memory requirements
   
4. **Interpretation Guidelines**
   - Consider both statistical significance (p-value) and coefficient magnitude
   - Remember that correlation does not imply causation
   - Validate findings through additional analysis or domain knowledge

5. **Integration Guidelines**
   - Use `DataSource` to provide flexible data access
   - Always provide a reporter for tracking operation progress
   - Handle the returned `OperationResult` appropriately
   - Use correlation findings to inform anonymization strategies

## Conclusion

The PAMOLA CORE Correlation Analysis module provides comprehensive tools for understanding relationships between fields in datasets. It supports various correlation methods suitable for different data types, produces informative visualizations, and integrates seamlessly with the PAMOLA CORE profiling system. 

By leveraging this module, users can identify meaningful patterns and dependencies in their data, which is crucial for making informed decisions during the anonymization process while preserving the analytical value of the data.