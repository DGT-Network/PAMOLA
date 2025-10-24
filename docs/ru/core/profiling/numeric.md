# Numeric Analysis Module Documentation

## Overview

The `numeric.py` module is a pamola core component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) profiling system, designed for comprehensive analysis of numeric fields within datasets. It provides statistical analysis, distribution visualization, outlier detection, and normality testing for numeric data, making it essential for data profiling tasks that involve continuous or discrete numeric variables.

## Architecture

The module follows a clean separation of concerns with two primary classes:

1. **NumericAnalyzer**: Contains pure analytical logic for calculating statistics and performing numerical analysis.
2. **NumericOperation**: Orchestrates the profiling workflow, including data acquisition, analysis execution, artifact generation, and result reporting.

This architecture adheres to the operation framework of the PAMOLA.CORE system:

```
┌─────────────────┐      ┌───────────────────┐      ┌────────────────────┐
│   DataSource    │──────▶  NumericOperation │─────▶   OperationResult   │
└─────────────────┘      └────────┬──────────┘      └────────────────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │  NumericAnalyzer  │
                         └───────────────────┘
```

The module also interacts with several supporting components:
- `numeric_utils.py`: Contains low-level statistical functions
- `io.py`: Handles saving analysis results and other artifacts
- `visualization.py`: Creates statistical visualizations
- `op_result.py`: Structures operation results and artifacts

## Key Capabilities

The module performs the following types of analysis:

1. **Basic Statistics**: Calculates min, max, mean, median, standard deviation, variance
2. **Extended Statistics**: Analyzes skewness, kurtosis, percentiles
3. **Distribution Analysis**: Generates histogram data and visualizations
4. **Special Value Analysis**: Identifies zero, near-zero, negative, and positive values
5. **Outlier Detection**: Uses IQR method to identify statistical outliers
6. **Normality Testing**: Applies multiple tests to determine if data follows a normal distribution

## Generated Artifacts

The module produces the following artifacts:

1. **JSON Statistics Report** (`{field_name}_stats.json`):
   - Basic statistics (min, max, mean, median, etc.)
   - Null/missing value analysis
   - Distribution metrics
   - Outlier information
   - Normality test results

2. **Histogram Visualization** (`{field_name}_distribution.png`):
   - Frequency distribution of values
   - Shows the overall shape of the data

3. **Box Plot** (`{field_name}_boxplot.png`):
   - Displays the five-number summary (min, Q1, median, Q3, max)
   - Visualizes outliers

4. **Q-Q Plot** (`{field_name}_qq_plot.png`):
   - Shows how the data distribution compares to a normal distribution
   - Generated only when normality testing is enabled

## Usage Example

Here's an example of how to use `NumericOperation` to analyze a numeric field:

```python
from pathlib import Path
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.utils.ops.op_data_source import DataSource
from your_reporting_module import Reporter  # Your report system

# Create a reporter instance
reporter = Reporter()

# Prepare data source (DataFrame with numeric field)
data_source = DataSource.from_dataframe(df, "main")

# Create output directory
task_dir = Path("output/analysis")

# Create and run the operation
operation = NumericOperation(
    field_name="salary",
    bins=15,
    detect_outliers=True,
    test_normality=True
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
    print(f"Mean value: {stats['mean']}")
    print(f"Outlier count: {stats['outliers_count']}")

    # List the generated artifacts
    for artifact in result.artifacts:
        print(f"Artifact: {artifact.artifact_type}, Path: {artifact.path}")
```

## Parameters

### NumericOperation Class

**Constructor Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `field_name` | str | Yes | - | Name of the numeric field to analyze |
| `bins` | int | No | 10 | Number of bins for histogram generation |
| `detect_outliers` | bool | No | True | Whether to perform outlier detection |
| `test_normality` | bool | No | True | Whether to perform normality testing |
| `description` | str | No | "" | Custom description of the operation |

**Execute Method Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data_source` | DataSource | Yes | - | Source of data for the operation |
| `task_dir` | Path | Yes | - | Directory for saving artifacts |
| `reporter` | Any | Yes | - | Reporter object for tracking artifacts |
| `progress_tracker` | ProgressTracker | No | None | Progress tracking object |
| `near_zero_threshold` | float | No | 1e-10 | Threshold for near-zero detection |
| `generate_visualization` | bool | No | True | Whether to create visualizations |
| `include_timestamp` | bool | No | True | Whether to include timestamps in filenames |
| `profile_type` | str | No | "numeric" | Type of profiling for organizing artifacts |

## Return Value

The `execute` method returns an `OperationResult` object with the following properties:

| Property | Type | Description |
|----------|------|-------------|
| `status` | OperationStatus | SUCCESS or ERROR indicating operation outcome |
| `artifacts` | List[OperationArtifact] | List of generated files (JSONs, PNGs) |
| `metrics` | Dict[str, Any] | Key statistical measures extracted from the analysis |
| `error_message` | str | Error details if the operation failed |
| `execution_time` | float | Time taken for the operation in seconds |

## Metrics and Statistical Calculations

The module calculates the following metrics for numeric fields:

### Basic Statistics
| Metric | Description |
|--------|-------------|
| `min` | Minimum value in the dataset |
| `max` | Maximum value in the dataset |
| `mean` | Arithmetic mean (average) |
| `median` | Middle value when data is sorted |
| `std` | Standard deviation (measure of dispersion) |
| `variance` | Square of standard deviation |
| `sum` | Sum of all values |

### Distribution Metrics
| Metric | Description |
|--------|-------------|
| `skewness` | Measure of asymmetry in the distribution |
| `kurtosis` | Measure of "tailedness" of the distribution |
| `percentiles` | Values at specific positions (p0.1, p1, p5, p10, p25, p50, p75, p90, p95, p99, p99.9) |
| `histogram` | Bin edges and counts for distribution visualization |

### Value Categorization
| Metric | Description |
|--------|-------------|
| `zero_count` | Number of exact zero values |
| `zero_percentage` | Percentage of exact zero values |
| `near_zero_count` | Number of values close to zero (below threshold) |
| `near_zero_percentage` | Percentage of near-zero values |
| `positive_count` | Number of positive values |
| `positive_percentage` | Percentage of positive values |
| `negative_count` | Number of negative values |
| `negative_percentage` | Percentage of negative values |

### Missing Value Analysis
| Metric | Description |
|--------|-------------|
| `total_rows` | Total number of rows in the dataset |
| `null_count` | Number of missing (NULL) values |
| `non_null_count` | Number of non-missing values |
| `null_percentage` | Percentage of missing values |
| `valid_count` | Number of valid numeric values after conversion |

### Outlier Detection (when enabled)
| Metric | Description |
|--------|-------------|
| `iqr` | Interquartile range (Q3 - Q1) |
| `lower_bound` | Lower threshold for outliers (Q1 - 1.5*IQR) |
| `upper_bound` | Upper threshold for outliers (Q3 + 1.5*IQR) |
| `count` | Number of detected outliers |
| `percentage` | Percentage of outliers in valid data |
| `sample` | Sample of outlier values (up to 10) |

### Normality Testing (when enabled)
| Metric | Description |
|--------|-------------|
| `shapiro` | Results of Shapiro-Wilk normality test (statistic, p-value, normal) |
| `anderson` | Results of Anderson-Darling normality test |
| `ks` | Results of Kolmogorov-Smirnov normality test |
| `is_normal` | Boolean indicating whether the data follows a normal distribution |
| `normal_test_count` | Number of normality tests performed |
| `normal_test_passed` | Number of normality tests that indicate normal distribution |

## Handling Large Datasets

For large datasets, the module:
1. Automatically detects dataset size and switches to chunk-based processing when needed
2. Processes data in chunks to maintain memory efficiency
3. Combines chunk results to produce accurate statistics for the entire dataset
4. Provides progress tracking with memory usage monitoring

## Integration Points

The module integrates with:

1. **Data Source System**: Gets data through the `DataSource` abstraction
2. **Operation Framework**: Follows the standard operation workflow
3. **I/O System**: Uses `io.py` for saving results and artifacts
4. **Visualization System**: Uses `visualization.py` for creating plots
5. **Reporting System**: Reports progress and results to the reporter object

## Conclusion

The `numeric.py` module provides comprehensive analysis of numeric fields with robust statistical calculations, visualization capabilities, and performance optimizations for large datasets. It follows a clean architecture that separates analytical logic from infrastructure concerns, making it both powerful and maintainable.

By delivering insights about distributions, outliers, and normality, it helps data scientists and analysts better understand the characteristics of numeric fields, which is essential for data preparation, quality assessment, and feature engineering tasks.