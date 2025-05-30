# PAMOLA.CORE Visualization Utilities

## 1. Overview

The `visualization_utils.py` module provides a set of helper utilities for creating visualizations in anonymization operations, simplifying interactions with the pamola core visualization system. This module is designed to work closely with the broader PAMOLA.CORE anonymization framework to provide consistent, standardized visualizations for comparing original and anonymized data.

### 1.1 Purpose

The primary purpose of this module is to:

- Generate standardized filenames for visualization artifacts
- Prepare data for different types of visualizations based on data characteristics
- Handle sampling of large datasets for visualization
- Create optimal visualization parameters (like bin counts for histograms)
- Simplify the creation of visualization file paths

### 1.2 Key Features

- **Standardized Naming**: Consistent filename generation for all visualization types
- **Data Preparation**: Automatic preparation of data for different visualization types
- **Large Dataset Support**: Efficient sampling of large datasets for visualization
- **Optimal Parameter Calculation**: Automatic calculation of optimal visualization parameters
- **Path Management**: Simple path creation and management for visualization artifacts
- **Format Agnostic**: Support for both numeric and categorical data visualizations

## 2. Module Architecture

This module serves as a helper layer between anonymization operations and the pamola core visualization system:

```
┌─────────────────────┐
│ anonymization_op.py │
└──────────┬──────────┘
           │ calls
           ▼
┌─────────────────────────┐     uses     ┌─────────────────┐
│ visualization_utils.py   ├────────────►│ visualization.py │
└─────────────────────────┘              └─────────────────┘
```

The module consists of a set of standalone functions that handle different aspects of visualization preparation, without maintaining any state. This makes it highly composable and easy to use in different contexts.

## 3. Key Functions

### 3.1 `generate_visualization_filename`

```python
generate_visualization_filename(
    field_name: str,
    operation_name: str,
    visualization_type: str,
    timestamp: Optional[str] = None,
    extension: str = "png"
) -> str
```

Generates a standardized filename for a visualization.

**Parameters:**
- `field_name`: Name of the field being visualized
- `operation_name`: Name of the operation creating the visualization
- `visualization_type`: Type of visualization (e.g., "histogram", "distribution")
- `timestamp`: Timestamp for file naming. If None, current timestamp is used.
- `extension`: File extension (default: "png")

**Returns:**
- Standardized filename string

**Example:**
```python
filename = generate_visualization_filename("income", "binning", "histogram")
# Result: "income_binning_histogram_20250504_152345.png"
```

### 3.2 `register_visualization_artifact`

```python
register_visualization_artifact(
    result: Any,
    reporter: Any,
    path: Path,
    field_name: str,
    visualization_type: str,
    description: Optional[str] = None
) -> None
```

Registers a visualization artifact with the result and reporter.

**Parameters:**
- `result`: Operation result to add the artifact to
- `reporter`: Reporter to add the artifact to
- `path`: Path to the visualization file
- `field_name`: Name of the field being visualized
- `visualization_type`: Type of visualization
- `description`: Custom description of the visualization

**Example:**
```python
register_visualization_artifact(
    result, 
    reporter, 
    Path("/path/to/visualization.png"), 
    "income", 
    "histogram",
    "Income distribution comparison before and after anonymization"
)
```

### 3.3 `sample_large_dataset`

```python
sample_large_dataset(
    data: pd.Series,
    max_samples: int = 10000,
    random_state: int = 42
) -> pd.Series
```

Samples a large dataset to a manageable size for visualization.

**Parameters:**
- `data`: Original large dataset
- `max_samples`: Maximum number of samples to return
- `random_state`: Random seed for reproducibility

**Returns:**
- Sampled dataset as pd.Series

**Example:**
```python
original_data = pd.Series(range(1000000))
sampled_data = sample_large_dataset(original_data, max_samples=5000)
# Result: Series with 5000 randomly sampled values
```

### 3.4 `prepare_comparison_data`

```python
prepare_comparison_data(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    data_type: str = "auto",
    max_categories: int = 10
) -> Tuple[Dict[str, Any], str]
```

Prepares data for comparison visualizations based on data type.

**Parameters:**
- `original_data`: Original data
- `anonymized_data`: Anonymized data
- `data_type`: Force specific data type ('numeric', 'categorical') or 'auto' to detect
- `max_categories`: Maximum number of categories for categorical data

**Returns:**
- Tuple containing prepared data dictionary and detected data type

**Example:**
```python
original = pd.Series([1, 2, 3, 4, 5])
anonymized = pd.Series([1, 2, 3, 3, 4])
data, dtype = prepare_comparison_data(original, anonymized)
# Result: ({"Original": [1, 2, 3, 4, 5], "Anonymized": [1, 2, 3, 3, 4]}, "numeric")
```

### 3.5 `calculate_optimal_bins`

```python
calculate_optimal_bins(
    data: pd.Series,
    min_bins: int = 5,
    max_bins: int = 30
) -> int
```

Calculates optimal number of bins for histograms using square root rule.

**Parameters:**
- `data`: Data to calculate bins for
- `min_bins`: Minimum number of bins
- `max_bins`: Maximum number of bins

**Returns:**
- Optimal number of bins

**Example:**
```python
data = pd.Series(np.random.normal(size=10000))
optimal_bins = calculate_optimal_bins(data)
# Result: An integer value between 5 and 30 based on square root rule
```

### 3.6 `create_visualization_path`

```python
create_visualization_path(
    task_dir: Path,
    filename: str
) -> Path
```

Creates full path for visualization with directory creation if needed.

**Parameters:**
- `task_dir`: Base task directory
- `filename`: Filename for the visualization

**Returns:**
- Full path to the visualization file

**Example:**
```python
task_dir = Path("/path/to/task_directory")
path = create_visualization_path(task_dir, "income_histogram_20250504.png")
# Result: Path object pointing to "/path/to/task_directory/income_histogram_20250504.png"
```

## 4. Usage Patterns

### 4.1 Typical Usage Flow

A typical usage pattern in an anonymization operation would be:

1. Process the original and anonymized data
2. Prepare the data for visualization using `prepare_comparison_data`
3. Generate a standardized filename with `generate_visualization_filename`
4. Create the visualization path with `create_visualization_path`
5. Call the pamola core visualization function from `visualization.py`
6. Register the visualization artifact with `register_visualization_artifact`

Example flow:

```python
# In an anonymization operation
def _generate_visualizations(self, original_data, anonymized_data, task_dir):
    # Prepare data based on its type
    prepared_data, data_type = prepare_comparison_data(original_data, anonymized_data)
    
    # Generate standardized filename
    filename = generate_visualization_filename(
        self.field_name, self.__class__.__name__, "distribution"
    )
    
    # Create full path
    viz_path = create_visualization_path(task_dir, filename)
    
    # Create visualization based on data type
    if data_type == "numeric":
        bin_count = calculate_optimal_bins(original_data)
        create_histogram(prepared_data, str(viz_path), f"{self.field_name} Distribution", 
                        self.field_name, "Frequency", bins=bin_count)
    else:
        create_bar_plot(prepared_data, str(viz_path), f"{self.field_name} Categories", 
                       "Category", "Count")
    
    # Register the visualization
    register_visualization_artifact(self.result, self.reporter, viz_path, 
                                   self.field_name, "distribution")
    
    return {"distribution": viz_path}
```

### 4.2 Handling Large Datasets

For large datasets, use the sampling utility:

```python
# If dataset is large, sample it before visualization
if len(original_data) > 10000:
    original_data_sample = sample_large_dataset(original_data)
    anonymized_data_sample = sample_large_dataset(anonymized_data)
    prepared_data, data_type = prepare_comparison_data(
        original_data_sample, anonymized_data_sample
    )
else:
    prepared_data, data_type = prepare_comparison_data(original_data, anonymized_data)
```

## 5. Implementation Considerations

### 5.1 Performance

- The module is designed to efficiently handle large datasets through sampling
- For visualization preparation, data is processed efficiently with vectorized operations
- File paths are created with minimal overhead

### 5.2 Memory Usage

- Large datasets are sampled to reduce memory usage during visualization
- Data processing is done in-place wherever possible

### 5.3 Error Handling

- Functions use appropriate exception handling and logging
- Non-critical errors are logged and graceful fallbacks are provided
- Return sensible defaults when calculations fail

### 5.4 Dependencies

- The module depends on:
  - `pandas` and `numpy` for data manipulation
  - `datetime` for timestamp generation
  - `pathlib.Path` for path handling
  - Pamola Core logging utilities

## 6. Future Improvements

1. Add support for more specialized visualization types (time series, geospatial)
2. Implement adaptive sampling based on data characteristics
3. Add theme support for consistent visualization styling
4. Support for interactive visualization formats (HTML, SVG)
5. Add visualization metadata capture for better reporting

## 7. Integration Guidelines

When using this module in anonymization operations:

1. Always use standardized naming conventions
2. Handle numeric and categorical data appropriately
3. Use sampling for large datasets
4. Register visualizations with both result and reporter
5. Ensure visualization directories exist before saving files
6. Use appropriate visualization types for different data characteristics

## 8. Examples

### 8.1 Numeric Data Visualization

```python
# For numeric data
import pandas as pd
from pathlib import Path
from pamola_core.anonymization.commons.visualization_utils import *

# Sample data
original_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
anonymized_data = pd.Series([1, 3, 3, 5, 5, 7, 7, 9, 9, 10])

# Prepare data
prepared_data, data_type = prepare_comparison_data(original_data, anonymized_data)

# Create visualization
task_dir = Path("/path/to/task")
filename = generate_visualization_filename("sample_field", "numeric_gen", "histogram")
viz_path = create_visualization_path(task_dir, filename)

# Use create_histogram from pamola_core.utils.visualization
from pamola_core.utils.visualization import create_histogram
create_histogram(
    prepared_data,
    str(viz_path),
    "Sample Distribution Comparison",
    "Value",
    "Frequency",
    bins=10
)
```

### 8.2 Categorical Data Visualization

```python
# For categorical data
import pandas as pd
from pathlib import Path
from pamola_core.anonymization.commons.visualization_utils import *

# Sample data
original_data = pd.Series(["A", "B", "C", "A", "B", "D", "E", "F", "G", "H"])
anonymized_data = pd.Series(["Other", "Other", "C", "Other", "Other", "D", "Other", "Other", "Other", "H"])

# Prepare data
prepared_data, data_type = prepare_comparison_data(original_data, anonymized_data)

# Create visualization
task_dir = Path("/path/to/task")
filename = generate_visualization_filename("category_field", "cat_gen", "categories")
viz_path = create_visualization_path(task_dir, filename)

# Use create_bar_plot from pamola_core.utils.visualization
from pamola_core.utils.visualization import create_bar_plot
create_bar_plot(
    prepared_data,
    str(viz_path),
    "Category Comparison",
    "Category",
    "Count",
    orientation="v"
)
```

## 9. Conclusion

The `visualization_utils.py` module provides essential utilities for creating standardized visualizations in anonymization operations. By following consistent patterns and leveraging these utilities, operations can produce informative, comparable, and well-organized visualizations that help users understand the impact of anonymization techniques on their data.