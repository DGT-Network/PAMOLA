# PAMOLA.CORE Visualization System Documentation

## Overview

The PAMOLA.CORE Visualization System provides a unified interface for creating various types of data visualizations primarily using Plotly. This system is designed to support operations that analyze and profile data within the PAMOLA.CORE (Privacy-Preserving AI Data Processors) project, generating standardized PNG visualizations that help identify patterns, distributions, and relationships within the data.

The visualization system focuses on simplicity and consistency - operations pass in their data, specify an output path, and receive either the path to the saved PNG file or an error message. This streamlined workflow makes it easy to integrate visualizations into profiling and analysis pipelines without dealing with the complexity of visualization libraries directly.

## Pamola Core Philosophy

The visualization system is built around several key principles:

1. **Simplified API**: A clean interface focused on creating and saving PNG visualizations
2. **Consistent Workflow**: Every function accepts data, creates visualizations, saves to PNG, and returns the path
3. **Error Resilience**: Graceful handling of errors with clear error messages
4. **Plotly-First**: Primary focus on Plotly for high-quality visualizations
5. **Minimal Configuration**: Sensible defaults with optional customization when needed
6. **Path-Based Output**: Clear organization of visualization files by providing explicit output paths

This approach ensures that operations can easily generate visualizations without needing to understand the underlying visualization libraries.

## Architecture

The visualization system consists of a main API module (`visualization.py`) and a package of helper modules (`vis_helpers/`) that implement specific visualization types:

```
pamola_core/utils/
├── visualization.py             # Main API module
├── vis_helpers/                 # Helper package
│   ├── __init__.py              # Package initialization 
│   ├── base.py                  # Base classes and interfaces
│   ├── theme.py                 # Theme management
│   ├── bar_plots.py             # Bar chart implementations
│   ├── histograms.py            # Histogram implementations
│   ├── scatter_plots.py         # Scatter plot implementations
│   └── ... (other visualization types)
```

The system follows a consistent pattern:
- Operations call functions in the main API
- The API delegates to appropriate helpers for implementation
- Visualizations are saved as PNG files
- The path to the saved file is returned for further use

## Dependencies

The visualization system relies on the following libraries:

### Pamola Core Dependencies
- `plotly`: Primary visualization library for creating visualizations
- `pandas`: For data handling and manipulation
- `numpy`: For numerical operations
- `pathlib` (Path): For handling file paths consistently
- `kaleido`: Required for exporting Plotly figures to PNG format

### Optional Dependencies
- `matplotlib`: Used as a fallback for some visualization types
- `wordcloud` and `PIL`: Required only if using word cloud visualizations

Note that all dependencies are handled internally by the visualization system, so operations using the API don't need to import these libraries directly.

## API Reference

### Basic Visualization Functions

#### `create_bar_plot`

Creates a bar plot visualization and saves it as PNG.

```python
def create_bar_plot(
    data: Union[Dict[str, Any], pd.Series],
    output_path: Union[str, Path],
    title: str,
    orientation: str = "v",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    sort_by: str = "value",
    max_items: int = 15,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Data to visualize (dictionary or pandas Series)
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `orientation`: "v" for vertical bars, "h" for horizontal bars
- `x_label`: Label for x-axis (optional)
- `y_label`: Label for y-axis (optional)
- `sort_by`: How to sort data ("value" or "key")
- `max_items`: Maximum number of items to show
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import create_bar_plot

# Create a bar plot from a dictionary
categories = {"Category A": 42, "Category B": 18, "Category C": 34, "Category D": 55}
result = create_bar_plot(
    data=categories,
    output_path="output/categories.png",
    title="Sample Categories",
    x_label="Category",
    y_label="Count",
    orientation="v"
)
print(f"Visualization saved to: {result}")
```

#### `create_histogram`

Creates a histogram visualization and saves it as PNG.

```python
def create_histogram(
    data: Union[Dict[str, int], pd.Series, np.ndarray, List[float]],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = "Count",
    bins: int = 20,
    kde: bool = True,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Data to visualize (various formats supported)
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `x_label`: Label for x-axis (optional)
- `y_label`: Label for y-axis (optional)
- `bins`: Number of bins for the histogram
- `kde`: Whether to show kernel density estimate
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import create_histogram
import numpy as np

# Create a histogram with random data
data = np.random.normal(0, 1, 1000)  # 1000 points from standard normal distribution
result = create_histogram(
    data=data,
    output_path="output/normal_distribution.png",
    title="Normal Distribution",
    x_label="Value",
    y_label="Frequency",
    bins=30,
    kde=True
)
print(f"Visualization saved to: {result}")
```

#### `create_scatter_plot`

Creates a scatter plot visualization and saves it as PNG.

```python
def create_scatter_plot(
    x_data: Union[List[float], np.ndarray, pd.Series],
    y_data: Union[List[float], np.ndarray, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    add_trendline: bool = False,
    correlation: Optional[float] = None,
    method: Optional[str] = None,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `x_data`: Data for the x-axis
- `y_data`: Data for the y-axis
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `x_label`: Label for x-axis (optional)
- `y_label`: Label for y-axis (optional)
- `add_trendline`: Whether to add a trendline
- `correlation`: Correlation coefficient to display (optional)
- `method`: Correlation method used (e.g., "pearson")
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import create_scatter_plot
import numpy as np

# Create a scatter plot with correlated data
x = np.random.normal(0, 1, 100)
y = x * 2 + np.random.normal(0, 0.5, 100)  # y correlated with x plus noise

# Calculate correlation
correlation = np.corrcoef(x, y)[0, 1]

result = create_scatter_plot(
    x_data=x,
    y_data=y,
    output_path="output/correlation_example.png",
    title="Correlation Example",
    x_label="X Variable",
    y_label="Y Variable",
    add_trendline=True,
    correlation=correlation,
    method="Pearson"
)
print(f"Visualization saved to: {result}")
```

#### `create_boxplot`

Creates a box plot visualization and saves it as PNG.

```python
def create_boxplot(
    data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = "Category",
    y_label: Optional[str] = "Value",
    orientation: str = "v",
    show_points: bool = True,
    notched: bool = False,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Data to visualize
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `x_label`: Label for the x-axis (categorical axis for vertical orientation)
- `y_label`: Label for the y-axis (value axis for vertical orientation)
- `orientation`: Orientation of the boxes: "v" for vertical, "h" for horizontal
- `show_points`: Whether to show outlier points
- `notched`: Whether to show notched boxes
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import create_boxplot

# Creating a boxplot of salary by department
salary_data = {
    "IT": [75000, 85000, 92000, 105000, 120000],
    "Sales": [65000, 72000, 80000, 95000, 110000],
    "Marketing": [60000, 68000, 75000, 82000, 95000]
}
result = create_boxplot(
    data=salary_data,
    output_path="output/salary_by_department.png",
    title="Salary Distribution by Department",
    x_label="Department",
    y_label="Salary (USD)",
    show_points=True
)
print(f"Visualization saved to: {result}")
```

#### `create_heatmap`

Creates a heatmap visualization and saves it as PNG.

```python
def create_heatmap(
    data: Union[Dict[str, Dict[str, float]], pd.DataFrame, np.ndarray],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    colorscale: Optional[str] = None,
    annotate: bool = True,
    annotation_format: str = ".2f",
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Data to visualize
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `x_label`: Label for the x-axis
- `y_label`: Label for the y-axis
- `colorscale`: Colorscale to use (default from theme if None)
- `annotate`: Whether to annotate the heatmap with values
- `annotation_format`: Format string for annotations (e.g., ".2f" for 2 decimal places)
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
import pandas as pd
from pamola_core.utils.visualization import create_heatmap

# Creating a heatmap of values
data = pd.DataFrame([
    [10, 15, 20, 25],
    [15, 25, 30, 35],
    [20, 30, 40, 45],
    [25, 35, 45, 55]
], index=["A", "B", "C", "D"], columns=["W", "X", "Y", "Z"])
result = create_heatmap(
    data=data,
    output_path="output/value_heatmap.png",
    title="Value Heatmap",
    x_label="Columns",
    y_label="Rows",
    colorscale="Blues",
    annotate=True
)
print(f"Visualization saved to: {result}")
```

#### `create_line_plot`

Creates a line plot visualization and saves it as PNG.

```python
def create_line_plot(
    data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    add_markers: bool = True,
    add_area: bool = False,
    smooth: bool = False,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Data to visualize
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `x_data`: Data for the x-axis. If None, indices are used
- `x_label`: Label for the x-axis
- `y_label`: Label for the y-axis
- `add_markers`: Whether to add markers at data points
- `add_area`: Whether to fill area under lines
- `smooth`: Whether to use spline interpolation for smoother lines
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import create_line_plot

# Creating a line plot of trends over time
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
data = {
    "Product A": [100, 120, 130, 125, 150, 170],
    "Product B": [80, 85, 90, 100, 110, 115]
}
result = create_line_plot(
    data=data,
    output_path="output/product_trends.png",
    title="Product Sales Trends",
    x_data=months,
    x_label="Month",
    y_label="Sales",
    add_markers=True
)
print(f"Visualization saved to: {result}")
```

#### `create_correlation_matrix`

Creates a correlation matrix visualization and saves it as PNG.

```python
def create_correlation_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    colorscale: Optional[str] = None,
    annotate: bool = True,
    annotation_format: str = ".2f",
    mask_upper: bool = False,
    mask_diagonal: bool = False,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Correlation matrix data
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `x_label`: Label for the x-axis
- `y_label`: Label for the y-axis
- `colorscale`: Colorscale to use (default from theme if None)
- `annotate`: Whether to annotate the matrix with correlation values
- `annotation_format`: Format string for annotations (e.g., ".2f" for 2 decimal places)
- `mask_upper`: Whether to mask the upper triangle (above diagonal)
- `mask_diagonal`: Whether to mask the diagonal
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
import pandas as pd
from pamola_core.utils.visualization import create_correlation_matrix

# Creating a correlation matrix
df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [5, 4, 3, 2, 1],
    "C": [1, 3, 5, 7, 9],
    "D": [9, 7, 5, 3, 1]
})
corr_matrix = df.corr()
result = create_correlation_matrix(
    data=corr_matrix,
    output_path="output/correlation_matrix.png",
    title="Feature Correlation Matrix",
    colorscale="RdBu_r",
    mask_upper=True
)
print(f"Visualization saved to: {result}")
```

#### `create_correlation_pair`

Creates a correlation plot for a pair of variables and saves it as PNG.

```python
def create_correlation_pair(
    x_data: Union[List[float], np.ndarray, pd.Series],
    y_data: Union[List[float], np.ndarray, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    correlation: Optional[float] = None,
    method: Optional[str] = "Pearson",
    add_trendline: bool = True,
    add_histogram: bool = True,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `x_data`: Data for the x-axis
- `y_data`: Data for the y-axis
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `x_label`: Label for the x-axis
- `y_label`: Label for the y-axis
- `correlation`: Correlation coefficient to display on the plot
- `method`: Correlation method name (for annotation)
- `add_trendline`: Whether to add a trendline
- `add_histogram`: Whether to add histograms for both variables
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
import numpy as np
from pamola_core.utils.visualization import create_correlation_pair

# Creating a correlation pair plot
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
correlation = np.corrcoef(x, y)[0, 1]
result = create_correlation_pair(
    x_data=x,
    y_data=y,
    output_path="output/xy_correlation.png",
    title="X vs Y Correlation",
    x_label="X Values",
    y_label="Y Values",
    correlation=correlation,
    method="Pearson"
)
print(f"Visualization saved to: {result}")
```

### Specialized Profiling Visualization Functions

#### `plot_completeness`

Visualizes data completeness for each column in a DataFrame and saves as PNG.

```python
def plot_completeness(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Completeness Analysis",
    min_percent: float = 0,
    max_fields: int = 50,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `df`: DataFrame to analyze
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `min_percent`: Minimum completeness percentage to include
- `max_fields`: Maximum number of fields to show
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
import pandas as pd
from pamola_core.utils.visualization import plot_completeness

# Create a DataFrame with some missing values
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, None, 30, None, 50],
    'C': [None, None, 'x', 'y', 'z'],
    'D': [True, False, True, False, True]
})

# Plot completeness
result = plot_completeness(
    df=df,
    output_path="output/data_completeness.png",
    title="Data Completeness",
    min_percent=0,  # Show all fields
    max_fields=10
)
print(f"Visualization saved to: {result}")
```

#### `plot_value_distribution`

Visualizes the distribution of values and saves as PNG.

```python
def plot_value_distribution(
    data: Dict[str, int],
    output_path: Union[str, Path],
    title: str,
    max_items: int = 15,
    sort_by: str = "value",
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Dictionary with values and their counts
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `max_items`: Maximum number of items to show
- `sort_by`: How to sort data ("value" or "key")
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import plot_value_distribution

# Create a dictionary of values and their frequencies
education_levels = {
    "High School": 1200,
    "Bachelor's": 2500,
    "Master's": 1800,
    "PhD": 450,
    "Associate's": 950,
    "Other": 350
}

# Plot distribution
result = plot_value_distribution(
    data=education_levels,
    output_path="output/education_levels.png",
    title="Education Level Distribution",
    max_items=10,
    sort_by="value"  # Sort by frequency
)
print(f"Visualization saved to: {result}")
```

#### `plot_numeric_distribution`

Visualizes the distribution of numeric values and saves as PNG.

```python
def plot_numeric_distribution(
    data: Union[Dict[str, int], pd.Series, np.ndarray, List[float]],
    output_path: Union[str, Path],
    title: str,
    bins: int = 20,
    kde: bool = True,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `data`: Data to visualize
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `bins`: Number of bins for the histogram
- `kde`: Whether to show a kernel density estimate
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import plot_numeric_distribution

# Creating a numeric distribution plot
ages = [25, 28, 30, 32, 35, 35, 36, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60]
result = plot_numeric_distribution(
    data=ages,
    output_path="output/age_distribution.png",
    title="Age Distribution",
    bins=10
)
print(f"Visualization saved to: {result}")
```

#### `plot_date_distribution`

Visualizes the distribution of dates by year and saves as PNG.

```python
def plot_date_distribution(
    date_stats: Dict[str, Any],
    output_path: Union[str, Path],
    title: str,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `date_stats`: Dictionary with date statistics including 'year_distribution'
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import plot_date_distribution

# Create a dictionary with year distribution data
date_stats = {
    'year_distribution': {
        '2010': 120,
        '2011': 150,
        '2012': 180,
        '2013': 210,
        '2014': 250,
        '2015': 220,
        '2016': 240,
        '2017': 260,
        '2018': 290,
        '2019': 320,
        '2020': 280
    }
}

# Plot distribution
result = plot_date_distribution(
    date_stats=date_stats,
    output_path="output/year_distribution.png",
    title="Events by Year"
)
print(f"Visualization saved to: {result}")
```

#### `plot_email_domains`

Visualizes the distribution of email domains and saves as PNG.

```python
def plot_email_domains(
    domains: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "Email Domain Distribution",
    max_domains: int = 15,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `domains`: Dictionary with domain names and their counts
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `max_domains`: Maximum number of domains to show
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import plot_email_domains

# Create domain distribution data
domains = {
    "gmail.com": 350,
    "yahoo.com": 180,
    "hotmail.com": 120,
    "outlook.com": 110,
    "company.com": 80
}
result = plot_email_domains(
    domains=domains,
    output_path="output/email_domains.png",
    title="Email Domain Distribution",
    max_domains=10
)
print(f"Visualization saved to: {result}")
```

#### `plot_phone_distribution`

Visualizes the distribution of phone components and saves as PNG.

```python
def plot_phone_distribution(
    phone_data: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "Phone Code Distribution",
    field_name: str = "Country Code",
    max_items: int = 15,
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `phone_data`: Dictionary with codes and their counts
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `field_name`: The name of the field being visualized (for axis labels)
- `max_items`: Maximum number of items to show
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import plot_phone_distribution

# Create phone country code distribution data
country_codes = {
    "+1": 450,
    "+44": 180,
    "+49": 120,
    "+33": 90,
    "+81": 60
}
result = plot_phone_distribution(
    phone_data=country_codes,
    output_path="output/country_codes.png",
    title="Phone Country Code Distribution",
    field_name="Country Code"
)
print(f"Visualization saved to: {result}")
```

#### `plot_text_length_distribution`

Visualizes the distribution of text lengths and saves as PNG.

```python
def plot_text_length_distribution(
    length_data: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "Text Length Distribution",
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `length_data`: Dictionary with length ranges and their counts
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import plot_text_length_distribution

# Create text length distribution data
length_ranges = {
    "0-50": 120,
    "51-100": 180,
    "101-200": 150,
    "201-500": 80,
    "501+": 30
}
result = plot_text_length_distribution(
    length_data=length_ranges,
    output_path="output/text_length_distribution.png",
    title="Description Length Distribution"
)
print(f"Visualization saved to: {result}")
```

#### `plot_group_variation_distribution`

Visualizes the distribution of group variation values and saves as PNG.

```python
def plot_group_variation_distribution(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Group Variation Distribution",
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `results`: Dictionary with group variation results
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the plot
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import plot_group_variation_distribution

# Create group variation distribution data
variation_results = {
    "variation_distribution": {
        "0-0.1": 30,
        "0.1-0.2": 45,
        "0.2-0.3": 60,
        "0.3-0.4": 40,
        "0.4-0.5": 25,
        "0.5+": 10
    }
}
result = plot_group_variation_distribution(
    results=variation_results,
    output_path="output/group_variation.png",
    title="Group Variation Distribution"
)
print(f"Visualization saved to: {result}")
```

#### `plot_multiple_fields`

Creates a comparison plot for multiple fields and saves as PNG.

```python
def plot_multiple_fields(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    fields: List[str],
    plot_type: str = "bar",
    title: str = "Field Comparison",
    theme: Optional[str] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `df`: DataFrame containing the data
- `output_path`: Path where the PNG file should be saved
- `fields`: List of fields to compare
- `plot_type`: Type of plot: "bar", "line", or "boxplot"
- `title`: Title for the plot
- `theme`: Theme name to use (optional)
- `**kwargs`: Additional parameters for the underlying plotting function

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
import pandas as pd
from pamola_core.utils.visualization import plot_multiple_fields

# Create a DataFrame with multiple fields
df = pd.DataFrame({
    "Field1": [10, 15, 20, 25, 30],
    "Field2": [15, 20, 25, 30, 35],
    "Field3": [5, 10, 15, 20, 25]
})
result = plot_multiple_fields(
    df=df,
    output_path="output/field_comparison.png",
    fields=["Field1", "Field2", "Field3"],
    plot_type="bar",
    title="Comparison of Fields"
)
print(f"Visualization saved to: {result}")
```

## Additional Visualization Types

In addition to the visualizations documented above, the system also supports word cloud visualizations, though this functionality requires additional dependencies:

### Word Cloud Visualization

```python
def create_wordcloud(
    text_data: Union[str, List[str], Dict[str, float]],
    output_path: Union[str, Path],
    title: str,
    max_words: int = 200,
    background_color: str = "white",
    colormap: Optional[str] = None,
    exclude_words: Optional[List[str]] = None,
    **kwargs
) -> str:
```

**Parameters:**
- `text_data`: Text to visualize (raw text string, list of documents, or word-frequency dictionary)
- `output_path`: Path where the PNG file should be saved
- `title`: Title for the visualization
- `max_words`: Maximum number of words to include in the word cloud
- `background_color`: Background color for the word cloud
- `colormap`: Matplotlib colormap name for word coloring
- `exclude_words`: List of words to exclude from the word cloud
- `**kwargs`: Additional parameters for the underlying wordcloud implementation

**Returns:**
- Path to the saved PNG file or error message

**Example:**

```python
from pamola_core.utils.visualization import create_wordcloud

# Create a word cloud from text
text = """
Data visualization is the graphic representation of data. 
It involves producing images that communicate relationships 
among the represented data to viewers. Visualizing data is 
an important step in data analysis and helps in understanding
patterns, trends, and outliers in data.
"""

result = create_wordcloud(
    text_data=text,
    output_path="output/data_viz_cloud.png",
    title="Data Visualization Concepts",
    max_words=100,
    exclude_words=["is", "the", "and", "to", "in"]
)
print(f"Word cloud saved to: {result}")
```

**Note:** This function requires the optional `wordcloud` and `PIL` packages to be installed.

## Theme Management

The visualization system supports theming to maintain visual consistency across all outputs. Themes can be specified via the `theme` parameter in most visualization functions.

### Available Themes

The visualization system comes with several built-in themes:

1. **default**: Clean professional style with blue accent colors
2. **dark**: Dark background with vibrant colors for better contrast
3. **pastel**: Soft pastel colors for less intense visualizations
4. **professional**: Business-oriented theme with corporate color scheme

When a theme is specified, all visual elements including colors, font sizes, margins, and other styling options are adjusted accordingly to maintain a consistent appearance.

## Error Handling

All visualization functions return the path to the saved PNG file if successful, or an error message string if an error occurs. This allows operations to handle errors gracefully without crashing.

Example error handling:

```python
result = create_bar_plot(data, "output/plot.png", "My Plot")
if result.startswith("Error"):
    # Handle error
    print(f"Visualization failed: {result}")
else:
    # Success
    print(f"Visualization saved to: {result}")
```

The system handles many common error scenarios:
- Empty data sets
- Data with NaN or missing values
- Type mismatches
- Directory access issues
- Rendering problems
```

## Integration with Profiling Operations

The visualization system is designed to integrate seamlessly with profiling operations:

```python
import pandas as pd
from pamola_core.utils.visualization import plot_completeness, plot_value_distribution

# Load data
df = pd.read_csv("resume_data.csv")

# Analyze completeness
completeness_result = plot_completeness(
    df=df,
    output_path="output/profiling/completeness.png",
    title="Resume Data Completeness",
    min_percent=50  # Only show fields with >50% completeness
)

# Analyze categorical field
education_counts = df['education_level'].value_counts().to_dict()
education_result = plot_value_distribution(
    data=education_counts,
    output_path="output/profiling/education_levels.png",
    title="Education Level Distribution",
    max_items=10
)

# Return paths for integration with reporting
return {
    'completeness_plot': completeness_result,
    'education_plot': education_result
}
```

## Best Practices

1. **Always provide meaningful titles** for visualizations to make them self-explanatory
2. **Use specific plot types** appropriate for your data (e.g., bar plots for categories, histograms for distributions)
3. **Provide axis labels** to clarify what the data represents
4. **Limit the number of items** shown in visualizations to maintain readability
5. **Structure output paths** logically to organize visualization outputs
6. **Handle error results** properly in operations that use the visualization system
7. **Ensure output directories exist** before calling visualization functions, or handle path errors

## Performance Considerations

For optimal performance with the visualization system:

1. **Handle large datasets appropriately**:
   - For scatter plots with large datasets, consider sampling or aggregation
   - For bar plots, use the `max_items` parameter to show only the most relevant categories

2. **Be mindful of file paths**:
   - Use consistent naming conventions
   - Organize by profile type or analysis category

3. **Error handling**:
   - Always check return values for potential errors
   - Log any visualization failures appropriately
   - Provide fallback options when critical visualizations fail

## Conclusion

The PAMOLA.CORE visualization system provides a streamlined interface for creating visualizations within the project. With its focus on simplicity and consistency, operations can easily generate standardized PNG visualizations without dealing with the complexities of visualization libraries directly.

The system's consistent pattern - accept data, create visualization, save to PNG, return path - makes it easy to integrate into various analysis and profiling workflows. By following the guidelines and examples in this documentation, you can effectively use the system to create informative visualizations for your data.