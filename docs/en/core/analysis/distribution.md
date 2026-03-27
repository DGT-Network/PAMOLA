# Distribution Visualization

**Module:** `pamola_core.analysis.distribution`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

The `distribution` module generates interactive distribution visualizations (histograms and bar charts) for pandas DataFrame fields. It automates chart generation with intelligent field-type routing, timestamped file naming, and integration with the PAMOLA visualization subsystem.

Numeric fields are visualized as histograms or binned bar charts; categorical fields as frequency bar charts. All output is generated in configurable formats (HTML, PNG, SVG, JPG) suitable for EDA reports, dashboards, and governance documentation.

## Key Features

- **Type-Aware Routing**: Automatically routes numeric vs. categorical fields to appropriate chart types
- **Configurable Binning**: Customize number of bins for numeric histograms
- **Multiple Formats**: Supports HTML (interactive), PNG, SVG, JPG (static)
- **Timestamped Output**: Auto-generates unique filenames to prevent overwrites
- **Batch Generation**: Process multiple fields in single call; returns path mapping
- **Safe Handling**: Gracefully skips empty or problematic fields with logging

## Core Function

### `visualize_distribution_df()`

Generate distribution visualizations for DataFrame fields.

**Signature:**
```python
def visualize_distribution_df(
    df: pd.DataFrame,
    viz_dir: Path,
    numeric_bar_charts: bool = False,
    n_bins: int = 10,
    field_names: Optional[List[str]] = None,
    viz_format: str = "html",
) -> Dict[str, Path]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | — | Input dataset |
| `viz_dir` | Path | — | Output directory for visualizations (created if doesn't exist) |
| `numeric_bar_charts` | bool | False | If True, visualize numerics as binned bar charts; else histograms |
| `n_bins` | int | 10 | Number of bins for numeric visualizations |
| `field_names` | List[str] \| None | All columns | Specific fields to visualize; if None, analyzes all |
| `viz_format` | str | "html" | Output format: "html", "png", "svg", "jpg" |

**Returns:**
```python
{
    "field_name_1": Path("/path/to/analysis_histogram_field_name_1_20260323_120530.html"),
    "field_name_2": Path("/path/to/analysis_bar_chart_field_name_2_20260323_120530.html"),
    ...
}
```

Maps field name to output file Path. Skipped fields (empty, unsupported dtype) are omitted from result.

**Raises/Logs:**

| Message | Condition |
|---------|-----------|
| Warning: "Skipping empty field: {field}" | Field has only NaN values |
| Warning: "Skipping unsupported dtype for field: {field}" | Unsupported data type (rare) |
| Error: "Failed to create visualization for {field}: {msg}" | Chart generation failed |

### Examples

#### Basic Usage
```python
import pandas as pd
from pathlib import Path
from pamola_core.analysis import visualize_distribution_df

df = pd.read_csv("sales_data.csv")

paths = visualize_distribution_df(
    df,
    viz_dir=Path("output/distributions"),
    n_bins=15,
    viz_format="html"
)

# Result:
# {
#     'age': Path('output/distributions/analysis_histogram_age_20260323_120530.html'),
#     'salary': Path('output/distributions/analysis_histogram_salary_20260323_120530.html'),
#     'department': Path('output/distributions/analysis_bar_chart_department_20260323_120530.html'),
#     ...
# }

print(f"Generated {len(paths)} visualizations")
for field, path in paths.items():
    print(f"  {field}: {path.name}")
```

#### Selective Fields
```python
# Visualize only numeric columns
paths = visualize_distribution_df(
    df,
    viz_dir=Path("eda/numeric"),
    field_names=['age', 'salary', 'years_experience']
)
```

#### Numeric as Bar Charts
```python
# For numeric fields, use binned bar charts instead of histograms
paths = visualize_distribution_df(
    df,
    viz_dir=Path("eda/distributions"),
    numeric_bar_charts=True,
    n_bins=20,
    viz_format="png"
)
# Numeric fields now rendered as bar charts with labeled bins
```

#### Static Format (PNG)
```python
# Generate static PNG images for reports
paths = visualize_distribution_df(
    df,
    viz_dir=Path("report/charts"),
    n_bins=12,
    field_names=['age', 'income', 'region'],
    viz_format="png"
)
```

## Visualization Types

### Numeric Fields

#### Histogram (default, `numeric_bar_charts=False`)
- X-axis: Continuous value ranges binned by n_bins
- Y-axis: Frequency (count)
- Display: Bars showing distribution shape
- Use: Detect skewness, multimodality, outliers

**Example:**
```
Histogram: age
Frequency
10 |  ███
   |  ███ ██
   |  ███ ██ █
 5 |  ███ ██ █
   |  ███ ██ █
   |__________________
   18  30  40  50  60
```

#### Binned Bar Chart (`numeric_bar_charts=True`)
- X-axis: Bin ranges with formatted labels (e.g., "18.50–25.50" for floats, "18–25" for ints)
- Y-axis: Frequency
- Display: Discrete bars per bin
- Use: Clearer bin boundaries, easier reading

**Label Formatting:**
| Data Type | Bin Width | Label Format | Example |
|-----------|-----------|--------------|---------|
| Float | Any | `{left:.2f}–{right:.2f}` | "18.50–25.50" |
| Integer | 1.0 | `{left}` (single value) | "18" |
| Integer | >1.0 | `{left}–{right}` | "18–24" |

### Categorical Fields

#### Bar Chart
- X-axis: Category values
- Y-axis: Frequency (normalized to [0,1] or count)
- Display: Bars ordered by frequency
- Use: Compare category popularity, identify dominant values

**Example:**
```
Bar Chart: department
Frequency
0.30 | ███
     | ███ ██
     | ███ ██ █
0.15 | ███ ██ █
     | _____________
       Sales  IT  HR  Ops
```

## Configuration Parameters

### Bins (`n_bins`)

| Value | Use Case |
|-------|----------|
| 5–10 | Large datasets, smooth distributions |
| 10–20 | Standard EDA, balanced detail/readability |
| 20–50 | Small datasets, fine granularity |
| >50 | Special cases (e.g., count data with many unique values) |

**Default:** 10 (suitable for most cases)

### Format (`viz_format`)

| Format | Pros | Cons | Use Case |
|--------|------|------|----------|
| html | Interactive, zoomable, small file | Requires viewer | Web dashboards, reports |
| png | Static, universal, clear | Larger files, no zoom | PDFs, presentations |
| svg | Scalable, editable, small | Less browser support | Technical docs, posters |
| jpg | Compact, compatible | Lossy, less sharp | Email, web thumbnails |

**Default:** "html" (interactive, best for exploration)

## Output File Naming

Files are timestamped to prevent overwrites:

```
analysis_{chart_type}_{field_name}_{timestamp}.{format}
       ↑                      ↑          ↑
    fixed prefix          field name   YYYYMMDD_HHMMSS

Example: analysis_histogram_salary_20260323_120530.html
```

## Best Practices

1. **Create Output Directory First**: Ensure `viz_dir` exists and is writable before calling function.
   ```python
   viz_dir = Path("output/distributions")
   viz_dir.mkdir(parents=True, exist_ok=True)
   visualize_distribution_df(df, viz_dir)
   ```

2. **Handle Empty Fields**: Function logs warnings and skips empty fields. Check returned `paths` dict size vs. expected fields.
   ```python
   paths = visualize_distribution_df(df, viz_dir, field_names=['age', 'salary'])
   if len(paths) < 2:
       print("WARNING: Some fields were skipped (likely empty)")
   ```

3. **Choose Format by Use Case**:
   - **HTML**: Best for exploration, dashboards, interactive reports
   - **PNG**: Best for static reports, PDFs, presentations
   - **SVG**: Best for publication, technical documents
   - **JPG**: Best for web/email distribution

4. **Adjust Bins for Data Distribution**:
   ```python
   # For sparse data or outliers
   paths = visualize_distribution_df(df, viz_dir, n_bins=5)

   # For fine detail
   paths = visualize_distribution_df(df, viz_dir, n_bins=30)
   ```

5. **Combine with Descriptive Stats**:
   ```python
   stats = analyze_descriptive_stats(df)
   paths = visualize_distribution_df(df, viz_dir)
   # Use stats to validate chart ranges and identify anomalies
   ```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty output dict | All fields empty or skipped | Check DataFrame for NaN values; pre-filter |
| "viz_dir not found" | Directory doesn't exist | Create: `viz_dir.mkdir(parents=True, exist_ok=True)` |
| Files not created | Permission denied | Verify viz_dir is writable; check OS permissions |
| Chart looks wrong | Wrong `numeric_bar_charts` setting | Try opposite setting; or adjust n_bins |
| Missing field in output | Field had only NaN values | Check input data; pre-impute if needed |
| Large file sizes | HTML format with large dataset | Use PNG or reduce dataset size |

## Related Components

- [`analyze_descriptive_stats()`](./descriptive_stats.md) - Numeric statistics backing visualizations
- [`analyze_dataset_summary()`](./dataset_summary.md) - Field types and outliers
- [`analyze_correlation()`](./correlation.md) - Bivariate relationships
- [`utils.visualization`](../utils/) - Underlying chart generation APIs

## Implementation Details

### Type Routing

1. Check if field is numeric (`pd.api.types.is_numeric_dtype()`)
   - Yes → Histogram or binned bar chart (per `numeric_bar_charts`)
   - No → Check categorical

2. Check if field is categorical (`pd.api.types.is_categorical_dtype()`)
   - Yes → Bar chart
   - No → Log warning, skip

### Memory Efficiency

- Drops NaN values before visualization (reduces memory)
- Generates one chart at a time (doesn't hold all in memory)
- Output file size depends on format (HTML typically 50KB–200KB per chart)

## Summary Analysis

**Purpose**: Generate distribution visualizations for all fields in a DataFrame for EDA and reporting.

**Inputs**: DataFrame, output directory, field names, bins, format.

**Outputs**: Interactive/static charts (HTML/PNG/SVG/JPG) + path mapping.

**Strengths**:
- Automatic type detection and routing
- Batch processing with timestamped files
- Multiple output formats
- Graceful handling of edge cases

**Limitations**:
- No support for time series or hierarchical data
- Histograms assume continuous data (not suitable for discrete counts)
- No faceting or multi-variable visualizations

**Typical Usage**: Generate distribution charts for EDA reports, exploratory analysis dashboards, and data quality documentation.
