# Correlation Analysis

**Module:** `pamola_core.analysis.correlation`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

The `correlation` module performs correlation analysis on pandas DataFrames with support for multiple methods (Pearson, Spearman, Kendall) and optional visualization generation. All results are normalized to DataFrames for consistent downstream processing.

The module handles automatic categorical-to-numeric mapping (e.g., "yes"/"no" → 1/0), filters zero-variance columns, and generates correlation matrices, heatmaps, or detailed single-variable analyses. Output includes both normalized DataFrame results and backward-compatible raw results.

## Key Features

- **Multiple Methods**: Pearson (linear), Spearman (ordinal), Kendall (rank)
- **Automatic Type Conversion**: Maps binary categorical (yes/no, true/false) to numeric
- **Zero-Variance Filtering**: Automatically removes constant columns
- **Flexible Analysis Types**: All variables, single variable, pairwise, or selected variables
- **Optional Visualization**: Generate correlation matrices, heatmaps with automatic scaling
- **Normalized DataFrame Output**: Consistent API across all result types
- **Backward Compatibility**: Includes raw results for legacy code

## Architecture

```
CorrelationAnalyzer (class)
├── _validate_method(method)
├── _validate_viz_format(format)
├── _validate_output_chart(output_chart)
├── _validate_columns(df, columns)
├── _map_binary_to_numeric(series) → Series
├── _prepare_data(df, columns) → DataFrame
├── _calculate_correlation_result(clean_data, columns, method) → (DataFrame, type_str)
├── _generate_charts(result_df, result_type, method, chart_types, analysis_dir, viz_format) → Path|List[Path]|None
└── analyze_correlation(df, columns, method, plot, output_chart, analysis_dir, viz_format) → Dict

analyze_correlation() (convenience function wrapper)
```

## Core Classes/Methods

### `CorrelationAnalyzer`

**Constructor:**
```python
analyzer = CorrelationAnalyzer()
```

No parameters required; stateless initialization.

### `analyze_correlation()`

Main analysis function with optional chart generation.

**Signature:**
```python
def analyze_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    plot: bool = True,
    output_chart: Union[str, List[str]] = "heatmap",
    analysis_dir: str = "",
    viz_format: str = "html",
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | — | Input dataset |
| `columns` | List[str] \| None | None | Specific columns to analyze; None = all numeric/boolean columns |
| `method` | str | "pearson" | Correlation method: "pearson", "spearman", "kendall" |
| `plot` | bool | True | If True, generate charts based on output_chart |
| `output_chart` | str \| List[str] | "heatmap" | Chart type(s): "matrix" (annotated), "heatmap" (color), or list of both |
| `analysis_dir` | str | "" | Directory to save chart files; required if plot=True |
| `viz_format` | str | "html" | Output format: "html", "png", "jpg", "svg" |

**Returns:**
```python
{
    "result": pd.DataFrame,           # Normalized correlation result (always DataFrame)
    "result_type": str,               # Analysis type: "all_variables", "single_variable", "pairwise", "selected_variables"
    "raw_result": Series|float|DataFrame,  # Backward-compatible original result type
    "path": Path|List[Path]|None      # Chart file path(s) or None if skipped
}
```

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValidationError` | Invalid method, format, or no suitable numeric columns |
| `ColumnNotFoundError` | Requested column not found in DataFrame |
| `DataError` | DataFrame is empty or all columns are constant |
| `TypeValidationError` | output_chart has wrong type |

### Result Types

| Result Type | Input | Output Shape | Use Case |
|-------------|-------|--------------|----------|
| `all_variables` | `columns=None` | N×N matrix | Full correlation across all fields |
| `single_variable` | `columns=['field']` | N×1 vertical | Correlation of one field with all others |
| `pairwise` | `columns=['field1', 'field2']` | 2×2 matrix | Detailed 2-variable correlation |
| `selected_variables` | `columns=['f1', 'f2', 'f3', ...]` | M×M matrix | Subset correlation analysis |

## Usage Examples

### All-Variables Correlation
```python
import pandas as pd
from pamola_core.analysis import analyze_correlation
from pathlib import Path

df = pd.read_csv("sales_data.csv")

result = analyze_correlation(
    df,
    method="pearson",
    plot=True,
    output_chart="heatmap",
    analysis_dir="output/correlation",
    viz_format="html"
)

print(f"Correlation matrix shape: {result['result'].shape}")
print(f"Analysis type: {result['result_type']}")
print(f"Chart saved to: {result['path']}")
```

### Single-Variable Analysis
```python
# Correlate 'salary' with all other numeric fields
result = analyze_correlation(
    df,
    columns=['salary'],
    method="pearson",
    plot=False  # Skip visualization
)

# result['result'] is N×1 DataFrame showing salary correlations
print(result['result'])
#          salary_correlation
# age                    0.65
# experience             0.72
# years_in_company       0.58
```

### Pairwise Analysis
```python
# Analyze correlation between exactly 2 fields
result = analyze_correlation(
    df,
    columns=['salary', 'experience'],
    method="spearman"
)

# result['result'] is 2×2 matrix
# result['raw_result'] is scalar correlation value (backward compat)
print(f"Correlation: {result['raw_result']:.3f}")
```

### Selected Variables (Multiple)
```python
# Correlate specific subset of fields
result = analyze_correlation(
    df,
    columns=['salary', 'experience', 'age', 'education_years'],
    method="kendall",
    plot=True,
    output_chart=["matrix", "heatmap"],  # Generate both chart types
    analysis_dir="output",
    viz_format="png"
)

print(f"Charts generated: {len(result['path'])} files")
```

### Spearman (Ordinal Data)
```python
# For ordinal data (ratings, rankings, etc.)
result = analyze_correlation(
    df[['satisfaction_rating', 'service_quality', 'price_value']],
    method="spearman",
    plot=True,
    output_chart="heatmap",
    analysis_dir="output"
)
```

### Kendall (Small Samples)
```python
# For small samples where rank-based correlation is more robust
result = analyze_correlation(
    df.sample(n=50),  # Sample of 50 records
    method="kendall",
    plot=False
)

print(result['result'])
```

## Type Conversion & Data Preparation

### Binary Categorical Mapping

Boolean and categorical columns with ≤2 unique values are automatically mapped to numeric:

```python
CATEGORICAL_MAPPINGS = {
    "yes": 1, "no": 0,
    "y": 1, "n": 0,
    "true": 1, "false": 0,
    "t": 1, "f": 0,
}
```

**Examples:**
```python
df = pd.DataFrame({
    'active': ['yes', 'no', 'yes', 'no'],
    'verified': [True, False, True, False],
    'score': [1, 2, 3, 4]
})

result = analyze_correlation(df)
# 'active' and 'verified' automatically converted to numeric
# Correlation calculated across all 3 fields
```

### Zero-Variance Filtering

Columns with constant values (variance ≈ 0) are automatically removed:

```python
df = pd.DataFrame({
    'always_5': [5, 5, 5, 5],       # Zero variance
    'region_code': [1, 1, 1, 1],    # Zero variance
    'salary': [50k, 60k, 70k, 80k]  # Non-zero variance ✓
})

result = analyze_correlation(df)
# Only 'salary' included in result (others have no variance)
```

## Visualization

### Chart Types

| Type | Appearance | Use Case | Annotation |
|------|-----------|----------|-----------|
| `matrix` | Grid with values | Exact value reading | Always annotated (0.3f precision) |
| `heatmap` | Color gradient | Pattern recognition | Conditional (annotated if ≤10 vars) |

### Automatic Skipping

Charts are skipped if DataFrame shape is insufficient:
- Requires ≥2×2 for meaningful correlation visualization
- Single-variable results may skip if only 1 row/column
- Pairwise results (2 cols) use full 2×2 matrix, not reduced 1×1

### Format & Backend

- **Backend**: Plotly (interactive HTML) or static export (PNG/SVG/JPG)
- **Default format**: HTML (interactive, zoomable)
- **Colorscale**:
  - Matrix: RdBu_r (red=negative, blue=positive)
  - Heatmap: Viridis (sequential, better for pattern recognition)

### Examples

```python
# Generate annotated correlation matrix
result = analyze_correlation(
    df,
    columns=['salary', 'age', 'experience'],
    plot=True,
    output_chart="matrix",
    analysis_dir="output",
    viz_format="html"
)
# Output: correlation_pearson_matrix_20260323_120530.html
# Includes exact correlation values in each cell
```

```python
# Generate heatmap for large correlation set
result = analyze_correlation(
    df,
    method="spearman",
    plot=True,
    output_chart="heatmap",
    viz_format="png"
)
# Output: correlation_spearman_heatmap_20260323_120530.png
# Color gradient; values only if ≤10 variables
```

## Best Practices

1. **Validate Input Columns**: Check that requested columns exist in DataFrame.
   ```python
   required_cols = ['salary', 'experience', 'age']
   missing = [c for c in required_cols if c not in df.columns]
   if missing:
       print(f"Warning: Missing columns {missing}")
   ```

2. **Choose Appropriate Method**:
   - **Pearson**: Linear relationships, normally distributed data
   - **Spearman**: Ordinal data, non-linear monotonic relationships
   - **Kendall**: Small samples, robust to outliers

3. **Exclude Non-Numeric Fields**: Non-numeric non-boolean columns are automatically filtered.
   ```python
   # Pre-filter if you want explicit control:
   numeric_cols = df.select_dtypes(include=['number']).columns
   result = analyze_correlation(df[numeric_cols], method="pearson")
   ```

4. **Check Result Type**: Different result types have different shapes.
   ```python
   if result['result_type'] == 'single_variable':
       print(f"Correlations with {result['result'].columns[0]}")
   elif result['result_type'] == 'pairwise':
       print("2-variable correlation")
   ```

5. **Handle Chart Skipping**: Charts may be skipped if data is insufficient.
   ```python
   if result['path'] is None:
       print("Chart skipped (insufficient data)")
   else:
       print(f"Chart generated: {result['path']}")
   ```

6. **Post-Process Results**: Normalize or filter correlations for reporting.
   ```python
   corr_matrix = result['result']
   strong_corr = corr_matrix.abs()[corr_matrix.abs() > 0.7]
   print("Strong correlations (>0.7):")
   print(strong_corr)
   ```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No suitable columns found" | All columns constant or non-numeric | Pre-filter data; check for constant columns |
| "ColumnNotFoundError" | Requested column not in DataFrame | Verify column names (case-sensitive) |
| Charts not generated | result['path'] is None | Check if data sufficient (≥2×2); review logs |
| Unexpected correlations | Binary categorical mapping | Verify mapping in CATEGORICAL_MAPPINGS |
| NaN in correlation | Missing values in columns | Pre-impute or drop NaN; use `.dropna()` |

## Backward Compatibility

The `raw_result` field maintains compatibility with legacy code:

```python
# Old code expecting Series for single-variable:
result = analyze_correlation(df, columns=['salary'])
corr_series = result['raw_result']  # Still a Series (not DataFrame)

# Old code expecting scalar for pairwise:
result = analyze_correlation(df, columns=['salary', 'age'])
scalar_corr = result['raw_result']  # Still a scalar (not DataFrame)

# New code uses normalized DataFrame:
corr_df = result['result']  # Always DataFrame
```

## Performance Considerations

- **Time Complexity**: O(n·m²) where n=rows, m=numeric columns
- **Memory**: Loads full DataFrame + correlation matrix
- **Suitable for**: <1M rows, <100 numeric columns
- **Optimization**: For large datasets, sample or pre-select columns

## Related Components

- [`analyze_descriptive_stats()`](./descriptive_stats.md) - Univariate statistics for correlation basis
- [`visualize_distribution_df()`](./distribution.md) - Marginal distributions of correlated variables
- [`analyze_dataset_summary()`](./dataset_summary.md) - Field types before correlation
- [`utils.visualization`](../utils/) - Underlying chart generation

## Summary Analysis

**Purpose**: Compute and visualize correlations between numeric/boolean fields using multiple methods.

**Inputs**: DataFrame, column selection, method, visualization preferences.

**Outputs**: Correlation matrix/series as DataFrame, optional charts, backward-compatible raw results.

**Strengths**:
- Automatic type detection and binary mapping
- Zero-variance column filtering
- Multiple correlation methods
- Optional interactive/static visualization
- Flexible result types (all/single/pairwise/selected)

**Limitations**:
- Assumes linear or monotonic relationships
- No causal inference
- No time series support
- Visualization scaling challenges with >50 variables

**Typical Workflow**: Explore relationships → Identify strong correlations → Visualize → Validate assumptions → Report findings.
