# k-Anonymity Visualization Documentation

**Module:** `pamola_core.privacy_models.k_anonymity.ka_visualization`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Core Functions](#core-functions)
3. [Usage Examples](#usage-examples)
4. [Configuration](#configuration)
5. [Output Format](#output-format)
6. [Best Practices](#best-practices)
7. [Related Components](#related-components)

## Overview

The k-Anonymity visualization module provides specialized visualization functions for analyzing k-Anonymity properties of datasets. It helps communicate privacy guarantees and assess anonymization effectiveness.

**Purpose:** Generate professional visualizations for k-Anonymity assessment and reporting.

**Location:** `pamola_core/privacy_models/k_anonymity/ka_visualization.py`

## Core Functions

### visualize_k_distribution()

**Purpose:** Visualize distribution of k-values (group sizes) across records.

**Signature:**
```python
def visualize_k_distribution(
    data: pd.DataFrame,
    k_column: str,
    save_path: Optional[str] = None,
    title: str = "Distribution of k-Anonymity Group Sizes",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 20,
    save_format: str = "png",
) -> Tuple[plt.Figure, Optional[str]]:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | — | Dataset with k-values |
| `k_column` | str | — | Column name containing k-values |
| `save_path` | str | None | Directory or file path to save |
| `title` | str | See above | Plot title |
| `figsize` | Tuple | (10, 6) | Figure size (width, height) |
| `bins` | int | 20 | Number of histogram bins |
| `save_format` | str | "png" | File format (png, pdf, svg) |

**Returns:** Tuple of (Figure, file_path or None)

**Example:**
```python
from pamola_core.privacy_models.k_anonymity.ka_visualization import visualize_k_distribution

processor = KAnonymityProcessor(k=5)
df_enriched = processor.enrich_with_k_values(df, ['age', 'zip_code'])

fig, path = visualize_k_distribution(
    df_enriched,
    k_column='k_value',
    save_path='/reports/visualizations/',
    title='k-Anonymity Distribution After Transformation'
)

print(f"Saved to: {path}")
```

### visualize_risk_heatmap()

**Purpose:** Create heatmap visualizing re-identification risk patterns.

**Signature:**
```python
def visualize_risk_heatmap(
    data: pd.DataFrame,
    risk_column: str,
    quasi_identifiers: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdYlGn_r",
    save_format: str = "png",
) -> Tuple[plt.Figure, Optional[str]]:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | — | Dataset with risk scores |
| `risk_column` | str | — | Risk score column |
| `quasi_identifiers` | List[str] | — | Quasi-identifier columns for grouping |
| `save_path` | str | None | Save location |
| `figsize` | Tuple | (12, 8) | Figure dimensions |
| `cmap` | str | "RdYlGn_r" | Matplotlib colormap |
| `save_format` | str | "png" | File format |

**Returns:** Tuple of (Figure, file_path)

**Example:**
```python
from pamola_core.privacy_models.k_anonymity.ka_visualization import visualize_risk_heatmap

processor = KAnonymityProcessor(k=5)
df_risk = processor.calculate_risk(df, ['age', 'zip_code'])

fig, path = visualize_risk_heatmap(
    df_risk,
    risk_column='risk_score',
    quasi_identifiers=['age', 'zip_code'],
    save_path='/reports/visualizations/'
)

print(f"Risk heatmap saved to: {path}")
```

## Usage Examples

### Example 1: Generate Single Visualization

```python
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity.ka_visualization import visualize_k_distribution

# Process data
processor = KAnonymityProcessor(k=5)
df_anonymized = processor.apply_model(df, ['age', 'zip_code'])
df_enriched = processor.enrich_with_k_values(df_anonymized, ['age', 'zip_code'])

# Generate visualization
fig, save_path = visualize_k_distribution(
    df_enriched,
    k_column='k_value',
    save_path='output/',
    title='k-Anonymity Group Sizes'
)

print(f"Visualization saved: {save_path}")
```

### Example 2: Generate Multiple Visualizations

```python
import os
from pathlib import Path
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity.ka_visualization import (
    visualize_k_distribution,
    visualize_risk_heatmap
)

processor = KAnonymityProcessor(k=5)
output_dir = Path('reports/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare data
df_anon = processor.apply_model(df, quasi_ids)
df_enriched = processor.enrich_with_k_values(df_anon, quasi_ids)
df_risk = processor.calculate_risk(df_anon, quasi_ids)

# Generate visualizations
fig1, path1 = visualize_k_distribution(
    df_enriched,
    'k_value',
    save_path=str(output_dir)
)

fig2, path2 = visualize_risk_heatmap(
    df_risk,
    'risk_score',
    quasi_identifiers=quasi_ids,
    save_path=str(output_dir)
)

print(f"Distribution plot: {path1}")
print(f"Risk heatmap: {path2}")
```

### Example 3: Custom Styling

```python
import matplotlib.pyplot as plt
from pamola_core.privacy_models.k_anonymity.ka_visualization import visualize_k_distribution

# Generate with custom settings
fig, path = visualize_k_distribution(
    df_enriched,
    k_column='k_value',
    save_path='output/',
    title='k-Anonymity Results (k=5)',
    figsize=(12, 8),
    bins=30,
    save_format='pdf'  # High-quality PDF
)

# Further customize if needed
ax = fig.get_axes()[0]
ax.set_xlabel('Group Size (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('custom_output.pdf', dpi=300)
```

### Example 4: Comparative Visualization

```python
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity.ka_visualization import visualize_k_distribution
import matplotlib.pyplot as plt

# Compare different k values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

k_values = [3, 5, 10]
for idx, k in enumerate(k_values):
    processor = KAnonymityProcessor(k=k)
    df_anon = processor.apply_model(df, quasi_ids)
    df_enriched = processor.enrich_with_k_values(df_anon, quasi_ids)

    ax = axes[idx]
    df_enriched['k_value'].hist(bins=20, ax=ax)
    ax.set_title(f'k={k}')
    ax.set_xlabel('Group Size')
    ax.set_ylabel('Number of Records')

plt.tight_layout()
plt.savefig('k_comparison.png', dpi=300)
plt.close()
```

### Example 5: Report-Ready Visualizations

```python
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity.ka_visualization import (
    visualize_k_distribution,
    visualize_risk_heatmap
)
from pamola_core.privacy_models.k_anonymity import KAnonymityReport
import json

# Process data
processor = KAnonymityProcessor(k=5)
df_anon = processor.apply_model(df, quasi_ids)
df_enriched = processor.enrich_with_k_values(df_anon, quasi_ids)
df_risk = processor.calculate_risk(df_anon, quasi_ids)

# Generate visualizations
viz_dir = 'report_visualizations/'
fig1, path1 = visualize_k_distribution(df_enriched, 'k_value', save_path=viz_dir)
fig2, path2 = visualize_risk_heatmap(df_risk, 'risk_score', quasi_ids, save_path=viz_dir)

# Build report with visualization paths
report_data = {
    'k_anonymity_configuration': {'k': 5},
    'privacy_evaluation': processor.evaluate_privacy(df_anon, quasi_ids),
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(df_anon)
    },
    'visualizations': {
        'k_distribution': path1,
        'risk_heatmap': path2
    }
}

# Generate full report
report = KAnonymityReport(report_data)
final_report = report.generate(include_visualizations=True)

# Save
with open('report.json', 'w') as f:
    json.dump(final_report, f, indent=2)
```

## Configuration

### Color Maps

Available colormaps for heatmaps:
- "RdYlGn_r" (default) — Red (high risk) to Green (low risk)
- "viridis" — Yellow (high) to Blue (low)
- "plasma" — Yellow (high) to Purple (low)
- "coolwarm" — Blue (low) to Red (high)

### File Formats

Supported save formats:
- "png" (default) — Raster, good for web/email
- "pdf" — Vector, good for printing
- "svg" — Vector, good for editing
- "jpg" — Compressed raster

### Figure Sizes

Common figure sizes (width, height in inches):
- (10, 6) — Default, good for most uses
- (12, 8) — Larger, for presentations
- (14, 10) — Extra large, for detailed analysis
- (8, 5) — Smaller, for compact reports

## Output Format

### File Naming

When save_path is a directory, files are automatically named with timestamp:
```
k_distribution_20260323_103045.png
risk_heatmap_20260323_103046.png
```

When save_path includes filename, that name is used:
```
/path/to/my_distribution.png
```

### Return Values

Both functions return tuple `(Figure, str)`:
- **Figure:** matplotlib Figure object (can be further customized)
- **str:** Path to saved file (None if not saved)

```python
fig, path = visualize_k_distribution(df, 'k_value', save_path='output/')
# fig is matplotlib.figure.Figure object
# path is '/output/k_distribution_20260323_103045.png' (or None)
```

## Best Practices

1. **Save Before Customizing:**
   ```python
   fig, path = visualize_k_distribution(df, 'k_value', save_path='output/')
   # File already saved, further changes to fig won't affect saved file
   ```

2. **Use Descriptive Titles:**
   ```python
   fig, path = visualize_k_distribution(
       df,
       'k_value',
       title='k-Anonymity Results After Generalization (k=5)'
   )
   ```

3. **Consistent File Formats:**
   ```python
   # Use PNG for reports and emails
   fig, path = visualize_k_distribution(df, 'k_value', save_format='png')

   # Use PDF for printing
   fig, path = visualize_k_distribution(df, 'k_value', save_format='pdf')
   ```

4. **Create Output Directory:**
   ```python
   from pathlib import Path

   output_dir = Path('reports/visualizations')
   output_dir.mkdir(parents=True, exist_ok=True)

   fig, path = visualize_k_distribution(df, 'k_value', save_path=str(output_dir))
   ```

5. **Pair with Reports:**
   ```python
   # Always include visualizations in reports
   report_data['visualizations'] = {
       'k_distribution': path1,
       'risk_heatmap': path2
   }
   ```

## Related Components

- [KAnonymityProcessor](./k_anonymity_processor.md) — Generates data for visualization
- [KAnonymityReport](./k_anonymity_report.md) — Includes visualization paths in reports
- [Privacy Models Overview](../privacy_models_overview.md) — Model concepts

## Summary

k-Anonymity visualizations help:
- **Communicate** privacy guarantees to stakeholders
- **Identify** high-risk groups and patterns
- **Document** anonymization effectiveness
- **Assess** group distribution after transformation

Use `visualize_k_distribution()` for overview analysis and `visualize_risk_heatmap()` for detailed risk assessment. Always include visualizations in compliance reports.
