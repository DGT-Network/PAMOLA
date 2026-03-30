# Visualization Helpers Module Documentation

**Package:** `pamola_core.utils.vis_helpers`
**Version:** 1.0
**Last Updated:** 2026-03-23
**Type:** Internal (Non-Public API)

## Overview

The `vis_helpers` package provides comprehensive visualization utilities for PAMOLA.CORE operations. It supports both Matplotlib and Plotly backends with a unified interface, enabling operations to generate various plot types, heatmaps, correlation matrices, word clouds, and network diagrams with consistent styling and theming.

## Architecture

```
pamola_core.utils.vis_helpers/
├── Core Framework
│   ├── base.py - BaseFigure, FigureFactory, FigureRegistry
│   ├── registry.py - Figure registration
│   └── context.py - Execution contexts
│
├── Plot Types
│   ├── bar_plots.py - Bar charts (Matplotlib, Plotly)
│   ├── histograms.py - Histograms (Matplotlib, Plotly)
│   ├── line_plots.py - Line plots (Matplotlib, Plotly)
│   ├── scatter_plots.py - Scatter plots (Matplotlib, Plotly)
│   ├── box_plot.py - Box plots (Matplotlib, Plotly)
│   ├── pie_charts.py - Pie and sunburst charts (Plotly)
│   └── combined_charts.py - Multi-series charts
│
├── Advanced Visualizations
│   ├── cor_matrix.py - Correlation matrices
│   ├── cor_pair.py - Pairwise correlations
│   ├── heatmap.py - General heatmaps
│   ├── spider_charts.py - Radar/spider charts
│   ├── word_clouds.py - Word cloud generation
│   ├── venn_diagram.py - Venn diagrams
│   └── network_diagram.py - Network graphs
│
├── Correlation Utilities
│   ├── cor_utils.py - Correlation calculation and masking
│   └── Significance testing
│
├── Theming & Styling
│   ├── theme.py - Theme management and customization
│   └── Color schemes
│
└── __init__.py - Public API exports
```

## Component Files

| File | Purpose | Key Classes |
|------|---------|---|
| `base.py` | Foundation classes | `BaseFigure`, `MatplotlibFigure`, `PlotlyFigure`, `FigureFactory`, `FigureRegistry` |
| `bar_plots.py` | Bar charts | `MatplotlibBarPlot`, `PlotlyBarPlot` |
| `histograms.py` | Histograms | `MatplotlibHistogram`, `PlotlyHistogram` |
| `line_plots.py` | Line plots | `MatplotlibLinePlot`, `PlotlyLinePlot` |
| `scatter_plots.py` | Scatter plots | `MatplotlibScatterPlot`, `PlotlyScatterPlot` |
| `boxplot.py` | Box plots | `MatplotlibBoxPlot`, `PlotlyBoxPlot` |
| `pie_charts.py` | Pie/Sunburst | `MatplotlibPieChart`, `PlotlyPieChart`, `PlotlySunburstChart` |
| `combined_charts.py` | Multi-series | `MatplotlibCombinedChart`, `PlotlyCombinedChart` |
| `cor_matrix.py` | Correlation matrices | `MatplotlibCorrelationMatrix`, `PlotlyCorrelationMatrix` |
| `cor_pair.py` | Pairwise correlation | `MatplotlibCorrelationPair`, `PlotlyCorrelationPair` |
| `heatmap.py` | General heatmaps | `MatplotlibHeatmap`, `PlotlyHeatmap` |
| `spider_charts.py` | Radar charts | `MatplotlibSpiderChart`, `PlotlySpiderChart` |
| `word_clouds.py` | Word clouds | `WordCloudGenerator` |
| `venn_diagram.py` | Venn diagrams | `MatplotlibVennDiagram`, `PlotlyVennDiagram` |
| `network_diagram.py` | Network graphs | `MatplotlibNetworkDiagram`, `PlotlyNetworkDiagram` |
| `theme.py` | Theming | Theme management and color schemes |
| `context.py` | Contexts | Visualization execution contexts |
| `registry.py` | Registration | `register_builtin_figures()` |

## Key Concepts

### Dual Backend Support

Visualizations support both Matplotlib and Plotly:

- **Matplotlib**: Publication-quality static images
- **Plotly**: Interactive web-based visualizations

Switch backends dynamically:

```python
from pamola_core.utils.vis_helpers import set_backend

set_backend("plotly")  # Use Plotly
set_backend("matplotlib")  # Use Matplotlib
```

### Unified Figure Interface

All figure types inherit from `BaseFigure` with consistent API:

```python
figure = BarPlot(
    data=df,
    x_col="category",
    y_col="value",
    title="Sales by Category",
    theme="light"
)

# Render to different formats
html = figure.to_html()
image = figure.to_image()
json_data = figure.to_json()
```

### Theme Management

Consistent styling across all visualizations:

```python
from pamola_core.utils.vis_helpers import set_theme, create_custom_theme

# Use built-in theme
set_theme("dark")

# Create custom theme
custom = create_custom_theme(
    primary_color="#1f77b4",
    background_color="#ffffff",
    font_family="Arial"
)
```

### Correlation Analysis

Advanced correlation visualization with significance testing:

```python
from pamola_core.utils.vis_helpers import PlotlyCorrelationMatrix

corr_matrix = PlotlyCorrelationMatrix(
    data=df,
    method="pearson",
    significance_level=0.05,
    mask_insignificant=True
)
```

### Word Cloud Generation

Generate word clouds from text data:

```python
from pamola_core.utils.vis_helpers import WordCloudGenerator

wc = WordCloudGenerator(
    texts=text_list,
    max_words=100,
    width=800,
    height=400
)

image = wc.generate()
```

## Usage Patterns

### Pattern 1: Simple Bar Plot

```python
from pamola_core.utils.vis_helpers import PlotlyBarPlot
import pandas as pd

df = pd.DataFrame({
    "category": ["A", "B", "C"],
    "value": [10, 20, 15]
})

plot = PlotlyBarPlot(
    data=df,
    x_col="category",
    y_col="value",
    title="Sales by Category"
)

html = plot.to_html()
```

### Pattern 2: Correlation Matrix with Significance

```python
from pamola_core.utils.vis_helpers import PlotlyCorrelationMatrix

corr = PlotlyCorrelationMatrix(
    data=df,
    method="spearman",
    significance_level=0.05,
    mask_insignificant=True,
    annotation_format=".2f"
)

html = corr.to_html()
```

### Pattern 3: Multi-Series Comparison

```python
from pamola_core.utils.vis_helpers import MatplotlibCombinedChart

chart = MatplotlibCombinedChart(
    data=df,
    x_col="date",
    y_cols=["revenue", "cost", "profit"],
    chart_type="line",
    title="Financial Overview"
)

image = chart.to_image("png")
```

### Pattern 4: Theme-Styled Visualization

```python
from pamola_core.utils.vis_helpers import (
    set_theme,
    PlotlyScatterPlot
)

set_theme("dark")

plot = PlotlyScatterPlot(
    data=df,
    x_col="age",
    y_col="income",
    color_col="category",
    title="Income vs Age by Category"
)

html = plot.to_html()
```

### Pattern 5: Network Diagram

```python
from pamola_core.utils.vis_helpers import PlotlyNetworkDiagram

diagram = PlotlyNetworkDiagram(
    nodes=nodes_df,
    edges=edges_df,
    node_size_col="size",
    edge_weight_col="weight",
    title="Relationship Network"
)

html = diagram.to_html()
```

## Supported Plot Types

| Plot Type | Matplotlib | Plotly | Use Cases |
|-----------|-----------|--------|-----------|
| Bar Plot | ✓ | ✓ | Categorical comparisons |
| Histogram | ✓ | ✓ | Distribution analysis |
| Line Plot | ✓ | ✓ | Time series, trends |
| Scatter Plot | ✓ | ✓ | Relationship analysis |
| Box Plot | ✓ | ✓ | Distribution comparison |
| Pie Chart | ✓ | ✓ | Composition |
| Sunburst Chart | - | ✓ | Hierarchical composition |
| Heatmap | ✓ | ✓ | Matrix visualization |
| Correlation Matrix | ✓ | ✓ | Feature relationships |
| Spider/Radar | ✓ | ✓ | Multi-dimensional comparison |
| Word Cloud | ✓ | - | Text frequency analysis |
| Venn Diagram | ✓ | ✓ | Set relationships |
| Network Diagram | ✓ | ✓ | Graph visualization |

## API Reference

### Core Classes

#### BaseFigure

Base class for all visualization types.

```python
class BaseFigure:
    def render(self) -> str
    def to_html(self) -> str
    def to_json(self) -> Dict
    def to_image(self, format: str) -> bytes
    def save(self, filepath: str)
```

#### MatplotlibFigure

Matplotlib backend implementation.

#### PlotlyFigure

Plotly backend implementation.

#### FigureFactory

Factory for creating appropriate figure types:

```python
from pamola_core.utils.vis_helpers import FigureFactory

figure = FigureFactory.create(
    figure_type="bar",
    backend="plotly",
    data=df,
    x_col="x",
    y_col="y"
)
```

#### FigureRegistry

Registry for managing figure types:

```python
from pamola_core.utils.vis_helpers import FigureRegistry

registry = FigureRegistry()
available_types = registry.get_available_types()
figure_class = registry.get_figure_class("bar")
```

### Theming Functions

#### set_theme()

```python
def set_theme(theme_name: str) -> None
```

Set global theme (e.g., "light", "dark", "minimal")

#### create_custom_theme()

```python
def create_custom_theme(
    primary_color: str,
    secondary_color: str,
    background_color: str,
    font_family: str,
    **kwargs
) -> Dict
```

Create custom theme

#### get_current_theme()

```python
def get_current_theme() -> Dict
```

Get currently active theme

## Integration Points

`vis_helpers` is used by:

- **Profiling Operations**: Visualize data distributions
- **Analysis Reports**: Generate analysis visualizations
- **Dashboards**: Web-based visualization display
- **Web UI**: Result visualization
- **Export Modules**: Visual report generation

## Best Practices

1. **Choose Backend Appropriately**
   - Use Plotly for web/interactive visualization
   - Use Matplotlib for publication/static images
   - Consider use case before choosing

2. **Apply Consistent Theming**
   - Set theme once per session
   - Use built-in themes for consistency
   - Create custom theme for branding

3. **Optimize for Large Datasets**
   - Sample data for very large datasets
   - Use appropriate visualization types
   - Consider performance implications

4. **Handle Missing Data**
   - Clean data before visualization
   - Handle NaN values appropriately
   - Document any data exclusions

5. **Provide Context**
   - Include descriptive titles
   - Add axis labels
   - Use clear legends

## Troubleshooting

### Issue: Missing Required Data Columns

**Solution:**
```python
# Verify columns exist in DataFrame
required_cols = [x_col, y_col]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
```

### Issue: Plotly Not Rendering in Jupyter

**Solution:**
```python
# Ensure plotly is properly installed
import plotly.io as pio
pio.renderers.default = "notebook"  # or "vscode", "browser"
```

### Issue: Memory Issues with Large Correlation Matrix

**Solution:**
```python
# Sample data if too large
if len(df) > 10000:
    df_sample = df.sample(n=10000, random_state=42)
else:
    df_sample = df

corr = PlotlyCorrelationMatrix(data=df_sample)
```

## Related Documentation

- [base.py](./base.md) - Figure base classes and factory
- [theme.py](./theme.md) - Theming system
- [cor_utils.py](./cor_utils.md) - Correlation utilities
- [Visualization Guide](../../../visualization-guide.md) - Comprehensive usage guide

## Summary

The `vis_helpers` package provides comprehensive visualization infrastructure for PAMOLA.CORE operations. It offers dual backend support, consistent styling, and a wide variety of plot types.

Key strengths:
- Unified interface for multiple backends
- Comprehensive plot type coverage
- Advanced correlation analysis
- Theme management system
- Extension-friendly architecture

Use `vis_helpers` to:
- Visualize operation results
- Generate analytical reports
- Create interactive dashboards
- Build data exploration tools
- Document analysis findings
