# Analysis Module Documentation

This directory contains comprehensive documentation for the `pamola_core.analysis` module—a suite of dataset analysis and privacy assessment tools.

## Documentation Files

| File | Purpose | Key Content |
|------|---------|-------------|
| **[analysis_overview.md](./analysis_overview.md)** | Module overview and quick start | Public API, architecture, dependencies, common use cases |
| **[dataset_summary.md](./dataset_summary.md)** | Dataset overview analysis | Field type detection, missing values, outlier identification |
| **[privacy_risk.md](./privacy_risk.md)** | Privacy risk assessment | k-anonymity, l-diversity, attack simulation, risk scoring |
| **[descriptive_stats.md](./descriptive_stats.md)** | Statistical summaries | Mean, std, median, mode, unique counts per field |
| **[distribution.md](./distribution.md)** | Distribution visualization | Histograms, bar charts, multiple output formats |
| **[correlation.md](./correlation.md)** | Correlation analysis | Pearson/Spearman/Kendall methods, matrix visualization |

## Quick Navigation

### Getting Started
Start with [analysis_overview.md](./analysis_overview.md) for:
- Overview of all 5 public functions
- Quick start examples
- Module architecture
- Common use cases

### Function-Specific Docs
Each function has dedicated documentation:
1. **`analyze_dataset_summary()`** → [dataset_summary.md](./dataset_summary.md)
2. **`calculate_full_risk()`** → [privacy_risk.md](./privacy_risk.md)
3. **`analyze_descriptive_stats()`** → [descriptive_stats.md](./descriptive_stats.md)
4. **`visualize_distribution_df()`** → [distribution.md](./distribution.md)
5. **`analyze_correlation()`** → [correlation.md](./correlation.md)

## Module at a Glance

**Location:** `pamola_core/analysis/`

**Public API (5 functions):**
```python
from pamola_core.analysis import (
    analyze_dataset_summary,        # Overview: rows, columns, types, outliers
    calculate_full_risk,            # Privacy: k-anonymity, l-diversity, attacks
    analyze_descriptive_stats,      # Statistics: mean, std, median, mode, etc.
    visualize_distribution_df,      # Visualizations: histograms, bar charts
    analyze_correlation             # Correlations: Pearson, Spearman, Kendall
)
```

**Typical Workflow:**
```
1. analyze_dataset_summary()      ← Understand structure
2. analyze_descriptive_stats()    ← Detailed field statistics
3. calculate_full_risk()          ← Assess privacy before anonymization
4. visualize_distribution_df()    ← Explore distributions
5. analyze_correlation()          ← Identify relationships
```

## Integration Points

The `analysis` module integrates with:
- **`profiling`** - Field type detection, statistical analysis
- **`anonymization`** - Apply transformations to reduce risk
- **`metrics`** - Measure utility after anonymization
- **`utils.visualization`** - Backend-agnostic chart generation
- **`errors`** - Structured error handling

## Key Features

- **Automatic field type detection** with numeric coercion
- **Formal privacy models** (k-anonymity, l-diversity, t-closeness)
- **Simulated attack metrics** (re-identification, attribute disclosure, membership inference)
- **Type-aware statistics** (mean/std for numeric, mode/unique for categorical)
- **Distribution visualization** with multiple formats (HTML, PNG, SVG, JPG)
- **Correlation analysis** with 3 methods and optional charting
- **JSON-serializable output** suitable for governance systems and dashboards

## Documentation Standards

Each function documentation includes:
- **Overview**: Purpose and key features
- **Key Features**: Bulleted capability list
- **Signature**: Function/method definitions with parameter tables
- **Returns**: Output schema with field descriptions
- **Usage Examples**: Real, runnable code examples
- **Best Practices**: Guidelines for effective usage
- **Troubleshooting**: Common issues and solutions
- **Related Components**: Links to related documentation

## Code References

All documentation is verified against actual source code in:
- `pamola_core/analysis/dataset_summary.py` (DatasetAnalyzer class)
- `pamola_core/analysis/privacy_risk.py` (calculate_full_risk + helpers)
- `pamola_core/analysis/descriptive_stats.py` (analyze_descriptive_stats)
- `pamola_core/analysis/distribution.py` (visualize_distribution_df)
- `pamola_core/analysis/correlation.py` (CorrelationAnalyzer class)
- `pamola_core/analysis/field_analysis.py` (Field-level analysis helper)

## Usage Examples

### Dataset Summary
```python
from pamola_core.analysis import analyze_dataset_summary
summary = analyze_dataset_summary(df)
print(f"Rows: {summary['rows']}, Columns: {summary['columns']}")
```

### Privacy Risk
```python
from pamola_core.analysis import calculate_full_risk
risk = calculate_full_risk(df, quasi_ids=['age', 'zip_code'], sensitive=['disease'])
print(f"Risk score: {risk['risk_assessment']}%")
```

### Descriptive Statistics
```python
from pamola_core.analysis import analyze_descriptive_stats
stats = analyze_descriptive_stats(df, field_names=['salary', 'age'])
print(stats['salary']['mean'])
```

### Distribution Visualization
```python
from pamola_core.analysis import visualize_distribution_df
paths = visualize_distribution_df(df, viz_dir=Path('output'))
```

### Correlation Analysis
```python
from pamola_core.analysis import analyze_correlation
result = analyze_correlation(df, method='spearman', plot=True)
```

## Documentation Maintenance

**Last Updated:** 2026-03-23

**Covered Version:** `pamola_core` 0.0.1

**Changes tracked in:** `pamola_core/analysis/__init__.py` (public API)

When updating the module:
1. Update relevant documentation file(s)
2. Verify code examples remain functional
3. Update parameter tables if signatures change
4. Add new function docs if public API expands
5. Update this README with new files/functions

## Related Documentation

- [Project Overview](../../project-overview-pdr.md)
- [Code Standards](../../code-standards.md)
- [System Architecture](../../system-architecture.md)
- [Profiling Module](../profiling/)
- [Anonymization Module](../anonymization/)
- [Metrics Module](../metrics/)
