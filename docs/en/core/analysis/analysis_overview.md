# Analysis Module Documentation

**Module:** `pamola_core.analysis`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

The `analysis` module provides comprehensive dataset analysis and privacy risk assessment tools for pandas DataFrames. It offers five public functions that enable developers to understand data structure, compute privacy metrics, analyze distributions, and evaluate correlation patterns.

This module integrates seamlessly with the broader PAMOLA ecosystem for privacy-preserving data operations. All functions return structured, JSON-serializable dictionaries suitable for downstream processing, visualization, and governance workflows.

Key capabilities include automatic field type detection, privacy risk scoring using formal models (k-anonymity, l-diversity), distribution visualization, and correlation analysis with optional chart generation.

## Public API Functions

| Function | Module | Purpose |
|----------|--------|---------|
| [`analyze_dataset_summary()`](./dataset_summary.md) | `dataset_summary.py` | Overview stats: rows, columns, missing values, field types, outliers |
| [`calculate_full_risk()`](./privacy_risk.md) | `privacy_risk.py` | Privacy risk assessment using k-anonymity, l-diversity, attack simulation |
| [`analyze_descriptive_stats()`](./descriptive_stats.md) | `descriptive_stats.py` | Statistical summaries (mean, std, median, mode, unique counts) |
| [`visualize_distribution_df()`](./distribution.md) | `distribution.py` | Generate histograms/bar charts for numeric and categorical fields |
| [`analyze_correlation()`](./correlation.md) | `correlation.py` | Correlation analysis with Pearson/Spearman/Kendall methods |

## Module Architecture

```
pamola_core.analysis/
├── __init__.py                # Public API exports
├── dataset_summary.py         # DatasetAnalyzer, analyze_dataset_summary()
├── privacy_risk.py            # Privacy risk metrics + attack simulation
├── descriptive_stats.py       # Normalized descriptive statistics
├── distribution.py            # Visualization generation
├── correlation.py             # CorrelationAnalyzer, analyze_correlation()
└── field_analysis.py          # Field-level analysis helper
```

## Core Dependencies

**External packages:**
- `pandas` - DataFrame operations, type inference, groupby/aggregation
- `numpy` - Numeric operations, entropy calculations, binning
- `scipy.stats.wasserstein_distance` - Earth Mover's Distance for t-closeness
- `pathlib` / `datetime` - File paths and timestamps

**Internal dependencies:**
- `pamola_core.profiling.commons.statistical_analysis` - IQR outlier detection
- `pamola_core.utils.visualization` - Chart generation (histograms, heatmaps)
- `pamola_core.utils.logging` - Structured logging
- `pamola_core.errors.exceptions` - Error handling (ValidationError, ColumnNotFoundError)

## Quick Start

### 1. Dataset Summary
```python
import pandas as pd
from pamola_core.analysis import analyze_dataset_summary

df = pd.read_csv("data.csv")
summary = analyze_dataset_summary(df)
print(f"Rows: {summary['rows']}, Columns: {summary['columns']}")
print(f"Missing values: {summary['missing_values']['value']}")
print(f"Outliers detected: {summary['outliers']['count']}")
```

### 2. Privacy Risk Assessment
```python
from pamola_core.analysis import calculate_full_risk

quasi_ids = ["age", "zip_code", "gender"]
sensitive_attrs = ["disease", "salary"]

risk = calculate_full_risk(df, quasi_ids, sensitive_attrs)
print(f"Overall risk score: {risk['risk_assessment']}%")
print(f"k-anonymity: {risk['k_anonymity']['k']}")
print(f"Re-identification risk: {risk['reidentification_risk']}")
```

### 3. Descriptive Statistics
```python
from pamola_core.analysis import analyze_descriptive_stats

stats = analyze_descriptive_stats(
    df,
    field_names=["salary", "experience"],
    extra_statistics=["median", "mode", "unique"]
)
for field, values in stats.items():
    print(f"{field}: mean={values.get('mean')}, std={values.get('std')}")
```

### 4. Distribution Visualization
```python
from pamola_core.analysis import visualize_distribution_df
from pathlib import Path

viz_paths = visualize_distribution_df(
    df,
    viz_dir=Path("output/distributions"),
    n_bins=15,
    field_names=["age", "salary"],
    viz_format="html"
)
print(f"Visualizations saved: {list(viz_paths.keys())}")
```

### 5. Correlation Analysis
```python
from pamola_core.analysis import analyze_correlation

result = analyze_correlation(
    df,
    columns=["salary", "experience", "age"],
    method="pearson",
    plot=True,
    output_chart=["heatmap", "matrix"],
    analysis_dir="output/correlation",
    viz_format="html"
)
print(f"Correlation matrix shape: {result['result'].shape}")
print(f"Charts saved: {result['path']}")
```

## Design Patterns

### 1. Normalized Return Values
All functions return structured dictionaries with consistent schemas:
- **Dataset Summary**: `{rows, columns, total_cells, missing_values, numeric_fields, ...}`
- **Privacy Risk**: `{k_anonymity, l_diversity, risk_assessment, ...}`
- **Descriptive Stats**: `{field_name: {count, mean, std, median, mode, ...}, ...}`
- **Distribution**: `{field_name: Path, ...}`
- **Correlation**: `{result: DataFrame, result_type: str, path: str|List[str]|None, ...}`

### 2. Class + Function Pattern
Complex modules (e.g., `CorrelationAnalyzer`, `DatasetAnalyzer`) provide:
- **Class**: Full internal control, method composition, logging
- **Convenience function**: Wrapper for simple one-off use cases

Example:
```python
# Class-based (advanced control)
analyzer = CorrelationAnalyzer()
result = analyzer.analyze_correlation(df, method="spearman")

# Function-based (quick usage)
result = analyze_correlation(df, method="spearman")
```

### 3. Error Handling
All functions validate inputs and raise typed exceptions:
- `ValidationError` - Invalid parameters or missing data
- `ColumnNotFoundError` - Referenced columns don't exist
- `DataError` - Empty or unsuitable data
- `TypeValidationError` - Wrong type for parameter

```python
try:
    risk = calculate_full_risk(df, ["nonexistent_col"], ["sensitive"])
except ColumnNotFoundError as e:
    print(f"Missing: {e.column_name}")
```

## Common Use Cases

### Case 1: Pre-Anonymization Analysis
Before applying anonymization operations:
```python
summary = analyze_dataset_summary(df)
risk_before = calculate_full_risk(df, quasi_ids, sensitive_attrs)
print(f"Risk before: {risk_before['risk_assessment']}%")
```

### Case 2: Data Quality Assessment
Understand data structure and completeness:
```python
summary = analyze_dataset_summary(df)
stats = analyze_descriptive_stats(df)
print(f"Data quality: {100 - (summary['missing_values']['value'] / summary['total_cells']) * 100}%")
```

### Case 3: Privacy Monitoring
Track privacy metrics over data transformations:
```python
risks = []
for transformed_df in transformation_pipeline:
    risk = calculate_full_risk(transformed_df, quasi_ids, sensitive_attrs)
    risks.append(risk['risk_assessment'])
    if risk['risk_assessment'] > 50:
        print("WARNING: Privacy threshold exceeded")
```

### Case 4: Exploratory Data Analysis (EDA)
Generate comprehensive analysis reports:
```python
# Summary statistics
summary = analyze_dataset_summary(df)
stats = analyze_descriptive_stats(df)

# Visualizations
analyze_distribution_df(df, viz_dir=Path("eda/distributions"))
analyze_correlation(df, plot=True, analysis_dir="eda/correlations")
```

## Best Practices

1. **Validate Quasi-Identifiers**: Ensure QI columns represent realistic re-identification risks. Include only attributes an attacker could know (e.g., age, gender, ZIP code), not unique identifiers (SSN, email).

2. **Choose Appropriate Methods**: Use `spearman` for ordinal data, `pearson` for linear relationships, `kendall` for small samples.

3. **Handle Missing Values**: Functions treat NaN as missing data. Pre-impute or filter before analysis if needed.

4. **Specify Visualization Directories**: Always create output directories before calling visualization functions. Use timestamped filenames to avoid overwrites.

5. **Set Realistic Weights**: When customizing privacy risk weights, ensure they sum to 1.0 and reflect organizational risk tolerance.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ColumnNotFoundError` when analyzing | Check column names match DataFrame columns exactly (case-sensitive) |
| Empty visualization output | Ensure viz_dir exists and is writable; check field_names parameter |
| Constant correlation values | Remove zero-variance columns; filter out constant fields |
| High memory usage | Analyze subsets of data; use `field_names` to limit scope |
| Risk score always 0 | Verify quasi_identifiers form meaningful equivalence classes |

## Related Modules

- [`profiling`](../profiling/) - Field type detection and data profiling
- [`anonymization`](../anonymization/) - Applies transformations to reduce privacy risk
- [`metrics`](../metrics/) - Measures utility and privacy after anonymization
- [`utils.visualization`](../utils/) - Backend-agnostic plotting utilities

## Summary Analysis

**Module Purpose**: Enable comprehensive dataset analysis, privacy assessment, and exploratory data analysis (EDA) workflows.

**Target Users**: Data scientists, privacy engineers, governance teams evaluating or monitoring data.

**Key Strengths**:
- Formal privacy models (k-anonymity, l-diversity, t-closeness) integrated with simulated attacks
- Automatic field type detection with numeric coercion
- Optional visualization generation with configurable formats
- Backward-compatible convenience functions alongside class-based APIs

**Limitations**:
- Simulated attacks are heuristic-based, not cryptographically rigorous
- No support for temporal or hierarchical data patterns
- Correlation analysis requires at least 2 variables; single-variable results may not visualize

**Integration**: Works with any pandas DataFrame; output format (dicts/DataFrames) integrates cleanly with downstream governance systems, BI tools, and ML pipelines.
