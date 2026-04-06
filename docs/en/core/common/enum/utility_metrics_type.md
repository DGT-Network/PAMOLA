# UtilityMetricsType Enumeration

**Module:** `pamola_core.common.enum.utility_metrics_type`
**Class:** `UtilityMetricsType`
**Version:** 1.0
**Last Updated:** 2026-03-24

## Table of Contents
1. [Overview](#overview)
2. [Enum Members](#enum-members)
3. [Usage Examples](#usage-examples)
4. [Related Components](#related-components)
5. [Summary Analysis](#summary-analysis)

## Overview

`UtilityMetricsType` defines supported utility metrics for evaluating the functional quality of data post-anonymization. These metrics quantify how well anonymized data preserves the predictive power of the original dataset.

Two categories: **Regression** metrics (R2, MSE, MAE) for continuous targets and **Classification** metrics (AUROC, ACCURACY, F1, PRECISION, RECALL) for categorical targets.

## Enum Members

### Regression Metrics

| Member | Value | Description |
|--------|-------|-------------|
| `R2` | `"r2"` | Coefficient of Determination (R²) — proportion of variance explained |
| `MSE` | `"mse"` | Mean Squared Error — average squared prediction error |
| `MAE` | `"mae"` | Mean Absolute Error — average absolute prediction error |

### Classification Metrics

| Member | Value | Description |
|--------|-------|-------------|
| `AUROC` | `"auroc"` | Area Under ROC Curve — discrimination ability |
| `ACCURACY` | `"accuracy"` | Classification Accuracy — correct predictions ratio |
| `F1` | `"f1"` | F1-Score — harmonic mean of precision and recall |
| `PRECISION` | `"precision"` | Precision — positive predictive value |
| `RECALL` | `"recall"` | Recall — sensitivity / true positive rate |

## Usage Examples

### Basic Usage
```python
from pamola_core.common.enum.utility_metrics_type import UtilityMetricsType

# Select regression metric
metric = UtilityMetricsType.R2
print(metric.value)  # "r2"

# Select classification metric
metric = UtilityMetricsType.F1
print(metric.value)  # "f1"
```

### With UtilityMetricOperation
```python
from pamola_core.metrics import UtilityMetricOperation
from pamola_core.common.enum.utility_metrics_type import UtilityMetricsType

op = UtilityMetricOperation(
    metric_type=UtilityMetricsType.R2,
    target_column="salary"
)
```

### Iterate All Members
```python
for metric in UtilityMetricsType:
    print(f"{metric.name}: {metric.value}")
```

## Related Components
- [fidelity_metrics.md](fidelity_metrics.md) — Fidelity metrics (KS, KL, JS, Wasserstein)
- [privacy_metrics_type.md](privacy_metrics_type.md) — Privacy metrics (k-anonymity, DCR, NNDR)
- [enums_reference.md](enums_reference.md) — Quick reference for all enums
- [../../metrics/operations/utility_ops.md](../../metrics/operations/utility_ops.md) — UtilityMetricOperation

## Summary Analysis
- 8 members across regression (3) and classification (5) categories
- String enum (`str, Enum`) for JSON serialization
- Used by `UtilityMetricOperation` to select evaluation metric
- Covers standard ML evaluation metrics
