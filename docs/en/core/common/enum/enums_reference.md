# Enumerations Quick Reference

**Module:** `pamola_core.common.enum`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

This document provides a comprehensive quick-reference table of all enumerations available in the common module, including member values and usage context.

## Quick Reference Table

### Encryption & Security

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **EncryptionMode** | NONE, SIMPLE, AGE | "none", "simple", "age" | Task framework encryption configuration |

### Masking Strategies

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **MaskStrategyEnum** | FIXED, PATTERN, RANDOM, WORDS | "fixed", "pattern", "random", "words" | Character-level masking strategies |

### Distance Metrics

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **DistanceMetricType** | EUCLIDEAN, MANHATTAN, COSINE, MAHALANOBIS | "euclidean", "manhattan", "cosine", "mahalanobis" | Distance calculations in privacy/utility metrics |

### Fidelity Metrics

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **FidelityMetrics** | KS, KL, JS, WASSERSTEIN | "ks", "kl", "js", "wasserstein" | Statistical similarity between datasets |
| **FidelityMetricsType** | KS, KL, JS, WASSERSTEIN | "ks", "kl", "js", "wasserstein" | Type variant of fidelity metrics |

### Privacy Metrics

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **PrivacyMetricsType** | DCR, NNDR, UNIQUENESS, K_ANONYMITY, L_DIVERSITY, T_CLOSENESS | "dcr", "nndr", "uniqueness", "k_anonymity", "l_diversity", "t_closeness" | Privacy preservation evaluation |

### Utility Metrics

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **UtilityMetricsType** | R2, MSE, MAE, AUROC, ACCURACY, F1, PRECISION, RECALL | "r2", "mse", "mae", "auroc", "accuracy", "f1", "precision", "recall" | Data utility and model performance |

### Generalization Methods

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **DatetimeMethod** | RANGE, PERIOD, FORMAT | "range", "period", "format" | DateTime generalization methods |
| **DatePeriod** | YEAR, QUARTER, MONTH, WEEKDAY, HOUR | "year", "quarter", "month", "weekday", "hour" | DateTime period levels |
| **NumericMethod** | BINNING, ROUNDING, SCALING | "binning", "rounding", "scaling" | Numeric generalization methods |

### Analysis & Processing

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **AnalysisMode** | ANALYZE, ENRICH, BOTH | "ANALYZE", "ENRICH", "BOTH" | K-anonymity analysis modes |
| **StatisticalMethod** | MEAN, MEDIAN, MODE | "mean", "median", "mode" | Statistical aggregation methods |
| **RelationshipType** | AUTO, ONE_TO_ONE, ONE_TO_MANY | "auto", "one-to-one", "one-to-many" | Data relationship types |

### Model & Language

| Enum | Members | Values | Purpose |
|------|---------|--------|---------|
| **ModelType** | KNN, RANDOM_FOREST, LINEAR_REGRESSION | "knn", "random_forest", "linear_regression" | ML model types |
| **Language** | ENGLISH, VIETNAMESE, RUSSIAN | "en", "vi", "ru" | Language support |

## Detailed Enum Listings

### EncryptionMode
```python
from pamola_core.common.enum.encryption_mode import EncryptionMode

EncryptionMode.NONE          # "none"      - No encryption
EncryptionMode.SIMPLE        # "simple"    - Simple symmetric encryption
EncryptionMode.AGE           # "age"       - AGE encryption with key rotation

# Method
EncryptionMode.from_string(value)  # Convert string to enum
```

**Documentation:** [EncryptionMode](./encryption_mode.md)

### MaskStrategyEnum
```python
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum

MaskStrategyEnum.FIXED       # "fixed"     - Fixed position masking
MaskStrategyEnum.PATTERN     # "pattern"   - Pattern-based masking
MaskStrategyEnum.RANDOM      # "random"    - Random character masking
MaskStrategyEnum.WORDS       # "words"     - Word-level masking
```

**Documentation:** [MaskStrategyEnum](./mask_strategy.md)

### DistanceMetricType
```python
from pamola_core.common.enum.distance_metric_type import DistanceMetricType

DistanceMetricType.EUCLIDEAN      # "euclidean"     - L2 distance
DistanceMetricType.MANHATTAN      # "manhattan"     - L1 distance
DistanceMetricType.COSINE         # "cosine"        - Angular distance
DistanceMetricType.MAHALANOBIS    # "mahalanobis"   - Statistical distance
```

**Documentation:** [DistanceMetricType](./distance_metric_type.md)

### FidelityMetrics / FidelityMetricsType
```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics
from pamola_core.common.enum.fidelity_metrics_type import FidelityMetricsType

FidelityMetrics.KS             # "ks"          - Kolmogorov-Smirnov test
FidelityMetrics.KL             # "kl"          - Kullback-Leibler divergence
FidelityMetrics.JS             # "js"          - Jensen-Shannon divergence
FidelityMetrics.WASSERSTEIN    # "wasserstein" - Wasserstein distance
```

### PrivacyMetricsType
```python
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

# Distance-based
PrivacyMetricsType.DCR         # "dcr"         - Distance to Closest Record
PrivacyMetricsType.NNDR        # "nndr"        - Nearest Neighbor Distance Ratio

# Uniqueness-based
PrivacyMetricsType.UNIQUENESS  # "uniqueness"  - Re-identification risk
PrivacyMetricsType.K_ANONYMITY # "k_anonymity" - k-Anonymity
PrivacyMetricsType.L_DIVERSITY # "l_diversity" - l-Diversity
PrivacyMetricsType.T_CLOSENESS # "t_closeness" - t-Closeness
```

### UtilityMetricsType
```python
from pamola_core.common.enum.utility_metrics_type import UtilityMetricsType

# Regression metrics
UtilityMetricsType.R2          # "r2"          - R² coefficient
UtilityMetricsType.MSE         # "mse"         - Mean Squared Error
UtilityMetricsType.MAE         # "mae"         - Mean Absolute Error

# Classification metrics
UtilityMetricsType.AUROC       # "auroc"       - Area Under ROC Curve
UtilityMetricsType.ACCURACY    # "accuracy"    - Classification Accuracy
UtilityMetricsType.F1          # "f1"          - F1-Score
UtilityMetricsType.PRECISION   # "precision"   - Precision Score
UtilityMetricsType.RECALL      # "recall"      - Recall Score
```

### DatetimeMethod & DatePeriod
```python
from pamola_core.common.enum.datetime_generalization import DatetimeMethod, DatePeriod

DatetimeMethod.RANGE          # "range"   - Range-based generalization
DatetimeMethod.PERIOD         # "period"  - Period-based generalization
DatetimeMethod.FORMAT         # "format"  - Format-based generalization

DatePeriod.YEAR               # "year"    - Year level
DatePeriod.QUARTER            # "quarter" - Quarter level
DatePeriod.MONTH              # "month"   - Month level
DatePeriod.WEEKDAY            # "weekday" - Weekday level
DatePeriod.HOUR               # "hour"    - Hour level
```

### NumericMethod
```python
from pamola_core.common.enum.numeric_generalization import NumericMethod

NumericMethod.BINNING         # "binning"  - Bin into ranges
NumericMethod.ROUNDING        # "rounding" - Round to nearest value
NumericMethod.SCALING         # "scaling"  - Scale to normalized range
```

### AnalysisMode
```python
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode

AnalysisMode.ANALYZE          # "ANALYZE" - Generate metrics and reports
AnalysisMode.ENRICH           # "ENRICH"  - Add k-values to DataFrame
AnalysisMode.BOTH             # "BOTH"    - Both analyze and enrich
```

### Language
```python
from pamola_core.common.enum.language_enum import Language

Language.ENGLISH              # "en"  - English
Language.VIETNAMESE           # "vi"  - Vietnamese
Language.RUSSIAN              # "ru"  - Russian
```

### StatisticalMethod
```python
from pamola_core.common.enum.statistical_method import StatisticalMethod

StatisticalMethod.MEAN        # "mean"   - Mean/average
StatisticalMethod.MEDIAN      # "median" - Median
StatisticalMethod.MODE        # "mode"   - Mode/most frequent
```

### RelationshipType
```python
from pamola_core.common.enum.relationship_type import RelationshipType

RelationshipType.AUTO         # "auto"         - Automatic detection
RelationshipType.ONE_TO_ONE   # "one-to-one"   - One-to-one relationship
RelationshipType.ONE_TO_MANY  # "one-to-many"  - One-to-many relationship
```

### ModelType
```python
from pamola_core.common.enum.model_type import ModelType

ModelType.KNN                 # "knn"               - K-Nearest Neighbors
ModelType.RANDOM_FOREST       # "random_forest"    - Random Forest
ModelType.LINEAR_REGRESSION   # "linear_regression" - Linear Regression
```

## UI/Form Enumerations

### CustomComponents
```python
from pamola_core.common.enum.custom_components import CustomComponents

# Component names for UI schema x-component attributes
CustomComponents.NUMERIC_RANGE_MODE
CustomComponents.DATE_FORMAT_ARRAY
CustomComponents.DATE_PICKER_ARRAY
CustomComponents.STRING_ARRAY
CustomComponents.UPLOAD
CustomComponents.DEPEND_SELECT
# ... and 20+ more components
```

### CustomFunctions
```python
from pamola_core.common.enum.custom_functions import CustomFunctions

# Function names for UI schema x-custom-function attributes
CustomFunctions.UPDATE_FIELD_OPTIONS
CustomFunctions.UPDATE_CONDITION_OPERATOR
CustomFunctions.INIT_FIELD_DOUBLE_SELECT
# ... and 8+ more functions
```

### GroupName
```python
from pamola_core.common.enum.form_groups import GroupName

# Form field grouping for UI organization
GroupName.CORE_GENERALIZATION_STRATEGY
GroupName.CONDITIONAL_LOGIC
GroupName.OPERATION_BEHAVIOR_OUTPUT
# ... and 70+ more group names

# Helper functions
from pamola_core.common.enum.form_groups import (
    get_groups_for_operation,
    get_groups_with_titles
)

groups = get_groups_for_operation("NumericGeneralizationConfig")
groups_with_titles = get_groups_with_titles("NumericGeneralizationConfig")
```

## Import Summary

```python
# Core enumerations
from pamola_core.common.enum.encryption_mode import EncryptionMode
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.common.enum.distance_metric_type import DistanceMetricType
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType
from pamola_core.common.enum.utility_metrics_type import UtilityMetricsType

# Generalization methods
from pamola_core.common.enum.datetime_generalization import DatetimeMethod, DatePeriod
from pamola_core.common.enum.numeric_generalization import NumericMethod

# Analysis & processing
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode
from pamola_core.common.enum.statistical_method import StatisticalMethod
from pamola_core.common.enum.relationship_type import RelationshipType

# Models and language
from pamola_core.common.enum.model_type import ModelType
from pamola_core.common.enum.language_enum import Language

# UI/Form
from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName, get_groups_for_operation
```

## Selection Guide

### By Use Case

**Privacy Analysis**
- `PrivacyMetricsType` - Select privacy metric
- `DistanceMetricType` - Choose distance calculation
- `AnalysisMode` - Configure analysis output

**Anonymization**
- `MaskStrategyEnum` - Select masking method
- `DatetimeMethod` / `NumericMethod` - Choose generalization
- `EncryptionMode` - Configure encryption

**Data Quality**
- `UtilityMetricsType` - Measure data utility
- `StatisticalMethod` - Aggregation method
- `FidelityMetrics` - Statistical similarity

**ML Operations**
- `ModelType` - Select model algorithm
- `UtilityMetricsType` - Evaluate performance
- `DistanceMetricType` - Distance calculations

**Configuration**
- `Language` - Language support
- `RelationshipType` - Data relationships
- `GroupName` - Form organization

## Common Patterns

### Convert String to Enum
```python
from pamola_core.common.enum.encryption_mode import EncryptionMode

mode = EncryptionMode.from_string("age")  # Returns EncryptionMode.AGE
```

### Check Enum Membership
```python
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

if metric == PrivacyMetricsType.K_ANONYMITY:
    # Handle k-anonymity specific logic
    pass
```

### Iterate Over Enum Members
```python
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode

for mode in AnalysisMode:
    print(f"Mode: {mode.name}, Value: {mode.value}")
```

### Get All Values
```python
from pamola_core.common.enum.language_enum import Language

languages = [lang.value for lang in Language]
# ['en', 'vi', 'ru']
```

## Related Documentation

- [Common Module Overview](../common_overview.md)
- Individual enum documentation in `/enum/` directory
- Constants and Type Aliases in [type_aliases.md](../type_aliases.md)
- Form groups in [Form Groups Reference](./form_groups.md)

## Maintenance Notes

All enums are maintained in `pamola_core/common/enum/` directory. When adding new enums:
1. Use descriptive lowercase values
2. Include docstring explaining purpose
3. Add to this quick reference
4. Create dedicated documentation file if substantial
5. Update related component documentation
