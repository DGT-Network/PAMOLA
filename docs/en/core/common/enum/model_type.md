# ModelType Enumeration

**Module:** `pamola_core.common.enum.model_type`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

ModelType defines the machine learning model types supported by PAMOLA.CORE. It provides type-safe selection of algorithms for various privacy-preserving data operations and analysis tasks.

## Members

| Member | Value | Algorithm Type | Purpose |
|--------|-------|-----------------|---------|
| `KNN` | `"knn"` | K-Nearest Neighbors | Distance-based classification and regression |
| `RANDOM_FOREST` | `"random_forest"` | Random Forest | Ensemble-based classification and regression |
| `LINEAR_REGRESSION` | `"linear_regression"` | Linear Regression | Continuous value prediction |

## Usage

### Basic Enumeration Access

```python
from pamola_core.common.enum.model_type import ModelType

# Access members
model = ModelType.KNN
print(model.value)  # Output: "knn"
print(model.name)   # Output: "KNN"
```

### Algorithm Selection

```python
from pamola_core.common.enum.model_type import ModelType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def instantiate_model(model_type: ModelType, **kwargs):
    """Create model instance based on type."""
    if model_type == ModelType.KNN:
        return KNeighborsClassifier(**kwargs)
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(**kwargs)
    elif model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegression(**kwargs)

# Usage
model = instantiate_model(ModelType.RANDOM_FOREST, n_estimators=100)
```

### Supported Model Listing

```python
from pamola_core.common.enum.model_type import ModelType

# Get all available models
available_models = [m.value for m in ModelType]
print(available_models)  # ["knn", "random_forest", "linear_regression"]

# Get model names
model_names = [m.name for m in ModelType]
print(model_names)  # ["KNN", "RANDOM_FOREST", "LINEAR_REGRESSION"]
```

## Member Descriptions

### KNN
**Value:** `"knn"`

K-Nearest Neighbors - A distance-based algorithm that classifies/predicts by finding the k nearest data points.

**Characteristics:**
- Non-parametric algorithm
- Lazy learning (no training phase)
- Distance-based decision making

**Use cases:**
- Classification and regression with small to medium datasets
- Privacy-preserving analysis (distance calculations can be obfuscated)
- Local pattern detection

**Considerations:**
- Sensitive to feature scaling and irrelevant features
- Computationally expensive for large datasets
- Works well with anonymized data using distance metrics

### RANDOM_FOREST
**Value:** `"random_forest"`

Random Forest - An ensemble method combining multiple decision trees for robust predictions.

**Characteristics:**
- Ensemble-based learning
- Handles both classification and regression
- Reduces overfitting through tree aggregation
- Feature importance available

**Use cases:**
- Classification and regression with high-dimensional data
- Feature importance analysis on anonymized datasets
- Robust predictions with privacy-preserving capabilities
- Dealing with mixed data types

**Considerations:**
- Requires more computational resources than single models
- Feature importance may leak sensitive information
- Less interpretable than individual decision trees

### LINEAR_REGRESSION
**Value:** `"linear_regression"`

Linear Regression - A parametric method for predicting continuous values using linear relationships.

**Characteristics:**
- Parametric algorithm
- Interpretable coefficients
- Computationally efficient
- Assumes linear relationships

**Use cases:**
- Continuous value prediction
- Understanding relationships between variables
- Baseline model for comparison
- Computationally lightweight privacy-preserving analysis

**Considerations:**
- Assumes linear relationships (may underfit complex data)
- Sensitive to outliers
- Model coefficients can be privacy-sensitive

## Selection Guide

### By Task Type

**Classification**
- Use `KNN` for local pattern-based classification
- Use `RANDOM_FOREST` for robust multi-class classification

**Regression**
- Use `LINEAR_REGRESSION` for simple linear relationships
- Use `KNN` for local value estimation
- Use `RANDOM_FOREST` for complex non-linear relationships

### By Dataset Characteristics

**Small to Medium Size**
- All models suitable; `KNN` or `LINEAR_REGRESSION` preferred

**Large Scale**
- `RANDOM_FOREST` for parallel processing
- `LINEAR_REGRESSION` for efficiency

**High Dimensionality**
- `RANDOM_FOREST` handles feature selection automatically
- Avoid `KNN` without dimensionality reduction

## Related Components

- **Utility Metrics:** Used with `UtilityMetricsType` to evaluate model performance
- **Privacy Metrics:** Evaluated alongside model accuracy to balance privacy-utility tradeoff
- **Distance Metrics:** `DistanceMetricType` used by KNN for similarity calculation

## Common Patterns

### Model Configuration with Type

```python
from pamola_core.common.enum.model_type import ModelType

model_config = {
    ModelType.KNN: {"n_neighbors": 5, "weights": "uniform"},
    ModelType.RANDOM_FOREST: {"n_estimators": 100, "max_depth": 10},
    ModelType.LINEAR_REGRESSION: {"fit_intercept": True}
}

def train_model(model_type: ModelType, X_train, y_train, **kwargs):
    """Train model with type-specific configuration."""
    base_config = model_config.get(model_type, {})
    config = {**base_config, **kwargs}

    if model_type == ModelType.KNN:
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**config).fit(X_train, y_train)
    # ... handle other types
```

### Model Performance Comparison

```python
from pamola_core.common.enum.model_type import ModelType
from pamola_core.common.enum.utility_metrics_type import UtilityMetricsType

def evaluate_models(X_test, y_test) -> dict:
    """Compare all model types."""
    results = {}
    for model_type in ModelType:
        model = train_model(model_type, X_train, y_train)
        score = model.score(X_test, y_test)
        results[model_type.value] = score
    return results
```

## Best Practices

1. **Use Enum for Type Safety**
   ```python
   # Good
   model_type = ModelType.RANDOM_FOREST

   # Avoid
   model_type = "random_forest"
   ```

2. **Match Model to Privacy Requirements**
   ```python
   # For highly sensitive data, prefer interpretable models
   if privacy_critical:
       model_type = ModelType.LINEAR_REGRESSION
   else:
       model_type = ModelType.RANDOM_FOREST
   ```

3. **Document Model Choice**
   ```python
   # Good - explains reasoning
   model_type = ModelType.KNN  # Minimal parameter storage, good for privacy
   ```

4. **Validate Dataset Size**
   ```python
   def select_model_for_dataset(df: pd.DataFrame) -> ModelType:
       """Choose model based on dataset characteristics."""
       if len(df) < 1000:
           return ModelType.KNN
       elif df.shape[1] > 50:
           return ModelType.RANDOM_FOREST
       else:
           return ModelType.LINEAR_REGRESSION
   ```

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [Utility Metrics Type](./utility_metrics_type.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- Three supported model types
- Detailed characteristics and use cases
