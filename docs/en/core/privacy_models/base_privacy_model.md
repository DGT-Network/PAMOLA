# BasePrivacyModelProcessor Documentation

**Module:** `pamola_core.privacy_models.base`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Methods](#core-methods)
4. [Inheritance Pattern](#inheritance-pattern)
5. [Usage Examples](#usage-examples)
6. [Implementation Guide](#implementation-guide)
7. [Best Practices](#best-practices)
8. [Related Components](#related-components)
9. [Summary Analysis](#summary-analysis)

## Overview

`BasePrivacyModelProcessor` is an abstract base class that defines the standard interface for all privacy model implementations in PAMOLA.CORE. It enforces a consistent API across different anonymization models (k-Anonymity, l-Diversity, t-Closeness, Differential Privacy).

**Purpose:** Ensure uniform behavior and interchangeability among privacy models.

**Location:** `pamola_core/privacy_models/base.py`

**Dependencies:**
- `pandas` — DataFrame manipulation
- `abc` — Abstract base class support

## Architecture

### Class Hierarchy

```
BasePrivacyModelProcessor (ABC)
├── KAnonymityProcessor
├── LDiversityCalculator
├── TCloseness
└── DifferentialPrivacyProcessor
```

All concrete processors inherit from this base class and implement its abstract methods.

### Design Principles

1. **Abstraction:** Hide implementation details behind a common interface
2. **Consistency:** Ensure all models behave predictably
3. **Extensibility:** Easy to add new privacy models
4. **Flexibility:** Support diverse anonymization strategies

## Core Methods

### 1. process(data)

**Signature:**
```python
@abstractmethod
def process(self, data: Any) -> Any:
```

**Purpose:** Process input data according to the specific processor logic.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | Any | Yes | Input data to be processed |

**Returns:** Processed data transformed according to processor-specific logic

**Notes:**
- Implementation varies by processor
- k-Anonymity: Applies grouping and suppression
- l-Diversity: Ensures attribute diversity
- Differential Privacy: Adds noise

### 2. evaluate_privacy(data, quasi_identifiers, **kwargs)

**Signature:**
```python
@abstractmethod
def evaluate_privacy(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str],
    **kwargs
) -> dict:
```

**Purpose:** Assess anonymization risks and model compliance without modifying data.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | pd.DataFrame | Yes | Dataset to evaluate |
| `quasi_identifiers` | list[str] | Yes | Columns used as quasi-identifiers |
| `**kwargs` | dict | No | Model-specific evaluation parameters |

**Returns:** Dictionary containing:
- Model compliance status (e.g., `is_k_anonymous`, `is_l_diverse`)
- Risk metrics (re-identification risks, disclosure rates)
- Dataset information (record counts, group sizes)
- Model-specific metrics

**Return Structure (Example):**
```python
{
    'is_k_anonymous': True,
    'min_k': 3,
    'max_k': 150,
    'avg_k': 25.4,
    'records_at_risk': 12,
    'dataset_info': {
        'total_records': 10000,
        'total_groups': 3000
    }
}
```

**Usage Example:**
```python
from pamola_core.privacy_models import KAnonymityProcessor

processor = KAnonymityProcessor(k=5)
evaluation = processor.evaluate_privacy(df, quasi_identifiers=['age', 'city'])

if not evaluation['is_k_anonymous']:
    print(f"Privacy violated: min_k={evaluation['min_k']} < required k=5")
```

### 3. apply_model(data, quasi_identifiers, suppression=True, **kwargs)

**Signature:**
```python
@abstractmethod
def apply_model(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str],
    suppression: bool = True,
    **kwargs
) -> pd.DataFrame:
```

**Purpose:** Apply privacy transformation to dataset to achieve privacy guarantees.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data` | pd.DataFrame | Yes | — | Input dataset |
| `quasi_identifiers` | list[str] | Yes | — | Columns to use for grouping |
| `suppression` | bool | No | True | Whether to suppress non-compliant records |
| `**kwargs` | dict | No | — | Model-specific application parameters |

**Returns:** `pd.DataFrame` with privacy transformations applied

**Behavior with suppression=True:**
- Non-compliant groups removed entirely
- Higher information loss but stronger guarantee

**Behavior with suppression=False:**
- Groups masked/generalized instead of removed
- Retains more records but weaker guarantee

**Usage Example:**
```python
processor = KAnonymityProcessor(k=3)

# Apply with suppression (removes non-compliant rows)
anonymized_strict = processor.apply_model(
    df,
    quasi_identifiers=['age', 'city'],
    suppression=True
)

# Apply without suppression (masks values instead)
anonymized_lenient = processor.apply_model(
    df,
    quasi_identifiers=['age', 'city'],
    suppression=False
)
```

## Inheritance Pattern

### Creating a Custom Privacy Model

Extend `BasePrivacyModelProcessor` to implement a new privacy model:

```python
from pamola_core.privacy_models.base import BasePrivacyModelProcessor
import pandas as pd

class CustomPrivacyModel(BasePrivacyModelProcessor):
    """Custom privacy model processor"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def process(self, data):
        """Process data using custom logic"""
        return data  # Your implementation

    def evaluate_privacy(
        self,
        data: pd.DataFrame,
        quasi_identifiers: list[str],
        **kwargs
    ) -> dict:
        """Evaluate privacy compliance"""
        # Implement evaluation logic
        return {
            'is_compliant': True,
            'risk_score': 0.25,
            'details': {}
        }

    def apply_model(
        self,
        data: pd.DataFrame,
        quasi_identifiers: list[str],
        suppression: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Apply privacy transformation"""
        # Implement anonymization logic
        return data.copy()
```

### Contract Obligations

When extending `BasePrivacyModelProcessor`, you must:

1. **Implement all abstract methods** — No exceptions
2. **Maintain method signatures** — Parameters and return types must match
3. **Handle edge cases** — Empty data, missing columns, invalid parameters
4. **Provide meaningful returns** — Evaluation should include compliance status
5. **Log operations** — Include logging for debugging and audit trails

## Usage Examples

### Example 1: Evaluate Without Transformation

```python
from pamola_core.privacy_models import KAnonymityProcessor
import pandas as pd

# Load data
df = pd.read_csv('customer_data.csv')

# Create processor
processor = KAnonymityProcessor(k=5)

# Evaluate current privacy level
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code', 'gender']
)

# Inspect results
if evaluation['is_k_anonymous']:
    print(f"Dataset is k-anonymous with min_k={evaluation['min_k']}")
else:
    print(f"Dataset NOT k-anonymous (min_k={evaluation['min_k']}, required=5)")
```

### Example 2: Apply Transformation

```python
from pamola_core.privacy_models import LDiversityCalculator

# Load data
df = pd.read_csv('patient_data.csv')

# Create processor
processor = LDiversityCalculator(l=3, diversity_type='distinct')

# Transform to l-diversity
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'zip_code'],
    suppression=True
)

print(f"Original records: {len(df)}")
print(f"Anonymized records: {len(anonymized)}")
print(f"Records removed: {len(df) - len(anonymized)}")
```

### Example 3: Workflow with Both Steps

```python
from pamola_core.privacy_models import TCloseness

# Create processor
processor = TCloseness(
    quasi_identifiers=['age', 'location'],
    sensitive_column='salary',
    t=0.15
)

# Step 1: Evaluate current privacy
eval_before = processor.evaluate_privacy(df, quasi_identifiers=['age', 'location'])
print(f"Before: is_t_close={eval_before['is_t_close']}, max_t={eval_before.get('max_t_value')}")

# Step 2: Apply transformation
anonymized = processor.apply_model(df, quasi_identifiers=['age', 'location'])

# Step 3: Verify privacy post-transformation
eval_after = processor.evaluate_privacy(
    anonymized,
    quasi_identifiers=['age', 'location']
)
print(f"After: is_t_close={eval_after['is_t_close']}")
```

## Implementation Guide

### Step 1: Choose Base Class

Inherit from `BasePrivacyModelProcessor`:

```python
from pamola_core.privacy_models.base import BasePrivacyModelProcessor

class MyPrivacyModel(BasePrivacyModelProcessor):
    pass
```

### Step 2: Initialize Configuration

Store model parameters in `__init__`:

```python
def __init__(self, parameter1: float, parameter2: str = "default"):
    self.parameter1 = parameter1
    self.parameter2 = parameter2
    self.logger = logging.getLogger(__name__)
```

### Step 3: Implement process()

Handle generic data processing:

```python
def process(self, data):
    """Generic processing implementation"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be pandas DataFrame")
    return data.copy()
```

### Step 4: Implement evaluate_privacy()

Calculate privacy metrics without modification:

```python
def evaluate_privacy(self, data: pd.DataFrame, quasi_identifiers: list[str], **kwargs) -> dict:
    """Evaluate privacy compliance"""
    try:
        # Validate inputs
        if not all(col in data.columns for col in quasi_identifiers):
            raise ValidationError("Missing quasi-identifier columns")

        # Calculate privacy metrics
        group_sizes = data.groupby(quasi_identifiers).size()
        min_group_size = group_sizes.min()

        # Return evaluation results
        return {
            'is_compliant': min_group_size >= self.parameter1,
            'min_group_size': int(min_group_size),
            'max_group_size': int(group_sizes.max()),
            'total_groups': len(group_sizes)
        }
    except Exception as e:
        self.logger.error(f"Evaluation failed: {e}")
        raise
```

### Step 5: Implement apply_model()

Transform data to achieve privacy:

```python
def apply_model(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str],
    suppression: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Apply privacy transformation"""
    result = data.copy()

    # Apply privacy logic
    group_sizes = result.groupby(quasi_identifiers).size()
    non_compliant = group_sizes[group_sizes < self.parameter1].index

    if suppression:
        # Remove non-compliant groups
        result = result[~result.set_index(quasi_identifiers).index.isin(non_compliant)]
    else:
        # Mask non-compliant groups
        mask = result.set_index(quasi_identifiers).index.isin(non_compliant)
        result.loc[mask, quasi_identifiers] = "MASKED"

    return result
```

## Best Practices

1. **Always Validate Inputs:**
   ```python
   if not isinstance(data, pd.DataFrame):
       raise TypeError("Expected pandas DataFrame")
   if not quasi_identifiers:
       raise ValueError("At least one quasi-identifier required")
   ```

2. **Use Consistent Error Handling:**
   ```python
   from pamola_core.errors.exceptions import ValidationError

   try:
       # Processing
   except Exception as e:
       self.logger.error(f"Operation failed: {e}")
       raise ValidationError(str(e)) from e
   ```

3. **Include Detailed Logging:**
   ```python
   logger = logging.getLogger(__name__)
   logger.info(f"Processing {len(data)} records")
   logger.debug(f"Quasi-identifiers: {quasi_identifiers}")
   ```

4. **Document Return Values:**
   ```python
   evaluation = processor.evaluate_privacy(df, quasi_ids)
   # Always include status field (is_k_anonymous, is_l_diverse, etc.)
   # Always include metric details (min/max values, risk scores)
   ```

5. **Support Both Pandas and Dask (if applicable):**
   ```python
   if isinstance(data, dd.DataFrame):
       # Dask-specific logic
   else:
       # Pandas-specific logic
   ```

## Related Components

- [KAnonymityProcessor](./k_anonymity/k_anonymity_processor.md)
- [LDiversityCalculator](./l_diversity/l_diversity_calculator.md)
- [TCloseness](./t_closeness/t_closeness_processor.md)
- [DifferentialPrivacyProcessor](./differential_privacy/dp_processor.md)

## Summary Analysis

`BasePrivacyModelProcessor` provides a flexible, extensible foundation for privacy models. It enforces a common contract while allowing implementation freedom. Use it to:

- **Ensure consistency** across different privacy models
- **Enable polymorphism** in privacy evaluation workflows
- **Standardize interfaces** for integration with other modules
- **Simplify extensibility** for custom privacy models

The three-method interface (process, evaluate_privacy, apply_model) covers all typical privacy workflows. Implementations should prioritize correctness and clear error handling over performance optimization.
