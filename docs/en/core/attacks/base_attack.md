# Base Attack Documentation

**Module:** `pamola_core.attacks.base`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Class Reference](#class-reference)
3. [Abstract Methods](#abstract-methods)
4. [Implementation Notes](#implementation-notes)
5. [Related Classes](#related-classes)
6. [Summary](#summary)

## Overview

`AttackInitialization` is the abstract base class for all attack classes in PAMOLA.CORE. It defines the contract that all concrete attack implementations must follow.

**Purpose:** Establish consistent interface for attack simulation classes:
- Data preprocessing pipeline
- Attack execution pattern
- Error handling and validation

**Inheritance Hierarchy:**
```
AttackInitialization (ABC)
  └─ PreprocessData
       ├─ LinkageAttack
       ├─ MembershipInference
       ├─ AttributeInference
       ├─ DistanceToClosestRecord
       ├─ NearestNeighborDistanceRatio
       └─ AttackMetrics
```

## Class Reference

### AttackInitialization

Abstract base class for all attack implementations in PAMOLA.CORE.

```python
from pamola_core.attacks import AttackInitialization

class AttackInitialization(ABC):
    """
    Abstract base class for attack simulation in PAMOLA.CORE.
    This class extends BaseProcessor and declare methods used for attack simulation.
    """
```

**Availability:** Both `AttackInitialization` and `BaseAttack` names are exported (aliases)
```python
from pamola_core.attacks import AttackInitialization
from pamola_core.attacks import BaseAttack  # Same class
```

## Abstract Methods

### preprocess_data

Convert input datasets into numeric feature vectors.

```python
@abstractmethod
def preprocess_data(self, data1, data2):
    """
    Data preprocessing: Convert all string elements of data1 and data2 to numbers.
    Uses TF-IDF for categorical columns and StandardScaler for numeric columns.

    Parameters
    -----------
    data1: pd.DataFrame
        First dataset (reference/training set).
        Used to fit vectorizer and scaler vocabularies.

    data2: pd.DataFrame
        Second dataset (query/test set).
        Transformed using vocabularies learned from data1.

    Returns
    -----------
    data1_final: np.ndarray
        Numeric feature matrix for data1 (TF-IDF + scaled numeric).
        Shape: (n_samples_1, n_features)

    data2_final: np.ndarray
        Numeric feature matrix for data2 (TF-IDF + scaled numeric).
        Shape: (n_samples_2, n_features)
    """
    pass
```

**Implementation Details:**
- Subclasses must implement TF-IDF vectorization for categorical columns
- Subclasses must implement StandardScaler for numeric columns
- Both datasets must use the same vocabulary (fit on data1, transform both)

### process

Execute core attack logic on input data.

```python
@abstractmethod
def process(self, data):
    """
    Process the input data according to specific attack logic.

    Parameters
    -----------
    data : Any
        The input data to be processed (typically pd.DataFrame).
        Type depends on specific attack implementation.

    Returns
    --------
    Any
        Processed result. Type depends on attack:
        - LinkageAttack: DataFrame with match pairs
        - MembershipInference: numpy array of binary predictions
        - AttributeInference: pandas Series of predicted values
    """
    pass
```

**Concrete Implementations:**
| Class | process() Returns | Purpose |
|-------|------------------|---------|
| LinkageAttack | DataFrame | Record pairs that match |
| MembershipInference | np.ndarray | Binary membership predictions (0/1) |
| AttributeInference | pd.Series | Inferred attribute values |
| DistanceToClosestRecord | np.ndarray | Distance scores |
| NearestNeighborDistanceRatio | np.ndarray | Distance ratio scores |
| AttackMetrics | dict | Evaluation metrics |

## Implementation Notes

### Mandatory Subclass Behavior

Every concrete attack class **must:**

1. **Implement preprocess_data()**
   - Combine TF-IDF (categorical) + StandardScaler (numeric)
   - Fit vectorizer/scaler on data1 only
   - Transform both data1 and data2 consistently

2. **Implement process()**
   - Accept data input (usually one or two DataFrames)
   - Return structured output (DataFrame, array, Series, or dict)
   - Validate inputs and raise appropriate errors

3. **Use PreprocessData as parent**
   - Don't inherit directly from AttackInitialization
   - Inherit from PreprocessData to get working implementations
   - PreprocessData provides TF-IDF + scaling out-of-the-box

### Error Handling

All implementations should catch and raise:
```python
from pamola_core.errors.exceptions import ValidationError, FieldNotFoundError

# Validate non-null inputs
if data1 is None or data2 is None:
    raise ValidationError("Input datasets cannot be None.")

# Validate required columns
if target_field not in data.columns:
    raise FieldNotFoundError(
        target_field,
        list(data.columns),
        dataset_name="target data"
    )
```

### Logging

Use PAMOLA logger in all implementations:
```python
import pamola_core.utils.logging as pamola_logging
logger = pamola_logging.getLogger(__name__)

logger.info(f"Attack execution: {description}")
logger.warning(f"Potential issue: {message}")
```

## Related Classes

### PreprocessData (Recommended Parent)

All concrete attacks inherit from PreprocessData, which provides:

```python
class PreprocessData(AttackInitialization):
    def preprocess_data(self, data1, data2, max_features=5000):
        """
        Fully implemented preprocessing combining:
        - TF-IDF vectorization for categorical columns
        - StandardScaler normalization for numeric columns
        - Alignment of features between both datasets
        """
```

**Example Usage:**
```python
from pamola_core.attacks import PreprocessData
import pandas as pd
import numpy as np

class MyAttack(PreprocessData):
    def process(self, data):
        # Use inherited preprocess_data()
        X1, X2 = self.preprocess_data(data['original'], data['anonymized'])
        # Now X1, X2 are numeric arrays ready for distance/similarity computation
        return self._compute_attack_metric(X1, X2)
```

## Summary

**AttackInitialization Key Points:**

1. **Abstract Interface** — Defines contract for all attack classes
2. **Two Methods** — preprocess_data() and process()
3. **Always Use PreprocessData** — Don't inherit AttackInitialization directly
4. **Preprocessing Pipeline** — TF-IDF + StandardScaler built-in
5. **Error Handling** — Use ValidationError and FieldNotFoundError
6. **Logging** — Integration with PAMOLA logging system

**When Implementing a New Attack:**
```python
from pamola_core.attacks.preprocess_data import PreprocessData
from pamola_core.errors.exceptions import ValidationError

class NewAttack(PreprocessData):
    def __init__(self):
        super().__init__()
        # Your initialization

    def process(self, data):
        # Implement core attack logic
        # Use self.preprocess_data() to get numeric arrays
        pass
```

See [LinkageAttack](./linkage_attack.md), [MembershipInference](./membership_inference.md), or [AttributeInference](./attribute_inference.md) for concrete examples.
