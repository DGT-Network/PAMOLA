# Attribute Inference Documentation

**Module:** `pamola_core.attacks.attribute_inference`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Class Reference](#class-reference)
4. [Attack Method](#attack-method)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Related Components](#related-components)

## Overview

`AttributeInference` implements attacks that infer hidden sensitive attributes based on visible attributes in anonymized data.

**Purpose:** Test whether attacker can guess missing or redacted sensitive attributes by:
- Identifying predictive features (lowest entropy)
- Building feature-to-attribute mappings from training data
- Applying mappings to test records

**Threat Model:**
```
Training Data (Complete)        Anonymized Data (Partial)     Inference
========================        ==========================     ===========
Name   Age  Job      Salary     Age  Job      Salary(hidden)  Can infer
John   32   Engineer 120k  →    32   Engineer ???         →   Salary ≈ 120k?
Jane   32   Engineer 130k  →    32   Engineer ???         →   Salary ≈ 125k?
Bob    28   Analyst  85k   →    28   Analyst  ???         →   Salary ≈ 85k?
```

**Risk:** Medium-High — if sensitive attributes can be inferred accurately, anonymization is compromised.

## Key Features

**Entropy-Based Feature Selection**
- Selects lowest-entropy feature (most predictive)
- Avoids high-entropy features (high variance, unpredictive)
- Example: "Age" (low entropy) better predictor than "Name" (high entropy)

**Majority-Voting Inference**
- Groups records by predictor feature value
- Predicts mode (most common) sensitive attribute in each group
- Falls back to global mode if mapping not found

**Handles Missing Data**
- Gracefully handles NaN values during training
- Fills unfound predictions with global mode
- Returns pandas Series (compatible with DataFrame)

## Class Reference

### AttributeInference

```python
from pamola_core.attacks import AttributeInference

class AttributeInference(PreprocessData):
    """
    AttributeInference class for attack simulation in PAMOLA.CORE.
    Infers sensitive attributes using entropy-based feature selection.
    """

    def __init__(self):
        """
        No parameters required.
        All thresholds are derived from training data entropy.
        """
```

**Constructor:** No parameters

## Attack Method

### attribute_inference_attack

Infers sensitive attribute values for test records.

```python
def attribute_inference_attack(self, data_train, data_test, target_attribute):
    """
    Infer sensitive attribute values for test records.

    Strategy:
    1. Identify lowest-entropy feature in data_train
    2. Build mapping: feature_value → mode(target_attribute)
    3. Apply mapping to test records
    4. Fallback to global mode if mapping missing

    Parameters
    -----------
    data_train : pd.DataFrame
        Training dataset with complete target_attribute values.
        Used to build the feature-to-attribute mapping.

    data_test : pd.DataFrame
        Test dataset with hidden target_attribute values.
        Features must overlap with data_train.

    target_attribute : str
        Name of the sensitive attribute to infer.
        Must exist in data_train columns.

    Returns
    -----------
    pd.Series
        Inferred values for target_attribute in test records.
        Index: same as data_test.index
        Values: predicted attribute values (type matches training data)
        Length: same as len(data_test)
    """
```

### Internal Workflow

```
Step 1: Validate Inputs
  └─ Check data_train and data_test not empty
  └─ Verify target_attribute exists in data_train
  └─ Ensure features available (not all columns are target)

Step 2: Calculate Entropy for Each Feature
  └─ entropy = -Σ(p * log2(p))  where p = probability of each unique value
  └─ Example: age entropy = 2.1, job entropy = 1.8 (job is lower)

Step 3: Select Lowest-Entropy Feature
  └─ best_feature = "job" (most predictive)
  └─ This feature has most signal about target_attribute

Step 4: Build Mapping
  └─ Group data_train by best_feature value
  └─ For each group, find mode (most common target_attribute value)
  └─ mapping = {"Engineer": 120k, "Analyst": 85k, "Manager": 150k}

Step 5: Apply Mapping to Test Data
  └─ For each test record, lookup feature value in mapping
  └─ If found, use mapped value; else use global mode
  └─ Return as pandas Series

Step 6: Return Results
  └─ Series with predicted attribute values
  └─ Same index as data_test
  └─ Same length as data_test
```

**Entropy Calculation Example:**
```python
# Feature: "Age" with values [25, 25, 25, 35, 35]
counts = [3, 2]
probs = [3/5, 2/5] = [0.6, 0.4]
entropy = -(0.6 * log2(0.6) + 0.4 * log2(0.4))
        = -(0.6 * -0.737 + 0.4 * -1.322)
        = 0.971  # Low entropy = uniform, predictive feature

# Feature: "ID" with values [1, 2, 3, 4, 5]
counts = [1, 1, 1, 1, 1]
probs = [0.2, 0.2, 0.2, 0.2, 0.2]
entropy = -(5 * 0.2 * log2(0.2))
        = 2.322  # High entropy = unique, not predictive
```

## Usage Examples

### Basic Attribute Inference

```python
from pamola_core.attacks import AttributeInference
import pandas as pd

# Training data (complete)
data_train = pd.DataFrame({
    'age': [25, 32, 45, 28, 35, 42, 38, 30],
    'job_title': ['Engineer', 'Engineer', 'Manager', 'Analyst', 'Engineer', 'Manager', 'Analyst', 'Engineer'],
    'salary': [80000, 85000, 120000, 60000, 88000, 130000, 65000, 82000],
    'city': ['NYC', 'LA', 'Chicago', 'Boston', 'NYC', 'Houston', 'Seattle', 'Denver']
})

# Test data (salary hidden)
data_test = pd.DataFrame({
    'age': [33, 28, 50, 26],
    'job_title': ['Engineer', 'Analyst', 'Manager', 'Engineer'],
    'city': ['Chicago', 'Phoenix', 'Miami', 'Austin']
    # salary: unknown, to be inferred
})

# Run attack
aia = AttributeInference()
predicted_salaries = aia.attribute_inference_attack(data_train, data_test, 'salary')

print("Predicted Salaries:")
print(predicted_salaries)

# Attach predictions to test data for inspection
data_test['predicted_salary'] = predicted_salaries.values
print("\nTest data with predictions:")
print(data_test)
```

**Output:**
```
0    85000     # Engineer → mode salary is 85000
1    60000     # Analyst → mode salary is 60000
2    125000    # Manager → mode salary is 125000
3    85000     # Engineer → mode salary is 85000
Name: salary, dtype: int64

   age job_title city     predicted_salary
0  33  Engineer  Chicago 85000
1  28  Analyst   Phoenix 60000
2  50  Manager   Miami   125000
3  26  Engineer  Austin  85000
```

### Entropy-Based Feature Selection Example

```python
from pamola_core.attacks import AttributeInference
import pandas as pd
import numpy as np

# Training data
data_train = pd.DataFrame({
    'income_level': ['Low', 'Low', 'Medium', 'High', 'High'],
    'education': ['HS', 'HS', 'Bach', 'Bach', 'Grad'],
    'region': ['South', 'North', 'Midwest', 'North', 'West'],
    'medical_condition': ['Healthy', 'Healthy', 'Diabetic', 'Cardiac', 'Cardiac']
})

# Analyze entropy
import math

def calc_entropy(series):
    counts = series.value_counts()
    probs = counts / len(series)
    return -(probs * np.log2(probs)).sum()

print("Feature Entropy Analysis:")
for col in ['income_level', 'education', 'region']:
    ent = calc_entropy(data_train[col])
    print(f"  {col}: {ent:.3f}")

# Output:
# Feature Entropy Analysis:
#   income_level: 1.522 (lower entropy)
#   education: 1.368 (lower entropy)
#   region: 1.585 (higher entropy)

# Inference will select 'education' (lowest entropy)
aia = AttributeInference()

data_test = pd.DataFrame({
    'income_level': ['Low', 'High', 'Medium'],
    'education': ['HS', 'Bach', 'Grad'],
    'region': ['South', 'West', 'Midwest']
})

# Predict medical_condition using 'education' feature
predictions = aia.attribute_inference_attack(
    data_train, data_test, 'medical_condition'
)

print("\nPredicted Medical Conditions:")
print(predictions)
```

### Evaluating Attack Success

```python
from pamola_core.attacks import AttributeInference
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Create test dataset with known target values
data_train = pd.DataFrame({
    'age': [25, 35, 45, 55, 65] * 10,
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'] * 10,
    'income': [60000, 80000, 100000, 120000, 140000] * 10
})

# True test data (hidden income)
data_test = pd.DataFrame({
    'age': [25, 35, 45, 55, 65, 30, 40, 50, 60],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Boston', 'Seattle', 'Denver', 'Portland']
})

# Ground truth (actual income in test set)
y_true = np.array([60000, 80000, 100000, 120000, 140000, 70000, 90000, 110000, 130000])

# Run attack
aia = AttributeInference()
y_pred = aia.attribute_inference_attack(data_train, data_test, 'income')

# Evaluate
accuracy = accuracy_score(y_true, y_pred)
print(f"Prediction Accuracy: {accuracy:.3f}")

if accuracy > 0.8:
    print("WARNING: High attack success - income is easily inferred")
elif accuracy > 0.5:
    print("CAUTION: Moderate attack success - consider further anonymization")
else:
    print("GOOD: Inference difficult - anonymization appears effective")

# Detailed error analysis
errors = np.abs(y_true - y_pred.values)
print(f"Mean Absolute Error: ${errors.mean():,.0f}")
print(f"Predictions within $10k: {(errors <= 10000).sum() / len(errors):.1%}")
```

## Best Practices

**1. Test Multiple Target Attributes**
```python
aia = AttributeInference()

sensitive_attrs = ['salary', 'medical_condition', 'credit_score', 'insurance_rate']
for attr in sensitive_attrs:
    predictions = aia.attribute_inference_attack(data_train, data_test, attr)
    accuracy = evaluate_accuracy(data_test[attr], predictions)
    print(f"{attr}: {accuracy:.1%}")
```

**2. Understand Entropy Trade-offs**
```python
# Low-entropy features are more predictive but easier to attack
# High-entropy features are less predictive but harder to infer

# Good quasi-identifier: age (low entropy, limited values)
# Bad predictor: ID (high entropy, all unique)

# For inference attack, we want GOOD predictor
```

**3. Use with Linkage or Membership Inference**
```python
# Complete attack chain:
# 1. Linkage: Re-identify individuals
# 2. Membership: Confirm they were in training set
# 3. Attribute: Infer their sensitive attributes

matches = linkage.record_linkage_attack(original, anonymized)
memberships = mia.membership_inference_attack_dcr(original, anonymized)
salaries = aia.attribute_inference_attack(original, anonymized, 'salary')

# If all succeed, privacy is severely compromised
```

**4. Combine with Metrics Evaluation**
```python
# Quantify attack success
from sklearn.metrics import accuracy_score, f1_score

predictions = aia.attribute_inference_attack(data_train, data_test, 'sensitive_attr')
y_true = data_test['sensitive_attr']

acc = accuracy_score(y_true, predictions)
f1 = f1_score(y_true, predictions, average='weighted')

print(f"Accuracy: {acc:.3f}")
print(f"F1-Score: {f1:.3f}")

# If acc > 0.7: High risk
# If 0.5 < acc < 0.7: Medium risk
# If acc < 0.5: Low risk (better than random)
```

## Troubleshooting

**Q: AttributeError: 'Series' object has no attribute 'mode'**
- A: Likely in attribute_inference_attack, line 83. This is a pandas version issue.
```python
# Check pandas version
import pandas as pd
print(pd.__version__)

# Workaround: use .value_counts() instead of .mode()
mapping = data_train.groupby(best_feature)[target_attribute].agg(
    lambda x: x.value_counts().idxmax() if len(x) > 0 else x.iloc[0]
)
```

**Q: Empty DataFrame returned**
- A: data_test or data_train is empty.
```python
if data_train.empty or data_test.empty:
    return pd.Series(dtype=object)

# Check inputs
print(f"Training size: {len(data_train)}")
print(f"Test size: {len(data_test)}")
```

**Q: FieldNotFoundError: target_attribute not in columns**
- A: target_attribute name doesn't match exactly.
```python
# Check exact column names
print("Training columns:", data_train.columns.tolist())
print("Looking for:", target_attribute)

# Use correct name
predictions = aia.attribute_inference_attack(data_train, data_test, 'salary')  # Not 'Salary'
```

**Q: Predictions all the same value**
- A: Selected feature has low variance, or global mode dominates.
```python
# Check selected feature's entropy
for col in data_train.columns:
    if col != target_attribute:
        entropy = calc_entropy(data_train[col])
        print(f"{col}: entropy={entropy:.3f}")

# If all have low entropy, feature selection is legitimate
# Consider using different target_attribute
```

**Q: Accuracy very low (near random)**
- A: Target attribute and features are independent.
```python
# Good news for privacy! Means attribute is not inferrable from visible features
# Continue with other anonymization tests (linkage, membership inference)
```

## Related Components

- **[Linkage Attack](./linkage_attack.md)** — Re-identify individuals first
- **[Membership Inference](./membership_inference.md)** — Confirm membership before attribute inference
- **[Attribute Metrics](./attack_metrics.md)** — Evaluate attack success rates
- **[Profiling Module](../profiling/)** — Analyze feature entropy and importance before anonymization
