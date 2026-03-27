# DifferentialPrivacyProcessor Documentation

**Module:** `pamola_core.privacy_models.differential_privacy.calculation`
**Class:** `DifferentialPrivacyProcessor`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Initialization](#initialization)
4. [Core Methods](#core-methods)
5. [Noise Mechanisms](#noise-mechanisms)
6. [Epsilon Parameter](#epsilon-parameter)
7. [Usage Examples](#usage-examples)
8. [Mathematical Foundation](#mathematical-foundation)
9. [Best Practices](#best-practices)
10. [Related Components](#related-components)

## Overview

`DifferentialPrivacyProcessor` implements epsilon-Differential Privacy using Laplace and Gaussian noise mechanisms. It adds calibrated noise to ensure output indistinguishability, providing formal privacy guarantees.

**Key Guarantee:** Query outputs are statistically indistinguishable whether or not any individual is in the dataset.

**Location:** `pamola_core/privacy_models/differential_privacy/calculation.py`

## Key Features

- **Multiple Mechanisms:** Laplace and Gaussian noise addition
- **Epsilon-Based Privacy Budget:** Configurable privacy level
- **Sensitivity Parameter:** Query-specific sensitivity control
- **Formal Privacy Proof:** Mathematical privacy guarantees
- **Data Evaluation:** Pre-transformation privacy assessment
- **Transformation Support:** Apply differential privacy to datasets

## Initialization

### Constructor Signature

```python
def __init__(
    self,
    epsilon: float,
    sensitivity: float,
    mechanism: str = "laplace"
):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon` | float | — | Privacy budget (smaller = more private, less accurate) |
| `sensitivity` | float | — | Query sensitivity (max change in output from 1-record change) |
| `mechanism` | str | "laplace" | "laplace" or "gaussian" |

### Epsilon Guidelines

| Epsilon | Privacy Level | Use Case |
|---------|---|----------|
| 0.1-0.5 | Very High | Research, strict requirements |
| 0.5-1.0 | High | Sensitive data, GDPR-aligned |
| 1.0-2.0 | Moderate | Balanced privacy-utility |
| 2.0-5.0 | Low | Utility-focused applications |
| > 5.0 | Minimal | Non-sensitive data |

### Usage Examples

```python
# High privacy (epsilon=0.5)
processor = DifferentialPrivacyProcessor(
    epsilon=0.5,
    sensitivity=1.0,
    mechanism='laplace'
)

# Balanced privacy (epsilon=1.0)
processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=10.0,
    mechanism='gaussian'
)

# Utility-focused (epsilon=3.0)
processor = DifferentialPrivacyProcessor(
    epsilon=3.0,
    sensitivity=5.0,
    mechanism='laplace'
)
```

## Core Methods

### evaluate_privacy(data, quasi_identifiers, **kwargs)

**Purpose:** Evaluate differential privacy guarantees on dataset.

**Signature:**
```python
def evaluate_privacy(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    **kwargs
) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'epsilon': float,
    'mechanism': str,
    'sensitivity': float,
    'privacy_guarantee': str,
    'original_means': {...},          # Mean per numeric column
    'dp_means': {...},                # Mean after noise addition
    'mean_differences': {...},        # Difference between original and DP means
    'noise_scale': float
}
```

**Example:**
```python
processor = DifferentialPrivacyProcessor(epsilon=0.8, sensitivity=1.0)

evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code']
)

print(f"Privacy Epsilon: {evaluation['epsilon']}")
print(f"Mechanism: {evaluation['mechanism']}")
print(f"Privacy guarantee: {evaluation['privacy_guarantee']}")
```

### apply_model(data, quasi_identifiers, suppression=True, **kwargs)

**Purpose:** Apply differential privacy to dataset by adding noise.

**Signature:**
```python
def apply_model(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    suppression: bool = True,
    **kwargs
) -> pd.DataFrame:
```

**Returns:** DataFrame with noise added to numeric columns

**Note:** `suppression` parameter not applicable for differential privacy (noise is always added)

**Example:**
```python
processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=5.0)

# Apply differential privacy
dp_data = processor.apply_model(
    df,
    quasi_identifiers=['age', 'zip_code']
)

# Result has noise added to numeric columns
print(f"Original salary range: {df['salary'].min()}-{df['salary'].max()}")
print(f"DP salary range: {dp_data['salary'].min()}-{dp_data['salary'].max()}")
```

### add_noise(value)

**Purpose:** Add noise to a single value based on mechanism and epsilon.

**Signature:**
```python
def add_noise(self, value: float) -> float:
```

**Example:**
```python
processor = DifferentialPrivacyProcessor(epsilon=0.8, sensitivity=1.0)

# Add noise to individual values
noisy_age = processor.add_noise(30.0)
noisy_salary = processor.add_noise(50000.0)
```

### process(data)

**Purpose:** Generic processing interface (applies differential privacy).

**Signature:**
```python
def process(self, data):
```

## Noise Mechanisms

### 1. Laplace Mechanism

**Formula:** noise = Laplace(0, sensitivity/epsilon)

**Pros:**
- Provides (epsilon)-differential privacy
- Lower computation cost
- Good for aggregate queries

**Cons:**
- Larger outlier noise values
- Less smooth noise distribution

**Used when:** Simpler privacy needs, computational efficiency important

```python
processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=1.0,
    mechanism='laplace'
)
```

### 2. Gaussian Mechanism

**Formula:** noise = Gaussian(0, sqrt(2*ln(1.25))*sensitivity/epsilon)

**Pros:**
- Smooth, natural-looking noise
- Better for composition of queries
- Gaussian Differential Privacy (weaker but more practical)

**Cons:**
- Slightly higher computation
- Provides (epsilon, delta)-DP, not pure epsilon-DP

**Used when:** Multiple queries on same data, smooth noise preferred

```python
processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=1.0,
    mechanism='gaussian'
)
```

## Epsilon Parameter

### Privacy Budget Concept

Epsilon represents the privacy budget—lower epsilon = stronger privacy:

- **epsilon = 0.5:** Very strong privacy, significant noise
- **epsilon = 1.0:** Good privacy-utility balance
- **epsilon = 5.0:** Weak privacy, minimal noise

### Sensitivity

Sensitivity measures how much a single record can change query output:

```python
# For average age query
sensitivity = (max_age - min_age) / n_records  # How much 1 record affects average

processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=sensitivity
)
```

## Usage Examples

### Example 1: Basic Differential Privacy

```python
from pamola_core.privacy_models import DifferentialPrivacyProcessor
import pandas as pd

df = pd.read_csv('salary_data.csv')

processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=10000.0,
    mechanism='laplace'
)

# Evaluate privacy
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'department']
)

print(f"Epsilon: {evaluation['epsilon']}")
print(f"Mechanism: {evaluation['mechanism']}")

# Apply differential privacy
dp_data = processor.apply_model(
    df,
    quasi_identifiers=['age', 'department']
)

# Compare means
print("\nOriginal means:")
print(evaluation['original_means'])
print("\nDP means:")
print(evaluation['dp_means'])
```

### Example 2: Privacy-Utility Trade-off

```python
# Try different epsilon values
for epsilon in [0.5, 1.0, 2.0, 5.0]:
    processor = DifferentialPrivacyProcessor(
        epsilon=epsilon,
        sensitivity=5000.0,
        mechanism='gaussian'
    )

    evaluation = processor.evaluate_privacy(df, ['age', 'department'])

    # Check mean difference (utility loss)
    mean_diff = sum(abs(evaluation['mean_differences'].values())) / len(evaluation['mean_differences'])

    print(f"epsilon={epsilon}: mean_difference={mean_diff:.2f}")
```

### Example 3: Query-Specific Sensitivity

```python
# Count query: sensitivity = 1 (adding/removing person changes count by 1)
count_processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=1.0,
    mechanism='laplace'
)

# Sum query: sensitivity = max_value
sum_processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=1000000.0,  # Max salary
    mechanism='laplace'
)

# Average query: sensitivity = (max-min)/n
avg_processor = DifferentialPrivacyProcessor(
    epsilon=1.0,
    sensitivity=(100000-20000)/len(df),
    mechanism='gaussian'
)
```

### Example 4: Multi-Query Composition

```python
# For multiple queries on same data, use smaller epsilon per query
total_epsilon = 1.0
num_queries = 3
epsilon_per_query = total_epsilon / num_queries

processors = [
    DifferentialPrivacyProcessor(
        epsilon=epsilon_per_query,
        sensitivity=sensitivity,
        mechanism='gaussian'
    )
    for sensitivity in [1000, 5000, 10000]
]
```

## Mathematical Foundation

### Epsilon-Differential Privacy Definition

A mechanism M provides epsilon-differential privacy if for adjacent datasets D and D' differing by one record:

```
P(M(D) ∈ S) ≤ e^ε * P(M(D') ∈ S)
```

For all output sets S, meaning output distributions are similar regardless of individual presence.

### Laplace Mechanism

For query f(D) with sensitivity Δf:

```
M(D) = f(D) + Laplace(0, Δf/ε)
```

Provides pure ε-differential privacy.

### Gaussian Mechanism

For query f(D) with sensitivity Δf:

```
M(D) = f(D) + Gaussian(0, σ²) where σ = sqrt(2*ln(1.25))*Δf/ε
```

Provides (ε, δ)-differential privacy.

## Best Practices

1. **Choose Appropriate Epsilon:**
   ```python
   # For sensitive data (healthcare, finance)
   epsilon = 0.5  # to 1.0

   # For balanced requirements
   epsilon = 1.0  # to 2.0

   # For utility-focused applications
   epsilon = 3.0  # to 5.0
   ```

2. **Calculate Sensitivity Correctly:**
   ```python
   # For count queries
   sensitivity = 1

   # For sum queries
   sensitivity = max(abs(column.min()), abs(column.max()))

   # For average queries
   sensitivity = (column.max() - column.min()) / len(df)
   ```

3. **Consider Query Composition:**
   ```python
   # For multiple queries, compose epsilon budgets
   total_epsilon = 1.0
   num_queries = 5
   epsilon_per_query = total_epsilon / num_queries

   processor = DifferentialPrivacyProcessor(
       epsilon=epsilon_per_query,
       sensitivity=sensitivity
   )
   ```

4. **Test Mechanism Choice:**
   ```python
   # Compare Laplace vs Gaussian
   for mechanism in ['laplace', 'gaussian']:
       processor = DifferentialPrivacyProcessor(
           epsilon=1.0,
           sensitivity=sensitivity,
           mechanism=mechanism
       )
       eval_result = processor.evaluate_privacy(df, quasi_ids)
   ```

## Troubleshooting

### Issue: Too Much Noise

**Symptom:** Data statistics unreliable

**Solutions:**
- Increase epsilon (weaker privacy)
- Increase sensitivity (larger scale for noise)
- Aggregate before adding noise
- Use larger dataset

### Issue: Weak Privacy Guarantees

**Symptom:** Need stronger privacy

**Solutions:**
- Decrease epsilon (more noise)
- Use Gaussian mechanism
- Compose multiple queries carefully
- Consider k-Anonymity or l-Diversity instead

## Related Components

- [BasePrivacyModelProcessor](../base_privacy_model.md)
- [KAnonymityProcessor](../k_anonymity/k_anonymity_processor.md)
- [LDiversityCalculator](../l_diversity/l_diversity_calculator.md)

## Summary

`DifferentialPrivacyProcessor` provides mathematically-proven privacy guarantees by adding calibrated noise. Use it for:
- Research datasets requiring strong privacy proofs
- Query responses on sensitive data
- Federated learning scenarios
- Compliance with strict privacy regulations

Start with epsilon=1.0 for balanced privacy-utility, adjusting based on specific requirements. Combine with other models for defense-in-depth privacy strategies.
