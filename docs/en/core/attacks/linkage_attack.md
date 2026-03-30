# Linkage Attack Documentation

**Module:** `pamola_core.attacks.linkage_attack`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Class Reference](#class-reference)
4. [Attack Methods](#attack-methods)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Related Components](#related-components)

## Overview

`LinkageAttack` implements record linkage techniques to test whether an attacker can re-identify individuals by matching records between original and anonymized datasets.

**Purpose:** Simulate re-identification attacks where:
- Attacker has external dataset with identifiable information
- Attacker tries to match records using quasi-identifiers
- Goal: Determine if anonymization was effective

**Threat Model:**
```
Original Data          Quasi-Identifiers       Anonymized Data
[John, 35, NYC]  --→  [age, city]        <--  [35, NYC, Disease=X]
[Jane, 28, LA]   ---→  Match?              <--  [28, LA, Disease=Y]
```

If records match, attacker reveals the disease associated with each person.

## Key Features

| Method | Technique | Complexity | Best For |
|--------|-----------|-----------|----------|
| **Record Linkage** | Exact key matching | Low | Deterministic attributes (ID, exact name) |
| **Probabilistic Linkage** | Fellegi-Sunter fuzzy matching | Medium | Noisy attributes (misspelled names, typos) |
| **Cluster Vector Linkage** | PCA + cosine similarity | High | High-dimensional, mixed data |

## Class Reference

### LinkageAttack

```python
from pamola_core.attacks import LinkageAttack

class LinkageAttack(PreprocessData):
    """
    LinkageAttack class for attack simulation in PAMOLA.CORE.
    Implements three record linkage strategies.
    """

    def __init__(self, fs_threshold=None, n_components=2):
        """
        Parameters
        -----------
        fs_threshold: float, optional (default=None)
            Threshold for Fellegi-Sunter similarity score (0.0-1.0).
            Higher threshold = stricter matching. If None, defaults to 0.85.

        n_components: int, optional (default=2)
            Number of components for PCA dimensionality reduction
            in cluster vector linkage. Default=2.
        """
```

### Constructor Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `fs_threshold` | float | None (→0.85) | Fellegi-Sunter matching confidence threshold |
| `n_components` | int | 2 | PCA dimensions for CVPLA |

## Attack Methods

### 1. record_linkage_attack

Exact record matching by comparing values of common columns.

```python
def record_linkage_attack(self, data1, data2, linkage_keys):
    """
    Direct comparison of common columns to find matching record pairs.

    Parameters
    -----------
    data1 : pd.DataFrame
        Original dataset (reference set).

    data2 : pd.DataFrame
        Anonymized dataset (target set).

    linkage_keys : list of str, optional
        Column names to compare for matching.
        If None, automatically use common columns between datasets.

    Returns
    -----------
    pd.DataFrame
        Matched record pairs with structure:
        - Original columns suffixed _data1
        - Anonymized columns suffixed _data2
        - Index: pair ID (0, 1, 2, ...)
    """
```

**Output Columns:**
```
linkage_keys | column1_data1 | column1_data2 | ...
---------------------------------------------------
val1         | val2          | val2_anon     | ...
val1         | val3          | val3_anon     | ...
```

**Example:**
```python
from pamola_core.attacks import LinkageAttack
import pandas as pd

original = pd.DataFrame({
    'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'age': [35, 28, 42],
    'city': ['NYC', 'LA', 'Chicago'],
    'salary': [80000, 90000, 75000]
})

anonymized = pd.DataFrame({
    'name': ['***', '***', '***'],      # Masked
    'age': [35, 28, 42],                 # Not masked
    'city': ['NYC', 'LA', 'Chicago'],    # Not masked
    'salary_binned': ['60-90k', '60-90k', '60-90k']  # Generalized
})

linkage = LinkageAttack()
matches = linkage.record_linkage_attack(original, anonymized, linkage_keys=['age', 'city'])

print(f"Found {len(matches)} matching records")
print(matches.head())
```

**Output:**
```
   age city name_data1 name_data2 ...
0  35  NYC John Doe   ***
1  28  LA  Jane Smith ***
2  42  Chicago Bob Johnson ***
```

**Raises:**
- `ValidationError` — If data1 or data2 is None
- Warning: "Multiple matches detected" — If 1-to-many or many-to-many matches occur

### 2. probabilistic_linkage_attack

Fuzzy matching using Fellegi-Sunter model (handles typos, variations).

```python
def probabilistic_linkage_attack(self, data1, data2, keys=None):
    """
    Fellegi-Sunter probabilistic record linkage using string similarity.

    Parameters
    -----------
    data1 : pd.DataFrame
        Original dataset.

    data2 : pd.DataFrame
        Anonymized dataset.

    keys : list of str, optional
        Column names for comparison. If None, auto-detect common columns.

    Returns
    -----------
    pd.DataFrame
        Matched pairs with columns:
        - level_0 (tuple): (index_data1, index_data2)
        - Other columns: comparison scores for each key
        - similarity_score: Aggregated score (0.0-1.0)

        Only returns pairs with similarity_score >= fs_threshold.
    """
```

**Output Structure:**
```
level_0        | name | age  | city | ... | similarity_score
(0, 0)         | 0.85 | 1.0  | 1.0  | ... | 0.95
(1, 1)         | 0.80 | 1.0  | 1.0  | ... | 0.93
```

**Features:**
- Jaro-Winkler string similarity (handles typos, misspellings)
- Automatic threshold determination (0.85 if not specified)
- Blocking optimization (faster, skips unlikely pairs)

**Example with Typos:**
```python
original = pd.DataFrame({
    'name': ['John Doe', 'Jane Smith'],
    'email': ['john@example.com', 'jane@example.com'],
    'dob': ['1990-01-15', '1995-06-20']
})

anonymized = pd.DataFrame({
    'name': ['Jon Doe', 'Jane Smyth'],  # Typos
    'email': ['jon@example.com', 'jane@example.com'],
    'dob': ['1990-01-15', '1995-06-20']  # Exact match
})

linkage = LinkageAttack(fs_threshold=0.85)
matches = linkage.probabilistic_linkage_attack(original, anonymized, keys=['name', 'email', 'dob'])

print(f"Matches found: {len(matches)}")
print(matches[['name', 'email', 'dob', 'similarity_score']])
```

**Output:**
```
   name email dob similarity_score
0  0.90 1.0   1.0   0.93
1  0.88 1.0   1.0   0.96
```

### 3. cluster_vector_linkage_attack

High-dimensional linkage using PCA dimensionality reduction and cosine similarity (CVPLA).

```python
def cluster_vector_linkage_attack(
    self,
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    similarity_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Cluster-Vector Probabilistic Linkage Attack (CVPLA).
    Uses latent vector representation for matching complex datasets.

    Parameters
    ----------
    data1 : pd.DataFrame
        First dataset (target/reference).

    data2 : pd.DataFrame
        Second dataset (external/query).

    similarity_threshold : float, optional (default=0.8)
        Minimum cosine similarity to consider a match (0.0-1.0).
        Higher value = stricter matching.

    Returns
    -------
    pd.DataFrame
        Matched records with columns:
        - ID_DF1: index in data1
        - ID_DF2: index in data2
        - Score: cosine similarity (0.0-1.0)

        Rows: Only matches with Score >= similarity_threshold, one-to-one.
    """
```

**Processing Pipeline:**
```
TF-IDF + Scale  →  Normalize  →  PCA (2D)  →  Cosine Sim  →  Filter + Resolve Dups
```

**Output Structure:**
```
ID_DF1  ID_DF2  Score
0       0       0.95
1       1       0.88
2       3       0.82
```

**Features:**
- Handles mixed numeric/categorical data (TF-IDF + scaling)
- Dimensionality reduction (PCA) for efficiency and noise reduction
- Resolves duplicate matches (keeps highest-scoring pair)
- Returns one-to-one mapping only

**Example with Complex Data:**
```python
import numpy as np

original = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Bob Wilson'],
    'job_title': ['Software Engineer', 'Product Manager', 'Data Scientist'],
    'salary': [120000, 130000, 110000],
    'city': ['San Francisco', 'New York', 'Boston']
})

anonymized = pd.DataFrame({
    'name': ['***', '***', '***'],                      # Masked
    'job_title': ['***', '***', '***'],                # Masked
    'salary_range': ['100-150k', '100-150k', '100-150k'],  # Generalized
    'region': ['West', 'Northeast', 'Northeast']       # Generalized
})

linkage = LinkageAttack(n_components=2)
matches = linkage.cluster_vector_linkage_attack(
    original, anonymized, similarity_threshold=0.75
)

print(f"CVPLA Matches: {len(matches)}")
print(matches)
```

**Output:**
```
   ID_DF1  ID_DF2  Score
0  0       0       0.92
1  1       1       0.88
2  2       2       0.79
```

## Usage Examples

### Complete Re-identification Test

```python
from pamola_core.attacks import LinkageAttack
import pandas as pd

# Simulate original and anonymized datasets
original = pd.DataFrame({
    'person_id': [1, 2, 3, 4, 5],
    'name': ['Alice Brown', 'Bob Jones', 'Charlie Davis', 'Diana Lee', 'Eve White'],
    'age': [32, 45, 28, 55, 38],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
    'income': [75000, 95000, 60000, 120000, 85000]
})

anonymized = pd.DataFrame({
    'age': [32, 45, 28, 55, 38],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
    'income_bin': ['60-80k', '80-100k', '60-80k', '100-150k', '80-100k'],
    'sensitive_attribute': ['A', 'B', 'C', 'D', 'E']
})

# Test all three methods
linkage = LinkageAttack(fs_threshold=0.8, n_components=2)

# Method 1: Exact matching
exact_matches = linkage.record_linkage_attack(original, anonymized, ['age', 'city'])
print(f"Exact matches: {len(exact_matches)}")

# Method 2: Fuzzy matching
fuzzy_matches = linkage.probabilistic_linkage_attack(original, anonymized, ['age', 'city'])
print(f"Fuzzy matches: {len(fuzzy_matches)}")

# Method 3: Vector linkage
vector_matches = linkage.cluster_vector_linkage_attack(original, anonymized, 0.80)
print(f"Vector matches: {len(vector_matches)}")

# Risk assessment
if len(exact_matches) > 0 or len(vector_matches) > 0:
    print("WARNING: High re-identification risk!")
```

## Best Practices

**1. Choose Method Based on Data Quality**
- Clean, exact data → `record_linkage_attack()`
- Noisy, misspelled data → `probabilistic_linkage_attack()`
- High-dimensional mixed data → `cluster_vector_linkage_attack()`

**2. Select Quasi-Identifiers Carefully**
- Include only attributes that could link to external data
- Use age ranges, not exact birth dates
- Combine multiple weak identifiers

**3. Test Before Anonymization**
```python
# Establish baseline - all should match (identical data)
perfect_matches = linkage.record_linkage_attack(data, data, keys)
assert len(perfect_matches) == len(data)  # Should be 100% match
```

**4. Adjust Thresholds Appropriately**
```python
# Strict matching (fewer false positives)
linkage = LinkageAttack(fs_threshold=0.90, n_components=3)

# Lenient matching (finds more potential matches)
linkage = LinkageAttack(fs_threshold=0.70, n_components=2)
```

**5. Combine with Other Attacks**
```python
# Use linkage to establish baseline, then membership inference
linkage_results = linkage.record_linkage_attack(...)
if len(linkage_results) > threshold:
    print("Proceed to MembershipInference for additional testing")
```

## Troubleshooting

**Q: No matches found when expecting many**
- A: Linkage keys too specific. Use broader quasi-identifiers.
```python
# Too strict
matches = linkage.record_linkage_attack(data1, data2, ['exact_name', 'exact_dob'])

# Better
matches = linkage.record_linkage_attack(data1, data2, ['age_range', 'city'])
```

**Q: Too many false positive matches (noise)**
- A: Increase fs_threshold (fuzzy) or similarity_threshold (CVPLA).
```python
# Stricter matching
linkage = LinkageAttack(fs_threshold=0.92)  # Was 0.85
matches = linkage.probabilistic_linkage_attack(data1, data2)
```

**Q: Runtime too slow with large datasets**
- A: Use probabilistic_linkage (blocking reduces candidates).
```python
# record_linkage compares all pairs: O(n*m)
exact = linkage.record_linkage_attack(data1, data2, keys)  # Slow for 100k+ rows

# probabilistic uses blocking: O(n*log(m))
fuzzy = linkage.probabilistic_linkage_attack(data1, data2, keys)  # Faster
```

**Q: Error: "ValidationError: Input datasets cannot be None"**
- A: Check that data1 and data2 are valid DataFrames.
```python
if data1 is None or data2 is None:
    raise ValidationError("Load datasets first")
```

## Related Components

- **[MembershipInference](./membership_inference.md)** — Test if records were in training set
- **[DistanceToClosestRecord](./distance_to_closest_record.md)** — Measure dataset dissimilarity
- **[Anonymization Module](../anonymization/)** — Apply protections before linkage testing
- **[Profiling Module](../profiling/)** — Analyze data before anonymization
