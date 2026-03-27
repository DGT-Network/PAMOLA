# KL Divergence Metric
**Module:** pamola_core.metrics.fidelity.distribution.kl_divergence  
**Version:** 1.0.0  
**Status:** Stable  
**Last Updated:** July 23, 2025

## Table of Contents
1. [Module Overview](#1-module-overview)
2. [Source Code Hierarchy](#2-source-code-hierarchy)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Main Functionalities & Features](#4-main-functionalities--features)
5. [API Reference & Key Methods](#5-api-reference--key-methods)
6. [Usage Examples](#6-usage-examples)
7. [Troubleshooting & Investigation Guide](#7-troubleshooting--investigation-guide)
8. [Summary Analysis](#8-summary-analysis)
9. [Challenges, Limitations & Enhancement Opportunities](#9-challenges-limitations--enhancement-opportunities)
10. [Related Components & References](#10-related-components--references)
11. [Change Log & Contributors](#11-change-log--contributors)

## 1. Module Overview
KL Divergence (Kullback-Leibler Divergence) measures how one probability distribution diverges from a second, expected probability distribution. Used to compare similarity between distributions in information theory and statistics.

## 2. Source Code Hierarchy
- pamola_core/metrics/fidelity/distribution/kl_divergence.py
  - class KLDivergence
    - __init__
    - calculate_metric
    - _calculate_kl_from_dicts
    - _prepare_distributions
    - _safe_entropy
    - _jensen_shannon_distance
    - _calculate_effect_size
    - _calculate_confidence_interval
    - _is_statistically_significant
    - _get_normalization_description
    - _interpret_kl

## 3. Architecture & Data Flow
- Used by fidelity metric operations and wrappers
- Static method for direct calculation

## 4. Main Functionalities & Features
- Computes KL divergence between two probability distributions
- Handles normalization if needed

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__(normalize_features)` | Constructor |
| `calculate_metric(original_df, transformed_df)` | Computes KL divergence between distributions |
| `_calculate_kl_from_dicts(...)` | Calculate KL from value dictionaries |
| `_prepare_distributions(...)` | Prepares and normalizes distributions |
| `_safe_entropy(p, q)` | Safely computes entropy |
| `_jensen_shannon_distance(p, q)` | Computes Jensen-Shannon distance |
| `_calculate_effect_size(kl_value)` | Interprets effect size |
| `_calculate_confidence_interval(...)` | Calculates confidence intervals |
| `_is_statistically_significant(kl_value)` | Tests statistical significance |
| `_get_normalization_description()` | Gets normalization information |
| `_interpret_kl(kl_value)` | Interprets KL value as similarity |

## 6. Usage Examples
```python
import numpy as np
from pamola_core.metrics.fidelity.distribution.kl_divergence import KLDivergence
p = np.array([0.4, 0.6])
q = np.array([0.5, 0.5])
kl = KLDivergence.kl_divergence(p, q)
```

## 7. Troubleshooting & Investigation Guide
- Ensure input arrays are valid probability distributions (sum to 1, non-negative)
- Check for shape mismatches

## 8. Summary Analysis
- Efficient, robust implementation for distribution comparison
- Fully covered by unit tests

## 9. Challenges, Limitations & Enhancement Opportunities
- Only valid for probability distributions
- Future: add support for batch/multivariate KL

## 10. Related Components & References
- pamola_core/metrics/fidelity/distribution/ks_test.py

## 11. Change Log & Contributors
- v4.0.0: Initial metric release (2025-07-22)
- Contributors: Metrics team

---

*This documentation is auto-generated as part of the PAMOLA Core Metrics package coverage process.*
