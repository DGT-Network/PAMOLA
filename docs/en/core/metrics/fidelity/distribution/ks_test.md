# Kolmogorov-Smirnov Test Metric
**Module:** pamola_core.metrics.fidelity.distribution.ks_test  
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
The Kolmogorov-Smirnov (KS) test is a nonparametric test that compares the distributions of two samples. Quantifies the distance between empirical distribution functions and provides a p-value for the null hypothesis.

## 2. Source Code Hierarchy
- pamola_core/metrics/fidelity/distribution/ks_test.py
  - class KolmogorovSmirnovTest
    - ks_test (static)

## 3. Architecture & Data Flow
- Used by fidelity metric operations and wrappers
- Static method for direct calculation

## 4. Main Functionalities & Features
- Computes KS statistic and p-value for two samples
- Handles normalization if needed

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `ks_test(data1, data2)` | Computes KS statistic and p-value |

## 6. Usage Examples
```python
import numpy as np
from pamola_core.metrics.fidelity.distribution.ks_test import KolmogorovSmirnovTest
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(0, 1, 100)
stat, pval = KolmogorovSmirnovTest.ks_test(data1, data2)
```

## 7. Troubleshooting & Investigation Guide
- Ensure input arrays are valid and have compatible shapes
- Check for shape mismatches

## 8. Summary Analysis
- Efficient, robust implementation for distribution comparison
- Fully covered by unit tests

## 9. Challenges, Limitations & Enhancement Opportunities
- Only valid for 1D arrays
- Future: add support for batch/multivariate KS

## 10. Related Components & References
- pamola_core/metrics/fidelity/distribution/kl_divergence.py

## 11. Change Log & Contributors
- v4.0.0: Initial metric release (2025-07-22)
- Contributors: Metrics team

---

*This documentation is auto-generated as part of the PAMOLA Core Metrics package coverage process.*
