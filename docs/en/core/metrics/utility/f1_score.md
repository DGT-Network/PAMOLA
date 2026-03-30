# F1 Score Utility Metric
**Module:** pamola_core.metrics.utility.f1_score
**Version:** 1.0.0
**Status:** Stable
**Last Updated:** March 2025

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
Computes F1 Score for evaluating utility of anonymized datasets in classification tasks. F1 Score is the harmonic mean of precision and recall, providing a single metric that balances false positives and false negatives. Essential for assessing how well classification models perform on anonymized data versus original data.

## 2. Source Code Hierarchy
- pamola_core/metrics/utility/f1_score.py
  - class F1Score
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Original and anonymized classification datasets with labels
- Processing: Train or apply classifier, compute precision/recall, calculate F1
- Output: F1 Score (0-1 range), component scores (precision, recall)

## 4. Main Functionalities & Features
- Calculate F1 Score for binary and multi-class classification
- Support for different averaging modes (macro, micro, weighted)
- Comparison of F1 between original and anonymized data
- Component metrics: precision, recall, support
- Utility degradation assessment through F1 comparison

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize F1 Score metric |
| `calculate_metric(original_labels, predicted_labels, average)` | Compute F1 Score |

## 6. Usage Examples
```python
from pamola_core.metrics.utility.f1_score import F1Score
import pandas as pd
import numpy as np

# Classification results
original_labels = [0, 1, 1, 0, 1, 0]
predicted_labels_original = [0, 1, 1, 0, 1, 0]
predicted_labels_anon = [0, 1, 0, 0, 1, 0]

# Calculate F1 Score
f1 = F1Score()
result_original = f1.calculate_metric(original_labels, predicted_labels_original)
result_anon = f1.calculate_metric(original_labels, predicted_labels_anon)

print(result_original['f1_score'])  # F1 on original data
print(result_anon['f1_score'])      # F1 on anonymized data
print(result_anon['precision'])     # Precision component
print(result_anon['recall'])        # Recall component
```

## 7. Troubleshooting & Investigation Guide
- Ensure labels and predictions have same length
- Use consistent label encoding across original and anonymized data
- For imbalanced datasets, prefer weighted averaging
- Check for missing classes in predictions

## 8. Summary Analysis
- Standard metric for classification utility assessment
- Widely recognized and easy to interpret (0-1 scale)
- Production-ready for classification task evaluation

## 9. Challenges, Limitations & Enhancement Opportunities
- Sensitive to class imbalance; weighted averaging recommended
- Binary classification different from multi-class
- Threshold selection can impact F1 calculation
- Future: add ROC-AUC and confusion matrix visualization

## 10. Related Components & References
- Part of utility metrics suite in operations/utility_ops.py
- Complements classification.py for comprehensive classification evaluation
- Used for assessing utility of anonymized training data

## 11. Change Log & Contributors
- v1.0.0: F1 Score metric implementation (2025-03)
- Contributors: Metrics team
