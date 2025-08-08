# Safe Instantiation Utility
**Module:** pamola_core.metrics.commons.safe_instantiate  
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

# Safe Instantiation Utility (`safe_instantiate.py`)

## 1. Module Overview
Utility for safely instantiating metric classes by filtering out invalid constructor parameters. Ensures robust metric object creation from dynamic/user-supplied configs.

## 2. Source Code Hierarchy
- pamola_core/metrics/commons/safe_instantiate.py
  - safe_instantiate

## 3. Architecture & Data Flow
- Used by operation wrappers/factories to instantiate metric classes from config

## 4. Main Functionalities & Features
- Filters out invalid constructor parameters
- Prevents runtime errors from unexpected arguments

## 5. API Reference & Key Methods
| Function | Description |
|----------|-------------|
| `safe_instantiate(metric_class, params)` | Instantiates class with valid args |

## 6. Usage Examples
```python
class MyMetric:
    def __init__(self, a, b=2):
        self.a = a
        self.b = b
obj = safe_instantiate(MyMetric, {"a": 1, "b": 3, "c": 99})
# obj.a == 1, obj.b == 3, 'c' is ignored
```

## 7. Troubleshooting & Investigation Guide
- Ensure class constructor matches provided params
- Unexpected params are ignored, not an error

## 8. Summary Analysis
- Simple, robust utility for safe instantiation

## 9. Challenges, Limitations & Enhancement Opportunities
- Only works with standard Python constructors
- Future: add support for advanced class patterns

## 10. Related Components & References
- Used by all metrics operation wrappers

## 11. Change Log & Contributors
- v4.0.0: Initial utility release (2025-07-22)
- Contributors: Metrics team
