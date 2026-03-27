# Privacy Models Overview

**Module:** `pamola_core.privacy_models`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Supported Privacy Models](#supported-privacy-models)
3. [Model Comparison](#model-comparison)
4. [Architecture](#architecture)
5. [Quick Start](#quick-start)
6. [Usage Patterns](#usage-patterns)
7. [Integration Guide](#integration-guide)
8. [Performance Considerations](#performance-considerations)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The `privacy_models` module provides a comprehensive suite of privacy-preserving anonymization models. It implements four major privacy models: k-Anonymity, l-Diversity, t-Closeness, and Differential Privacy. Each model offers different privacy guarantees and utility trade-offs suitable for various use cases.

All models inherit from `BasePrivacyModelProcessor`, which defines a consistent interface for evaluation and application.

## Supported Privacy Models

### 1. k-Anonymity
Ensures that each record in a dataset is **indistinguishable from at least k-1 others** based on quasi-identifiers.

**Key Features:**
- Group-based indistinguishability guarantee
- Quasi-identifier suppression/masking
- Adaptive k-levels per group
- Re-identification risk calculation
- Progress tracking and parallel processing

**Best For:** Basic privacy requirements, regulatory compliance, moderate datasets

### 2. l-Diversity
Extends k-Anonymity by ensuring **diverse sensitive attribute values** within each group.

**Key Features:**
- Multiple diversity types: distinct, entropy, recursive (c,l)-diversity
- Comprehensive risk assessment with caching
- Attribute disclosure risk analysis
- Flexible anonymization strategies (suppression, masking)
- Advanced visualization and reporting

**Best For:** Sensitive healthcare/financial data, attribute disclosure protection, detailed compliance reporting

### 3. t-Closeness
Maintains **statistical similarity** between group distributions and the overall dataset distribution.

**Key Features:**
- Distribution-based privacy guarantee
- Wasserstein distance computation
- Sensitive attribute distribution matching
- Complements k-Anonymity and l-Diversity

**Best For:** Preserving statistical properties, data utility sensitive applications, publishing aggregated statistics

### 4. Differential Privacy
Adds **calibrated noise** to ensure output indistinguishability via Laplace or Gaussian mechanisms.

**Key Features:**
- Epsilon-based privacy budgets
- Laplace and Gaussian noise mechanisms
- Query-level privacy guarantees
- Sensitivity parameter customization

**Best For:** Strong theoretical privacy, query responses, statistical analysis, federated learning

## Model Comparison

| Aspect | k-Anonymity | l-Diversity | t-Closeness | Differential Privacy |
|--------|-------------|-------------|-------------|----------------------|
| **Privacy Guarantee** | Group indistinguishability | Attribute diversity | Distribution closeness | Output indistinguishability |
| **Attack Model** | Identity disclosure | Attribute disclosure | Inference attacks | Composition attacks |
| **Computational Cost** | Low-Medium | High | Medium-High | Medium |
| **Data Utility** | High | Medium | High | Medium-Low (ε dependent) |
| **Quasi-Identifier Approach** | Required | Required | Optional | Not applicable |
| **Scalability** | Excellent | Good (with caching) | Good | Excellent |
| **Regulatory Fit** | GDPR, HIPAA | HIPAA, HIPAA+ | GDPR with utility focus | Research, federated systems |
| **Implementation Complexity** | Simple | Complex | Medium | Medium |

## Architecture

### Core Components

```
pamola_core.privacy_models/
├── base.py                          # BasePrivacyModelProcessor interface
├── k_anonymity/
│   ├── calculation.py               # KAnonymityProcessor
│   ├── ka_reporting.py              # KAnonymityReport
│   └── ka_visualization.py          # Visualization functions
├── l_diversity/
│   ├── calculation.py               # LDiversityCalculator
│   ├── metrics.py                   # LDiversityMetricsCalculator
│   ├── privacy.py                   # LDiversityPrivacyRiskAssessor
│   ├── attribute_risk.py            # AttributeDisclosureRiskAnalyzer
│   ├── interpretation.py            # RiskInterpreter
│   ├── apply_model.py               # Anonymization strategies
│   ├── reporting.py                 # LDiversityReport
│   ├── visualization.py             # LDiversityVisualizer
│   └── report_generators/           # Compliance and technical reports
├── t_closeness/
│   └── calculation.py               # TCloseness processor
└── differential_privacy/
    └── calculation.py               # DifferentialPrivacyProcessor
```

### Common Interface

All processors implement `BasePrivacyModelProcessor` with these methods:

```python
def process(data) -> Processed data
def evaluate_privacy(data, quasi_identifiers, **kwargs) -> dict
def apply_model(data, quasi_identifiers, suppression=True, **kwargs) -> pd.DataFrame
```

## Quick Start

### Example 1: k-Anonymity

```python
from pamola_core.privacy_models import KAnonymityProcessor
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Create processor
processor = KAnonymityProcessor(k=3)

# Evaluate privacy
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code']
)
print(evaluation)

# Apply anonymization
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'zip_code'],
    suppression=True
)
```

### Example 2: l-Diversity

```python
from pamola_core.privacy_models import LDiversityCalculator

# Create processor
processor = LDiversityCalculator(
    l=3,
    diversity_type='distinct'  # or 'entropy', 'recursive'
)

# Evaluate privacy
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code']
)

# Apply anonymization
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'zip_code'],
    suppression=True
)

# Generate comprehensive report
report = processor.generate_report()
```

### Example 3: t-Closeness

```python
from pamola_core.privacy_models import TCloseness

# Create processor with t threshold
processor = TCloseness(
    quasi_identifiers=['age', 'zip_code'],
    sensitive_column='salary',
    t=0.1  # Maximum allowed distribution distance
)

# Evaluate t-closeness
evaluation = processor.evaluate_privacy(df, quasi_identifiers=['age', 'zip_code'])
print(f"Is t-close: {evaluation['is_t_close']}")
```

### Example 4: Differential Privacy

```python
from pamola_core.privacy_models import DifferentialPrivacyProcessor

# Create processor
processor = DifferentialPrivacyProcessor(
    epsilon=0.5,        # Privacy budget
    sensitivity=1.0,    # Query sensitivity
    mechanism='laplace' # or 'gaussian'
)

# Apply differential privacy
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code']
)
```

## Usage Patterns

### Pattern 1: Privacy Evaluation Only

Assess privacy level without transformation:

```python
from pamola_core.privacy_models import KAnonymityProcessor

processor = KAnonymityProcessor(k=5)
evaluation = processor.evaluate_privacy(df, quasi_identifiers=['age', 'city'])

if not evaluation.get('is_k_anonymous'):
    print(f"Dataset fails k-anonymity: min_k={evaluation['min_k']}")
```

### Pattern 2: Privacy Transformation

Transform data to meet privacy requirements:

```python
from pamola_core.privacy_models import LDiversityCalculator

processor = LDiversityCalculator(l=4, diversity_type='distinct')
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'occupation'],
    suppression=True
)
```

### Pattern 3: Multi-Model Evaluation

Evaluate data against multiple models:

```python
from pamola_core.privacy_models import (
    KAnonymityProcessor,
    LDiversityCalculator,
    TCloseness
)

k_proc = KAnonymityProcessor(k=3)
l_proc = LDiversityCalculator(l=3)
t_proc = TCloseness(['age', 'zip'], 'salary', t=0.1)

k_eval = k_proc.evaluate_privacy(df, quasi_ids)
l_eval = l_proc.evaluate_privacy(df, quasi_ids)
t_eval = t_proc.evaluate_privacy(df, quasi_ids)

# Check which models are satisfied
print(f"k-Anonymity satisfied: {k_eval.get('is_k_anonymous')}")
print(f"l-Diversity satisfied: {l_eval.get('is_l_diverse')}")
print(f"t-Closeness satisfied: {t_eval.get('is_t_close')}")
```

### Pattern 4: Adaptive Anonymization

Use adaptive k/l values for different groups:

```python
from pamola_core.privacy_models import KAnonymityProcessor

# Define custom k values for different groups
adaptive_k = {
    ('adult', 'urban'): 5,
    ('adult', 'rural'): 3,
    ('senior', 'urban'): 4,
}

processor = KAnonymityProcessor(adaptive_k=adaptive_k)
anonymized = processor.apply_model(df, quasi_identifiers=['age_group', 'location'])
```

## Integration Guide

### With Data Processing Pipeline

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.transformations import DataTransformer

# Load data
df = load_data()

# Apply transformations
transformer = DataTransformer()
df_transformed = transformer.apply_transformations(df, transformations)

# Apply privacy model
privacy_processor = LDiversityCalculator(l=4, diversity_type='entropy')
df_anonymized = privacy_processor.apply_model(
    df_transformed,
    quasi_identifiers=['age', 'zip_code']
)

# Save result
save_data(df_anonymized)
```

### With Profiling

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.profiling import ProfileAnalyzer

# Analyze dataset
analyzer = ProfileAnalyzer()
profile = analyzer.analyze_dataset(df)

# Extract quasi-identifiers from profile
quasi_ids = profile['quasi_identifiers']

# Apply privacy model
processor = LDiversityCalculator(l=3)
anonymized = processor.apply_model(df, quasi_ids)
```

## Performance Considerations

### Memory Optimization

```python
# Use Dask for large datasets (>1GB)
processor = LDiversityCalculator(use_dask=True)

# Processor will automatically use Dask DataFrame
anonymized = processor.apply_model(df, quasi_identifiers)
```

### Caching Strategy

l-Diversity processor includes centralized caching to avoid redundant calculations:

```python
processor = LDiversityCalculator(l=3)

# First call calculates and caches results
eval1 = processor.evaluate_privacy(df, quasi_ids)

# Subsequent calls use cached results (much faster)
eval2 = processor.evaluate_privacy(df, quasi_ids)  # Uses cache
```

### Parallel Processing

```python
# k-Anonymity supports parallel processing via Dask
processor = KAnonymityProcessor(k=5, use_dask=True)

# Large datasets will be processed in parallel
anonymized = processor.apply_model(df, quasi_ids)
```

## Best Practices

1. **Define Clear Quasi-Identifiers:** Only include attributes that could be linked to external data.

2. **Choose Appropriate Model:** Match privacy model to your threat model and compliance requirements.

3. **Test on Samples:** Validate privacy transformation on sample data before processing production datasets.

4. **Monitor Information Loss:** Use metrics to understand utility trade-offs during anonymization.

5. **Document Decisions:** Keep records of parameter choices and privacy trade-offs for audit trails.

6. **Iterative Refinement:** Adjust k/l/t parameters based on privacy evaluation results.

7. **Cache Results:** Reuse evaluation results when possible to improve performance.

8. **Combine Models:** Use multiple models for defense-in-depth privacy strategy.

## Troubleshooting

### Issue: Dataset fails k-anonymity evaluation

**Symptom:** `is_k_anonymous: False` in evaluation results

**Solutions:**
- Increase k parameter
- Add more quasi-identifiers for grouping
- Consider using l-Diversity or t-Closeness for attribute-level privacy

### Issue: High information loss after anonymization

**Symptom:** Anonymized data has too many masked values

**Solutions:**
- Reduce k/l parameters (lower privacy guarantee)
- Use partial masking strategies
- Generalize quasi-identifiers instead of suppressing
- Consider t-Closeness to preserve distributions

### Issue: Performance degradation on large datasets

**Symptom:** Processing takes excessive time

**Solutions:**
- Enable Dask with `use_dask=True`
- Sample data for initial evaluation
- Use adaptive k/l levels
- Enable caching (automatic for l-Diversity)

### Issue: Memory exhaustion

**Symptom:** Out-of-memory errors

**Solutions:**
- Enable Dask processing
- Use adaptive k/l to reduce group computation
- Process in batches
- Clear cache periodically for long-running jobs

## Related Documentation

- [k-Anonymity Documentation](./k_anonymity/k_anonymity_processor.md)
- [l-Diversity Documentation](./l_diversity/l_diversity_calculator.md)
- [t-Closeness Documentation](./t_closeness/t_closeness_processor.md)
- [Differential Privacy Documentation](./differential_privacy/dp_processor.md)

## Summary

The `privacy_models` module provides flexible, production-ready privacy-preserving techniques. Start with k-Anonymity for basic privacy, graduate to l-Diversity for sensitive attributes, and consider t-Closeness or Differential Privacy for specific use cases. Use adaptive parameters and multi-model evaluation for robust privacy guarantees.
