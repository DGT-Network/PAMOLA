# KAnonymityReport Documentation

**Module:** `pamola_core.privacy_models.k_anonymity.ka_reporting`
**Class:** `KAnonymityReport`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Methods](#core-methods)
4. [Report Structure](#report-structure)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Related Components](#related-components)

## Overview

`KAnonymityReport` generates comprehensive reports for k-Anonymity anonymization, extending `PrivacyReport` with k-Anonymity-specific metrics, visualizations, and compliance documentation.

**Purpose:** Document k-Anonymity transformations for compliance, audit, and analysis.

**Location:** `pamola_core/privacy_models/k_anonymity/ka_reporting.py`

## Architecture

### Class Hierarchy

```
PrivacyReport (base)
    ↓
KAnonymityReport (specialized reporting)
    ├── metadata (timestamp, version)
    ├── k_anonymity_configuration (parameters used)
    ├── dataset_information (before/after stats)
    ├── privacy_evaluation (metrics)
    ├── anonymization_result (transformation details)
    └── information_loss (utility metrics)
```

## Core Methods

### 1. __init__(report_data)

**Signature:**
```python
def __init__(self, report_data: Dict[str, Any]):
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `report_data` | Dict[str, Any] | Dictionary containing k-Anonymity metrics and metadata |

**Example:**
```python
report_data = {
    'k_anonymity_configuration': {'k': 5},
    'privacy_evaluation': {'is_k_anonymous': True, 'min_k': 5},
    'dataset_information': {'original_records': 10000}
}

report = KAnonymityReport(report_data)
```

### 2. generate(include_visualizations=True)

**Signature:**
```python
def generate(self, include_visualizations: bool = True) -> Dict[str, Any]:
```

**Purpose:** Compile comprehensive k-Anonymity report.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_visualizations` | bool | True | Include paths to visualization files |

**Returns:** Complete report dictionary with sections:
- report_metadata
- k_anonymity_configuration
- dataset_information
- privacy_evaluation
- anonymization_result
- information_loss
- visualizations (if requested)

**Example:**
```python
processor = KAnonymityProcessor(k=5)
evaluation = processor.evaluate_privacy(df, quasi_ids)
anonymized = processor.apply_model(df, quasi_ids)

report_data = {
    'k_anonymity_configuration': {'k': processor.k},
    'privacy_evaluation': evaluation,
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(anonymized)
    }
}

report = KAnonymityReport(report_data)
report_output = report.generate(include_visualizations=True)
```

## Report Structure

### Report Sections

#### 1. Report Metadata

```python
{
    'report_type': 'k-anonymity',
    'timestamp': '2026-03-23T10:30:00Z',
    'pamola_version': '0.0.1',
    'report_id': 'ka_20260323_103000'
}
```

#### 2. k-Anonymity Configuration

```python
{
    'k': 5,
    'adaptive_k': None,
    'suppression': True,
    'mask_value': 'MASKED',
    'use_dask': False
}
```

#### 3. Dataset Information

```python
{
    'original_records': 10000,
    'anonymized_records': 9856,
    'records_suppressed': 144,
    'suppression_rate': 1.44,
    'quasi_identifiers': ['age', 'zip_code', 'gender'],
    'total_groups_before': 3200,
    'total_groups_after': 3000
}
```

#### 4. Privacy Evaluation

```python
{
    'is_k_anonymous': True,
    'min_k': 5,
    'max_k': 250,
    'avg_k': 27.5,
    'records_at_risk': 0,
    're_identification_risk': 0.0,
    'privacy_guarantee': 'All records belong to groups of size >= 5'
}
```

#### 5. Anonymization Result

```python
{
    'transformation_type': 'suppression',
    'status': 'SUCCESS',
    'execution_time_seconds': 2.34,
    'compliance_status': 'COMPLIANT',
    'notes': 'All records meet k-anonymity requirement'
}
```

#### 6. Information Loss

```python
{
    'information_loss_percentage': 1.44,
    'data_utility_score': 98.56,
    'completeness': 98.56,
    'accuracy': 100.0,
    'attribute_preservation': {
        'age': 100.0,
        'zip_code': 100.0,
        'gender': 100.0
    }
}
```

#### 7. Visualizations

```python
{
    'k_distribution': '/path/to/k_distribution.png',
    'risk_heatmap': '/path/to/risk_heatmap.png',
    'group_size_histogram': '/path/to/histogram.png'
}
```

## Usage Examples

### Example 1: Basic Report Generation

```python
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity import KAnonymityReport

# Process data
processor = KAnonymityProcessor(k=5)
eval_result = processor.evaluate_privacy(df, quasi_ids)
anonymized = processor.apply_model(df, quasi_ids)

# Prepare report data
report_data = {
    'k_anonymity_configuration': {
        'k': processor.k,
        'suppression': True
    },
    'privacy_evaluation': eval_result,
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(anonymized),
        'quasi_identifiers': quasi_ids
    }
}

# Generate report
report = KAnonymityReport(report_data)
report_output = report.generate()

print(f"Report ID: {report_output['report_metadata']['report_id']}")
print(f"Compliance: {report_output['anonymization_result']['compliance_status']}")
```

### Example 2: Full Workflow with Reporting

```python
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity import KAnonymityReport
import json

# Step 1: Evaluate
processor = KAnonymityProcessor(k=5)
evaluation = processor.evaluate_privacy(df, ['age', 'zip_code'])

# Step 2: Transform
anonymized = processor.apply_model(df, ['age', 'zip_code'], suppression=True)

# Step 3: Calculate metrics
metrics = processor.calculate_metrics(anonymized, ['age', 'zip_code'])

# Step 4: Build report data
report_data = {
    'k_anonymity_configuration': {
        'k': 5,
        'adaptive_k': None,
        'suppression': True
    },
    'privacy_evaluation': evaluation,
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(anonymized),
        'quasi_identifiers': ['age', 'zip_code']
    },
    'anonymization_result': {
        'status': 'SUCCESS',
        'compliance_status': 'COMPLIANT'
    },
    'information_loss': metrics
}

# Step 5: Generate report
report = KAnonymityReport(report_data)
final_report = report.generate(include_visualizations=True)

# Step 6: Save report
with open('k_anonymity_report.json', 'w') as f:
    json.dump(final_report, f, indent=2)
```

### Example 3: Compliance Documentation

```python
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity import KAnonymityReport

# Process with required k=10 for compliance
processor = KAnonymityProcessor(k=10)  # HIPAA requirement
eval_result = processor.evaluate_privacy(df, ['age', 'zip_code'])

# Prepare compliance-focused report
report_data = {
    'k_anonymity_configuration': {
        'k': 10,
        'compliance_framework': 'HIPAA',
        'suppression': True
    },
    'privacy_evaluation': eval_result,
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(anonymized),
        'quasi_identifiers': ['age', 'zip_code'],
        'regulatory_framework': 'HIPAA'
    },
    'anonymization_result': {
        'status': 'SUCCESS',
        'compliance_status': 'COMPLIANT' if eval_result['is_k_anonymous'] else 'NON_COMPLIANT',
        'notes': f"All records meet HIPAA k-anonymity requirement (k>=10)"
    }
}

# Generate compliance report
report = KAnonymityReport(report_data)
compliance_report = report.generate()

# Verify compliance
if compliance_report['anonymization_result']['compliance_status'] == 'COMPLIANT':
    print("✓ Dataset meets HIPAA requirements")
    with open('hipaa_compliance_report.json', 'w') as f:
        json.dump(compliance_report, f)
```

### Example 4: Report with Custom Visualization Paths

```python
from pamola_core.privacy_models import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity import KAnonymityReport

processor = KAnonymityProcessor(k=5)
anonymized = processor.apply_model(df, quasi_ids)

# Generate visualizations
fig_dist, path_dist = visualize_k_distribution(
    anonymized,
    'k_value',
    save_path='/reports/visualizations/'
)

# Build report with visualization paths
report_data = {
    'k_anonymity_configuration': {'k': 5},
    'privacy_evaluation': processor.evaluate_privacy(anonymized, quasi_ids),
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(anonymized),
        'quasi_identifiers': quasi_ids
    },
    'visualizations': {
        'k_distribution': path_dist
    }
}

report = KAnonymityReport(report_data)
final_report = report.generate(include_visualizations=True)
```

## Best Practices

1. **Include All Context:**
   ```python
   report_data = {
       'k_anonymity_configuration': processor.__dict__,
       'privacy_evaluation': eval_result,
       'dataset_information': {...},
       'anonymization_result': {...}
   }
   ```

2. **Generate Visualizations:**
   ```python
   report = report.generate(include_visualizations=True)
   # Include visualization paths for stakeholder review
   ```

3. **Document Decisions:**
   ```python
   report_data['anonymization_result']['notes'] = (
       "Selected k=5 based on privacy-utility trade-off analysis. "
       "Suppression strategy used to maintain data integrity."
   )
   ```

4. **Save for Compliance:**
   ```python
   import json
   from datetime import datetime

   timestamp = datetime.now().isoformat()
   filename = f"k_anonymity_report_{timestamp}.json"

   with open(filename, 'w') as f:
       json.dump(report.generate(), f, indent=2)
   ```

5. **Version Report Metadata:**
   ```python
   report_data = {
       'metadata': {
           'report_version': '1.0',
           'pamola_version': '0.0.1',
           'framework': 'GDPR'
       },
       ...
   }
   ```

## Related Components

- [KAnonymityProcessor](./k_anonymity_processor.md) — Processor that generates data for reports
- [KAnonymityVisualization](./k_anonymity_visualization.md) — Visualization functions for reports
- [Privacy Models Overview](../privacy_models_overview.md) — Model comparison

## Summary

`KAnonymityReport` transforms k-Anonymity evaluation results into professional, audit-ready documentation. Use it to:
- Document anonymization parameters and results
- Generate compliance reports
- Track before/after metrics
- Provide stakeholder communication
- Maintain audit trails

Combine with `KAnonymityProcessor` for complete k-Anonymity workflows.
