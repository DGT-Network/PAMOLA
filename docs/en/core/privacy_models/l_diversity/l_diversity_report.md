# LDiversityReport Documentation

**Module:** `pamola_core.privacy_models.l_diversity.reporting`
**Class:** `LDiversityReport`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Core Methods](#core-methods)
3. [Report Structure](#report-structure)
4. [Usage Examples](#usage-examples)
5. [Report Sections](#report-sections)
6. [Best Practices](#best-practices)
7. [Related Components](#related-components)

## Overview

`LDiversityReport` generates comprehensive reports for l-Diversity anonymization. It extends `PrivacyReport` with l-Diversity-specific metrics, including diversity calculations, risk assessments, and compliance status.

**Purpose:** Document l-Diversity transformations for compliance, stakeholder communication, and audit trails.

**Location:** `pamola_core/privacy_models/l_diversity/reporting.py`

## Core Methods

### 1. __init__(processor=None, report_data=None, diversity_type=None)

**Signature:**
```python
def __init__(
    self,
    processor=None,
    report_data: Optional[Dict[str, Any]] = None,
    diversity_type: str = None
):
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `processor` | object | LDiversityCalculator instance for cache access |
| `report_data` | Dict | Explicit report data (if not using processor) |
| `diversity_type` | str | Type of l-diversity ("distinct", "entropy", "recursive") |

**Example:**
```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity import LDiversityReport

processor = LDiversityCalculator(l=3, diversity_type='distinct')
report = LDiversityReport(processor=processor)
```

### 2. generate(output_format='comprehensive')

**Signature:**
```python
def generate(self, output_format: str = 'comprehensive') -> Dict[str, Any]:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | str | 'comprehensive' | 'comprehensive', 'compliance', or 'technical' |

**Returns:** Complete report dictionary

**Example:**
```python
# Comprehensive report (all sections)
full_report = report.generate(output_format='comprehensive')

# Compliance-focused report
compliance_report = report.generate(output_format='compliance')

# Technical details only
tech_report = report.generate(output_format='technical')
```

## Report Structure

### Full Report Sections

```python
{
    'report_metadata': {
        'timestamp': '2026-03-23T10:30:00Z',
        'report_type': 'l-diversity',
        'report_id': 'ld_20260323_103000',
        'pamola_version': '0.0.1'
    },
    'l_diversity_configuration': {
        'l': 3,
        'diversity_type': 'distinct',
        'k': 2,
        'c_value': 1.0,
        'adaptive_l': None
    },
    'dataset_information': {
        'original_records': 10000,
        'anonymized_records': 9856,
        'quasi_identifiers': ['age', 'zip_code'],
        'sensitive_attributes': ['diagnosis'],
        'total_groups': 3000
    },
    'privacy_evaluation': {
        'is_l_diverse': True,
        'l_value': 3,
        'min_diversity': 3.0,
        'max_diversity': 145.0,
        'avg_diversity': 15.2,
        'non_diverse_groups': 0
    },
    'anonymization_result': {
        'status': 'SUCCESS',
        'compliance_status': 'COMPLIANT',
        'transformation_type': 'suppression',
        'execution_time_seconds': 3.45
    },
    'information_loss': {
        'information_loss_percentage': 1.44,
        'data_utility_score': 98.56,
        'completeness': 98.56,
        'accuracy': 100.0
    },
    'risk_assessment': {
        'attribute_disclosure_risk': {...},
        're_identification_risk': 0.5,
        'privacy_risk_level': 'LOW'
    },
    'visualizations': {
        'diversity_distribution': '/path/to/viz.png',
        'risk_heatmap': '/path/to/heatmap.png'
    }
}
```

## Usage Examples

### Example 1: Generate Basic Report

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity import LDiversityReport

processor = LDiversityCalculator(l=3, diversity_type='distinct')
anonymized = processor.apply_model(df, quasi_ids)

# Generate report
report = LDiversityReport(processor=processor)
report_output = report.generate()

# Access sections
print(f"Compliance: {report_output['anonymization_result']['compliance_status']}")
print(f"Min diversity: {report_output['privacy_evaluation']['min_diversity']}")
```

### Example 2: Processor-Based Report (Recommended)

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity import LDiversityReport
import json

# Process with caching
processor = LDiversityCalculator(l=4, diversity_type='entropy')
anonymized = processor.apply_model(df, quasi_ids)

# Report uses processor's cached results
report = LDiversityReport(processor=processor)
report_output = report.generate(output_format='comprehensive')

# Save
with open('ldiversity_report.json', 'w') as f:
    json.dump(report_output, f, indent=2)
```

### Example 3: Manual Report Data

```python
from pamola_core.privacy_models.l_diversity import LDiversityReport

# Manually assembled report data
report_data = {
    'l_diversity_configuration': {
        'l': 3,
        'diversity_type': 'distinct'
    },
    'privacy_evaluation': {
        'is_l_diverse': True,
        'min_diversity': 3.0,
        'max_diversity': 100.0
    },
    'dataset_information': {
        'original_records': 10000,
        'anonymized_records': 9800,
        'quasi_identifiers': ['age', 'zip_code']
    }
}

report = LDiversityReport(report_data=report_data, diversity_type='distinct')
report_output = report.generate()
```

### Example 4: Multi-Attribute Report

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity import LDiversityReport
from pamola_core.privacy_models.l_diversity.attribute_risk import AttributeDisclosureRiskAnalyzer

processor = LDiversityCalculator(l=3, diversity_type='entropy')
anonymized = processor.apply_model(df, quasi_ids)

# Assess each attribute
risk_analyzer = AttributeDisclosureRiskAnalyzer(l_threshold=3)
risks = {}
for attr in ['diagnosis', 'treatment', 'marital_status']:
    risk = risk_analyzer.calculate_attribute_disclosure_risk(
        anonymized, quasi_ids, attr
    )
    risks[attr] = risk

# Build report with attribute risks
report_data = {
    'l_diversity_configuration': {'l': 3, 'diversity_type': 'entropy'},
    'privacy_evaluation': processor.evaluate_privacy(anonymized, quasi_ids),
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(anonymized),
        'quasi_identifiers': quasi_ids,
        'sensitive_attributes': list(risks.keys())
    },
    'risk_assessment': {
        'attribute_risks': risks
    }
}

report = LDiversityReport(report_data=report_data)
report_output = report.generate()
```

### Example 5: Compliance-Focused Report

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity import LDiversityReport

# HIPAA-compliant l-diversity
processor = LDiversityCalculator(
    l=4,
    diversity_type='entropy',
    k=5
)

anonymized = processor.apply_model(df, quasi_ids, suppression=True)

# Build compliance report
report_data = {
    'l_diversity_configuration': {
        'l': 4,
        'diversity_type': 'entropy',
        'k': 5,
        'regulatory_framework': 'HIPAA'
    },
    'privacy_evaluation': processor.evaluate_privacy(anonymized, quasi_ids),
    'dataset_information': {
        'original_records': len(df),
        'anonymized_records': len(anonymized),
        'regulatory_framework': 'HIPAA',
        'compliance_required': True
    },
    'anonymization_result': {
        'status': 'SUCCESS',
        'compliance_status': 'COMPLIANT',
        'compliance_certification': 'HIPAA Privacy Rule §164.514'
    }
}

report = LDiversityReport(
    report_data=report_data,
    diversity_type='entropy'
)
compliance_report = report.generate(output_format='compliance')

# Verify compliance
if compliance_report['anonymization_result']['compliance_status'] == 'COMPLIANT':
    print("✓ HIPAA Compliance Certified")
```

## Report Sections

### Report Metadata

Automatically generated:
- Timestamp of report generation
- Report ID (unique identifier)
- PAMOLA version
- Report type

### l-Diversity Configuration

Documents the parameters used:
- l value
- diversity_type (distinct/entropy/recursive)
- k value (if k-anonymity also enforced)
- c_value (if recursive diversity)
- adaptive_l (if group-specific l values)

### Dataset Information

Before/after statistics:
- Original record count
- Anonymized record count
- Quasi-identifier columns
- Sensitive attribute columns
- Group statistics

### Privacy Evaluation

Diversity metrics:
- Compliance status
- Minimum diversity value
- Maximum diversity value
- Average diversity value
- Non-compliant groups count

### Anonymization Result

Transformation details:
- Status (SUCCESS/FAILED)
- Compliance status
- Transformation type
- Execution time
- Success notes/error messages

### Information Loss

Data utility metrics:
- Information loss percentage
- Data utility score
- Completeness percentage
- Accuracy score
- Per-attribute preservation

### Risk Assessment

Privacy risk analysis:
- Attribute disclosure risks
- Re-identification risk percentages
- Overall privacy risk level
- Risk recommendations

### Visualizations

Paths to generated visualizations (if available):
- Diversity distribution plots
- Risk heatmaps
- Attribute distribution visualizations

## Best Practices

1. **Always Use Processor-Based Reports:**
   ```python
   # Recommended: processor's cache is utilized
   processor = LDiversityCalculator(l=3)
   report = LDiversityReport(processor=processor)

   # Avoid: manual data assembly is error-prone
   report_data = {...}
   report = LDiversityReport(report_data=report_data)
   ```

2. **Document Decisions:**
   ```python
   report_data = {
       ...
       'anonymization_result': {
           'notes': 'Selected l=3 based on privacy-utility trade-off. '
                    'Entropy diversity used for skewed distributions.',
           'justification': 'Balances HIPAA compliance with data utility'
       }
   }
   ```

3. **Include Visualizations:**
   ```python
   report_output = report.generate(output_format='comprehensive')
   # Ensures visualization paths are included for stakeholder review
   ```

4. **Save for Audit Trail:**
   ```python
   import json
   from datetime import datetime

   timestamp = datetime.now().isoformat()
   filename = f"ldiversity_report_{timestamp}.json"

   with open(filename, 'w') as f:
       json.dump(report_output, f, indent=2)
   ```

5. **Choose Appropriate Output Format:**
   ```python
   # For technical team
   tech_report = report.generate(output_format='technical')

   # For compliance/audit
   compliance_report = report.generate(output_format='compliance')

   # For stakeholders
   full_report = report.generate(output_format='comprehensive')
   ```

## Related Components

- [LDiversityCalculator](./l_diversity_calculator.md) — Processor for report data
- [LDiversityMetricsCalculator](./l_diversity_metrics.md) — Metrics computation
- [LDiversityPrivacyRiskAssessor](./privacy_risk_assessor.md) — Risk assessment
- [AttributeDisclosureRiskAnalyzer](./attribute_disclosure.md) — Attribute risk analysis

## Summary

`LDiversityReport` transforms l-Diversity evaluation results into professional documentation suitable for:
- Compliance verification and audits
- Stakeholder communication
- Risk documentation
- Regulatory reporting (HIPAA, GDPR, CCPA)

Always generate reports through the processor's cache for efficiency. Choose output format based on intended audience and use case.
