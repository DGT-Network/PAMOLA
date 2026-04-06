# LDiversityPrivacyRiskAssessor Documentation

**Module:** `pamola_core.privacy_models.l_diversity.privacy`
**Class:** `LDiversityPrivacyRiskAssessor`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

`LDiversityPrivacyRiskAssessor` evaluates privacy risks for l-Diversity datasets using multiple attack models. It provides comprehensive risk assessment with cache-aware computation.

**Location:** `pamola_core/privacy_models/l_diversity/privacy.py`

## Core Methods

### __init__(processor=None, risk_threshold=0.5)

**Signature:**
```python
def __init__(self, processor=None, risk_threshold: float = 0.5):
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `processor` | object | None | LDiversityCalculator for cached calculations |
| `risk_threshold` | float | 0.5 | Threshold for determining high-risk groups |

### assess_privacy_risks(data, quasi_identifiers, sensitive_attributes, diversity_type="distinct", **kwargs)

**Purpose:** Comprehensive privacy risk assessment for l-Diversity dataset.

**Signature:**
```python
def assess_privacy_risks(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
    diversity_type: str = "distinct",
    **kwargs
) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'overall_risk': float,           # Overall privacy risk %
    'risk_level': str,               # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    'high_risk_groups': int,         # Number of high-risk groups
    'record_risk_distribution': {...},  # Risk distribution by record
    'attack_models': {
        'identity_disclosure_risk': float,
        'attribute_disclosure_risk': float,
        'inference_risk': float
    },
    'recommendations': [...]         # Mitigations
}
```

## Usage Examples

### Example 1: Basic Risk Assessment

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity.privacy import LDiversityPrivacyRiskAssessor

processor = LDiversityCalculator(l=3, diversity_type='distinct')
anonymized = processor.apply_model(df, quasi_ids)

assessor = LDiversityPrivacyRiskAssessor(processor=processor, risk_threshold=0.3)
risk_assessment = assessor.assess_privacy_risks(
    anonymized,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)

print(f"Overall Risk: {risk_assessment['overall_risk']:.1f}%")
print(f"Risk Level: {risk_assessment['risk_level']}")
print(f"High-Risk Groups: {risk_assessment['high_risk_groups']}")
```

### Example 2: Multi-Attribute Risk

```python
assessor = LDiversityPrivacyRiskAssessor(processor=processor)

for attr in ['diagnosis', 'treatment', 'marital_status']:
    risk = assessor.assess_privacy_risks(
        anonymized,
        quasi_identifiers=['age', 'zip_code'],
        sensitive_attributes=[attr]
    )
    print(f"{attr}: {risk['risk_level']} (overall={risk['overall_risk']:.1f}%)")
```

### Example 3: Attack Model Analysis

```python
risk_assessment = assessor.assess_privacy_risks(
    anonymized,
    quasi_identifiers=quasi_ids,
    sensitive_attributes=sens_attrs
)

# Analyze specific attack risks
attacks = risk_assessment['attack_models']
print(f"Identity Disclosure: {attacks['identity_disclosure_risk']:.1f}%")
print(f"Attribute Disclosure: {attacks['attribute_disclosure_risk']:.1f}%")
print(f"Inference Attack: {attacks['inference_risk']:.1f}%")

# Review recommendations
print("\nMitigations:")
for rec in risk_assessment['recommendations']:
    print(f"  - {rec}")
```

## Risk Levels

- **LOW** (< 10%): Acceptable risk, limited privacy concerns
- **MODERATE** (10-30%): Manageable with controls
- **HIGH** (30-50%): Requires mitigation
- **CRITICAL** (> 50%): Unacceptable risk, needs stronger privacy model

## Best Practices

1. **Assess All Sensitive Attributes:**
   ```python
   for attr in sensitive_attrs:
       risk = assessor.assess_privacy_risks(df, quasi_ids, [attr])
   ```

2. **Compare Across Models:**
   ```python
   # Evaluate same data with different diversity types
   for div_type in ['distinct', 'entropy', 'recursive']:
       processor = LDiversityCalculator(l=3, diversity_type=div_type)
       anonymized = processor.apply_model(df, quasi_ids)
       risk = assessor.assess_privacy_risks(anonymized, quasi_ids, sens_attrs)
   ```

3. **Use Processor's Cache:**
   ```python
   # Reuses processor's cached calculations
   assessor = LDiversityPrivacyRiskAssessor(processor=processor)
   ```

4. **Document High-Risk Groups:**
   ```python
   risk = assessor.assess_privacy_risks(df, quasi_ids, sens_attrs)
   if risk['high_risk_groups'] > 0:
       print(f"⚠️ {risk['high_risk_groups']} high-risk groups detected")
       for rec in risk['recommendations']:
           print(f"  Mitigation: {rec}")
   ```

## Related Components

- [LDiversityCalculator](./l_diversity_calculator.md)
- [AttributeDisclosureRiskAnalyzer](./attribute_disclosure.md)
- [RiskInterpreter](./risk_interpreter.md)
