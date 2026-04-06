# RiskInterpreter Documentation

**Module:** `pamola_core.privacy_models.l_diversity.interpretation`
**Class:** `RiskInterpreter`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

`RiskInterpreter` translates numeric risk values into human-readable assessments. It provides context-aware interpretation based on domain and regulatory requirements.

**Location:** `pamola_core/privacy_models/l_diversity/interpretation.py`

## Core Methods

### __init__(domain="general", regulation=None, custom_thresholds=None)

**Signature:**
```python
def __init__(
    self,
    domain: str = "general",
    regulation: str = None,
    custom_thresholds: Dict[str, List[float]] = None
):
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | str | "general" | Domain context (healthcare, finance, education, telecom) |
| `regulation` | str | None | Regulatory framework (GDPR, HIPAA, CCPA, PIPEDA, APPI) |
| `custom_thresholds` | Dict | None | Custom risk thresholds |

## Supported Domains

- **general** — Default thresholds
- **healthcare** — Stricter thresholds (medical data sensitivity)
- **finance** — Strict thresholds (financial data sensitivity)
- **education** — Moderate thresholds
- **telecom** — Moderate-strict thresholds

## Supported Regulations

- **GDPR** — EU data protection (l >= 3)
- **HIPAA** — US healthcare (l >= 4)
- **CCPA** — California privacy (l >= 3)
- **PIPEDA** — Canadian privacy (l >= 3)
- **APPI** — Japanese privacy (l >= 3)

## Usage Examples

### Example 1: Healthcare Domain Interpretation

```python
from pamola_core.privacy_models.l_diversity.interpretation import RiskInterpreter

# Healthcare-specific interpretation
interpreter = RiskInterpreter(
    domain='healthcare',
    regulation='HIPAA'
)

risk_score = 25.5
interpretation = interpreter.interpret_risk(risk_score)

print(f"Risk Score: {risk_score}")
print(f"Level: {interpretation['level']}")
print(f"Compliant: {interpretation['regulatory_compliant']}")
```

### Example 2: Financial Data Interpretation

```python
interpreter = RiskInterpreter(
    domain='finance',
    regulation='None'
)

# Financial data requires stricter thresholds
risk_score = 15.0
interpretation = interpreter.interpret_risk(risk_score)

if not interpretation['acceptable']:
    print(f"⚠️ Risk unacceptable in finance domain")
    print(f"   Threshold: {interpretation['thresholds']['high']}")
```

### Example 3: Custom Thresholds

```python
# Define custom risk thresholds
custom_thresholds = {
    'general': [5.0, 15.0, 30.0, 50.0]  # Very low, low, moderate, high
}

interpreter = RiskInterpreter(custom_thresholds=custom_thresholds)
interpretation = interpreter.interpret_risk(18.5)
```

### Example 4: Regulatory Compliance Check

```python
interpreter = RiskInterpreter(regulation='GDPR')

# Check multiple attributes
for attr, risk_score in attribute_risks.items():
    interp = interpreter.interpret_risk(risk_score, attribute=attr)

    if interp['regulatory_compliant']:
        print(f"✓ {attr}: GDPR compliant")
    else:
        print(f"✗ {attr}: GDPR non-compliant (risk={risk_score:.1f}%)")
```

## Risk Interpretation Levels

- **Very Low** (0-5%) — Minimal risk, widely acceptable
- **Low** (5-15%) — Limited risk, generally acceptable
- **Moderate** (15-30%) — Manageable with controls
- **High** (30-50%) — Significant risk, requires mitigation
- **Very High** (50%+) — Unacceptable risk, needs stronger privacy

Note: Thresholds vary by domain and regulation.

## Best Practices

1. **Match Domain and Regulation:**
   ```python
   # For HIPAA-covered healthcare data
   interpreter = RiskInterpreter(
       domain='healthcare',
       regulation='HIPAA'
   )
   ```

2. **Interpret All Sensitive Attributes:**
   ```python
   for attr in sensitive_attrs:
       risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, attr)
       interp = interpreter.interpret_risk(risk['overall_risk'], attribute=attr)
       print(f"{attr}: {interp['level']}")
   ```

3. **Document Regulatory Compliance:**
   ```python
   interpreter = RiskInterpreter(regulation='GDPR')

   for attr, risk_score in risks.items():
       interp = interpreter.interpret_risk(risk_score)
       compliance_status = 'COMPLIANT' if interp['regulatory_compliant'] else 'NON-COMPLIANT'
       print(f"{attr}: {compliance_status} (GDPR)")
   ```

4. **Use for Risk Communication:**
   ```python
   # Present interpretation to stakeholders
   interp = interpreter.interpret_risk(risk_score)

   print(f"Risk Level: {interp['level']}")
   print(f"Description: {interp['description']}")
   print(f"Recommendation: {interp['recommendation']}")
   ```

## Related Components

- [LDiversityPrivacyRiskAssessor](./privacy_risk_assessor.md)
- [AttributeDisclosureRiskAnalyzer](./attribute_disclosure.md)
