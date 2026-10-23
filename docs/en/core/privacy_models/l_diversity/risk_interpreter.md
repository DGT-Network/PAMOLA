# RiskInterpreter Documentation

**Module:** `pamola_core.privacy_models.l_diversity.interpretation`
**Class:** `RiskInterpreter`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

`RiskInterpreter` translates numeric risk values into human-readable assessments. It provides context-aware interpretation based on the data domain and a caller-chosen l-diversity threshold. Results are technical evidence; no law or regulation is mapped to a threshold and nothing here is a statement of legal compliance.

**Location:** `pamola_core/privacy_models/l_diversity/interpretation.py`

## Core Methods

### __init__(domain="general", l_threshold=3, custom_thresholds=None)

**Signature:**
```python
def __init__(
    self,
    domain: str = "general",
    l_threshold: int = 3,
    custom_thresholds: Dict[str, List[float]] = None
):
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | str | "general" | Domain context (healthcare, finance, education, telecom) |
| `l_threshold` | int | 3 | l-diversity threshold used by `interpret_threshold()`; chosen by the caller |
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
    l_threshold=4
)

risk_score = 25.5
interpretation = interpreter.interpret_risk(risk_score)

print(f"Risk Score: {risk_score}")
print(f"Level: {interpretation['level']}")
print(f"Threshold check: {interpreter.interpret_threshold(2.0)['threshold_met']}")
```

### Example 2: Financial Data Interpretation

```python
interpreter = RiskInterpreter(
    domain='finance'
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

### Example 4: Threshold Check

```python
interpreter = RiskInterpreter(l_threshold=3)

# Check multiple attributes
for attr, min_l in attribute_min_l.items():
    check = interpreter.interpret_threshold(min_l, 'distinct')

    if check['threshold_met']:
        print(f"✓ {attr}: risk within the configured threshold")
    else:
        print(f"✗ {attr}: below the configured threshold (l={min_l})")
    # A threshold check is technical evidence, not a legal determination.
```

## Risk Interpretation Levels

- **Very Low** (0-5%) — Minimal risk, widely acceptable
- **Low** (5-15%) — Limited risk, generally acceptable
- **Moderate** (15-30%) — Manageable with controls
- **High** (30-50%) — Significant risk, requires mitigation
- **Very High** (50%+) — Unacceptable risk, needs stronger privacy

Note: Thresholds vary by domain and regulation.

## Best Practices

1. **Match the domain and choose the threshold deliberately:**
   ```python
   # Healthcare data, stricter l chosen by the data owner
   interpreter = RiskInterpreter(
       domain='healthcare',
       l_threshold=4
   )
   ```

2. **Interpret All Sensitive Attributes:**
   ```python
   for attr in sensitive_attrs:
       risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, attr)
       interp = interpreter.interpret_risk(risk['overall_risk'], attribute=attr)
       print(f"{attr}: {interp['level']}")
   ```

3. **Record the threshold result as evidence (not as a compliance verdict):**
   ```python
   interpreter = RiskInterpreter(l_threshold=3)

   for attr, min_l in min_l_by_attribute.items():
       check = interpreter.interpret_threshold(min_l)
       print(f"{attr}: threshold {'met' if check['threshold_met'] else 'not met'} (l >= 3)")
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
