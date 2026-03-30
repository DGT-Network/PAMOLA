# AttributeDisclosureRiskAnalyzer Documentation

**Module:** `pamola_core.privacy_models.l_diversity.attribute_risk`
**Class:** `AttributeDisclosureRiskAnalyzer`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

`AttributeDisclosureRiskAnalyzer` evaluates attribute disclosure risk in l-Diversity datasets. It focuses on the risk of revealing sensitive attribute values even when identity is protected.

**Location:** `pamola_core/privacy_models/l_diversity/attribute_risk.py`

## Core Methods

### __init__(l_threshold=3, diversity_type="distinct", c_value=1.0, high_risk_threshold=50.0)

**Signature:**
```python
def __init__(
    self,
    l_threshold: int = 3,
    diversity_type: str = "distinct",
    c_value: float = 1.0,
    high_risk_threshold: float = 50.0,
):
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `l_threshold` | int | 3 | Minimum acceptable l value |
| `diversity_type` | str | "distinct" | Type of l-diversity |
| `c_value` | float | 1.0 | Parameter for recursive diversity |
| `high_risk_threshold` | float | 50.0 | Risk percentage for high risk |

### calculate_attribute_disclosure_risk(data, quasi_identifiers, sensitive_attribute, **kwargs)

**Purpose:** Calculate comprehensive attribute disclosure risk.

**Signature:**
```python
def calculate_attribute_disclosure_risk(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attribute: str,
    **kwargs
) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'attribute': str,
    'overall_risk': float,           # Risk percentage
    'high_risk_groups': int,
    'disclosure_rates': {...},       # Per-value disclosure rates
    'value_distribution': {...},     # Value frequency distribution
    'homogeneity_score': float,      # Group homogeneity (0-100)
    'predictability': float,         # Value predictability
    'risk_level': str,               # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    'recommendations': [...]
}
```

## Usage Examples

### Example 1: Single Attribute Risk

```python
from pamola_core.privacy_models.l_diversity.attribute_risk import AttributeDisclosureRiskAnalyzer

analyzer = AttributeDisclosureRiskAnalyzer(
    l_threshold=3,
    diversity_type='distinct',
    high_risk_threshold=50.0
)

risk = analyzer.calculate_attribute_disclosure_risk(
    df,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attribute='diagnosis'
)

print(f"Attribute: {risk['attribute']}")
print(f"Overall Risk: {risk['overall_risk']:.1f}%")
print(f"Risk Level: {risk['risk_level']}")
print(f"High-Risk Groups: {risk['high_risk_groups']}")
```

### Example 2: Multi-Attribute Comparison

```python
analyzer = AttributeDisclosureRiskAnalyzer(l_threshold=3)

sensitive_attrs = ['diagnosis', 'treatment', 'procedure']
risks = {}

for attr in sensitive_attrs:
    risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, attr)
    risks[attr] = risk
    print(f"\n{attr}:")
    print(f"  Risk: {risk['overall_risk']:.1f}%")
    print(f"  Homogeneity: {risk['homogeneity_score']:.1f}%")
    print(f"  Predictability: {risk['predictability']:.1f}%")
```

### Example 3: Risk-Based Mitigation

```python
analyzer = AttributeDisclosureRiskAnalyzer(high_risk_threshold=40.0)

for attr in sensitive_attrs:
    risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, attr)

    if risk['overall_risk'] > 40.0:
        print(f"⚠️ HIGH RISK: {attr}")
        for rec in risk['recommendations']:
            print(f"   → {rec}")
    else:
        print(f"✓ {attr}: {risk['risk_level']}")
```

### Example 4: Value-Specific Risk Analysis

```python
risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, 'diagnosis')

# Check risk for specific values
print("Disclosure rates by value:")
for value, rate in risk['disclosure_rates'].items():
    print(f"  {value}: {rate:.1f}%")

# Value distribution
print("\nValue distribution:")
for value, count in risk['value_distribution'].items():
    print(f"  {value}: {count} records")
```

## Risk Metrics

### Homogeneity Score
Measures how uniform sensitive attribute values are within groups:
- 0% = All groups contain all different values
- 100% = All groups contain same value

Higher homogeneity = Higher disclosure risk

### Predictability
Measures how easily attacker can predict sensitive value:
- 0% = Uniform distribution (impossible to predict)
- 100% = Single dominant value (easy to predict)

### Disclosure Rates
Per-value percentage of records vulnerable to disclosure.

## Best Practices

1. **Assess All Sensitive Attributes:**
   ```python
   for attr in df[sensitive_columns]:
       risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, attr)
   ```

2. **Use Domain-Specific Thresholds:**
   ```python
   # Healthcare domain (strict)
   analyzer = AttributeDisclosureRiskAnalyzer(
       high_risk_threshold=30.0,  # Lower threshold
       l_threshold=5               # Higher l
   )

   # Marketing domain (lenient)
   analyzer = AttributeDisclosureRiskAnalyzer(
       high_risk_threshold=60.0,  # Higher threshold
       l_threshold=2               # Lower l
   )
   ```

3. **Track High-Risk Values:**
   ```python
   risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, 'diagnosis')

   high_risk_values = {
       v: rate for v, rate in risk['disclosure_rates'].items()
       if rate > 50.0
   }
   print(f"High-risk values: {high_risk_values}")
   ```

4. **Document Mitigation Decisions:**
   ```python
   risk = analyzer.calculate_attribute_disclosure_risk(df, quasi_ids, attr)

   if risk['overall_risk'] > threshold:
       action = "Increase l-diversity" if risk['homogeneity_score'] > 70 else "Suppress value"
       print(f"Recommended action: {action}")
   ```

## Related Components

- [LDiversityCalculator](./l_diversity_calculator.md)
- [LDiversityPrivacyRiskAssessor](./privacy_risk_assessor.md)
- [RiskInterpreter](./risk_interpreter.md)
