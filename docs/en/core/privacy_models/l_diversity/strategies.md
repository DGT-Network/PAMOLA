# Anonymization Strategies Documentation

**Module:** `pamola_core.privacy_models.l_diversity.apply_model`
**Classes:** `AnonymizationStrategy`, `SuppressionStrategy`, `FullMaskingStrategy`, `PartialMaskingStrategy`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

Anonymization strategies define how to handle non-diverse groups in l-Diversity. Different strategies offer different privacy-utility trade-offs.

**Location:** `pamola_core/privacy_models/l_diversity/apply_model.py`

## Strategy Types

### 1. SuppressionStrategy

**Behavior:** Remove entire groups that don't meet l-Diversity requirements.

**Pros:**
- Strongest privacy guarantee
- Simple implementation
- Predictable behavior

**Cons:**
- Highest information loss
- Records may be permanently removed

**Example:**
```python
from pamola_core.privacy_models.l_diversity.apply_model import SuppressionStrategy
from pamola_core.privacy_models import LDiversityCalculator

processor = LDiversityCalculator(l=3)
strategy = SuppressionStrategy(processor)

# Non-diverse groups are completely removed
anonymized = strategy.apply(
    df,
    non_diverse_groups=[(25, '12345'), (30, '67890')],
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)
```

### 2. FullMaskingStrategy

**Behavior:** Replace all quasi-identifier values in non-diverse groups with mask string.

**Pros:**
- Strong privacy
- Retains all records
- Clear masking indicators

**Cons:**
- Complete information loss for masked groups
- May make patterns obvious

**Example:**
```python
from pamola_core.privacy_models.l_diversity.apply_model import FullMaskingStrategy

processor = LDiversityCalculator(l=3)
strategy = FullMaskingStrategy(processor, mask_value="***")

anonymized = strategy.apply(
    df,
    non_diverse_groups=non_diverse,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)
# Non-diverse records have age='***', zip_code='***'
```

### 3. PartialMaskingStrategy

**Behavior:** Mask only specific quasi-identifier values in non-diverse groups.

**Pros:**
- Balanced privacy and utility
- Retains some information
- Flexible per-attribute control

**Cons:**
- More complex configuration
- May still allow linkage attacks

**Example:**
```python
from pamola_core.privacy_models.l_diversity.apply_model import PartialMaskingStrategy

processor = LDiversityCalculator(l=3)
strategy = PartialMaskingStrategy(
    processor,
    mask_columns=['age'],  # Only mask age, keep zip_code
    mask_value="**"
)

anonymized = strategy.apply(
    df,
    non_diverse_groups=non_diverse,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)
```

## Usage Examples

### Example 1: Choose Strategy Based on Requirements

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity.apply_model import (
    SuppressionStrategy,
    FullMaskingStrategy,
    PartialMaskingStrategy
)

processor = LDiversityCalculator(l=3)
df_anonymous = processor.apply_model(df, quasi_ids)

# Evaluate which groups are non-diverse
eval_result = processor.evaluate_privacy(df, quasi_ids)
non_diverse = eval_result.get('non_diverse_groups', [])

# Choose strategy
if len(non_diverse) < 10:
    # Few non-diverse groups → suppression acceptable
    strategy = SuppressionStrategy(processor)
    print("Using suppression for few non-diverse groups")
elif len(non_diverse) < 100:
    # Moderate non-diverse groups → full masking
    strategy = FullMaskingStrategy(processor, mask_value="***")
    print("Using full masking for moderate non-diversity")
else:
    # Many non-diverse groups → partial masking
    strategy = PartialMaskingStrategy(processor, mask_columns=['age'])
    print("Using partial masking for many non-diverse groups")

result = strategy.apply(df, non_diverse, quasi_ids, sensitive_attrs)
```

### Example 2: Compare Strategies

```python
from pamola_core.privacy_models import LDiversityCalculator

processor = LDiversityCalculator(l=3)

# Strategy 1: Suppression
strategy_suppression = SuppressionStrategy(processor)
result_suppress = strategy_suppression.apply(df, non_diverse, quasi_ids, sens_attrs)
print(f"Suppression: {len(result_suppress)} records retained ({len(result_suppress)/len(df)*100:.1f}%)")

# Strategy 2: Full Masking
strategy_full = FullMaskingStrategy(processor)
result_full = strategy_full.apply(df, non_diverse, quasi_ids, sens_attrs)
print(f"Full Masking: {len(result_full)} records retained ({len(result_full)/len(df)*100:.1f}%)")

# Strategy 3: Partial Masking
strategy_partial = PartialMaskingStrategy(processor, mask_columns=['age'])
result_partial = strategy_partial.apply(df, non_diverse, quasi_ids, sens_attrs)
print(f"Partial Masking: {len(result_partial)} records retained ({len(result_partial)/len(df)*100:.1f}%)")
```

### Example 3: Regulatory-Driven Strategy Selection

```python
# HIPAA: strict privacy, suppression acceptable if < 5% data loss
if len(non_diverse) / len(df) < 0.05:
    strategy = SuppressionStrategy(processor)
else:
    # If too much data lost, use masking
    strategy = FullMaskingStrategy(processor, mask_value="MASKED")

anonymized = strategy.apply(df, non_diverse, quasi_ids, sens_attrs)
```

## Best Practices

1. **Evaluate Non-Diversity First:**
   ```python
   eval_result = processor.evaluate_privacy(df, quasi_ids)
   non_diverse = eval_result.get('non_diverse_groups', [])
   pct_affected = len(non_diverse) / len(df) * 100

   if pct_affected > 10:
       print(f"Warning: {pct_affected:.1f}% data affected by non-diversity")
   ```

2. **Match Strategy to Data Loss Tolerance:**
   ```python
   tolerance = 0.05  # 5% data loss acceptable

   if len(non_diverse) / len(df) <= tolerance:
       strategy = SuppressionStrategy(processor)
   else:
       strategy = FullMaskingStrategy(processor)
   ```

3. **Document Strategy Choice:**
   ```python
   report_data = {
       'anonymization_strategy': {
           'type': 'FullMasking',
           'mask_value': '***',
           'justification': 'Balance privacy and data retention'
       }
   }
   ```

4. **Verify Strategy Effectiveness:**
   ```python
   anonymized = strategy.apply(df, non_diverse, quasi_ids, sens_attrs)
   eval_after = processor.evaluate_privacy(anonymized, quasi_ids)

   if eval_after['is_l_diverse']:
       print("✓ Strategy effective: dataset now l-diverse")
   ```

## Related Components

- [LDiversityCalculator](./l_diversity_calculator.md)
- [LDiversityModelApplicator](./model_applicator.md)
