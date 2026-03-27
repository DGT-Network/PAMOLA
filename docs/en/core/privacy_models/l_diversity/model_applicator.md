# LDiversityModelApplicator Documentation

**Module:** `pamola_core.privacy_models.l_diversity.apply_model`
**Class:** `LDiversityModelApplicator`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

`LDiversityModelApplicator` orchestrates l-Diversity application with flexible anonymization strategies. It manages the end-to-end anonymization workflow including strategy selection, application, and result tracking.

**Location:** `pamola_core/privacy_models/l_diversity/apply_model.py`

## Core Methods

### __init__(processor, strategy="suppression")

**Signature:**
```python
def __init__(self, processor: LDiversityCalculator, strategy: str = "suppression"):
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `processor` | LDiversityCalculator | — | Processor with l-diversity configuration |
| `strategy` | str | "suppression" | "suppression", "full_masking", or "partial_masking" |

### apply(data, quasi_identifiers, sensitive_attributes, **kwargs)

**Purpose:** Apply l-Diversity with selected strategy.

**Signature:**
```python
def apply(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
    **kwargs
) -> pd.DataFrame:
```

**Returns:** Anonymized DataFrame with l-Diversity guaranteed.

## Usage Examples

### Example 1: Basic Application

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity.apply_model import LDiversityModelApplicator

processor = LDiversityCalculator(l=3, diversity_type='distinct')
applicator = LDiversityModelApplicator(processor, strategy='suppression')

anonymized = applicator.apply(
    df,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)

print(f"Original: {len(df)} records")
print(f"Anonymized: {len(anonymized)} records")
```

### Example 2: Strategy Selection

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity.apply_model import LDiversityModelApplicator

processor = LDiversityCalculator(l=4, diversity_type='entropy')

# Evaluate first
eval_result = processor.evaluate_privacy(df, quasi_ids)
non_diverse_pct = (eval_result.get('non_diverse_groups', 0) / len(df)) * 100

# Choose strategy based on impact
if non_diverse_pct < 2:
    strategy = 'suppression'
elif non_diverse_pct < 10:
    strategy = 'full_masking'
else:
    strategy = 'partial_masking'

applicator = LDiversityModelApplicator(processor, strategy=strategy)
anonymized = applicator.apply(df, quasi_ids, sensitive_attrs)
```

### Example 3: Workflow with Verification

```python
# Setup
processor = LDiversityCalculator(l=3)
applicator = LDiversityModelApplicator(processor, strategy='full_masking')

# Evaluate before
before = processor.evaluate_privacy(df, quasi_ids)
print(f"Before: is_l_diverse={before['is_l_diverse']}")

# Apply
anonymized = applicator.apply(df, quasi_ids, sens_attrs)

# Verify
after = processor.evaluate_privacy(anonymized, quasi_ids)
assert after['is_l_diverse'], "Anonymization failed"
print(f"After: is_l_diverse={after['is_l_diverse']}")
print(f"Records retained: {len(anonymized)/len(df)*100:.1f}%")
```

## Best Practices

1. **Always Evaluate Before Applying:**
   ```python
   eval_result = processor.evaluate_privacy(df, quasi_ids)
   if eval_result['is_l_diverse']:
       print("Already l-diverse, application not needed")
   ```

2. **Verify After Application:**
   ```python
   anonymized = applicator.apply(df, quasi_ids, sens_attrs)
   verification = processor.evaluate_privacy(anonymized, quasi_ids)
   assert verification['is_l_diverse'], "Verification failed"
   ```

3. **Document Strategy Choice:**
   ```python
   metadata = {
       'strategy': strategy,
       'original_records': len(df),
       'anonymized_records': len(anonymized),
       'information_loss': (len(df) - len(anonymized)) / len(df) * 100
   }
   ```

4. **Monitor Data Quality:**
   ```python
   # Track information loss
   loss_pct = (1 - len(anonymized) / len(df)) * 100
   if loss_pct > 20:
       print(f"⚠️ High information loss: {loss_pct:.1f}%")
       # Consider less strict privacy requirements
   ```

## Related Components

- [LDiversityCalculator](./l_diversity_calculator.md)
- [Anonymization Strategies](./strategies.md)
