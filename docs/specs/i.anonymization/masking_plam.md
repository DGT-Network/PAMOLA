# PAMOLA.CORE Masking Operations Implementation Plan

## Package Structure

```
pamola_core/anonymization/masking/
├── __init__.py                 # Package exports
├── full_masking.py            # Full masking operation
├── partial_masking.py         # Partial masking operation
├── patterns.py                # Masking patterns library
└── presets.py                 # Entity-specific presets
```

## Implementation Modules

### 1. **patterns.py** - Masking Patterns Library
```python
# Core pattern definitions
class MaskingPatterns:
    PATTERNS = {
        'email': {...},
        'phone': {...},
        'ssn': {...},
        'credit_card': {...},
        'ip_address': {...}
    }
    
# Pattern matching utilities
def apply_pattern_mask(value: str, pattern_config: dict, mask_char: str) -> str
def detect_pattern_type(value: str) -> Optional[str]
```

### 2. **presets.py** - Entity-Specific Configurations
```python
# Preset configurations for common entities
class EmailMaskingPresets:
    FULL_DOMAIN = {...}
    PARTIAL_DOMAIN = {...}
    PRIVACY_FOCUSED = {...}

class PhoneMaskingPresets:
    US_STANDARD = {...}
    INTERNATIONAL = {...}

class CreditCardMaskingPresets:
    PCI_COMPLIANT = {...}
    STRICT = {...}

class SSNMaskingPresets:
    LAST_FOUR = {...}
    FULL_MASK = {...}
```

### 3. **full_masking.py** - Full Masking Operation
```python
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.validation import (
    FieldExistsValidator,
    validation_handler,
    requires_field
)
from pamola_core.anonymization.commons.data_utils import (
    process_nulls,
    filter_records_conditionally
)

class FullMaskingOperation(AnonymizationOperation):
    def __init__(self, **kwargs):
        # Use base class initialization
        super().__init__(**kwargs)
        
    def _setup_operation(self):
        # Initialize format patterns from presets
        # Setup mask character validation
        # Configure caching if enabled
        
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Use commons.data_utils.filter_records_conditionally for conditions
        # Apply masking logic
        # Return transformed batch
        
    def _mask_value(self, value: Any) -> str:
        # Core masking logic
        # Format preservation if configured
        # Type-specific handling
        
    def _collect_specific_metrics(self, original: pd.Series, 
                                 anonymized: pd.Series) -> Dict[str, Any]:
        # Use commons.metric_utils for standard metrics
        # Add masking-specific metrics
```

### 4. **partial_masking.py** - Partial Masking Operation
```python
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.validation import (
    NumericRangeValidator,
    PatternValidator,
    validate_types
)
from pamola_core.anonymization.commons.text_processing_utils import (
    normalize_text,
    fuzzy_match
)
from .patterns import MaskingPatterns, apply_pattern_mask

class PartialMaskingOperation(AnonymizationOperation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _setup_operation(self):
        # Load pattern configurations
        # Setup consistency mapping
        # Initialize position validators
        
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Build consistency map using commons.category_utils
        # Apply partial masking strategies
        # Handle multi-field consistency
        
    def _apply_partial_mask(self, value: str) -> str:
        # Router for different strategies
        # Use patterns.apply_pattern_mask for pattern-based
        # Position-based logic
        # Random percentage logic
        
    def _create_consistency_map(self, batch: pd.DataFrame) -> Dict[str, str]:
        # Use commons.category_utils.get_category_distribution
        # Create consistent masks for unique values
```

### 5. **__init__.py** - Package Exports
```python
from .full_masking import FullMaskingOperation
from .partial_masking import PartialMaskingOperation
from .patterns import MaskingPatterns
from .presets import (
    EmailMaskingPresets,
    PhoneMaskingPresets,
    CreditCardMaskingPresets,
    SSNMaskingPresets
)

__all__ = [
    'FullMaskingOperation',
    'PartialMaskingOperation',
    'MaskingPatterns',
    'EmailMaskingPresets',
    'PhoneMaskingPresets',
    'CreditCardMaskingPresets',
    'SSNMaskingPresets'
]
```

## Integration Points with Commons

### 1. **Validation** (commons.validation)
- `FieldExistsValidator` - Field existence checks
- `PatternValidator` - Regex pattern validation
- `NumericRangeValidator` - Validate prefix/suffix lengths
- `@validation_handler` - Error handling decorator
- `@requires_field` - Field requirement decorator

### 2. **Data Processing** (commons.data_utils)
- `process_nulls()` - Null value handling strategies
- `filter_records_conditionally()` - Conditional masking
- `handle_vulnerable_records()` - K-anonymity risk handling

### 3. **Text Processing** (commons.text_processing_utils)
- `normalize_text()` - Text normalization before masking
- `detect_encoding()` - Handle various text encodings
- Pattern matching utilities

### 4. **Category Analysis** (commons.category_utils)
- `get_category_distribution()` - For consistency mapping
- `analyze_category_statistics()` - Distribution metrics

### 5. **Metrics** (commons.metric_utils)
- `calculate_information_loss()` - Privacy metrics
- `calculate_suppression_rate()` - Masking coverage
- Standard anonymization metrics

### 6. **Visualization** (commons.visualization_utils)
- `create_value_comparison()` - Before/after samples
- `create_distribution_chart()` - Masking distribution
- `register_visualization_artifact()` - Result integration

## Implementation Order

### Phase 1: Foundation
1. Create package structure
2. Implement `patterns.py` with core pattern definitions
3. Implement `presets.py` with entity configurations

### Phase 2: Full Masking
1. Implement `FullMaskingOperation` class structure
2. Add `_mask_value()` core logic
3. Integrate format preservation
4. Add type-specific handlers
5. Implement metrics collection

### Phase 3: Partial Masking
1. Implement `PartialMaskingOperation` class structure
2. Add position-based masking
3. Add pattern-based masking using patterns.py
4. Implement consistency mapping
5. Add advanced strategies (random, word-based)

### Phase 4: Integration
1. Complete commons integration points
2. Add caching mechanisms
3. Implement Dask support in process_batch_dask
4. Add comprehensive validation
5. Create __init__.py with proper exports

## Key Design Decisions

### 1. **No Separate Base Class**
- Inherit directly from `pamola_core.anonymization.base_anonymization_op.AnonymizationOperation`
- Leverage all base functionality (batch processing, metrics, visualization)

### 2. **Maximum Commons Reuse**
- Use commons validation instead of custom validators
- Leverage data_utils for all data processing
- Use standard metric calculations from metric_utils

### 3. **Modular Pattern System**
- Separate pattern definitions (patterns.py) from operations
- Presets provide ready-to-use configurations
- Easy to extend with new patterns

### 4. **Performance Focus**
- Built-in caching using base class cache mechanism
- Vectorized operations where possible
- Minimal overhead on top of base class

### 5. **Security by Design**
- Validation of mask characters
- Pattern security checks in validators
- No logging of sensitive values