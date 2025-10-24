# PAMOLA.CORE Masking Operations Software Requirements Sub-Specification

**Document Version:** 1.0.0
**Parent Document:** PAMOLA.CORE Anonymization Package SRS v4.1.0
**Last Updated:** 2025-06-15
**Status:** Draft

## 1. Introduction

### 1.1 Purpose

This Software Requirements Sub-Specification (Sub-SRS) defines the detailed requirements for masking operations within the PAMOLA.CORE anonymization package. Masking operations hide or obfuscate sensitive data by replacing characters with masking symbols while preserving data format and partial information as needed.

### 1.2 Scope

This document covers two masking operations for MVP:
- **Full Masking Operation**: Complete replacement of field values with masking characters
- **Partial Masking Operation**: Selective masking while preserving specific portions of data

All operations follow the base anonymization framework defined in the parent SRS.

### 1.3 Document Conventions

- **REQ-MASK-XXX**: General masking requirements
- **REQ-FULL-XXX**: Full masking specific requirements
- **REQ-PARTIAL-XXX**: Partial masking specific requirements

## 2. Common Masking Requirements

### 2.1 Base Class Inheritance

**REQ-MASK-001 [MUST]** All masking operations SHALL inherit from `AnonymizationOperation` and follow the standard operation contract defined in the parent SRS (REQ-ANON-001).

### 2.2 Masking Character Support

**REQ-MASK-002 [MUST]** Operations SHALL support configurable masking characters:
- Default masking character: `*`
- Support for any single character or Unicode symbol
- Support for random character selection from a pool
- Validation that masking character differs from common data characters

### 2.3 Data Type Handling

**REQ-MASK-003 [MUST]** Masking operations SHALL handle multiple data types:
- **Strings**: Direct character replacement
- **Numbers**: Convert to string, mask, optionally convert back
- **Dates**: Format-aware masking (preserve separators)
- **Complex Types**: JSON, structured data with nested masking

### 2.4 Format Preservation

**REQ-MASK-004 [MUST]** Operations SHALL preserve data format when required:
- Maintain separators (-, /, ., spaces)
- Preserve length or use fixed length
- Maintain data type structure for validation

## 3. Full Masking Operation

### 3.1 Overview

**REQ-FULL-001 [MUST]** The `FullMaskingOperation` replaces entire field values with masking characters, providing complete obfuscation.

### 3.2 Constructor Interface

**REQ-FULL-002 [MUST]** Constructor signature:

```python
class FullMaskingOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Field to mask
                 # Masking configuration
                 mask_char: str = "*",               # Character to use for masking
                 preserve_length: bool = True,       # Keep original length
                 fixed_length: Optional[int] = None, # Use fixed mask length
                 random_mask: bool = False,          # Use random characters
                 mask_char_pool: Optional[str] = None,  # Pool for random selection
                 # Format handling
                 preserve_format: bool = False,      # Keep separators/structure
                 format_patterns: Optional[Dict[str, str]] = None,  # Custom patterns
                 # Type-specific handling
                 numeric_output: str = "string",     # string, numeric, preserve
                 date_format: Optional[str] = None,  # For date parsing
                 # Conditional masking
                 condition_field: Optional[str] = None,
                 condition_operator: Optional[str] = None,
                 condition_value: Optional[Any] = None,
                 condition_list: Optional[List[Dict[str, Any]]] = None,
                 # K-anonymity integration
                 ka_risk_field: Optional[str] = None,
                 risk_threshold: float = 5.0,
                 vulnerable_record_strategy: str = "mask",
                 # Standard parameters
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 3.3 Masking Implementation

**REQ-FULL-003 [MUST]** Implement comprehensive masking logic:

```python
def _mask_value(self, value: Any) -> str:
    """Apply full masking to a single value."""
    if pd.isna(value) and self.null_strategy == "PRESERVE":
        return value
    
    # Convert to string for masking
    str_value = str(value)
    
    # Determine mask length
    if self.fixed_length:
        mask_length = self.fixed_length
    elif self.preserve_length:
        mask_length = len(str_value)
    else:
        mask_length = 8  # Default fixed length
    
    # Generate mask
    if self.random_mask:
        if self.mask_char_pool:
            mask = ''.join(random.choice(self.mask_char_pool) 
                          for _ in range(mask_length))
        else:
            # Default pool of safe characters
            pool = string.ascii_letters + string.digits + "!@#$%^&*"
            mask = ''.join(random.choice(pool) 
                          for _ in range(mask_length))
    else:
        mask = self.mask_char * mask_length
    
    # Handle numeric output if needed
    if self.numeric_output == "numeric" and str_value.replace('.', '').isdigit():
        # Convert mask to numeric representation
        return self._mask_to_numeric(mask)
    
    return mask
```

### 3.4 Format Preservation

**REQ-FULL-004 [SHOULD]** Support format preservation for structured data:

```python
def _mask_with_format(self, value: str) -> str:
    """Mask while preserving format structure."""
    if not self.preserve_format:
        return self._mask_value(value)
    
    # Common format patterns
    default_patterns = {
        'phone': r'(\d{3})-(\d{3})-(\d{4})',
        'ssn': r'(\d{3})-(\d{2})-(\d{4})',
        'credit_card': r'(\d{4})-(\d{4})-(\d{4})-(\d{4})',
        'email': r'([^@]+)@([^.]+)\.(.+)',
        'date': r'(\d{4})-(\d{2})-(\d{2})'
    }
    
    patterns = self.format_patterns or default_patterns
    
    # Try to match patterns
    for pattern_name, pattern in patterns.items():
        match = re.match(pattern, value)
        if match:
            # Mask each group separately
            masked_groups = []
            for group in match.groups():
                if group.isdigit():
                    masked_groups.append(self.mask_char * len(group))
                else:
                    # Preserve non-digit groups (like @ in email)
                    masked_groups.append(group)
            
            # Reconstruct with original separators
            return re.sub(pattern, 
                         lambda m: self._reconstruct_format(m, masked_groups),
                         value)
    
    # No pattern matched, use default masking
    return self._mask_value(value)
```

### 3.5 Batch Processing

**REQ-FULL-005 [MUST]** Process batches efficiently:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process batch with full masking."""
    result = batch.copy()
    
    # Apply conditional filtering if specified
    if self.condition_field:
        mask_indices = self._get_conditional_indices(batch)
    else:
        mask_indices = batch.index
    
    # Apply k-anonymity risk filtering if specified
    if self.ka_risk_field and self.ka_risk_field in batch.columns:
        risk_indices = batch[batch[self.ka_risk_field] < self.risk_threshold].index
        mask_indices = mask_indices.intersection(risk_indices)
    
    # Process target field
    output_col = self.output_field_name or self.field_name
    
    if self.preserve_format and self._is_string_field(batch[self.field_name]):
        # Format-aware masking
        result.loc[mask_indices, output_col] = batch.loc[mask_indices, self.field_name].apply(
            self._mask_with_format
        )
    else:
        # Standard masking
        result.loc[mask_indices, output_col] = batch.loc[mask_indices, self.field_name].apply(
            self._mask_value
        )
    
    return result
```

### 3.6 Type-Specific Handling

**REQ-FULL-006 [MUST]** Handle different data types appropriately:

```python
def _handle_numeric_field(self, series: pd.Series) -> pd.Series:
    """Handle numeric fields based on configuration."""
    if self.numeric_output == "preserve":
        # Keep as numeric with special values
        return series.apply(lambda x: np.nan if pd.notna(x) else x)
    elif self.numeric_output == "numeric":
        # Convert to numeric mask (e.g., 9999999)
        return series.apply(lambda x: int('9' * len(str(int(x)))) if pd.notna(x) else x)
    else:  # string
        # Standard string masking
        return series.apply(self._mask_value)

def _handle_date_field(self, series: pd.Series) -> pd.Series:
    """Handle date fields with format awareness."""
    if self.date_format:
        # Parse dates and mask components
        return series.apply(
            lambda x: self._mask_date_components(x, self.date_format) 
            if pd.notna(x) else x
        )
    else:
        # Convert to string and mask
        return series.apply(self._mask_value)
```

### 3.7 Metrics Collection

**REQ-FULL-007 [MUST]** Collect masking-specific metrics:
- `values_masked`: Number of values masked
- `masking_rate`: Percentage of non-null values masked
- `format_preserved_count`: Number of format-preserved masks
- `conditional_mask_count`: Values masked due to conditions
- `risk_based_mask_count`: Values masked due to k-anonymity risk

## 4. Partial Masking Operation

### 4.1 Overview

**REQ-PARTIAL-001 [MUST]** The `PartialMaskingOperation` selectively masks portions of data while preserving specified parts for utility.

### 4.2 Constructor Interface

**REQ-PARTIAL-002 [MUST]** Constructor signature:

```python
class PartialMaskingOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Field to mask
                 # Masking configuration
                 mask_char: str = "*",               # Character for masking
                 # Position-based masking
                 unmasked_prefix: int = 0,           # Characters to keep at start
                 unmasked_suffix: int = 0,           # Characters to keep at end
                 unmasked_positions: Optional[List[int]] = None,  # Specific positions
                 # Pattern-based masking
                 mask_pattern: Optional[str] = None,  # Regex pattern to mask
                 preserve_pattern: Optional[str] = None,  # Regex pattern to preserve
                 # Common patterns
                 pattern_type: Optional[str] = None,  # email, phone, ssn, credit_card
                 # Advanced masking
                 mask_percentage: Optional[float] = None,  # Random % to mask
                 mask_strategy: str = "fixed",       # fixed, random, pattern
                 # Type-specific handling
                 case_sensitive: bool = True,        # For string operations
                 preserve_word_boundaries: bool = False,  # Keep word structure
                 # Multi-field consistency
                 consistency_fields: Optional[List[str]] = None,  # Apply same mask
                 # Conditional masking
                 condition_field: Optional[str] = None,
                 condition_operator: Optional[str] = None,
                 condition_value: Optional[Any] = None,
                 condition_list: Optional[List[Dict[str, Any]]] = None,
                 # K-anonymity integration
                 ka_risk_field: Optional[str] = None,
                 risk_threshold: float = 5.0,
                 vulnerable_record_strategy: str = "full_mask",
                 # Standard parameters
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 4.3 Pattern Library

**REQ-PARTIAL-003 [MUST]** Implement common masking patterns:

```python
class MaskingPatterns:
    """Library of common partial masking patterns."""
    
    PATTERNS = {
        'email': {
            'regex': r'^([^@]{2})([^@]+)(@.+)$',
            'mask_groups': [2],  # Mask middle part
            'description': 'Keep first 2 chars and domain'
        },
        'phone': {
            'regex': r'^(\d{3})-?(\d{3})-?(\d{4})$',
            'mask_groups': [2],  # Mask middle 3 digits
            'description': 'Keep area code and last 4'
        },
        'phone_international': {
            'regex': r'^(\+\d{1,3})-?(\d+)-?(\d{4})$',
            'mask_groups': [2],  # Mask middle section
            'description': 'Keep country code and last 4'
        },
        'ssn': {
            'regex': r'^(\d{3})-?(\d{2})-?(\d{4})$',
            'mask_groups': [1, 2],  # Mask first two groups
            'description': 'Keep last 4 digits only'
        },
        'credit_card': {
            'regex': r'^(\d{4})-?(\d{4})-?(\d{4})-?(\d{4})$',
            'mask_groups': [2, 3],  # Mask middle 8 digits
            'description': 'Keep first 4 and last 4'
        },
        'ip_address': {
            'regex': r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$',
            'mask_groups': [3, 4],  # Mask last two octets
            'description': 'Keep first two octets'
        },
        'account_number': {
            'regex': r'^(.{2})(.+)(.{4})$',
            'mask_groups': [2],  # Mask middle section
            'description': 'Keep first 2 and last 4 chars'
        }
    }
    
    @classmethod
    def get_pattern(cls, pattern_type: str) -> Dict[str, Any]:
        """Get pattern configuration by type."""
        return cls.PATTERNS.get(pattern_type, {})
```

### 4.4 Partial Masking Logic

**REQ-PARTIAL-004 [MUST]** Implement flexible partial masking:

```python
def _apply_partial_mask(self, value: str) -> str:
    """Apply partial masking based on configuration."""
    if pd.isna(value) and self.null_strategy == "PRESERVE":
        return value
    
    str_value = str(value)
    
    # Strategy 1: Position-based masking
    if self.mask_strategy == "fixed":
        return self._position_based_mask(str_value)
    
    # Strategy 2: Pattern-based masking
    elif self.mask_strategy == "pattern":
        return self._pattern_based_mask(str_value)
    
    # Strategy 3: Random percentage masking
    elif self.mask_strategy == "random":
        return self._random_percentage_mask(str_value)
    
    # Strategy 4: Word boundary preserving
    elif self.mask_strategy == "words":
        return self._word_based_mask(str_value)
    
    return str_value

def _position_based_mask(self, value: str) -> str:
    """Mask based on position configuration."""
    if not value:
        return value
    
    result = list(value)
    value_len = len(value)
    
    # Determine positions to mask
    if self.unmasked_positions:
        # Mask everything except specified positions
        for i in range(value_len):
            if i not in self.unmasked_positions:
                result[i] = self.mask_char
    else:
        # Use prefix/suffix configuration
        mask_start = self.unmasked_prefix
        mask_end = value_len - self.unmasked_suffix
        
        if mask_start < mask_end:
            for i in range(mask_start, mask_end):
                result[i] = self.mask_char
    
    return ''.join(result)

def _pattern_based_mask(self, value: str) -> str:
    """Mask based on regex patterns."""
    # Use predefined pattern if specified
    if self.pattern_type:
        pattern_config = MaskingPatterns.get_pattern(self.pattern_type)
        if pattern_config:
            return self._apply_pattern_config(value, pattern_config)
    
    # Use custom patterns
    if self.mask_pattern:
        # Replace all matches with masks
        return re.sub(
            self.mask_pattern,
            lambda m: self.mask_char * len(m.group()),
            value
        )
    
    if self.preserve_pattern:
        # Mask everything except matches
        matches = list(re.finditer(self.preserve_pattern, value))
        result = [self.mask_char] * len(value)
        
        for match in matches:
            for i in range(match.start(), match.end()):
                result[i] = value[i]
        
        return ''.join(result)
    
    return value
```

### 4.5 Consistency Across Fields

**REQ-PARTIAL-005 [SHOULD]** Maintain consistency when masking related fields:

```python
def _create_consistency_map(self, batch: pd.DataFrame) -> Dict[str, str]:
    """Create mapping for consistent masking across fields."""
    consistency_map = {}
    
    if not self.consistency_fields:
        return consistency_map
    
    # Get unique values from all consistency fields
    all_values = set()
    for field in [self.field_name] + self.consistency_fields:
        if field in batch.columns:
            all_values.update(batch[field].dropna().unique())
    
    # Create consistent masks for each unique value
    for value in all_values:
        str_value = str(value)
        masked = self._apply_partial_mask(str_value)
        consistency_map[str_value] = masked
    
    return consistency_map

def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process with consistency support."""
    result = batch.copy()
    
    # Create consistency map if needed
    if self.consistency_fields:
        consistency_map = self._create_consistency_map(batch)
        
        # Apply to all fields
        for field in [self.field_name] + self.consistency_fields:
            if field in result.columns:
                output_field = f"{field}_masked" if field != self.field_name else (self.output_field_name or field)
                result[output_field] = result[field].apply(
                    lambda x: consistency_map.get(str(x), x) if pd.notna(x) else x
                )
    else:
        # Standard processing
        output_col = self.output_field_name or self.field_name
        result[output_col] = result[self.field_name].apply(self._apply_partial_mask)
    
    return result
```

### 4.6 Advanced Masking Strategies

**REQ-PARTIAL-006 [SHOULD]** Implement advanced masking strategies:

```python
def _random_percentage_mask(self, value: str) -> str:
    """Mask random percentage of characters."""
    if not value or not self.mask_percentage:
        return value
    
    value_len = len(value)
    num_to_mask = int(value_len * self.mask_percentage)
    
    if num_to_mask == 0:
        return value
    
    # Select random positions to mask
    positions = random.sample(range(value_len), num_to_mask)
    
    result = list(value)
    for pos in positions:
        result[pos] = self.mask_char
    
    return ''.join(result)

def _word_based_mask(self, value: str) -> str:
    """Mask while preserving word boundaries."""
    if not self.preserve_word_boundaries:
        return self._apply_partial_mask(value)
    
    # Split into words
    words = value.split()
    
    # Mask each word individually
    masked_words = []
    for word in words:
        if len(word) <= 3:
            # Short words fully masked
            masked_words.append(self.mask_char * len(word))
        else:
            # Longer words partially masked
            masked = self._position_based_mask(word)
            masked_words.append(masked)
    
    return ' '.join(masked_words)
```

### 4.7 Metrics Collection

**REQ-PARTIAL-007 [MUST]** Collect partial masking metrics:
- `partial_mask_rate`: Percentage of characters masked per value
- `pattern_matches`: Number of pattern-based masks applied
- `consistency_fields_processed`: Number of fields with consistent masking
- `average_visibility`: Average percentage of data remaining visible
- `masking_strategy_distribution`: Count by strategy type

## 5. Common Masking Features

### 5.1 Validation

**REQ-MASK-005 [MUST]** Validate masking configuration:

```python
def validate_configuration(self) -> None:
    """Validate masking operation configuration."""
    super().validate_configuration()
    
    # Validate mask character
    if not self.mask_char or len(self.mask_char) != 1:
        raise ValueError("mask_char must be a single character")
    
    # Validate partial masking settings
    if isinstance(self, PartialMaskingOperation):
        if self.unmasked_prefix < 0 or self.unmasked_suffix < 0:
            raise ValueError("unmasked_prefix/suffix must be non-negative")
        
        if self.mask_percentage is not None:
            if not 0 <= self.mask_percentage <= 1:
                raise ValueError("mask_percentage must be between 0 and 1")
        
        if self.pattern_type and self.pattern_type not in MaskingPatterns.PATTERNS:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
```

### 5.2 Caching Support

**REQ-MASK-006 [SHOULD]** Implement caching for repeated values:

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    
    # Initialize cache if enabled
    if self.use_cache:
        self._mask_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

def _get_cached_mask(self, value: str) -> Optional[str]:
    """Get mask from cache if available."""
    if not self.use_cache:
        return None
    
    cache_key = f"{value}:{self.mask_strategy}"
    if cache_key in self._mask_cache:
        self._cache_hits += 1
        return self._mask_cache[cache_key]
    
    self._cache_misses += 1
    return None

def _cache_mask(self, value: str, masked: str) -> None:
    """Cache masked value."""
    if self.use_cache:
        cache_key = f"{value}:{self.mask_strategy}"
        self._mask_cache[cache_key] = masked
```

### 5.3 Visualization

**REQ-MASK-007 [SHOULD]** Generate masking visualizations:

```python
def _create_visualization(self, metrics: Dict[str, Any]) -> None:
    """Create masking operation visualization."""
    # Sample before/after comparison
    sample_data = []
    for original, masked in self._sample_pairs[:10]:
        sample_data.append({
            'Original': original[:50] + '...' if len(original) > 50 else original,
            'Masked': masked[:50] + '...' if len(masked) > 50 else masked,
            'Visibility': f"{(len([c for c in masked if c != self.mask_char]) / len(masked) * 100):.1f}%"
        })
    
    # Create comparison table visualization
    self._create_comparison_table(sample_data)
    
    # Create masking distribution chart if partial
    if isinstance(self, PartialMaskingOperation):
        self._create_visibility_distribution()
```

## 6. Entity-Specific Masking Configurations

### 6.1 Email Masking

**REQ-MASK-008 [SHOULD]** Provide optimized email masking:

```python
class EmailMaskingPresets:
    """Predefined configurations for email masking."""
    
    FULL_DOMAIN = {
        'pattern_type': 'email',
        'unmasked_prefix': 2,
        'preserve_pattern': r'@[\w\.-]+\.[a-zA-Z]{2,}$'
    }
    
    PARTIAL_DOMAIN = {
        'mask_pattern': r'[^@]+(?=@)',  # Mask local part
        'preserve_pattern': r'@[\w\.-]+(?=\.)'  # Keep domain minus TLD
    }
    
    PRIVACY_FOCUSED = {
        'unmasked_prefix': 1,
        'mask_pattern': r'[^@]+',
        'fixed_length': 8
    }
```

### 6.2 Phone Number Masking

**REQ-MASK-009 [SHOULD]** Support international phone formats:

```python
class PhoneMaskingPresets:
    """Phone number masking configurations."""
    
    US_STANDARD = {
        'pattern_type': 'phone',
        'preserve_format': True,
        'format_patterns': {
            'us': r'(\d{3})-(\d{3})-(\d{4})',
            'us_dots': r'(\d{3})\.(\d{3})\.(\d{4})',
            'us_parens': r'\((\d{3})\) (\d{3})-(\d{4})'
        }
    }
    
    INTERNATIONAL = {
        'pattern_type': 'phone_international',
        'unmasked_prefix': 4,  # Keep country code
        'unmasked_suffix': 4   # Keep last 4
    }
```

### 6.3 Credit Card Masking

**REQ-MASK-010 [SHOULD]** Implement PCI-compliant masking:

```python
class CreditCardMaskingPresets:
    """Credit card masking following PCI standards."""
    
    PCI_COMPLIANT = {
        'pattern_type': 'credit_card',
        'preserve_format': True,
        'unmasked_prefix': 6,   # First 6 (BIN)
        'unmasked_suffix': 4    # Last 4
    }
    
    STRICT = {
        'unmasked_suffix': 4,   # Only last 4
        'fixed_length': 16,
        'mask_char': 'X'
    }
```

## 7. Performance Optimization

### 7.1 Vectorization

**REQ-MASK-011 [SHOULD]** Use vectorized operations where possible:

```python
def _vectorized_mask(self, series: pd.Series) -> pd.Series:
    """Apply masking using vectorized operations."""
    if self.mask_strategy == "fixed" and not self.preserve_format:
        # Simple position-based masking can be vectorized
        if self.fixed_length:
            return pd.Series([self.mask_char * self.fixed_length] * len(series),
                           index=series.index)
        
        # Use string methods for efficiency
        if self.unmasked_prefix == 0 and self.unmasked_suffix > 0:
            # Keep only suffix
            return series.str[-self.unmasked_suffix:].str.pad(
                width=series.str.len(), 
                side='left', 
                fillchar=self.mask_char
            )
    
    # Fall back to apply for complex logic
    return series.apply(self._apply_mask_function)
```

### 7.2 Parallel Processing

**REQ-MASK-012 [SHOULD]** Support parallel processing for large datasets:

```python
def _process_batch_dask(self, ddf: dd.DataFrame) -> dd.DataFrame:
    """Process Dask DataFrame with masking."""
    import dask.dataframe as dd
    
    # Define masking function for map_partitions
    def mask_partition(partition):
        return self.process_batch(partition)
    
    # Apply masking to each partition
    result = ddf.map_partitions(
        mask_partition,
        meta=ddf._meta
    )
    
    return result
```

## 8. Security Considerations

### 8.1 Mask Character Validation

**REQ-MASK-013 [MUST]** Validate mask characters don't reveal information:

```python
def _validate_mask_security(self) -> None:
    """Ensure mask character doesn't compromise security."""
    # Forbidden mask characters that might reveal information
    forbidden_chars = set('0123456789') | set(string.ascii_letters)
    
    if self.mask_char in forbidden_chars and not self.random_mask:
        logger.warning(
            f"Mask character '{self.mask_char}' might reveal information. "
            "Consider using symbols like *, #, or X"
        )
```

### 8.2 Pattern Security

**REQ-MASK-014 [MUST]** Validate masking patterns maintain privacy:

```python
def _validate_pattern_security(self) -> None:
    """Ensure patterns don't expose too much information."""
    if isinstance(self, PartialMaskingOperation):
        # Calculate worst-case visibility
        if self.unmasked_prefix + self.unmasked_suffix > 10:
            logger.warning(
                "High visibility configuration: "
                f"{self.unmasked_prefix + self.unmasked_suffix} characters visible"
            )
        
        # Check pattern-based exposure
        if self.preserve_pattern:
            # Estimate exposure based on pattern
            test_values = [
                "test@example.com",
                "123-45-6789",
                "4111-1111-1111-1111"
            ]
            
            for test in test_values:
                masked = self._pattern_based_mask(test)
                visibility = len([c for c in masked if c != self.mask_char]) / len(masked)
                if visibility > 0.5:
                    logger.warning(
                        f"Pattern may expose >50% of data: {self.preserve_pattern}"
                    )
                    break
```

## 9. Testing Requirements

### 9.1 Unit Tests

**REQ-MASK-015 [MUST]** Test coverage must include:

1. **Full Masking Tests**:
   - All data types (string, numeric, date)
   - Format preservation
   - Length preservation vs fixed length
   - Random masking
   - Null handling

2. **Partial Masking Tests**:
   - Position-based masking
   - Pattern-based masking
   - Consistency across fields
   - All preset patterns
   - Edge cases (empty, single char)

### 9.2 Pattern Tests

**REQ-MASK-016 [MUST]** Test all predefined patterns:

```python
def test_email_pattern_masking():
    """Test email masking patterns."""
    op = PartialMaskingOperation(
        field_name="email",
        pattern_type="email"
    )
    
    test_cases = [
        ("user@example.com", "us**@example.com"),
        ("a@domain.org", "a@domain.org"),  # Too short to mask
        ("longusername@company.co.uk", "lo**********@company.co.uk")
    ]
    
    for email, expected_pattern in test_cases:
        masked = op._apply_partial_mask(email)
        # Verify pattern matches expected format
        assert masked.endswith("@example.com") or masked.endswith("@domain.org")
        assert masked.count('*') > 0 or len(email) <= 3
```

## 10. Example Implementations

### 10.1 Complete Full Masking Implementation

```python
class FullMaskingOperation(AnonymizationOperation):
    """Complete field value masking operation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_mask_security()
        
        # Initialize format patterns
        self._setup_format_patterns()
        
        # Cache for performance
        if self.use_cache:
            self._mask_cache = {}
    
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Apply full masking to batch."""
        result = batch.copy()
        
        # Determine records to mask
        mask_indices = self._get_masking_indices(batch)
        
        # Apply masking
        output_col = self.output_field_name or self.field_name
        
        if len(mask_indices) > 0:
            if self._can_vectorize():
                result.loc[mask_indices, output_col] = self._vectorized_mask(
                    batch.loc[mask_indices, self.field_name]
                )
            else:
                result.loc[mask_indices, output_col] = batch.loc[mask_indices, self.field_name].apply(
                    self._mask_value
                )
        
        return result
    
    def _collect_specific_metrics(self, original_data: pd.Series,
                                 anonymized_data: pd.Series) -> Dict[str, Any]:
        """Collect masking-specific metrics."""
        # Count masked values
        masked_count = (anonymized_data != original_data).sum()
        
        return {
            "values_masked": masked_count,
            "masking_rate": masked_count / len(original_data) if len(original_data) > 0 else 0,
            "mask_character": self.mask_char,
            "preserve_length": self.preserve_length,
            "fixed_length": self.fixed_length
        }
```

### 10.2 Complete Partial Masking Implementation

```python
class PartialMaskingOperation(AnonymizationOperation):
    """Selective field value masking operation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Setup pattern library
        self._pattern_config = None
        if self.pattern_type:
            self._pattern_config = MaskingPatterns.get_pattern(self.pattern_type)
        
        # Initialize consistency tracking
        self._consistency_map = {}
        
        # Sample pairs for visualization
        self._sample_pairs = []
    
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Apply partial masking with consistency support."""
        result = batch.copy()
        
        # Build consistency map if needed
        if self.consistency_fields:
            self._consistency_map = self._create_consistency_map(batch)
        
        # Process primary field
        output_col = self.output_field_name or self.field_name
        result[output_col] = self._apply_masking_with_cache(
            batch[self.field_name]
        )
        
        # Process consistency fields
        if self.consistency_fields:
            for field in self.consistency_fields:
                if field in batch.columns and field != self.field_name:
                    result[f"{field}_masked"] = self._apply_masking_with_cache(
                        batch[field]
                    )
        
        # Collect samples for visualization
        if len(self._sample_pairs) < 20:
            for idx in batch.index[:5]:
                if idx in result.index:
                    original = str(batch.loc[idx, self.field_name])
                    masked = str(result.loc[idx, output_col])
                    if original != masked:
                        self._sample_pairs.append((original, masked))
        
        return result
    
    def _apply_masking_with_cache(self, series: pd.Series) -> pd.Series:
        """Apply masking with caching support."""
        def mask_with_cache(value):
            if pd.isna(value):
                return value
            
            str_val = str(value)
            
            # Check cache
            if self.use_cache:
                cached = self._get_cached_mask(str_val)
                if cached is not None:
                    return cached
            
            # Apply masking
            if str_val in self._consistency_map:
                masked = self._consistency_map[str_val]
            else:
                masked = self._apply_partial_mask(str_val)
            
            # Cache result
            if self.use_cache:
                self._cache_mask(str_val, masked)
            
            return masked
        
        return series.apply(mask_with_cache)
    
    def _collect_specific_metrics(self, original_data: pd.Series,
                                 anonymized_data: pd.Series) -> Dict[str, Any]:
        """Collect partial masking metrics."""
        metrics = super()._collect_specific_metrics(original_data, anonymized_data)
        
        # Calculate average visibility
        visibility_scores = []
        for orig, anon in zip(original_data.dropna(), anonymized_data.dropna()):
            if str(orig) != str(anon):
                orig_str = str(orig)
                anon_str = str(anon)
                if len(orig_str) > 0:
                    visible_chars = sum(1 for i, c in enumerate(anon_str) 
                                      if i < len(orig_str) and c != self.mask_char)
                    visibility_scores.append(visible_chars / len(orig_str))
        
        avg_visibility = np.mean(visibility_scores) if visibility_scores else 0
        
        metrics.update({
            "partial_mask_rate": len(visibility_scores) / len(original_data),
            "average_visibility": avg_visibility,
            "mask_strategy": self.mask_strategy,
            "pattern_type": self.pattern_type,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) 
                             if self.use_cache and (self._cache_hits + self._cache_misses) > 0 else 0
        })
        
        return metrics
```

## 11. Integration Examples

### 11.1 Basic Usage

```python
# Full masking example
full_mask_op = FullMaskingOperation(
    field_name="credit_card",
    mask_char="X",
    preserve_format=True,
    fixed_length=None,
    preserve_length=True
)

# Partial masking example
partial_mask_op = PartialMaskingOperation(
    field_name="email",
    pattern_type="email",
    mask_char="*",
    consistency_fields=["alternate_email", "recovery_email"]
)

# Execute operations
masked_data = full_mask_op.execute(data_source, task_dir)
```

### 11.2 Advanced Configuration

```python
# Risk-based masking
risk_mask_op = PartialMaskingOperation(
    field_name="phone",
    pattern_type="phone",
    ka_risk_field="k_anonymity_phone",
    risk_threshold=3.0,
    vulnerable_record_strategy="full_mask",
    # High-risk records get full masking
    unmasked_prefix=0,
    unmasked_suffix=0
)

# Multi-condition masking
conditional_mask_op = FullMaskingOperation(
    field_name="salary",
    condition_list=[
        {"field": "department", "operator": "in", "value": ["HR", "Finance"]},
        {"field": "level", "operator": ">=", "value": 5}
    ],
    numeric_output="numeric",
    fixed_length=7,
    mask_char="9"
)
```

## 12. Summary

The masking operations provide flexible data obfuscation through:

- **Full Masking**: Complete value replacement with configurable format preservation
- **Partial Masking**: Selective character masking with pattern support and consistency

Both operations integrate with the PAMOLA.CORE framework, support conditional processing, handle multiple data types, and provide comprehensive metrics. The implementation prioritizes security, performance, and usability while maintaining the framework's architectural principles.