# PAMOLA.CORE Text Processing Utilities for Anonymization Documentation

## Overview

The `text_processing_utils.py` module is a specialized component of the PAMOLA.CORE anonymization package that provides text processing utilities specifically tailored for privacy-preserving operations. Built on top of the general-purpose `pamola_core.utils.nlp.text_utils` module, it extends and adapts text processing functions with anonymization-specific behaviors, defaults, and safety constraints.

### Purpose

This module addresses the unique requirements of text processing in anonymization contexts:
- **Safe Category Generation**: Ensuring category names are database-safe and don't leak information
- **Hierarchy Management**: Supporting hierarchical generalization with level-appropriate constraints
- **Composite Value Handling**: Managing multi-value fields common in real-world data
- **Privacy-Aware Matching**: Fuzzy matching that doesn't compromise privacy
- **Controlled Tokenization**: Limiting token extraction to prevent information leakage

## Key Features

### 1. **Anonymization-Safe Text Processing**
- More restrictive character sets for field naming
- Automatic fallback to safe default values ("OTHER", "SUPPRESSED")
- Length constraints appropriate for anonymized fields
- Removal of stopwords disabled by default to preserve category integrity

### 2. **Enhanced Category Matching**
- Preparation functions that remove common prefixes/suffixes
- Configurable fallback strategies for unmatched values
- Support for composite category merging
- Hierarchy-aware validation

### 3. **Privacy-Preserving Constraints**
- Limited number of components in composite values
- Maximum token extraction limits
- Shorter default lengths for anonymized categories
- Safe hierarchy path generation

### 4. **Specialized Anonymization Functions**
- Merge strategies for multiple category assignments
- Value preparation for improved matching accuracy
- Hierarchy level validation
- Safe path representation for hierarchical data

## Dependencies

### Internal Dependencies
- `pamola_core.utils.nlp.text_utils`: Base text processing functions
  - `normalize_text`
  - `clean_category_name`
  - `calculate_string_similarity`
  - `find_closest_match`
  - `find_closest_category`
  - `split_composite_value`
  - `extract_tokens`
  - `truncate_text`
  - `is_valid_category_name`

### External Dependencies
- `re`: Regular expression operations
- `logging`: Structured logging
- `typing`: Type hints for better code clarity

## Component Architecture

```
text_processing_utils.py
├── Wrapped Functions (Anonymization-Aware)
│   ├── normalize_text()              # No stopword removal by default
│   ├── clean_category_name()         # More restrictive character set
│   ├── find_closest_category()       # With fallback handling
│   ├── split_composite_value()       # Component limit enforcement
│   └── extract_tokens()              # Token limit enforcement
├── Anonymization-Specific Functions
│   ├── merge_composite_categories()  # Multiple category handling
│   ├── prepare_value_for_matching()  # Pre-processing for matching
│   ├── validate_hierarchy_value()    # Level-aware validation
│   └── create_safe_hierarchy_path()  # Hierarchy visualization
└── Re-exported Functions
    ├── calculate_string_similarity()
    ├── find_closest_match()
    ├── truncate_text()
    └── is_valid_category_name()
```

## Function Signatures and Parameters

### Wrapped Functions with Anonymization Defaults

#### `normalize_text()`
```python
def normalize_text(
    text: str,
    level: str = "basic",
    preserve_case: bool = False
) -> str
```

**Parameters:**
- `text`: Text to normalize
- `level`: Normalization level ("none", "basic", "advanced", "aggressive")
- `preserve_case`: Whether to preserve original case

**Returns:** Normalized text (without stopword removal)

**Anonymization Note:** Stopword removal is disabled to preserve category matching accuracy

#### `clean_category_name()`
```python
def clean_category_name(
    name: str,
    max_length: int = MAX_CATEGORY_LENGTH,
    invalid_chars: str = ANONYMIZATION_SAFE_CHARS,
    separator: str = CATEGORY_SEPARATOR
) -> str
```

**Parameters:**
- `name`: Category name to clean
- `max_length`: Maximum length (default: 40)
- `invalid_chars`: More restrictive pattern for anonymization
- `separator`: Standard separator character (default: "_")

**Returns:** Cleaned category name or "OTHER" if empty

#### `find_closest_category()`
```python
def find_closest_category(
    value: str,
    categories: List[str],
    threshold: float = 0.8,
    method: str = "ratio",
    normalize_value: bool = True,
    fallback: str = DEFAULT_UNKNOWN_VALUE
) -> str
```

**Parameters:**
- `value`: Value to categorize
- `categories`: List of possible categories
- `threshold`: Minimum similarity threshold
- `method`: Similarity calculation method
- `normalize_value`: Whether to normalize input value
- `fallback`: Value to return if no match (default: "OTHER")

**Returns:** Best matching category or fallback value

#### `split_composite_value()`
```python
def split_composite_value(
    value: str,
    separators: Optional[List[str]] = None,
    normalize: bool = True,
    max_components: int = 10
) -> List[str]
```

**Parameters:**
- `value`: Composite value to split
- `separators`: List of separators (default: ["|", "/", ",", ";", "&"])
- `normalize`: Whether to normalize components
- `max_components`: Maximum components to return (privacy limit)

**Returns:** List of components (limited to max_components)

#### `extract_tokens()`
```python
def extract_tokens(
    text: str,
    min_length: int = 3,
    pattern: Optional[str] = None,
    lowercase: bool = True,
    max_tokens: int = 50
) -> List[str]
```

**Parameters:**
- `text`: Text to tokenize
- `min_length`: Minimum token length (default: 3, longer for privacy)
- `pattern`: Token extraction pattern
- `lowercase`: Whether to lowercase tokens
- `max_tokens`: Maximum tokens to return (privacy limit)

**Returns:** List of tokens (limited to max_tokens)

### Anonymization-Specific Functions

#### `merge_composite_categories()`
```python
def merge_composite_categories(
    categories: List[str],
    strategy: str = "first",
    separator: str = CATEGORY_SEPARATOR,
    max_length: int = MAX_CATEGORY_LENGTH
) -> str
```

**Parameters:**
- `categories`: List of categories to merge
- `strategy`: Merging strategy
  - `"first"`: Use first category
  - `"all"`: Concatenate all categories
  - `"most_specific"`: Use longest category
  - `"shortest"`: Use shortest category
- `separator`: Separator for "all" strategy
- `max_length`: Maximum length for result

**Returns:** Merged category name

#### `prepare_value_for_matching()`
```python
def prepare_value_for_matching(
    value: str,
    remove_common_prefixes: bool = True,
    remove_common_suffixes: bool = True
) -> str
```

**Parameters:**
- `value`: Value to prepare for matching
- `remove_common_prefixes`: Remove prefixes like "Senior", "Junior"
- `remove_common_suffixes`: Remove suffixes like "Manager", "Specialist"

**Returns:** Prepared value with common modifiers removed

#### `validate_hierarchy_value()`
```python
def validate_hierarchy_value(
    value: str,
    level: int,
    max_length_by_level: Optional[Dict[int, int]] = None
) -> Tuple[bool, Optional[str]]
```

**Parameters:**
- `value`: Value to validate
- `level`: Hierarchy level (0 = most general)
- `max_length_by_level`: Custom max lengths per level

**Returns:** Tuple of (is_valid, error_message)

#### `create_safe_hierarchy_path()`
```python
def create_safe_hierarchy_path(
    components: List[str],
    separator: str = " > ",
    max_length: int = 100
) -> str
```

**Parameters:**
- `components`: Hierarchy components from general to specific
- `separator`: Visual separator between levels
- `max_length`: Maximum total length

**Returns:** Safe hierarchy path string

## Usage Examples

### Example 1: Basic Category Cleaning for Anonymization

```python
from pamola_core.anonymization.commons.text_processing_utils import clean_category_name

# Clean potentially problematic category names
raw_category = "Sales & Marketing (2023)"
clean = clean_category_name(raw_category)
print(f"Clean: {clean}")  # Output: Clean: Sales_Marketing_2023

# Empty input returns safe default
empty_clean = clean_category_name("")
print(f"Empty: {empty_clean}")  # Output: Empty: OTHER
```

### Example 2: Category Matching with Fallback

```python
from pamola_core.anonymization.commons.text_processing_utils import (
    find_closest_category,
    prepare_value_for_matching
)

# Define generalization categories
categories = ["Engineering", "Management", "Sales", "Support"]

# Match with automatic fallback
job1 = "Senior Software Engineer"
match1 = find_closest_category(job1, categories, threshold=0.7)
print(f"Match 1: {match1}")  # Output: Match 1: Engineering

# No match returns fallback
job2 = "Janitor"
match2 = find_closest_category(job2, categories, threshold=0.8)
print(f"Match 2: {match2}")  # Output: Match 2: OTHER

# Improve matching by removing common prefixes
prepared_job = prepare_value_for_matching("Senior Engineering Manager")
print(f"Prepared: {prepared_job}")  # Output: Prepared: engineering
match3 = find_closest_category(prepared_job, categories)
print(f"Match 3: {match3}")  # Output: Match 3: Engineering
```

### Example 3: Handling Composite Values Safely

```python
from pamola_core.anonymization.commons.text_processing_utils import (
    split_composite_value,
    merge_composite_categories
)

# Split multi-department assignment
departments = "IT|Finance|HR|Legal|Marketing|Sales|R&D|QA|Support|Admin|Security"
components = split_composite_value(departments, max_components=5)
print(f"Components: {components}")  
# Output: Components: ['it', 'finance', 'hr', 'legal', 'marketing']
# Note: Limited to 5 components for privacy

# Merge multiple assignments
assignments = ["Junior Developer", "Senior Developer", "Team Lead"]
merged = merge_composite_categories(assignments, strategy="most_specific")
print(f"Merged: {merged}")  # Output: Merged: Junior_Developer
```

### Example 4: Hierarchy Management

```python
from pamola_core.anonymization.commons.text_processing_utils import (
    validate_hierarchy_value,
    create_safe_hierarchy_path
)

# Validate values for different hierarchy levels
is_valid, error = validate_hierarchy_value("Technology", level=0)
print(f"Level 0 valid: {is_valid}")  # Output: Level 0 valid: True

is_valid, error = validate_hierarchy_value(
    "Senior Software Engineer III", 
    level=0
)
print(f"Too specific: {is_valid}, {error}")  
# Output: Too specific: False, Value too specific for level 0

# Create safe hierarchy path
hierarchy = ["Technology", "Software Development", "Backend Engineering"]
path = create_safe_hierarchy_path(hierarchy)
print(f"Path: {path}")  
# Output: Path: Technology > Software_Development > Backend_Engineering
```

### Example 5: Token Extraction with Privacy Limits

```python
from pamola_core.anonymization.commons.text_processing_utils import extract_tokens

# Extract tokens with anonymization constraints
text = "Senior Software Engineer with 10+ years experience in Python, Java, " \
       "JavaScript, C++, Go, Rust, Scala, Kotlin, Swift, Objective-C, " \
       "Ruby, PHP, C#, TypeScript, Dart, R, MATLAB, Julia, Haskell..."

tokens = extract_tokens(text, min_length=3, max_tokens=10)
print(f"Tokens: {tokens}")
# Output: Tokens: ['senior', 'software', 'engineer', 'with', 'years', 
#                  'experience', 'python', 'java', 'javascript']
# Note: Limited to 10 tokens for privacy
```

### Example 6: Complete Anonymization Workflow

```python
from pamola_core.anonymization.commons.text_processing_utils import (
    normalize_text,
    prepare_value_for_matching,
    find_closest_category,
    clean_category_name
)

# Anonymization workflow for job titles
def anonymize_job_title(job_title: str, hierarchy: Dict[str, List[str]]) -> str:
    """Anonymize job title using hierarchical generalization."""
    
    # Step 1: Normalize
    normalized = normalize_text(job_title, level="basic")
    
    # Step 2: Prepare for matching
    prepared = prepare_value_for_matching(normalized)
    
    # Step 3: Find best category
    all_categories = []
    for level, categories in hierarchy.items():
        all_categories.extend(categories)
    
    matched = find_closest_category(
        prepared, 
        all_categories, 
        threshold=0.7,
        fallback="OTHER_ROLE"
    )
    
    # Step 4: Clean for safe storage
    safe_category = clean_category_name(matched)
    
    return safe_category

# Example hierarchy
job_hierarchy = {
    "level_0": ["Technical", "Business", "Support"],
    "level_1": ["Engineering", "Sales", "Customer Service"],
    "level_2": ["Software Engineering", "Account Management", "Technical Support"]
}

# Anonymize various job titles
titles = [
    "Senior Software Engineer",
    "Jr. Sales Representative", 
    "Customer Success Manager",
    "Chief Technology Officer"
]

for title in titles:
    anon = anonymize_job_title(title, job_hierarchy)
    print(f"{title} -> {anon}")

# Output:
# Senior Software Engineer -> Software_Engineering
# Jr. Sales Representative -> Sales
# Customer Success Manager -> Customer_Service
# Chief Technology Officer -> Technical
```

## Constants

### Anonymization-Specific Constants

```python
ANONYMIZATION_SAFE_CHARS = r'[^a-zA-Z0-9\s\-_]'  # More restrictive character set
DEFAULT_UNKNOWN_VALUE = "OTHER"                   # Default for unknown categories
DEFAULT_SUPPRESSED_VALUE = "SUPPRESSED"           # Value for suppressed data
CATEGORY_SEPARATOR = "_"                          # Standard separator
MAX_CATEGORY_LENGTH = 40                          # Shorter for anonymization
```

## Integration with Anonymization Operations

This module is designed to work seamlessly with PAMOLA.CORE anonymization operations:

1. **Categorical Generalization**: Provides category matching and hierarchy navigation
2. **Value Suppression**: Safe fallback values for unmatchable data
3. **Synthetic Data Generation**: Category name generation that preserves privacy
4. **Batch Processing**: Efficient handling of large categorical datasets

## Performance Considerations

- **Caching**: Frequently matched categories can be cached at the operation level
- **Length Limits**: Enforced limits prevent memory issues with malformed data
- **Component Limits**: Maximum components/tokens prevent combinatorial explosion
- **Efficient Matching**: Pre-processing improves matching accuracy and speed

## Best Practices

1. **Use Appropriate Normalization**:
   - Keep normalization minimal to preserve category distinctions
   - Use `prepare_value_for_matching()` to improve matching accuracy

2. **Set Reasonable Thresholds**:
   - Lower thresholds (0.7-0.8) for general categories
   - Higher thresholds (0.9+) for sensitive categories

3. **Handle Fallbacks Properly**:
   - Always provide meaningful fallback values
   - Consider using "OTHER" or domain-specific unknowns

4. **Validate Hierarchy Levels**:
   - Ensure values are appropriate for their hierarchy level
   - Use stricter constraints for higher (more general) levels

5. **Monitor Component Limits**:
   - Log warnings when limits are exceeded
   - Adjust limits based on privacy requirements

This module provides the text processing foundation for privacy-preserving categorical data anonymization in PAMOLA.CORE, ensuring consistent and safe handling of textual data throughout the anonymization pipeline.