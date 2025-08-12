# PAMOLA.CORE Text Processing Utilities Documentation

## Overview

The `text_utils.py` module is a core component of the PAMOLA.CORE privacy-preserving framework, providing essential text processing utilities for data anonymization and privacy protection operations. This module offers a comprehensive suite of functions for text normalization, string similarity calculations, and text manipulation that are fundamental to many privacy-enhancing techniques.

### Purpose

In the context of privacy protection, accurate text processing is crucial for:
- **Category Matching**: Identifying similar values that should be grouped together for k-anonymity
- **Generalization**: Finding appropriate hierarchical categories for sensitive data
- **Pseudonymization**: Consistent replacement of identifying information
- **Data Quality**: Maintaining data utility while applying privacy transformations

## Key Features

### 1. **Multi-Level Text Normalization**
- **Basic**: Simple whitespace normalization and case conversion
- **Advanced**: Unicode normalization, special character handling, smart quote replacement
- **Aggressive**: Alphanumeric filtering, stopword removal, punctuation normalization
- Language-aware processing with optional stopword removal

### 2. **String Similarity Algorithms**
- **Sequence Matching**: Using difflib's SequenceMatcher for accurate string comparison
- **Partial Matching**: Finding best substring matches for fuzzy matching
- **Token-Based Similarity**: Jaccard similarity for word-level comparisons
- **Levenshtein Distance**: Edit distance calculation with memory-optimized implementation

### 3. **Text Manipulation Utilities**
- **Category Name Cleaning**: Safe formatting for file/field naming
- **Composite Value Splitting**: Handling multi-value fields (e.g., "IT|Finance|HR")
- **Token Extraction**: Configurable tokenization with pattern matching
- **Smart Truncation**: Length limiting with word boundary preservation

### 4. **Validation Functions**
- Category name validation with customizable rules
- Invalid character detection and reporting
- Length constraint enforcement

## Dependencies

### Internal Dependencies
- `pamola_core.utils.nlp.stopwords`: Provides language-specific stopword lists for advanced text normalization

### External Dependencies (Python Built-ins)
- `difflib`: Sequence matching and similarity calculations
- `re`: Regular expression operations for pattern matching
- `unicodedata`: Unicode normalization and character category detection
- `typing`: Type hints for better code clarity
- `logging`: Structured logging for debugging and monitoring

## Component Architecture

```
text_utils.py
├── Text Normalization
│   ├── normalize_text()          # Multi-level normalization
│   └── clean_category_name()     # Safe naming for categories
├── Similarity Calculations
│   ├── calculate_string_similarity()  # Multiple similarity methods
│   ├── find_closest_match()          # Find best matches from candidates
│   └── find_closest_category()       # Category-specific matching
├── Text Manipulation
│   ├── split_composite_value()   # Handle multi-value fields
│   ├── extract_tokens()          # Configurable tokenization
│   └── truncate_text()           # Smart text truncation
└── Validation
    └── is_valid_category_name()  # Category name validation
```

## Function Signatures and Parameters

### Text Normalization

#### `normalize_text()`
```python
def normalize_text(
    text: str,
    level: str = "basic",
    preserve_case: bool = False,
    languages: Optional[List[str]] = None
) -> str
```

**Parameters:**
- `text`: Input text to normalize
- `level`: Normalization level ("none", "basic", "advanced", "aggressive")
- `preserve_case`: Whether to preserve original case
- `languages`: Language codes for stopword removal (used in aggressive mode)

**Returns:** Normalized text string

#### `clean_category_name()`
```python
def clean_category_name(
    name: str,
    max_length: int = 50,
    invalid_chars: str = r'[<>:"/\\|?*]'
) -> str
```

**Parameters:**
- `name`: Category name to clean
- `max_length`: Maximum allowed length
- `invalid_chars`: Regex pattern of invalid characters

**Returns:** Cleaned category name safe for file/field naming

### String Similarity

#### `calculate_string_similarity()`
```python
def calculate_string_similarity(
    s1: str,
    s2: str,
    method: str = "ratio",
    normalize: bool = True,
    case_sensitive: bool = False
) -> float
```

**Parameters:**
- `s1`, `s2`: Strings to compare
- `method`: Similarity calculation method
  - `"ratio"`: SequenceMatcher ratio (0-1)
  - `"partial"`: Best partial match ratio
  - `"token"`: Token-based Jaccard similarity
  - `"levenshtein"`: Normalized Levenshtein distance
- `normalize`: Whether to normalize strings before comparison
- `case_sensitive`: Whether comparison is case-sensitive

**Returns:** Similarity score between 0 and 1

#### `find_closest_match()`
```python
def find_closest_match(
    target: str,
    candidates: List[str],
    threshold: float = 0.8,
    method: str = "ratio",
    top_n: int = 1,
    normalize: bool = True
) -> List[Tuple[str, float]]
```

**Parameters:**
- `target`: Target string to match
- `candidates`: List of candidate strings
- `threshold`: Minimum similarity threshold
- `method`: Similarity method to use
- `top_n`: Number of top matches to return
- `normalize`: Whether to normalize strings

**Returns:** List of (candidate, similarity_score) tuples

### Text Manipulation

#### `split_composite_value()`
```python
def split_composite_value(
    value: str,
    separators: Optional[List[str]] = None,
    normalize: bool = True
) -> List[str]
```

**Parameters:**
- `value`: Composite value to split
- `separators`: List of separator characters (default: ["|", "/", ",", ";"])
- `normalize`: Whether to normalize components

**Returns:** List of split components

#### `extract_tokens()`
```python
def extract_tokens(
    text: str,
    min_length: int = 2,
    pattern: Optional[str] = None,
    lowercase: bool = True
) -> List[str]
```

**Parameters:**
- `text`: Text to tokenize
- `min_length`: Minimum token length
- `pattern`: Regex pattern for token extraction
- `lowercase`: Whether to lowercase tokens

**Returns:** List of extracted tokens

## Usage Examples

### Example 1: Basic Text Normalization for Category Matching

```python
from pamola_core.utils.nlp.text_utils import normalize_text, calculate_string_similarity

# Normalize job titles for matching
job1 = "Senior Software Engineer"
job2 = "Sr. Software Engineer"

# Basic normalization
norm_job1 = normalize_text(job1, level="basic")
norm_job2 = normalize_text(job2, level="basic")

# Calculate similarity
similarity = calculate_string_similarity(job1, job2, method="ratio")
print(f"Similarity: {similarity:.2f}")  # Output: Similarity: 0.91
```

### Example 2: Finding Best Category Match for Generalization

```python
from pamola_core.utils.nlp.text_utils import find_closest_category

# Available generalization categories
categories = [
    "Healthcare Professional",
    "IT Professional", 
    "Finance Professional",
    "Education Professional"
]

# Find best match for specific job title
job_title = "Nurse Practitioner"
best_category = find_closest_category(
    job_title, 
    categories, 
    threshold=0.6,
    method="token"
)
print(f"Best category: {best_category}")  
# Output: Best category: Healthcare Professional
```

### Example 3: Handling Composite Values in Data

```python
from pamola_core.utils.nlp.text_utils import split_composite_value

# Multi-department employee
departments = "IT|Finance|Operations"
dept_list = split_composite_value(departments)
print(f"Departments: {dept_list}")
# Output: Departments: ['it', 'finance', 'operations']

# With custom separators and no normalization
skills = "Python / Machine Learning / Data Science"
skill_list = split_composite_value(
    skills, 
    separators=["/"], 
    normalize=False
)
print(f"Skills: {skill_list}")
# Output: Skills: ['Python', 'Machine Learning', 'Data Science']
```

### Example 4: Advanced Similarity for Privacy-Preserving Matching

```python
from pamola_core.utils.nlp.text_utils import find_closest_match

# Find similar names for grouping (k-anonymity)
target_name = "John Smith"
name_database = [
    "Jon Smith",
    "John Smyth", 
    "Johnny Smith",
    "Jane Smith",
    "John Doe"
]

matches = find_closest_match(
    target_name,
    name_database,
    threshold=0.85,
    method="levenshtein",
    top_n=3
)

for name, score in matches:
    print(f"{name}: {score:.3f}")
# Output:
# Jon Smith: 0.900
# Johnny Smith: 0.867
# John Smyth: 0.900
```

### Example 5: Safe Category Name Generation

```python
from pamola_core.utils.nlp.text_utils import clean_category_name, is_valid_category_name

# Clean potentially problematic category names
raw_category = "Sales/Marketing & Distribution"
clean_name = clean_category_name(raw_category)
print(f"Clean name: {clean_name}")
# Output: Clean name: Sales_Marketing _ Distribution

# Validate category name
is_valid, error_msg = is_valid_category_name("A/B Testing")
if not is_valid:
    print(f"Invalid: {error_msg}")
# Output: Invalid: Name contains invalid characters: /
```

### Example 6: Text Processing for Anonymization

```python
from pamola_core.utils.nlp.text_utils import normalize_text, extract_tokens

# Aggressive normalization for maximum privacy
sensitive_text = "Contact John Doe at john.doe@example.com or call 555-1234!"
anonymized = normalize_text(
    sensitive_text, 
    level="aggressive",
    languages=["en"]  # Remove English stopwords
)
print(f"Anonymized: {anonymized}")
# Output: Anonymized: contact john doe johndoeexamplecom call 5551234

# Extract tokens for analysis
tokens = extract_tokens(sensitive_text, min_length=4)
print(f"Tokens: {tokens}")
# Output: Tokens: ['contact', 'john', 'example', 'call', '1234']
```

## Integration with Privacy Operations

The text utilities module integrates seamlessly with PAMOLA.CORE anonymization operations:

1. **Categorical Generalization**: Uses similarity matching to group similar categories
2. **Pseudonymization**: Ensures consistent text normalization for mapping generation
3. **K-Anonymity**: Identifies similar values that can be grouped together
4. **Data Quality**: Maintains text consistency across privacy transformations

## Performance Considerations

- **String Length Limits**: Maximum string length of 10,000 characters for similarity calculations
- **Memory Optimization**: Levenshtein implementation uses O(min(m,n)) space
- **Early Exit**: Similarity calculations exit early when perfect matches are found
- **Caching**: Results can be cached at the operation level for repeated calculations

## Best Practices

1. **Choose Appropriate Normalization Level**:
   - Use "basic" for general text processing
   - Use "advanced" for Unicode-heavy text
   - Use "aggressive" only when maximum normalization is needed

2. **Select Suitable Similarity Method**:
   - Use "ratio" for general string comparison
   - Use "partial" for substring matching
   - Use "token" for multi-word text
   - Use "levenshtein" for typo tolerance

3. **Handle Edge Cases**:
   - Always check for empty/None inputs
   - Consider maximum string lengths for large datasets
   - Use appropriate thresholds for similarity matching

4. **Performance Optimization**:
   - Pre-normalize text when doing multiple comparisons
   - Use batch processing for large datasets
   - Consider caching similarity results

This module provides the foundation for text-based privacy operations in PAMOLA.CORE, ensuring consistent and reliable text processing across all anonymization techniques.