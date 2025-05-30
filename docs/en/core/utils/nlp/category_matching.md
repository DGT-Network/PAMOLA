# PAMOLA Core: Category Matching Module

## Overview

The `pamola_core.utils.nlp.category_matching` module provides advanced category matching capabilities for the PAMOLA Core framework. It enables mapping of free-form text to predefined categories using hierarchical dictionaries, supports conflict resolution strategies, and is optimized for efficient, large-scale NLP pipelines.

---

## Key Features

- **Hierarchical Category Dictionaries**: Supports both flat and hierarchical category structures.
- **Flexible Conflict Resolution**: Multiple strategies for resolving ambiguous matches (e.g., specificity, domain, alias, user override).
- **File-Based and In-Memory Caching**: Efficient dictionary loading and reuse with cache integration.
- **Batch Processing**: Parallelized matching for large datasets.
- **Fallback Handling**: Robust fallback category logic for low-confidence matches.
- **Detailed Hierarchy Analysis**: Extracts statistics and structure from category hierarchies.

---

## Dependencies

### Standard Library
- `json`
- `logging`
- `pathlib.Path`
- `typing`

### Internal Modules
- `pamola_core.utils.nlp.base.ConfigurationError`
- `pamola_core.utils.nlp.base.batch_process`
- `pamola_core.utils.nlp.cache.get_cache`, `cache_function`

---

## Exception Classes

### - **ConfigurationError**
Base exception for configuration-related errors, imported from `pamola_core.utils.nlp.base`.

**Example Handling:**
```python
from pamola_core.utils.nlp.base import ConfigurationError
try:
    # Attempt to load a malformed dictionary file
    cat_dict = CategoryDictionary.from_file('bad_dict.json')
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
```
**When Raised:**
- Raised when dictionary or hierarchy data is missing required structure or is otherwise invalid.

---

## Main Classes

### CategoryDictionary

**Purpose:** Manages category dictionaries, supports hierarchical structures, and provides matching and analysis methods.

#### Constructor
```python
CategoryDictionary(
    dictionary_data: Dict[str, Any] = None,
    hierarchy_data: Dict[str, Any] = None
)
```
**Parameters:**
- `dictionary_data`: Category dictionary data (optional).
- `hierarchy_data`: Category hierarchy data (optional).

#### Key Attributes
- `dictionary`: The main category dictionary (flattened for matching).
- `hierarchy`: The full hierarchy structure (if provided).

#### Public Methods
- `from_file(file_path: Union[str, Path]) -> CategoryDictionary`
    - Load a category dictionary from a JSON file (cached).
    - **Parameters:**
        - `file_path`: Path to the dictionary JSON file.
    - **Returns:** Loaded `CategoryDictionary` instance.
    - **Raises:** `ConfigurationError` (re-raised), generic exceptions logged and return empty dictionary.

- `get_best_match(
    self,
    text: str,
    strategy: str = "specific_first"
) -> Tuple[Optional[str], float, List[str]]`
    - Find the best matching category for a text.
    - **Parameters:**
        - `text`: Text to match.
        - `strategy`: Conflict resolution strategy (`specific_first`, `domain_prefer`, `alias_only`, `user_override`).
    - **Returns:** Tuple of (matched category, confidence score, conflict candidates).

- `get_category_info(
    self,
    category: str
) -> Dict[str, Any]`
    - Get information about a category.
    - **Parameters:**
        - `category`: Category name.
    - **Returns:** Category info dict or empty dict if not found.

- `analyze_hierarchy(self) -> Dict[str, Any]`
    - Analyze the category hierarchy for statistics and structure.
    - **Returns:** Dict with counts by level, domain, and child relationships.

- `get_fallback_category(
    self,
    confidence_threshold: float = 0.5
) -> Optional[str]`
    - Get a fallback category for low-confidence matches.
    - **Parameters:**
        - `confidence_threshold`: Minimum score for reliable matches.
    - **Returns:** Fallback category name or `None`.

#### Example Usage
```python
# Load a category dictionary from file
cat_dict = CategoryDictionary.from_file('categories.json')

# Match a text to a category
category, score, conflicts = cat_dict.get_best_match('Senior Data Scientist')

# Get info about a category
info = cat_dict.get_category_info(category)

# Analyze the hierarchy
analysis = cat_dict.analyze_hierarchy()
```

---

## Top-Level Functions

### get_best_match
```python
get_best_match(
    text: str,
    dictionary: Dict[str, Dict[str, Any]],
    match_strategy: str = "specific_first"
) -> Tuple[Optional[str], float, List[str]]
```
- Find the best matching category for a text using a provided dictionary.
- **Parameters:**
    - `text`: Text to categorize.
    - `dictionary`: Category definitions.
    - `match_strategy`: Conflict resolution strategy.
- **Returns:** Tuple of (matched category, confidence score, conflict candidates).

### analyze_hierarchy
```python
analyze_hierarchy(
    hierarchy: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]
```
- Analyze a category hierarchy for statistics and structure.
- **Parameters:**
    - `hierarchy`: Category hierarchy data.
- **Returns:** Analysis results dict.

### batch_match_categories
```python
batch_match_categories(
    texts: List[str],
    dictionary_path: Union[str, Path],
    match_strategy: str = "specific_first",
    processes: Optional[int] = None
) -> List[Tuple[Optional[str], float, List[str]]]
```
- Match multiple texts to categories in parallel.
- **Parameters:**
    - `texts`: List of texts to categorize.
    - `dictionary_path`: Path to category dictionary file.
    - `match_strategy`: Conflict resolution strategy.
    - `processes`: Number of parallel processes (optional).
- **Returns:** List of (category, score, conflicts) tuples for each text.

#### Example Usage
```python
# Batch match a list of job titles
results = batch_match_categories(
    ["Data Scientist", "Software Engineer"],
    dictionary_path="categories.json"
)
for category, score, conflicts in results:
    print(f"Category: {category}, Score: {score}, Conflicts: {conflicts}")
```

---

## Dependency Resolution and Validation Logic

- **Dictionary Loading**: Uses file-based cache for efficient, up-to-date dictionary loading. Automatically reloads if the file changes.
- **Conflict Resolution**: Multiple strategies (specificity, domain, alias, user override) for ambiguous matches.
- **Batch Processing**: Uses `batch_process` for parallelized matching, improving throughput for large datasets.

---

## Usage Examples

### Accessing Outputs and Validating Categories
```python
# Load and match a single text
cat_dict = CategoryDictionary.from_file('categories.json')
category, score, conflicts = cat_dict.get_best_match('Lead Data Engineer')

# Get fallback if confidence is low
if score < 0.5:
    category = cat_dict.get_fallback_category()
```

### Handling Failed Dependencies
```python
try:
    cat_dict = CategoryDictionary.from_file('missing.json')
except ConfigurationError as e:
    logger.error(f"Failed to load dictionary: {e}")
    # Fallback to default or empty dictionary
    cat_dict = CategoryDictionary()
```

### Using in a Pipeline (e.g., with BaseTask)
```python
# In a BaseTask subclass
class MyTask(BaseTask):
    def run(self):
        cat_dict = CategoryDictionary.from_file(self.config['dict_path'])
        for record in self.data:
            category, score, _ = cat_dict.get_best_match(record['title'])
            record['category'] = category
```

### Continue-on-Error with Logging
```python
for text in texts:
    try:
        category, score, _ = cat_dict.get_best_match(text)
    except Exception as e:
        logger.warning(f"Failed to match category for '{text}': {e}")
        continue  # Continue processing other texts
```

---

## Integration Notes

- Designed for use in NLP pipelines and with `BaseTask` classes.
- Use `CategoryDictionary.from_file()` for efficient, cached dictionary loading.
- Use batch functions for high-throughput scenarios.

---

## Error Handling and Exception Hierarchy

- **ConfigurationError**: Raised for invalid or missing dictionary/hierarchy data.
- Other exceptions (e.g., file I/O, JSON decode errors) are logged and result in empty or default dictionaries.
- Always check for `None` or empty results from matching methods.

---

## Configuration Requirements

- Dictionary files must be valid JSON and follow the expected structure (flat or hierarchical with `categories_hierarchy`).
- For best performance, store dictionaries in a location accessible to the pipeline and avoid frequent changes.

---

## Security Considerations and Best Practices

- **Path Security**: Only load dictionaries from trusted locations. Avoid user-supplied paths without validation.
- **Cache Poisoning**: Ensure cache keys are not user-controlled to prevent overwriting or leaking sensitive data.
- **Parallel Processing**: When using batch processing, ensure that dictionary files are not modified during execution.

### Example: Security Failure and Handling
```python
# BAD: Using user-supplied path directly
user_path = get_user_input()
cat_dict = CategoryDictionary.from_file(user_path)  # Risk: may load malicious or corrupted file

# GOOD: Validate and sanitize path
if is_valid_path(user_path):
    cat_dict = CategoryDictionary.from_file(user_path)
else:
    raise ValueError("Invalid dictionary path")
```
**Risks of Disabling Path Security:**
- Loading untrusted files may result in code execution, data leaks, or pipeline corruption.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical resource names or config-driven paths for dictionaries managed within the pipeline.
- **External (Absolute Path) Dependencies**: Only use for static, trusted resources. Always validate before loading.

---

## Best Practices

1. **Use Logical Keys for Internal Data**: Reference dictionaries by config or resource name, not user input.
2. **Validate All Paths**: Never load dictionaries from untrusted or user-supplied locations without validation.
3. **Monitor Cache and Batch Performance**: Use caching and batch processing for large-scale tasks.
4. **Handle Low-Confidence Matches**: Always check confidence scores and use fallback categories as needed.
5. **Log and Handle Errors Gracefully**: Catch and log exceptions, fallback to defaults where possible.
6. **Keep Dictionary Files Consistent**: Avoid modifying dictionary files during batch operations.
