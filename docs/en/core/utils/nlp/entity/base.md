# PAMOLA Core NLP Entity Extraction Base Module

## Overview

This module provides the foundational classes and utilities for entity extraction within the PAMOLA Core framework. It defines abstract base classes, result structures, and dictionary management logic for all entity extractors, supporting both dictionary-based and NER-based extraction. The module is designed for extensibility, allowing new entity types and extraction strategies to be implemented with minimal effort.

## Key Features

- Abstract base class for entity extractors (`BaseEntityExtractor`)
- Unified result structure for entity matches (`EntityMatchResult`)
- Flexible dictionary file resolution and loading
- Support for both dictionary and NER-based extraction
- Caching for efficient dictionary reuse
- Language detection and normalization
- Progress bar support for batch extraction
- Extensible for custom entity types and extraction logic

## Dependencies

### Standard Library
- `os`, `json`, `logging`, `abc`, `pathlib`, `typing`

### Third-Party
- `tqdm` (for progress bars)

### Internal Modules
- `pamola_core.utils.nlp.base` (normalize_language_code)
- `pamola_core.utils.nlp.cache` (get_cache)
- `pamola_core.utils.nlp.category_matching` (CategoryDictionary)
- `pamola_core.utils.nlp.language` (detect_language)
- `pamola_core.utils.nlp.model_manager` (NLPModelManager)

## Exception Classes

> **Note:** This module does not define custom exception classes. All errors are handled using standard Python exceptions (e.g., `Exception`).

### Example: Handling Dictionary Load Errors
```python
try:
    extractor.load_dictionary('invalid_path.json')
except Exception as e:
    # Handle file not found or JSON decode errors
    print(f"Failed to load dictionary: {e}")
```
**When Raised:**
- Errors are raised when dictionary files are missing, corrupted, or have invalid format.

## Main Classes

### EntityMatchResult

Represents a single entity match result.

#### Constructor
```python
EntityMatchResult(
    original_text: str,
    normalized_text: str,
    category: Optional[str] = None,
    alias: Optional[str] = None,
    domain: Optional[str] = None,
    level: Optional[int] = None,
    seniority: Optional[str] = None,
    confidence: float = 0.0,
    method: str = "unknown",
    language: str = "unknown",
    conflicts: Optional[List[str]] = None,
    record_id: Optional[str] = None
)
```
**Parameters:**
- `original_text`: The original matched text.
- `normalized_text`: Lowercased, cleaned version of the text.
- `category`: Matched category name.
- `alias`: Category alias (for replacement).
- `domain`: Domain of the category.
- `level`: Hierarchy level.
- `seniority`: Seniority level.
- `confidence`: Confidence score (0-1).
- `method`: Matching method (dictionary, ner, etc.).
- `language`: Detected language.
- `conflicts`: List of conflicting categories.
- `record_id`: Optional record identifier.

#### Key Attributes
- `original_text`, `normalized_text`, `category`, `alias`, `domain`, `level`, `seniority`, `confidence`, `method`, `language`, `conflicts`, `record_id`

#### Public Methods
- `to_dict(self) -> Dict[str, Any]`
    - Converts the match result to a dictionary.
    - **Returns:** Dictionary representation of the match result.

### BaseEntityExtractor (Abstract)

Abstract base class for all entity extractors.

#### Constructor
```python
BaseEntityExtractor(
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    min_confidence: float = 0.5,
    use_cache: bool = True,
    **kwargs
)
```
**Parameters:**
- `language`: Language code or "auto" for detection.
- `dictionary_path`: Path to the dictionary file.
- `match_strategy`: Strategy for resolving matches.
- `use_ner`: Whether to use NER models if dictionary match fails.
- `min_confidence`: Minimum confidence threshold.
- `use_cache`: Whether to use caching.
- `**kwargs`: Additional parameters for subclasses.

#### Key Attributes
- `language`, `dictionary_path`, `match_strategy`, `use_ner`, `min_confidence`, `use_cache`, `category_dictionary`, `hierarchy`, `extra_params`

#### Public Methods
- `load_dictionary(self, dictionary_path: str) -> bool`
    - Loads the entity dictionary from the specified path.
    - **Parameters:**
        - `dictionary_path`: Path to the dictionary file.
    - **Returns:** `True` if loaded successfully, `False` otherwise.
    - **Raises:** Standard exceptions on file or format errors.

- `ensure_dictionary_loaded(self, entity_type: str) -> bool`
    - Ensures a dictionary is loaded, searching for a suitable one if needed.
    - **Parameters:**
        - `entity_type`: Type of entity.
    - **Returns:** `True` if loaded, `False` otherwise.

- `extract_entities(self, texts: List[str], record_ids: Optional[List[str]] = None, show_progress: bool = False) -> Dict[str, Any]`
    - Extracts entities from a list of texts.
    - **Parameters:**
        - `texts`: List of text strings.
        - `record_ids`: Optional list of record IDs.
        - `show_progress`: Show progress bar.
    - **Returns:** Extraction results as a dictionary.

- `_process_text(self, text: str, record_id: Optional[str] = None, language: str = "en") -> Optional[EntityMatchResult]`
    - Processes a single text and extracts entities.
    - **Parameters:**
        - `text`: Text to process.
        - `record_id`: Optional record ID.
        - `language`: Language code.
    - **Returns:** `EntityMatchResult` or `None`.

- `_extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]` (abstract)
    - Extracts entities using NER models. Must be implemented by subclasses.
    - **Parameters:**
        - `text`: Original text.
        - `normalized_text`: Normalized text.
        - `language`: Language code.
    - **Returns:** `EntityMatchResult` or `None`.

- `_get_entity_type(self) -> str` (abstract)
    - Returns the entity type string. Must be implemented by subclasses.
    - **Returns:** Entity type string.

- `_empty_result(self) -> Dict[str, Any]`
    - Returns an empty result structure.

- `_compile_results(entity_matches: List[EntityMatchResult], unresolved: List[Dict[str, Any]], language: str) -> Dict[str, Any]` (static)
    - Compiles final results from matches and unresolved texts.

#### Dependency Resolution and Completion Validation
- Dictionary files are resolved in the following order:
    1. `PAMOLA_ENTITIES_DIR` environment variable
    2. `data_repository` in `prj_config.json` under `external_dictionaries/entities`
    3. Default path in package resources
- If a dictionary is not found, extraction falls back to NER (if enabled).
- Completion is validated by checking if a suitable dictionary is loaded and if all texts are processed.

## Usage Examples

### Basic Extraction
```python
from pamola_core.utils.nlp.entity.base import BaseEntityExtractor

# Subclass BaseEntityExtractor and implement abstract methods
class JobTitleExtractor(BaseEntityExtractor):
    def _extract_with_ner(self, text, normalized_text, language):
        # Custom NER logic here
        return None
    def _get_entity_type(self):
        return "job"

# Instantiate extractor
extractor = JobTitleExtractor(language="en")

# Extract entities from texts
results = extractor.extract_entities([
    "Senior Data Scientist",
    "Chief Technology Officer"
])

# Access results
for entity in results["entities"]:
    print(entity)
```

### Handling Dictionary Load Failures
```python
try:
    extractor.load_dictionary('missing_file.json')
except Exception as e:
    # Handle missing or invalid dictionary
    print(f"Error: {e}")
```

### Using Progress Bar
```python
results = extractor.extract_entities(texts, show_progress=True)
```

### Integration with BaseTask
```python
# Inherit from BaseTask and use BaseEntityExtractor for entity extraction
class MyTask(BaseTask):
    def run(self):
        extractor = JobTitleExtractor(language="auto")
        entities = extractor.extract_entities(self.input_texts)
        self.save_results(entities)
```

## Error Handling and Exception Hierarchy
- All errors are raised as standard Python exceptions (e.g., `FileNotFoundError`, `json.JSONDecodeError`).
- Errors are logged using the module logger.
- Extraction continues for other texts even if some fail (unresolved entries are reported).

## Configuration Requirements
- If using the `data_repository` config, ensure `prj_config.json` is present and contains the correct path.
- Set the `PAMOLA_ENTITIES_DIR` environment variable to override dictionary location.

## Security Considerations and Best Practices

### Security Failure Example
```python
import os
os.environ['PAMOLA_ENTITIES_DIR'] = 'C:/untrusted/path'  # Risk: loading untrusted dictionaries
extractor = JobTitleExtractor()
extractor.ensure_dictionary_loaded('job')  # May load malicious data
```
**How It Is Handled:**
- Always validate and restrict dictionary paths to trusted locations.
- Avoid setting `PAMOLA_ENTITIES_DIR` to untrusted directories.

### Risks of Disabling Path Security
- Loading dictionaries from arbitrary paths can introduce malicious data or code.
- Always review and validate external dictionary files before use.

## Internal vs. External Dependencies
- **Internal:** Use project-relative or config-based dictionary paths for consistency and security.
- **External (Absolute):** Only use absolute paths for trusted, external dictionaries not managed by the pipeline.

## Best Practices
1. **Use Configured Paths:** Prefer config or environment variable-based dictionary resolution for maintainability.
2. **Implement Abstract Methods:** Always implement `_extract_with_ner` and `_get_entity_type` in subclasses.
3. **Validate Inputs:** Ensure all input texts are preprocessed and sanitized.
4. **Handle Errors Gracefully:** Log errors and report unresolved entries without interrupting the pipeline.
5. **Cache Dictionaries:** Enable caching for large dictionaries to improve performance.
6. **Restrict External Paths:** Avoid using untrusted absolute paths for dictionary files.
7. **Document Custom Extractors:** Provide clear docstrings and usage examples for new entity extractors.
