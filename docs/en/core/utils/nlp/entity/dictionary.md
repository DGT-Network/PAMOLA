# PAMOLA Core: Generic Dictionary-Based Entity Extractor Module

## Overview

The `dictionary.py` module provides a universal, dictionary-based entity extractor for the PAMOLA Core framework. It enables flexible and extensible entity extraction using custom or pre-defined dictionaries, supporting a wide range of entity types beyond specialized extractors. This module is designed for seamless integration into NLP pipelines, offering fallback mechanisms and robust error handling.

## Key Features

- **Generic Entity Extraction**: Works with any entity dictionary, supporting custom entity types.
- **NER Fallback**: Optionally falls back to Named Entity Recognition (NER) if dictionary matching fails.
- **Automatic Dictionary Resolution**: Locates appropriate dictionary files based on entity type and language.
- **Logging and Error Handling**: Provides detailed logging and robust exception management.
- **Integration Ready**: Designed for use with PAMOLA Core's `BaseTask` and other pipeline components.

## Dependencies

### Standard Library
- `logging`
- `typing.Optional`

### Internal Modules
- `pamola_core.utils.nlp.entity.base`  
  - `BaseEntityExtractor`, `EntityMatchResult`, `find_dictionary_file`
- `pamola_core.utils.nlp.model_manager`  
  - `NLPModelManager`

## Exception Classes

> **Note:** This module does not define custom exception classes directly. All exceptions are handled via logging and standard Python exceptions. If you extend this module, consider defining custom exceptions for advanced error handling.

### Example: Handling Extraction Errors
```python
try:
    extractor = GenericDictionaryExtractor(entity_type='product', language='en')
    result = extractor.extract(text)
except FileNotFoundError as e:
    # Handle missing dictionary file
    logger.error(f"Dictionary not found: {e}")
except Exception as e:
    # Handle other errors
    logger.error(f"Extraction failed: {e}")
```

## Main Classes

### `GenericDictionaryExtractor`

#### Constructor
```python
GenericDictionaryExtractor(
    entity_type: str = 'generic',
    fallback_to_ner: bool = True,
    **kwargs
)
```
**Parameters:**
- `entity_type`: Type of entities to extract (used for dictionary lookup)
- `fallback_to_ner`: Whether to fall back to NER if dictionary matching fails
- `**kwargs`: Additional parameters for the base extractor

#### Key Attributes
- `entity_type`: The type of entity to extract
- `fallback_to_ner`: Whether to use NER as a fallback
- `dictionary_path`: Path to the dictionary file (auto-resolved if not provided)

#### Public Methods

##### `def _get_entity_type(self) -> str`
Returns the entity type string for this extractor.

##### `def _find_dictionary(self) -> Optional[str]`
Finds the dictionary file for the current entity type and language.
- **Returns:** Path to the dictionary file if found, `None` otherwise.
- **Raises:** Logs a warning if not found.

##### `def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]`
Attempts to extract entities using NER as a fallback.
- **Parameters:**
    - `text`: Original text
    - `normalized_text`: Normalized text
    - `language`: Language of the text
- **Returns:** `EntityMatchResult` if a relevant entity is found, `None` otherwise.
- **Raises:** Logs errors on failure.

## Dependency Resolution and Completion Validation

- **Dictionary Resolution**: The extractor attempts to locate a dictionary file based on the `entity_type` and `language`. If not found, it logs a warning and may fall back to NER if enabled.
- **NER Fallback**: If dictionary extraction fails and `fallback_to_ner` is `True`, the extractor uses a generic NER model (e.g., spaCy) to identify entities.

## Usage Examples

### Basic Extraction
```python
# Create a generic dictionary extractor for 'product' entities
extractor = GenericDictionaryExtractor(entity_type='product', language='en')

# Extract entities from text
result = extractor.extract("Apple released a new iPhone.")
if result:
    print(f"Entity: {result.category}, Text: {result.original_text}")
else:
    print("No entity found.")
```

### Handling Missing Dictionaries
```python
try:
    extractor = GenericDictionaryExtractor(entity_type='unknown_type', language='en')
    result = extractor.extract("Some text.")
except FileNotFoundError:
    # Handle missing dictionary gracefully
    print("Dictionary file not found for the specified entity type.")
```

### Using NER Fallback
```python
# Enable NER fallback (default)
extractor = GenericDictionaryExtractor(entity_type='person', fallback_to_ner=True, language='en')
result = extractor.extract("Barack Obama visited Berlin.")
if result:
    print(f"Extracted entity: {result.category}")
```

### Integration with BaseTask
```python
from pamola_core.base_processor import BaseTask

class MyEntityTask(BaseTask):
    def run(self, text):
        extractor = GenericDictionaryExtractor(entity_type='org', language='en')
        return extractor.extract(text)
```

### Continue-on-Error Example
```python
try:
    extractor = GenericDictionaryExtractor(entity_type='event', language='en')
    result = extractor.extract("The Olympics will be held in Paris.")
except Exception as e:
    # Continue processing other tasks
    logger.warning(f"Extraction failed, continuing: {e}")
```

## Error Handling and Exception Hierarchy

- **FileNotFoundError**: Raised if the dictionary file cannot be found (handled internally with logging).
- **Generic Exception**: Any other errors during extraction are logged and do not halt the pipeline if handled appropriately.

## Configuration Requirements

- **entity_type**: Must correspond to a valid dictionary or NER-supported type.
- **language**: Should match the language of the input text and available models/dictionaries.
- **dictionary_path**: Optional; if not provided, auto-resolved.
- **fallback_to_ner**: Boolean; enables NER fallback if dictionary extraction fails.

## Security Considerations and Best Practices

- **Path Security**: Ensure that dictionary paths are validated and not user-controlled to prevent path traversal or unauthorized file access.
- **Model Security**: Only use trusted NER models to avoid malicious code execution.

### Security Failure Example
```python
# BAD: Allowing user input for dictionary_path without validation
user_path = input("Enter dictionary path: ")
extractor = GenericDictionaryExtractor(dictionary_path=user_path)
# This can lead to path traversal or loading malicious files!
```

#### How It Is Handled
- Always validate or restrict dictionary paths to known directories.
- Never allow arbitrary user input for file paths.

### Risks of Disabling Path Security
- Disabling path checks can expose the system to unauthorized file access, data leaks, or code execution vulnerabilities.

## Internal vs. External Dependencies

- **Internal (Task ID-based)**: Use entity types and language to resolve dictionaries within the project structure.
- **External (Absolute Path)**: Only specify absolute paths for dictionaries when using trusted, external resources.

## Best Practices

1. **Use Entity Types for Internal Dictionaries**: Maintain logical connections and consistency by using entity types and language for dictionary resolution.
2. **Use Absolute Paths Judiciously**: Only use absolute paths for trusted, external dictionaries.
3. **Enable NER Fallback for Robustness**: Allow NER fallback to handle cases where dictionaries are incomplete or missing.
4. **Validate All Paths**: Never accept unvalidated user input for file paths.
5. **Integrate with BaseTask**: Use the extractor within PAMOLA Core tasks for seamless pipeline integration.
6. **Log All Errors**: Ensure all exceptions are logged for traceability and debugging.
