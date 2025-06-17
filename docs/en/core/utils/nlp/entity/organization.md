# PAMOLA Core: Organization Entity Extractor Module

**Module Path:** `pamola_core.utils.nlp.entity.organization`

---

## Overview

The Organization Entity Extractor module provides robust functionality for extracting organization names (such as companies, educational institutions, government agencies, and nonprofits) from text. It is a core component of the PAMOLA Core framework's NLP utilities, enabling downstream tasks to identify and classify organization entities in multilingual and domain-specific contexts.

This module is designed for integration with the broader PAMOLA pipeline, supporting both rule-based and model-based extraction, and is extensible for custom organization types and extraction strategies.

---

## Key Features

- **Named Entity Recognition (NER) Integration:** Utilizes spaCy and custom NER models for organization extraction.
- **Organization Type Filtering:** Supports filtering by organization type (e.g., company, university, government, nonprofit).
- **Abbreviation Handling:** Optionally includes abbreviations in extraction results.
- **Multilingual Support:** Handles organization extraction in multiple languages.
- **Extensible Terms Dictionary:** Customizable terms for different organization types.
- **Graceful Fallbacks:** Falls back to spaCy models if specialized extractors are unavailable.
- **Logging and Error Handling:** Comprehensive logging and robust exception management.

---

## Dependencies

### Standard Library
- `logging`
- `typing` (Dict, List, Any, Optional, Set, Union, Tuple)

### Internal Modules
- `pamola_core.utils.nlp.entity.base`  
  - `BaseEntityExtractor`, `EntityMatchResult`
- `pamola_core.utils.nlp.model_manager`  
  - `NLPModelManager`

---

## Exception Classes

> **Note:** This module does not define custom exception classes. All exceptions are handled internally and logged. Standard exceptions (e.g., `Exception`) may be raised in rare cases of critical failure.

### Example: Handling Extraction Errors
```python
try:
    result = extractor._extract_with_ner(text, normalized_text, language)
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    # Handle or propagate as needed
```
**When Raised:**
- If the NER model is unavailable or fails to process the input.
- If the model manager encounters an unexpected error.

---

## Main Classes

### OrganizationExtractor

#### Constructor
```python
def __init__(
    self,
    organization_type: str = 'any',
    include_abbreviations: bool = True,
    **kwargs
)
```
**Parameters:**
- `organization_type`: Type of organizations to extract (`'company'`, `'university'`, `'government'`, `'nonprofit'`, or `'any'`).
- `include_abbreviations`: Whether to include abbreviations in extraction.
- `**kwargs`: Additional parameters for the base extractor.

#### Key Attributes
- `organization_type`: Current filter for organization type.
- `include_abbreviations`: Whether abbreviations are included.
- `organization_terms`: Dictionary of terms for each organization type.

#### Public Methods

##### _get_entity_type
```python
def _get_entity_type(self) -> str
```
- **Returns:** Entity type string (`"organization"`).

##### _extract_with_ner
```python
def _extract_with_ner(
    self,
    text: str,
    normalized_text: str,
    language: str
) -> Optional[EntityMatchResult]
```
- **Parameters:**
    - `text`: Original text.
    - `normalized_text`: Normalized text.
    - `language`: Language of the text.
- **Returns:** `EntityMatchResult` if a match is found, else `None`.
- **Raises:** Logs and returns `None` on error.

##### _matches_organization_type
```python
def _matches_organization_type(
    self,
    text: str,
    org_type: str
) -> bool
```
- **Parameters:**
    - `text`: Organization name.
    - `org_type`: Organization type to check.
- **Returns:** `True` if the organization matches the type, else `False`.

---

## Dependency Resolution and Completion Validation

- **Model Selection:** Attempts to use a specialized NER extractor from the model manager. If unavailable, falls back to a spaCy model for the specified language.
- **Type Filtering:** After extraction, organization names are filtered by the specified type using the `organization_terms` dictionary.
- **Validation:** If no valid organization is found, returns `None`.
- **Logging:** All errors and warnings are logged for traceability.

---

## Usage Examples

### Basic Extraction
```python
from pamola_core.utils.nlp.entity.organization import OrganizationExtractor

# Create an extractor for companies
extractor = OrganizationExtractor(organization_type='company')

text = "OpenAI Inc. and Google LLC are leading AI companies."
normalized_text = text.lower()
language = 'en'

# Extract organization
result = extractor._extract_with_ner(text, normalized_text, language)
if result:
    print(result.category, result.alias)
else:
    print("No organization found.")
```

### Handling Extraction Errors
```python
try:
    result = extractor._extract_with_ner(text, normalized_text, language)
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    # Handle error gracefully
```

### Filtering by Organization Type
```python
# Only extract universities
extractor = OrganizationExtractor(organization_type='university')
```

### Integration with Pipeline (e.g., BaseTask)
```python
# Example integration with a task pipeline
class MyTask(BaseTask):
    def run(self, text):
        extractor = OrganizationExtractor()
        result = extractor._extract_with_ner(text, text.lower(), 'en')
        # Use result in downstream processing
```

---

## Error Handling and Exception Hierarchy

- **Standard Exceptions:** All errors are caught and logged. No custom exceptions are defined in this module.
- **Logging:** Errors are logged with context for debugging and traceability.
- **Best Practice:** Always check for `None` before using extraction results.

---

## Configuration Requirements

- **Model Manager:** Ensure that `NLPModelManager` is properly configured with the required NER models for your target languages.
- **Organization Terms:** Extend `organization_terms` as needed for domain-specific organization types.

---

## Security Considerations and Best Practices

- **Model Trust:** Only use trusted and validated NER models to avoid data leakage or misclassification.
- **Input Sanitization:** Always sanitize input text to prevent injection attacks or model misuse.
- **Logging:** Avoid logging sensitive data in production environments.

### Example: Security Failure and Handling
```python
# Risk: Using an untrusted spaCy model
model = nlp_model_manager.get_model('spacy', 'en')
if not model:
    logger.warning("No trusted model available. Extraction aborted.")
    return None
```
**Risk:** Using untrusted models may lead to data leakage or incorrect extraction.
**Mitigation:** Always validate and restrict model sources.

---

## Internal vs. External Dependencies

- **Internal:** Organization extraction is typically used as an internal step in NLP pipelines.
- **External:** If using external models or data, ensure they are properly validated and secured.

---

## Best Practices

1. **Specify Organization Type:** Use the `organization_type` parameter to improve extraction precision.
2. **Extend Terms Dictionary:** Add domain-specific terms to `organization_terms` for better coverage.
3. **Handle None Results:** Always check for `None` before using extraction results.
4. **Integrate with Logging:** Use logging to monitor extraction quality and errors.
5. **Validate Models:** Only use trusted and validated NER models.
