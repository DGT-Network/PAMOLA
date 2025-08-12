# PAMOLA Core: Job Position Entity Extractor (`job.py`) Module

## Overview

The `job.py` module provides robust functionality for extracting job position entities from text within the PAMOLA Core framework. It leverages dictionary-based matching, Named Entity Recognition (NER) models, and specialized rules to identify job titles, seniority levels, and domains. This module is designed to support multilingual extraction (including English and Russian) and is intended for use in data processing pipelines where accurate job position identification is critical.

## Key Features

- **Dictionary and NER-based Extraction**: Combines rule-based and model-based approaches for high accuracy.
- **Seniority and Domain Detection**: Identifies both the seniority (e.g., junior, senior) and the professional domain (e.g., software development, data science) of job titles.
- **Multilingual Support**: Handles both English and Russian job titles and domains.
- **Configurable Extraction**: Allows enabling/disabling seniority detection and skill inclusion.
- **Integration with PAMOLA Core**: Designed to work seamlessly with the broader PAMOLA pipeline and entity extraction framework.

## Dependencies

### Standard Library
- `logging`
- `typing.Optional`

### Internal Modules
- `pamola_core.utils.nlp.entity.base`  
  - `BaseEntityExtractor`, `EntityMatchResult`
- `pamola_core.utils.nlp.model_manager`  
  - `NLPModelManager`

## Exception Classes

> **Note:** This module does not define custom exception classes. All errors are handled using standard Python exceptions and logging. If you wish to extend error handling, subclass `Exception` and follow the pattern below:

```python
class JobExtractionError(Exception):
    """Raised when job extraction fails."""
    pass

try:
    # Extraction logic
    ...
except JobExtractionError as e:
    print(f"Extraction failed: {e}")
```

- **When Raised**: Custom exceptions would be raised for unrecoverable extraction errors, such as model loading failures or invalid input formats.

## Main Classes

### `JobPositionExtractor`

#### Constructor
```python
JobPositionExtractor(
    seniority_detection: bool = True,
    include_skills: bool = True,
    **kwargs
)
```
**Parameters:**
- `seniority_detection`: Enable/disable seniority level detection (default: `True`).
- `include_skills`: Include skill extraction (default: `True`).
- `**kwargs`: Additional parameters for `BaseEntityExtractor`.

#### Key Attributes
- `seniority_detection`: Whether to detect seniority levels.
- `include_skills`: Whether to include skills in extraction.
- `domains`: Dictionary mapping domain names to job-related terms.
- `seniority_terms`: Dictionary mapping seniority levels to keywords.

#### Public Methods

##### `def _get_entity_type(self) -> str`
Returns the entity type string for this extractor.
- **Returns:** `str` — Always returns `'job'`.

##### `def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]`
Extracts job positions using NER models.
- **Parameters:**
    - `text`: Original text.
    - `normalized_text`: Normalized version of the text.
    - `language`: Language code (e.g., 'en', 'ru').
- **Returns:** `EntityMatchResult` or `None` — Extraction result if found.
- **Raises:** Logs errors internally; does not raise exceptions outward.

##### `def _determine_domain(self, text: str) -> str`
Determines the professional domain for a job position.
- **Parameters:**
    - `text`: Job position text.
- **Returns:** `str` — Domain name (e.g., 'software_development').

##### `def _determine_seniority(self, text: str) -> str`
Determines the seniority level from a job position.
- **Parameters:**
    - `text`: Job position text.
- **Returns:** `str` — Seniority level (e.g., 'junior', 'senior', 'Any').

## Dependency Resolution and Completion Validation

- **Dependency Resolution**: The extractor relies on the `NLPModelManager` to provide the appropriate NER model for the specified language and entity type. If the model is unavailable, extraction falls back to dictionary-based or rule-based methods.
- **Completion Validation**: Extraction is considered complete if a job position is found and classified with a domain and (optionally) seniority. If no match is found, the method returns `None`.

## Usage Examples

### Basic Extraction
```python
from pamola_core.utils.nlp.entity.job import JobPositionExtractor

# Create the extractor
extractor = JobPositionExtractor(seniority_detection=True)

# Example text
text = "Senior Data Scientist at Acme Corp"
normalized_text = text.lower()
language = 'en'

# Extract job entity
result = extractor._extract_with_ner(text, normalized_text, language)
if result:
    print(f"Job: {result.category}, Domain: {result.domain}, Seniority: {result.seniority}")
else:
    print("No job entity found.")
```

### Handling Extraction Failures
```python
try:
    result = extractor._extract_with_ner(text, normalized_text, language)
    if not result:
        raise ValueError("No job entity found.")
except Exception as e:
    # Log and handle extraction errors
    print(f"Extraction error: {e}")
```

### Integration with a Pipeline Task
```python
# Example integration with a BaseTask
class JobExtractionTask(BaseTask):
    def run(self, input_text):
        extractor = JobPositionExtractor()
        result = extractor._extract_with_ner(input_text, input_text.lower(), 'en')
        if result:
            # Process result
            ...
        else:
            # Handle missing entity
            ...
```

### Continue-on-Error Mode with Logging
```python
import logging
logger = logging.getLogger(__name__)

try:
    result = extractor._extract_with_ner(text, normalized_text, language)
except Exception as e:
    logger.warning(f"Extraction failed, continuing: {e}")
    result = None
```

## Configuration Requirements

- The `NLPModelManager` must be properly configured and able to provide an entity extractor for the 'job' entity type and the desired language.
- The extractor can be customized via `seniority_detection` and `include_skills` parameters.

## Security Considerations and Best Practices

- **Path Security**: This module does not handle file paths directly, but if integrating with external data sources, always validate and sanitize paths.
- **Risks of Disabling Path Security**: Disabling path security (e.g., allowing arbitrary file access) can expose the system to path traversal and data leakage attacks. Always restrict file access to trusted directories.

#### Example of a Security Failure
```python
# BAD: Loading a model from an untrusted path
model_path = user_input_path  # User-controlled
model = load_model(model_path)  # Potential path traversal!

# GOOD: Restrict to a safe directory
import os
SAFE_DIR = '/models/'
model_path = os.path.join(SAFE_DIR, os.path.basename(user_input_path))
model = load_model(model_path)
```

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical task IDs and internal references for data produced within the pipeline.
- **External Dependencies**: Use absolute paths only for data that is external to the pipeline and cannot be referenced by task ID.

## Best Practices

1. **Use Logical Task IDs**: Reference internal data by task ID to maintain clear data lineage.
2. **Limit Absolute Path Usage**: Only use absolute paths for external, non-pipeline data.
3. **Enable Seniority Detection When Needed**: For detailed analytics, enable `seniority_detection`.
4. **Handle Extraction Failures Gracefully**: Always check for `None` results and handle errors with logging.
5. **Validate Model Availability**: Ensure the `NLPModelManager` is configured for all required languages and entity types.
6. **Log Warnings for Missing Models**: The extractor logs a warning if a model is missing; monitor logs for such events.
