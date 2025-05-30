# PAMOLA Core: Entity Extraction Utilities

## Overview

The `entity_extraction` module provides a unified, high-level API for extracting structured entities from unstructured text within the PAMOLA Core framework. It supports a variety of entity types, including job positions, organizations, skills, transaction purposes, and custom entities, making it a central component for NLP-driven data enrichment and profiling tasks.

This module is designed for extensibility and ease of integration, offering both ready-to-use extractors and the ability to create custom entity extractors for specialized use cases. It leverages dictionary-based and NER (Named Entity Recognition) approaches, with flexible configuration for language, matching strategies, and more.

---

## Key Features

- Unified API for extracting multiple entity types from text
- Support for job positions, organizations, universities, skills, transaction purposes, and custom entities
- Dictionary-based and NER-based extraction with configurable fallback
- Language auto-detection and multi-language support
- Caching for efficient repeated extraction
- Progress bar support for large batch processing
- Extensible: create custom extractors for new entity types

---

## Dependencies

### Standard Library
- `logging`
- `typing` (`Dict`, `List`, `Any`, `Optional`)

### Internal Modules
- `pamola_core.utils.nlp.cache` (`cache_function`)
- `pamola_core.utils.nlp.entity` (`create_entity_extractor`, `extract_entities`)

---

## Exception Classes

> **Note:** This module does not define custom exception classes directly. Exceptions are typically raised by underlying extractors or dependencies. Handle exceptions as shown below.

### Example: Handling Extraction Errors
```python
try:
    results = extract_entities(["Sample text"], entity_type="job")
except Exception as e:
    # Handle extraction or configuration errors
    print(f"Entity extraction failed: {e}")
```
- **When raised:** Errors may occur due to invalid configuration, missing dictionary files, or issues in the underlying NER models.

---

## Main Functions and Usage

### 1. `extract_entities`

```python
def extract_entities(
    texts: List[str],
    entity_type: str = "generic",
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    record_ids: Optional[List[str]] = None,
    show_progress: bool = False,
    **kwargs
) -> Dict[str, Any]:
```
**Parameters:**
- `texts`: List of text strings to process
- `entity_type`: Type of entities to extract (e.g., "job", "organization", "skill", "transaction", "generic")
- `language`: Language code or "auto" for detection
- `dictionary_path`: Path to dictionary file (optional)
- `match_strategy`: Matching strategy (e.g., "specific_first", "domain_prefer")
- `use_ner`: Use NER models if dictionary match fails
- `record_ids`: Optional list of record IDs
- `show_progress`: Show progress bar
- `**kwargs`: Additional extractor parameters

**Returns:**
- Dictionary with extraction results, including entities, categories, and statistics

**Raises:**
- Exceptions from underlying extractors (e.g., file not found, model errors)

#### Example Usage
```python
# Extract job positions from a list of texts
results = extract_entities([
    "John is a Senior Data Scientist at Acme Corp.",
    "Jane works as a Software Engineer."
], entity_type="job")
```

---

### 2. `extract_job_positions`

```python
def extract_job_positions(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    seniority_detection: bool = True,
    **kwargs
) -> Dict[str, Any]:
```
**Parameters:**
- `texts`: List of text strings
- `language`: Language code or "auto"
- `dictionary_path`: Path to dictionary file (optional)
- `use_ner`: Use NER if dictionary match fails
- `seniority_detection`: Detect seniority levels
- `**kwargs`: Additional parameters

**Returns:** Extraction results dictionary

#### Example
```python
# Extract job positions with seniority detection
data = ["Alice is a Junior Analyst."]
results = extract_job_positions(data, seniority_detection=True)
```

---

### 3. `extract_organizations`

```python
def extract_organizations(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    organization_type: str = "any",
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
```
**Parameters:**
- `texts`: List of text strings
- `language`: Language code or "auto"
- `dictionary_path`: Path to dictionary file (optional)
- `organization_type`: Type of organization (e.g., 'company', 'university')
- `use_ner`: Use NER if dictionary match fails
- `**kwargs`: Additional parameters

**Returns:** Extraction results dictionary

#### Example
```python
# Extract company names from text
results = extract_organizations(["Worked at Google and MIT."], organization_type="company")
```

---

### 4. `extract_universities`

```python
def extract_universities(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
```
**Parameters:**
- `texts`: List of text strings
- `language`: Language code or "auto"
- `dictionary_path`: Path to dictionary file (optional)
- `use_ner`: Use NER if dictionary match fails
- `**kwargs`: Additional parameters

**Returns:** Extraction results dictionary

#### Example
```python
# Extract university names
results = extract_universities(["Graduated from Stanford University."])
```

---

### 5. `extract_skills`

```python
def extract_skills(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    skill_type: str = "technical",
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
```
**Parameters:**
- `texts`: List of text strings
- `language`: Language code or "auto"
- `dictionary_path`: Path to dictionary file (optional)
- `skill_type`: Type of skills (e.g., 'technical', 'soft')
- `use_ner`: Use NER if dictionary match fails
- `**kwargs`: Additional parameters

**Returns:** Extraction results dictionary

#### Example
```python
# Extract technical skills
results = extract_skills(["Python, SQL, and communication skills."])
```

---

### 6. `extract_transaction_purposes`

```python
def extract_transaction_purposes(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
```
**Parameters:**
- `texts`: List of text strings
- `language`: Language code or "auto"
- `dictionary_path`: Path to dictionary file (optional)
- `use_ner`: Use NER if dictionary match fails
- `**kwargs`: Additional parameters

**Returns:** Extraction results dictionary

#### Example
```python
# Extract transaction purposes
results = extract_transaction_purposes(["Payment for consulting services."])
```

---

### 7. `create_custom_entity_extractor`

```python
def create_custom_entity_extractor(
    entity_type: str,
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    **kwargs
) -> Any:
```
**Parameters:**
- `entity_type`: Type of entities to extract
- `language`: Language code or "auto"
- `dictionary_path`: Path to dictionary file (optional)
- `match_strategy`: Matching strategy
- `use_ner`: Use NER if dictionary match fails
- `**kwargs`: Additional parameters

**Returns:** Entity extractor instance

#### Example
```python
# Create a custom extractor for certifications
extractor = create_custom_entity_extractor(
    entity_type="certification",
    dictionary_path="/path/to/certifications.json"
)
results = extractor.extract(["Certified Kubernetes Administrator"])
```

---

## Dependency Resolution and Completion Validation

- The module relies on the `pamola_core.utils.nlp.entity` package for actual extraction logic, which handles dictionary loading, NER model invocation, and match strategy resolution.
- Caching is provided via `cache_function` to avoid redundant computation.
- Completion validation (e.g., ensuring all texts are processed) is handled by the underlying extractor logic.

---

## Usage Scenarios

### Accessing Outputs
```python
# Extract organizations and access the results
data = ["Worked at OpenAI and Harvard University."]
results = extract_organizations(data)
print(results["entities"])
```

### Handling Failed Dependencies
```python
try:
    # Attempt extraction with a missing dictionary
    extract_skills(["Python"], dictionary_path="/invalid/path.json")
except Exception as e:
    # Log or handle the error
    print(f"Extraction failed: {e}")
```

### Using in a Pipeline (with BaseTask)
```python
class MyTask(BaseTask):
    def run(self):
        # Use entity extraction as part of a processing pipeline
        orgs = extract_organizations(self.input_texts)
        self.save_results(orgs)
```

### Continue-on-Error with Logging
```python
for text in texts:
    try:
        result = extract_entities([text], entity_type="job")
    except Exception as e:
        logger.warning(f"Failed to extract from: {text} | Error: {e}")
        continue
```

---

## Integration Notes

- Designed for seamless integration with pipeline tasks (e.g., `BaseTask`).
- Can be used as a standalone utility or as part of a larger data processing workflow.
- Supports both batch and single-record extraction.

---

## Error Handling and Exception Hierarchy

- Most errors are propagated from the underlying extractor or file system (e.g., missing dictionary, invalid parameters).
- Use try/except blocks to handle errors gracefully in production pipelines.
- No custom exception hierarchy is defined in this module; refer to the extractor's documentation for specific error types.

---

## Configuration Requirements

- For dictionary-based extraction, provide a valid `dictionary_path`.
- For NER-based extraction, ensure the required models are available and properly configured.
- `language` can be set to "auto" for automatic detection, or specify a language code (e.g., "en").
- `match_strategy` controls how matches are resolved; choose based on your use case.

---

## Security Considerations and Best Practices

- **Path Security:** Only use trusted dictionary files and NER models. Avoid using untrusted or user-supplied paths.
- **Risks of Disabling Path Security:**
    - Using unvalidated dictionary paths can lead to code execution or data leakage.
    - Always validate and sanitize external paths before use.

### Security Failure Example
```python
# BAD: Using an untrusted dictionary path
untrusted_path = input("Enter dictionary path: ")
results = extract_entities(["text"], dictionary_path=untrusted_path)
# This can lead to security issues if the path is malicious.

# GOOD: Validate the path before use
import os
if os.path.exists(trusted_path):
    results = extract_entities(["text"], dictionary_path=trusted_path)
else:
    raise ValueError("Dictionary path is invalid or untrusted.")
```

---

## Internal vs. External Dependencies

- **Internal:** Use built-in entity types and dictionaries provided by the PAMOLA Core framework for most use cases.
- **External (Absolute Path):** Only specify absolute paths for custom dictionaries or models that are not part of the standard pipeline. Ensure these are secure and trusted.

---

## Best Practices

1. **Use Built-in Entity Types:** Prefer standard entity types ("job", "organization", etc.) for consistency and maintainability.
2. **Validate Custom Dictionaries:** Always validate the existence and integrity of custom dictionary files before use.
3. **Handle Errors Gracefully:** Use try/except blocks to catch and log extraction errors, especially in batch processing.
4. **Leverage Caching:** Take advantage of the built-in caching to improve performance for repeated extractions.
5. **Document Custom Extractors:** When creating custom entity extractors, document their configuration and intended use.
6. **Secure Path Usage:** Never use untrusted paths for dictionary or model files.
7. **Monitor Extraction Quality:** Regularly review extraction results and update dictionaries/models as needed.
