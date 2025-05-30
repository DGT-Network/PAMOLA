# PAMOLA Core: Skill Entity Extractor Module

## Overview

The Skill Entity Extractor module is a core component of the PAMOLA Core framework, designed to identify and extract skills, technologies, and competencies from unstructured text. It is tailored for technical and professional domains, supporting both keyword-based and Named Entity Recognition (NER) approaches. This module enables downstream tasks such as candidate profiling, job matching, and analytics by providing structured skill information from raw text.

---

## Key Features

- **Skill Extraction**: Identifies technical, soft, and language skills from text using both keyword matching and NER models.
- **Category Classification**: Classifies extracted skills into categories (e.g., programming languages, databases, frameworks, tools, soft skills, languages).
- **Customizable Parameters**: Supports configuration of skill type and minimum skill length.
- **Language Support**: Utilizes language-specific NER models for broader applicability.
- **Fallback Logic**: Uses noun phrase extraction when explicit skill entities are not found.
- **Integration Ready**: Designed for seamless use with PAMOLA's BaseEntityExtractor and pipeline components.

---

## Dependencies

### Standard Library
- `logging`
- `typing` (List, Optional, Tuple)

### Internal Modules
- `pamola_core.utils.nlp.entity.base` (`BaseEntityExtractor`, `EntityMatchResult`)
- `pamola_core.utils.nlp.model_manager` (`NLPModelManager`)
- `pamola_core.utils.nlp.tokenization` (`tokenize`)
- `pamola_core.utils.nlp.stopwords` (`remove_stopwords`)

---

## Exception Classes

This module does not define custom exception classes. All exceptions are handled internally, typically logged and returned as `None` from extraction methods. Standard Python exceptions (e.g., `Exception`) may be raised in rare cases of critical failure.

**Example: Handling Extraction Errors**
```python
try:
    result = skill_extractor._extract_with_ner(text, normalized_text, language)
    if result is None:
        # Handle no skill found
        print("No skills extracted.")
except Exception as e:
    # Handle unexpected errors
    print(f"Skill extraction failed: {e}")
```
**When Raised:**
- Errors may occur if the NER model is unavailable, or if there are issues with text processing. These are logged and do not propagate unless explicitly re-raised.

---

## Main Classes

### `SkillExtractor`

#### Constructor
```python
def __init__(
    self,
    skill_type: str = 'technical',
    min_skill_length: int = 3,
    **kwargs
)
```
**Parameters:**
- `skill_type`: Type of skills to extract (e.g., 'technical', 'soft', 'language').
- `min_skill_length`: Minimum length for a skill term.
- `**kwargs`: Additional parameters for the base extractor.

#### Key Attributes
- `skill_type`: The type of skills targeted for extraction.
- `min_skill_length`: Minimum length for a skill to be considered.
- `skill_categories`: Dictionary of skill categories and their associated skills.
- `all_skills`: Flattened list of all known skills.

#### Public Methods

##### `def _get_entity_type(self) -> str`
Returns the entity type string for this extractor.
- **Returns:** `str` — Always returns "skill".

##### `def _extract_with_ner(
    self,
    text: str,
    normalized_text: str,
    language: str
) -> Optional[EntityMatchResult]`
Extracts skills using both keyword matching and NER models.
- **Parameters:**
    - `text`: Original input text.
    - `normalized_text`: Preprocessed version of the text.
    - `language`: Language code (e.g., 'en').
- **Returns:** `EntityMatchResult` or `None` if no skills are found.
- **Raises:** Logs and handles all exceptions internally.

##### `def _extract_skills_from_text(self, text: str) -> List[str]`
Extracts skills from text using keyword matching.
- **Parameters:**
    - `text`: Text to extract skills from.
- **Returns:** `List[str]` — List of matched skills.

##### `def _determine_skill_category(self, skills: List[str]) -> Tuple[str, float]`
Determines the most likely skill category from a list of skills.
- **Parameters:**
    - `skills`: List of extracted skills.
- **Returns:** `Tuple[str, float]` — Category name and confidence score.

---

## Dependency Resolution and Completion Validation

- **NER Model Management**: Uses `NLPModelManager` to load and cache language-specific NER models. If a model is unavailable, logs a warning and falls back to noun phrase extraction.
- **Keyword Matching**: Prioritizes known skills for high-confidence extraction. If no matches, tokenizes and removes stopwords for broader matching.
- **Category Determination**: Tallies matches by category and computes a confidence score for the best match.

---

## Usage Examples

### Basic Skill Extraction
```python
from pamola_core.utils.nlp.entity.skill import SkillExtractor

# Create a skill extractor for technical skills
skill_extractor = SkillExtractor(skill_type='technical', min_skill_length=3)

# Example text
text = "Experienced in Python, SQL, and Docker. Strong communication skills."
normalized_text = text.lower()
language = 'en'

# Extract skills
result = skill_extractor._extract_with_ner(text, normalized_text, language)
if result:
    print(result.category)  # e.g., SKILL_PROGRAMMING_LANGUAGES
    print(result.alias)     # e.g., skill_programming_languages
else:
    print("No skills found.")
```

### Handling Extraction Failures
```python
try:
    result = skill_extractor._extract_with_ner(text, normalized_text, language)
    if not result:
        # No skills found, handle gracefully
        print("No skills detected.")
except Exception as e:
    # Log or handle unexpected errors
    print(f"Extraction error: {e}")
```

### Integration with BaseTask
```python
from pamola_core.utils.nlp.entity.skill import SkillExtractor
from pamola_core.pipeline.base_task import BaseTask

class SkillExtractionTask(BaseTask):
    def run(self, text):
        extractor = SkillExtractor()
        result = extractor._extract_with_ner(text, text.lower(), 'en')
        # Use result in pipeline
```

---

## Error Handling and Exception Hierarchy

- All exceptions are caught and logged within extraction methods.
- No custom exception classes are defined in this module.
- Errors such as missing NER models or text processing failures are handled gracefully, returning `None`.

---

## Configuration Requirements

- **Skill Type**: Set via `skill_type` parameter (e.g., 'technical', 'soft', 'language').
- **Minimum Skill Length**: Set via `min_skill_length` parameter.
- **NER Model Availability**: Ensure language-specific models are available in `NLPModelManager`.

---

## Security Considerations and Best Practices

- **Path Security**: This module does not process file paths directly, but downstream usage should avoid exposing sensitive data in extracted text.
- **Model Security**: Ensure that NER models are trusted and not tampered with.
- **Input Validation**: Always sanitize and validate input text to prevent injection attacks.

**Example: Security Failure**
```python
# Risk: Using untrusted NER models
model = nlp_model_manager.get_model('spacy', 'en')
# If the model is compromised, extraction results may be manipulated.
# Always verify model sources and integrity.
```
**Risks of Disabling Path Security**
- If downstream code disables path or data validation, sensitive information may be leaked or processed incorrectly.

---

## Internal vs. External Dependencies

- **Internal**: All skill definitions and categories are managed within the module.
- **External**: NER models are loaded via `NLPModelManager` and must be available in the environment.

---

## Best Practices

1. **Configure Skill Types Appropriately**: Use the `skill_type` parameter to target relevant skills for your use case.
2. **Validate Input Text**: Always preprocess and normalize input text for consistent extraction.
3. **Handle Extraction Failures Gracefully**: Check for `None` results and log or handle as needed.
4. **Integrate with Pipeline Components**: Use with `BaseTask` or other pipeline elements for end-to-end processing.
5. **Monitor Model Availability**: Ensure required NER models are present and up-to-date.
6. **Avoid Hardcoding Skills**: Extend `skill_categories` as needed for your domain, but avoid hardcoding outside the module.
7. **Log All Errors**: Use logging to capture and diagnose extraction issues.
