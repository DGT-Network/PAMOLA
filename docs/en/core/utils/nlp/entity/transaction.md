# PAMOLA Core: Transaction Purpose Entity Extractor Module

## Overview

The `transaction.py` module provides advanced functionality for extracting transaction purposes and financial categories from transaction descriptions. It is a core component of the PAMOLA Core framework, enabling automated financial data analysis, categorization, and enrichment by leveraging both pattern-based and NLP-based extraction techniques.

## Key Features

- **Automatic Extraction of Transaction Purposes**: Identifies the purpose and category of financial transactions from raw text descriptions.
- **Pattern and NLP-Based Methods**: Combines regular expression patterns, keyword matching, and Named Entity Recognition (NER) for robust extraction.
- **Customizable Cleaning**: Supports removal of account numbers, IDs, and dates from transaction text for privacy and accuracy.
- **Multi-Language Support**: Integrates with language-specific NLP models for internationalization.
- **Extensible Categories**: Easily extendable set of transaction categories and keywords.
- **Integration with PAMOLA Core NLP Model Manager**: Utilizes internal model management for advanced entity extraction.

## Dependencies

### Standard Library
- `logging`: For logging and debugging.
- `re`: Regular expressions for pattern matching.
- `typing`: Type hints for better code clarity and safety.

### Internal Modules
- `pamola_core.utils.nlp.entity.base`: Provides `BaseEntityExtractor` and `EntityMatchResult` base classes.
- `pamola_core.utils.nlp.model_manager`: Provides `NLPModelManager` for model and extractor management.

## Exception Classes

This module does not define custom exception classes. Instead, it uses standard Python exceptions and logs errors internally. Example error handling:

```python
try:
    result = extractor._extract_with_ner(text, normalized_text, language)
except Exception as e:
    logger.error(f"Extraction failed: {e}")
```

- **When exceptions are raised**: Exceptions are typically raised during model loading, NER extraction, or pattern matching failures. These are caught and logged, ensuring the pipeline continues without crashing.

## Main Classes

### TransactionPurposeExtractor

#### Constructor
```python
def __init__(
    self,
    remove_account_numbers: bool = True,
    remove_dates: bool = True,
    **kwargs
)
```
**Parameters:**
- `remove_account_numbers`: Whether to remove account numbers and IDs from text (default: True).
- `remove_dates`: Whether to remove dates from text (default: True).
- `**kwargs`: Additional parameters for the base extractor.

#### Key Attributes
- `remove_account_numbers`: Controls removal of account numbers/IDs.
- `remove_dates`: Controls removal of dates.
- `transaction_categories`: Dictionary of transaction categories and associated keywords.
- `account_pattern`: Regex pattern for account numbers/IDs.
- `date_pattern`: Regex pattern for dates.

#### Public Methods

##### _get_entity_type
```python
def _get_entity_type(self) -> str
```
- **Returns:** Entity type string ("transaction").

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
    - `text`: Original transaction description.
    - `normalized_text`: Preprocessed/normalized text.
    - `language`: Language code (e.g., 'en', 'ru').
- **Returns:** `EntityMatchResult` if a match is found, else `None`.
- **Raises:** Logs and handles all exceptions internally.

##### _clean_transaction_text
```python
def _clean_transaction_text(self, text: str) -> str
```
- **Parameters:**
    - `text`: Text to clean.
- **Returns:** Cleaned text with optional removal of account numbers and dates.

##### _determine_transaction_category
```python
def _determine_transaction_category(self, text: str) -> Tuple[str, float]
```
- **Parameters:**
    - `text`: Transaction text.
- **Returns:** Tuple of (category name, confidence score).

## Dependency Resolution and Completion Validation

- **Model Management**: Uses `NLPModelManager` to dynamically load and manage language-specific models and extractors.
- **Fallback Logic**: If pattern matching fails, attempts specialized extractors, then general NER, ensuring robust extraction.
- **Validation**: Confidence scores are used to validate extraction quality; only results above a minimum threshold are returned.

## Usage Examples

### Basic Extraction
```python
from pamola_core.utils.nlp.entity.transaction import TransactionPurposeExtractor

# Create the extractor
extractor = TransactionPurposeExtractor(remove_account_numbers=True, remove_dates=True)

# Example transaction text
text = "Salary payment 2024-05-01 to account 12345678"
normalized_text = text.lower()
language = 'en'

# Extract purpose
result = extractor._extract_with_ner(text, normalized_text, language)
if result:
    print(result.category)  # e.g., TRANSACTION_SALARY
```

### Handling Extraction Failures
```python
try:
    result = extractor._extract_with_ner(text, normalized_text, language)
    if not result:
        print("No transaction purpose found.")
except Exception as e:
    # Handle unexpected errors
    print(f"Extraction failed: {e}")
```

### Continue-on-Error Example
```python
try:
    result = extractor._extract_with_ner(text, normalized_text, language)
except Exception as e:
    logger.warning(f"Extraction error, continuing: {e}")
    result = None
```

## Error Handling and Exception Hierarchy

- **Standard Exceptions**: All errors are caught and logged. No custom exceptions are defined in this module.
- **Logging**: Errors are logged at debug or error level for traceability.
- **Best Practice**: Always check for `None` before using extraction results.

## Configuration Requirements

- **NLP Model Manager**: Requires `NLPModelManager` to be properly configured and available.
- **Language Support**: Ensure language models for required languages are installed and accessible.
- **Category Extension**: To add new categories, update the `transaction_categories` dictionary.

## Security Considerations and Best Practices

- **Sensitive Data Handling**: By default, account numbers and dates are removed to prevent leakage of sensitive information.
- **Risks of Disabling Path Security**: If `remove_account_numbers` or `remove_dates` is set to `False`, sensitive data may be exposed in logs or outputs.

### Security Failure Example
```python
# BAD: Disabling account number removal exposes sensitive data
extractor = TransactionPurposeExtractor(remove_account_numbers=False)
result = extractor._extract_with_ner("Transfer to account 9876543210", "transfer to account 9876543210", "en")
# The result may include the account number, which is a security risk.
```

### Secure Usage Example
```python
# GOOD: Always remove account numbers and dates
extractor = TransactionPurposeExtractor(remove_account_numbers=True, remove_dates=True)
```

## Internal vs. External Dependencies

- **Internal**: Uses internal NLP models and extractors for entity recognition.
- **External**: No direct support for absolute path dependencies; all models are managed via the internal model manager.

## Best Practices

1. **Always Remove Sensitive Data**: Keep `remove_account_numbers` and `remove_dates` enabled unless absolutely necessary.
2. **Extend Categories Carefully**: When adding new transaction categories, ensure keywords are unique and relevant.
3. **Check Extraction Results**: Always verify that extraction results are not `None` before use.
4. **Integrate with Model Manager**: Use the provided `NLPModelManager` for all model and extractor needs.
5. **Log Errors**: Use logging to capture and debug extraction issues.
6. **Internationalization**: Ensure language models are available for all supported languages in your data.
