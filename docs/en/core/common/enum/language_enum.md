# Language Enumeration

**Module:** `pamola_core.common.enum.language_enum`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

Language defines the supported languages for PAMOLA.CORE operations. It enables multi-language support for UI generation, text processing, and localization across the framework.

## Members

| Member | Value | Language Name | Code |
|--------|-------|---------------|------|
| `ENGLISH` | `"en"` | English | ISO 639-1 |
| `VIETNAMESE` | `"vi"` | Vietnamese (Tiếng Việt) | ISO 639-1 |
| `RUSSIAN` | `"ru"` | Russian (Русский) | ISO 639-1 |

## Usage

### Basic Enumeration Access

```python
from pamola_core.common.enum.language_enum import Language

# Access members
lang = Language.ENGLISH
print(lang.value)  # Output: "en"
print(lang.name)   # Output: "ENGLISH"

# All supported languages
languages = [lang for lang in Language]
```

### Language-Based Conditionals

```python
from pamola_core.common.enum.language_enum import Language

def get_greeting(language: Language) -> str:
    greetings = {
        Language.ENGLISH: "Hello",
        Language.VIETNAMESE: "Xin chào",
        Language.RUSSIAN: "Привет"
    }
    return greetings.get(language, "Hello")

msg = get_greeting(Language.VIETNAMESE)  # "Xin chào"
```

### Listing Available Languages

```python
from pamola_core.common.enum.language_enum import Language

# Get all language codes
codes = [lang.value for lang in Language]
print(codes)  # ["en", "vi", "ru"]

# Get language names
names = [lang.name for lang in Language]
print(names)  # ["ENGLISH", "VIETNAMESE", "RUSSIAN"]
```

## Member Descriptions

### ENGLISH
**Value:** `"en"`

English language support. Used as the default language in the framework.

**Use case:** Default UI text, error messages, documentation, and international contexts.

**Region Coverage:** United States, United Kingdom, Canada, Australia, etc.

### VIETNAMESE
**Value:** `"vi"`

Vietnamese language support for localization in Vietnamese-speaking regions.

**Use case:** UI text, error messages, and documentation in Vietnamese.

**Region Coverage:** Vietnam and Vietnamese diaspora communities.

### RUSSIAN
**Value:** `"ru"`

Russian language support for localization in Russian-speaking regions.

**Use case:** UI text, error messages, and documentation in Russian.

**Region Coverage:** Russia, Belarus, Kazakhstan, and Russian diaspora communities.

## Related Components

- **UI Localization:** Language selection controls form generation and display
- **Form Generation:** Custom components and field labels adapt based on language
- **Error Messages:** Exception text can be rendered in selected language
- **Text Processing:** Language-specific patterns for validation

## Common Patterns

### Detect Language from Code

```python
from pamola_core.common.enum.language_enum import Language

def get_language_by_code(code: str) -> Language:
    """Get language enum from ISO 639-1 code."""
    try:
        return Language[code.upper()]  # Name lookup
    except KeyError:
        # Fallback to value lookup
        return next(
            (l for l in Language if l.value == code.lower()),
            Language.ENGLISH
        )

lang = get_language_by_code("vi")  # Returns Language.VIETNAMESE
```

### Multilingual Configuration

```python
from pamola_core.common.enum.language_enum import Language
from typing import Dict

multilingual_config = {
    Language.ENGLISH: {
        "locale": "en_US",
        "encoding": "UTF-8",
        "date_format": "%Y-%m-%d"
    },
    Language.VIETNAMESE: {
        "locale": "vi_VN",
        "encoding": "UTF-8",
        "date_format": "%d/%m/%Y"
    },
    Language.RUSSIAN: {
        "locale": "ru_RU",
        "encoding": "UTF-8",
        "date_format": "%d.%m.%Y"
    }
}
```

### Language-Specific Validation

```python
from pamola_core.common.enum.language_enum import Language

def get_validation_patterns(language: Language) -> dict:
    """Get language-specific regex patterns."""
    patterns = {
        Language.ENGLISH: {
            "phone": r"^\+?1?\d{10}$",  # US format
            "postal": r"^\d{5}(?:-\d{4})?$"
        },
        Language.VIETNAMESE: {
            "phone": r"^\+?84\d{9,10}$",  # Vietnam format
            "postal": None
        },
        Language.RUSSIAN: {
            "phone": r"^\+?7\d{10}$",  # Russia format
            "postal": r"^\d{6}$"
        }
    }
    return patterns.get(language, patterns[Language.ENGLISH])
```

## Best Practices

1. **Use Enums for Type Safety**
   ```python
   # Good
   config = {"language": Language.VIETNAMESE}

   # Avoid
   config = {"language": "vi"}  # String is less safe
   ```

2. **Default to English**
   ```python
   def process_with_language(language: Language = Language.ENGLISH):
       # English is safe default
       pass
   ```

3. **Validate Language Input**
   ```python
   def set_language(lang_code: str) -> Language:
       try:
           return Language[lang_code.upper()]
       except KeyError:
           # Log warning and use default
           return Language.ENGLISH
   ```

4. **Document Language-Specific Behavior**
   ```python
   def format_date(date, language: Language):
       """
       Format date according to language locale.

       - ENGLISH: YYYY-MM-DD
       - VIETNAMESE: DD/MM/YYYY
       - RUSSIAN: DD.MM.YYYY
       """
   ```

## Extensibility

To add a new language:

1. Add to the Language enum:
   ```python
   FRENCH = "fr"  # French
   ```

2. Update related components that use language-specific data
3. Update documentation with new language information
4. Test with language-specific validation and formatting

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [Form Groups](./form_groups.md) - uses language for localization

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- Three supported languages: English, Vietnamese, Russian
- Usage patterns for multilingual applications
