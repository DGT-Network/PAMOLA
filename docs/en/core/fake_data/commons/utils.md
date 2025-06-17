# fake_data.commons.utils Module Documentation

## Overview

The `utils.py` module is a foundation component of the `fake_data` package, providing essential utilities for generating, manipulating, and transforming data during anonymization processes. This module offers a rich set of functions for string handling, language detection, data transformation, and other common operations needed when creating realistic synthetic data that preserves statistical properties while protecting privacy.

## Architecture

The module follows a functional programming approach with utility functions organized into logical categories:

1. **String Processing**: Functions for normalizing, transforming, and validating text
2. **Language Utilities**: Tools for language detection and transliteration
3. **Data Transformation**: Functions for generating deterministic values and consistent profiles
4. **File and Dictionary Handling**: Utilities for loading and managing dictionaries of replacement values
5. **Integration Layer**: Wrappers around pamola core PAMOLA.CORE infrastructure for seamless operation

This architecture integrates with the PAMOLA.CORE system infrastructure while providing specialized functionality for fake data generation:

```
┌─────────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│    PAMOLA.CORE Core Utils   │     │ fake_data.utils   │     │   Other fake_data   │
│   (io, logging)     │◄────┤   Integration     │────►│     Components      │
└─────────────────────┘     └───────┬───────────┘     └─────────────────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────────┐
              │        Utility Function Categories         │
              │                                            │
              │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
              │  │  String  │  │ Language │  │  Data    │  │
              │  │Processing│  │ Utilities│  │Transform │  │
              │  └──────────┘  └──────────┘  └──────────┘  │
              └────────────────────────────────────────────┘
```

## Key Capabilities

The module provides the following categories of functionality:

1. **String Handling**: Normalization, case preservation, punctuation removal
2. **Cryptographic Functions**: Hashing with various algorithms and salting
3. **Deterministic Generation**: Creating consistent replacement values based on seeds
4. **Language Processing**: Detecting languages, transliterating between writing systems
5. **Phone Formatting**: Region-aware phone number formatting
6. **Name Processing**: Gender detection from names, username generation
7. **Data Analysis**: Counting values, analyzing distributions
8. **Dictionary Management**: Loading, caching, and accessing replacement dictionaries
9. **Progress Tracking**: Integration with PAMOLA.CORE progress reporting
10. **Demographic Profile Generation**: Creating consistent demographics for synthetic identities

## Key Components

### Pamola Core String Utilities

| Function | Purpose |
|----------|---------|
| `normalize_string` | Standardizes string format, case, and punctuation |
| `hash_value` | Creates cryptographic hash of values with multiple algorithm options |
| `generate_deterministic_value` | Creates consistent pseudorandom values based on seeds |
| `validate_with_pattern` | Validates string against regular expression patterns |

### Language and Internationalization

| Function | Purpose |
|----------|---------|
| `detect_language` | Identifies the most likely language of text |
| `detect_gender_from_name` | Determines probable gender from a personal name |
| `transliterate` | Converts text between different writing systems |
| `format_phone_number` | Formats phone numbers according to regional standards |
| `get_language_for_region` | Maps country/region codes to primary language codes |

### Data Generation

| Function | Purpose |
|----------|---------|
| `create_username_from_name` | Generates usernames from personal names |
| `generate_random_date` | Creates random dates within specified ranges |
| `create_consistent_profile` | Builds internally consistent demographic profiles |
| `weighted_random_choice` | Selects items with probability proportional to weights |

### Dictionary and Resource Management

| Function | Purpose |
|----------|---------|
| `load_dictionary` | Loads reference dictionaries from various sources with caching |
| `clear_dictionary_cache` | Releases memory used by cached dictionaries |
| `ensure_dir` | Creates directories if they don't exist |
| `get_timestamped_filename` | Generates filenames with timestamps |
| `get_file_stats` | Retrieves statistics about files |

### Integration with PAMOLA.CORE Infrastructure

| Function | Purpose |
|----------|---------|
| `get_progress_bar` | Creates standardized progress tracking |
| `save_dataframe` | Saves pandas DataFrame with progress tracking |
| `save_visualization` | Saves visualization artifacts |

## Usage Examples

### String Processing

```python
from pamola_core.fake_data.commons.utils import normalize_string, transliterate, hash_value

# Normalize a string
original = "John Smith-Williams, Jr."
normalized = normalize_string(original, keep_case=False, remove_punctuation=True)
# result: "john smith williams jr"

# Transliterate between writing systems
russian_name = "Иван Петров"
latin_name = transliterate(russian_name, source_lang="ru", target_lang="en")
# result: "Ivan Petrov"

# Create a deterministic hash
hash_value("sensitive-data", salt="mysalt", algorithm="sha256")
# result: [sha256 hash hex string]
```

### Profile Generation

```python
from pamola_core.fake_data.commons.utils import create_consistent_profile, create_username_from_name

# Generate a consistent demographic profile
profile = create_consistent_profile(
    base_seed="user123",
    gender="F",
    region="US",
    age_range=(25, 35)
)
# result: {
#   'gender': 'F',
#   'region': 'US',
#   'birth_date': datetime.datetime(1991, 5, 12, 0, 0),
#   'age': 33,
#   'language': 'en'
# }

# Create a username
username = create_username_from_name(
    first_name="Jane",
    last_name="Smith",
    style="firstlast",
    min_length=6,
    max_length=15,
    exclude_list=["janesmith", "jane.smith"]
)
# result: "janesmith123"
```

### Dictionary Handling

```python
from pamola_core.fake_data.commons.utils import load_dictionary

# Load a dictionary of first names
first_names = load_dictionary(
    source="data/dictionaries/first_names_en.csv",
    encoding="utf-8",
    cache=True
)

# Load from a string or dictionary object
raw_data = """id,value,gender
1,John,M
2,Jane,F
3,Alex,U"""

names_df = load_dictionary(
    source=raw_data,
    delimiter=",",
    cache=False
)
```

### Regional Formatting

```python
from pamola_core.fake_data.commons.utils import format_phone_number, get_language_for_region

# Format phone numbers by region
us_phone = format_phone_number("12345678901", region="US")
# result: "+1 (234) 567-8901"

ru_phone = format_phone_number("79051234567", region="RU")
# result: "+7 (905) 123-45-67"

# Get default language for a region
language = get_language_for_region("DE")
# result: "de"
```

## Parameters and Return Values

### String Normalization

```python
def normalize_string(
    value: str,           # Input string to normalize
    keep_case: bool=False,  # Whether to preserve case
    remove_punctuation: bool=False  # Whether to remove punctuation
) -> str:  # Normalized string
```

### Deterministic Value Generation

```python
def generate_deterministic_value(
    seed_value: Any,        # Seed for reproducible generation
    length: int=10,         # Length of generated value
    chars: Optional[str]=None  # Character set (default: alphanumeric)
) -> str:  # Deterministically generated string
```

### Language Detection

```python
def detect_language(
    text: str  # Text sample to analyze
) -> str:  # ISO language code ('en', 'ru', etc.) or 'unknown'
```

### Transliteration

```python
def transliterate(
    text: str,                # Text to transliterate
    source_lang: str="ru",    # Source language ISO code
    target_lang: str="en"     # Target language ISO code
) -> str:  # Transliterated text
```

### Dictionary Loading

```python
def load_dictionary(
    source: Union[str, Path, IO, Dict],  # Dictionary source
    io_module: Optional[Any]=None,       # IO module for loading
    encoding: str="utf-8",               # File encoding
    delimiter: str=",",                  # CSV delimiter
    cache: bool=True                     # Whether to cache dictionary
) -> pd.DataFrame:  # Dictionary data as DataFrame
```

## Integration with PAMOLA.CORE Systems

The `utils.py` module integrates with several PAMOLA.CORE components:

1. **PAMOLA.CORE IO System**: Uses `pamola_core.utils.io` for file operations
2. **PAMOLA.CORE Logging**: Uses `pamola_core.utils.logging` for consistent log formatting
3. **PAMOLA.CORE Progress**: Uses `pamola_core.utils.progress` for operation tracking

This integration allows the module to leverage the PAMOLA.CORE infrastructure while providing specialized functionality for fake data generation. The module follows PAMOLA.CORE conventions for:

- Error handling and logging
- File path management
- Progress reporting
- Artifact generation

## Fallback Mechanisms

The module includes robust fallback mechanisms for pamola core functionality to ensure operation even when optional dependencies are unavailable:

1. **Language Detection**: Falls back to regex-based detection if `langdetect` library is missing
2. **Transliteration**: Uses basic character mapping if `transliterate` library is absent
3. **File Operations**: Provides basic implementations if PAMOLA.CORE IO functions fail

## Design Considerations

The module was designed with the following principles in mind:

1. **Determinism**: Functions produce consistent outputs for the same inputs
2. **Extensibility**: Easy to add support for new languages, regions, or data types
3. **Performance**: Optimized for both speed and memory usage
4. **Robustness**: Graceful fallbacks and error handling
5. **Integration**: Seamless operation with PAMOLA.CORE infrastructure
6. **Internationalization**: Strong support for multiple languages and regions

## Performance Considerations

For performance-sensitive operations, the module provides:

1. **Dictionary Caching**: Avoids repeated loading of dictionaries
2. **Memory-Efficient Processing**: Careful use of resources for large datasets
3. **Optimized Algorithms**: Efficient implementations of common operations
4. **Progress Tracking**: Provides visibility into long-running operations

## Module Naming Considerations

The current module name `pamola_core.fake_data.commons.utils` may cause confusion with `pamola_core.utils`. Consider renaming options:

1. **`pamola_core.fake_data.commons.data_utils`**: Clarifies its focus on data manipulation
2. **`pamola_core.fake_data.commons.fake_utils`**: Emphasizes its role in fake data generation
3. **`pamola_core.fake_data.commons.transform_utils`**: Highlights its transformation capabilities

Renaming would require updating imports in dependent modules but would improve code clarity and maintainability.

## Conclusion

The `utils.py` module provides essential functionality for the `fake_data` package, enabling the creation of realistic synthetic data while preserving privacy. With robust implementations of string processing, language handling, and data transformation utilities, it serves as a foundation for the entire anonymization system.

Its integration with PAMOLA.CORE infrastructure ensures consistent operation within the larger system while providing specialized capabilities for fake data generation. The module's comprehensive approach to internationalization, deterministic generation, and fallback mechanisms makes it resilient and adaptable to various anonymization scenarios.