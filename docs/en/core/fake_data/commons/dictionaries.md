# Dictionary Subsystem Documentation

## 1. Overview

The dictionary subsystem provides a comprehensive infrastructure for managing, loading, and manipulating dictionaries used in fake data generation. It consists of a pamola core utility module `dict_helpers.py` and a dedicated package `dictionaries` containing embedded dictionaries for various data types.

This subsystem is designed to efficiently handle different dictionary formats, provide fallback mechanisms, and optimize performance through caching and validation strategies.

## 2. Architecture

### 2.1 Components

1. **Core Utility Module (`dict_helpers.py`)**:
    - Central module for dictionary management
    - Provides functions for loading, validating, and accessing dictionaries
    - Handles caching and optimization
2. **Embedded Dictionaries Package (`dictionaries/`)**:
    - Organized by data type (names, domains, phones, addresses, organizations)
    - Provides structured access to built-in dictionaries
    - Implements data-specific operations and formats
3. **Integration with Pamola Core Utilities**:
    - Utilizes project's IO utilities (`pamola_core.utils.io`)
    - Leverages logging infrastructure (`pamola_core.utils.logging`)
    - Designed for compatibility with other project components

### 2.2 Data Flow

```
┌─────────────────────┐    ┌─────────────────────────┐
│ External Dictionary │    │ Embedded Dictionaries   │
│ Files               │    │ ┌───────────────────┐   │
│                     │    │ │ names.py          │   │
│ - Text Files        │    │ │ domains.py        │   │
│ - CSV Files         │    │ │ phones.py         │   │
│ - JSON Files        │    │ │ addresses.py      │   │
└─────────┬───────────┘    │ │ organizations.py  │   │
          │                │ └───────────────────┘   │
          │                └──────────┬──────────────┘
          │                           │
          ▼                           ▼
┌────────────────────────────────────────────────────┐
│ dict_helpers.py                                    │
│                                                    │
│ 1. Attempt to load from explicit path              │
│ 2. Try finding by convention                       │
│ 3. Fall back to embedded dictionaries              │
│                                                    │
│ ┌─────────────────┐    ┌─────────────────────┐     │
│ │ Validation      │    │ Transformation      │     │
│ │ - Format Check  │    │ - Parse Full Names  │     │
│ │ - Pattern Match │    │ - Combine Sources   │     │
│ │ - Length Check  │    │ - Deduplication     │     │
│ └─────────────────┘    └─────────────────────┘     │
└─────────────────────────┬──────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Dictionary    │
                  │ Cache         │
                  └───────┬───────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │ Generator Components          │
          │                               │
          │ - Name Generator              │
          │ - Email Generator             │
          │ - Address Generator           │
          │ - Phone Generator             │
          │ - Organization Generator      │
          └───────────────────────────────┘
```

## 3. Key Functions and Capabilities

### 3.1 Dictionary Loading

The subsystem supports multiple strategies for dictionary loading:

1. **Direct Loading**:
    - Load dictionaries from specified file paths
    - Support for plain text, CSV, and JSON formats
2. **Convention-Based Loading**:
    - Find dictionaries based on naming conventions
    - Support for language, gender, and type-specific patterns
3. **Fallback Mechanism**:
    - Cascade from explicit path to conventional path to embedded dictionaries
    - Ensures availability even when external dictionaries are missing

### 3.2 Dictionary Management

1. **Caching**:
    - Efficient memory caching to avoid repeated file operations
    - Cache invalidation and clearing functions
2. **Validation**:
    - Pattern-based validation for different dictionary types
    - Length constraints and format checks
    - Statistics gathering for validation results
3. **Combination and Transformation**:
    - Merging multiple dictionaries
    - Deduplication and filtering
    - Format conversion (e.g., parsing full names into components)

## 4. Module: dict_helpers.py

### 4.1 Primary Functions

|Function|Purpose|Parameters|Return Value|
|---|---|---|---|
|`find_dictionary()`|Locates appropriate dictionary file|`dictionary_path`, `language`, `gender`, `name_type`, `dict_dir`|`Optional[Path]`|
|`load_dictionary_from_text()`|Loads dictionary from text file|`path`, `cache`, `encoding`, `validate_pattern`, `min_length`, `max_length`|`List[str]`|
|`load_dictionary_with_stats()`|Loads dictionary with statistical information|`path`, `language`, `gender`, `name_type`, `cache`, `validate_pattern`|`Dict[str, Any]`|
|`clear_dictionary_cache()`|Clears all dictionary caches|None|None|
|`get_random_items()`|Selects random items from dictionary|`dictionary`, `count`, `seed`|`List[str]`|
|`validate_dictionary()`|Validates and filters dictionary entries|`dictionary`, `dict_type`, `min_length`, `max_length`|`Tuple[List[str], Dict[str, Any]]`|
|`load_multi_dictionary()`|Loads dictionary based on parameters|`dict_type`, `params`, `fallback_to_embedded`|`List[str]`|
|`parse_full_name()`|Parses full name into components|`full_name`, `language`, `name_format`|`Dict[str, str]`|
|`get_embedded_dictionary()`|Returns embedded dictionary|`name_type`, `gender`, `language`|`List[str]`|
|`load_csv_dictionary()`|Loads dictionary from CSV file|`path`, `column_name`, `delimiter`, `encoding`, `cache`|`List[str]`|
|`combine_dictionaries()`|Combines multiple dictionaries|`dictionaries`, `dedup`, `min_length`, `max_length`|`List[str]`|

### 4.2 Configuration Options

|Option|Description|Default|
|---|---|---|
|`_DICTIONARY_PATTERNS`|Regex patterns for dictionary validation|Dictionary of patterns|
|`_dictionary_cache`|Global cache for loaded dictionaries|Empty dictionary|

## 5. Package: dictionaries

### 5.1 Module: names.py

Provides embedded dictionaries for personal names across different languages and genders.

#### Key Functions

|Function|Purpose|Parameters|Return Value|
|---|---|---|---|
|`get_ru_male_first_names()`|Gets Russian male first names|None|`List[str]`|
|`get_ru_female_first_names()`|Gets Russian female first names|None|`List[str]`|
|`get_en_male_first_names()`|Gets English male first names|None|`List[str]`|
|`get_en_female_first_names()`|Gets English female first names|None|`List[str]`|
|`get_vn_male_names()`|Gets Vietnamese male names|None|`List[str]`|
|`get_vn_female_names()`|Gets Vietnamese female names|None|`List[str]`|
|`get_names()`|Gets names by language, gender, and type|`language`, `gender`, `name_type`|`List[str]`|
|`clear_cache()`|Clears the names dictionary cache|None|None|

### 5.2 Module: domains.py

Provides embedded dictionaries for email domains and username patterns.

#### Key Functions

|Function|Purpose|Parameters|Return Value|
|---|---|---|---|
|`get_common_email_domains()`|Gets common email domains|None|`List[str]`|
|`get_business_email_domains()`|Gets business email domains|None|`List[str]`|
|`get_educational_email_domains()`|Gets educational email domains|None|`List[str]`|
|`get_username_prefixes()`|Gets common username prefixes|None|`List[str]`|
|`get_tlds_by_country()`|Gets TLDs by country code|None|`Dict[str, str]`|
|`get_domain_by_country()`|Gets domains for specific country|`country_code`|`List[str]`|
|`clear_cache()`|Clears the domains dictionary cache|None|None|

### 5.3 Module: phones.py

Provides embedded dictionaries for phone numbers, country codes, and area codes.

#### Key Functions

|Function|Purpose|Parameters|Return Value|
|---|---|---|---|
|`get_country_codes()`|Gets country phone codes|None|`Dict[str, str]`|
|`get_area_codes()`|Gets area codes for a country|`country_code`|`List[str]`|
|`get_phone_format()`|Gets phone format for a country|`country_code`|`Dict[str, str]`|
|`clear_cache()`|Clears the phones dictionary cache|None|None|

### 5.4 Module: addresses.py

Provides embedded dictionaries for address components across different countries.

#### Key Functions

|Function|Purpose|Parameters|Return Value|
|---|---|---|---|
|`get_ru_street_names()`|Gets Russian street names|None|`List[str]`|
|`get_us_street_names()`|Gets US street names|None|`List[str]`|
|`get_ru_cities()`|Gets Russian city names|None|`List[str]`|
|`get_us_cities()`|Gets US city names|None|`List[str]`|
|`get_address_component()`|Gets address components|`country_code`, `component_type`|`List[str]`|
|`get_postal_code_for_city()`|Gets postal code for a city|`country_code`, `city`|`str`|
|`clear_cache()`|Clears the addresses dictionary cache|None|None|

### 5.5 Module: organizations.py

Provides embedded dictionaries for organization names across different countries and types.

#### Key Functions

|Function|Purpose|Parameters|Return Value|
|---|---|---|---|
|`get_educational_institutions()`|Gets educational institution names|`country_code`|`List[str]`|
|`get_business_organizations()`|Gets business organization names|`country_code`, `industry`|`List[str]`|
|`get_government_organizations()`|Gets government organization names|`country_code`|`List[str]`|
|`get_organization_names()`|Gets organization names|`country_code`, `org_type`, `industry`|`List[str]`|
|`clear_cache()`|Clears the organizations dictionary cache|None|None|

## 6. External Dictionary Format Specifications

### 6.1 Text File Format

- Plain text files with one entry per line
- UTF-8 encoding
- No header row
- Trimmed whitespace
- No empty lines

### 6.2 CSV File Format

- CSV files with header row
- Delimiter can be comma, semicolon, or tab
- UTF-8 encoding
- First column used by default if column name not specified

### 6.3 Naming Conventions

For name dictionaries, files should follow the pattern:

```
{language}_{gender}_{name_type}.txt
```

- `language`: Language code (e.g., "ru", "en", "vn")
- `gender`: Gender code ("m"/"f" or omitted for gender-neutral)
- `name_type`: Type of names ("first_names", "last_names", "middle_names", "names")

Examples:

- `ru_m_first_names.txt` - Russian male first names
- `en_last_names.txt` - English last names (gender-neutral)
- `vn_f_names.txt` - Vietnamese female full names

## 7. Usage Examples

### 7.1 Loading an External Dictionary



```python
from pamola_core.fake_data.commons import dict_helpers

# Load dictionary with explicit path
names = dict_helpers.load_dictionary_from_text("path/to/dictionary.txt")

# Load dictionary with validation
valid_emails = dict_helpers.load_dictionary_from_text(
    "path/to/emails.txt",
    validate_pattern="email",
    min_length=5
)
```

### 7.2 Using Convention-Based Loading



```python
# Find and load a dictionary based on parameters
dict_path = dict_helpers.find_dictionary(
    language="ru",
    gender="M",
    name_type="first_name",
    dict_dir="DATA/dictionaries"
)

if dict_path:
    names = dict_helpers.load_dictionary_from_text(dict_path)
```

### 7.3 Using Embedded Dictionaries



```python
from pamola_core.fake_data.dictionaries import names

# Get names by language and gender
ru_male_names = names.get_names(language="ru", gender="M", name_type="first_name")
en_female_names = names.get_names(language="en", gender="F", name_type="first_name")

# Get domains
from pamola_core.fake_data.dictionaries import domains
email_domains = domains.get_common_email_domains()
```

### 7.4 Multi-Dictionary Loading with Fallback



```python
# Load with fallback to embedded dictionary
params = {
    "language": "ru",
    "gender": "F",
    "name_type": "last_name",
    "dict_dir": "DATA/dictionaries"
}

last_names = dict_helpers.load_multi_dictionary("name", params, fallback_to_embedded=True)
```

### 7.5 Working with Full Names



```python
# Parse a full name into components
parsed_name = dict_helpers.parse_full_name(
    "Иванов Иван Иванович", 
    language="ru"
)
# Result: {'first_name': 'Иван', 'middle_name': 'Иванович', 'last_name': 'Иванов'}

# Parse with explicit format
parsed_name = dict_helpers.parse_full_name(
    "John Smith", 
    name_format="first_last"
)
# Result: {'first_name': 'John', 'middle_name': '', 'last_name': 'Smith'}
```

## 8. Performance Considerations

1. **Caching**:
    - Dictionary loading is optimized through caching
    - Clear cache when memory usage is a concern
2. **Validation**:
    - Validation can impact performance for large dictionaries
    - Use appropriate validation based on use case
3. **Embedded vs External**:
    - Embedded dictionaries are faster to access but more limited
    - External dictionaries provide more flexibility but have I/O overhead
4. **Memory Usage**:
    - Monitor memory usage when loading large dictionaries
    - Consider batch processing for very large datasets

## 9. Extension Points

The dictionary subsystem is designed to be extensible:

1. **Adding New Dictionary Types**:
    - Add new modules to the `dictionaries` package
    - Implement clear API with proper caching
2. **Custom Validation Patterns**:
    - Add new validation patterns to `_DICTIONARY_PATTERNS`
    - Implement specialized validation functions if needed
3. **Additional Formats**:
    - Implement new loading functions for additional formats
    - Ensure proper integration with caching mechanism

By following these guidelines, the dictionary subsystem can be extended to support new data types, formats, and validation requirements while maintaining performance and reliability.