# pamola_core.fake_data.commons Package Documentation

## Overview

The `pamola_core.fake_data.commons` package provides the foundation for the PAMOLA.CORE fake data generation system. It contains pamola core abstractions, utility functions, and validation capabilities that enable the creation of realistic synthetic data for anonymization purposes. This package serves as the architectural backbone for the entire `fake_data` module, establishing patterns and interfaces that ensure consistency and extensibility across the system.

Key modules in this package include:
- `base.py`: Pamola Core abstract classes and interfaces
- `utils.py`: Utility functions for data manipulation and transformation
- `validators.py`: Validation functions for various data types

Together, these modules provide a comprehensive foundation that supports the generation of high-quality synthetic data while maintaining structural integrity and statistical properties of the original data.

## Package Architecture

The package follows a layered architecture that separates concerns and promotes reusability:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        pamola_core.fake_data.commons                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐ │
│  │       base.py       │  │      utils.py       │  │   validators.py  │ │
│  │                     │  │                     │  │                  │ │
│  │ • Abstract classes  │  │ • String utilities  │  │ • Data validation│ │
│  │ • Interfaces        │  │ • Data manipulation │  │ • Format checking│ │
│  │ • Core structures   │  │ • File operations   │  │ • Error reporting│ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └────────┬─────────┘ │
│             │                        │                       │          │
└─────────────┼────────────────────────┼───────────────────────┼──────────┘
              │                        │                       │
              ▼                        ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Higher-level Components                           │
│                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐ │
│  │     Generators      │  │      Mappers        │  │    Operations    │ │
│  └─────────────────────┘  └─────────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

The architecture is designed to:
- Establish clear separation of concerns
- Enable independent development of components
- Facilitate testing and maintenance
- Support extensibility for new data types and generation strategies

## Module Dependencies

The internal dependencies between modules in the package are designed to minimize coupling:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   base.py   │◄────┤   utils.py  │────►│validators.py│
└─────────────┘     └─────────────┘     └─────────────┘
      ▲                   ▲                   ▲
      │                   │                   │
      └───────────────────┴───────────────────┘
                          │
                          ▼
                  ┌─────────────────┐
                  │ External PAMOLA.CORE    │
                  │ Infrastructure  │
                  └─────────────────┘
```

- `base.py` has no dependencies on other modules in the package
- `utils.py` may use functionality from `base.py` but not `validators.py`
- `validators.py` may use functionality from both `base.py` and `utils.py`
- All modules may use external PAMOLA.CORE infrastructure components

This design allows for incremental development and testing, as well as the ability to use modules independently when needed.

## Key Components

### Abstract Base Classes (from base.py)

The package defines several critical abstract base classes that serve as the foundation for the entire fake data system:

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `BaseGenerator` | Template for data generators | `generate()`, `generate_like()`, `analyze_value()` |
| `BaseMapper` | Interface for mapping components | `map()`, `restore()`, `add_mapping()` |
| `BaseOperation` | Pamola Core interface for operations | `execute()` |
| `FieldOperation` | Extended interface for field operations | `process_batch()`, `handle_null_values()` |
| `MappingStore` | Repository for mappings | `add_mapping()`, `get_mapping()`, `restore_original()` |

### Utility Functions (from utils.py)

The package provides numerous utility functions for common tasks:

| Category | Key Functions |
|----------|--------------|
| String Processing | `normalize_string()`, `hash_value()`, `generate_deterministic_value()` |
| Language Utilities | `detect_language()`, `transliterate()`, `detect_gender_from_name()` |
| Data Generation | `create_username_from_name()`, `format_phone_number()` |
| File Operations | `load_dictionary()`, `save_dataframe()`, `ensure_dir()` |
| Progress Tracking | `get_progress_bar()` |

### Validators (from validators.py)

The package includes validation functions for various data types:

| Function | Purpose |
|----------|---------|
| `validate_name()` | Validates personal names with language-specific rules |
| `validate_email()` | Verifies email address format and extracts components |
| `validate_phone()` | Validates phone numbers using region-specific rules |
| `validate_id_number()` | Validates ID documents like passports, SSNs, etc. |
| `validate_format()` | Generic validation against regular expression patterns |

## Integration with PAMOLA.CORE Infrastructure

The `commons` package integrates with the broader PAMOLA.CORE system through several mechanisms:

1. **I/O Integration**: Uses `pamola_core.utils.io` for file operations and data loading
2. **Logging**: Leverages `pamola_core.utils.logging` for consistent log formatting
3. **Progress Tracking**: Utilizes `pamola_core.utils.progress` for operation monitoring
4. **Operations Framework**: Aligns with the PAMOLA.CORE operations registry and reporting system

This integration ensures that the fake data system operates seamlessly within the larger PAMOLA.CORE ecosystem while maintaining its modular architecture.

## Cross-Cutting Concerns

The package addresses several cross-cutting concerns that affect all modules:

### Error Handling

A consistent error handling strategy is employed throughout the package:

- Hierarchical exception classes derived from `FakeDataError`
- Detailed error messages for troubleshooting
- Consistent patterns for error propagation
- Balance between fail-fast and graceful degradation

### Internationalization

The package is designed to support multiple languages and regions:

- Language detection and classification
- Region-specific validation rules
- Transliteration between writing systems
- Dictionaries organized by language and region

### Performance

Performance considerations are addressed at the architectural level:

- Resource estimation capabilities
- Batch processing for large datasets
- Dictionary caching mechanisms
- Memory-efficient data handling

### Security

The package implements security measures for sensitive data:

- Value hashing for audit logs
- Secure dictionary handling
- Protection against data leakage
- Integration with PAMOLA.CORE encryption capabilities

## Usage Examples

### Creating a Complete Anonymization Process

```python
from pamola_core.fake_data.commons.base import FieldOperation, NullStrategy
from pamola_core.fake_data.commons.utils import load_dictionary, ensure_dir
from pamola_core.fake_data.commons.validators import validate_email
import pandas as pd

# 1. Load dictionaries
first_names = load_dictionary("data/dictionaries/first_names_en.csv")
domains = load_dictionary("data/dictionaries/email_domains.csv")

# 2. Create generators and mappers
# (Assuming these are defined in higher-level modules)
name_generator = NameGenerator(first_names, language="en")
name_mapper = OneToOneMapper(fallback_generator=name_generator)
email_generator = EmailGenerator(domains)
email_mapper = OneToOneMapper(fallback_generator=email_generator)

# 3. Define operations
name_operation = NameGenerationOperation(
    field_name="first_name",
    generator=name_generator,
    mapper=name_mapper,
    mode="REPLACE",
    null_strategy=NullStrategy.PRESERVE
)

email_operation = EmailGenerationOperation(
    field_name="email",
    generator=email_generator,
    mapper=email_mapper,
    mode="ENRICH",
    output_field_name="anonymized_email"
)


# 4. Process data
def anonymize_dataset(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Apply operations
    df = name_operation.execute(df, ensure_dir("task/name_op"), reporter)
    df = email_operation.execute(df, ensure_dir("task/email_op"), reporter)

    # Validate results
    valid_emails = df["anonymized_email"].apply(
        lambda x: validate_email(x)["valid"] if pd.notna(x) else True
    )

    if not valid_emails.all():
        logger.warning(f"Found {(~valid_emails).sum()} invalid emails")

    # Save results
    df.to_csv(output_path, index=False)

    return df
```

### Implementing Custom Generators and Validators

```python
from pamola_core.fake_data.commons.base import BaseGenerator
from pamola_core.fake_data.commons.utils import transliterate, hash_value
from pamola_core.fake_data.commons.validators import validate_format


# Custom generator for company names
class CompanyGenerator(BaseGenerator):
    def __init__(self, company_dict, suffix_dict):
        self.companies = company_dict
        self.suffixes = suffix_dict

    def generate(self, count, **params):

    # Implementation for generating multiple company names
    # ...

    def generate_like(self, original_value, **params):
        # Generate a name similar to the original
        properties = self.analyze_value(original_value)

        # Use properties to guide generation
        if properties.get("has_suffix"):
        # Generate with similar suffix
        # ...
        else:
        # Generate without suffix
        # ...

        return new_company_name

    def analyze_value(self, value):
        # Extract properties from the company name
        return {
            "length": len(value),
            "has_suffix": any(suffix in value for suffix in self.suffixes),
            "industry": self._detect_industry(value)
        }

    def _detect_industry(self, name):
# Custom logic to infer industry from name
# ...


# Custom validator for company registration numbers
def validate_company_reg_num(reg_num, region="US"):
    result = {
        "valid": False,
        "errors": [],
        "properties": {}
    }

    if not reg_num:
        result["errors"].append("Registration number is empty")
        return result

    # Region-specific validation
    if region == "US":
        # US EIN validation (9 digits)
        if validate_format(reg_num, r'^\d{2}-\d{7}$')["valid"]:
            result["valid"] = True
            result["properties"] = {
                "type": "EIN",
                "area": reg_num[:2]
            }
        else:
            result["errors"].append("Invalid EIN format (should be XX-XXXXXXX)")
    elif region == "UK":
    # UK Company Number validation (8 digits)
    # ...

    return result
```

## Extending the Package

The `commons` package is designed to be extended in various ways:

### Adding New Data Types

To add support for a new data type:

1. Create appropriate validators in `validators.py` or extend it in a higher-level module
2. Use existing utility functions from `utils.py` to support the new data type
3. Implement generators and mappers that extend the base classes from `base.py`
4. Add operations that use the new components

### Supporting New Languages or Regions

To add support for a new language or region:

1. Extend language detection and transliteration in `utils.py`
2. Add region-specific validation rules to validators
3. Create appropriate dictionaries for the language/region
4. Update any region-specific formatting functions

### Enhancing Performance

To optimize performance for specific use cases:

1. Implement specialized generators with optimized algorithms
2. Create custom mappers with more efficient storage strategies
3. Use batch processing and parallelization in operations
4. Implement memory-efficient handling for large datasets

## Best Practices

When working with the `commons` package:

1. **Follow Interface Contracts**: Adhere to the behavior defined in abstract base classes
2. **Validate Early**: Use validators to check data before processing
3. **Handle Edge Cases**: Consider empty values, NULL values, and unusual inputs
4. **Internationalize Properly**: Support multiple languages and regions consistently
5. **Document Extensions**: Provide clear documentation for any extensions or customizations
6. **Test Thoroughly**: Create comprehensive tests for all components

## Common Pitfalls

Watch out for these common issues:

1. **Forgetting NULL Handling**: Always specify a NULL strategy for operations
2. **Ignoring Resource Constraints**: Be mindful of memory usage and performance
3. **Breaking Transitivity**: Ensure mapping consistency in complex scenarios
4. **Language Assumptions**: Avoid making assumptions about character sets or formats
5. **Ignoring Error Handling**: Always check return values from validators and handle errors gracefully

## Conclusion

The `pamola_core.fake_data.commons` package provides a solid foundation for building a comprehensive fake data generation system. Its well-designed abstractions, utility functions, and validation capabilities enable the creation of high-quality synthetic data that maintains the structural integrity and statistical properties of the original data.

By leveraging the components in this package, developers can create sophisticated anonymization solutions that protect privacy while preserving the utility of datasets for analysis and testing. The extensible design allows for adaptation to new data types, languages, and regions, making it a versatile tool for a wide range of anonymization scenarios.

As the fake data system evolves, the `commons` package will continue to serve as its architectural backbone, providing the consistent interfaces and pamola core functionality that ensure the system's reliability, maintainability, and extensibility.