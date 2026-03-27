# Field Definitions Module Documentation

**Module:** `pamola_core.configs.field_definitions`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Enumerations](#enumerations)
3. [Field Metadata Structure](#field-metadata-structure)
4. [Table Organization](#table-organization)
5. [Field Categories](#field-categories)
6. [Anonymization Strategies](#anonymization-strategies)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Technical Summary](#technical-summary)

## Overview

The `field_definitions.py` module provides comprehensive metadata for data fields in the PAMOLA ecosystem, particularly focused on resume/employment datasets. It defines field types, privacy categories, anonymization strategies, and profiling tasks required for proper data handling and anonymization.

### Purpose

This module:
- Defines metadata for all dataset fields
- Specifies privacy risk levels for each field
- Prescribes anonymization strategies appropriate to field sensitivity
- Organizes fields into logical tables
- Defines profiling tasks for data quality assessment
- Provides enumerations for type safety

### Module Structure

```
pamola_core/configs/field_definitions.py
├── FieldType enum                # Data type enumeration
├── PrivacyCategory enum          # Privacy level enumeration
├── AnonymizationStrategy enum    # Anonymization method enumeration
├── ProfilingTask enum            # Data profiling tasks
├── TABLES dict                   # Field organization by table
└── FIELD_DEFINITIONS dict        # Comprehensive field metadata
```

## Enumerations

### FieldType Enum

Classifies the data type of each field.

```python
class FieldType(str, Enum):
    """Enumeration of field data types."""
    SHORT_TEXT = "short_text"   # Text < 100 characters
    LONG_TEXT = "long_text"     # Text > 100 characters, paragraphs
    DOUBLE = "double"           # Floating-point numbers
    LONG = "long"               # Integer numbers
    DATE = "date"               # Date values (may be stored as text)
```

| Type | Use Case | Example Fields |
|------|----------|-----------------|
| `SHORT_TEXT` | Single-value text fields | name, email, job_title |
| `LONG_TEXT` | Multi-value or paragraph text | descriptions, skills, experience |
| `DOUBLE` | Decimal numeric values | salary, commission, rate |
| `LONG` | Integer numeric values | age, count, ID |
| `DATE` | Date values (any storage format) | birth_date, hire_date, end_date |

### PrivacyCategory Enum

Classifies privacy risk level of each field.

```python
class PrivacyCategory(str, Enum):
    """Enumeration of privacy categories for fields."""
    DIRECT_IDENTIFIER = "Direct Identifier"
    INDIRECT_IDENTIFIER = "Indirect Identifier"
    QUASI_IDENTIFIER = "Quasi-Identifier"
    SENSITIVE_ATTRIBUTE = "Sensitive Attribute"
    NON_SENSITIVE = "Non-Sensitive"
```

| Category | Definition | Risk Level | Examples |
|----------|-----------|-----------|----------|
| **Direct Identifier** | Uniquely identifies individual | Critical | name, email, phone, SSN |
| **Indirect Identifier** | Identifies when combined with other data | High | resume_id, employee_id |
| **Quasi-Identifier** | May contribute to identification in combination | Medium | age, gender, location |
| **Sensitive Attribute** | Private information requiring protection | Medium-High | salary, medical history, religion |
| **Non-Sensitive** | Poses minimal privacy risk | Low | job_title, education_level |

**Privacy Strategy by Category:**
- Direct Identifier → SUPPRESSION or PSEUDONYMIZATION
- Indirect Identifier → PSEUDONYMIZATION
- Quasi-Identifier → GENERALIZATION or KEEP_ORIGINAL
- Sensitive Attribute → NOISE_ADDITION or SUPPRESSION
- Non-Sensitive → KEEP_ORIGINAL or GENERALIZATION

### AnonymizationStrategy Enum

Specifies how each field should be anonymized.

```python
class AnonymizationStrategy(str, Enum):
    """Enumeration of anonymization strategies."""
    PSEUDONYMIZATION = "PSEUDONYMIZATION"
    GENERALIZATION = "GENERALIZATION"
    SUPPRESSION = "SUPPRESSION"
    KEEP_ORIGINAL = "KEEP_ORIGINAL"
    NOISE_ADDITION = "NOISE_ADDITION"
    NER_LLM_RECONSTRUCTION = "NER/LLM RECONSTRUCTION"
```

| Strategy | Description | When to Use | Output |
|----------|-------------|------------|--------|
| **PSEUDONYMIZATION** | Replace with reversible synthetic identifier | Indirect identifiers, names | Consistent masked value per original |
| **GENERALIZATION** | Replace with less specific value | Quasi-identifiers, locations | Age groups, region instead of city |
| **SUPPRESSION** | Remove data completely | Direct identifiers, sensitive | NULL or deleted |
| **KEEP_ORIGINAL** | Retain unchanged | Non-sensitive fields | Original value |
| **NOISE_ADDITION** | Add random variation | Numeric sensitive data | Original + random offset |
| **NER/LLM_RECONSTRUCTION** | NLP-based entity replacement | Text with named entities | Replaced entity names |

### ProfilingTask Enum

Defines data profiling operations for quality assessment.

```python
class ProfilingTask(str, Enum):
    """Enumeration of profiling tasks."""
    COMPLETENESS = "completeness"           # Missing value check
    UNIQUENESS = "uniqueness"               # Count distinct values
    DISTRIBUTION = "distribution"           # Value distribution analysis
    FREQUENCY = "frequency"                 # Frequency counts
    OUTLIERS = "outliers"                   # Identify outlier values
    RARE_VALUES = "rare_values"             # Find infrequent values
    FORMAT_VALIDATION = "format_validation" # Format compliance check
    CORRELATION = "correlation"             # Cross-field relationships
    TEXT_ANALYSIS = "text_analysis"         # Text content analysis
    LENGTH_ANALYSIS = "length_analysis"     # Text length distribution
    PATTERN_DETECTION = "pattern_detection" # Pattern discovery
```

| Task | Purpose | Output |
|------|---------|--------|
| `COMPLETENESS` | Check missing/null rates | % complete, # nulls |
| `UNIQUENESS` | Count distinct values | # distinct, cardinality |
| `DISTRIBUTION` | Statistical distribution | histogram, percentiles |
| `FREQUENCY` | Most/least common values | frequency table |
| `OUTLIERS` | Identify anomalous values | outlier values, % outliers |
| `RARE_VALUES` | Find infrequent values | rare value list, counts |
| `FORMAT_VALIDATION` | Validate format patterns | # valid, # invalid, invalid samples |
| `CORRELATION` | Cross-field relationships | correlation coefficients |
| `TEXT_ANALYSIS` | Text content analysis | word frequency, entities |
| `LENGTH_ANALYSIS` | Text length distribution | length histogram, statistics |
| `PATTERN_DETECTION` | Pattern discovery | identified patterns, % matching |

## Field Metadata Structure

Each field in `FIELD_DEFINITIONS` contains:

```python
"field_name": {
    "id": int,                              # Unique field ID
    "type": FieldType,                      # Field data type
    "category": PrivacyCategory,            # Privacy level
    "strategy": AnonymizationStrategy,      # Anonymization method
    "description": str,                     # Human-readable description
    "profiling_tasks": List[ProfilingTask], # Applicable profiling tasks
    "table": str                            # Parent table name
}
```

### Example Field Definition

```python
"salary": {
    "id": 9,
    "type": FieldType.DOUBLE,
    "category": PrivacyCategory.SENSITIVE_ATTRIBUTE,
    "strategy": AnonymizationStrategy.NOISE_ADDITION,
    "description": "Зарплата (много нулей), добавление случайного шума",
    "profiling_tasks": [
        ProfilingTask.COMPLETENESS,
        ProfilingTask.DISTRIBUTION,
        ProfilingTask.OUTLIERS,
        ProfilingTask.CORRELATION
    ],
    "table": "RESUME_DETAILS"
}
```

## Table Organization

Fields are organized into logical tables:

```python
TABLES = {
    "IDENTIFICATION": [...],         # Personal identifiers
    "RESUME_DETAILS": [...],         # Employment details
    "CONTACTS": [...],               # Contact information
    "SPECIALIZATION": [...],         # Skills and certifications
    "ATTESTATION": [...],            # Attestation education
    "PRIMARY_EDU": [...],            # Primary education
    "ADDITIONAL_EDU": [...],         # Additional education
    "ELEMENTARY_EDU": [...],         # Elementary education
    "EXPERIENCE": [...]              # Work experience
}
```

### Table Details

| Table | Purpose | Fields | Privacy Level |
|-------|---------|--------|--------------|
| **IDENTIFICATION** | Personal identity | resume_id, first_name, last_name, gender, birth_day | HIGH |
| **RESUME_DETAILS** | Job preferences & attributes | post, education_level, salary, area_name | MEDIUM |
| **CONTACTS** | Communication channels | email, phones | CRITICAL |
| **SPECIALIZATION** | Skills & expertise | key_skill_names, specialization_names | MEDIUM |
| **ATTESTATION** | Certifications | attestation_education_* | MEDIUM |
| **PRIMARY_EDU** | Formal education | primary_education_* | LOW-MEDIUM |
| **ADDITIONAL_EDU** | Supplementary training | additional_education_* | LOW-MEDIUM |
| **ELEMENTARY_EDU** | Basic education | elementary_education_* | LOW |
| **EXPERIENCE** | Work history | experience_* | MEDIUM |

## Field Categories

### Direct Identifiers (CRITICAL Risk)

Directly and uniquely identify individuals. Require immediate action.

**Fields:**
- `first_name`, `last_name`: Person's name
- `email`: Email address
- `home_phone`, `work_phone`, `cell_phone`: Phone numbers
- `file_as`: Name as filed (variant)

**Strategy:** SUPPRESSION or PSEUDONYMIZATION
**Rationale:** Direct matching possible without additional context

### Indirect Identifiers (HIGH Risk)

Identify when combined with external data or public records.

**Fields:**
- `resume_id`: Unique resume identifier
- `UID`: User ID within system
- `middle_name`: Patronymic or middle name

**Strategy:** PSEUDONYMIZATION
**Rationale:** Reversible mapping required to link records; direct deletion breaks relational integrity

### Quasi-Identifiers (MEDIUM Risk)

May facilitate re-identification when combined with publicly available data.

**Fields:**
- `birth_day`: Date of birth
- `gender`: Gender
- `area_name`: Geographic region/city
- `metro_station_name`: Specific location
- `post`: Job title

**Strategy:** GENERALIZATION or KEEP_ORIGINAL
**Rationale:** Less sensitive individually but risky in combination; generalization reduces specificity

### Sensitive Attributes (MEDIUM-HIGH Risk)

Private information that individuals typically wish to keep confidential.

**Fields:**
- `salary`: Compensation information
- `business_trip_readiness`: Work preferences
- `driver_license_types`: Personal capabilities

**Strategy:** NOISE_ADDITION, GENERALIZATION, or SUPPRESSION
**Rationale:** Disclosure harmful even without identity link; noise preserves distributions

### Non-Sensitive (LOW Risk)

Information poses minimal privacy risk on its own.

**Fields:**
- `education_level`: Formal education type
- `relocation`: Job mobility preference
- `work_schedules`: Work schedule type
- `employments`: Employment type
- `has_vehicle`: Vehicle ownership
- `key_skill_names`: Skills and expertise
- `specialization_names`: Specializations
- Education-related fields

**Strategy:** KEEP_ORIGINAL or GENERALIZATION
**Rationale:** Low sensitivity; can be retained or generalized for utility

## Anonymization Strategies

### PSEUDONYMIZATION Strategy

**Purpose:** Reversible replacement with synthetic identifier.

**Implementation:**
- Map original value to consistent pseudonym
- Maintain one-to-one relationship (same original → same pseudonym)
- Preserve linkage between records

**Use Cases:**
- Names (first_name, last_name, middle_name)
- IDs (resume_id, UID)

**Example:**
```
Original: "John Smith", "john.smith@email.com"
Anonymized: "Pseudo_ID_001", "Pseudo_ID_001"
(Both map to same individual)
```

### GENERALIZATION Strategy

**Purpose:** Replace specific values with less specific categories.

**Implementation:**
- Group similar values into categories
- Suppress granular details
- Reduce uniqueness

**Use Cases:**
- Birth date → Age group (20-30, 30-40, etc.)
- City name → Region name
- Job title → Job category
- Skills → Skill category

**Example:**
```
Original: "2001-03-15" (age 22)
Anonymized: "20-30" (age group)

Original: "Moscow"
Anonymized: "Western Russia"
```

### SUPPRESSION Strategy

**Purpose:** Remove data completely.

**Implementation:**
- Delete field value
- Replace with NULL or empty string
- Lose all field information

**Use Cases:**
- Email addresses (direct identifiers)
- Phone numbers (direct identifiers)

**Example:**
```
Original: "john@email.com"
Anonymized: NULL or ""
```

### KEEP_ORIGINAL Strategy

**Purpose:** Retain data without modification.

**Implementation:**
- No transformation applied
- Data disclosed as-is

**Use Cases:**
- Non-sensitive fields (education_level, relocation)
- Publicly available information
- Required for analysis without privacy risk

**Example:**
```
Original: "Bachelor"
Anonymized: "Bachelor" (unchanged)
```

### NOISE_ADDITION Strategy

**Purpose:** Add random variation to numeric data.

**Implementation:**
- Add Gaussian or Laplacian noise to values
- Preserve statistical distribution
- Mask exact individual values

**Use Cases:**
- Salary information (sensitive numeric)
- Age (quasi-identifier numeric)
- Quantities

**Example:**
```
Original: 50000
Anonymized: 50000 + random_noise
(e.g., 49850, 50150, 50025)
```

### NER/LLM_RECONSTRUCTION Strategy

**Purpose:** NLP-based entity identification and replacement.

**Implementation:**
- Use Named Entity Recognition to identify named entities
- Replace entity references with generic categories or pseudonyms
- Preserve text structure

**Use Cases:**
- Organization names (additional_education_organizations)
- Education program names (additional_education_names)
- Company names (experience_organizations)

**Example:**
```
Original: "XYZ Corporation", "Harvard University"
Anonymized: "Org_001", "Org_002" or "Company", "University"
```

## Usage Examples

### Example 1: Get Field Definition

```python
from pamola_core.configs.field_definitions import FIELD_DEFINITIONS

# Get field metadata
salary_def = FIELD_DEFINITIONS['salary']

print(f"Type: {salary_def['type']}")
print(f"Category: {salary_def['category']}")
print(f"Strategy: {salary_def['strategy']}")

# Output:
# Type: double
# Category: Sensitive Attribute
# Strategy: NOISE_ADDITION
```

### Example 2: Find All Direct Identifiers

```python
from pamola_core.configs.field_definitions import (
    FIELD_DEFINITIONS,
    PrivacyCategory
)

# Get all direct identifiers
direct_ids = {
    name: defn for name, defn in FIELD_DEFINITIONS.items()
    if defn['category'] == PrivacyCategory.DIRECT_IDENTIFIER
}

print("Direct Identifiers to suppress:")
for field in direct_ids:
    print(f"  - {field}")

# Output:
# Direct Identifiers to suppress:
#   - first_name
#   - last_name
#   - email
#   - home_phone
#   - work_phone
#   - cell_phone
```

### Example 3: Find Fields Requiring Specific Strategy

```python
from pamola_core.configs.field_definitions import (
    FIELD_DEFINITIONS,
    AnonymizationStrategy
)

# Get fields requiring GENERALIZATION
generalization_fields = {
    name: defn for name, defn in FIELD_DEFINITIONS.items()
    if defn['strategy'] == AnonymizationStrategy.GENERALIZATION
}

print(f"Fields requiring generalization: {len(generalization_fields)}")
for field in generalization_fields:
    print(f"  - {field}")
```

### Example 4: Get Profiling Tasks for Field

```python
from pamola_core.configs.field_definitions import FIELD_DEFINITIONS

# Get profiling requirements
field_def = FIELD_DEFINITIONS['salary']

print(f"Profiling tasks for 'salary':")
for task in field_def['profiling_tasks']:
    print(f"  - {task.value}")

# Output:
# Profiling tasks for 'salary':
#   - completeness
#   - distribution
#   - outliers
#   - correlation
```

### Example 5: Fields by Privacy Category

```python
from pamola_core.configs.field_definitions import (
    FIELD_DEFINITIONS,
    PrivacyCategory
)

# Count fields by category
categories = {}
for name, defn in FIELD_DEFINITIONS.items():
    cat = defn['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(name)

for category, fields in categories.items():
    print(f"{category}: {len(fields)} fields")

# Output example:
# Direct Identifier: 6 fields
# Indirect Identifier: 2 fields
# Quasi-Identifier: 5 fields
# Sensitive Attribute: 3 fields
# Non-Sensitive: 15 fields
```

### Example 6: Fields by Type

```python
from pamola_core.configs.field_definitions import (
    FIELD_DEFINITIONS,
    FieldType
)

# Get all text fields
text_fields = {
    name: defn for name, defn in FIELD_DEFINITIONS.items()
    if defn['type'] in [FieldType.SHORT_TEXT, FieldType.LONG_TEXT]
}

print(f"Text fields: {len(text_fields)}")

# Get numeric fields
numeric_fields = {
    name: defn for name, defn in FIELD_DEFINITIONS.items()
    if defn['type'] in [FieldType.DOUBLE, FieldType.LONG]
}

print(f"Numeric fields: {len(numeric_fields)}")
```

### Example 7: Fields in Specific Table

```python
from pamola_core.configs.field_definitions import (
    TABLES,
    FIELD_DEFINITIONS
)

# Get all fields in CONTACTS table
contact_fields = TABLES['CONTACTS']

for field_name in contact_fields:
    defn = FIELD_DEFINITIONS[field_name]
    print(f"{field_name}: {defn['strategy']}")

# Output:
# email: SUPPRESSION
# home_phone: SUPPRESSION
# work_phone: SUPPRESSION
# cell_phone: SUPPRESSION
```

## Best Practices

### 1. **Use Type-Safe Enumerations**

```python
from pamola_core.configs.field_definitions import (
    FieldType,
    PrivacyCategory,
    AnonymizationStrategy
)

# Good: Use enum values for comparison
if defn['category'] == PrivacyCategory.DIRECT_IDENTIFIER:
    apply_suppression()

# Avoid: String comparison (error-prone)
if defn['category'] == "Direct Identifier":  # OK but less safe
    apply_suppression()
```

### 2. **Validate Field References**

```python
from pamola_core.configs.field_definitions import FIELD_DEFINITIONS

# Check field exists before access
field_name = "user_input_field"

if field_name in FIELD_DEFINITIONS:
    defn = FIELD_DEFINITIONS[field_name]
else:
    raise ValueError(f"Unknown field: {field_name}")
```

### 3. **Respect Profiling Tasks**

```python
from pamola_core.configs.field_definitions import (
    FIELD_DEFINITIONS,
    ProfilingTask
)

# Profile only required tasks
defn = FIELD_DEFINITIONS['salary']

if ProfilingTask.OUTLIERS in defn['profiling_tasks']:
    # Perform outlier detection
    detect_outliers(data['salary'])
```

### 4. **Maintain Privacy Classification Discipline**

Document decisions when assigning privacy categories:

```python
# When adding new fields, establish:
# 1. Data type (from FieldType enum)
# 2. Privacy risk (from PrivacyCategory)
# 3. Appropriate strategy (from AnonymizationStrategy)
# 4. Profiling needs (from ProfilingTask list)

new_field = {
    "id": 999,
    "type": FieldType.LONG_TEXT,           # Rationale: multi-value text
    "category": PrivacyCategory.NON_SENSITIVE,  # Rationale: public job titles
    "strategy": AnonymizationStrategy.KEEP_ORIGINAL,  # Rationale: no privacy risk
    "description": "Job titles (non-sensitive)",
    "profiling_tasks": [ProfilingTask.FREQUENCY],  # Rationale: frequency analysis useful
    "table": "RESUME_DETAILS"
}
```

### 5. **Use Table Organization**

```python
from pamola_core.configs.field_definitions import TABLES

# Batch operations by table
for table_name, field_names in TABLES.items():
    print(f"Processing table: {table_name}")
    # Load table-specific configuration
    # Perform table-level operations
```

## Technical Summary

The `field_definitions.py` module provides comprehensive data governance metadata:

- **Type Safety**: Enum-based definitions prevent invalid values
- **Privacy-First**: Clear categorization guides anonymization strategy
- **Comprehensive Coverage**: Metadata for 50+ fields with profiling requirements
- **Table Organization**: Logical grouping supports batch operations
- **Strategy Guidance**: Explicit recommendations aligned with privacy risk
- **Extensibility**: Simple structure enables adding new fields/categories
- **Non-Intrusive**: Definitions are declarative; no operational side effects

The module serves as the single source of truth for field-level privacy requirements and anonymization strategies in PAMOLA.CORE.
