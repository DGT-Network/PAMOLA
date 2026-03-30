# RelationshipType Enumeration

**Module:** `pamola_core.common.enum.relationship_type`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

RelationshipType defines the cardinality relationships between tables in multi-table data operations. It controls how datasets are joined, aggregated, and related during privacy-preserving processing.

## Members

| Member | Value | Description |
|--------|-------|-------------|
| `AUTO` | `"auto"` | Automatically detect relationship type from data |
| `ONE_TO_ONE` | `"one-to-one"` | Each row in table A maps to exactly one row in table B |
| `ONE_TO_MANY` | `"one-to-many"` | Each row in table A maps to multiple rows in table B |

## Usage

### Basic Enumeration Access

```python
from pamola_core.common.enum.relationship_type import RelationshipType

# Access members
rel = RelationshipType.ONE_TO_MANY
print(rel.value)  # Output: "one-to-many"
print(rel.name)   # Output: "ONE_TO_MANY"
```

### Conditional Join Logic

```python
from pamola_core.common.enum.relationship_type import RelationshipType
import pandas as pd

def join_tables(left: pd.DataFrame, right: pd.DataFrame,
                rel_type: RelationshipType, on: str) -> pd.DataFrame:
    """Join tables based on relationship type."""
    if rel_type == RelationshipType.ONE_TO_ONE:
        return left.merge(right, on=on, how="inner")
    elif rel_type == RelationshipType.ONE_TO_MANY:
        return left.merge(right, on=on, how="left")
    else:  # AUTO
        # Detect based on data
        return detect_and_join(left, right, on)

# Usage
df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
df2 = pd.DataFrame({"id": [1, 1, 2, 3], "value": [10, 20, 30, 40]})
result = join_tables(df1, df2, RelationshipType.ONE_TO_MANY, on="id")
```

### Listing Available Relationships

```python
from pamola_core.common.enum.relationship_type import RelationshipType

types = [t.value for t in RelationshipType]
print(types)  # ["auto", "one-to-one", "one-to-many"]
```

## Member Descriptions

### AUTO
**Value:** `"auto"`

Automatically detects the relationship type by analyzing the data cardinality. The system examines key distributions and row counts to determine the actual relationship structure.

**Detection Logic:**
- Checks for duplicate keys in each table
- Compares cardinality ratios
- Infers relationship from data patterns

**Use cases:**
- Unknown data structures requiring automatic analysis
- Dynamic processing pipelines
- Exploratory data analysis

**Trade-offs:**
- Requires data scanning (may be slow for large datasets)
- Detection could fail on edge cases
- Generally preferred for robustness

### ONE_TO_ONE
**Value:** `"one-to-one"`

Each record in the left table corresponds to exactly one record in the right table. Both tables have unique identifiers for the join key.

**Example Structure:**
```
Table A (Users)       Table B (Profiles)
id | name            | user_id | email
1  | John            | 1       | john@example.com
2  | Jane            | 2       | jane@example.com
3  | Bob             | 3       | bob@example.com
```

**Characteristics:**
- Both tables have unique key values
- Join produces result with same row count as input
- No duplication of data

**Use cases:**
- Joining user master with user settings
- Combining person record with unique identifier
- Linking primary and secondary identifiers

**Privacy Consideration:**
- Data volume remains same after join
- No aggregation-based privacy amplification

### ONE_TO_MANY
**Value:** `"one-to-many"`

Each record in the left table may correspond to multiple records in the right table. The left table has unique keys, but the right table may have duplicates.

**Example Structure:**
```
Table A (Orders)      Table B (Order Items)
order_id | date       | order_id | product | qty
1        | 2024-01-01 | 1        | Widget  | 2
2        | 2024-01-02 | 1        | Gadget  | 1
3        | 2024-01-03 | 2        | Widget  | 3
                      | 3        | Tool    | 1
```

**Characteristics:**
- Left table has unique keys
- Right table may have duplicate keys
- Join produces result with more rows than left table
- May cause data duplication issues

**Use cases:**
- Customer with multiple orders
- Patient with multiple medical visits
- Company with multiple employees
- Person with multiple transactions

**Privacy Consideration:**
- Data duplication may increase identifiability
- Requires careful aggregation for privacy preservation

## Selection Guide

### Detect Relationship Type

```python
from pamola_core.common.enum.relationship_type import RelationshipType

def detect_relationship(left: pd.DataFrame, right: pd.DataFrame,
                       join_key: str) -> RelationshipType:
    """Detect relationship type from data."""
    left_uniq = left[join_key].nunique()
    right_uniq = right[join_key].nunique()
    left_count = len(left)
    right_count = len(right)

    # Check for one-to-one
    if left_uniq == left_count and right_uniq == right_count:
        if left_uniq == right_uniq:
            return RelationshipType.ONE_TO_ONE

    # Check for one-to-many
    if left_uniq == left_count and right_uniq < right_count:
        return RelationshipType.ONE_TO_MANY

    # Ambiguous or many-to-many
    return RelationshipType.AUTO
```

### By Use Case

**Simple Master-Detail**
- `ONE_TO_MANY` for typical business relationships

**Identity Linking**
- `ONE_TO_ONE` for unique identifier matching

**Unknown Structure**
- `AUTO` for automatic detection

## Related Components

- **Join Operations:** Controls how tables are combined
- **Aggregation Logic:** Impacts data deduplication strategies
- **Privacy Preservation:** Influences privacy leakage assessment

## Common Patterns

### Multi-Table Processing with Relationship Awareness

```python
from pamola_core.common.enum.relationship_type import RelationshipType

def process_related_tables(tables: dict, relationships: dict) -> pd.DataFrame:
    """Process multiple tables with specified relationships."""
    result = tables["primary"]

    for table_name, rel_type in relationships.items():
        secondary = tables[table_name]
        if rel_type == RelationshipType.ONE_TO_ONE:
            result = result.merge(secondary, on="id", how="inner")
        elif rel_type == RelationshipType.ONE_TO_MANY:
            result = result.merge(secondary, on="id", how="left")

    return result
```

### Privacy-Aware Aggregation

```python
from pamola_core.common.enum.relationship_type import RelationshipType
import pandas as pd

def aggregate_with_relationship(df: pd.DataFrame,
                               rel_type: RelationshipType) -> pd.DataFrame:
    """Apply appropriate aggregation based on relationship."""
    if rel_type == RelationshipType.ONE_TO_MANY:
        # Sum counts or aggregate for privacy
        return df.groupby("id").agg({"value": "sum"})
    else:
        # No aggregation needed for one-to-one
        return df
```

## Best Practices

1. **Explicit Over Automatic**
   ```python
   # Good - explicit relationship declaration
   rel = RelationshipType.ONE_TO_MANY

   # Acceptable - auto-detection when uncertain
   rel = RelationshipType.AUTO
   ```

2. **Validate Before Processing**
   ```python
   detected = detect_relationship(df1, df2, "id")
   if detected != expected:
       raise ValueError(f"Expected {expected}, got {detected}")
   ```

3. **Document Relationships**
   ```python
   # Good - explains data model
   relationships = {
       "orders": RelationshipType.ONE_TO_MANY,  # Customer has multiple orders
       "profile": RelationshipType.ONE_TO_ONE   # User has one profile
   }
   ```

4. **Consider Privacy Impact**
   ```python
   # One-to-many may increase identifiability
   if rel_type == RelationshipType.ONE_TO_MANY:
       apply_additional_privacy_protection()
   ```

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- Three relationship types: AUTO, ONE_TO_ONE, ONE_TO_MANY
- Detection and usage patterns
