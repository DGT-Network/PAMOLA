# PAMOLA.CORE NLP Category Matching Module Documentation

**Module:** `pamola_core.utils.nlp.category_matching`  
**Version:** 1.0.0  
**Last Updated:** December 2024  
**Status:** Production Ready

## 1. Overview

### 1.1 Purpose

The Category Matching module provides sophisticated functionality for matching text to predefined categories within hierarchical dictionaries. It implements advanced conflict resolution strategies, caching mechanisms, and support for multilingual category definitions, making it an essential component for classification tasks in the PAMOLA.CORE framework.

### 1.2 Key Features

- **Hierarchical Category Support**: Multi-level category hierarchies with parent-child relationships
- **Flexible Matching Strategies**: Multiple conflict resolution approaches for ambiguous matches
- **Performance Optimization**: Built-in caching for dictionaries and matching results
- **Batch Processing**: Parallel processing support for large text collections
- **Domain Awareness**: Category organization by business domains
- **Confidence Scoring**: Match quality assessment with confidence metrics
- **Fallback Handling**: Intelligent handling of unmatched texts

### 1.3 Integration Points

```
pamola_core.utils.nlp.category_matching
├── Depends On:
│   ├── pamola_core.utils.nlp.base (ConfigurationError, batch_process)
│   └── pamola_core.utils.nlp.cache (caching functionality)
└── Used By:
    ├── Data profiling operations
    ├── Text classification tasks
    └── Entity categorization pipelines
```

## 2. Architecture

### 2.1 Core Classes

#### CategoryDictionary

The main class that manages category dictionaries and provides matching functionality.

```python
class CategoryDictionary:
    """
    Manages category dictionaries with support for hierarchical structures,
    efficient matching algorithms, and multiple resolution strategies.
    """
```

**Key Responsibilities:**
- Dictionary loading and management
- Text-to-category matching
- Hierarchy analysis
- Conflict resolution
- Performance optimization through caching

### 2.2 Dictionary Structure

The module supports two dictionary formats:

#### Hierarchical Format (Recommended)
```json
{
    "categories_hierarchy": {
        "Software Engineer": {
            "keywords": ["software engineer", "developer", "programmer"],
            "level": 2,
            "domain": "Technology",
            "alias": "software_engineer",
            "seniority": "Any",
            "children": ["Senior Software Engineer", "Junior Developer"]
        },
        "Manager": {
            "keywords": ["manager", "lead", "supervisor"],
            "level": 1,
            "domain": "Management",
            "alias": "manager"
        }
    },
    "Unclassified": {
        "keywords": [],
        "level": 0,
        "domain": "General"
    }
}
```

#### Flat Format (Legacy)
```json
{
    "Software Engineer": {
        "keywords": ["software engineer", "developer"],
        "level": 2,
        "domain": "Technology"
    }
}
```

### 2.3 Matching Algorithm

The matching process follows these steps:

1. **Text Normalization**: Convert to lowercase, strip whitespace
2. **Keyword Scanning**: Check all category keywords against text
3. **Score Calculation**: Compute match scores based on keyword coverage
4. **Conflict Resolution**: Apply strategy when multiple matches found
5. **Confidence Assessment**: Evaluate match quality
6. **Result Generation**: Return category, score, and alternatives

## 3. API Reference

### 3.1 CategoryDictionary Class

#### Constructor

```python
CategoryDictionary(
    dictionary_data: Dict[str, Any] = None,
    hierarchy_data: Dict[str, Any] = None
)
```

**Parameters:**
- `dictionary_data`: Flattened category definitions
- `hierarchy_data`: Hierarchical category structure

#### Class Methods

##### from_file

```python
@classmethod
@cache_function(ttl=3600, cache_type='file')
def from_file(cls, file_path: Union[str, Path]) -> 'CategoryDictionary'
```

Load a category dictionary from a JSON file with caching support.

**Parameters:**
- `file_path`: Path to the dictionary JSON file

**Returns:**
- `CategoryDictionary`: Loaded and cached dictionary instance

**Example:**
```python
dict_path = Path("configs/dictionaries/job_categories.json")
category_dict = CategoryDictionary.from_file(dict_path)
```

#### Instance Methods

##### get_best_match

```python
def get_best_match(
    self, 
    text: str, 
    strategy: str = "specific_first"
) -> Tuple[Optional[str], float, List[str]]
```

Find the best matching category for given text.

**Parameters:**
- `text`: Text to categorize
- `strategy`: Conflict resolution strategy
  - `"specific_first"`: Prefer more specific categories (higher level)
  - `"domain_prefer"`: Prioritize domain-specific matches
  - `"alias_only"`: Use only score for resolution
  - `"user_override"`: Support external override mappings

**Returns:**
- `Tuple[Optional[str], float, List[str]]`:
  - Matched category name (or None)
  - Confidence score (0.0 to 1.0)
  - List of alternative categories (conflicts)

**Example:**
```python
category, score, conflicts = category_dict.get_best_match(
    "Senior Software Engineer", 
    strategy="specific_first"
)
# Returns: ("Senior Software Engineer", 0.85, ["Software Engineer"])
```

##### get_category_info

```python
def get_category_info(self, category: str) -> Dict[str, Any]
```

Retrieve detailed information about a specific category.

**Parameters:**
- `category`: Category name

**Returns:**
- `Dict[str, Any]`: Category metadata including keywords, level, domain

##### analyze_hierarchy

```python
def analyze_hierarchy(self) -> Dict[str, Any]
```

Analyze the category hierarchy structure.

**Returns:**
- `Dict[str, Any]`: Analysis results including:
  - `total_categories`: Total category count
  - `level_counts`: Categories per hierarchy level
  - `domain_counts`: Categories per domain
  - `categories_by_level`: Grouped by level
  - `categories_by_domain`: Grouped by domain

##### get_fallback_category

```python
def get_fallback_category(
    self, 
    confidence_threshold: float = 0.5
) -> Optional[str]
```

Get the default category for low-confidence matches.

**Parameters:**
- `confidence_threshold`: Minimum acceptable confidence

**Returns:**
- `Optional[str]`: Fallback category name or None

### 3.2 Module Functions

#### get_best_match

```python
@cache_function(ttl=3600, cache_type='memory')
def get_best_match(
    text: str,
    dictionary: Dict[str, Dict[str, Any]],
    match_strategy: str = "specific_first"
) -> Tuple[Optional[str], float, List[str]]
```

Standalone function for one-off matching without creating a CategoryDictionary instance.

#### batch_match_categories

```python
def batch_match_categories(
    texts: List[str],
    dictionary_path: Union[str, Path],
    match_strategy: str = "specific_first",
    processes: Optional[int] = None
) -> List[Tuple[Optional[str], float, List[str]]]
```

Process multiple texts in parallel for improved performance.

**Parameters:**
- `texts`: List of texts to categorize
- `dictionary_path`: Path to dictionary file
- `match_strategy`: Resolution strategy
- `processes`: Number of parallel processes (None for auto)

**Example:**
```python
job_titles = ["Software Engineer", "Project Manager", "Data Analyst"]
results = batch_match_categories(
    job_titles,
    "configs/dictionaries/jobs.json",
    processes=4
)
```

## 4. Dictionary Configuration

### 4.1 Category Properties

Each category can have the following properties:

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| keywords | List[str] | Matching keywords/phrases | Yes |
| level | int | Hierarchy level (0=root) | No |
| domain | str | Business domain | No |
| alias | str | Programmatic identifier | No |
| seniority | str | Seniority level | No |
| language | List[str] | Supported languages | No |
| children | List[str] | Child category names | No |

### 4.2 Best Practices

1. **Keyword Selection**:
   - Use complete phrases for accuracy
   - Include common variations and abbreviations
   - Order keywords by specificity

2. **Hierarchy Design**:
   - Start with broad categories at level 0
   - Increase specificity with each level
   - Maintain consistent depth across domains

3. **Domain Organization**:
   - Group related categories by domain
   - Use consistent domain naming
   - Consider cross-domain categories carefully

## 5. Usage Examples

### 5.1 Basic Category Matching

```python
from pathlib import Path
from pamola_core.utils.nlp.category_matching import CategoryDictionary

# Load dictionary
dict_path = Path("configs/dictionaries/job_titles.json")
category_dict = CategoryDictionary.from_file(dict_path)

# Match a single text
text = "Senior Data Scientist with ML expertise"
category, score, conflicts = category_dict.get_best_match(text)

if category:
    print(f"Category: {category}")
    print(f"Confidence: {score:.2f}")
    if conflicts:
        print(f"Alternatives: {', '.join(conflicts)}")
else:
    # Use fallback
    fallback = category_dict.get_fallback_category()
    print(f"No match found, using: {fallback}")
```

### 5.2 Batch Processing with Strategy Selection

```python
from pamola_core.utils.nlp.category_matching import batch_match_categories

# List of job titles to categorize
job_titles = [
    "Senior Software Engineer",
    "Marketing Manager",
    "Data Analyst",
    "DevOps Engineer",
    "Unknown Position XYZ"
]

# Process with domain preference strategy
results = batch_match_categories(
    job_titles,
    "configs/dictionaries/jobs.json",
    match_strategy="domain_prefer",
    processes=4
)

# Analyze results
for title, (category, score, conflicts) in zip(job_titles, results):
    print(f"\nTitle: {title}")
    print(f"  Category: {category or 'Unmatched'}")
    print(f"  Score: {score:.2f}")
    if conflicts:
        print(f"  Also considered: {conflicts}")
```

### 5.3 Dictionary Analysis

```python
# Analyze dictionary structure
category_dict = CategoryDictionary.from_file("configs/dictionaries/jobs.json")
analysis = category_dict.analyze_hierarchy()

print(f"Total categories: {analysis['total_categories']}")
print("\nCategories by level:")
for level, count in analysis['level_counts'].items():
    print(f"  Level {level}: {count} categories")

print("\nCategories by domain:")
for domain, categories in analysis['categories_by_domain'].items():
    print(f"  {domain}: {len(categories)} categories")
    print(f"    Examples: {', '.join(categories[:3])}")
```

### 5.4 Custom Resolution Strategy

```python
class CustomCategoryMatcher:
    def __init__(self, dictionary_path, user_overrides=None):
        self.category_dict = CategoryDictionary.from_file(dictionary_path)
        self.user_overrides = user_overrides or {}
    
    def match_with_override(self, text):
        # Check user overrides first
        if text.lower() in self.user_overrides:
            return self.user_overrides[text.lower()], 1.0, []
        
        # Fall back to dictionary matching
        return self.category_dict.get_best_match(
            text, 
            strategy="specific_first"
        )

# Usage
matcher = CustomCategoryMatcher(
    "configs/dictionaries/jobs.json",
    user_overrides={
        "cto": "Chief Technology Officer",
        "ceo": "Chief Executive Officer"
    }
)

category, score, _ = matcher.match_with_override("CTO")
print(f"Matched: {category} (score: {score})")
```

## 6. Performance Considerations

### 6.1 Caching Strategy

The module implements multi-level caching:

1. **File Cache**: Dictionary files cached for 1 hour
2. **Memory Cache**: Match results cached in memory
3. **Batch Optimization**: Parallel processing for multiple texts

### 6.2 Performance Tips

1. **Dictionary Size**: Keep dictionaries focused and domain-specific
2. **Keyword Count**: Balance coverage with performance (5-10 keywords per category)
3. **Batch Processing**: Use `batch_match_categories` for multiple texts
4. **Cache Warming**: Pre-load frequently used dictionaries

### 6.3 Benchmarks

Typical performance on standard hardware:

| Operation | Items | Time | Throughput |
|-----------|-------|------|------------|
| Single match | 1 | ~0.1ms | 10,000/sec |
| Batch match | 1,000 | ~50ms | 20,000/sec |
| Dictionary load | 1 | ~10ms | N/A |
| Hierarchy analysis | 1 | ~5ms | N/A |

## 7. Error Handling

### 7.1 Common Errors

1. **ConfigurationError**: Invalid dictionary format
2. **FileNotFoundError**: Dictionary file missing
3. **JSONDecodeError**: Malformed dictionary JSON
4. **ValueError**: Invalid matching strategy

### 7.2 Error Handling Example

```python
import logging
from pamola_core.utils.nlp.category_matching import CategoryDictionary
from pamola_core.utils.nlp.base import ConfigurationError

logger = logging.getLogger(__name__)

try:
    category_dict = CategoryDictionary.from_file("path/to/dict.json")
    category, score, _ = category_dict.get_best_match("test text")
    
except ConfigurationError as e:
    logger.error(f"Dictionary configuration error: {e}")
    # Use default categories
    
except FileNotFoundError:
    logger.warning("Dictionary not found, using empty dictionary")
    category_dict = CategoryDictionary()
    
except Exception as e:
    logger.error(f"Unexpected error in category matching: {e}")
    # Fallback logic
```

## 8. Testing

### 8.1 Unit Test Example

```python
import pytest
from pamola_core.utils.nlp.category_matching import CategoryDictionary

class TestCategoryMatching:
    def test_exact_match(self):
        dict_data = {
            "Engineer": {
                "keywords": ["engineer", "engineering"],
                "level": 1
            }
        }
        cat_dict = CategoryDictionary(dict_data)
        
        category, score, _ = cat_dict.get_best_match("Engineer")
        assert category == "Engineer"
        assert score == 1.0
    
    def test_partial_match(self):
        dict_data = {
            "Software Engineer": {
                "keywords": ["software engineer"],
                "level": 2
            }
        }
        cat_dict = CategoryDictionary(dict_data)
        
        category, score, _ = cat_dict.get_best_match(
            "Senior Software Engineer"
        )
        assert category == "Software Engineer"
        assert 0 < score < 1
    
    def test_conflict_resolution(self):
        dict_data = {
            "Engineer": {"keywords": ["engineer"], "level": 1},
            "Software Engineer": {
                "keywords": ["software engineer"], 
                "level": 2
            }
        }
        cat_dict = CategoryDictionary(dict_data)
        
        category, _, conflicts = cat_dict.get_best_match(
            "software engineer",
            strategy="specific_first"
        )
        assert category == "Software Engineer"
        assert "Engineer" in conflicts
```

## 9. Migration Guide

### 9.1 From Legacy Systems

If migrating from older classification systems:

```python
# Legacy format
old_categories = {
    "tech": ["developer", "programmer"],
    "mgmt": ["manager", "lead"]
}

# Convert to new format
new_format = {
    "categories_hierarchy": {}
}

for cat_id, keywords in old_categories.items():
    new_format["categories_hierarchy"][cat_id] = {
        "keywords": keywords,
        "level": 1,
        "domain": "General",
        "alias": cat_id
    }

# Save new format
import json
with open("new_categories.json", "w") as f:
    json.dump(new_format, f, indent=2)
```

## 10. Future Enhancements

### 10.1 Planned Features

1. **ML-Based Matching**: Integration with transformer models
2. **Dynamic Dictionary Updates**: Real-time category learning
3. **Multi-language Support**: Enhanced cross-language matching
4. **Fuzzy Matching**: Typo-tolerant keyword matching
5. **Contextual Matching**: Consider surrounding text context

### 10.2 Extension Points

The module is designed for extensibility:

```python
# Custom matching strategy
class MLCategoryMatcher(CategoryDictionary):
    def get_best_match(self, text, strategy="ml_enhanced"):
        if strategy == "ml_enhanced":
            # Custom ML-based matching logic
            embeddings = self.encode_text(text)
            similarities = self.compute_similarities(embeddings)
            return self.select_best_category(similarities)
        else:
            return super().get_best_match(text, strategy)
```

## 11. Conclusion

The Category Matching module provides a robust, performant, and extensible solution for text categorization within the PAMOLA.CORE framework. Its support for hierarchical dictionaries, multiple resolution strategies, and efficient caching makes it suitable for both simple classification tasks and complex, domain-specific categorization requirements.

Key takeaways:
- Use hierarchical dictionaries for better organization
- Choose appropriate resolution strategies for your use case
- Leverage batch processing for large-scale operations
- Monitor and optimize dictionary design for best results
- Implement proper error handling for production systems

For additional support or feature requests, please refer to the PAMOLA.CORE documentation or contact the NLP utilities team.