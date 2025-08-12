# PAMOLA.CORE NLP Entity Extraction Module Documentation

**Module:** `pamola_core.utils.nlp.entity_extraction`  
**Version:** 1.0.0  
**Last Updated:** December 2024

## 1. Overview

The `entity_extraction` module provides a high-level, unified API for extracting various types of entities from text data. It serves as the main entry point to the entity extraction functionality within the PAMOLA.CORE NLP utilities, supporting multiple entity types including job positions, organizations, skills, transaction purposes, and custom entities.

### 1.1 Purpose

This module simplifies entity extraction by:
- Providing a single, consistent interface for all entity types
- Supporting multiple extraction strategies (dictionary-based, NER-based, hybrid)
- Offering specialized functions for common entity types
- Enabling custom entity extractor creation
- Implementing efficient caching for improved performance

### 1.2 Key Features

- **Unified API**: Single interface for all entity extraction needs
- **Multiple Entity Types**: Pre-configured support for jobs, organizations, skills, transactions
- **Language Support**: Automatic language detection or explicit language specification
- **Hybrid Approach**: Combines dictionary matching with NER models
- **Caching**: Memory-based caching for efficient repeated extractions
- **Progress Tracking**: Optional progress indicators for batch processing
- **Custom Extractors**: Support for creating domain-specific extractors

## 2. Architecture

### 2.1 Module Structure

```
entity_extraction.py
├── Imports and Dependencies
├── Logger Configuration
├── Main Functions
│   ├── extract_entities()         # Main entry point
│   ├── extract_job_positions()    # Job-specific extraction
│   ├── extract_organizations()    # Organization extraction
│   ├── extract_universities()     # University extraction
│   ├── extract_skills()           # Skill extraction
│   └── extract_transaction_purposes() # Transaction purpose extraction
└── Factory Function
    └── create_custom_entity_extractor() # Custom extractor creation
```

### 2.2 Dependencies

- **Internal**:
  - `pamola_core.utils.nlp.cache`: Caching functionality
  - `pamola_core.utils.nlp.entity`: Core entity extraction implementation
  
- **External**:
  - `logging`: Standard Python logging
  - `typing`: Type hints support

## 3. API Reference

### 3.1 Main Function: `extract_entities()`

The primary function for entity extraction, delegating to specialized extractors based on entity type.

```python
@cache_function(ttl=3600, cache_type='memory')
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
) -> Dict[str, Any]
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `texts` | `List[str]` | List of text strings to process | Required |
| `entity_type` | `str` | Type of entities to extract | `"generic"` |
| `language` | `str` | Language code or "auto" for detection | `"auto"` |
| `dictionary_path` | `Optional[str]` | Path to custom dictionary file | `None` |
| `match_strategy` | `str` | Strategy for resolving matches | `"specific_first"` |
| `use_ner` | `bool` | Whether to use NER models | `True` |
| `record_ids` | `Optional[List[str]]` | Record IDs for tracking | `None` |
| `show_progress` | `bool` | Show progress bar | `False` |
| `**kwargs` | `Any` | Additional extractor-specific parameters | - |

#### Entity Types

- `"generic"`: General entity extraction
- `"job"`: Job positions and titles
- `"organization"`: Company and organization names
- `"skill"`: Technical and soft skills
- `"transaction"`: Transaction purposes and categories

#### Match Strategies

- `"specific_first"`: Prioritize more specific matches
- `"domain_prefer"`: Prefer domain-specific matches
- `"alias_only"`: Use only alias matching
- `"user_override"`: Apply user-defined overrides

#### Returns

```python
{
    "entities": [
        {
            "text": str,           # Original text
            "entity": str,         # Extracted entity
            "category": str,       # Entity category
            "confidence": float,   # Confidence score (0-1)
            "method": str,        # Extraction method used
            "record_id": str      # Record ID if provided
        }
    ],
    "statistics": {
        "total_processed": int,
        "extracted": int,
        "dictionary_matches": int,
        "ner_matches": int,
        "unmatched": int
    },
    "metadata": {
        "entity_type": str,
        "language": str,
        "dictionary_used": str
    }
}
```

### 3.2 Specialized Extraction Functions

#### 3.2.1 `extract_job_positions()`

Extract job positions with optional seniority detection.

```python
def extract_job_positions(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    seniority_detection: bool = True,
    **kwargs
) -> Dict[str, Any]
```

**Additional Parameters:**
- `seniority_detection`: Enable detection of seniority levels (Junior, Senior, etc.)

#### 3.2.2 `extract_organizations()`

Extract organization names with type filtering.

```python
def extract_organizations(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    organization_type: str = "any",
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]
```

**Organization Types:**
- `"any"`: All organization types
- `"company"`: Commercial companies
- `"university"`: Educational institutions
- `"government"`: Government bodies
- `"nonprofit"`: Non-profit organizations

#### 3.2.3 `extract_universities()`

Specialized function for educational institution extraction.

```python
def extract_universities(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]
```

#### 3.2.4 `extract_skills()`

Extract skills with type categorization.

```python
def extract_skills(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    skill_type: str = "technical",
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]
```

**Skill Types:**
- `"technical"`: Programming languages, tools, technologies
- `"soft"`: Communication, leadership, teamwork
- `"language"`: Natural languages
- `"domain"`: Domain-specific expertise

#### 3.2.5 `extract_transaction_purposes()`

Extract transaction purposes and categories.

```python
def extract_transaction_purposes(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]
```

### 3.3 Factory Function

#### `create_custom_entity_extractor()`

Create a custom entity extractor for specific use cases.

```python
def create_custom_entity_extractor(
    entity_type: str,
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    **kwargs
) -> Any
```

**Returns:** A configured entity extractor instance that can be reused for multiple extractions.

## 4. Usage Examples

### 4.1 Basic Entity Extraction

```python
from pamola_core.utils.nlp.entity_extraction import extract_entities

# Extract generic entities
texts = ["Apple Inc. is looking for a Senior Python Developer"]
results = extract_entities(texts, entity_type="generic")

# Access results
for entity in results["entities"]:
    print(f"Found: {entity['entity']} ({entity['category']})")
```

### 4.2 Job Position Extraction

```python
from pamola_core.utils.nlp.entity_extraction import extract_job_positions

texts = [
    "We need a Senior Data Scientist",
    "Junior Frontend Developer position available",
    "Looking for ML Engineer with 5 years experience"
]

results = extract_job_positions(
    texts, 
    language="en",
    seniority_detection=True
)

# Results include seniority levels
for entity in results["entities"]:
    print(f"Position: {entity['entity']}, Seniority: {entity.get('seniority', 'N/A')}")
```

### 4.3 Organization Extraction with Type Filtering

```python
from pamola_core.utils.nlp.entity_extraction import extract_organizations

texts = [
    "Stanford University research shows...",
    "Microsoft Corporation announced...",
    "The Federal Reserve decided..."
]

# Extract only universities
universities = extract_organizations(
    texts,
    organization_type="university"
)

# Extract all organizations
all_orgs = extract_organizations(
    texts,
    organization_type="any"
)
```

### 4.4 Custom Dictionary Usage

```python
from pamola_core.utils.nlp.entity_extraction import extract_skills

# Use custom skill dictionary
results = extract_skills(
    texts=["Proficient in React and Node.js"],
    dictionary_path="/path/to/custom_skills.json",
    skill_type="technical"
)
```

### 4.5 Batch Processing with Progress

```python
from pamola_core.utils.nlp.entity_extraction import extract_entities

# Large batch with progress tracking
texts = ["Text " + str(i) for i in range(10000)]
record_ids = [f"REC_{i}" for i in range(10000)]

results = extract_entities(
    texts,
    entity_type="job",
    record_ids=record_ids,
    show_progress=True
)

print(f"Processed: {results['statistics']['total_processed']}")
print(f"Extracted: {results['statistics']['extracted']}")
```

### 4.6 Creating Custom Extractor

```python
from pamola_core.utils.nlp.entity_extraction import create_custom_entity_extractor

# Create reusable extractor
medical_extractor = create_custom_entity_extractor(
    entity_type="medical",
    dictionary_path="/path/to/medical_terms.json",
    match_strategy="domain_prefer"
)

# Use for multiple extractions
texts1 = ["Patient diagnosed with Type 2 Diabetes"]
texts2 = ["Prescribed metformin for treatment"]

results1 = medical_extractor.extract(texts1)
results2 = medical_extractor.extract(texts2)
```

## 5. Best Practices

### 5.1 Performance Optimization

1. **Use Caching**: The module automatically caches results for 1 hour
2. **Batch Processing**: Process multiple texts in a single call
3. **Disable NER**: Set `use_ner=False` if dictionary matching is sufficient
4. **Custom Dictionaries**: Use domain-specific dictionaries for better accuracy

### 5.2 Accuracy Improvement

1. **Language Specification**: Specify language explicitly when known
2. **Match Strategy**: Choose appropriate strategy for your domain
3. **Custom Dictionaries**: Create comprehensive domain dictionaries
4. **Entity Type Selection**: Use specific entity types over generic

### 5.3 Memory Management

1. **Large Batches**: Process in chunks for very large datasets
2. **Progress Tracking**: Enable only when needed (adds overhead)
3. **Custom Extractors**: Reuse extractors for multiple operations

## 6. Error Handling

The module implements robust error handling:

```python
try:
    results = extract_entities(texts, entity_type="job")
except ValueError as e:
    # Invalid parameters
    logger.error(f"Invalid parameters: {e}")
except FileNotFoundError as e:
    # Dictionary file not found
    logger.error(f"Dictionary not found: {e}")
except Exception as e:
    # Other errors
    logger.error(f"Extraction failed: {e}")
```

## 7. Configuration

### 7.1 Logging

Configure logging level:

```python
import logging
logging.getLogger("pamola_core.utils.nlp.entity_extraction").setLevel(logging.INFO)
```

### 7.2 Cache Settings

The module uses memory caching with 1-hour TTL. This can be modified by adjusting the `@cache_function` decorator parameters.

## 8. Integration with PAMOLA.CORE

### 8.1 Operation Integration

```python
from pamola_core.operations import DataEnrichmentOperation
from pamola_core.utils.nlp.entity_extraction import extract_job_positions

class JobExtractionOperation(DataEnrichmentOperation):
    def process(self, df):
        results = extract_job_positions(
            df['job_title'].tolist(),
            show_progress=True
        )
        # Process results...
```

### 8.2 Pipeline Integration

```python
from pamola_core.pipeline import Pipeline
from pamola_core.utils.nlp.entity_extraction import extract_entities

# Add entity extraction step
pipeline.add_step(
    lambda df: extract_entities(
        df['text'].tolist(),
        entity_type="organization"
    )
)
```

## 9. Troubleshooting

### 9.1 Common Issues

1. **No Entities Found**
   - Check dictionary path and format
   - Verify language setting
   - Enable NER if disabled

2. **Low Confidence Scores**
   - Use more specific entity types
   - Improve dictionary quality
   - Adjust match strategy

3. **Performance Issues**
   - Process in smaller batches
   - Use custom extractors for repeated operations
   - Disable progress tracking

### 9.2 Debug Mode

Enable debug logging:

```python
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pamola_core.utils.nlp.entity_extraction")
```

## 10. Future Enhancements

- Support for additional entity types
- Multi-lingual entity linking
- Confidence score calibration
- GPU acceleration for NER models
- Streaming API for real-time extraction