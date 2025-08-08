# PAMOLA.CORE NLP Package Documentation

## Overview

The `pamola_core.utils.nlp` package is a comprehensive natural language processing framework within PAMOLA.CORE, designed specifically for privacy-preserving data operations. It provides advanced text processing capabilities essential for anonymization, entity extraction, and data transformation workflows while maintaining graceful degradation when optional dependencies are unavailable.

## Package Purpose

The NLP package serves as the foundation for text-based privacy operations in PAMOLA.CORE by providing:

- **Privacy-Preserving Text Processing**: LLM-powered anonymization and transformation
- **Entity Recognition and Extraction**: Multi-language identification of sensitive entities
- **Text Analysis and Clustering**: Similarity matching and diversity assessment for k-anonymity
- **Efficient Data Processing**: Scalable handling of large text datasets with intelligent caching

## Architecture

```
pamola_core/utils/nlp/
├── Core Infrastructure
│   ├── base.py                    # Exception hierarchy, utilities, dependency management
│   ├── compatibility.py           # Graceful degradation for optional dependencies
│   └── cache.py                   # Multi-backend caching system (memory/file/model/text)
│
├── Text Processing
│   ├── text_utils.py              # Text normalization, similarity calculations
│   ├── tokenization.py            # Multi-method tokenization and lemmatization
│   ├── tokenization_ext.py        # Extended tokenization capabilities
│   ├── tokenization_helpers.py    # Resource management for tokenization
│   ├── stopwords.py               # Multi-language stopword management
│   └── language.py                # Language detection and analysis
│
├── Entity Processing
│   ├── entity_extraction.py       # Generic entity extraction interface
│   └── entity/                    # Specialized entity extractors
│       ├── __init__.py
│       ├── job_positions.py       # Job title extraction
│       ├── organizations.py       # Company/organization extraction
│       ├── skills.py              # Technical/soft skills extraction
│       └── locations.py           # Geographic entity extraction
│
├── Analysis and Metrics
│   ├── clustering.py              # Text clustering algorithms
│   ├── minhash.py                 # Efficient similarity computation
│   ├── diversity_metrics.py       # Text diversity assessment
│   └── category_matching.py       # Hierarchical category classification
│
├── LLM Integration (llm/)
│   ├── __init__.py
│   ├── client.py                  # Multi-provider LLM connectivity
│   ├── config.py                  # Model configurations and presets
│   ├── processing.py              # Pre/post-processing pipeline
│   ├── prompt.py                  # Template management
│   └── metrics.py                 # Performance tracking
│
├── Data Operations
│   ├── text_transformer.py        # High-level LLM text processing interface
│   ├── dataframe_utils.py         # Pandas DataFrame utilities
│   └── model_manager.py           # NLP model lifecycle management
│
└── __init__.py                    # Package exports and initialization
```

## Key Design Principles

### 1. **Graceful Degradation**
All modules implement fallback strategies to ensure functionality even when specialized NLP libraries (NLTK, spaCy, transformers) are unavailable. This is managed through the `compatibility.py` module and `DependencyManager` class.

### 2. **Privacy-First Design**
Every component is designed with privacy operations in mind:
- Entity extraction identifies sensitive information for anonymization
- Text clustering supports k-anonymity grouping
- LLM integration includes built-in anonymization prompts
- Caching respects privacy boundaries

### 3. **Performance Optimization**
- **Multi-level Caching**: Memory, file, model, and text-specific caches
- **Batch Processing**: Efficient handling of large datasets
- **Resource Management**: Intelligent loading and unloading of models
- **Parallel Processing**: Support for multi-core operations

### 4. **Multilingual Support**
- Language detection for 20+ languages
- Language-specific tokenization and lemmatization
- Multilingual stopword management
- Cross-language entity extraction

## Core Components

### Infrastructure Layer

#### base.py
Provides foundational utilities including:
- **Exception Hierarchy**: `NLPError`, `ResourceNotFoundError`, `ModelNotAvailableError`, `LLMError` family
- **Dependency Management**: `DependencyManager` for checking and loading optional modules
- **Common Utilities**: `batch_process()`, `normalize_language_code()`, resource path management

#### cache.py
Implements comprehensive caching with multiple backends:
- **MemoryCache**: In-memory with LRU/LFU/FIFO policies
- **FileCache**: Disk-based with modification tracking
- **ModelCache**: Memory-aware caching for ML models
- **TextCache**: Specialized for text with canonicalization

#### compatibility.py
Manages optional dependencies and provides fallback mechanisms for all NLP libraries.

### Text Processing Layer

#### text_utils.py
Essential text manipulation functions:
- **Normalization**: Multi-level text normalization (basic/advanced/aggressive)
- **Similarity**: Multiple algorithms (ratio, partial, token, Levenshtein)
- **Utilities**: Category name cleaning, composite value splitting

#### tokenization.py
Flexible tokenization with multiple backends:
- **SimpleTokenizer**: Always available, regex-based
- **NLTKTokenizer**: Advanced with language-specific rules
- **SpacyTokenizer**: Linguistic-aware processing
- **TransformersTokenizer**: Neural subword tokenization

#### language.py
Robust language detection:
- Multiple detection methods (FastText, langdetect, heuristics)
- Mixed-language analysis
- Script detection (Latin, Cyrillic, CJK)
- Confidence scoring

### Entity Extraction Layer

#### entity_extraction.py
Unified interface for entity extraction with specialized extractors:
- Organizations (companies, institutions)
- Job positions (titles, seniority levels)
- Skills (technical, soft skills)
- Locations (cities, countries, regions)
- Personal identifiers (names, emails, phones)

### Analysis Layer

#### clustering.py
Text clustering for grouping similar content:
- Multiple distance metrics
- Configurable thresholds
- Batch processing support

#### minhash.py
Efficient similarity computation using MinHash signatures for large-scale deduplication.

#### diversity_metrics.py
Assessment of text diversity in datasets:
- Lexical diversity (TTR, MTLD)
- Semantic diversity (token overlap)
- Statistical measures

### LLM Integration Layer

#### text_transformer.py
Primary interface for LLM-based text processing:
- Unified API for multiple LLM providers
- Built-in anonymization templates
- Intelligent caching and batch processing
- Checkpoint support for long operations
- Comprehensive metrics collection

#### llm/ subsystem
Complete LLM integration framework:
- **client.py**: WebSocket/HTTP connectivity for LMStudio, OpenAI
- **config.py**: Model presets and parameter management
- **processing.py**: Text preprocessing and response validation
- **prompt.py**: Template library with anonymization prompts
- **metrics.py**: Token usage and performance tracking

### Data Operations Layer

#### dataframe_utils.py
Specialized pandas operations:
- Marker-based processing tracking
- Batch processing utilities
- Progress monitoring
- Error handling

#### model_manager.py
Lifecycle management for NLP models:
- Automatic loading/unloading
- Memory pressure monitoring
- Model metadata tracking

## Usage Patterns

### Basic Text Processing
```python
from pamola_core.utils.nlp import normalize_text, tokenize, extract_entities

# Normalize text for consistency
normalized = normalize_text(text, level="advanced")

# Tokenize with language detection
tokens = tokenize(text, language="auto")

# Extract entities
entities = extract_entities(text, entity_type="all")
```

### LLM-Based Anonymization
```python
from pamola_core.utils.nlp.text_transformer import TextTransformer

# Initialize transformer
transformer = TextTransformer(
    llm_config={'provider': 'lmstudio', 'model_name': 'gemma-2-2b-it'},
    processing_config={'batch_size': 10},
    generation_config={'temperature': 0.3},
    prompt_template='anonymize_experience_ru',
    task_dir='./anonymization_output'
)

# Process DataFrame
result_df = transformer.process_dataframe(
    df, 
    source_column='biography',
    target_column='biography_anonymized'
)
```

### Entity-Based Anonymization
```python
from pamola_core.utils.nlp import extract_entities, CategoryDictionary

# Extract sensitive entities
entities = extract_entities(texts, entity_type="person")

# Categorize for generalization
category_dict = CategoryDictionary.from_file('job_categories.json')
category, score = category_dict.get_best_match(job_title)
```

### Text Clustering for K-Anonymity
```python
from pamola_core.utils.nlp.clustering import TextClusterer

# Cluster similar texts
clusterer = TextClusterer(threshold=0.8)
clusters = clusterer.cluster_texts(job_descriptions)

# Use clusters for k-anonymity grouping
for cluster_id, indices in clusters.items():
    # Apply same anonymization to cluster members
    generalized_value = generalize_cluster(df.iloc[indices])
```

## Integration with PAMOLA.CORE Operations

The NLP package integrates seamlessly with PAMOLA.CORE anonymization operations:

1. **Pre-processing**: Entity extraction identifies sensitive data
2. **Anonymization**: Text transformation removes/generalizes PII
3. **Post-processing**: Validation and quality assessment
4. **Metrics**: Privacy and utility measurement

## Performance Characteristics

- **Caching**: Reduces redundant processing by up to 90%
- **Batch Processing**: Handles millions of records efficiently
- **Memory Management**: Automatic cleanup for large datasets
- **Parallel Processing**: Utilizes all available cores

## Configuration

### Environment Variables
```bash
# Resource directories
export PAMOLA_NLP_RESOURCES=/path/to/resources
export PAMOLA_STOPWORDS_DIR=/path/to/stopwords
export PAMOLA_TOKENIZATION_DIR=/path/to/tokenization

# Cache configuration
export PAMOLA_CACHE_SIZE=1000
export PAMOLA_CACHE_TTL=3600
export PAMOLA_DISABLE_CACHE=0

# Model paths
export PAMOLA_FASTTEXT_MODEL=/path/to/fasttext/model
```

### Dependency Management
The package automatically detects available libraries and adjusts functionality:
- **Required**: Python 3.6+, pandas
- **Optional**: nltk, spacy, transformers, langdetect, fasttext, pymorphy2

## Best Practices

1. **Use Appropriate Cache Backend**
   - Memory cache for speed
   - File cache for persistence
   - Redis cache for distributed systems

2. **Configure Batch Sizes**
   - Start small (5-10) for LLM operations
   - Increase for simple operations (100-1000)

3. **Monitor Resource Usage**
   - Enable memory profiling for large datasets
   - Use ModelCache for ML models
   - Set appropriate cleanup intervals

4. **Handle Multilingual Content**
   - Let language detection run automatically
   - Provide language hints when known
   - Use language-specific resources
