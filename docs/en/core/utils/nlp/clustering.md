# PAMOLA.CORE NLP Clustering Module Technical Documentation

**Module:** `pamola_core.utils.nlp.clustering`  
**Version:** 1.0.0  
**Last Updated:** December 2024  
**Status:** Active  

## 1. Overview

### 1.1 Purpose

The clustering module provides functionality for grouping similar text items based on token overlap and various similarity metrics. It is designed to efficiently identify and cluster related texts, supporting multiple languages and parallel processing for large-scale text analysis within the PAMOLA.CORE framework.

### 1.2 Key Features

- **Token-based Clustering**: Groups texts based on shared tokens
- **Multiple Similarity Metrics**: Supports Jaccard, overlap, and cosine similarity
- **Language-aware Processing**: Automatic language detection and language-specific tokenization
- **Configurable Thresholds**: Adjustable similarity thresholds for clustering
- **Batch Processing**: Parallel processing support for multiple text batches
- **Caching Support**: Integrated caching for performance optimization

### 1.3 Use Cases

- **Duplicate Detection**: Identifying similar or duplicate text entries
- **Content Grouping**: Organizing related documents or messages
- **Data Deduplication**: Preprocessing step for data cleaning
- **Pattern Recognition**: Finding groups of similar text patterns
- **Category Discovery**: Unsupervised discovery of text categories

## 2. Architecture

### 2.1 Module Structure

```
clustering.py
├── TextClusterer (class)
│   ├── __init__()
│   ├── cluster_texts()
│   └── calculate_similarity() (static)
├── cluster_by_similarity() (cached function)
└── batch_cluster_texts() (batch processing)
```

### 2.2 Dependencies

- `pamola_core.utils.nlp.base`: Base utilities and batch processing
- `pamola_core.utils.nlp.cache`: Caching functionality
- `pamola_core.utils.nlp.language`: Language detection
- `pamola_core.utils.nlp.tokenization`: Text tokenization
- `logging`: Standard Python logging

### 2.3 Integration Points

The module integrates with:
- **Tokenization Module**: For text preprocessing
- **Language Module**: For automatic language detection
- **Cache Module**: For performance optimization
- **Base Module**: For batch processing capabilities

## 3. Core Components

### 3.1 TextClusterer Class

#### 3.1.1 Purpose
The main class responsible for clustering text items based on token similarity.

#### 3.1.2 Constructor

```python
TextClusterer(threshold: float = 0.7, tokenize_fn=None, language: str = "auto")
```

**Parameters:**
- `threshold` (float): Similarity threshold for clustering (0-1). Default: 0.7
- `tokenize_fn` (callable, optional): Custom tokenization function. Default: uses standard tokenize
- `language` (str): Language code or "auto" for automatic detection. Default: "auto"

#### 3.1.3 Methods

##### cluster_texts()
```python
cluster_texts(texts: List[str]) -> Dict[str, List[int]]
```

Clusters texts by similarity using token overlap.

**Parameters:**
- `texts` (List[str]): List of text strings to cluster

**Returns:**
- Dict[str, List[int]]: Dictionary mapping cluster labels to lists of text indices

**Algorithm:**
1. Detects language if set to "auto"
2. Tokenizes all texts into sets of tokens
3. Iteratively builds clusters by comparing token sets
4. Assigns texts to clusters based on similarity threshold

**Example:**
```python
clusterer = TextClusterer(threshold=0.8)
texts = ["Python programming", "Python coding", "Java development"]
clusters = clusterer.cluster_texts(texts)
# Result: {'CLUSTER_0': [0, 1], 'CLUSTER_1': [2]}
```

##### calculate_similarity() (static)
```python
calculate_similarity(tokens1: Set[str], tokens2: Set[str], method: str = "jaccard") -> float
```

Calculates similarity between two token sets.

**Parameters:**
- `tokens1` (Set[str]): First set of tokens
- `tokens2` (Set[str]): Second set of tokens
- `method` (str): Similarity method - "jaccard", "overlap", or "cosine"

**Returns:**
- float: Similarity score (0-1)

**Similarity Methods:**

1. **Jaccard Similarity**:
   - Formula: |A ∩ B| / |A ∪ B|
   - Range: [0, 1]
   - Use case: General purpose similarity

2. **Overlap Coefficient**:
   - Formula: |A ∩ B| / min(|A|, |B|)
   - Range: [0, 1]
   - Use case: When one text might be a subset of another

3. **Cosine Similarity** (simplified):
   - Formula: |A ∩ B| / √(|A| × |B|)
   - Range: [0, 1]
   - Use case: Normalized similarity measure

### 3.2 Module Functions

#### 3.2.1 cluster_by_similarity()
```python
@cache_function(ttl=3600, cache_type='memory')
def cluster_by_similarity(texts: List[str], threshold: float = 0.7, language: str = "auto") -> Dict[str, List[int]]
```

Cached wrapper function for text clustering.

**Features:**
- Memory caching with 1-hour TTL
- Simplified interface for common use cases
- Automatic cache key generation based on inputs

#### 3.2.2 batch_cluster_texts()
```python
def batch_cluster_texts(texts_list: List[List[str]], threshold: float = 0.7,
                       language: str = "auto", processes: Optional[int] = None) -> List[Dict[str, List[int]]]
```

Performs clustering on multiple batches of texts in parallel.

**Parameters:**
- `texts_list` (List[List[str]]): List of text batches to cluster
- `threshold` (float): Similarity threshold
- `language` (str): Language code or "auto"
- `processes` (int, optional): Number of parallel processes

**Returns:**
- List[Dict[str, List[int]]]: List of clustering results, one per batch

**Use Case:**
Processing large datasets by dividing them into batches and clustering in parallel.

## 4. Usage Examples

### 4.1 Basic Clustering

```python
from pamola_core.utils.nlp.clustering import TextClusterer

# Create clusterer with custom threshold
clusterer = TextClusterer(threshold=0.75)

# Cluster similar job titles
job_titles = [
    "Senior Software Engineer",
    "Sr. Software Engineer", 
    "Lead Developer",
    "Software Engineer Senior",
    "Data Scientist",
    "Senior Data Scientist"
]

clusters = clusterer.cluster_texts(job_titles)
# Result: Groups similar job titles together
```

### 4.2 Using Different Similarity Methods

```python
from pamola_core.utils.nlp.clustering import TextClusterer

clusterer = TextClusterer()

# Tokenize texts
tokens1 = {"python", "programming", "language"}
tokens2 = {"python", "coding", "language"}

# Compare different methods
jaccard = clusterer.calculate_similarity(tokens1, tokens2, method="jaccard")
overlap = clusterer.calculate_similarity(tokens1, tokens2, method="overlap")
cosine = clusterer.calculate_similarity(tokens1, tokens2, method="cosine")

print(f"Jaccard: {jaccard:.2f}")  # 0.50
print(f"Overlap: {overlap:.2f}")  # 0.67
print(f"Cosine: {cosine:.2f}")   # 0.58
```

### 4.3 Batch Processing

```python
from pamola_core.utils.nlp.clustering import batch_cluster_texts

# Multiple batches of texts
batch1 = ["text analysis", "text processing", "data mining"]
batch2 = ["machine learning", "deep learning", "neural networks"]
batch3 = ["web development", "frontend development", "backend development"]

# Cluster all batches in parallel
results = batch_cluster_texts(
    [batch1, batch2, batch3],
    threshold=0.6,
    processes=3
)

# Process results
for i, clusters in enumerate(results):
    print(f"Batch {i+1}: {len(clusters)} clusters found")
```

### 4.4 Language-Specific Clustering

```python
from pamola_core.utils.nlp.clustering import cluster_by_similarity

# Russian texts
russian_texts = [
    "программирование на Python",
    "разработка на Python",
    "веб-разработка"
]

# Cluster with automatic language detection
clusters = cluster_by_similarity(
    russian_texts,
    threshold=0.7,
    language="auto"  # Will detect Russian
)
```

## 5. Best Practices

### 5.1 Threshold Selection

- **High Threshold (0.8-0.9)**: For finding near-duplicates or very similar texts
- **Medium Threshold (0.6-0.8)**: For general clustering of related content
- **Low Threshold (0.4-0.6)**: For loose grouping or topic discovery

### 5.2 Performance Optimization

1. **Use Caching**: The `cluster_by_similarity` function is cached by default
2. **Batch Processing**: For large datasets, use `batch_cluster_texts`
3. **Appropriate Tokenization**: Use minimum token length to reduce noise
4. **Language Specification**: Specify language when known to avoid detection overhead

### 5.3 Memory Management

- Process large datasets in batches to control memory usage
- Clear cache periodically for long-running processes
- Monitor cluster sizes to detect potential memory issues

## 6. Error Handling

### 6.1 Common Scenarios

1. **Empty Input**: Returns empty dictionary for empty text list
2. **Empty Texts**: Skips empty strings in clustering
3. **Invalid Method**: Falls back to Jaccard similarity with warning
4. **Language Detection Failure**: Falls back to English tokenization

### 6.2 Logging

The module uses standard Python logging:
```python
logger = logging.getLogger(__name__)
```

Key log messages:
- Warnings for unknown similarity methods
- Debug information for clustering progress (if enabled)

## 7. Performance Characteristics

### 7.1 Time Complexity

- **Clustering**: O(n²) where n is the number of texts
- **Similarity Calculation**: O(m) where m is the average number of tokens
- **Tokenization**: O(n × l) where l is average text length

### 7.2 Space Complexity

- **Token Storage**: O(n × m) for n texts with m average tokens
- **Cluster Storage**: O(n) for cluster assignments
- **Cache Storage**: Depends on cache configuration

### 7.3 Optimization Strategies

1. **Early Termination**: Skip already assigned texts
2. **Set Operations**: Efficient token comparison using Python sets
3. **Lazy Evaluation**: Process only when needed
4. **Parallel Processing**: Available through batch functions

## 8. Limitations and Considerations

### 8.1 Current Limitations

1. **Simple Clustering**: Uses greedy single-pass clustering
2. **Token-Based**: May miss semantic similarities
3. **Memory-Based**: All texts must fit in memory
4. **No Hierarchical Clustering**: Produces flat clusters only

### 8.2 Future Enhancements

1. **Advanced Algorithms**: Support for hierarchical and density-based clustering
2. **Semantic Similarity**: Integration with word embeddings
3. **Streaming Support**: Process texts as they arrive
4. **Cluster Refinement**: Post-processing to improve cluster quality

## 9. Related Modules

- `pamola_core.utils.nlp.minhash`: For more efficient similarity computation at scale
- `pamola_core.utils.nlp.entity_extraction`: For clustering based on entities
- `pamola_core.utils.nlp.text_transformer`: For advanced text preprocessing
- `pamola_core.utils.nlp.diversity_metrics`: For measuring cluster diversity

## 10. Conclusion

The clustering module provides a robust foundation for text clustering within the PAMOLA.CORE framework. Its token-based approach, multiple similarity metrics, and parallel processing capabilities make it suitable for a wide range of text clustering tasks. The module's integration with caching and language detection ensures both performance and accuracy in multilingual environments.