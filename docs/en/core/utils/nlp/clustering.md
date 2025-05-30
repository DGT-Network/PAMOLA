# PAMOLA Core: Text Clustering Module

## Overview

The `clustering.py` module provides robust text clustering capabilities for the PAMOLA Core framework. It enables grouping of similar text items using configurable similarity metrics and tokenization strategies. This module is designed for flexible integration into NLP pipelines, supporting batch processing and parallel execution for large-scale text datasets.

---

## Key Features

- **Text Clustering**: Groups similar texts based on token overlap and configurable similarity thresholds.
- **Multiple Similarity Metrics**: Supports Jaccard, Overlap, and Cosine similarity for flexible clustering.
- **Language Detection**: Automatically detects language for tokenization if not specified.
- **Custom Tokenization**: Allows injection of custom tokenization functions.
- **Batch Processing**: Efficiently clusters multiple batches of texts in parallel.
- **Caching**: Results of clustering can be cached for performance.

---

## Dependencies

### Standard Library
- `logging`
- `typing` (`Dict`, `List`, `Set`, `Optional`)

### Internal Modules
- `pamola_core.utils.nlp.base.batch_process`
- `pamola_core.utils.nlp.cache.cache_function`
- `pamola_core.utils.nlp.language.detect_language`
- `pamola_core.utils.nlp.tokenization.tokenize`

---

## Exception Classes

> **Note:** This module does not define custom exception classes. Standard Python exceptions (e.g., `ValueError`, `TypeError`) may be raised in case of invalid input or configuration. Handle exceptions as shown below:

```python
try:
    clusters = cluster_by_similarity(["text1", "text2"])
except Exception as e:
    # Handle unexpected errors
    print(f"Clustering failed: {e}")
```

---

## Main Classes and Functions

### `TextClusterer`

#### Constructor
```python
TextClusterer(
    threshold: float = 0.7,
    tokenize_fn=None,
    language: str = "auto"
)
```
**Parameters:**
- `threshold`: Similarity threshold for clustering (0-1).
- `tokenize_fn`: Optional custom tokenization function.
- `language`: Language code or "auto" for detection.

#### Key Attributes
- `threshold`: Similarity threshold for clustering.
- `tokenize_fn`: Tokenization function used for processing texts.
- `language`: Language code for tokenization.

#### Public Methods

##### `cluster_texts`
```python
def cluster_texts(
    self,
    texts: List[str]
) -> Dict[str, List[int]]
```
- **texts**: List of text strings to cluster.
- **Returns**: Dictionary mapping cluster labels to lists of text indices.
- **Raises**: None (returns empty dict if input is empty).

##### `calculate_similarity`
```python
@staticmethod
def calculate_similarity(
    tokens1: Set[str],
    tokens2: Set[str],
    method: str = "jaccard"
) -> float
```
- **tokens1**: First set of tokens.
- **tokens2**: Second set of tokens.
- **method**: Similarity method: "jaccard", "overlap", "cosine".
- **Returns**: Similarity score (0-1).
- **Raises**: None (logs warning and defaults to Jaccard if method is unknown).

---

### Top-Level Functions

#### `cluster_by_similarity`
```python
def cluster_by_similarity(
    texts: List[str],
    threshold: float = 0.7,
    language: str = "auto"
) -> Dict[str, List[int]]
```
- **texts**: List of text strings.
- **threshold**: Similarity threshold for clustering (0-1).
- **language**: Language code or "auto" to detect.
- **Returns**: Dictionary mapping cluster labels to lists of text indices.
- **Raises**: None.
- **Notes**: Results are cached in memory for 1 hour.

#### `batch_cluster_texts`
```python
def batch_cluster_texts(
    texts_list: List[List[str]],
    threshold: float = 0.7,
    language: str = "auto",
    processes: Optional[int] = None
) -> List[Dict[str, List[int]]]
```
- **texts_list**: List of text batches to cluster.
- **threshold**: Similarity threshold.
- **language**: Language code or "auto".
- **processes**: Number of processes for parallel execution.
- **Returns**: List of clustering results, one per batch.

---

## Dependency Resolution and Completion Validation

- **Language Detection**: If `language` is set to "auto", the language of the first text is detected and used for tokenization.
- **Tokenization**: All texts are tokenized using the provided or default function. Empty texts are handled gracefully.
- **Clustering Logic**: Each text is compared to others using the selected similarity metric. If similarity exceeds the threshold, texts are grouped into the same cluster.
- **Batch Processing**: Uses `batch_process` to parallelize clustering across multiple text batches.

---

## Usage Examples

### Basic Clustering
```python
from pamola_core.utils.nlp.clustering import cluster_by_similarity

# List of texts to cluster
texts = [
    "The quick brown fox",
    "A fast brown fox",
    "Lazy dog sleeps",
    "Dog is sleeping lazily"
]

# Cluster texts with default settings
clusters = cluster_by_similarity(texts)
print(clusters)
# Output: {'CLUSTER_0': [0, 1], 'CLUSTER_1': [2, 3]}
```

### Custom Tokenization and Similarity
```python
def custom_tokenizer(text, language=None, min_length=2):
    # Example: split on whitespace and filter short tokens
    return [t for t in text.lower().split() if len(t) >= min_length]

from pamola_core.utils.nlp.clustering import TextClusterer

clusterer = TextClusterer(threshold=0.5, tokenize_fn=custom_tokenizer, language="en")
clusters = clusterer.cluster_texts(["apple orange", "orange banana", "car bus"])
print(clusters)
```

### Batch Clustering
```python
from pamola_core.utils.nlp.clustering import batch_cluster_texts

batches = [
    ["cat dog", "dog cat", "fish bird"],
    ["apple pie", "pie apple", "banana split"]
]
results = batch_cluster_texts(batches, threshold=0.6)
print(results)
```

### Handling Empty or Invalid Input
```python
try:
    clusters = cluster_by_similarity([])
    print(clusters)  # {}
except Exception as e:
    print(f"Error: {e}")
```

---

## Integration Notes

- The clustering module is designed for easy integration with other NLP and data processing components in PAMOLA Core.
- Can be used as a standalone utility or as part of a larger pipeline (e.g., with BaseTask or batch processing workflows).
- Results are compatible with downstream analytics and reporting modules.

---

## Error Handling and Exception Hierarchy

- No custom exceptions are defined in this module.
- Standard exceptions may be raised by dependencies (e.g., tokenization, language detection). Handle these using try/except blocks as shown above.
- All warnings (e.g., unknown similarity method) are logged using the standard `logging` module.

---

## Configuration Requirements

- No special configuration is required for basic usage.
- For custom tokenization or language support, provide appropriate functions or language codes.
- For batch processing, ensure that the `batch_process` utility is available and configured for your environment.

---

## Security Considerations and Best Practices

- **Input Validation**: Always validate and sanitize input texts to avoid processing unexpected or malicious data.
- **Custom Tokenizers**: Ensure custom tokenization functions do not execute unsafe code or expose sensitive data.
- **Caching**: Cached results are stored in memory; avoid caching sensitive data if running in shared environments.

### Security Failure Example
```python
def insecure_tokenizer(text, language=None, min_length=2):
    # BAD: Using eval on user input is dangerous!
    return eval(text)

try:
    clusterer = TextClusterer(tokenize_fn=insecure_tokenizer)
    clusterer.cluster_texts(["__import__('os').system('rm -rf /')"])
except Exception as e:
    print("Security failure handled:", e)
```
**Risk**: Using unsafe tokenization can lead to code execution vulnerabilities. Always use safe, deterministic tokenization.

---

## Internal vs. External Dependencies

- **Internal**: Uses internal PAMOLA Core utilities for tokenization, language detection, and batch processing.
- **External**: No external (absolute path) dependencies are required for this module.

---

## Best Practices

1. **Use Default Tokenization for Most Cases**: The built-in tokenizer is robust for general text. Only override if you have special requirements.
2. **Set Thresholds Appropriately**: Adjust the similarity threshold based on your use case (e.g., higher for stricter clustering).
3. **Batch Large Datasets**: Use `batch_cluster_texts` for large-scale or parallel clustering.
4. **Handle Empty Inputs Gracefully**: Always check for empty or invalid input lists.
5. **Log Warnings and Errors**: Use the logging output to monitor for unexpected behavior or configuration issues.
6. **Avoid Unsafe Tokenization**: Never use `eval` or similar unsafe operations in custom tokenizers.
