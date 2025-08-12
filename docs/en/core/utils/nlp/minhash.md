# MinHash Module Documentation

## Overview

The `minhash.py` module is part of the PAMOLA.CORE framework for privacy-preserving AI data processors. It implements MinHash signature generation for text data to enable efficient similarity comparisons and diversity evaluation across records in databases or datasets, particularly for anonymization and clustering applications.

## Purpose

This module computes MinHash signatures for input text fields, enabling:

1. Fast similarity comparison between text records
2. Efficient diversity evaluation across records
3. Clustering of similar text content
4. Anonymization risk assessment

The module is particularly useful for processing:
- Long free-text fields (e.g., professional experience descriptions)
- Multi-valued text fields (e.g., lists of skills, certifications)
- Educational and professional qualifications
- Any text content requiring similarity analysis

## Key Features

- **Language-agnostic processing**: Full Unicode support for mixed-language content, particularly Russian and English
- **Memory-efficient batch processing**: Designed for handling large datasets with manageable memory usage
- **Configurable signatures**: Adjustable permutation count and shingle size for different precision/performance trade-offs
- **Serialization functionality**: Support for converting signatures to storable formats for CSV/database persistence
- **Performance optimization**: Caching options for repeated operations on the same text
- **Integration with datasketch**: Leverages the efficient MinHash implementation from the datasketch library

## Architecture

The module follows a layered architecture:

1. **Core Functions Layer**
   - Fundamental operations: preprocessing, shingling, MinHash computation
   - Pure functions with minimal dependencies

2. **Utility Layer**
   - Helper functions: serialization, deserialization, similarity calculation
   - Caching mechanisms

3. **Batch Processing Layer**
   - File handling: CSV reading/writing
   - Progress tracking and reporting

4. **Integration Layer**
   - Compatibility with PAMOLA.CORE operation framework
   - Integration with other modules (e.g., stopwords)

## Main Functions

### `compute_minhash(text, num_perm=128, shingle_size=2)`

Computes a MinHash signature for given text.

**Parameters:**
- `text` (str): Input text to compute a signature for
- `num_perm` (int): Number of hash permutations (signature length); typical values: 64, 128, 256
- `shingle_size` (int): Size of n-grams (shingles); typical values: 2 (bigrams), 3 (trigrams)

**Returns:**
- List[int]: A list of integers representing the MinHash signature

**Process:**
1. Creates a MinHash object with specified permutations
2. Preprocesses the input text (normalization, cleaning)
3. Generates shingles (n-grams) from the text
4. Updates the MinHash with encoded shingles
5. Returns the resulting signature as a list of integers

### `preprocess_text(text, lowercase=True, remove_punctuation=True, remove_extra_spaces=True, remove_stopwords=False, languages=None)`

Normalizes and cleans input text.

**Parameters:**
- `text` (str): Input text to preprocess
- `lowercase` (bool): Whether to convert text to lowercase
- `remove_punctuation` (bool): Whether to remove punctuation
- `remove_extra_spaces` (bool): Whether to normalize whitespace
- `remove_stopwords` (bool): Whether to remove stopwords
- `languages` (List[str]): Languages for stopword removal, defaults to ['en', 'ru']

**Returns:**
- str: Preprocessed text

### `create_shingles(text, shingle_size=2, method='character')`

Creates shingles (n-grams) from input text.

**Parameters:**
- `text` (str): Input text to create shingles from
- `shingle_size` (int): Size of n-grams (shingles)
- `method` (str): Shingling method: 'character', 'word', or 'unicode'

**Returns:**
- Set[str]: Set of shingles

### `serialize_signature(signature, delimiter=";")`

Converts a signature vector into a string for storage.

**Parameters:**
- `signature` (List[int]): MinHash signature as list of integers
- `delimiter` (str): Delimiter character for joining values

**Returns:**
- str: Serialized signature string

### `deserialize_signature(signature_str, delimiter=";")`

Converts a serialized signature string back into a list of integers.

**Parameters:**
- `signature_str` (str): Serialized signature string
- `delimiter` (str): Delimiter character used in the serialized string

**Returns:**
- List[int]: MinHash signature as list of integers

### `calculate_jaccard_similarity(signature1, signature2)`

Calculates Jaccard similarity between two MinHash signatures.

**Parameters:**
- `signature1` (List[int]): First MinHash signature
- `signature2` (List[int]): Second MinHash signature

**Returns:**
- float: Estimated Jaccard similarity (0-1 range)

### `process_csv_file(input_path, output_path, field_name, id_field, num_perm=128, shingle_size=2, batch_size=1000, preprocessing_params=None)`

Processes a CSV file, computing MinHash signatures for a specified field.

**Parameters:**
- `input_path` (str): Path to input CSV file
- `output_path` (str): Path to output CSV file
- `field_name` (str): Name of the field to compute signatures for
- `id_field` (str): Name of the ID field to include in output
- `num_perm` (int): Number of hash permutations
- `shingle_size` (int): Size of n-grams (shingles)
- `batch_size` (int): Number of records to process in each batch
- `preprocessing_params` (Dict): Parameters for text preprocessing

**Returns:**
- OperationResult: Result of the operation

## Usage Examples

### Basic Example: Computing a MinHash Signature

```python
from pamola_core.utils.nlp.minhash import compute_minhash

# Compute a signature for a text string
text = "Project manager with extensive experience in software development."
signature = compute_minhash(text, num_perm=128, shingle_size=2)

print(signature)  # Output: [12345, 67890, ...]
```

### Comparing Two Texts for Similarity

```python
from pamola_core.utils.nlp.minhash import compute_minhash, calculate_jaccard_similarity

# Compute signatures for two texts
text1 = "Software developer with Java and Python experience"
text2 = "Python and Java software developer with 5 years experience"

sig1 = compute_minhash(text1)
sig2 = compute_minhash(text2)

# Calculate similarity
similarity = calculate_jaccard_similarity(sig1, sig2)
print(f"Similarity: {similarity:.2f}")  # Output: Similarity: 0.73
```

### Processing a CSV File

```python
from pamola_core.utils.nlp.minhash import process_csv_file
from pamola_core.utils.ops.op_result import OperationStatus

# Process a CSV file with experience descriptions
result = process_csv_file(
    input_path="resumes.csv",
    output_path="resume_signatures.csv",
    field_name="experience_description",
    id_field="resume_id",
    num_perm=256,
    shingle_size=3,
    batch_size=500
)

if result.status == OperationStatus.SUCCESS:
    print(f"Successfully processed {result.get_metric('processed_records')} records")
else:
    print(f"Error: {result.error_message}")
```

### Custom Text Preprocessing

```python
from pamola_core.utils.nlp.minhash import compute_minhash, preprocess_text

# Custom preprocessing
text = "PROJECT MANAGER (2015-2020): Led development teams on multiple projects."
preprocessed = preprocess_text(
    text,
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    languages=['en']
)

# Compute MinHash with custom preprocessing
signature = compute_minhash(preprocessed)
```

### Serializing Signatures for Storage

```python
from pamola_core.utils.nlp.minhash import compute_minhash, serialize_signature, deserialize_signature

# Compute a signature
text = "Data scientist with ML expertise"
signature = compute_minhash(text)

# Serialize for storage
serialized = serialize_signature(signature)
print(serialized)  # Output: "12345;67890;..."

# Store in database or CSV...

# Later, deserialize
restored_signature = deserialize_signature(serialized)
```

## Performance Considerations

- **Memory Usage**: For large datasets, use batch processing to avoid loading all data into memory
- **Signature Size**: Larger `num_perm` values increase accuracy but also increase memory and storage requirements
- **Shingle Size**: Larger shingle sizes capture more context but may reduce sensitivity to small changes
- **Unicode Support**: Unicode processing is more resource-intensive but necessary for multilingual text

## Integration with Other Modules

The MinHash module integrates with:

1. **Stopwords Module**: For enhanced text preprocessing with multilingual stopword removal
2. **Operation Cache**: For performance optimization with repeated operations
3. **Operation Result**: For standardized reporting and error handling

## Future Extensions

The module is designed to be compatible with:

1. **Locality-Sensitive Hashing (LSH)**: For efficient nearest-neighbor search across large datasets
2. **Parallel Processing**: For improved performance on multi-core systems
3. **Additional Shingling Methods**: For specialized text types and languages