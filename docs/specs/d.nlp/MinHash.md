# Specification: Python MinHash Module

## Module Name

`minhash_generator.py`

## Purpose

This module computes MinHash signatures for input text fields, typically long free-text fields or multi-valued text fields (e.g., lists of skills, experience descriptions) to enable fast similarity comparison and diversity evaluation across records.

The module receives a text string, processes it into a set of shingles (n-grams), computes multiple hash functions, and outputs a MinHash signature vector.

## Input

| Parameter      | Type   | Description                                                                  |
| -------------- | ------ | ---------------------------------------------------------------------------- |
| `text`         | string | Input text to be processed (single field or concatenated fields)             |
| `num_perm`     | int    | Number of hash permutations (signature length); typical values: 64, 128, 256 |
| `shingle_size` | int    | Size of n-grams (shingles); typical values: 2 (bigrams), 3 (trigrams)        |

## Output

| Output             | Type             | Description                                                               |
| ------------------ | ---------------- | ------------------------------------------------------------------------- |
| `signature_vector` | list of integers | The computed MinHash signature as a list of integers (length = num\_perm) |

## Functional Description

1. **Preprocessing**
   Normalize input text (e.g., lowercase, strip punctuation, remove extra spaces).

2. **Shingling**
   Split text into shingles of the specified size (e.g., bigrams).

3. **MinHash Computation**
   Use a MinHash library (such as `datasketch.MinHash`) to compute the signature over the shingle set.

4. **Serialization (optional)**
   Convert the signature vector into a serializable string format (e.g., `"12;45;78;..."`) for saving into CSV or database.

## Example Usage

```python
from minhash_generator import compute_minhash

text = "Project manager with extensive experience in software development."
signature = compute_minhash(text, num_perm=128, shingle_size=2)
print(signature)  # Output: [123, 456, 789, ...]
```

## External Dependencies

* Python 3.x
* `datasketch` library (for efficient MinHash computation)
* Optional: `re` or `nltk` for preprocessing and tokenization

## Functions

### `compute_minhash(text: str, num_perm: int = 128, shingle_size: int = 2) -> List[int]`

* **Description**: Computes a MinHash signature for the given text.
* **Parameters**:

  * `text`: The input string.
  * `num_perm`: Number of hash permutations.
  * `shingle_size`: Size of n-grams for shingling.
* **Returns**: A list of integers representing the MinHash signature.

### `serialize_signature(signature: List[int], delimiter: str = ";") -> str`

* **Description**: Converts a signature vector into a string for CSV storage.
* **Parameters**:

  * `signature`: The list of integers.
  * `delimiter`: Delimiter character (default: `;`).
* **Returns**: A serialized string, e.g., `"12;45;78;..."`.

### `deserialize_signature(signature_str: str, delimiter: str = ";") -> List[int]`

* **Description**: Converts a serialized string back into a signature list.
* **Parameters**:

  * `signature_str`: The serialized string.
  * `delimiter`: Delimiter character.
* **Returns**: A list of integers.

## Notes

* Ensure deterministic results by setting a random seed if needed.
* For long texts, consider performance optimizations (e.g., limiting number of shingles).
* Integrate with data pipelines to preprocess CSV datasets and append signature columns.

---

Let me know if you want me to write the actual Python implementation!
