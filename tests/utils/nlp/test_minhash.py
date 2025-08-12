"""
Tests for the utils.nlp.minhash module.

This module contains unit tests for the abstract base classes and concrete
implementations in the minhash.py module.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
from typing import List
import tempfile
import os
import csv

from pamola_core.utils.nlp.minhash import (
    compute_minhash,
    preprocess_text,
    create_shingles,
    serialize_signature,
    deserialize_signature,
    calculate_jaccard_similarity,
    cached_compute_minhash,
    estimate_optimal_num_perm,
    get_cache_key,
    batch_compute_minhash,
    create_minhash_generator,
    process_csv_file
)



class TestMinHashUtils(unittest.TestCase):

    def test_preprocess_text_basic(self):
        text = "This is a test.!! With punctuation and UPPERCASE letters."
        processed = preprocess_text(text)
        expected = "this is a test with punctuation and uppercase letters"
        self.assertEqual(processed, expected)

    def test_create_shingles_character(self):
        text = "abcd"
        shingles = create_shingles(text, shingle_size=2)
        expected = {'ab', 'bc', 'cd'}
        self.assertEqual(shingles, expected)

    def test_create_shingles_word(self):
        text = "the quick brown fox"
        shingles = create_shingles(text, shingle_size=2, method='word')
        expected = {'the quick', 'quick brown', 'brown fox'}
        self.assertEqual(shingles, expected)

    def test_compute_minhash_signature_length(self):
        text = "example text for hashing"
        signature = compute_minhash(text, num_perm=64)
        self.assertEqual(len(signature), 64)
        self.assertTrue(all(isinstance(i, int) for i in signature))

    def test_serialize_and_deserialize_signature(self):
        signature = list(range(10))
        serialized = serialize_signature(signature)
        deserialized = deserialize_signature(serialized)
        self.assertEqual(signature, deserialized)

    def test_calculate_jaccard_similarity_identical(self):
        sig1 = list(range(100))
        sig2 = list(range(100))
        similarity = calculate_jaccard_similarity(sig1, sig2)
        self.assertEqual(similarity, 1.0)

    def test_calculate_jaccard_similarity_partial(self):
        sig1 = list(range(50)) + list(range(100, 150))
        sig2 = list(range(25)) + list(range(100, 125)) + list(range(200, 225))
        similarity = calculate_jaccard_similarity(sig1, sig2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_cached_compute_minhash_consistency(self):
        text = "minhash caching test"
        sig1 = cached_compute_minhash(text, use_cache=False)
        sig2 = cached_compute_minhash(text, use_cache=True)
        self.assertEqual(sig1, sig2)

    def test_estimate_optimal_num_perm_default(self):
        num_perm = estimate_optimal_num_perm()
        self.assertIn(num_perm, [32, 64, 128, 256, 512])

    def test_estimate_optimal_num_perm_custom(self):
        num_perm = estimate_optimal_num_perm(0.01)
        self.assertGreaterEqual(num_perm, 10000 ** 0.5)

    def test_get_cache_key_consistency(self):
        key1 = get_cache_key("sample text", 128, 2)
        key2 = get_cache_key("sample text", 128, 2)
        self.assertEqual(key1, key2)

    def test_batch_compute_minhash(self):
        texts = ["one", "two", "three"]
        result = batch_compute_minhash(texts, num_perm=32)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(len(sig) == 32 for sig in result))

    def test_create_minhash_generator(self):
        generator = create_minhash_generator()
        sig = generator("hello world", num_perm=32)
        self.assertEqual(len(sig), 32)
        self.assertTrue(all(isinstance(i, int) for i in sig))

    @patch("pandas.read_csv")
    def test_process_csv_file(self, mock_read_csv):
        # Create temp input CSV
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as temp_input:
            writer = csv.writer(temp_input)
            writer.writerow(["id", "text"])
            writer.writerow(["1", "example text one"])
            writer.writerow(["2", "example text two"])
            input_path = temp_input.name

        # Create temp output CSV
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

        # Patch pandas.read_csv to return a DataFrame
        import pandas as pd
        df = pd.DataFrame({
            "id": [1, 2],
            "text": ["example text one", "example text two"]
        })
        mock_read_csv.return_value = [df]

        from pamola_core.utils.ops.op_result import OperationStatus

        result = process_csv_file(
            input_path=input_path,
            output_path=output_path,
            field_name="text",
            id_field="id",
            num_perm=32,
            shingle_size=2
        )

        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.assertEqual(result.metrics.get("processed_records"), 2)
        self.assertTrue(os.path.exists(output_path))

        # Clean up
        os.remove(input_path)
        os.remove(output_path)



if __name__ == '__main__':
    unittest.main()
