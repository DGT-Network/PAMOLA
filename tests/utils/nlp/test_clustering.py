"""
Tests for the utils.nlp.clustering module.

This module contains unit tests for the abstract base classes and concrete
implementations in the clustering.py module.
"""
import unittest
from unittest.mock import patch
from pamola_core.utils.nlp.clustering import (
    TextClusterer,
    cluster_by_similarity,
    batch_cluster_texts
)

class TestTextClusterer(unittest.TestCase):
    """Tests for the TextClusterer class."""

    def test_calculate_similarity_jaccard(self):
        tokens1 = {"apple", "banana", "cherry"}
        tokens2 = {"banana", "cherry", "date"}
        similarity = TextClusterer.calculate_similarity(tokens1, tokens2, method="jaccard")
        expected = 2 / 4  # 2 shared / 4 total
        self.assertAlmostEqual(similarity, expected)

    def test_calculate_similarity_overlap(self):
        tokens1 = {"a", "b", "c"}
        tokens2 = {"b", "c", "d"}
        similarity = TextClusterer.calculate_similarity(tokens1, tokens2, method="overlap")
        expected = 2 / 3  # 2 shared / 3 in smaller set
        self.assertAlmostEqual(similarity, expected)

    def test_calculate_similarity_cosine(self):
        tokens1 = {"x", "y"}
        tokens2 = {"y", "z"}
        similarity = TextClusterer.calculate_similarity(tokens1, tokens2, method="cosine")
        expected = 1 / (2 * 2) ** 0.5
        self.assertAlmostEqual(similarity, expected)



if __name__ == '__main__':
    unittest.main()
