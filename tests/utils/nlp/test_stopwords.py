"""
Tests for the utils.nlp.stopwords module.

This module contains unit tests for the abstract base classes and concrete
implementations in the stopwords.py module.
"""
import os
import tempfile
import unittest
from pathlib import Path
from pamola_core.utils.nlp.stopwords import (
    load_stopwords_from_file,
    remove_stopwords,
    save_stopwords_to_file,
    load_stopwords_from_sources,
    get_stopwords,
)

class TestStopwordsUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_words = {"hello", "world", "test"}
        self.test_file_path = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.test_words))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_stopwords_from_file(self):
        loaded = load_stopwords_from_file(self.test_file_path)
        self.assertEqual(loaded, {word.lower() for word in self.test_words})

    def test_remove_stopwords(self):
        tokens = ["Hello", "there", "World"]
        stopwords = {"hello", "world"}
        result = remove_stopwords(tokens, stop_words=stopwords)
        self.assertEqual(result, ["there"])

    def test_save_stopwords_to_file_txt(self):
        save_path = os.path.join(self.temp_dir.name, "output.txt")
        result = save_stopwords_to_file(self.test_words, save_path)
        self.assertTrue(result)
        with open(save_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.assertEqual(set(lines), {word.lower() for word in self.test_words})

    def test_load_stopwords_from_sources_with_file(self):
        sources = [self.test_file_path]
        loaded = load_stopwords_from_sources(sources)
        self.assertIn("hello", loaded)

    def test_get_stopwords_with_custom_source(self):
        stopwords = get_stopwords(custom_sources=[self.test_file_path], include_defaults=False, use_nltk=False)
        self.assertIn("hello", stopwords)

if __name__ == '__main__':
    unittest.main()
