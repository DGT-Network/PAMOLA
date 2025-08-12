"""
Tests for the utils.nlp.tokenization_ext module.

This module contains unit tests for the abstract base classes and concrete
implementations in the tokenization_ext.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

from pamola_core.utils.nlp.tokenization_ext import (
    load_ngram_dictionary,
    extract_ngrams,
    extract_multi_ngrams,
    extract_keyphrases,
    filter_tokens_by_pos,
    extract_collocations,
    extract_sentiment_words,
    tokenize_and_analyze,
    NGramExtractor,
    AdvancedTextProcessor
)

class TestTokenizationExt(unittest.TestCase):

    def setUp(self):
        self.sample_text = "The quick brown fox jumps over the lazy dog."
        self.tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        self.language = 'en'

    @patch('pamola_core.utils.nlp.tokenization_helpers.load_ngram_dictionary')
    def test_load_ngram_dictionary_returns_set(self, mock_load):
        # Setup: mock return
        mock_load.return_value = {'machine learning', 'natural language'}

        # Call the function
        result = load_ngram_dictionary(sources="mock_source", language="en")

        # Assert
        self.assertIsInstance(result, set)
        self.assertIn("machine learning", result)
        self.assertIn("natural language", result)

        # Check that the mock was called correctly
        mock_load.assert_called_once_with("mock_source", "en")

    @patch('pamola_core.utils.nlp.tokenization_helpers.load_ngram_dictionary')
    def test_load_ngram_dictionary_empty(self, mock_load):
        mock_load.return_value = set()
        result = load_ngram_dictionary(sources=None, language=None)
        self.assertEqual(result, set())

    @patch('pamola_core.utils.nlp.tokenization_helpers.load_ngram_dictionary')
    def test_load_ngram_dictionary_multiple_sources(self, mock_load):
        mock_load.return_value = {'data science', 'big data'}
        result = load_ngram_dictionary(sources=["source1", "source2"], language="en")
        self.assertIn('data science', result)
        self.assertEqual(mock_load.call_args[0][0], ["source1", "source2"])
        self.assertEqual(mock_load.call_args[0][1], "en")

    def test_extract_ngrams(self):
        result = extract_ngrams(self.tokens, n=2)
        self.assertIn("quick brown", result)
        self.assertEqual(len(result), len(self.tokens) - 1)

    def test_extract_multi_ngrams(self):
        result = extract_multi_ngrams(self.tokens, min_n=2, max_n=3)
        self.assertTrue(any("quick brown fox" in ngram for ngram in result))
        self.assertGreater(len(result), 0)

    def test_extract_keyphrases(self):
        result = extract_keyphrases(self.sample_text, language=self.language)
        self.assertIsInstance(result, list)
        if result:
            self.assertIn('phrase', result[0])
            self.assertIn('frequency', result[0])

    def test_filter_tokens_by_pos_basic(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = filter_tokens_by_pos(text, pos_tags=["NOUN", "ADJ"], language="en")
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(t, str) for t in result))
        self.assertIn("fox", result)
        self.assertIn("dog", result)

    def test_filter_tokens_by_pos_empty_text(self):
        result = filter_tokens_by_pos("", pos_tags=["NOUN"], language="en")
        self.assertEqual(result, [])

    @patch("pamola_core.utils.nlp.base.DependencyManager.check_dependency")
    def test_filter_tokens_by_pos_spacy_missing(self, mock_check_dep):
        mock_check_dep.return_value = False
        result = filter_tokens_by_pos("Some sample text", pos_tags=["NOUN"], language="en")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)  # Falls back to tokenizer

    def test_extract_collocations_basic(self):
        text = "The cat sat on the mat. The cat slept on the mat."
        results = extract_collocations(text, language="en", min_freq=1)
        self.assertIsInstance(results, list)
        self.assertTrue(all("collocation" in r for r in results))
        self.assertTrue(any("cat sat" in r["collocation"] or "cat slept" in r["collocation"] for r in results))

    def test_extract_collocations_empty_text(self):
        results = extract_collocations("", language="en")
        self.assertEqual(results, [])

    @patch("pamola_core.utils.nlp.base.DependencyManager.check_dependency")
    def test_extract_collocations_nltk_missing(self, mock_check_dep):
        mock_check_dep.return_value = False
        text = "Apples and oranges are fruits. Apples and oranges are healthy."
        results = extract_collocations(text, language="en", min_freq=1)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertTrue(all("collocation" in r for r in results))

    def test_extract_sentiment_words(self):
        text = "I love sunny days but hate rainy nights."
        result = extract_sentiment_words(text, language='en')
        self.assertIn('positive', result)
        self.assertIn('negative', result)

    def test_tokenize_and_analyze(self):
        result = tokenize_and_analyze(self.sample_text, language=self.language, include_ngrams=True)
        self.assertIn('tokens', result)

    def test_ngram_extractor_class(self):
        extractor = NGramExtractor()
        result = extractor.extract_ngrams(self.tokens, n=2)
        self.assertIn("quick brown", result)

    def test_advanced_text_processor_basic(self):
        processor = AdvancedTextProcessor(language='en')
        tokens = processor.tokenize(self.sample_text)
        self.assertIn("quick", tokens)

    def test_advanced_text_processor_keyphrases(self):
        processor = AdvancedTextProcessor(language='en')
        result = processor.extract_keyphrases(self.sample_text)
        self.assertIsInstance(result, list)
        if result:
            self.assertIn('phrase', result[0])




if __name__ == '__main__':
    unittest.main()
