"""
Tests for the utils.nlp.tokenization_helpers module.

This module contains unit tests for the abstract base classes and concrete
implementations in the tokenization_helpers.py module.
"""
import unittest
from unittest.mock import patch, mock_open, MagicMock
import json

from pamola_core.utils.nlp.tokenization_helpers import (
    load_tokenization_config,
    load_synonym_dictionary,
    load_ngram_dictionary,
    ProgressTracker
)


class TestTokenizationHelpers(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("os.path.exists", return_value=True)
    def test_load_tokenization_config(self, mock_exists, mock_file):
        config = load_tokenization_config(config_sources="dummy_path.json")
        self.assertEqual(config, {"key": "value"})
        mock_file.assert_called_with("dummy_path.json", "r", encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open, read_data='{"term": ["syn1", "syn2"]}')
    @patch("os.path.exists", return_value=True)
    def test_load_synonym_dictionary_dict_format(self, mock_exists, mock_file):
        synonyms = load_synonym_dictionary(sources="dummy_path.json")
        self.assertEqual(synonyms, {"term": ["syn1", "syn2"]})
        mock_file.assert_called_with("dummy_path.json", "r", encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps([
        {"canonical": "term", "synonyms": ["syn1", "syn2"]}
    ]))
    @patch("os.path.exists", return_value=True)
    def test_load_synonym_dictionary_list_format(self, mock_exists, mock_file):
        synonyms = load_synonym_dictionary(sources="dummy_path.json")
        self.assertEqual(synonyms, {"term": ["syn1", "syn2"]})

    @patch("builtins.open", new_callable=mock_open, read_data='["ng1", "ng2"]')
    @patch("os.path.exists", return_value=True)
    def test_load_ngram_dictionary_json_list(self, mock_exists, mock_file):
        ngrams = load_ngram_dictionary(sources="dummy_path.json")
        self.assertSetEqual(ngrams, {"ng1", "ng2"})

    @patch("builtins.open", new_callable=mock_open, read_data='{"ngrams": ["ng1", "ng2"]}')
    @patch("os.path.exists", return_value=True)
    def test_load_ngram_dictionary_json_dict(self, mock_exists, mock_file):
        ngrams = load_ngram_dictionary(sources="dummy_path.json")
        self.assertSetEqual(ngrams, {"ng1", "ng2"})

    @patch("builtins.open", new_callable=mock_open, read_data="ng1\nng2\n")
    @patch("os.path.exists", return_value=True)
    def test_load_ngram_dictionary_txt(self, mock_exists, mock_file):
        ngrams = load_ngram_dictionary(sources="dummy_path.txt")
        self.assertSetEqual(ngrams, {"ng1", "ng2"})

    @patch("os.path.exists", return_value=False)
    def test_load_tokenization_config_file_not_found(self, mock_exists):
        config = load_tokenization_config(config_sources="nonexistent.json")
        self.assertEqual(config, {})

    @patch("os.path.exists", return_value=False)
    def test_load_synonym_dictionary_file_not_found(self, mock_exists):
        synonyms = load_synonym_dictionary(sources="nonexistent.json")
        self.assertEqual(synonyms, {})

    @patch("os.path.exists", return_value=False)
    def test_load_ngram_dictionary_file_not_found(self, mock_exists):
        ngrams = load_ngram_dictionary(sources="nonexistent.txt")
        self.assertEqual(ngrams, set())



class TestProgressTracker(unittest.TestCase):

    @patch("pamola_core.utils.nlp.tokenization_helpers.logger")
    @patch.dict("sys.modules", {"tqdm": None})  # Simulate no tqdm installed
    def test_update_logging_without_tqdm(self, mock_logger):
        tracker = ProgressTracker(total=10, description="Testing", show=True)
        tracker.update()  # Should log at 10% (1/10)

        # Check logger.info was called at least once with percentage log
        mock_logger.info.assert_any_call("Testing: 10% (1/10)")

        tracker.update(8)  # Total progress: 9/10
        tracker.update()   # Final increment to 10/10, should log 100%
        mock_logger.info.assert_any_call("Testing: 100% (10/10)")

        tracker.close()
        mock_logger.info.assert_any_call("Testing: Completed (10/10)")

    @patch("tqdm.tqdm")
    def test_with_tqdm_available(self, mock_tqdm_cls):
        mock_tqdm = MagicMock()
        mock_tqdm_cls.return_value = mock_tqdm

        tracker = ProgressTracker(total=5, description="With tqdm", show=True)

        tracker.update(1)
        tracker.update(2)
        tracker.close()

        mock_tqdm.update.assert_any_call(1)
        mock_tqdm.update.assert_any_call(2)
        mock_tqdm.close.assert_called_once()

    def test_without_show(self):
        tracker = ProgressTracker(total=5, description="No show", show=False)
        tracker.update(1)
        self.assertEqual(tracker.current, 1)
        tracker.close()
        self.assertEqual(tracker.current, 1)



if __name__ == '__main__':
    unittest.main()
