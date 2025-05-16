"""
Tests for the utils.nlp.entity.base module.

This module contains unit tests for the abstract base classes and concrete
implementations in the base.py module.
"""
import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
import tempfile
from pathlib import Path
from typing import Optional

from pamola_core.utils.nlp.entity.base import (
    get_dictionaries_path,
    find_dictionary_file,
    BaseEntityExtractor,
    EntityMatchResult
)

class TestDictionaryPathFunctions(unittest.TestCase):

    @patch.dict(os.environ, {"PAMOLA_ENTITIES_DIR": "/mock/env/path"})
    @patch("os.path.exists", return_value=True)
    def test_get_dictionaries_path_from_env(self, mock_exists):
        path = get_dictionaries_path()
        self.assertEqual(path, Path("/mock/env/path"))
        mock_exists.assert_called_with("/mock/env/path")

    @patch("builtins.open", new_callable=mock_open, read_data='{"data_repository": "/data/repo"}')
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_get_dictionaries_path_from_config(self, mock_makedirs, mock_exists, mock_open_file):
        # Mock file lookup order: no env var, no default found, config exists
        def exists_side_effect(path):
            if "configs\\prj_config.json" in path:
                return True
            if "external_dictionaries\\entities" in path:
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        with patch.dict(os.environ, {}, clear=True):
            path = get_dictionaries_path()
            self.assertIn("external_dictionaries\\entities", str(path))

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_get_dictionaries_path_default(self, mock_makedirs, mock_exists):
        with patch.dict(os.environ, {}, clear=True):
            path = get_dictionaries_path()
            self.assertIn("resources\\entities", str(path))
            self.assertTrue(mock_makedirs.called)

    @patch("pamola_core.utils.nlp.entity.base.get_dictionaries_path", return_value=Path("/mock/dictionaries"))
    @patch("pathlib.Path.exists")
    def test_find_dictionary_file_with_language(self, mock_exists, mock_dict_path):
        # Setup: Only one file will "exist"
        def exists_side_effect():
            return True if str(Path("/mock/dictionaries/job_title_en.json")) else False

        mock_exists.side_effect = exists_side_effect

        result = find_dictionary_file("job_title", "en")
        self.assertEqual(result, str(Path("/mock/dictionaries/job_title_en.json")))

    @patch("pamola_core.utils.nlp.entity.base.get_dictionaries_path", return_value=Path("/mock/dictionaries"))
    @patch("pathlib.Path.exists")
    def test_find_dictionary_file_generic_fallback(self, mock_exists, mock_dict_path):
        # Setup: Only one file will "exist"
        def exists_side_effect():
            return True if str(Path("/mock/dictionaries/job_title_map.json")) else False

        mock_exists.side_effect = exists_side_effect

        result = find_dictionary_file("job_title", None)
        self.assertEqual(result, str(Path("/mock/dictionaries/job_title_map.json")))

    @patch("pamola_core.utils.nlp.entity.base.get_dictionaries_path", return_value=Path("/mock/dictionaries"))
    @patch("os.path.exists", return_value=False)
    def test_find_dictionary_file_not_found(self, mock_exists, mock_dict_path):
        result = find_dictionary_file("job_title", "fr")
        self.assertIsNone(result)

class DummyEntityExtractor(BaseEntityExtractor):
    def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]:
        # Simulate NER match
        return EntityMatchResult(
            original_text=text,
            normalized_text=normalized_text,
            category="TestCategory",
            confidence=0.7,
            method="ner",
            language=language
        )

    def _get_entity_type(self) -> str:
        return "test_entity"

class TestEntityMatchResult(unittest.TestCase):

    def test_to_dict_with_all_fields(self):
        result = EntityMatchResult(
            original_text="Original Text",
            normalized_text="original text",
            category="TestCategory",
            alias="test_alias",
            domain="TestDomain",
            level=3,
            seniority="Senior",
            confidence=0.85,
            method="dictionary",
            language="en",
            conflicts=["Alt1", "Alt2"],
            record_id="1234"
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict["original_text"], "Original Text")
        self.assertEqual(result_dict["matched_category"], "TestCategory")
        self.assertEqual(result_dict["matched_alias"], "test_alias")
        self.assertEqual(result_dict["matched_domain"], "TestDomain")
        self.assertEqual(result_dict["matched_level"], 3)
        self.assertEqual(result_dict["matched_seniority"], "Senior")
        self.assertEqual(result_dict["match_confidence"], 0.85)
        self.assertEqual(result_dict["match_method"], "dictionary")
        self.assertEqual(result_dict["language_detected"], "en")
        self.assertEqual(result_dict["conflict_candidates"], ["Alt1", "Alt2"])
        self.assertEqual(result_dict["record_id"], "1234")

    def test_to_dict_with_minimal_fields(self):
        result = EntityMatchResult(
            original_text="Sample",
            normalized_text="sample"
        )

        result_dict = result.to_dict()
        self.assertEqual(result_dict["original_text"], "Sample")
        self.assertEqual(result_dict["normalized_text"], "sample")
        self.assertIsNone(result_dict["matched_category"])
        self.assertEqual(result_dict["matched_alias"], None)
        self.assertEqual(result_dict["matched_domain"], "General")
        self.assertEqual(result_dict["matched_level"], 0)
        self.assertEqual(result_dict["matched_seniority"], "Any")

class TestEntityExtractor(unittest.TestCase):

    @patch("pamola_core.utils.nlp.cache.get_cache")
    @patch("pamola_core.utils.nlp.category_matching.CategoryDictionary.from_file")
    def test_load_dictionary_success(self, mock_from_file, mock_get_cache):
        mock_dict = MagicMock()
        mock_dict.dictionary = {"Test": {}}
        mock_dict.hierarchy = {"Test": []}
        mock_from_file.return_value = mock_dict

        extractor = DummyEntityExtractor(dictionary_path="fake_path.json", use_cache=False)
        success = extractor.load_dictionary("fake_path.json")

        self.assertTrue(success)
        self.assertEqual(extractor.category_dictionary, {"Test": {}})
        self.assertEqual(extractor.hierarchy, {"Test": []})

    @patch("pamola_core.utils.nlp.entity.base.find_dictionary_file")
    @patch("pamola_core.utils.nlp.category_matching.CategoryDictionary.from_file")
    def test_ensure_dictionary_loaded_fallback(self, mock_from_file, mock_find_dict):
        mock_find_dict.return_value = "found_dict.json"
        mock_dict = MagicMock()
        mock_dict.dictionary = {"Test": {}}
        mock_dict.hierarchy = {"Test": []}
        mock_from_file.return_value = mock_dict

        extractor = DummyEntityExtractor(language="en", use_cache=False)
        loaded = extractor.ensure_dictionary_loaded("test_entity")

        self.assertTrue(loaded)
        self.assertEqual(extractor.dictionary_path, "found_dict.json")

    def test_extract_entities_with_ner_fallback(self):
        extractor = DummyEntityExtractor(use_ner=True, use_cache=False)
        result = extractor.extract_entities(["Some random entity"])

        self.assertEqual(result["summary"]["matched_count"], 1)
        self.assertEqual(result["entities"][0]["matched_category"], "TestCategory")
        self.assertEqual(result["entities"][0]["match_method"], "ner")

    def test_empty_texts_returns_empty_result(self):
        extractor = DummyEntityExtractor()
        result = extractor.extract_entities([])

        self.assertEqual(result["summary"]["total_texts"], 0)
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["unresolved"], [])

    @patch("pamola_core.utils.nlp.category_matching.CategoryDictionary.get_best_match")
    @patch("pamola_core.utils.nlp.category_matching.CategoryDictionary.get_category_info")
    def test_extract_entities_with_dictionary_match(self, mock_get_info, mock_best_match):
        extractor = DummyEntityExtractor(use_ner=False, use_cache=False)
        extractor.category_dictionary = {"test": {}}
        extractor.hierarchy = {}

        mock_get_info.return_value = {
            "alias": "test_alias",
            "domain": "test_domain",
            "level": 1,
            "seniority": "Senior"
        }
        mock_best_match.return_value = ("test", 0.9, [])

        result = extractor.extract_entities(["Test text"])

        self.assertEqual(result["summary"]["dictionary_matches"], 1)
        self.assertEqual(result["entities"][0]["matched_category"], "test")
        self.assertEqual(result["entities"][0]["matched_alias"], "test_alias")

if __name__ == "__main__":
    unittest.main()
