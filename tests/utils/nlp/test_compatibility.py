"""
Tests for the utils.nlp.compatibility module.

This module contains unit tests for the abstract base classes and concrete
implementations in the compatibility.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

# Import the module you're testing
from pamola_core.utils.nlp.compatibility import (
    check_dependency,
    log_nlp_status,
    dependency_info,
    get_best_available_module,
    clear_dependency_cache,
    check_nlp_requirements,
    setup_nlp_resources
)

class TestNLPCompatibilityUtils(unittest.TestCase):

    @patch("pamola_core.utils.nlp.base.DependencyManager.check_dependency", return_value=True)
    def test_check_dependency(self, mock_check):
        result = check_dependency("nltk")
        mock_check.assert_called_once_with("nltk")
        self.assertTrue(result)

    @patch("pamola_core.utils.nlp.compatibility.logger")
    @patch("pamola_core.utils.nlp.base.DependencyManager.get_nlp_status", return_value={"nltk": True, "spacy": False})
    def test_log_nlp_status(self, mock_status, mock_logger):
        log_nlp_status()
        self.assertTrue(mock_logger.info.called)

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_nlp_status", return_value={"nltk": True})
    def test_dependency_info_basic(self, mock_status):
        result = dependency_info(verbose=False)
        self.assertIn("available", result)
        self.assertIn("count", result)

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_nlp_status", return_value={"nltk": True})
    @patch("pamola_core.utils.nlp.base.DependencyManager.get_module")
    @patch("pamola_core.utils.nlp.base.DependencyManager.check_version", return_value=(True, "3.0.0"))
    def test_dependency_info_verbose(self, mock_ver, mock_mod, mock_status):
        mock_mod.return_value.__file__ = "/fake/path"
        result = dependency_info(verbose=True)
        self.assertIn("versions", result)
        self.assertEqual(result["versions"]["nltk"], "3.0.0")

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_best_available_module", return_value="nltk")
    def test_get_best_available_module(self, mock_best):
        result = get_best_available_module(["nltk", "spacy"])
        mock_best.assert_called_once()
        self.assertEqual(result, "nltk")

    @patch("pamola_core.utils.nlp.base.DependencyManager.clear_cache")
    @patch("pamola_core.utils.nlp.compatibility.logger")
    def test_clear_dependency_cache(self, mock_logger, mock_clear):
        clear_dependency_cache()
        mock_clear.assert_called_once()
        mock_logger.debug.assert_called_once()

    @patch("pamola_core.utils.nlp.base.DependencyManager.check_dependency", side_effect=[False, True])
    def test_check_nlp_requirements(self, mock_check):
        requirements = {
            "tokenizer": ["nonexistent", "nltk"]
        }
        result = check_nlp_requirements(requirements)
        self.assertEqual(result["tokenizer"], True)
        self.assertEqual(mock_check.call_count, 2)



if __name__ == "__main__":
    unittest.main()
