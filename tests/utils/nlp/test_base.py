"""
Tests for the utils.nlp.base module.

This module contains unit tests for the abstract base classes and concrete
implementations in the base.py module.
"""

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

# Import the module to test
from pamola_core.utils.nlp.base import (
    NLPError,
    ResourceNotFoundError,
    ModelNotAvailableError,
    UnsupportedLanguageError,
    ConfigurationError,
    DependencyManager,
    normalize_language_code,
    batch_process,
    _Missing,
    CacheBase
)



class TestExceptions(unittest.TestCase):
    """Tests for the exception classes in the base module."""

    def test_exception_hierarchy(self):
        """Test that exceptions inherit from the correct base classes."""
        self.assertTrue(issubclass(NLPError, Exception))
        self.assertTrue(issubclass(ResourceNotFoundError, NLPError))
        self.assertTrue(issubclass(ModelNotAvailableError, NLPError))
        self.assertTrue(issubclass(UnsupportedLanguageError, NLPError))
        self.assertTrue(issubclass(ConfigurationError, NLPError))

    def test_exception_instantiation(self):
        """Test that exceptions can be instantiated with messages."""
        nlp_error = NLPError("NLPError")
        self.assertEqual(str(nlp_error), "NLPError")

        resource_not_found_error = ResourceNotFoundError("ResourceNotFoundError")
        self.assertEqual(str(resource_not_found_error), "ResourceNotFoundError")

        model_not_available_error = ModelNotAvailableError("ModelNotAvailableError")
        self.assertEqual(str(model_not_available_error), "ModelNotAvailableError")

        unsupported_language_error = UnsupportedLanguageError("UnsupportedLanguageError")
        self.assertEqual(str(unsupported_language_error), "UnsupportedLanguageError")

        configuration_error = ConfigurationError("ConfigurationError")
        self.assertEqual(str(configuration_error), "ConfigurationError")



class TestDependencyManager(unittest.TestCase):
    """Tests for the DependencyManager class."""

    def setUp(self):
        """Set up test fixtures."""
        DependencyManager.clear_cache()

    @patch("importlib.import_module")
    def test_check_dependency_success(self, mock_import):
        """Test check_dependency of DependencyManager."""
        mock_import.return_value = MagicMock()
        result = DependencyManager.check_dependency("some_module")
        self.assertTrue(result)
        self.assertIn("some_module", DependencyManager._dependency_cache)
        self.assertTrue(DependencyManager._dependency_cache["some_module"])

    @patch("importlib.import_module", side_effect=ImportError)
    def test_check_dependency_failure(self, mock_import):
        """Test check_dependency of DependencyManager."""
        result = DependencyManager.check_dependency("nonexistent_module")
        self.assertFalse(result)
        self.assertIn("nonexistent_module", DependencyManager._dependency_cache)
        self.assertFalse(DependencyManager._dependency_cache["nonexistent_module"])

    @patch("importlib.import_module")
    def test_get_module_success(self, mock_import):
        """Test get_module of DependencyManager."""
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        result = DependencyManager.get_module("existing_module")
        self.assertEqual(result, mock_module)

    @patch("importlib.import_module", side_effect=ImportError)
    def test_get_module_failure(self, mock_import):
        """Test get_module of DependencyManager."""
        result = DependencyManager.get_module("missing_module")
        self.assertIsNone(result)

    @patch("importlib.import_module")
    def test_check_version_success(self, mock_import):
        """Test check_version of DependencyManager."""
        mock_module = MagicMock()
        mock_module.__version__ = "1.2.3"
        mock_import.return_value = mock_module

        with patch("packaging.version.Version") as mock_version:
            mock_version.side_effect = lambda x: x  # Bypass version comparison
            is_valid, version = DependencyManager.check_version("some_module", "1.0.0", "2.0.0")
            self.assertTrue(is_valid)
            self.assertEqual(version, "1.2.3")

    @patch("importlib.import_module", side_effect=ImportError)
    def test_check_version_module_not_found(self, mock_import):
        """Test check_version of DependencyManager."""
        is_valid, version = DependencyManager.check_version("missing_module", "1.0.0")
        self.assertFalse(is_valid)
        self.assertIsNone(version)

    def test_clear_cache(self):
        """Test clear_cache of DependencyManager."""
        DependencyManager._dependency_cache = {"test": True}
        DependencyManager.clear_cache()
        self.assertEqual(len(DependencyManager._dependency_cache), 0)

    @patch.object(DependencyManager, "check_dependency", return_value=True)
    def test_get_nlp_status_all_available(self, mock_check):
        """Test get_nlp_status of DependencyManager."""
        expected: Dict[str, bool] = {
            'nltk': True,
            'spacy': True,
            'pymorphy2': True,
            'langdetect': True,
            'fasttext': True,
            'transformers': True,
            'wordcloud': True,
        }
        status = DependencyManager.get_nlp_status()
        self.assertEqual(status, expected)

    @patch.object(DependencyManager, "check_dependency", side_effect=[False, False, True])
    def test_get_best_available_module(self, mock_check):
        """Test get_best_available_module of DependencyManager."""
        prefs = ["nltk", "spacy", "pymorphy2"]
        result = DependencyManager.get_best_available_module(prefs)
        self.assertEqual(result, "pymorphy2")

    @patch.object(DependencyManager, "check_dependency", return_value=False)
    def test_get_best_available_module_none(self, mock_check):
        """Test get_best_available_module of DependencyManager."""
        prefs = ["fake1", "fake2"]
        result = DependencyManager.get_best_available_module(prefs)
        self.assertIsNone(result)



class TestNormalizeLanguageCode(unittest.TestCase):
    """Test normalize_language_code function."""

    def test_valid_known_language_codes(self):
        # Test normalization for various known language codes
        self.assertEqual(normalize_language_code('english'), 'en')
        self.assertEqual(normalize_language_code('eng'), 'en')
        self.assertEqual(normalize_language_code('en_us'), 'en')
        self.assertEqual(normalize_language_code('en_gb'), 'en')
        self.assertEqual(normalize_language_code('en-us'), 'en')
        self.assertEqual(normalize_language_code('en-gb'), 'en')
        self.assertEqual(normalize_language_code('russian'), 'ru')
        self.assertEqual(normalize_language_code('rus'), 'ru')
        self.assertEqual(normalize_language_code('ru_ru'), 'ru')
        self.assertEqual(normalize_language_code('ru-ru'), 'ru')
        self.assertEqual(normalize_language_code('german'), 'de')
        self.assertEqual(normalize_language_code('deu'), 'de')
        self.assertEqual(normalize_language_code('ger'), 'de')
        self.assertEqual(normalize_language_code('de_de'), 'de')
        self.assertEqual(normalize_language_code('de-de'), 'de')
        self.assertEqual(normalize_language_code('french'), 'fr')
        self.assertEqual(normalize_language_code('fra'), 'fr')
        self.assertEqual(normalize_language_code('fre'), 'fr')
        self.assertEqual(normalize_language_code('fr_fr'), 'fr')
        self.assertEqual(normalize_language_code('fr-fr'), 'fr')
        self.assertEqual(normalize_language_code('spanish'), 'es')
        self.assertEqual(normalize_language_code('spa'), 'es')
        self.assertEqual(normalize_language_code('es_es'), 'es')
        self.assertEqual(normalize_language_code('es-es'), 'es')

    def test_valid_two_letter_codes(self):
        # Test that valid two-letter language codes return themselves
        self.assertEqual(normalize_language_code('en'), 'en')
        self.assertEqual(normalize_language_code('ru'), 'ru')
        self.assertEqual(normalize_language_code('de'), 'de')
        self.assertEqual(normalize_language_code('fr'), 'fr')
        self.assertEqual(normalize_language_code('es'), 'es')

    def test_invalid_two_letter_codes(self):
        # Test invalid two-letter codes (e.g., not known)
        self.assertEqual(normalize_language_code('zz'), 'zz')
        self.assertEqual(normalize_language_code('xy'), 'xy')

    def test_mixed_case_inputs(self):
        # Test mixed case inputs that should be normalized
        self.assertEqual(normalize_language_code('EnGlIsH'), 'en')
        self.assertEqual(normalize_language_code('RuS'), 'ru')
        self.assertEqual(normalize_language_code('Fr_Fr'), 'fr')

    def test_fallback_to_default(self):
        # Test fallback to 'en' for unknown inputs
        self.assertEqual(normalize_language_code('italian'), 'en')
        self.assertEqual(normalize_language_code('portuguese'), 'en')
        self.assertEqual(normalize_language_code('xyz'), 'en')

    def test_edge_case_empty_input(self):
        # Test empty input should fall back to 'en'
        self.assertEqual(normalize_language_code(''), 'en')

    def test_edge_case_non_alpha_input(self):
        # Test non-alphabetical inputs should fall back to 'en'
        self.assertEqual(normalize_language_code('123'), 'en')
        self.assertEqual(normalize_language_code('!!'), 'en')



class TestBatchProcess(unittest.TestCase):
    """Test batch_process function."""

    def test_empty_items(self):
        def dummy_process_func(x, multiplier=1):
            return x * multiplier

        result = batch_process([], dummy_process_func)
        self.assertEqual(result, [])

    def test_small_dataset(self):
        def dummy_process_func(x, multiplier=1):
            return x * multiplier

        items = [1, 2, 3]
        result = batch_process(items, dummy_process_func, multiplier=2)
        expected = [2, 4, 6]
        self.assertEqual(result, expected)

    @patch("multiprocessing.Pool")
    def test_large_dataset_with_mocked_pool(self, mock_pool_class):
        items = list(range(20))
        mock_pool = MagicMock()
        mock_pool.__enter__.return_value.map.return_value = [i * 2 for i in items]
        mock_pool_class.return_value = mock_pool

        def dummy_process_func(x, multiplier=1):
            return x * multiplier

        result = batch_process(items, dummy_process_func, multiplier=2)
        expected = [i * 2 for i in items]
        self.assertEqual(result, expected)
        mock_pool_class.assert_called()

    @patch("multiprocessing.cpu_count", return_value=4)
    @patch("multiprocessing.Pool")
    def test_large_dataset_cpu_limit(self, mock_pool_class, mock_cpu_count):
        items = list(range(20))
        mock_pool = MagicMock()
        mock_pool.__enter__.return_value.map.return_value = [i + 1 for i in items]
        mock_pool_class.return_value = mock_pool

        def add_one(x, **kwargs):
            return x + 1

        result = batch_process(items, add_one)
        self.assertEqual(result, [i + 1 for i in items])
        mock_cpu_count.assert_called_once()
        mock_pool_class.assert_called_with(4)



class TestMissing(unittest.TestCase):
    """Tests for the _Missing class."""

    def setUp(self):
        """Set up test fixtures."""
        self.missing = _Missing()

    def test_repr(self):
        """Test __repr__ of _Missing."""
        self.assertEqual(repr(self.missing), "MISSING")

    def test_bool(self):
        """Test __bool__ of _Missing."""
        self.assertFalse(bool(self.missing))



# Create a mock subclass of CacheBase to test the abstract methods
class CacheBaseMock(CacheBase):
    def get(self, key: str) -> Any:
        return None  # Return None to simulate a cache miss

    def set(self, key: str, value: Any, **kwargs) -> None:
        pass  # Do nothing

    def delete(self, key: str) -> bool:
        return False  # Simulate no key deletion

    def clear(self) -> None:
        pass  # Do nothing

    def get_stats(self) -> Dict[str, Any]:
        return {"hits": 0, "misses": 0}  # Return mock stats

    def get_model_info(self, key: Optional[str] = None) -> Dict[str, Any]:
        return {"key": key, "info": "some info"} if key else {"info": "no key"}

class TestCacheBase(unittest.TestCase):
    """Tests for the CacheBase class."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize mock cache object
        self.cache = CacheBaseMock()

    def test_get_not_implemented(self):
        """Test get of CacheBase."""
        with self.assertRaises(NotImplementedError):
            CacheBase().get("some_key")

    def test_set_not_implemented(self):
        """Test set of CacheBase."""
        with self.assertRaises(NotImplementedError):
            CacheBase().set("some_key", "value")

    def test_delete_not_implemented(self):
        """Test delete of CacheBase."""
        with self.assertRaises(NotImplementedError):
            CacheBase().delete("some_key")

    def test_clear_not_implemented(self):
        """Test clear of CacheBase."""
        with self.assertRaises(NotImplementedError):
            CacheBase().clear()

    def test_get_stats_not_implemented(self):
        """Test get_stats of CacheBase."""
        with self.assertRaises(NotImplementedError):
            CacheBase().get_stats()

    def test_get_or_set_cache_miss(self):
        """Test get_or_set of CacheBase."""
        # Test that get_or_set calls the default_func and sets the value
        def default_func():
            return "computed_value"

        # Test CacheBaseMock behavior
        self.cache.get = MagicMock(return_value=None)  # Simulate cache miss
        self.cache.set = MagicMock()  # Ensure that set is called

        result = self.cache.get_or_set("some_key", default_func)

        self.cache.set.assert_called_once_with("some_key", "computed_value", **{})
        self.assertEqual(result, "computed_value")

    def test_get_or_set_cache_hit(self):
        """Test get_or_set of CacheBase."""
        # Test that get_or_set returns the cached value when available
        self.cache.get = MagicMock(return_value="cached_value")
        self.cache.set = MagicMock()  # Ensure that set is not called

        result = self.cache.get_or_set("some_key", lambda: "computed_value")

        self.cache.set.assert_not_called()  # Set should not be called
        self.assertEqual(result, "cached_value")

    def test_get_model_info_not_implemented(self):
        """Test get_model_info of CacheBase."""
        with self.assertRaises(NotImplementedError):
            CacheBase().get_model_info()

    def test_get_model_info(self):
        """Test get_model_info of CacheBase."""
        # Test the get_model_info method of the mock subclass
        result = self.cache.get_model_info("some_key")
        self.assertEqual(result, {"key": "some_key", "info": "some info"})

        result = self.cache.get_model_info()
        self.assertEqual(result, {"info": "no key"})



if __name__ == "__main__":
    unittest.main()
