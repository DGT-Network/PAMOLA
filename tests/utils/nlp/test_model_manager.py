"""
Tests for the utils.nlp.model_manager module.

This module contains unit tests for the abstract base classes and concrete
implementations in the model_manager.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

from pamola_core.utils.nlp.model_manager import (
    NLPModelManager,
    ModelLoadError
)



class TestNLPModelManager(unittest.TestCase):

    def setUp(self):
        # Clear the singleton before each test
        NLPModelManager._instance = None
        self.manager = NLPModelManager()

    def test_singleton_behavior(self):
        m1 = NLPModelManager()
        m2 = NLPModelManager()
        self.assertIs(m1, m2, "NLPModelManager should be a singleton")

    @patch('pamola_core.utils.nlp.base.DependencyManager.check_dependency')
    def test_check_available_libraries(self, mock_check_dependency):
        # Simulate spaCy and transformers are available
        mock_check_dependency.side_effect = lambda x: x in ['spacy', 'transformers']
        self.manager._nlp_libraries = self.manager._check_available_libraries()
        self.assertIn('spacy', self.manager._nlp_libraries)
        self.assertIn('transformers', self.manager._nlp_libraries)

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager._load_spacy_model')
    @patch('pamola_core.utils.nlp.model_manager.get_cache')
    def test_get_model_with_cache(self, mock_get_cache, mock_load_model):
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        mock_cache.get.return_value = 'cached_model'

        # Reinitialize with mocked cache
        NLPModelManager._instance = None
        manager = NLPModelManager()

        model = manager.get_model('spacy', 'en')
        self.assertEqual(model, 'cached_model')
        mock_load_model.assert_not_called()

    @patch('pamola_core.utils.nlp.base.DependencyManager.check_dependency')
    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager._load_spacy_model')
    @patch('pamola_core.utils.nlp.model_manager.get_cache')
    def test_get_model_not_cached(self, mock_get_cache, mock_load_model, mock_check_dependency):
        mock_check_dependency.side_effect = lambda dep: dep == 'spacy'
        mock_load_model.return_value = 'mock_spacy_model'

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        NLPModelManager._instance = None
        manager = NLPModelManager()

        model = manager.get_model('spacy', 'en')
        self.assertEqual(model, 'mock_spacy_model')
        mock_load_model.assert_called_once()
        mock_cache.set.assert_called_once()

    def test_library_supported(self):
        self.manager._nlp_libraries = {'spacy'}
        self.assertTrue(self.manager._library_supported('spacy'))
        self.assertFalse(self.manager._library_supported('transformers'))

    def test_get_supported_model_types(self):
        self.manager._nlp_libraries = {'spacy', 'transformers'}
        supported = self.manager.get_supported_model_types()
        self.assertIn('spacy', supported)
        self.assertIn('transformers', supported)
        self.assertIn('entity_extractor', supported)

    def test_set_max_models_validation(self):
        self.manager.set_max_models(0)
        self.assertEqual(self.manager._max_models, 1)

    def test_set_model_expiry_validation(self):
        self.manager.set_model_expiry(10)
        self.assertEqual(self.manager._model_expiry, 60)

    @patch('psutil.Process')
    def test_get_memory_stats_success(self, mock_psutil_process):
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024  # 1MB
        mock_memory_info.vms = 2048 * 1024  # 2MB

        mock_process = MagicMock()
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 5.5

        mock_psutil_process.return_value = mock_process

        stats = self.manager.get_memory_stats()
        self.assertIn('rss_mb', stats)
        self.assertEqual(stats['rss_mb'], 1.0)

    def test_check_model_availability_unknown_type(self):
        result = self.manager.check_model_availability('unknown_type', 'en')
        self.assertFalse(result['library_available'])
        self.assertFalse(result['potentially_available'])

    def test_unload_model_and_clear_models(self):
        mock_cache = MagicMock()
        self.manager._model_cache = mock_cache

        self.manager.unload_model('some_key')
        mock_cache.delete.assert_called_once_with('some_key')

        self.manager.clear_models()
        mock_cache.clear.assert_called_once()


if __name__ == '__main__':
    unittest.main()
