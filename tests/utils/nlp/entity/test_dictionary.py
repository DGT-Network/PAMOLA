"""
Tests for the utils.nlp.entity.dictionary module.

This module contains unit tests for the abstract base classes and concrete
implementations in the dictionary.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

from pamola_core.utils.nlp.entity.dictionary import (
    GenericDictionaryExtractor
)

class TestGenericDictionaryExtractor(unittest.TestCase):

    @patch('pamola_core.utils.nlp.entity.base.find_dictionary_file')
    def test_find_dictionary_success(self, mock_find_dictionary_file):
        # Mocking dictionary file lookup
        mock_find_dictionary_file.return_value = '/path/to/dictionary.json'

        # Create instance of the extractor
        extractor = GenericDictionaryExtractor(entity_type='generic', fallback_to_ner=False)

        # Test dictionary lookup
        dict_path = extractor._find_dictionary()

        # Assert that the dictionary path is correct
        self.assertEqual(dict_path, '/path/to/dictionary.json')

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_ner_success(self, mock_get_model):
        # Mocking the NLP model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_model.return_value.ents = [
            MagicMock(text='John Doe', label_='PERSON'),
            MagicMock(text='New York', label_='GPE')
        ]

        # Create instance of the extractor
        extractor = GenericDictionaryExtractor(entity_type='person', fallback_to_ner=True)

        # Test NER extraction
        result = extractor._extract_with_ner("John Doe lives in New York.", "John Doe lives in New York.", "en")

        # Assert that result is not None and matches expected values
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "NER_PERSON")
        self.assertEqual(result.alias, "entity_person")
        self.assertEqual(result.original_text, "John Doe lives in New York.")

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_ner_no_entities(self, mock_get_model):
        # Mocking NER model but returning no entities
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_model.return_value.ents = []

        # Create instance of the extractor
        extractor = GenericDictionaryExtractor(entity_type='location', fallback_to_ner=True)

        # Test NER extraction with no entities
        result = extractor._extract_with_ner("The event was held in an undisclosed location.", "The event was held in an undisclosed location.", "en")

        # Assert that the result is None
        self.assertIsNone(result)

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_ner_no_model(self, mock_get_model):
        # Mocking no model available
        mock_get_model.return_value = None

        # Create instance of the extractor
        extractor = GenericDictionaryExtractor(entity_type='person', fallback_to_ner=True)

        # Test NER extraction when no model is available
        result = extractor._extract_with_ner("John Doe is a software engineer.", "John Doe is a software engineer.", "en")

        # Assert that the result is None
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
