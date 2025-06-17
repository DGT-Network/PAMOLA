"""
Tests for the utils.nlp.entity.transaction module.

This module contains unit tests for the abstract base classes and concrete
implementations in the transaction.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

from pamola_core.utils.nlp.entity.transaction import (
    TransactionPurposeExtractor
)

class TestTransactionPurposeExtractor(unittest.TestCase):

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_pattern_matching(self, mock_get_model):
        # Simulate pattern matching for a payment transaction
        extractor = TransactionPurposeExtractor()
        text = ("The payment for your invoice #12345 is complete. платеж оплата счет")

        # Test pattern matching
        result = extractor._extract_with_ner(text, text, "en")

        # Assert that the result is not None and matches the expected category
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "TRANSACTION_PAYMENT")
        self.assertEqual(result.alias, "transaction_payment")

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_fallback_ner(self, mock_get_model):
        # Simulate spaCy NER extraction as a fallback
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Simulate spaCy entities (organizations, products, etc.)
        mock_model.return_value.ents = [MagicMock(text="Invoice", label_="PRODUCT")]

        extractor = TransactionPurposeExtractor()
        text = "The invoice for your payment is ready."

        # Test fallback NER extraction
        result = extractor._extract_with_ner(text, text, "en")

        # Assert that the result is not None and matches the expected category
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "NER_ORGANIZATION")
        self.assertEqual(result.alias, "transaction_organization")
        self.assertEqual(result.confidence, 0.5)  # Lower confidence for fallback NER-based extraction

    def test_clean_transaction_text(self):
        # Test cleaning of account numbers and dates
        extractor = TransactionPurposeExtractor(remove_account_numbers=True, remove_dates=True)

        text = "Payment made to account 12345678 on 12/12/2023."
        cleaned_text = extractor._clean_transaction_text(text)

        # Assert that account numbers and dates are removed
        self.assertNotIn("12345678", cleaned_text)
        self.assertNotIn("12/12/2023", cleaned_text)

    def test_determine_transaction_category(self):
        # Test category determination
        extractor = TransactionPurposeExtractor()

        text = "The salary payment has been processed."
        category, confidence = extractor._determine_transaction_category(text)

        # Assert that the correct category and confidence are returned
        self.assertEqual(category, "payment")
        self.assertGreater(confidence, 0)

    def test_determine_transaction_category_no_matches(self):
        # Test category determination when no matches are found
        extractor = TransactionPurposeExtractor()

        text = "This is a general transaction."
        category, confidence = extractor._determine_transaction_category(text)

        # Assert that the category is "general" when no matches are found
        self.assertEqual(category, "general")
        self.assertEqual(confidence, 0.0)

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_no_matches(self, mock_get_model):
        # Simulate no NER model and no pattern match
        mock_get_model.return_value = None

        extractor = TransactionPurposeExtractor()
        text = "No transaction purpose found."

        # Test that no matches result in None
        result = extractor._extract_with_ner(text, text, "en")
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
