"""
Tests for the utils.nlp.entity_extraction module.

This module contains unit tests for the abstract base classes and concrete
implementations in the entity_extraction.py module.
"""
import unittest
from unittest.mock import patch

from pamola_core.utils.nlp.entity_extraction import (
    extract_entities,
    extract_job_positions,
    extract_organizations,
    extract_universities,
    extract_skills,
    extract_transaction_purposes,
    create_custom_entity_extractor
)



class TestEntityExtraction(unittest.TestCase):

    @patch("pamola_core.utils.nlp.entity_extraction.entity_extract_entities")
    def test_extract_entities(self, mock_extract):
        mock_extract.return_value = {"entities": []}
        result = extract_entities(["This is a test"], entity_type="job")
        mock_extract.assert_called_once()
        self.assertIn("entities", result)

    @patch("pamola_core.utils.nlp.entity_extraction.extract_entities")
    def test_extract_job_positions(self, mock_base):
        extract_job_positions(["Engineer at Google"])
        mock_base.assert_called_once_with(
            texts=["Engineer at Google"],
            entity_type="job",
            language="auto",
            dictionary_path=None,
            use_ner=True,
            seniority_detection=True
        )

    @patch("pamola_core.utils.nlp.entity_extraction.extract_entities")
    def test_extract_organizations(self, mock_base):
        extract_organizations(["Google, Inc."])
        mock_base.assert_called_once_with(
            texts=["Google, Inc."],
            entity_type="organization",
            language="auto",
            dictionary_path=None,
            use_ner=True,
            organization_type="any"
        )

    @patch("pamola_core.utils.nlp.entity_extraction.extract_entities")
    def test_extract_universities(self, mock_base):
        extract_universities(["Stanford University"])
        mock_base.assert_called_once_with(
            texts=["Stanford University"],
            entity_type="organization",
            language="auto",
            dictionary_path=None,
            use_ner=True,
            organization_type="university"
        )

    @patch("pamola_core.utils.nlp.entity_extraction.extract_entities")
    def test_extract_skills(self, mock_base):
        extract_skills(["Python and SQL"])
        mock_base.assert_called_once_with(
            texts=["Python and SQL"],
            entity_type="skill",
            language="auto",
            dictionary_path=None,
            use_ner=True,
            skill_type="technical"
        )

    @patch("pamola_core.utils.nlp.entity_extraction.extract_entities")
    def test_extract_transaction_purposes(self, mock_base):
        extract_transaction_purposes(["Payment for tuition"])
        mock_base.assert_called_once_with(
            texts=["Payment for tuition"],
            entity_type="transaction",
            language="auto",
            dictionary_path=None,
            use_ner=True
        )

    @patch("pamola_core.utils.nlp.entity_extraction.create_entity_extractor")
    def test_create_custom_entity_extractor(self, mock_create):
        mock_create.return_value = "Extractor"
        result = create_custom_entity_extractor("custom_type")
        mock_create.assert_called_once_with(
            entity_type="custom_type",
            language="auto",
            dictionary_path=None,
            match_strategy="specific_first",
            use_ner=True
        )
        self.assertEqual(result, "Extractor")



if __name__ == "__main__":
    unittest.main()
