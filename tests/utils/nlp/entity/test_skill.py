"""
Tests for the utils.nlp.entity.skill module.

This module contains unit tests for the abstract base classes and concrete
implementations in the skill.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

from pamola_core.utils.nlp.entity.skill import (
    SkillExtractor
)

class TestSkillExtractor(unittest.TestCase):

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_keyword_matching(self, mock_get_model):
        # Simulate keyword matching
        extractor = SkillExtractor(skill_type='technical')
        text = "I am proficient in Python, Java, and TypeScript."

        # Test skill extraction
        result = extractor._extract_with_ner(text, text, "en")

        # Assert that the result is not None and that it matches the expected skill category
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "SKILL_PROGRAMMING_LANGUAGES")
        self.assertEqual(result.alias, "skill_programming_languages")
        self.assertEqual(result.confidence, 1.0)  # Confidence should be high for keyword-based extraction

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_ner_success(self, mock_get_model):
        # Simulate spaCy model for NER extraction
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Simulate spaCy entities
        mock_model.return_value.ents = [MagicMock(text="Python", label_="SKILL")]

        # Create SkillExtractor instance
        extractor = SkillExtractor(skill_type='technical')
        text = "I have experience with Python and JavaScript."
        normalized_text = "I have 5 years experience"

        # Test skill extraction using NER
        result = extractor._extract_with_ner(text, normalized_text, "en")

        # Assert that the result is not None and that it matches the expected category
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "NER_SKILL")
        self.assertEqual(result.alias, "skill_general")
        self.assertEqual(result.confidence, 0.6)  # Lower confidence for NER-based extraction

    @patch('pamola_core.utils.nlp.model_manager.NLPModelManager.get_model')
    def test_extract_with_ner_no_model(self, mock_get_model):
        # Simulate no NER model available
        mock_get_model.return_value = None

        # Create SkillExtractor instance
        extractor = SkillExtractor(skill_type='technical')
        text = "I have experience with Java and C++."
        normalized_text = "I have 5 years experience"

        # Test skill extraction with no model
        result = extractor._extract_with_ner(text, normalized_text, "en")

        # Assert that result is None due to no model being available
        self.assertIsNone(result)

    def test_extract_skills_from_text(self):
        # Create SkillExtractor instance
        extractor = SkillExtractor(skill_type='technical')

        # Test skill extraction from text
        skills = extractor._extract_skills_from_text("I am skilled in Python, JavaScript, and SQL.")

        # Assert that the correct skills are extracted
        self.assertIn("python", skills)
        self.assertIn("javascript", skills)
        self.assertIn("sql", skills)

    def test_determine_skill_category(self):
        # Create SkillExtractor instance
        extractor = SkillExtractor(skill_type='technical')

        # Test skill categorization
        skills = ["python", "java", "typescript"]
        category, confidence = extractor._determine_skill_category(skills)

        # Assert the correct category is chosen
        self.assertEqual(category, "programming_languages")
        self.assertEqual(confidence, 1.0)  # Confidence should be 1.0 as all skills belong to the same category

    def test_determine_skill_category_empty(self):
        # Create SkillExtractor instance
        extractor = SkillExtractor(skill_type='technical')

        # Test skill categorization with no skills
        category, confidence = extractor._determine_skill_category([])

        # Assert the default "general" category is returned with low confidence
        self.assertEqual(category, "general")
        self.assertEqual(confidence, 0.0)

    def test_matches_skills_in_categories(self):
        # Create SkillExtractor instance
        extractor = SkillExtractor(skill_type='technical')

        # Test the skill categorization logic
        skills = ['python', 'mysql', 'docker']
        category, confidence = extractor._determine_skill_category(skills)

        # Assert that the category is correct
        self.assertEqual(category, "programming_languages")
        self.assertGreater(confidence, 0)

if __name__ == '__main__':
    unittest.main()
