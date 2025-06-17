"""
Tests for the utils.nlp.category_matching module.

This module contains unit tests for the abstract base classes and concrete
implementations in the category_matching.py module.
"""
import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import json

from pamola_core.utils.nlp.category_matching import (
    CategoryDictionary,
    get_best_match,
    analyze_hierarchy
)

class TestCategoryDictionary(unittest.TestCase):
    """Tests for the CategoryDictionary class."""

    def setUp(self):
        self.sample_dict = {
            "IT Support": {
                "keywords": ["support", "helpdesk"],
                "level": 2,
                "domain": "Tech"
            },
            "Customer Service": {
                "keywords": ["service", "customer"],
                "level": 1,
                "domain": "General"
            }
        }

        self.sample_hierarchy = {
            "IT Support": {
                "keywords": ["support", "helpdesk"],
                "level": 2,
                "domain": "Tech",
                "children": []
            },
            "Customer Service": {
                "keywords": ["service", "customer"],
                "level": 1,
                "domain": "General",
                "children": []
            }
        }

    def test_get_best_match_specific_first(self):
        cd = CategoryDictionary(self.sample_dict)
        category, score, conflicts = cd.get_best_match("I need IT support", "specific_first")
        self.assertEqual(category, "IT Support")
        self.assertGreater(score, 0)
        self.assertIsInstance(conflicts, list)

    def test_get_best_match_domain_prefer(self):
        cd = CategoryDictionary(self.sample_dict)
        category, score, conflicts = cd.get_best_match("customer helpdesk", "domain_prefer")
        self.assertIn(category, ["IT Support", "Customer Service"])

    def test_get_fallback_category(self):
        cd = CategoryDictionary({
            "Unclassified": {},
            "Other": {"level": 0, "domain": "General"}
        })
        self.assertEqual(cd.get_fallback_category(), "Unclassified")

        cd = CategoryDictionary({
            "Other": {"level": 0, "domain": "General"}
        })
        self.assertEqual(cd.get_fallback_category(), "Other")

        cd = CategoryDictionary({})
        self.assertIsNone(cd.get_fallback_category())

    def test_analyze_hierarchy(self):
        cd = CategoryDictionary(dictionary_data=self.sample_dict, hierarchy_data=self.sample_hierarchy)
        result = cd.analyze_hierarchy()
        self.assertIn("total_categories", result)
        self.assertEqual(result["total_categories"], 2)



    def test_global_cached_best_match(self):
        text = "Need customer service"
        result = get_best_match(text, self.sample_dict)
        self.assertEqual(result[0], "Customer Service")

    def test_global_analyze_hierarchy(self):
        result = analyze_hierarchy(self.sample_hierarchy)
        self.assertEqual(result["total_categories"], 2)



if __name__ == '__main__':
    unittest.main()
