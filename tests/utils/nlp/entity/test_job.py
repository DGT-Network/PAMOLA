"""
Tests for the utils.nlp.entity.job module.

This module contains unit tests for the abstract base classes and concrete
implementations in the job.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

from pamola_core.utils.nlp.entity.job import (
    JobPositionExtractor
)

class TestJobPositionExtractor(unittest.TestCase):

    def test_determine_domain(self):
        # Create instance of the extractor
        extractor = JobPositionExtractor(entity_type='job', seniority_detection=True)

        # Test domain determination
        domain_software = extractor._determine_domain("Software developer")
        domain_data = extractor._determine_domain("Data scientist")

        # Assert correct domain detection
        self.assertEqual(domain_software, 'software_development')
        self.assertEqual(domain_data, 'data_science')

    def test_determine_seniority(self):
        # Create instance of the extractor
        extractor = JobPositionExtractor(entity_type='job', seniority_detection=True)

        # Test seniority detection
        seniority_junior = extractor._determine_seniority("junior developer")
        seniority_senior = extractor._determine_seniority("senior developer")
        seniority_none = extractor._determine_seniority("developer")

        # Assert correct seniority detection
        self.assertEqual(seniority_junior, 'junior')
        self.assertEqual(seniority_senior, 'senior')
        self.assertEqual(seniority_none, 'Any')

if __name__ == '__main__':
    unittest.main()
