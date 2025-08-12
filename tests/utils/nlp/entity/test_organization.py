"""
Tests for the utils.nlp.entity.organization module.

This module contains unit tests for the abstract base classes and concrete
implementations in the organization.py module.
"""
import unittest
from unittest.mock import patch, MagicMock

from pamola_core.utils.nlp.entity.organization import (
    OrganizationExtractor
)

class TestOrganizationExtractor(unittest.TestCase):

    def test_matches_organization_type(self):
        # Create instance of the extractor
        extractor = OrganizationExtractor(organization_type='company')

        # Test matches for different types of organizations
        match_company = extractor._matches_organization_type("Acme Corp", "company")
        match_university = extractor._matches_organization_type("Oxford University", "university")
        match_nonprofit = extractor._matches_organization_type("Red Cross", "nonprofit")

        # Assert correct matches
        self.assertTrue(match_company)
        self.assertTrue(match_university)
        self.assertFalse(match_nonprofit)

if __name__ == '__main__':
    unittest.main()
