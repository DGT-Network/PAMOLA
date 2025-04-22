"""
Organization entity extractor.

This module provides functionality for extracting organization names from text,
including companies, educational institutions, government agencies, etc.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Union, Tuple

from pamola_core.utils.nlp.entity.base import BaseEntityExtractor, EntityMatchResult
from pamola_core.utils.nlp.model_manager import NLPModelManager

# Configure logger
logger = logging.getLogger(__name__)

# NLP model manager instance
nlp_model_manager = NLPModelManager()


class OrganizationExtractor(BaseEntityExtractor):
    """
    Extractor for organization names.
    """

    def __init__(self, **kwargs):
        """
        Initialize the organization extractor.

        Additional parameters:
        -----------
        organization_type : str
            Type of organizations to extract ('company', 'university', 'government', etc.)
        include_abbreviations : bool
            Whether to include abbreviations in extraction
        """
        super().__init__(**kwargs)

        # Organization-specific parameters
        self.organization_type = kwargs.get('organization_type', 'any')
        self.include_abbreviations = kwargs.get('include_abbreviations', True)

        # Common terms for different organization types
        self.organization_terms = {
            'company': [
                'ltd', 'inc', 'llc', 'corp', 'corporation', 'company', 'plc', 'gmbh',
                'ооо', 'зао', 'оао', 'ао', 'компания', 'корпорация'
            ],
            'university': [
                'university', 'college', 'institute', 'academy', 'school',
                'университет', 'колледж', 'институт', 'академия', 'школа', 'вуз'
            ],
            'government': [
                'ministry', 'department', 'agency', 'authority', 'commission',
                'министерство', 'департамент', 'агентство', 'комитет', 'служба'
            ],
            'nonprofit': [
                'foundation', 'association', 'society', 'trust', 'charity', 'ngo',
                'фонд', 'ассоциация', 'общество', 'траст', 'благотворительность', 'нко'
            ]
        }

    def _get_entity_type(self) -> str:
        """
        Get the entity type for this extractor.

        Returns:
        --------
        str
            Entity type string
        """
        return "organization"

    def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]:
        """
        Extract organization names using NER models.

        Parameters:
        -----------
        text : str
            Original text
        normalized_text : str
            Normalized text
        language : str
            Language of the text

        Returns:
        --------
        EntityMatchResult or None
            Match result if found, None otherwise
        """
        try:
            # Try to use entity extractor from model manager first
            extractor = nlp_model_manager.get_entity_extractor("ner", language)

            if not extractor:
                # If no specialized extractor is available, try to load a spaCy model
                model = nlp_model_manager.get_model('spacy', language)
                if not model:
                    logger.warning(f"No NER model available for language '{language}'")
                    return None

                # Use the spaCy model directly
                doc = model(text)

                # Extract organizations
                orgs = [ent.text for ent in doc.ents if ent.label_ in ('ORG', 'ORGANIZATION', 'FAC', 'GPE')]

                if not orgs:
                    return None

                # Use the first detected organization
                org_name = orgs[0]

            else:
                # Use the specialized entity extractor
                entities = extractor.extract_entities([text], language)

                # Check for organizations
                if not entities or 'organizations' not in entities or not entities['organizations']:
                    return None

                # Sort by count (descending)
                organizations = sorted(entities['organizations'], key=lambda x: x.get('count', 0), reverse=True)

                if not organizations:
                    return None

                # Get the top organization
                org_name = organizations[0].get('text', '')

            # Filter by organization type if specified
            if self.organization_type != 'any' and not self._matches_organization_type(org_name,
                                                                                       self.organization_type):
                return None

            # Create match result
            category = f"NER_{self.organization_type.upper()}"
            alias = f"org_{self.organization_type.lower()}"

            return EntityMatchResult(
                original_text=text,
                normalized_text=normalized_text,
                category=category,
                alias=alias,
                domain="Organization",
                level=1,  # Default level for NER matches
                seniority="Any",
                confidence=0.7,  # Arbitrary confidence for NER
                method="ner",
                language=language,
                conflicts=[]
            )

        except Exception as e:
            logger.error(f"Error in NER extraction for organizations: {e}")
            return None

    def _matches_organization_type(self, text: str, org_type: str) -> bool:
        """
        Check if an organization name matches a specific type.

        Parameters:
        -----------
        text : str
            Organization name
        org_type : str
            Organization type to check

        Returns:
        --------
        bool
            True if the organization matches the type, False otherwise
        """
        if org_type == 'any':
            return True

        text_lower = text.lower()

        # Check if the text contains any of the terms for the specified type
        type_terms = self.organization_terms.get(org_type, [])

        for term in type_terms:
            if term in text_lower:
                return True

        return False