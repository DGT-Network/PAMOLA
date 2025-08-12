"""
Job position entity extractor.

This module provides functionality for extracting job positions from text,
using dictionary-based matching, NER models, and specialized rules for
identifying job titles, seniority levels, and domains.
"""

import logging
from typing import Optional

from pamola_core.utils.nlp.entity.base import BaseEntityExtractor, EntityMatchResult
from pamola_core.utils.nlp.model_manager import NLPModelManager

# Configure logger
logger = logging.getLogger(__name__)

# NLP model manager instance
nlp_model_manager = NLPModelManager()


class JobPositionExtractor(BaseEntityExtractor):
    """
    Extractor for job positions and titles.
    """

    def __init__(self, **kwargs):
        """
        Initialize the job position extractor.

        All parameters from BaseEntityExtractor are supported.
        """
        super().__init__(**kwargs)

        # Job position-specific parameters
        self.seniority_detection = kwargs.get('seniority_detection', True)
        self.include_skills = kwargs.get('include_skills', True)

        # Domain terms for job positions
        self.domains = {
            'software_development': [
                'developer', 'programmer', 'coder', 'software', 'development',
                'разработчик', 'программист', 'разработка'
            ],
            'data_science': [
                'data scientist', 'data engineer', 'ml', 'machine learning',
                'data analyst', 'дата сайентист', 'инженер данных', 'аналитик данных'
            ],
            'devops': [
                'devops', 'sre', 'site reliability', 'infrastructure', 'cloud',
                'инфраструктура', 'облако', 'надежность'
            ],
            'design': [
                'designer', 'ux', 'ui', 'дизайнер', 'графический', 'design'
            ],
            'management': [
                'manager', 'director', 'head', 'lead', 'руководитель', 'директор',
                'менеджер', 'тимлид', 'team lead'
            ],
            'qa_testing': [
                'qa', 'test', 'quality', 'тестировщик', 'тестирование', 'testing'
            ],
            'security': [
                'security', 'безопасность', 'security engineer', 'инженер безопасности'
            ]
        }

        # Seniority terms
        self.seniority_terms = {
            'junior': ['junior', 'младший', 'джуниор', 'стажер', 'intern', 'trainee'],
            'middle': ['middle', 'миддл', 'regular', 'медиум'],
            'senior': ['senior', 'старший', 'сеньор', 'ведущий', 'lead', 'principal', 'staff'],
            'management': ['head', 'chief', 'manager', 'директор', 'руководитель', 'lead', 'team lead', 'тимлид']
        }

    def _get_entity_type(self) -> str:
        """
        Get the entity type for this extractor.

        Returns:
        --------
        str
            Entity type string
        """
        return "job"

    def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]:
        """
        Extract job positions using NER models.

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
            # Get job entity extractor from model manager
            extractor = nlp_model_manager.get_entity_extractor("job", language)
            if not extractor:
                logger.warning(f"No job entity extractor available for language '{language}'")
                return None

            # Extract job positions
            job_entities = extractor.extract_entities([text], language)

            # Check if we have any positions
            if not job_entities or 'positions' not in job_entities or not job_entities['positions']:
                return None

            # Get the most confident position
            positions = job_entities['positions']

            # Sort by count (descending)
            sorted_positions = sorted(positions, key=lambda x: x.get('count', 0), reverse=True)

            if not sorted_positions:
                return None

            # Get the top position
            top_position = sorted_positions[0]
            position_text = top_position.get('text', '')

            # Try to find domain based on position
            domain = self._determine_domain(position_text)

            # Try to find seniority based on position
            seniority = self._determine_seniority(position_text) if self.seniority_detection else "Any"

            # Create match result
            return EntityMatchResult(
                original_text=text,
                normalized_text=normalized_text,
                category=f"NER_JOB_{domain.upper()}",
                alias=f"job_{domain.lower()}",
                domain=domain,
                level=1,  # Default level for NER matches
                seniority=seniority,
                confidence=0.7,  # Arbitrary confidence for NER
                method="ner",
                language=language,
                conflicts=[]
            )

        except Exception as e:
            logger.error(f"Error in NER extraction for job positions: {e}")
            return None

    def _determine_domain(self, text: str) -> str:
        """
        Determine the domain for a job position.

        Parameters:
        -----------
        text : str
            Job position text

        Returns:
        --------
        str
            Domain name
        """
        text_lower = text.lower()

        for domain, terms in self.domains.items():
            for term in terms:
                if term in text_lower:
                    return domain

        return "general"

    def _determine_seniority(self, text: str) -> str:
        """
        Determine the seniority level from a job position.

        Parameters:
        -----------
        text : str
            Job position text

        Returns:
        --------
        str
            Seniority level
        """
        text_lower = text.lower()

        for level, terms in self.seniority_terms.items():
            for term in terms:
                if term in text_lower:
                    return level

        return "Any"