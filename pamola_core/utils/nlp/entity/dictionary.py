"""
Generic dictionary-based entity extractor.

This module provides a universal entity extractor that works with any type
of entity dictionaries, allowing for custom entity types beyond the
specialized extractors provided in the package.
"""

import logging
from typing import Optional

from pamola_core.utils.nlp.entity.base import BaseEntityExtractor, EntityMatchResult
from pamola_core.utils.nlp.model_manager import NLPModelManager

# Configure logger
logger = logging.getLogger(__name__)

# NLP model manager instance
nlp_model_manager = NLPModelManager()


class GenericDictionaryExtractor(BaseEntityExtractor):
    """
    Generic dictionary-based entity extractor.
    """

    def __init__(self, **kwargs):
        """
        Initialize the generic dictionary extractor.

        Additional parameters:
        -----------
        entity_type : str
            Type of entities to extract (used for dictionary lookup)
        fallback_to_ner : bool
            Whether to fall back to NER if dictionary matching fails
        """
        # Get entity type first from kwargs as it's used in finding dictionary
        self.entity_type = kwargs.pop('entity_type', 'generic')

        # Initialize base class
        super().__init__(**kwargs)

        # Special parameters
        self.fallback_to_ner = kwargs.get('fallback_to_ner', True)

        # Find dictionary file for this entity type if not provided
        if not self.dictionary_path:
            self.dictionary_path = self._find_dictionary()

    def _get_entity_type(self) -> str:
        """
        Get the entity type for this extractor.

        Returns:
        --------
        str
            Entity type string
        """
        return self.entity_type

    def _find_dictionary(self) -> Optional[str]:
        """
        Find a dictionary file for the current entity type.

        Returns:
        --------
        str or None
            Path to the dictionary file if found, None otherwise
        """
        from pamola_core.utils.nlp.entity.base import find_dictionary_file

        dict_path = find_dictionary_file(self.entity_type, self.language)
        if dict_path:
            logger.info(f"Found dictionary for {self.entity_type}: {dict_path}")
            return dict_path

        logger.warning(f"No dictionary found for entity type '{self.entity_type}'")
        return None

    def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]:
        """
        Extract entities using NER as a fallback.

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
        if not self.fallback_to_ner:
            return None

        try:
            # Try to get a generic NER model
            model = nlp_model_manager.get_model('spacy', language)
            if not model:
                logger.warning(f"No NER model available for language '{language}'")
                return None

            # Process the text
            doc = model(text)

            # Extract all entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            if not entities:
                return None

            # Sort by relevant entity types
            priority_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART']

            # Filter and sort entities
            relevant_entities = [
                (text, label) for text, label in entities
                if label in priority_types
            ]

            if not relevant_entities:
                # No relevant entities found
                return None

            # Use the first relevant entity
            entity_text, entity_label = relevant_entities[0]

            # Create match result
            return EntityMatchResult(
                original_text=text,
                normalized_text=normalized_text,
                category=f"NER_{entity_label}",
                alias=f"entity_{entity_label.lower()}",
                domain="Generic",
                level=1,
                seniority="Any",
                confidence=0.5,
                method="ner",
                language=language,
                conflicts=[]
            )

        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return None