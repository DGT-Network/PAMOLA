"""
Entity extraction package for the NLP module.

This package provides functionality for extracting various types of entities from text,
including job positions, organizations, skills, and transaction purposes. It supports
different extraction methods such as dictionary-based matching, NER models, and
clustering for unresolved entities.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type

from pamola_core.utils.nlp.entity.base import BaseEntityExtractor
from pamola_core.utils.nlp.entity.job import JobPositionExtractor
from pamola_core.utils.nlp.entity.organization import OrganizationExtractor
from pamola_core.utils.nlp.entity.skill import SkillExtractor
from pamola_core.utils.nlp.entity.transaction import TransactionPurposeExtractor
from pamola_core.utils.nlp.entity.dictionary import GenericDictionaryExtractor

# Configure logger
logger = logging.getLogger(__name__)

# Registry of available entity extractors
_EXTRACTOR_REGISTRY = {
    "job": JobPositionExtractor,
    "job_position": JobPositionExtractor,
    "organization": OrganizationExtractor,
    "university": OrganizationExtractor,  # Alias to organization with university filter
    "skill": SkillExtractor,
    "transaction": TransactionPurposeExtractor,
    "generic": GenericDictionaryExtractor,
    "dictionary": GenericDictionaryExtractor,  # Alias
}


def create_entity_extractor(
    entity_type: str,
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    min_confidence: float = 0.5,
    **kwargs
) -> BaseEntityExtractor:
    """
    Create an entity extractor based on the specified type.

    Parameters:
    -----------
    entity_type : str
        Type of entity extractor to create ("job", "organization", "skill", "transaction", "generic")
    language : str
        Language code or "auto" for automatic detection
    dictionary_path : str, optional
        Path to the dictionary file for the entity type
    match_strategy : str
        Strategy for resolving matches ("specific_first", "domain_prefer", "alias_only", "user_override")
    use_ner : bool
        Whether to use NER models if dictionary match fails
    min_confidence : float
        Minimum confidence threshold for entity recognition
    **kwargs
        Additional parameters for specific extractors

    Returns:
    --------
    BaseEntityExtractor
        An instance of the appropriate entity extractor
    """
    # Normalize entity type
    entity_type = entity_type.lower()

    # Check if entity type is supported
    if entity_type not in _EXTRACTOR_REGISTRY:
        logger.warning(f"Unknown entity type: '{entity_type}', using generic dictionary extractor")
        entity_type = "generic"

    # Get extractor class
    extractor_class = _EXTRACTOR_REGISTRY[entity_type]

    # Special case for university
    if entity_type == "university":
        kwargs["organization_type"] = "university"

    # Create and return the extractor
    return extractor_class(
        language=language,
        dictionary_path=dictionary_path,
        match_strategy=match_strategy,
        use_ner=use_ner,
        min_confidence=min_confidence,
        **kwargs
    )


def extract_entities(
    texts: List[str],
    entity_type: str = "generic",
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract entities from a list of texts.

    This is a convenience function that creates an appropriate extractor
    and calls its extract_entities method.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to process
    entity_type : str
        Type of entities to extract
    language : str
        Language code or "auto" for detection
    dictionary_path : str, optional
        Path to the dictionary file
    match_strategy : str
        Strategy for resolving matches
    use_ner : bool
        Whether to use NER models if dictionary match fails
    **kwargs
        Additional parameters for the extractor

    Returns:
    --------
    Dict[str, Any]
        Extraction results containing entities, categories, and statistics
    """
    # Create an extractor
    extractor = create_entity_extractor(
        entity_type=entity_type,
        language=language,
        dictionary_path=dictionary_path,
        match_strategy=match_strategy,
        use_ner=use_ner,
        **kwargs
    )

    # Extract entities
    return extractor.extract_entities(texts)