"""
Entity extraction utilities for the NLP module.

This module provides high-level functions for extracting various types of entities
from text, serving as a unified API to the entity package. It supports extracting
job positions, organizations, skills, transaction purposes, and custom entity types.
"""

import logging
from typing import Dict, List, Any, Optional

from pamola_core.utils.nlp.cache import cache_function
from pamola_core.utils.nlp.entity import (
    create_entity_extractor,
    extract_entities as entity_extract_entities
)

# Configure logger
logger = logging.getLogger(__name__)


@cache_function(ttl=3600, cache_type='memory')
def extract_entities(
    texts: List[str],
    entity_type: str = "generic",
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    record_ids: Optional[List[str]] = None,
    show_progress: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    High-level function to extract entities from texts.

    This function serves as the main entry point for entity extraction,
    delegating to specialized extractors based on the entity type.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to process
    entity_type : str
        Type of entities to extract ("job", "organization", "skill", "transaction", "generic")
    language : str
        Language code or "auto" for detection
    dictionary_path : str, optional
        Path to the dictionary file
    match_strategy : str
        Strategy for resolving matches ("specific_first", "domain_prefer", "alias_only", "user_override")
    use_ner : bool
        Whether to use NER models if dictionary match fails
    record_ids : List[str], optional
        List of record IDs corresponding to the texts
    show_progress : bool
        Whether to show a progress bar
    **kwargs
        Additional parameters for specific extractors

    Returns:
    --------
    Dict[str, Any]
        Extraction results containing entities, categories, and statistics
    """
    return entity_extract_entities(
        texts=texts,
        entity_type=entity_type,
        language=language,
        dictionary_path=dictionary_path,
        match_strategy=match_strategy,
        use_ner=use_ner,
        record_ids=record_ids,
        show_progress=show_progress,
        **kwargs
    )


def extract_job_positions(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    seniority_detection: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract job positions from texts.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to process
    language : str
        Language code or "auto" for detection
    dictionary_path : str, optional
        Path to the dictionary file
    use_ner : bool
        Whether to use NER models if dictionary match fails
    seniority_detection : bool
        Whether to detect seniority levels
    **kwargs
        Additional parameters for the extractor

    Returns:
    --------
    Dict[str, Any]
        Extraction results
    """
    return extract_entities(
        texts=texts,
        entity_type="job",
        language=language,
        dictionary_path=dictionary_path,
        use_ner=use_ner,
        seniority_detection=seniority_detection,
        **kwargs
    )


def extract_organizations(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    organization_type: str = "any",
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract organization names from texts.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to process
    language : str
        Language code or "auto" for detection
    dictionary_path : str, optional
        Path to the dictionary file
    organization_type : str
        Type of organizations to extract ('company', 'university', 'government', etc.)
    use_ner : bool
        Whether to use NER models if dictionary match fails
    **kwargs
        Additional parameters for the extractor

    Returns:
    --------
    Dict[str, Any]
        Extraction results
    """
    return extract_entities(
        texts=texts,
        entity_type="organization",
        language=language,
        dictionary_path=dictionary_path,
        use_ner=use_ner,
        organization_type=organization_type,
        **kwargs
    )


def extract_universities(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract university and educational institution names from texts.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to process
    language : str
        Language code or "auto" for detection
    dictionary_path : str, optional
        Path to the dictionary file
    use_ner : bool
        Whether to use NER models if dictionary match fails
    **kwargs
        Additional parameters for the extractor

    Returns:
    --------
    Dict[str, Any]
        Extraction results
    """
    return extract_entities(
        texts=texts,
        entity_type="organization",
        language=language,
        dictionary_path=dictionary_path,
        use_ner=use_ner,
        organization_type="university",
        **kwargs
    )


def extract_skills(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    skill_type: str = "technical",
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract skills from texts.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to process
    language : str
        Language code or "auto" for detection
    dictionary_path : str, optional
        Path to the dictionary file
    skill_type : str
        Type of skills to extract ('technical', 'soft', 'language', etc.)
    use_ner : bool
        Whether to use NER models if dictionary match fails
    **kwargs
        Additional parameters for the extractor

    Returns:
    --------
    Dict[str, Any]
        Extraction results
    """
    return extract_entities(
        texts=texts,
        entity_type="skill",
        language=language,
        dictionary_path=dictionary_path,
        use_ner=use_ner,
        skill_type=skill_type,
        **kwargs
    )


def extract_transaction_purposes(
    texts: List[str],
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    use_ner: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract transaction purposes from texts.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to process
    language : str
        Language code or "auto" for detection
    dictionary_path : str, optional
        Path to the dictionary file
    use_ner : bool
        Whether to use NER models if dictionary match fails
    **kwargs
        Additional parameters for the extractor

    Returns:
    --------
    Dict[str, Any]
        Extraction results
    """
    return extract_entities(
        texts=texts,
        entity_type="transaction",
        language=language,
        dictionary_path=dictionary_path,
        use_ner=use_ner,
        **kwargs
    )


def create_custom_entity_extractor(
    entity_type: str,
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    **kwargs
) -> Any:
    """
    Create a custom entity extractor for a specific use case.

    Parameters:
    -----------
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
    BaseEntityExtractor
        An entity extractor instance
    """
    return create_entity_extractor(
        entity_type=entity_type,
        language=language,
        dictionary_path=dictionary_path,
        match_strategy=match_strategy,
        use_ner=use_ner,
        **kwargs
    )