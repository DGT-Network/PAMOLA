"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Package: pamola_core.utils.nlp.entity
Type: Internal (Non-Public API)

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pamola_core.utils.nlp.entity.base import BaseEntityExtractor

from pamola_core.utils.nlp.entity.base import BaseEntityExtractor

from pamola_core.utils.nlp.entity.dictionary import GenericDictionaryExtractor

from pamola_core.utils.nlp.entity.job import JobPositionExtractor

from pamola_core.utils.nlp.entity.organization import OrganizationExtractor

from pamola_core.utils.nlp.entity.skill import SkillExtractor

from pamola_core.utils.nlp.entity.transaction import TransactionPurposeExtractor

_EXTRACTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "job": ("pamola_core.utils.nlp.entity.job", "JobPositionExtractor"),
    "job_position": ("pamola_core.utils.nlp.entity.job", "JobPositionExtractor"),
    "organization": ("pamola_core.utils.nlp.entity.organization", "OrganizationExtractor"),
    "university": ("pamola_core.utils.nlp.entity.organization", "OrganizationExtractor"),
    "skill": ("pamola_core.utils.nlp.entity.skill", "SkillExtractor"),
    "transaction": ("pamola_core.utils.nlp.entity.transaction", "TransactionPurposeExtractor"),
    "generic": ("pamola_core.utils.nlp.entity.dictionary", "GenericDictionaryExtractor"),
    "dictionary": ("pamola_core.utils.nlp.entity.dictionary", "GenericDictionaryExtractor"),
}

def _load_extractor_class(entity_type: str):
    _, class_name = _EXTRACTOR_REGISTRY[entity_type]
    return globals()[class_name]

def create_entity_extractor(
    entity_type: str,
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    min_confidence: float = 0.5,
    **kwargs,
) -> BaseEntityExtractor:
    """
    Create an entity extractor based on the specified type.

    Parameters
    ----------
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

    Returns
    -------
    BaseEntityExtractor
        An instance of the appropriate entity extractor
    """
    entity_type = entity_type.lower()

    if entity_type not in _EXTRACTOR_REGISTRY:
        logger.warning(
            f"Unknown entity type: '{entity_type}', using generic dictionary extractor"
        )
        entity_type = "generic"

    extractor_class = _load_extractor_class(entity_type)

    if entity_type == "university":
        kwargs["organization_type"] = "university"

    return extractor_class(
        language=language,
        dictionary_path=dictionary_path,
        match_strategy=match_strategy,
        use_ner=use_ner,
        min_confidence=min_confidence,
        **kwargs,
    )

def extract_entities(
    texts: List[str],
    entity_type: str = "generic",
    language: str = "auto",
    dictionary_path: Optional[str] = None,
    match_strategy: str = "specific_first",
    use_ner: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Extract entities from a list of texts.

    Parameters
    ----------
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

    Returns
    -------
    dict[str, Any]
        Extraction results containing entities, categories, and statistics
    """
    extractor = create_entity_extractor(
        entity_type=entity_type,
        language=language,
        dictionary_path=dictionary_path,
        match_strategy=match_strategy,
        use_ner=use_ner,
        **kwargs,
    )
    return extractor.extract_entities(texts)

__all__ = [
    # base classes
    "BaseEntityExtractor",
    "JobPositionExtractor",
    "OrganizationExtractor",
    "SkillExtractor",
    "TransactionPurposeExtractor",
    "GenericDictionaryExtractor",
    # convenience functions
    "create_entity_extractor",
    "extract_entities",
]
