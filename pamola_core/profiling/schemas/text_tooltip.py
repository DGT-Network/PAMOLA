"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Semantic Categorizer Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Provides detailed tooltips for text semantic categorization configuration fields in PAMOLA.CORE.
- Explains entity type recognition, NER, clustering, and dictionary-based categorization options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of text anonymization operations

Changelog:
1.0.0 - 2025-11-11 - Initial creation of text semantic categorizer tooltip file
"""


class TextSemanticCategorizerOperationTooltip:
    entity_type = (
        "What it does: Defines the semantic domain of the text content to guide entity extraction and categorization\n"
        "• 'Job' optimizes for job titles and positions\n"
        "• 'Organization' for company names\n"
        "• 'Skill' for technical skills\n"
        "• 'Generic' for general text"
    )

    min_word_length = (
        "What it does: Sets the minimum number of characters a word must have to be included in token-based analysis and keyword "
        "matching. Words shorter than this threshold are ignored in category keyword matching."
    )

    clustering_threshold = (
        "What it does: Sets the minimum similarity score (0-1) required for clustering unmatched text entries into groups. Only applies to "
        "items not matched by dictionary or NER\n\n"
        "Example: Threshold=0.7 groups 'Python Developer' and 'Python Engineer' (72% similar) together, but not 'Python Developer' and "
        "'Java Developer' (45% similar)"
    )

    match_strategy = (
        "What it does: Determines how to resolve conflicts when text matches multiple categories in the dictionary hierarchy\n"
        "• 'Specific First' prioritizes deeper hierarchy levels (e.g., 'Python Developer' over 'Developer')\n"
        "• 'Domain Prefer' favors categories from primary domain\n"
        "• 'Atlas Only' uses only atlas fields\n"
        "• 'User Override' applies manual prioritization rules"
    )

    use_ner = (
        "What it does: Activates Named Entity Recognition (NER) for text items not matched by the dictionary, using spaCy models to identify "
        "persons, organizations, skills, etc.\n\n"
        "Example: 'Google Inc.' not in dictionary is recognized by NER as ORGANIZATION and mapped to relevant category; 'Sergey Brin' "
        "recognized as PERSON"
    )

    perform_categorization = (
        "What it does: Activates similarity-based clustering for text entries not matched by dictionary or NER, grouping similar items "
        "together\n\n"
        "Example: Unmatched items 'Backend Dev', 'Backend Programmer', 'Back-end Developer' are clustered together as 'CLUSTER_001' "
        "for potential manual categorization or anonymization"
    )

    perform_clustering = (
        "What it does: Activates similarity-based clustering for text entries not matched by dictionary or NER, grouping similar items "
        "together\n\n"
        "Example: Unmatched items 'Backend Dev', 'Backend Programmer', 'Back-end Developer' are clustered together as 'CLUSTER_001' "
        "for potential manual categorization or anonymization"
    )

    dictionary_path = (
        "What it does: Path to a file containing semantic category definitions and patterns.\n\n"
        "Leave blank to use default categories. Use this to customize semantic classification with domain-specific categories or terms."
    )

    generate_visualization = (
        "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value "
        "count distributions"
    )

    force_recalculation = "What it does: Ignores saved results. Check this to force the operation to run again instead of using a cached result."

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "entity_type": cls.entity_type,
            "min_word_length": cls.min_word_length,
            "clustering_threshold": cls.clustering_threshold,
            "match_strategy": cls.match_strategy,
            "use_ner": cls.use_ner,
            "perform_categorization": cls.perform_categorization,
            "perform_clustering": cls.perform_clustering,
            "dictionary_path": cls.dictionary_path,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
