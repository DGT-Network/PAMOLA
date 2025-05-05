"""
Skill entity extractor.

This module provides functionality for extracting skills, technologies,
and competencies from text, with special focus on technical and professional skills.
"""

import logging
from typing import List, Optional, Tuple

from pamola_core.utils.nlp.entity.base import BaseEntityExtractor, EntityMatchResult
from pamola_core.utils.nlp.model_manager import NLPModelManager
from pamola_core.utils.nlp.tokenization import tokenize
from pamola_core.utils.nlp.stopwords import remove_stopwords

# Configure logger
logger = logging.getLogger(__name__)

# NLP model manager instance
nlp_model_manager = NLPModelManager()


class SkillExtractor(BaseEntityExtractor):
    """
    Extractor for skills and technologies.
    """

    def __init__(self, **kwargs):
        """
        Initialize the skill extractor.

        Additional parameters:
        -----------
        skill_type : str
            Type of skills to extract ('technical', 'soft', 'language', etc.)
        min_skill_length : int
            Minimum length for a skill term
        """
        super().__init__(**kwargs)

        # Skill-specific parameters
        self.skill_type = kwargs.get('skill_type', 'technical')
        self.min_skill_length = kwargs.get('min_skill_length', 3)

        # Common skills by category
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
                'go', 'rust', 'swift', 'kotlin', 'scala', 'perl', 'r', 'matlab'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'cassandra', 'redis',
                'elasticsearch', 'sqlite', 'dynamodb', 'neo4j', 'couchdb'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'laravel', 'express',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy'
            ],
            'tools': [
                'git', 'docker', 'kubernetes', 'jenkins', 'jira', 'confluence', 'aws',
                'azure', 'gcp', 'linux', 'unix', 'windows', 'macos'
            ],
            'soft_skills': [
                'communication', 'teamwork', 'leadership', 'problem-solving', 'creativity',
                'time management', 'adaptability', 'critical thinking', 'attention to detail'
            ],
            'languages': [
                'english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'russian',
                'chinese', 'japanese', 'korean', 'arabic'
            ]
        }

        # Combined list of all skills
        self.all_skills = []
        for category, skills in self.skill_categories.items():
            self.all_skills.extend(skills)

    def _get_entity_type(self) -> str:
        """
        Get the entity type for this extractor.

        Returns:
        --------
        str
            Entity type string
        """
        return "skill"

    def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]:
        """
        Extract skills using NER models and keyword matching.

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
            # First, try with keyword matching
            extracted_skills = self._extract_skills_from_text(normalized_text)

            if extracted_skills:
                # Determine the skill category
                skill_category, confidence = self._determine_skill_category(extracted_skills)

                # Create match result
                return EntityMatchResult(
                    original_text=text,
                    normalized_text=normalized_text,
                    category=f"SKILL_{skill_category.upper()}",
                    alias=f"skill_{skill_category.lower()}",
                    domain="Skill",
                    level=1,
                    seniority="Any",
                    confidence=confidence,
                    method="keyword",
                    language=language,
                    conflicts=[]
                )

            # If keyword matching fails, try NER
            # Use the model manager to get a suitable model
            model = nlp_model_manager.get_model('spacy', language)
            if not model:
                logger.warning(f"No NER model available for language '{language}'")
                return None

            # Use the model to extract entities
            doc = model(text)

            # Look for SKILL, PRODUCT, or TECH entities
            skill_entities = [ent.text for ent in doc.ents
                              if ent.label_ in ('SKILL', 'PRODUCT', 'TECH', 'WORK_OF_ART')]

            if not skill_entities:
                # If no explicit skills found, try noun phrases
                noun_phrases = [chunk.text for chunk in doc.noun_chunks
                                if len(chunk.text.split()) <= 3]  # Limit to short phrases

                if not noun_phrases:
                    return None

                # Use the first noun phrase
                skill_name = noun_phrases[0]
            else:
                # Use the first skill entity
                skill_name = skill_entities[0]

            # Create a match result
            return EntityMatchResult(
                original_text=text,
                normalized_text=normalized_text,
                category="NER_SKILL",
                alias="skill_general",
                domain="Skill",
                level=1,
                seniority="Any",
                confidence=0.6,  # Lower confidence for NER-based skills
                method="ner",
                language=language,
                conflicts=[]
            )

        except Exception as e:
            logger.error(f"Error in skill extraction: {e}")
            return None

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skills from text using keyword matching.

        Parameters:
        -----------
        text : str
            Text to extract skills from

        Returns:
        --------
        List[str]
            List of extracted skills
        """
        text_lower = text.lower()
        found_skills = []

        # Try to match known skills
        for skill in self.all_skills:
            if len(skill) >= self.min_skill_length and skill in text_lower:
                found_skills.append(skill)

        # If needed, tokenize and try to match individual tokens
        if not found_skills:
            tokens = tokenize(text_lower)
            # Use the imported remove_stopwords function
            tokens = remove_stopwords(tokens)

            for token in tokens:
                if len(token) >= self.min_skill_length and token in self.all_skills:
                    found_skills.append(token)

        return found_skills

    def _determine_skill_category(self, skills: List[str]) -> Tuple[str, float]:
        """
        Determine the most likely skill category from a list of skills.

        Parameters:
        -----------
        skills : List[str]
            List of extracted skills

        Returns:
        --------
        Tuple[str, float]
            Category name and confidence score
        """
        if not skills:
            return "general", 0.0

        # Count matches by category
        category_counts = {}

        for skill in skills:
            for category, category_skills in self.skill_categories.items():
                if skill in category_skills:
                    category_counts[category] = category_counts.get(category, 0) + 1

        # If no categories matched, return general
        if not category_counts:
            return "general", 0.5

        # Find the category with the most matches
        best_category = max(category_counts.items(), key=lambda x: x[1])
        category_name = best_category[0]

        # Calculate confidence
        total_matches = sum(category_counts.values())
        category_matches = category_counts[category_name]
        confidence = category_matches / total_matches if total_matches > 0 else 0.5

        return category_name, confidence