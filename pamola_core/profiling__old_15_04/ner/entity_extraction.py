"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor` for extracting named
entities from free-text fields using standard NLP libraries (spaCy, NLTK, etc.), identifying
and categorizing entities such as persons, organizations, locations, and more.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Named Entity Recognition Profiling Operations 
--------------------------------   
It includes the following capabilities:  
- Extracted entities by type
- Entity frequency counts
- Top entities by occurrence
- Entity distribution metrics
- Co-occurrence patterns
- Context analysis (when enabled)
- Entity type distribution visualization

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""


from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, OrderedDict
import difflib
import itertools
import json
import re
import sys
import natasha
import nltk
import pandas as pd
import subprocess
import spacy
import numpy as np
from scipy.stats import entropy
import logging
from natasha import Doc
from rapidfuzz import fuzz

from pamola_core.profiling.base import BaseProfilingProcessor
from pamola_core.utils.profiling_metrics import gini_index, simpsons_index

# Configure logging
logger = logging.getLogger(__name__)


class EntityExtractionProfilingProcessor(BaseProfilingProcessor):
    """
    Processor for extracting named entities from free-text fields using standard NLP
    libraries (spaCy, NLTK, etc.), identifying and categorizing entities such as persons,
    organizations, locations, and more.
    """
    
    def __init__(
        self,
        entity_types: List[str] = ["PERSON", "ORG", "LOC"], 
        nlp_library: str = "spacy",
        model_name: str = "en_core_web_md",
        min_frequency: int = 5, 
        max_entities: int = 100, 
        custom_entities: str = "",
        aggregate_similar: bool = True,
        collect_context: bool = False,
        context_window: int = 5,
    ):
        """
        Initializes the Entity Extraction Profiling Processor.

        Parameters:
        -----------
        entity_types : List[str], optional
            Types of entities to extract (default=["PERSON", "ORG", "LOC"]).
        nlp_library : str, optional
            NLP library to use for entity extraction (default="spacy").
        model_name : str, optional
            Name of the model to use (default="en_core_web_md").
        min_frequency : int, optional
            Minimum entity frequency to report (default=5).
        max_entities : int, optional
            Maximum number of entities to report (default=100).
        custom_entities: string, optional
            Path to file with custom entities (default="").
        aggregate_similar: bool, optional
            Whether to aggregate similar entities (default=True).
        collect_context: bool, optional
            Whether to collect entity context (default=False).
        context_window: int, optional
            Number of words to collect as context (default=5).

        """
        super().__init__()
        self.entity_types = entity_types
        self.nlp_library = nlp_library
        self.model_name = model_name
        self.min_frequency = min_frequency
        self.max_entities = max_entities
        self.custom_entities = custom_entities
        self.aggregate_similar = aggregate_similar
        self.collect_context = collect_context
        self.context_window = context_window
        self.nlp_models: Dict[str, Any] = {}
    
    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform categorical pattern analysis on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to analyze.
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all categorical columns will be selected.
        **kwargs : dict
            Dynamic parameter overrides:
            
            - `entity_types` (List[str], default=self.entity_types):
                Types of entities to extract.
            - `nlp_library` (str, default=self.nlp_library):
                NLP library to use for entity extraction.
            - `model_name` (str, default=self.model_name):
                Name of the model to use.
            - `min_frequency` (int, default=self.min_frequency):
                Minimum entity frequency to report.
            - `max_entities` (int, default=self.max_entities):
                Maximum number of entities to report.
            - `custom_entities` (str, default=self.custom_entities):
                Path to file with custom entities.
            - `aggregate_similar` (bool, default=self.aggregate_similar):
                Whether to aggregate similar entities.
            - `collect_context` (bool, default=self.collect_context):
                Whether to collect entity context.
            - `context_window` (int, default=self.context_window):
                Number of words to collect as context.
        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their pattern analysis results.
        """
        entity_types = kwargs.get("entity_types", self.entity_types)
        nlp_library = kwargs.get("nlp_library", self.nlp_library)
        model_name = kwargs.get("model_name", self.model_name)
        min_frequency = kwargs.get("min_frequency", self.min_frequency)
        max_entities = kwargs.get("max_entities", self.max_entities)
        custom_entities = kwargs.get("custom_entities", self.custom_entities)
        aggregate_similar = kwargs.get("aggregate_similar", self.aggregate_similar)
        collect_context = kwargs.get("collect_context", self.collect_context)
        context_window = kwargs.get("context_window", self.context_window)

        # Load the custom entities if provided
        entity_type_list = entity_types
        if custom_entities:
            custom_entity_list = self._load_custom_entities(custom_entities) or []
            entity_type_list.extend(custom_entity_list)  # Add custom entities to entity_type_list

        nlp = self._load_nlp_model(nlp_library, model_name)

        result = {}
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            entity_by_type = defaultdict(int)
            entity_pairs = defaultdict(int)
            context_data = {}
            entities = []
            context_lengths_all = []
            
            texts = df[col].dropna().to_numpy()
            
            for text in texts:
                try: 
                    extracted_entities, context, context_lengths = self._process_text(
                        text, nlp, nlp_library, entity_type_list, 
                        collect_context, context_window
                    )

                    if not extracted_entities:
                        continue

                    # Update entities list
                    entity_texts = [ent[0] for ent in extracted_entities]
                    entities.extend(entity_texts)
                    
                    # Update entity counts by type
                    for entity, entity_type in extracted_entities:
                        entity_by_type[(entity, entity_type)] += 1
                    
                    # Normalize pairs (sorted to avoid duplicates)
                    pairs = [tuple(sorted(pair)) for pair in itertools.combinations(set(entity_texts), 2)]
                    for pair in pairs:
                        entity_pairs[pair] += 1

                    # Collect context if enabled
                    if collect_context:
                        for k, v in context.items():
                            context_data.setdefault(k, []).extend(v)
                        context_lengths_all.extend(context_lengths)

                except Exception as e:
                    # Handle any errors during entity extraction
                    logger.error(f"Error processing text in column '{col}': {e}", exc_info=True)
                    continue
            
            # Calculate entity counts
            entity_counts = pd.Series(entities).value_counts()
            if aggregate_similar:
                entities = self._aggregate_similar_entities(entities)
                entity_counts = pd.Series(entities)

            # Filter entities by minimum frequency
            filtered_entities = {k: v for k, v in entity_counts.items() if v >= min_frequency}

            # Top N entities
            top_entities = {}
            if max_entities > 0:
                top_entities = entity_counts.nlargest(min(max_entities, len(entity_counts))).to_dict()

            entity_distribution = (entity_counts / entity_counts.sum()).round(4).to_dict()

            # Co-occurrence patterns
            co_occurrence = dict(sorted(entity_pairs.items(), key=lambda x: x[1], reverse=True)[:max_entities])
            co_occurrence = {f"{a}, {b}": v for (a, b), v in co_occurrence.items()}

            # Calculate average context length
            average_context_length = np.mean(context_lengths_all) if context_lengths_all else 0
            entity_by_type_str = {str(key): value for key, value in entity_by_type.items()}

            # Calculate entity distribution metrics
            entropy_value = entropy(entity_counts, base=2)
            gini_index_value = gini_index(entity_counts)
            simpsons_index_value = simpsons_index(entity_counts)

            # Store results for the column
            result[col] = {
                "entities_by_type": entity_by_type_str,
                "entity_counts": filtered_entities,
                "entity_distribution": entity_distribution,
                "entropy": entropy_value,
                "gini_index": gini_index_value,
                "simpsons_index": simpsons_index_value,
                "top_entities": top_entities,
                "co_occurrence_patterns": co_occurrence,
                "context": context_data if collect_context else None,
                "average_context_length": average_context_length,
            }

        return result
    
    def _process_text(
        self, text: str, nlp, library: str, entity_types: List[str],
        collect_context: bool, context_window: int
    ) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]], List[int]]:
        """
        Extract entities and (optionally) extract context around them from a given text string.

        Parameters:
        -----------
        text : str
            The input text to analyze.
        nlp : Any
            The preloaded NLP model (e.g., SpaCy, NLTK, or Natasha).
        library : str
            Name of the NLP library being used.
        entity_types : List[str]
            List of entity types to extract (e.g., PERSON, ORG, etc.).
        collect_context : bool
            Whether to collect context around the extracted entities.
        context_window : int
            Number of characters before and after the entity to extract as context.

        Returns:
        --------
        Tuple:
            - extracted_entities (List[Tuple[str, str]]): List of extracted entities with their types.
            - context_data (Dict[str, List[str]]): Context snippets per entity.
            - context_lengths (List[int]): Lengths of each context snippet.
        """
        try:
            # Extract entities from the text using the specified NLP library
            extracted_entities = self._extract_entities(text, nlp, library, entity_types)
            if not extracted_entities:
                return [], {}, []

            # Get just the text part of each entity
            entity_texts = [ent[0] for ent in extracted_entities]
            context_data = defaultdict(list)        # Stores context snippets for each entity
            context_lengths = []                    # Stores length of each context snippet for statistics

            if collect_context:
                for ent_text in entity_texts:
                    # Find all occurrences of the entity text in the input text
                    for match in re.finditer(re.escape(ent_text), text):
                        pos = match.start()
                        # Define start and end boundaries for the context window
                        start = max(0, pos - context_window)
                        end = min(len(text), pos + len(ent_text) + context_window)
                        context = text[start:end]

                        # Append the context snippet to the dictionary
                        context_data[ent_text].append(context)
                        # Track the length of the context snippet
                        context_lengths.append(len(context))

            return extracted_entities, dict(context_data), context_lengths

        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)
            return [], {}, []
    
    def _extract_entities(self, text: str, nlp, nlp_library: str, entity_types: List[str]) -> List[Tuple[str, str]]:
        """
        Main function to extract named entities from text using the specified NLP library.
        
        :param text: Input text
        :param nlp: NLP pipeline or components depending on the library
        :param nlp_library: Name of the NLP library ('spacy', 'nltk', or 'natasha')
        :param entity_types: List of entity types to include in the output
        :return: List of (entity_text, entity_type)
        """
        if not text.strip():
            return []

        try:
            if nlp_library == "spacy":
                return self._extract_with_spacy(text, nlp, entity_types)
            elif nlp_library == "nltk":
                return self._extract_with_nltk(text, entity_types)
            elif nlp_library == "natasha":
                return self._extract_with_natasha(text, nlp, entity_types)
            else:
                logger.error("Unsupported NLP library requested: %s", nlp_library)
                raise ValueError(f"Unsupported NLP library: {nlp_library}")
        except Exception as e:
            logger.error(f"Failed to extract entities with {nlp_library}: {e}", exc_info=True)
            return []


    def _extract_with_spacy(self, text: str, nlp, entity_types: List[str]) -> List[Tuple[str, str]]:
        """
        Extract entities using spaCy.
        
        :param text: Input text
        :param nlp: spaCy language model
        :param entity_types: List of desired entity types
        :return: List of (entity_text, entity_type)
        """
        doc = nlp(text)
        return [
            (ent.text, ent.label_)
            for ent in doc.ents if ent.label_ in entity_types
        ]


    def _extract_with_nltk(self, text: str, entity_types: List[str]) -> List[Tuple[str, str]]:
        """
        Extract entities using NLTK's named entity chunking.
        
        :param text: Input text
        :param entity_types: List of desired entity types
        :return: List of (entity_text, entity_type)
        """
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            named_entities = nltk.ne_chunk(pos_tags, binary=False)
        except Exception as e:
            logger.error("Error during NLTK processing: %s", e, exc_info=True)
            return []

        extracted = []
        for subtree in named_entities:
            if hasattr(subtree, 'label'):
                # Concatenate the words that make up the entity
                entity_name = " ".join([leaf[0] for leaf in subtree.leaves()])
                entity_type = subtree.label()
                if entity_type in entity_types:
                    extracted.append((entity_name, entity_type))

        return extracted


    def _extract_with_natasha(self, text: str, nlp_components, entity_types: List[str]) -> List[Tuple[str, str]]:
        """
        Extract entities using Natasha (Russian NLP).
        
        :param text: Input text
        :param nlp_components: Tuple of (segmenter, morph_tagger, ner_tagger)
        :param entity_types: List of desired entity types
        :return: List of (entity_text, entity_type)
        """
        segmenter, morph_tagger, ner_tagger = nlp_components

        # Create and process Natasha document
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.tag_ner(ner_tagger)

        if doc.spans is None:
            return []

        # Extract named entities with desired types
        return [
            (span.text, span.type)
            for span in doc.spans if span.type in entity_types
        ]
    
    def _load_custom_entities(self, file_path: str) -> List[str]:
        """
        Load custom entities from a CSV or JSON file.

        Parameters
        ----------
        file_path : str
            Path to the file containing custom entities.

        Returns
        -------
        List[str]
            A list of custom entity strings. Returns an empty list if the file is invalid or loading fails.
        """
        try:

            # CSV file - assumes entities are in the first column
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                return df.iloc[:, 0].dropna().tolist()
            
            #  JSON file - expects {"custom_entities": [ ... ]}
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "custom_entities" in data:
                        if isinstance(data["custom_entities"], list):
                            return [str(entity) for entity in data["custom_entities"] if entity]
                        else:
                            logger.warning(f"'custom_entities' in {file_path} is not a list.")
                            return []
                    else:
                        logger.warning(f"Invalid JSON structure in {file_path}. Expected a key 'custom_entities'.")
                        return []
            else:
                logger.warning(f"Unsupported file format for custom entities: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading custom entities from {file_path}: {e}")
            return []

    def _aggregate_similar_entities(self, entities: List[str], threshold: float = 80) -> Dict[str, int]:
        """
        Group similar entities based on string similarity using rapidfuzz and return entity counts.

        Parameters
        ----------
        entities : List[str]
            A list of entity strings that may contain duplicates or similar variations.
        threshold : float, default=80
            Similarity threshold (between 0 and 100) above which two entities are considered similar.

        Returns
        -------
        Dict[str, int]
            An ordered dictionary where keys are representative entity strings
            and values are the counts of similar entities grouped under them.
        """
        aggregated_entities = OrderedDict()  # Dictionary to store {entity: count}

        for entity in entities:
            matched = False

            # Compare with existing representative entities
            for rep in aggregated_entities:
                # Compute similarity score between current entity and representative
                if fuzz.ratio(entity, rep) >= threshold:
                    aggregated_entities[rep] += 1
                    matched = True
                    break

            # If no match is found, treat the entity as a new representative
            if not matched:
                aggregated_entities[entity] = 1

        return aggregated_entities
    
    def _load_nlp_model(self, nlp_library: str, model_name: str = "en_core_web_md"):
        """
        Load NLP model based on the specified library.

        Parameters
        ----------
        nlp_library : str
            The name of the NLP library to use ('spacy', 'nltk', or 'natasha').
        model_name : str, default="en_core_web_md"
            The model name to load for spaCy (ignored for other libraries).

        Returns
        -------
        Any
            Loaded NLP model or components depending on the library.
        """
        # Return from cache if already loaded
        if nlp_library in self.nlp_models:
            return self.nlp_models[nlp_library]

        try:
            if nlp_library == "spacy":
                model = self._load_spacy_model(model_name)

            elif nlp_library == "nltk":
                model = self._load_nltk_resources()

            elif nlp_library == "natasha":
                model = self._load_natasha_components()

            else:
                raise ValueError(f"Unsupported NLP library: {nlp_library}")

            # Cache the loaded model
            self.nlp_models[nlp_library] = model
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load {nlp_library} model: {e}")


    def _load_spacy_model(self, model_name: str):
        """
        Load spaCy model, download if missing.

        Parameters
        ----------
        model_name : str
            Name of the spaCy model to load.

        Returns
        -------
        spacy.lang.Language
            Loaded spaCy model.
        """
        try:
            return spacy.load(model_name)
        except OSError:
            logger.info(f"Model '{model_name}' not found. Downloading {model_name} as fallback.")
            is_poetry = "poetry" in sys.executable
            cmd = (
                ["poetry", "run", "python", "-m", "spacy", "download", model_name]
                if is_poetry
                else [sys.executable, "-m", "spacy", "download", model_name]
            )
            subprocess.run(cmd, check=True)
            return spacy.load(model_name)


    def _load_nltk_resources(self):
        """
        Ensure necessary NLTK resources are available.

        Returns
        -------
        module
            NLTK module with resources ensured.
        """
        resources = {
            "tokenizers/punkt": "punkt",
            "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
            "chunkers/maxent_ne_chunker": "maxent_ne_chunker",
            "corpora/words": "words"
        }

        for path, resource in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)

        return nltk


    def _load_natasha_components(self):
        """
        Load Natasha components for Russian NER.

        Returns
        -------
        tuple
            Natasha components: (segmenter, morph_tagger, ner_tagger)
        """
        emb = natasha.NewsEmbedding()
        return (
            natasha.Segmenter(),
            natasha.NewsMorphTagger(emb),
            natasha.NewsNERTagger(emb)
        )
