"""
Text clustering module for the HHR project.

This module provides functionality for clustering similar texts,
with support for different similarity metrics and algorithms.
"""

import logging
from typing import Dict, List, Set, Optional

from pamola_core.utils.nlp.base import batch_process
from pamola_core.utils.nlp.cache import cache_function
from pamola_core.utils.nlp.language import detect_language
from pamola_core.utils.nlp.tokenization import tokenize

# Configure logger
logger = logging.getLogger(__name__)


class TextClusterer:
    """
    Class for clustering similar text items.
    """

    def __init__(self, threshold: float = 0.7, tokenize_fn=None, language: str = "auto"):
        """
        Initialize the text clusterer.

        Parameters:
        -----------
        threshold : float
            Similarity threshold for clustering (0-1)
        tokenize_fn : callable, optional
            Function to use for tokenization
        language : str
            Language code or "auto" for detection
        """
        self.threshold = threshold
        self.tokenize_fn = tokenize_fn or tokenize
        self.language = language

    def cluster_texts(self, texts: List[str]) -> Dict[str, List[int]]:
        """
        Cluster texts by similarity.

        Parameters:
        -----------
        texts : List[str]
            List of text strings to cluster

        Returns:
        --------
        Dict[str, List[int]]
            Dictionary mapping cluster labels to lists of text indices
        """
        if not texts:
            return {}

        # Detect language if auto
        language = self.language
        if language == "auto" and texts:
            language = detect_language(texts[0])

        # Tokenize all texts
        tokenized_texts = []
        for text in texts:
            if not text:
                tokenized_texts.append(set())
                continue

            tokens = set(self.tokenize_fn(text, language=language, min_length=2))
            tokenized_texts.append(tokens)

        # Initialize clusters
        clusters = {}
        assigned = set()

        # Cluster texts
        for i, tokens_i in enumerate(tokenized_texts):
            if i in assigned or not tokens_i:
                continue

            # Create a new cluster
            cluster_label = f"CLUSTER_{len(clusters)}"
            clusters[cluster_label] = [i]
            assigned.add(i)

            # Find similar texts
            for j, tokens_j in enumerate(tokenized_texts):
                if j <= i or j in assigned or not tokens_j:
                    continue

                # Calculate similarity
                similarity = self.calculate_similarity(tokens_i, tokens_j)

                # Add to cluster if similar enough
                if similarity >= self.threshold:
                    clusters[cluster_label].append(j)
                    assigned.add(j)

        return clusters

    @staticmethod
    def calculate_similarity(tokens1: Set[str], tokens2: Set[str], method: str = "jaccard") -> float:
        """
        Calculate similarity between two token sets.

        Parameters:
        -----------
        tokens1 : Set[str]
            First set of tokens
        tokens2 : Set[str]
            Second set of tokens
        method : str
            Similarity method: "jaccard", "overlap", "cosine"

        Returns:
        --------
        float
            Similarity score (0-1)
        """
        if not tokens1 or not tokens2:
            return 0.0

        if method == "jaccard":
            # Jaccard similarity: size of intersection / size of union
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            return intersection / union if union > 0 else 0.0

        elif method == "overlap":
            # Overlap coefficient: size of intersection / size of smaller set
            intersection = len(tokens1.intersection(tokens2))
            min_size = min(len(tokens1), len(tokens2))
            return intersection / min_size if min_size > 0 else 0.0

        elif method == "cosine":
            # Simplified cosine similarity for token sets
            intersection = len(tokens1.intersection(tokens2))
            product = len(tokens1) * len(tokens2)
            return intersection / (product ** 0.5) if product > 0 else 0.0

        else:
            logger.warning(f"Unknown similarity method: {method}, using jaccard")
            # Default to jaccard
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            return intersection / union if union > 0 else 0.0


@cache_function(ttl=3600, cache_type='memory')
def cluster_by_similarity(texts: List[str], threshold: float = 0.7, language: str = "auto") -> Dict[str, List[int]]:
    """
    Cluster texts by token overlap similarity.

    Parameters:
    -----------
    texts : List[str]
        List of text strings
    threshold : float
        Similarity threshold for clustering (0-1)
    language : str
        Language code or "auto" to detect

    Returns:
    --------
    Dict[str, List[int]]
        Dictionary mapping cluster labels to lists of text indices
    """
    clusterer = TextClusterer(threshold=threshold, language=language)
    return clusterer.cluster_texts(texts)


def batch_cluster_texts(texts_list: List[List[str]], threshold: float = 0.7,
                        language: str = "auto", processes: Optional[int] = None) -> List[Dict[str, List[int]]]:
    """
    Perform clustering on multiple batches of texts in parallel.

    Parameters:
    -----------
    texts_list : List[List[str]]
        List of text batches to cluster
    threshold : float
        Similarity threshold
    language : str
        Language code or "auto"
    processes : int, optional
        Number of processes for parallel execution

    Returns:
    --------
    List[Dict[str, List[int]]]
        List of clustering results, one per batch
    """
    return batch_process(
        texts_list,
        cluster_by_similarity,
        processes=processes,
        threshold=threshold,
        language=language
    )