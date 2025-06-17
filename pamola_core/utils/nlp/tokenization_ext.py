"""
Extended tokenization utilities that build upon the pamola_core tokenization module.

This module provides specialized functionality including n-gram extraction,
advanced token filtering, and other text processing extensions.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set

# Import from base module for pamola_core functionality
from pamola_core.utils.nlp.base import batch_process
from pamola_core.utils.nlp.cache import get_cache, cache_function
# Import necessary functions from tokenization
from pamola_core.utils.nlp.tokenization import tokenize, _get_or_detect_language

# Configure logger
logger = logging.getLogger(__name__)

# Use memory cache for efficient lookup
memory_cache = get_cache('memory')


@cache_function(ttl=3600, cache_type='memory')
def load_ngram_dictionary(sources: Optional[Union[str, List[str]]] = None,
                          language: Optional[str] = None) -> Set[str]:
    """
    Load n-gram dictionary from files.

    Parameters
    ----------
    sources : str or List[str], optional
        Path(s) to n-gram dictionary files. If None, use default.
    language : str, optional
        Language code for language-specific dictionaries.

    Returns
    -------
    Set[str]
        Set of n-grams
    """
    from pamola_core.utils.nlp.tokenization_helpers import load_ngram_dictionary as load_ngram_dict
    return load_ngram_dict(sources, language)


@cache_function(ttl=3600, cache_type='memory')
def extract_ngrams(
        tokens: List[str],
        n: int = 2,
        ngram_sources: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None
) -> List[str]:
    """
    Extract n-grams of a specific size from tokens.

    Parameters
    ----------
    tokens : List[str]
        Tokens to extract n-grams from
    n : int
        Size of n-grams to extract
    ngram_sources : str or List[str], optional
        Path(s) to n-gram dictionaries
    language : str, optional
        Language code

    Returns
    -------
    List[str]
        List of n-grams
    """
    if len(tokens) < n:
        return []

    targeted_ngrams = set()
    if ngram_sources or language:
        targeted_ngrams = load_ngram_dictionary(ngram_sources, language)

    results = []
    for i in range(len(tokens) - n + 1):
        chunk = ' '.join(tokens[i: i + n])
        if not targeted_ngrams or chunk.lower() in targeted_ngrams:
            results.append(chunk)
    return results


@cache_function(ttl=3600, cache_type='memory')
def extract_multi_ngrams(
        tokens: List[str],
        min_n: int = 1,
        max_n: int = 3,
        ngram_sources: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None
) -> List[str]:
    """
    Extract n-grams of varying sizes from tokens.

    Parameters
    ----------
    tokens : List[str]
        Tokens to extract n-grams from
    min_n : int
        Minimum size of n-grams
    max_n : int
        Maximum size of n-grams
    ngram_sources : str or List[str], optional
        Path(s) to n-gram dictionaries
    language : str, optional
        Language code

    Returns
    -------
    List[str]
        List of n-grams
    """
    if not tokens:
        return []

    targeted_ngrams = set()
    if ngram_sources or language:
        targeted_ngrams = load_ngram_dictionary(ngram_sources, language)

    all_ngrams: List[str] = []
    for size in range(min_n, min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - size + 1):
            chunk = ' '.join(tokens[i: i + size])
            if not targeted_ngrams or chunk.lower() in targeted_ngrams:
                all_ngrams.append(chunk)

    return all_ngrams


def extract_keyphrases(
        text: str,
        language: Optional[str] = None,
        min_length: int = 2,
        max_length: int = 5,
        top_n: int = 10,
        tokenizer_type: str = 'auto'
) -> List[Dict[str, Any]]:
    """
    Extract key phrases from text using a combination of n-gram frequency
    and filtering techniques.

    Parameters
    ----------
    text : str
        Text to analyze
    language : str, optional
        Language code
    min_length : int
        Minimum n-gram length
    max_length : int
        Maximum n-gram length
    top_n : int
        Number of top phrases to return
    tokenizer_type : str
        Type of tokenizer to use

    Returns
    -------
    List[Dict[str, Any]]
        List of key phrases with scores
    """
    if not text:
        return []

    # Detect language if needed
    lang_code = None
    if language is None:
        lang_code = _get_or_detect_language(text)
    else:
        # Ensure language is a string
        lang_code = str(language) if language is not None else None

    # Import tokenize function for explicit clarity
    from pamola_core.utils.nlp.tokenization import tokenize

    # Tokenize text
    tokens = tokenize(
        text,
        language=lang_code,
        min_length=1,  # Use lower threshold here, will filter later
        tokenizer_type=tokenizer_type
    )

    # Filter stopwords and short tokens
    from pamola_core.utils.nlp.stopwords import get_stopwords

    # Use a safely wrapped version to avoid type issues
    def get_safe_stopwords(lang):
        try:
            result = get_stopwords(lang)
            if result is None:
                return set()
            elif isinstance(result, str):
                return {result}
            elif hasattr(result, '__iter__'):
                return set(result)
            else:
                return set()
        except Exception as e:
            logger.warning(f"Error getting stopwords: {e}")
            return set()

    stopwords_set = get_safe_stopwords(lang_code)
    filtered_tokens = [t for t in tokens if len(t) >= min_length and t.lower() not in stopwords_set]

    if not filtered_tokens:
        return []

    # Extract multi-grams
    ngrams = extract_multi_ngrams(
        filtered_tokens,
        min_n=min_length,
        max_n=max_length
    )

    # Filter ngrams containing stopwords
    filtered_ngrams = []
    for ng in ngrams:
        words = ng.split()
        if len(words) <= 1:  # Include single words
            filtered_ngrams.append(ng)
            continue

        # Filter out if any component is stopword
        if any(word.lower() in stopwords_set for word in words):
            continue

        filtered_ngrams.append(ng)

    # Count frequencies
    from collections import Counter
    counts = Counter(filtered_ngrams)

    # Score ngrams
    scored_phrases = []
    for phrase, count in counts.most_common(top_n * 2):  # Get more to allow for filtering
        # Calculate score based on frequency and length
        words = phrase.split()
        score = count * (0.5 + 0.5 * len(words))  # Favor longer phrases

        scored_phrases.append({
            'phrase': phrase,
            'frequency': count,
            'words': len(words),
            'score': score
        })

    # Sort by score and return top N
    result = sorted(scored_phrases, key=lambda x: x['score'], reverse=True)[:top_n]
    return result


def batch_extract_keyphrases(
        texts: List[str],
        language: Optional[str] = None,
        min_length: int = 2,
        max_length: int = 5,
        top_n: int = 10,
        processes: Optional[int] = None,
        tokenizer_type: str = 'auto'
) -> List[List[Dict[str, Any]]]:
    """
    Extract keyphrases from multiple texts in parallel.

    Parameters
    ----------
    texts : List[str]
        List of texts to process
    language : str, optional
        Language code
    min_length : int
        Minimum n-gram length
    max_length : int
        Maximum n-gram length
    top_n : int
        Number of top phrases to return per text
    processes : int, optional
        Number of processes to use
    tokenizer_type : str
        Type of tokenizer to use

    Returns
    -------
    List[List[Dict[str, Any]]]
        List of keyphrase results for each text
    """
    return batch_process(
        texts,
        extract_keyphrases,
        processes=processes,
        language=language,
        min_length=min_length,
        max_length=max_length,
        top_n=top_n,
        tokenizer_type=tokenizer_type
    )


def filter_tokens_by_pos(
        text: str,
        pos_tags: List[str] = None,
        language: Optional[str] = None
) -> List[str]:
    """
    Filter tokens by part-of-speech tags.
    Requires spaCy to be installed.

    Parameters
    ----------
    text : str
        Text to analyze
    pos_tags : List[str]
        List of POS tags to keep (e.g., ['NOUN', 'ADJ', 'VERB'])
    language : str, optional
        Language code

    Returns
    -------
    List[str]
        Tokens with the specified POS tags
    """
    if not text:
        return []

    if pos_tags is None:
        pos_tags = ['NOUN', 'PROPN', 'ADJ']  # Default to most informative tags

    # Check if spaCy is available
    from pamola_core.utils.nlp.base import DependencyManager
    if not DependencyManager.check_dependency('spacy'):
        logger.warning("spaCy not available, cannot filter by POS tags")
        return tokenize(text, language=language)

    # Detect language if needed
    if language is None:
        language = _get_or_detect_language(text)

    # Get spaCy
    spacy = DependencyManager.get_module('spacy')

    # Get appropriate model
    model_map = {
        'en': 'en_core_web_sm',
        'ru': 'ru_core_news_sm',
        'de': 'de_core_news_sm',
        'fr': 'fr_core_news_sm',
        'es': 'es_core_news_sm'
    }
    model_name = model_map.get(language, model_map.get('en'))

    try:
        # Try to get model
        nlp = spacy.load(model_name)

        # Process text
        doc = nlp(text)

        # Filter tokens by POS
        filtered_tokens = [token.text for token in doc if token.pos_ in pos_tags]
        return filtered_tokens

    except Exception as e:
        logger.error(f"Error using spaCy for POS filtering: {e}")
        return tokenize(text, language=language)


def extract_collocations(
        text: str,
        language: Optional[str] = None,
        min_freq: int = 2,
        window_size: int = 3,
        tokenizer_type: str = 'auto'
) -> List[Dict[str, Any]]:
    """
    Extract word collocations (words that appear together more often than by chance).

    Parameters
    ----------
    text : str
        Text to analyze
    language : str, optional
        Language code
    min_freq : int
        Minimum frequency of collocation
    window_size : int
        Window size for word co-occurrence
    tokenizer_type : str
        Type of tokenizer to use

    Returns
    -------
    List[Dict[str, Any]]
        List of collocations with scores
    """
    if not text:
        return []

    # Check for NLTK availability
    from pamola_core.utils.nlp.base import DependencyManager
    if not DependencyManager.check_dependency('nltk'):
        logger.warning("NLTK not available, falling back to simple n-gram extraction")
        tokens = tokenize(text, language=language, tokenizer_type=tokenizer_type)
        ngrams = extract_multi_ngrams(tokens, min_n=2, max_n=window_size)

        # Count frequencies
        from collections import Counter
        counts = Counter(ngrams)

        # Convert to result format
        result = [{'collocation': ng, 'frequency': freq, 'score': freq}
                  for ng, freq in counts.most_common() if freq >= min_freq]
        return result

    # Use NLTK's collocation functions
    try:
        from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
        from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder

        # Tokenize
        tokens = tokenize(text, language=language, tokenizer_type=tokenizer_type)

        # Filter stopwords
        from pamola_core.utils.nlp.stopwords import get_stopwords
        stopwords = get_stopwords(language)
        filtered_tokens = [t for t in tokens if t.lower() not in stopwords]

        results = []

        # Process bigrams
        bigram_measures = BigramAssocMeasures()
        bigram_finder = BigramCollocationFinder.from_words(filtered_tokens, window_size=window_size)
        bigram_finder.apply_freq_filter(min_freq)

        # Score using multiple measures
        for bigram, score in bigram_finder.score_ngrams(bigram_measures.pmi):
            results.append({
                'collocation': ' '.join(bigram),
                'words': bigram,
                'frequency': bigram_finder.ngram_fd[bigram],
                'score': score,
                'type': 'bigram'
            })

        # Process trigrams if we have enough tokens
        if len(filtered_tokens) >= 3:
            trigram_measures = TrigramAssocMeasures()
            trigram_finder = TrigramCollocationFinder.from_words(filtered_tokens)
            trigram_finder.apply_freq_filter(min_freq)

            for trigram, score in trigram_finder.score_ngrams(trigram_measures.pmi):
                results.append({
                    'collocation': ' '.join(trigram),
                    'words': trigram,
                    'frequency': trigram_finder.ngram_fd[trigram],
                    'score': score,
                    'type': 'trigram'
                })

        # Sort by score and return
        return sorted(results, key=lambda x: x['score'], reverse=True)

    except Exception as e:
        logger.error(f"Error using NLTK for collocation extraction: {e}")
        # Fallback to simple approach
        tokens = tokenize(text, language=language, tokenizer_type=tokenizer_type)
        ngrams = extract_multi_ngrams(tokens, min_n=2, max_n=window_size)

        from collections import Counter
        counts = Counter(ngrams)

        result = [{'collocation': ng, 'frequency': freq, 'score': freq}
                  for ng, freq in counts.most_common() if freq >= min_freq]
        return result


class NGramExtractor:
    """
    Utility class for extracting n-grams from a list of tokens,
    optionally filtered by a dictionary.
    """

    @staticmethod
    def extract_ngrams(
            tokens: List[str],
            n: int = 2,
            ngram_sources: Optional[Union[str, List[str]]] = None,
            language: Optional[str] = None
    ) -> List[str]:
        """
        Extract n-grams of a specific size from tokens. This is a wrapper
        around the extract_ngrams function for backward compatibility.

        Parameters
        ----------
        tokens : List[str]
            Tokens to extract n-grams from
        n : int
            Size of n-grams to extract
        ngram_sources : str or List[str], optional
            Path(s) to n-gram dictionaries
        language : str, optional
            Language code

        Returns
        -------
        List[str]
            List of n-grams
        """
        return extract_ngrams(tokens, n, ngram_sources, language)

    @staticmethod
    def extract_multi_ngrams(
            tokens: List[str],
            min_n: int = 1,
            max_n: int = 3,
            ngram_sources: Optional[Union[str, List[str]]] = None,
            language: Optional[str] = None
    ) -> List[str]:
        """
        Extract n-grams of varying sizes from tokens. This is a wrapper
        around the extract_multi_ngrams function for backward compatibility.

        Parameters
        ----------
        tokens : List[str]
            Tokens to extract n-grams from
        min_n : int
            Minimum size of n-grams
        max_n : int
            Maximum size of n-grams
        ngram_sources : str or List[str], optional
            Path(s) to n-gram dictionaries
        language : str, optional
            Language code

        Returns
        -------
        List[str]
            List of n-grams
        """
        return extract_multi_ngrams(tokens, min_n, max_n, ngram_sources, language)


class AdvancedTextProcessor:
    """
    Extended text processor with additional capabilities beyond the
    pamola_core TextProcessor in the main tokenization module.
    """

    def __init__(
            self,
            language: Optional[str] = None,
            tokenizer_type: str = 'auto',
            config_sources: Optional[Union[str, List[str]]] = None,
            lemma_dict_sources: Optional[Union[str, List[str]]] = None,
            ngram_sources: Optional[Union[str, List[str]]] = None,
            min_token_length: int = 2,
            preserve_case: bool = False
    ):
        """
        Initialize the advanced text processor.

        Parameters
        ----------
        language : str, optional
            Language code
        tokenizer_type : str
            Type of tokenizer to use
        config_sources : str or List[str], optional
            Path(s) to config files
        lemma_dict_sources : str or List[str], optional
            Path(s) to lemma dictionaries
        ngram_sources : str or List[str], optional
            Path(s) to n-gram dictionaries
        min_token_length : int
            Minimum token length
        preserve_case : bool
            Whether to preserve case
        """
        # Import TextProcessor from pamola_core tokenization
        from pamola_core.utils.nlp.tokenization import TextProcessor

        # Create underlying TextProcessor
        self.processor = TextProcessor(
            language=language,
            tokenizer_type=tokenizer_type,
            config_sources=config_sources,
            lemma_dict_sources=lemma_dict_sources,
            min_token_length=min_token_length,
            preserve_case=preserve_case
        )

        # Additional properties
        self.ngram_sources = ngram_sources

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text using the underlying TextProcessor.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        return self.processor.tokenize(text, **kwargs)

    def lemmatize(self, tokens: List[str], **kwargs) -> List[str]:
        """
        Lemmatize tokens using the underlying TextProcessor.

        Parameters
        ----------
        tokens : List[str]
            Tokens to lemmatize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            Lemmatized tokens
        """
        return self.processor.lemmatize(tokens, **kwargs)

    def tokenize_and_lemmatize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize and lemmatize text using the underlying TextProcessor.

        Parameters
        ----------
        text : str
            Text to process
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            Lemmatized tokens
        """
        return self.processor.tokenize_and_lemmatize(text, **kwargs)

    def extract_ngrams(
            self,
            tokens: List[str],
            n: int = 2,
            **kwargs
    ) -> List[str]:
        """
        Extract n-grams from tokens.

        Parameters
        ----------
        tokens : List[str]
            Tokens to extract n-grams from
        n : int
            Size of n-grams
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of n-grams
        """
        lang = kwargs.get('language', self.processor.language)
        ngram_srcs = kwargs.get('ngram_sources', self.ngram_sources)
        return extract_ngrams(tokens, n=n, ngram_sources=ngram_srcs, language=lang)

    def extract_multi_ngrams(
            self,
            tokens: List[str],
            min_n: int = 1,
            max_n: int = 3,
            **kwargs
    ) -> List[str]:
        """
        Extract n-grams of varying sizes from tokens.

        Parameters
        ----------
        tokens : List[str]
            Tokens to extract n-grams from
        min_n : int
            Minimum size of n-grams
        max_n : int
            Maximum size of n-grams
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of n-grams
        """
        lang = kwargs.get('language', self.processor.language)
        ngram_srcs = kwargs.get('ngram_sources', self.ngram_sources)
        return extract_multi_ngrams(
            tokens,
            min_n=min_n,
            max_n=max_n,
            ngram_sources=ngram_srcs,
            language=lang
        )

    def extract_keyphrases(
            self,
            text: str,
            min_length: int = 2,
            max_length: int = 5,
            top_n: int = 10,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract key phrases from text.

        Parameters
        ----------
        text : str
            Text to analyze
        min_length : int
            Minimum n-gram length
        max_length : int
            Maximum n-gram length
        top_n : int
            Number of top phrases to return
        **kwargs
            Additional parameters

        Returns
        -------
        List[Dict[str, Any]]
            List of key phrases with scores
        """
        lang = kwargs.get('language', self.processor.language)
        return extract_keyphrases(
            text,
            language=lang,
            min_length=min_length,
            max_length=max_length,
            top_n=top_n,
            tokenizer_type=self.processor.tokenizer_type
        )

    def process_text_advanced(
            self,
            text: str,
            lemmatize_tokens: bool = True,
            extract_ngrams_flag: bool = True,
            extract_keyphrases_flag: bool = False,
            pos_filter: Optional[List[str]] = None,
            ngram_sizes: List[int] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Full advanced pipeline with more features than the base process_text.

        Parameters
        ----------
        text : str
            Text to process
        lemmatize_tokens : bool
            Whether to lemmatize tokens
        extract_ngrams_flag : bool
            Whether to extract n-grams
        extract_keyphrases_flag : bool
            Whether to extract keyphrases
        pos_filter : List[str], optional
            Part-of-speech tags to filter by
        ngram_sizes : List[int], optional
            Sizes of n-grams to extract
        **kwargs
            Additional parameters

        Returns
        -------
        Dict[str, Any]
            Processing results
        """
        if ngram_sizes is None:
            ngram_sizes = [2, 3]

        # Start with basic processing using the TextProcessor
        result = self.processor.process_text(
            text,
            lemmatize_tokens=lemmatize_tokens,
            extract_ngrams_flag=False,  # We'll handle n-grams ourselves
            **kwargs
        )

        # No tokens? Return early
        if not result.get('tokens', []):
            return result

        # Get language and tokens
        lang = result.get('detected_language', result.get('language', self.processor.language))
        tokens = result.get('tokens', [])

        # Apply POS filtering if requested
        if pos_filter:
            pos_filtered = filter_tokens_by_pos(text, pos_tags=pos_filter, language=lang)
            result['pos_filtered_tokens'] = pos_filtered

        # Extract n-grams if requested
        if extract_ngrams_flag and tokens:
            result['ngrams'] = {}
            for n in ngram_sizes:
                ngrams = self.extract_ngrams(tokens, n=n, language=lang)
                result['ngrams'][n] = ngrams

        # Extract keyphrases if requested
        if extract_keyphrases_flag:
            result['keyphrases'] = self.extract_keyphrases(
                text,
                language=lang,
                **kwargs
            )

        return result


def extract_sentiment_words(
        text: str,
        language: Optional[str] = None,
        tokenizer_type: str = 'auto'
) -> Dict[str, List[str]]:
    """
    Extract words with positive and negative sentiment from text.
    This is a simple lexicon-based approach.

    Parameters
    ----------
    text : str
        Text to analyze
    language : str, optional
        Language code
    tokenizer_type : str
        Type of tokenizer to use

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'positive' and 'negative' word lists
    """
    if not text:
        return {'positive': [], 'negative': []}

    # Detect language if needed
    if language is None:
        language = _get_or_detect_language(text)

    # Tokenize and lemmatize
    from pamola_core.utils.nlp.tokenization import tokenize_and_lemmatize
    tokens = tokenize_and_lemmatize(
        text,
        language=language,
        tokenizer_type=tokenizer_type
    )

    if not tokens:
        return {'positive': [], 'negative': []}

    # Get sentiment lexicons
    try:
        # Try to find lexicon files
        from pamola_core.utils.nlp.base import RESOURCES_DIR
        import os
        import json

        lexicon_dir = os.path.join(RESOURCES_DIR, 'sentiment')
        pos_path = os.path.join(lexicon_dir, f'{language}_positive.txt')
        neg_path = os.path.join(lexicon_dir, f'{language}_negative.txt')

        positive_words = set()
        negative_words = set()

        # Load positive words
        if os.path.exists(pos_path):
            with open(pos_path, 'r', encoding='utf-8') as f:
                positive_words = {line.strip().lower() for line in f if line.strip()}

        # Load negative words
        if os.path.exists(neg_path):
            with open(neg_path, 'r', encoding='utf-8') as f:
                negative_words = {line.strip().lower() for line in f if line.strip()}

        # Fallback to English if specific language not found
        if not positive_words and not language == 'en':
            pos_path = os.path.join(lexicon_dir, 'en_positive.txt')
            if os.path.exists(pos_path):
                with open(pos_path, 'r', encoding='utf-8') as f:
                    positive_words = {line.strip().lower() for line in f if line.strip()}

        if not negative_words and not language == 'en':
            neg_path = os.path.join(lexicon_dir, 'en_negative.txt')
            if os.path.exists(neg_path):
                with open(neg_path, 'r', encoding='utf-8') as f:
                    negative_words = {line.strip().lower() for line in f if line.strip()}

        # Filter tokens by sentiment
        pos_matches = [t for t in tokens if t.lower() in positive_words]
        neg_matches = [t for t in tokens if t.lower() in negative_words]

        return {
            'positive': pos_matches,
            'negative': neg_matches
        }

    except Exception as e:
        logger.error(f"Error loading sentiment lexicons: {e}")
        return {'positive': [], 'negative': []}


def tokenize_and_analyze(
        text: str,
        language: Optional[str] = None,
        include_sentiment: bool = False,
        include_keyphrases: bool = False,
        include_ngrams: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Comprehensive tokenization and analysis function that combines
    multiple types of analysis in one call.

    Parameters
    ----------
    text : str
        Text to analyze
    language : str, optional
        Language code
    include_sentiment : bool
        Whether to include sentiment analysis
    include_keyphrases : bool
        Whether to include keyphrases
    include_ngrams : bool
        Whether to include n-grams
    **kwargs
        Additional parameters

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis results
    """
    if not text:
        return {"tokens": [], "analysis": {}}

    # Detect language if needed
    if language is None:
        language = _get_or_detect_language(text)

    # Create processor
    processor = AdvancedTextProcessor(language=language)

    # Get base results
    result = processor.process_text_advanced(
        text,
        lemmatize_tokens=True,
        extract_ngrams_flag=include_ngrams,
        extract_keyphrases_flag=include_keyphrases,
        **kwargs
    )

    # Add sentiment analysis if requested
    if include_sentiment:
        result['sentiment'] = extract_sentiment_words(text, language=language)

    return result