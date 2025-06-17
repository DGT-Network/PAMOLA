"""
Stop words handling utilities.

This module provides flexible functions for managing stop words with support for
multiple languages, custom stop word lists, and multiple source formats.
"""

import glob
import json
import logging
import os
from typing import List, Set, Optional, Union, Tuple, Dict

from pamola_core.utils.nlp.base import normalize_language_code
from pamola_core.utils.nlp.cache import get_cache, cache_function
from pamola_core.utils.nlp.compatibility import check_dependency

# Configure logger
logger = logging.getLogger(__name__)

# Get file cache from the cache module
file_cache = get_cache('file')
memory_cache = get_cache('memory')

# Optional NLTK integration
_NLTK_AVAILABLE = check_dependency('nltk')
if _NLTK_AVAILABLE:
    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords


        def _ensure_nltk_resources():
            """Ensure NLTK resources are available."""
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.info("Downloading NLTK stopwords")
                nltk.download('stopwords', quiet=True)


        _ensure_nltk_resources()
    except Exception as e:
        logger.warning(f"Error initializing NLTK: {e}")
        _NLTK_AVAILABLE = False


def get_config_paths() -> Dict[str, str]:
    """
    Get configuration paths from the project root config.

    Returns:
    --------
    Dict[str, str]
        Dictionary with configuration paths
    """
    # Cache the result to avoid repeated file operations
    cache_key = "pamola_config_paths"
    cached_paths = memory_cache.get(cache_key)
    if cached_paths:
        return cached_paths

    # Try to determine the project root
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    config_path = os.path.join(current_dir, 'configs', 'prj_config.json')

    # Default paths
    paths = {
        "data_repository": os.path.join(current_dir, 'data'),
        "external_dictionaries": os.path.join(current_dir, 'data', 'external_dictionaries'),
        "stopwords_dir": os.path.join(current_dir, 'data', 'external_dictionaries', 'stopwords')
    }

    # Try to load from config if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if "data_repository" in config:
                data_repo = config["data_repository"]
                paths["data_repository"] = data_repo
                paths["external_dictionaries"] = os.path.join(data_repo, 'external_dictionaries')
                paths["stopwords_dir"] = os.path.join(data_repo, 'external_dictionaries', 'stopwords')
        except Exception as e:
            logger.warning(f"Error loading project config: {e}")

    # Cache the paths
    memory_cache.set(cache_key, paths)
    return paths


def get_stopwords_dirs() -> List[str]:
    """
    Get all directories to search for stopwords.

    Returns:
    --------
    List[str]
        List of directories to search for stopwords
    """
    # Package resource locations
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_resources_dir = os.path.join(package_dir, 'resources', 'stopwords')

    # Override with environment variables if present
    env_stopwords_dir = os.environ.get('PAMOLA_STOPWORDS_DIR', default_resources_dir)

    # Get data repository paths
    config_paths = get_config_paths()
    external_stopwords_dir = config_paths["stopwords_dir"]

    # Collect all directories ensuring they exist
    dirs = []
    for directory in [env_stopwords_dir, external_stopwords_dir, default_resources_dir]:
        if directory and os.path.exists(directory):
            dirs.append(directory)
        elif directory:
            os.makedirs(directory, exist_ok=True)
            dirs.append(directory)

    return dirs


@cache_function(ttl=3600, cache_type='file')
def load_stopwords_from_file(file_path: str, encoding: str = 'utf-8') -> Set[str]:
    """
    Load stop words from a file.

    The file can be:
    - A simple text file with one word per line
    - A JSON file with a list of words or dictionary with 'stopwords' key
    - A CSV file with a header row and one word per line in the first column

    Parameters:
    -----------
    file_path : str
        Path to the stop words file
    encoding : str
        File encoding to use

    Returns:
    --------
    Set[str]
        Set of stop words
    """
    if not os.path.exists(file_path):
        logger.warning(f"Stop words file not found: {file_path}")
        return set()

    try:
        # Use file cache if available
        cache_key = f"stopwords_file:{file_path}:{encoding}"
        cached_data = file_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Handle based on file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        result = set()

        if file_extension == '.json':
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
                if isinstance(data, list):
                    result = set(word.lower().strip() for word in data if word)
                elif isinstance(data, dict) and 'stopwords' in data:
                    result = set(word.lower().strip() for word in data['stopwords'] if word)
                else:
                    logger.warning(f"Invalid JSON format in stop words file: {file_path}")

        elif file_extension == '.csv':
            import csv
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.reader(f)
                # Skip header row
                next(reader, None)
                result = set(row[0].lower().strip() for row in reader if row and row[0].strip())

        else:  # Default to text file with one word per line
            with open(file_path, 'r', encoding=encoding) as f:
                result = set(line.lower().strip() for line in f if line.strip())

        # Cache the result
        file_cache.set(cache_key, result, file_path=file_path)
        return result

    except Exception as e:
        logger.error(f"Error loading stop words from file {file_path}: {e}")
        return set()


def find_stopwords_files(language: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Find available stopwords files in all resource directories.

    Parameters:
    -----------
    language : str, optional
        Language code to filter files. If None, returns all files.

    Returns:
    --------
    List[Tuple[str, str]]
        List of tuples with (file_path, detected_language)
    """
    result = []

    # Get all stopwords directories
    stopwords_dirs = get_stopwords_dirs()

    for stopwords_dir in stopwords_dirs:
        if os.path.exists(stopwords_dir):
            # Find all text, JSON, and CSV files
            patterns = ['*.txt', '*.json', '*.csv']

            for pattern in patterns:
                for file_path in glob.glob(os.path.join(stopwords_dir, pattern)):
                    file_name = os.path.basename(file_path)
                    detected_lang = None

                    # Try to detect language from filename
                    file_stem = os.path.splitext(file_name)[0].lower()

                    # Common language code patterns in filenames
                    lang_patterns = {
                        'en': ['en', 'eng', 'english'],
                        'ru': ['ru', 'rus', 'russian'],
                        'fr': ['fr', 'fre', 'french'],
                        'de': ['de', 'ger', 'german'],
                        'es': ['es', 'spa', 'spanish'],
                        'nl': ['nl', 'dut', 'dutch'],
                        'ms': ['ms', 'may', 'malaysian'],
                        'uk': ['uk', 'ukr', 'ukrainian'],
                        'vi': ['vi', 'vie', 'vietnamese']
                    }

                    # Check for language codes in filename
                    for lang_code, patterns in lang_patterns.items():
                        if any(pattern in file_stem for pattern in patterns):
                            detected_lang = lang_code
                            break

                    # If language filter is specified, skip non-matching files
                    if language and detected_lang != language:
                        continue

                    result.append((file_path, detected_lang or 'unknown'))

            # Also check language subdirectories
            for lang_dir in glob.glob(os.path.join(stopwords_dir, '*')):
                if os.path.isdir(lang_dir):
                    lang_code = os.path.basename(lang_dir).lower()

                    # Skip if language filter is specified and doesn't match
                    if language and lang_code != language:
                        continue

                    for pattern in patterns:
                        for file_path in glob.glob(os.path.join(lang_dir, pattern)):
                            result.append((file_path, lang_code))

    return result


@cache_function(ttl=3600, cache_type='memory')
def get_nltk_stopwords(languages: List[str]) -> Set[str]:
    """
    Get stopwords from NLTK for the specified languages.

    Parameters:
    -----------
    languages : List[str]
        List of language codes

    Returns:
    --------
    Set[str]
        Set of stopwords
    """
    if not _NLTK_AVAILABLE:
        return set()

    stop_words = set()

    # Map common language codes to NLTK language names
    lang_map = {
        'en': 'english',
        'ru': 'russian',
        'fr': 'french',
        'de': 'german',
        'es': 'spanish',
        'it': 'italian',
        'pt': 'portuguese',
        'nl': 'dutch',
        'fi': 'finnish',
        'hu': 'hungarian',
        'da': 'danish',
        'no': 'norwegian',
        'sv': 'swedish',
        'tr': 'turkish'
    }

    for lang in languages:
        nltk_lang = lang_map.get(lang.lower(), lang.lower())

        try:
            lang_stopwords = set(nltk_stopwords.words(nltk_lang))
            stop_words.update(lang_stopwords)
            logger.debug(f"Loaded {len(lang_stopwords)} NLTK stopwords for language '{lang}'")
        except Exception as e:
            logger.debug(f"Failed to get NLTK stopwords for {lang}: {e}")

    return stop_words


@cache_function(ttl=3600, cache_type='memory')
def load_stopwords_from_sources(sources: List[Union[str, Set[str]]], encodings: Union[str, List[str]] = 'utf-8') -> Set[
    str]:
    """
    Load stopwords from multiple sources.

    Parameters:
    -----------
    sources : List[Union[str, Set[str]]]
        List of sources. Each source can be:
        - A file path (str)
        - A directory path (str) - all stopwords files in the directory will be loaded
        - A set of stopwords (Set[str])
    encodings : Union[str, List[str]]
        File encoding(s) to use. Can be a single encoding for all files or a list of encodings.

    Returns:
    --------
    Set[str]
        Combined set of stopwords from all sources
    """
    if not sources:
        return set()

    # Normalize encodings to a list
    if isinstance(encodings, str):
        encodings = [encodings] * len(sources)
    elif len(encodings) < len(sources):
        # If fewer encodings than sources, use the last encoding for remaining sources
        encodings = list(encodings) + [encodings[-1]] * (len(sources) - len(encodings))

    combined_stopwords = set()

    for i, source in enumerate(sources):
        encoding = encodings[i]

        if isinstance(source, set):
            # Source is already a set of stopwords
            combined_stopwords.update(word.lower().strip() for word in source if word)
        elif isinstance(source, str):
            if os.path.isdir(source):
                # Source is a directory, load all stopwords files
                for file_pattern in ['*.txt', '*.json', '*.csv']:
                    for file_path in glob.glob(os.path.join(source, file_pattern)):
                        combined_stopwords.update(load_stopwords_from_file(file_path, encoding))
            elif os.path.exists(source):
                # Source is a file
                combined_stopwords.update(load_stopwords_from_file(source, encoding))
            else:
                logger.warning(f"Stopwords source not found: {source}")

    return combined_stopwords


def get_stopwords(
        languages: Optional[List[str]] = None,
        custom_sources: Optional[List[Union[str, Set[str]]]] = None,
        include_defaults: bool = True,
        use_nltk: bool = True,
        encodings: Union[str, List[str]] = 'utf-8'
) -> Set[str]:
    """
    Get a comprehensive set of stopwords from multiple sources.

    Parameters:
    -----------
    languages : List[str], optional
        List of language codes. If None, defaults to ['en']
    custom_sources : List[Union[str, Set[str]]], optional
        Additional sources of stopwords. Each source can be:
        - A file path (str)
        - A directory path (str)
        - A set of stopwords (Set[str])
    include_defaults : bool
        Whether to include default stopwords files from the resources directory
    use_nltk : bool
        Whether to include NLTK stopwords if available
    encodings : Union[str, List[str]]
        File encoding(s) to use for custom sources

    Returns:
    --------
    Set[str]
        Combined set of stopwords
    """
    if languages is None:
        languages = ['en']

    # Normalize language codes to lowercase
    languages = [normalize_language_code(lang) for lang in languages]

    combined_stopwords = set()

    # 1. Include default stopwords files if requested
    if include_defaults:
        for language in languages:
            # Find language-specific stopwords files
            stopwords_files = find_stopwords_files(language)
            for file_path, _ in stopwords_files:
                combined_stopwords.update(load_stopwords_from_file(file_path))

    # 2. Include NLTK stopwords if requested and available
    if use_nltk and _NLTK_AVAILABLE:
        combined_stopwords.update(get_nltk_stopwords(languages))

    # 3. Include custom sources if provided
    if custom_sources:
        combined_stopwords.update(load_stopwords_from_sources(custom_sources, encodings))

    return combined_stopwords


def remove_stopwords(tokens: List[str], stop_words: Optional[Set[str]] = None,
                     languages: Optional[List[str]] = None,
                     custom_sources: Optional[List[Union[str, Set[str]]]] = None) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Parameters:
    -----------
    tokens : List[str]
        List of tokens to filter
    stop_words : Set[str], optional
        Set of stopwords to exclude. If None, gets default stopwords.
    languages : List[str], optional
        List of language codes for stopwords. Used only if stop_words is None.
    custom_sources : List[Union[str, Set[str]]], optional
        Additional sources of stopwords. Used only if stop_words is None.

    Returns:
    --------
    List[str]
        List of tokens with stopwords removed
    """
    if not tokens:
        return []

    # Get stopwords if not provided
    if stop_words is None:
        stop_words = get_stopwords(languages=languages, custom_sources=custom_sources)

    # Filter out stopwords
    return [token for token in tokens if token.lower() not in stop_words]


def save_stopwords_to_file(stop_words: Set[str], file_path: str, format: str = 'txt') -> bool:
    """
    Save a set of stopwords to a file.

    Parameters:
    -----------
    stop_words : Set[str]
        Set of stopwords to save
    file_path : str
        Path to save the stopwords file
    format : str
        File format: 'txt', 'json', or 'csv'

    Returns:
    --------
    bool
        True if the file was saved successfully, False otherwise
    """
    if not stop_words:
        logger.warning("No stopwords to save")
        return False

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Use lowercase sorted stopwords
        sorted_stopwords = sorted(word.lower().strip() for word in stop_words if word)

        # Save based on format
        if format.lower() == 'json' or file_path.endswith('.json'):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_stopwords, f, ensure_ascii=False, indent=2) # type: ignore

        elif format.lower() == 'csv' or file_path.endswith('.csv'):
            import csv
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)  # type: ignore
                writer.writerow(['stopword'])  # Header
                for word in sorted_stopwords:
                    writer.writerow([word])

        else:  # Default to text file with one word per line
            with open(file_path, 'w', encoding='utf-8') as f:
                for word in sorted_stopwords:
                    f.write(f"{word}\n")

        logger.info(f"Saved {len(sorted_stopwords)} stopwords to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving stopwords to file: {e}")
        return False


def combine_stopwords_files(
        input_files: List[str],
        output_file: str,
        encodings: Union[str, List[str]] = 'utf-8',
        format: str = 'txt'
) -> bool:
    """
    Combine multiple stopwords files into a single file.

    Parameters:
    -----------
    input_files : List[str]
        List of input file paths
    output_file : str
        Output file path
    encodings : Union[str, List[str]]
        File encoding(s) for input files
    format : str
        Output file format: 'txt', 'json', or 'csv'

    Returns:
    --------
    bool
        True if the combined file was saved successfully, False otherwise
    """
    combined_stopwords = load_stopwords_from_sources(input_files, encodings)
    return save_stopwords_to_file(combined_stopwords, output_file, format)


def create_external_stopwords_list(language: str, words: List[str], overwrite: bool = False) -> bool:
    """
    Create a new stopwords list in the external dictionaries directory.

    Parameters:
    -----------
    language : str
        Language code or name for the stopwords
    words : List[str]
        List of stopwords to save
    overwrite : bool
        Whether to overwrite an existing file

    Returns:
    --------
    bool
        True if the file was created successfully, False otherwise
    """
    # Normalize language to get standard code
    lang_code = normalize_language_code(language)

    # Get the external stopwords directory
    config_paths = get_config_paths()
    ext_stopwords_dir = config_paths["stopwords_dir"]

    # Ensure directory exists
    os.makedirs(ext_stopwords_dir, exist_ok=True)

    # Create file path
    filename = f"{lang_code}.txt"
    if lang_code == 'en':
        filename = "english.txt"
    elif lang_code == 'ru':
        filename = "russian.txt"
    elif lang_code == 'fr':
        filename = "french.txt"
    elif lang_code == 'de':
        filename = "german.txt"
    elif lang_code == 'es':
        filename = "spanish.txt"
    elif lang_code == 'nl':
        filename = "dutch.txt"

    file_path = os.path.join(ext_stopwords_dir, filename)

    # Check if file exists and we shouldn't overwrite
    if os.path.exists(file_path) and not overwrite:
        logger.warning(f"Stopwords file already exists: {file_path}. Use overwrite=True to replace.")
        return False

    # Save the stopwords
    return save_stopwords_to_file(set(words), file_path, 'txt')


def setup_nltk():
    """
    Ensure NLTK resources required for stopwords are available.

    Returns:
    --------
    bool
        True if all resources are available, False otherwise
    """
    if not _NLTK_AVAILABLE:
        logger.warning("NLTK is not available. Install it using: pip install nltk")
        return False

    try:
        import nltk
        resources = ['stopwords']

        for resource in resources:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

        return True
    except Exception as e:
        logger.error(f"Error setting up NLTK resources: {e}")
        return False


def list_available_stopwords() -> Dict[str, List[str]]:
    """
    List all available stopwords resources.

    Returns:
    --------
    Dict[str, List[str]]
        Dictionary of available stopwords resources by source
    """
    result = {
        "package": [],  # Internal package resources
        "external": [],  # External dictionaries
        "nltk": []  # NLTK resources
    }

    # Get all stopwords directories
    stopwords_dirs = get_stopwords_dirs()
    config_paths = get_config_paths()

    # Check package resources
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_resources_dir = os.path.join(package_dir, 'resources', 'stopwords')

    if os.path.exists(default_resources_dir):
        for file_pattern in ['*.txt', '*.json', '*.csv']:
            for file_path in glob.glob(os.path.join(default_resources_dir, file_pattern)):
                result["package"].append(os.path.basename(file_path))

    # Check external dictionaries
    external_dir = config_paths["stopwords_dir"]
    if os.path.exists(external_dir):
        for file_pattern in ['*.txt', '*.json', '*.csv']:
            for file_path in glob.glob(os.path.join(external_dir, file_pattern)):
                result["external"].append(os.path.basename(file_path))

    # Check NLTK resources
    if _NLTK_AVAILABLE:
        try:
            import nltk
            from nltk.corpus import stopwords as nltk_stopwords
            result["nltk"] = nltk_stopwords.fileids()
        except Exception as e:
            logger.debug(f"Error getting NLTK stopwords: {e}")

    return result