"""
Compatibility utilities for the NLP module.

This module handles checking and graceful degradation of NLP
functionality when dependencies are not available, by wrapping
DependencyManager with additional helpers.
"""

import logging
from typing import Dict, Any, Optional, List

# Import from base to avoid circular dependencies
from pamola_core.utils.nlp.base import DependencyManager
from pamola_core.utils.nlp.cache import cache_function

logger = logging.getLogger(__name__)


def check_dependency(module_name: str) -> bool:
    """
    Check if a specific Python module (dependency) is available.

    Thin wrapper around DependencyManager.check_dependency to provide
    a convenient function in this compatibility module.

    Parameters
    ----------
    module_name : str
        Name of the Python module to check

    Returns
    -------
    bool
        True if the module is installed and importable, else False
    """
    return DependencyManager.check_dependency(module_name)


def log_nlp_status() -> None:
    """
    Log the status of all NLP dependencies.
    """
    status = DependencyManager.get_nlp_status()

    logger.info("NLP dependencies status:")
    for module, available in status.items():
        logger.info(f"  - {module}: {'Available' if available else 'Not available'}")

    # Log overall status
    available_count = sum(status.values())
    total_count = len(status)
    logger.info(f"NLP availability: {available_count}/{total_count} dependencies available")


@cache_function(ttl=3600)
def dependency_info(verbose: bool = False) -> Dict[str, Any]:
    import sys
    basic_status = DependencyManager.get_nlp_status()

    result = {
        "available": basic_status,
        "count": {
            "total": len(basic_status),
            "available": sum(basic_status.values())
        },
        "python_version": sys.version,
        "system": sys.platform
    }

    if verbose:
        versions = {}
        details = {}

        for module_name, available in basic_status.items():
            if available:
                meets_req, version = DependencyManager.check_version(module_name)
                versions[module_name] = version

                # Initialize module_details
                module_details = {
                    "path": "Unknown",
                    "dependencies": []
                }

                try:
                    mod = DependencyManager.get_module(module_name)
                    module_details["path"] = getattr(mod, "__file__", "Unknown")

                    # Try to parse distribution metadata if Python >= 3.8
                    if sys.version_info >= (3, 8):
                        try:
                            # We do a broad `try/except ImportError` for the import
                            try:
                                import importlib.metadata as meta
                            except ImportError:
                                meta = None

                            if meta:
                                try:
                                    dist = meta.metadata(module_name)
                                    requires = dist.get_all('Requires-Dist')
                                    if requires:
                                        deps = []
                                        for req in requires:
                                            # If there's a semicolon, it might be a conditional
                                            if ';' in req:
                                                pkg_part, cond_part = req.split(';', 1)
                                                pkg_name = pkg_part.strip()
                                                spec = cond_part.strip()
                                            else:
                                                pkg_name = req.strip()
                                                spec = ""
                                            deps.append({"name": pkg_name, "specifier": spec})
                                        module_details["dependencies"] = deps
                                except (meta.PackageNotFoundError, AttributeError):
                                    pass
                                except Exception as e:
                                    logger.debug(f"Error processing metadata for {module_name}: {e}")
                        except Exception as e:
                            logger.debug(f"Could not import or use `importlib.metadata`: {e}")

                    details[module_name] = module_details

                except Exception as e:
                    details[module_name] = {"error": str(e)}

        result["versions"] = versions
        if verbose > 1:
            result["details"] = details

    return result



def get_best_available_module(module_preferences: List[str]) -> Optional[str]:
    """
    Return the first module from module_preferences that is installed,
    or None if none are installed.

    Parameters
    ----------
    module_preferences : List[str]
        List of module names in order of preference

    Returns
    -------
    str or None
        The name of the best available module or None
    """
    return DependencyManager.get_best_available_module(module_preferences)


def clear_dependency_cache() -> None:
    """
    Clear the dependency check cache from DependencyManager.
    """
    DependencyManager.clear_cache()
    logger.debug("Dependency cache cleared")


def check_nlp_requirements(requirements: Dict[str, List[str]]) -> Dict[str, bool]:
    """
    Check if specified NLP requirements are available.
    Each feature is considered available if at least one of its listed modules is installed.

    Parameters
    ----------
    requirements : Dict[str, List[str]]
        { feature_name: [list_of_required_modules] }

    Returns
    -------
    Dict[str, bool]
        { feature_name: True/False }
    """
    status = {}
    for feature, modules in requirements.items():
        # A feature is "available" if at least one required module is installed
        status[feature] = any(DependencyManager.check_dependency(m) for m in modules)
    return status


def setup_nlp_resources(download_if_missing: bool = True) -> Dict[str, bool]:
    """
    Check and (optionally) download key NLP resources (like NLTK or spaCy models).

    Parameters
    ----------
    download_if_missing : bool
        Whether to attempt auto-downloading missing data

    Returns
    -------
    Dict[str, bool]
        Mapping of resource_name -> bool (True if available after this check)
    """
    status = {}

    # Check NLTK resources
    if check_dependency('nltk'):
        import nltk
        nltk_resources = {
            'punkt': 'tokenizers/punkt',
            'wordnet': 'corpora/wordnet',
            'stopwords': 'corpora/stopwords'
        }
        for res_name, path in nltk_resources.items():
            try:
                nltk.data.find(path)
                status[f'nltk_{res_name}'] = True
            except LookupError:
                status[f'nltk_{res_name}'] = False
                if download_if_missing:
                    logger.info(f"Downloading NLTK resource: {res_name}")
                    try:
                        nltk.download(res_name, quiet=True)
                        status[f'nltk_{res_name}'] = True
                    except Exception as e:
                        logger.error(f"Failed to download NLTK resource {res_name}: {e}")

    # Check spaCy resources
    if check_dependency('spacy'):
        import spacy
        spacy_models = {
            'en_core_web_sm': 'English small model',
            'ru_core_news_sm': 'Russian small model'
        }
        for model_name, desc in spacy_models.items():
            try:
                spacy.load(model_name)
                status[f'spacy_{model_name}'] = True
            except OSError:
                status[f'spacy_{model_name}'] = False
                if download_if_missing:
                    logger.info(f"Downloading spaCy model: {model_name} ({desc})")
                    try:
                        spacy.cli.download(model_name)
                        # Just note success here; spacy.load check can be re-done as needed.
                        status[f'spacy_{model_name}'] = True
                    except Exception as e:
                        logger.error(f"Failed to download spaCy model {model_name}: {e}")

    return status
