"""
NLP model management module for the HHR project.

This module provides centralized management for NLP models with support for
caching, memory optimization, and graceful degradation when models are unavailable.
"""

import gc
import logging
import os
import sys
from typing import Dict, Any
from typing import Optional, List, Set, Callable

# Import from base to avoid circular dependencies
# Make sure these imports match your actual package structure
from pamola_core.utils.nlp.base import (
    DependencyManager,
    normalize_language_code,
    ModelNotAvailableError
)
from pamola_core.utils.nlp.cache import (
    get_cache,
    cache_function
)

logger = logging.getLogger(__name__)


class ModelLoadError(ModelNotAvailableError):
    """
    Exception raised when a model fails to load.
    """
    pass


class NLPModelManager:
    """
    Centralized manager for NLP models with caching and memory optimization.

    This class provides a unified interface for loading, accessing, and unloading
    various NLP models while ensuring memory efficiency and graceful fallback
    in case of missing libraries or uninstalled language models.
    """

    _instance = None

    def __new__(cls):
        """
        Enforce a singleton pattern for the NLPModelManager.
        """
        if cls._instance is None:
            cls._instance = super(NLPModelManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the model manager (only called once in the singleton pattern).
        Sets up caches, reads environment variables, and detects available libraries.
        """
        self._model_cache = get_cache('model')
        self._max_models = int(os.environ.get('HHR_MAX_MODELS', '5'))
        self._model_expiry = int(os.environ.get('HHR_MODEL_EXPIRY', '3600'))  # 1 hour in seconds
        self._nlp_libraries = self._check_available_libraries()

        # A dictionary mapping model_type strings to loader methods.
        # Each loader method should have the signature:
        #   (language: str, **kwargs) -> Any
        self._loaders: Dict[str, Callable[..., Any]] = {
            'spacy': self._load_spacy_model,
            'transformers': self._load_transformers_model,
            'ner': self._load_ner_model,
            'entity_extractor': self._load_entity_extractor
        }

        logger.info(
            "NLP Model Manager initialized. Available libraries: %s",
            ", ".join(sorted(self._nlp_libraries)) if self._nlp_libraries else "none"
        )

    def _check_available_libraries(self) -> Set[str]:
        """
        Check which NLP libraries are currently available in the environment.

        Returns:
        --------
        Set[str]
            Set of available NLP library names.
        """
        available = set()

        if DependencyManager.check_dependency('spacy'):
            available.add('spacy')
        if DependencyManager.check_dependency('nltk'):
            available.add('nltk')
        if DependencyManager.check_dependency('transformers'):
            available.add('transformers')
        if DependencyManager.check_dependency('torch'):
            available.add('torch')
        if DependencyManager.check_dependency('tensorflow'):
            available.add('tensorflow')

        return available

    def get_model(self, model_type: str, language: str, **params) -> Any:
        """
        Get or load a model of the specified type for the given language.
        First checks the cache; if not found, attempts to load a new model.

        Parameters:
        -----------
        model_type : str
            The type of model to retrieve (e.g., 'spacy', 'transformers', 'ner').
        language : str
            Language code (e.g., 'en', 'ru').
        **params : dict
            Additional parameters for model configuration.

        Returns:
        --------
        Any
            The loaded model instance.

        Raises:
        -------
        ModelNotAvailableError
            If the required library is not available for this model type.
        ModelLoadError
            If loading the model fails at runtime.
        """
        # Normalize language code to ensure consistency
        language = normalize_language_code(language)

        # Generate a unique key for caching
        model_key = self._get_model_key(model_type, language, params)

        # Attempt to retrieve from cache
        cached_model = self._model_cache.get(model_key)
        if cached_model is not None:
            return cached_model

        # Check if we actually have a loader for the requested model_type
        if model_type not in self._loaders:
            logger.warning("Unknown model type requested: '%s'", model_type)
            return None

        load_fn = self._loaders[model_type]

        # Check if required library is available for this model_type
        if not self._library_supported(model_type):
            raise ModelNotAvailableError(
                f"Library is not available for model type '{model_type}'. "
                f"Installed libraries: {', '.join(self._nlp_libraries)}"
            )

        # Attempt to load the model
        model = None
        try:
            model = load_fn(language, **params)
        except Exception as e:
            logger.error("Error loading model '%s' for language '%s': %s", model_type, language, e)
            raise ModelLoadError(f"Failed to load {model_type} model for {language}: {str(e)}")

        # If successfully loaded, store it in the cache
        if model is not None:
            metadata = {
                'type': model_type,
                'language': language,
                'params': str(params),
                'initialized_at': sys.version,  # Example of additional metadata
            }
            self._model_cache.set(model_key, model, metadata=metadata)

        return model

    def _library_supported(self, model_type: str) -> bool:
        """
        Check if the underlying library for the given model_type is present in the environment.

        Parameters:
        -----------
        model_type : str
            The requested model type.

        Returns:
        --------
        bool
            True if the required library is available, otherwise False.
        """
        if model_type in ('spacy', 'ner'):
            return 'spacy' in self._nlp_libraries
        elif model_type == 'transformers':
            return 'transformers' in self._nlp_libraries
        elif model_type == 'entity_extractor':
            # This might be library-agnostic or rely on spacy, etc.
            return True
        return False

    def _get_model_key(self, model_type: str, language: str, params: Dict[str, Any]) -> str:
        """
        Generate a unique key for identifying models in the cache.

        Parameters:
        -----------
        model_type : str
            The type of the model.
        language : str
            The normalized language code.
        params : Dict[str, Any]
            Model configuration parameters.

        Returns:
        --------
        str
            A unique cache key.
        """
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{model_type}_{language}_{param_str}"

    def _load_spacy_model(self, language: str, **kwargs) -> Any:
        """
        Load a spaCy model for the specified language. Installs it if missing.

        Parameters:
        -----------
        language : str
            Language code (e.g., 'en', 'ru').
        **kwargs : dict
            Accepts arbitrary parameters; if 'model_name' is provided, uses it instead of default.

        Returns:
        --------
        spacy.language.Language
            The loaded spaCy model.

        Raises:
        -------
        ModelLoadError
            If the spaCy model cannot be downloaded or loaded.
        """
        spacy_module = DependencyManager.get_module('spacy')
        if not spacy_module:
            raise ModelNotAvailableError("Failed to import spaCy even though it was detected.")

        # Extract model_name from kwargs if provided
        model_name = kwargs.pop("model_name", None)

        # Default model names by language
        default_map = {
            'en': 'en_core_web_sm',
            'ru': 'ru_core_news_sm',
            'de': 'de_core_news_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm'
        }
        if model_name is None:
            model_name = default_map.get(language.lower(), f"{language}_core_web_sm")

        try:
            return spacy_module.load(model_name)
        except OSError:
            # Attempt to download if not present
            logger.info("Attempting to download spaCy model: %s", model_name)
            try:
                spacy_module.cli.download(model_name)
                return spacy_module.load(model_name)
            except Exception as e:
                logger.error("Failed to download spaCy model %s: %s", model_name, e)
                # Try a multilingual fallback if available
                try:
                    return spacy_module.load("xx_ent_wiki_sm")
                except Exception as fallback_error:
                    raise ModelLoadError(
                        f"Failed to load any spaCy model for '{language}': {fallback_error}"
                    )

    def _load_transformers_model(self, language: str, **kwargs) -> Any:
        """
        Load a Hugging Face Transformers model and tokenizer.

        Parameters:
        -----------
        language : str
            Language code.
        **kwargs : dict
            Accepts arbitrary parameters:
            - 'model_name' (str): specific model name from the Hub,
            - 'task' (str): NLP task (classification, sequence-labeling, etc.).

        Returns:
        --------
        (transformers.PreTrainedModel, transformers.PreTrainedTokenizer)
            A tuple containing the loaded model and tokenizer.

        Raises:
        -------
        ModelLoadError
            If loading or downloading the model fails.
        """
        transformers_module = DependencyManager.get_module('transformers')
        if not transformers_module:
            raise ModelNotAvailableError("Failed to import 'transformers' library.")

        model_name = kwargs.pop("model_name", None)
        task = kwargs.pop("task", "classification")

        # Provide a default model if none is specified
        if model_name is None:
            if task == 'classification':
                model_name = 'bert-base-multilingual-cased'
            elif task == 'sequence-labeling':
                model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
            else:
                model_name = 'bert-base-multilingual-cased'

        try:
            tokenizer = transformers_module.AutoTokenizer.from_pretrained(model_name)
            model = transformers_module.AutoModel.from_pretrained(model_name)
            return (model, tokenizer)
        except Exception as e:
            raise ModelLoadError(f"Error loading transformers model '{model_name}': {str(e)}")

    def _load_ner_model(self, language: str, **kwargs) -> Any:
        """
        Load a Named Entity Recognition model. By default uses spaCy.

        Parameters:
        -----------
        language : str
            Language code.
        **kwargs : dict
            Additional parameters. 'model_type' or 'model_name' can be passed if needed.

        Returns:
        --------
        Any
            An NER-capable model or pipeline.

        Raises:
        -------
        ModelNotAvailableError
            If the required library is not available.
        ModelLoadError
            If loading the model fails.
        """
        # Decide which internal approach to use (e.g. spacy, huggingface, etc.)
        # but default is spaCy for now
        internal_model_type = kwargs.pop("model_type", "spacy")

        if internal_model_type == "spacy":
            if 'spacy' not in self._nlp_libraries:
                raise ModelNotAvailableError("spaCy is not available for NER.")
            # Reuse the spaCy loader
            spacy_model = self._load_spacy_model(language, **kwargs)
            if spacy_model is None:
                raise ModelLoadError(f"Failed to load spaCy for NER (lang='{language}').")
            return spacy_model

        raise ModelNotAvailableError(f"Unsupported NER model type '{internal_model_type}'.")

    def _load_entity_extractor(self, language: str, **kwargs) -> Any:
        """
        Load an entity extractor. This might rely on an external utility function.

        Parameters:
        -----------
        language : str
            Language code (not always necessary, depending on the extractor).
        **kwargs : dict
            Additional parameters, e.g. 'extractor_type': str for different strategies.

        Returns:
        --------
        Any
            The entity extractor instance.

        Raises:
        -------
        ModelLoadError
            If loading the extractor fails.
        """
        extractor_type = kwargs.pop("extractor_type", "auto")
        try:
            from pamola_core.utils.nlp.entity_extraction import create_entity_extractor
            return create_entity_extractor(extractor_type, language=language, **kwargs)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load entity extractor (type='{extractor_type}') for language '{language}': {str(e)}"
            )

    def unload_model(self, model_key: str) -> bool:
        """
        Unload a model from cache to free up memory.

        Parameters:
        -----------
        model_key : str
            The unique model key generated by _get_model_key.

        Returns:
        --------
        bool
            True if the model was in cache and successfully removed, False otherwise.
        """
        return self._model_cache.delete(model_key)

    def clear_models(self) -> None:
        """
        Unload all models and clear the cache entirely.
        Also triggers garbage collection.
        """
        logger.info("Clearing all cached NLP models.")
        self._model_cache.clear()
        gc.collect()

    def get_supported_model_types(self) -> List[str]:
        """
        Retrieve a list of supported model types based on detected libraries.

        Returns:
        --------
        List[str]
            A list of recognized model types that can be loaded.
        """
        supported_types = ['entity_extractor']  # Always possible by design

        if 'spacy' in self._nlp_libraries:
            supported_types.extend(['spacy', 'ner'])

        if 'transformers' in self._nlp_libraries:
            supported_types.append('transformers')

        if 'nltk' in self._nlp_libraries:
            supported_types.append('nltk')

        return supported_types

    def get_model_info(self, model_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about models stored in the cache.

        Parameters:
        -----------
        model_key : str, optional
            Specific model key for detailed info. If None, returns info for all.

        Returns:
        --------
        Dict[str, Any]
            A dictionary with metadata about the requested model(s).
        """
        return self._model_cache.get_model_info(model_key)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics (if psutil is available).

        Returns:
        --------
        Dict[str, Any]
            Memory usage information including RSS and VMS, or a fallback message.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms": memory_info.vms,
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent()
            }
        except ImportError:
            logger.warning("psutil is not installed, memory stats are not available.")
            return {"available": False}

    def set_max_models(self, max_models: int) -> None:
        """
        Configure the maximum number of models to keep loaded.

        Parameters:
        -----------
        max_models : int
            Maximum number of models allowed in memory.
        """
        if max_models < 1:
            logger.warning("Invalid 'max_models' value (%d). Using 1 instead.", max_models)
            max_models = 1
        self._max_models = max_models
        logger.info("Set maximum number of models to: %d", max_models)

    def set_model_expiry(self, expiry_seconds: int) -> None:
        """
        Configure time in seconds after which unused models may be unloaded.

        Parameters:
        -----------
        expiry_seconds : int
            Number of seconds to keep a model in the cache.
        """
        if expiry_seconds < 60:
            logger.warning("Invalid expiry value (%d). Using 60 seconds instead.", expiry_seconds)
            expiry_seconds = 60
        self._model_expiry = expiry_seconds
        logger.info("Set model expiry to: %d seconds", expiry_seconds)

    @cache_function(ttl=300, cache_type='memory')
    def check_model_availability(self, model_type: str, language: str) -> Dict[str, Any]:
        """
        Check if a model of the given type and language is potentially available without loading.

        Parameters
        ----------
        model_type : str
            The type of model (e.g., 'spacy', 'ner', 'transformers', 'nltk').
        language : str
            The language code (e.g. 'en', 'ru').

        Returns
        -------
        Dict[str, Any]
            Information about the model's potential availability.
        """
        # Normalize the language code
        language = normalize_language_code(language)

        # Prepare a result dictionary with some default fields
        result = {
            "model_type": model_type,
            "language": language,
            "library_available": False,
            "potentially_available": False,
            "details": {}
        }

        # ----------------------------------------------------------------
        # 1) spacy / ner
        # ----------------------------------------------------------------
        if model_type in ("spacy", "ner"):
            # We consider that 'ner' is typically powered by spaCy in this code
            result["library_available"] = "spacy" in self._nlp_libraries
            if result["library_available"]:
                # We can import spacy safely if library is known to be installed
                import spacy

                # For demonstration, define a default map for common languages
                default_map = {
                    'en': 'en_core_web_sm',
                    'ru': 'ru_core_news_sm',
                    'de': 'de_core_news_sm',
                    'fr': 'fr_core_news_sm',
                    'es': 'es_core_news_sm'
                }
                model_name = default_map.get(language, f"{language}_core_web_sm")

                try:
                    # Check if model is already installed locally
                    spacy.util.get_package_path(model_name)
                    result["potentially_available"] = True
                    result["details"]["model_name"] = model_name
                    result["details"]["installed"] = True
                except (OSError, AttributeError):
                    # If not installed, spaCy can still download it
                    result["potentially_available"] = True
                    result["details"]["model_name"] = model_name
                    result["details"]["installed"] = False

        # ----------------------------------------------------------------
        # 2) transformers
        # ----------------------------------------------------------------
        elif model_type == "transformers":
            result["library_available"] = "transformers" in self._nlp_libraries
            if result["library_available"]:
                # If library is installed, we can import it
                # Here you might do more advanced checks, e.g. if a certain model is in cache, etc.
                # By default, HuggingFace models can be downloaded from the Hub, so:
                result["potentially_available"] = True
                # For demonstration, we can store some detail about "online_download"
                result["details"]["online_download"] = True

        # ----------------------------------------------------------------
        # 3) nltk
        # ----------------------------------------------------------------
        elif model_type == "nltk":
            result["library_available"] = "nltk" in self._nlp_libraries
            if result["library_available"]:
                import nltk
                result["potentially_available"] = True

                # Optionally check specific resources (punkt, wordnet, stopwords)
                resources_to_check = {
                    'punkt': 'tokenizers/punkt',
                    'wordnet': 'corpora/wordnet',
                    'stopwords': 'corpora/stopwords'
                }
                resources_status = {}
                for name, path in resources_to_check.items():
                    try:
                        nltk.data.find(path)
                        resources_status[name] = True
                    except LookupError:
                        resources_status[name] = False

                result["details"]["resources"] = resources_status

        # ----------------------------------------------------------------
        # Otherwise
        # ----------------------------------------------------------------
        else:
            # If the model_type is unrecognized or not specifically handled
            # library_available remains False, potentially_available remains False
            pass

        return result


# Create a singleton instance for convenience
nlp_model_manager = NLPModelManager()
