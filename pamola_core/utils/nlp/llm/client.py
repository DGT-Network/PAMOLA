"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Client Interface
Package:       pamola_core.utils.nlp.llm.client
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides a unified client interface for connecting to and
communicating with Large Language Models (LLMs) through various APIs.
Currently supports LM Studio WebSocket connections with plans to extend
to other providers.

Key Features:
- WebSocket-based communication with LM Studio
- Automatic connection management and retry logic
- Support for multiple LLM providers through unified interface
- Thread-safe operations with connection pooling
- Comprehensive error handling and logging
- Performance monitoring and slow request detection
- Debug logging capabilities for troubleshooting

Framework:
Part of PAMOLA.CORE NLP utilities, providing low-level LLM connectivity
for higher-level text processing operations.

Changelog:
1.1.0 - Unified error classes, removed duplicate definitions
     - Fixed _supports_streaming initialization
     - Improved method calling logic
     - Better error handling for missing methods
1.0.0 - Initial implementation

Dependencies:
- Standard library only for core functionality
- Optional: lmstudio package for LM Studio support

TODO:
- Add support for OpenAI API
- Add support for Anthropic API
- Add support for local llama.cpp integration
- Implement connection pooling for high-throughput scenarios
- Add async/await support when providers support it
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, cast
from urllib.parse import urlparse, urlunparse

# Import error classes from base module to avoid duplication
from pamola_core.utils.nlp.base import (
    DependencyManager,
    LLMConnectionError,
    LLMResponseError
)

# Configure logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
SLOW_REQUEST_THRESHOLD = 5.0  # seconds for performance monitoring


class LLMProtocol(Protocol):
    """Protocol defining the interface for LLM model objects."""

    def respond(self, prompt: str, **kwargs) -> Any:
        """Generate response for prompt with optional parameters."""
        ...

    def generate(self, prompt: str, **kwargs) -> Any:
        """Alternative method for generating response."""
        ...

    def close(self) -> None:
        """Close connection (optional)."""
        ...

    def shutdown(self) -> None:
        """Shutdown connection (optional)."""
        ...


@dataclass
class LLMGenerationParams:
    """Parameters for text generation."""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 512
    repeat_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False



    def to_dict(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "stream": self.stream,
        }
        if self.stop_sequences:
            params["stop"] = self.stop_sequences
        return params


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    text: str
    raw_response: Any
    model_name: str
    response_time: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM client implementations."""

    def __init__(
            self,
            model_name: str,
            timeout: int = DEFAULT_TIMEOUT,
            max_retries: int = DEFAULT_MAX_RETRIES,
            retry_delay: float = DEFAULT_RETRY_DELAY,
            debug_logging: bool = False,
            debug_log_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize base LLM client.

        Parameters
        ----------
        model_name : str
            Name of the model to use
        timeout : int
            Request timeout in seconds
        max_retries : int
            Maximum number of retry attempts
        retry_delay : float
            Delay between retries in seconds
        debug_logging : bool
            Enable debug logging of requests/responses
        debug_log_file : Path, optional
            Path to debug log file
        """
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Debug logging setup
        self._debug_logger: Optional[logging.Logger] = None
        if debug_logging:
            self._setup_debug_logging(debug_log_file)

        # Performance monitoring
        self._request_count = 0
        self._slow_request_count = 0
        self._total_response_time = 0.0

        # Streaming support flag (to be set by subclasses)
        self._supports_streaming = False

    def _setup_debug_logging(self, debug_log_file: Optional[Union[str, Path]] = None) -> None:
        """Set up debug logging for requests and responses."""
        self._debug_logger = logging.getLogger(f"{__name__}.debug")
        self._debug_logger.setLevel(logging.DEBUG)

        # Determine log file path
        if debug_log_file:
            log_path = Path(debug_log_file)
        else:
            # Default to logs directory
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"llm_debug_{timestamp}.log"

        # Create file handler
        handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler
        self._debug_logger.addHandler(handler)
        self._debug_logger.info(f"Debug logging initialized. Log file: {log_path}")

    def _log_debug(self, request_type: str, content: str,
                   additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Log debug information for requests and responses."""
        if self._debug_logger:
            self._debug_logger.debug(f"\n{'=' * 60}")
            self._debug_logger.debug(f"{request_type} - {datetime.now().isoformat()}")
            if additional_info:
                for key, value in additional_info.items():
                    self._debug_logger.debug(f"{key}: {value}")
            self._debug_logger.debug(f"Content:\n{content}")
            self._debug_logger.debug(f"{'=' * 60}\n")

    def _monitor_performance(self, response_time: float) -> None:
        """Monitor request performance."""
        self._request_count += 1
        self._total_response_time += response_time

        if response_time > SLOW_REQUEST_THRESHOLD:
            self._slow_request_count += 1
            logger.warning(
                f"Slow LLM response detected: {response_time:.2f}s "
                f"(threshold: {SLOW_REQUEST_THRESHOLD}s)"
            )

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the LLM service."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the LLM service."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if client is connected."""
        pass

    @abstractmethod
    def _send_request(self, prompt: str, params: LLMGenerationParams) -> LLMResponse:
        """Send request to LLM service (implementation-specific)."""
        pass

    def generate(self, prompt: str, params: Optional[LLMGenerationParams] = None) -> LLMResponse:
        """
        Generate text from prompt with retry logic.

        Parameters
        ----------
        prompt : str
            Input prompt
        params : LLMGenerationParams, optional
            Generation parameters

        Returns
        -------
        LLMResponse
            Response from LLM

        Raises
        ------
        LLMConnectionError
            If all retry attempts fail
        """
        if params is None:
            params = LLMGenerationParams()

        # Log request if debug enabled
        if self._debug_logger:
            self._log_debug("REQUEST", prompt, {
                "model": self.model_name,
                "parameters": params.to_dict()
            })

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                # Ensure connection
                if not self.is_connected():
                    self.connect()

                # Time the request
                start_time = time.time()

                # Send request
                response = self._send_request(prompt, params)

                # Monitor performance
                response_time = time.time() - start_time
                response.response_time = response_time
                self._monitor_performance(response_time)

                # Log response if debug enabled
                if self._debug_logger:
                    self._log_debug("RESPONSE", response.text, {
                        "model": response.model_name,
                        "response_time": f"{response_time:.3f}s",
                        "tokens": {
                            "prompt": response.prompt_tokens,
                            "completion": response.completion_tokens,
                            "total": response.total_tokens
                        },
                        "metadata": response.metadata
                    })

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)

                    # Try to reconnect
                    try:
                        self.disconnect()
                        self.connect()
                    except Exception as reconnect_error:
                        logger.error(f"Reconnection failed: {reconnect_error}")

        # All retries failed
        error_msg = f"Failed after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise LLMConnectionError(error_msg)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        avg_response_time = (
            self._total_response_time / self._request_count
            if self._request_count > 0 else 0
        )

        return {
            "model_name": self.model_name,
            "request_count": self._request_count,
            "slow_request_count": self._slow_request_count,
            "average_response_time": avg_response_time,
            "slow_request_threshold": SLOW_REQUEST_THRESHOLD,
            "supports_streaming": self._supports_streaming
        }

    def close(self) -> None:
        """Clean up resources."""
        try:
            self.disconnect()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")

        # Close debug logger
        if self._debug_logger:
            self._debug_logger.info("Debug logging session ended")
            for handler in self._debug_logger.handlers[:]:
                handler.close()
                self._debug_logger.removeHandler(handler)


class LMStudioClient(BaseLLMClient):
    """Client for LM Studio WebSocket API."""

    def __init__(
            self,
            websocket_url: str,
            model_name: str,
            api_key: Optional[str] = None,
            ttl: int = 86400,
            **kwargs
    ):
        """
        Initialize LM Studio client.

        Parameters
        ----------
        websocket_url : str
            WebSocket URL for LM Studio
        model_name : str
            Name of the model in LM Studio
        api_key : str, optional
            API key (LM Studio requires one even if dummy)
        ttl : int
            Time-to-live for model cache in seconds
        **kwargs
            Additional parameters for base class
        """
        super().__init__(model_name=model_name, **kwargs)

        self.websocket_url = self._convert_to_websocket_url(websocket_url)
        self.api_key = api_key or os.environ.get("LM_STUDIO_API_KEY", "dummy-key")
        self.ttl = ttl
        self.model: Optional[LLMProtocol] = None

        # Store original environment variables to restore later
        self._original_env: Dict[str, Optional[str]] = {}

    def _convert_to_websocket_url(self, url: str) -> str:
        """Convert HTTP URL to WebSocket URL."""
        parsed = urlparse(url)

        # Convert scheme
        if parsed.scheme == 'http':
            parsed = parsed._replace(scheme='ws')
        elif parsed.scheme == 'https':
            parsed = parsed._replace(scheme='wss')
        elif parsed.scheme in ('ws', 'wss'):
            return url

        # Convert path if needed
        if '/v1' in parsed.path:
            new_path = parsed.path.replace('/v1', '/llm')
            parsed = parsed._replace(path=new_path)
        elif not parsed.path.endswith('/llm'):
            # Ensure path ends with /llm
            new_path = parsed.path.rstrip('/') + '/llm'
            parsed = parsed._replace(path=new_path)

        return cast(str, urlunparse(parsed))

    def _check_streaming_support(self) -> bool:
        """Check if the model supports streaming."""
        if self.model is None:
            return False

        try:
            # Check for _client attribute (LM Studio specific)
            if hasattr(self.model, '_client'):
                client = getattr(self.model, '_client')
                if hasattr(client, 'stream_supported'):
                    return bool(getattr(client, 'stream_supported'))

            # Check for stream_supported directly on model
            if hasattr(self.model, 'stream_supported'):
                return bool(getattr(self.model, 'stream_supported'))

            return False
        except Exception as e:
            logger.debug(f"Error checking streaming support: {e}")
            return False

    def connect(self) -> None:
        """Connect to LM Studio."""
        lms = DependencyManager.get_module('lmstudio')
        if lms is None:
            raise LLMConnectionError("lmstudio package not available")

        try:
            # Store original environment variables
            self._original_env['LM_STUDIO_WEBSOCKET_URL'] = os.environ.get('LM_STUDIO_WEBSOCKET_URL')
            self._original_env['LM_STUDIO_API_KEY'] = os.environ.get('LM_STUDIO_API_KEY')

            # Set environment variables
            os.environ["LM_STUDIO_WEBSOCKET_URL"] = self.websocket_url
            os.environ["LM_STUDIO_API_KEY"] = self.api_key

            # Initialize model
            self.model = lms.llm(self.model_name, ttl=self.ttl)

            # Check streaming support
            self._supports_streaming = self._check_streaming_support()

            logger.info(
                f"Connected to LM Studio: {self.model_name} "
                f"(streaming: {self._supports_streaming})"
            )

        except Exception as e:
            # Restore environment variables on failure
            self._restore_environment()
            raise LLMConnectionError(f"Failed to connect to LM Studio: {e}")

    def _restore_environment(self) -> None:
        """Restore original environment variables."""
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def disconnect(self) -> None:
        """Disconnect from LM Studio."""
        if self.model is not None:
            try:
                if hasattr(self.model, 'close'):
                    self.model.close()
                elif hasattr(self.model, 'shutdown'):
                    self.model.shutdown()
            except Exception as e:
                logger.warning(f"Error closing LM Studio connection: {e}")
            finally:
                self.model = None

        # Restore original environment variables
        self._restore_environment()

    def is_connected(self) -> bool:
        """Check if connected to LM Studio."""
        return self.model is not None

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text from LM Studio response."""
        # Handle None response
        if response is None:
            raise LLMResponseError("LLM returned None response")

        # If already a string, return it
        if isinstance(response, str):
            return response.strip()

        # Try common attribute names
        for attr in ['text', 'content', 'result', 'message', 'output', 'response']:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if value is not None:
                    return str(value).strip()

        # Try dictionary access
        if hasattr(response, '__getitem__'):
            for key in ['text', 'content', 'result', 'message', 'output', 'response']:
                try:
                    value = response[key]
                    if value is not None:
                        return str(value).strip()
                except (KeyError, TypeError):
                    continue

        # Last resort: convert to string
        try:
            text = str(response)
            if not text.startswith('<') and not text.startswith('object at'):
                return text.strip()
        except Exception:
            pass

        raise LLMResponseError(
            f"Cannot extract text from response of type {type(response).__name__}"
        )

    def _send_request(self, prompt: str, params: LLMGenerationParams) -> LLMResponse:
        """Send request to LM Studio."""
        if self.model is None:
            raise LLMConnectionError("Not connected to LM Studio")

        # Prepare parameters
        api_params = params.to_dict()

        # Remove stream parameter if not supported
        if not self._supports_streaming and 'stream' in api_params:
            api_params.pop('stream')

        # Try to call model with parameters
        raw_response = None
        method_used = None
        params_accepted = True

        try:
            # Determine which method to use
            if hasattr(self.model, 'respond'):
                method_used = 'respond'
                raw_response = self.model.respond(prompt, **api_params)
            elif hasattr(self.model, 'generate'):
                method_used = 'generate'
                raw_response = self.model.generate(prompt, **api_params)
            else:
                raise AttributeError("Model has no 'respond' or 'generate' method")

        except TypeError as e:
            # Model doesn't accept these parameters, try without
            logger.warning(f"Model doesn't accept generation parameters: {e}")
            params_accepted = False

            try:
                if method_used == 'respond':
                    raw_response = self.model.respond(prompt)
                elif method_used == 'generate':
                    raw_response = self.model.generate(prompt)
                else:
                    # Try both methods without parameters
                    if hasattr(self.model, 'respond'):
                        method_used = 'respond'
                        raw_response = self.model.respond(prompt)
                    elif hasattr(self.model, 'generate'):
                        method_used = 'generate'
                        raw_response = self.model.generate(prompt)
                    else:
                        raise AttributeError("Model has no 'respond' or 'generate' method")
            except Exception as e:
                raise LLMConnectionError(f"Failed to call model method: {e}")

        # Extract text from response
        text = self._extract_text_from_response(raw_response)

        # Create structured response
        return LLMResponse(
            text=text,
            raw_response=raw_response,
            model_name=self.model_name,
            response_time=0.0,  # Will be set by caller
            finish_reason=method_used,
            metadata={
                'method_used': method_used,
                'params_accepted': params_accepted,
                'streaming_enabled': self._supports_streaming
            }
        )


def create_llm_client(
        provider: str,
        **kwargs
) -> BaseLLMClient:
    """
    Factory function to create LLM client.

    Parameters
    ----------
    provider : str
        LLM provider name ('lmstudio', 'openai', etc.)
    **kwargs
        Provider-specific parameters

    Returns
    -------
    BaseLLMClient
        Configured LLM client

    Raises
    ------
    ValueError
        If provider is not supported
    """
    provider = provider.lower()

    if provider == 'lmstudio':
        required = ['websocket_url', 'model_name']
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter for LM Studio: {param}")
        return LMStudioClient(**kwargs)

    # Add other providers here as they are implemented
    # elif provider == 'openai':
    #     return OpenAIClient(**kwargs)
    # elif provider == 'anthropic':
    #     return AnthropicClient(**kwargs)

    else:
        supported = ['lmstudio']  # Add more as implemented
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {', '.join(supported)}"
        )