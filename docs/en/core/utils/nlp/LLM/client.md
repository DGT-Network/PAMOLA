# PAMOLA.CORE LLM Client Module Documentation

**Module:** `pamola_core.utils.nlp.llm.client`  
**Version:** 1.1.0  
**Status:** Stable  
**Author:** PAMOLA Core Team  
**Created:** 2025  
**License:** BSD 3-Clause

## 1. Purpose and Overview

The LLM Client module provides a unified interface for connecting to and communicating with Large Language Models (LLMs) through various APIs. It serves as the foundational connectivity layer for all LLM operations within the PAMOLA.CORE framework, abstracting the complexities of different LLM providers behind a consistent interface.

This module is designed to support multiple LLM providers while maintaining a single, predictable API for upstream operations. Currently, it provides full support for LM Studio with a WebSocket-based implementation, with the architecture ready to accommodate additional providers like OpenAI, Anthropic, and local llama.cpp integrations.

## 2. Key Features

### Core Capabilities
- **Unified Interface**: Abstract base class defining consistent LLM interaction patterns
- **Provider Abstraction**: Support for multiple LLM providers through a factory pattern
- **Connection Management**: Automatic connection establishment, health checking, and reconnection
- **Retry Logic**: Configurable retry mechanisms with exponential backoff
- **Performance Monitoring**: Request timing, slow request detection, and statistics collection
- **Debug Logging**: Comprehensive request/response logging for troubleshooting

### Advanced Features
- **Thread-Safe Operations**: Safe for use in multi-threaded environments
- **Environment Management**: Automatic handling of provider-specific environment variables
- **Streaming Support Detection**: Automatic detection of streaming capabilities
- **Flexible Response Extraction**: Robust extraction of text from various response formats
- **Protocol Compliance**: Type-safe interfaces using Python protocols

## 3. Architecture

### Module Structure

```
pamola_core/utils/nlp/llm/client.py
├── Imports and Dependencies
├── Constants and Configuration
├── Protocols
│   └── LLMProtocol
├── Data Classes
│   ├── LLMGenerationParams
│   └── LLMResponse
├── Abstract Base Class
│   └── BaseLLMClient
├── Concrete Implementations
│   └── LMStudioClient
└── Factory Functions
    └── create_llm_client
```

### Class Hierarchy

```
BaseLLMClient (ABC)
    ├── Abstract Methods
    │   ├── connect()
    │   ├── disconnect()
    │   ├── is_connected()
    │   └── _send_request()
    ├── Concrete Methods
    │   ├── generate()
    │   ├── get_stats()
    │   └── close()
    └── Implementations
        └── LMStudioClient
```

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│         (Text Processing, Anonymization, etc.)              │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│                     LLM Service Layer                       │
│              (pamola_core.utils.nlp.llm.service)                   │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│                    LLM Client Layer                         │
│              (pamola_core.utils.nlp.llm.client)                    │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ BaseLLMClient  │  │ LMStudioClient │  │ Future Impl  │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│                   External LLM Services                     │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │  LM Studio   │  │  OpenAI  │  │  Anthropic/Others   │   │
│  └──────────────┘  └──────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 4. Dependencies

### Required Dependencies
- **Standard Library**:
  - `logging`: For logging functionality
  - `os`: Environment variable management
  - `time`: Performance monitoring and delays
  - `abc`: Abstract base classes
  - `dataclasses`: Data structure definitions
  - `datetime`: Timestamp generation
  - `pathlib`: Path handling
  - `typing`: Type annotations
  - `urllib.parse`: URL manipulation

### External Dependencies
- **From PAMOLA.CORE**:
  - `pamola_core.utils.nlp.base.DependencyManager`: Optional dependency management
  - `pamola_core.utils.nlp.base.LLMConnectionError`: Connection error handling
  - `pamola_core.utils.nlp.base.LLMResponseError`: Response error handling

### Optional Dependencies
- **lmstudio**: Required only when using LM Studio provider

## 5. Core API Reference

### Data Classes

#### LLMGenerationParams

```python
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
        """Convert parameters to dictionary for API calls."""
```

**Parameters:**
- `temperature`: Controls randomness (0.0-1.0)
- `top_p`: Nucleus sampling threshold
- `top_k`: Top-k sampling parameter
- `max_tokens`: Maximum tokens to generate
- `repeat_penalty`: Penalty for repeated tokens
- `stop_sequences`: List of sequences that stop generation
- `stream`: Enable streaming responses

#### LLMResponse

```python
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
```

**Fields:**
- `text`: Generated text response
- `raw_response`: Original response object from provider
- `model_name`: Name of the model used
- `response_time`: Time taken for response in seconds
- `prompt_tokens`: Number of tokens in prompt
- `completion_tokens`: Number of generated tokens
- `total_tokens`: Total token count
- `finish_reason`: Reason for completion
- `metadata`: Additional provider-specific metadata

### Abstract Base Class

#### BaseLLMClient

```python
class BaseLLMClient(ABC):
    """Abstract base class for LLM client implementations."""
    
    def __init__(
        self,
        model_name: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        debug_logging: bool = False,
        debug_log_file: Optional[Union[str, Path]] = None
    ):
        """Initialize base LLM client."""
```

**Parameters:**
- `model_name`: Name of the model to use
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum number of retry attempts
- `retry_delay`: Initial delay between retries
- `debug_logging`: Enable debug logging
- `debug_log_file`: Path to debug log file

**Abstract Methods:**

```python
@abstractmethod
def connect(self) -> None:
    """Establish connection to the LLM service."""

@abstractmethod
def disconnect(self) -> None:
    """Close connection to the LLM service."""

@abstractmethod
def is_connected(self) -> bool:
    """Check if client is connected."""

@abstractmethod
def _send_request(self, prompt: str, params: LLMGenerationParams) -> LLMResponse:
    """Send request to LLM service (implementation-specific)."""
```

**Concrete Methods:**

```python
def generate(self, prompt: str, params: Optional[LLMGenerationParams] = None) -> LLMResponse:
    """Generate text from prompt with retry logic."""

def get_stats(self) -> Dict[str, Any]:
    """Get client statistics."""

def close(self) -> None:
    """Clean up resources."""
```

### Concrete Implementations

#### LMStudioClient

```python
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
        """Initialize LM Studio client."""
```

**Parameters:**
- `websocket_url`: WebSocket URL for LM Studio
- `model_name`: Name of the model in LM Studio
- `api_key`: API key (can be dummy for LM Studio)
- `ttl`: Time-to-live for model cache in seconds
- `**kwargs`: Additional parameters for base class

### Factory Function

```python
def create_llm_client(provider: str, **kwargs) -> BaseLLMClient:
    """
    Factory function to create LLM client.
    
    Parameters:
    -----------
    provider : str
        LLM provider name ('lmstudio', 'openai', etc.)
    **kwargs
        Provider-specific parameters
        
    Returns:
    --------
    BaseLLMClient
        Configured LLM client
    """
```

## 6. Usage Examples

### Basic Usage

```python
from pamola_core.utils.nlp.llm.client import create_llm_client, LLMGenerationParams

# Create client for LM Studio
client = create_llm_client(
    provider='lmstudio',
    websocket_url='http://localhost:1234/v1',
    model_name='mistral-7b-instruct'
)

try:
    # Connect to the service
    client.connect()
    
    # Generate text with default parameters
    response = client.generate("What is the capital of France?")
    print(f"Response: {response.text}")
    print(f"Response time: {response.response_time:.2f}s")
    
finally:
    # Always disconnect
    client.disconnect()
```

### Custom Generation Parameters

```python
from pamola_core.utils.nlp.llm.client import create_llm_client, LLMGenerationParams

# Create client
client = create_llm_client(
    provider='lmstudio',
    websocket_url='ws://localhost:1234/llm',
    model_name='llama-2-7b-chat'
)

# Configure generation parameters
params = LLMGenerationParams(
    temperature=0.3,        # Lower temperature for more focused responses
    top_p=0.9,
    max_tokens=256,
    repeat_penalty=1.1,
    stop_sequences=["Human:", "User:"]
)

# Use context manager for automatic cleanup
with client:
    client.connect()
    
    # Generate response
    response = client.generate(
        "Explain quantum computing in simple terms.",
        params=params
    )
    
    print(f"Model: {response.model_name}")
    print(f"Response: {response.text}")
    
    # Get client statistics
    stats = client.get_stats()
    print(f"Average response time: {stats['average_response_time']:.2f}s")
```

### Debug Logging

```python
from pamola_core.utils.nlp.llm.client import create_llm_client
from pathlib import Path

# Create client with debug logging
client = create_llm_client(
    provider='lmstudio',
    websocket_url='http://localhost:1234/v1',
    model_name='codellama-7b',
    debug_logging=True,
    debug_log_file=Path("logs/llm_debug.log")
)

client.connect()

# All requests and responses will be logged
response = client.generate("Write a Python function to sort a list")

client.disconnect()
```

### Error Handling with Retries

```python
from pamola_core.utils.nlp.llm.client import create_llm_client, LLMConnectionError

# Create client with custom retry configuration
client = create_llm_client(
    provider='lmstudio',
    websocket_url='http://localhost:1234/v1',
    model_name='mistral-7b',
    max_retries=5,
    retry_delay=2.0,  # 2 seconds initial delay
    timeout=60        # 60 seconds timeout
)

try:
    client.connect()
    
    # This will automatically retry on failures
    response = client.generate("Complex prompt that might fail")
    
except LLMConnectionError as e:
    print(f"Failed after all retries: {e}")
    # Handle permanent failure
    
finally:
    client.close()  # Alias for disconnect with cleanup
```

### Performance Monitoring

```python
from pamola_core.utils.nlp.llm.client import create_llm_client
import time

client = create_llm_client(
    provider='lmstudio',
    websocket_url='http://localhost:1234/v1',
    model_name='phi-2'
)

client.connect()

# Generate multiple responses
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What are transformers in AI?"
]

for prompt in prompts:
    response = client.generate(prompt)
    print(f"Prompt: {prompt[:30]}...")
    print(f"Response time: {response.response_time:.2f}s")
    time.sleep(1)  # Avoid overwhelming the service

# Get performance statistics
stats = client.get_stats()
print("\nPerformance Statistics:")
print(f"Total requests: {stats['request_count']}")
print(f"Average response time: {stats['average_response_time']:.2f}s")
print(f"Slow requests (>{stats['slow_request_threshold']}s): {stats['slow_request_count']}")

client.disconnect()
```

## 7. Error Handling

The module uses custom exception classes for clear error reporting:

### LLMConnectionError
Raised when:
- Connection to LLM service fails
- All retry attempts are exhausted
- Required dependencies are missing
- Invalid provider specified

### LLMResponseError
Raised when:
- Response extraction fails
- Invalid response format received
- Model returns None or empty response

### Example Error Handling

```python
from pamola_core.utils.nlp.llm.client import (
    create_llm_client, 
    LLMConnectionError, 
    LLMResponseError
)

client = create_llm_client(
    provider='lmstudio',
    websocket_url='http://localhost:1234/v1',
    model_name='llama-2-7b'
)

try:
    client.connect()
    response = client.generate("Test prompt")
    
except LLMConnectionError as e:
    # Handle connection issues
    print(f"Connection error: {e}")
    # Maybe try alternative service or fallback
    
except LLMResponseError as e:
    # Handle response parsing issues
    print(f"Response error: {e}")
    # Maybe retry with different parameters
    
finally:
    if client.is_connected():
        client.disconnect()
```

## 8. Best Practices

### Connection Management
1. **Always disconnect**: Use try-finally or context managers
2. **Check connection status**: Use `is_connected()` before operations
3. **Handle reconnection**: Implement reconnection logic for long-running processes

### Performance Optimization
1. **Reuse connections**: Don't create new clients for each request
2. **Batch requests**: Group related prompts when possible
3. **Monitor performance**: Use `get_stats()` to track performance
4. **Set appropriate timeouts**: Balance between reliability and responsiveness

### Debug and Monitoring
1. **Enable debug logging**: For development and troubleshooting
2. **Monitor slow requests**: Adjust thresholds based on your needs
3. **Log statistics**: Periodically log client statistics for monitoring

### Error Handling
1. **Catch specific exceptions**: Handle `LLMConnectionError` and `LLMResponseError` separately
2. **Implement fallbacks**: Have backup strategies for failures
3. **Log errors**: Maintain error logs for debugging

## 9. Future Enhancements

### Planned Features
1. **Additional Providers**: OpenAI, Anthropic, Hugging Face integrations
2. **Async Support**: Async/await patterns for better concurrency
3. **Connection Pooling**: For high-throughput scenarios
4. **Request Queuing**: Built-in request queue management
5. **Response Caching**: Optional caching layer
6. **Streaming Support**: Full streaming response handling

### Provider Extensions
The module is designed for easy extension. To add a new provider:

1. Create a new class inheriting from `BaseLLMClient`
2. Implement the four abstract methods
3. Add the provider to the factory function
4. Update documentation

## 10. Summary

The LLM Client module provides a robust, extensible foundation for LLM integration within the PAMOLA.CORE framework. Its abstract interface ensures consistency across providers while allowing for provider-specific optimizations. The built-in retry logic, performance monitoring, and debug capabilities make it suitable for both development and production use cases.