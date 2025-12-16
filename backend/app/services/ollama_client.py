"""
Ollama Client for Local LLM Communication

This module provides a client for interacting with Ollama, a local LLM runtime.
Ollama runs models locally and exposes them via HTTP API on port 11434.

Ollama Setup:
1. Install Ollama: https://ollama.ai/download
2. Pull a model: `ollama pull llama2` or `ollama pull mistral`
3. Verify it's running: `ollama list`

API Reference:
- POST /api/generate: Generate text from a prompt
- Request: {"model": "llama2", "prompt": "...", "stream": false, "options": {...}}
- Response: {"model": "...", "response": "...", "done": true}

Example Usage:
    >>> client = OllamaClient(model_name="llama2", base_url="http://localhost:11434")
    >>> response = client.generate("What is property law?")
    >>> print(response)
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """
    Configuration for Ollama client.
    
    Attributes:
        model_name: Name of the Ollama model
                   - "llama2" (base model, requires full GAP framework in prompt)
                   - "gap-english" (custom model with embedded framework)
                   - "gap-tamil" (custom model with embedded Tamil framework)
        base_url: Base URL for Ollama API (default: http://localhost:11434)
        temperature: Sampling temperature (0.0-2.0, higher = more random)
        top_p: Nucleus sampling threshold (0.0-1.0)
        max_tokens: Maximum tokens to generate (default: 2048)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts on failure
        use_custom_model: If True, model has GAP framework in system prompt
    
    Note:
        Custom models (gap-english, gap-tamil) must be created first:
        - cd backend/modelfiles
        - ollama create gap-english -f Modelfile.gap-english
        - ollama create gap-tamil -f Modelfile.gap-tamil
    """
    model_name: str = "llama2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout: int = 120  # 2 minutes
    max_retries: int = 2
    use_custom_model: bool = False  # Set True when using gap-english or gap-tamil


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when cannot connect to Ollama server."""
    pass


class OllamaModelError(OllamaClientError):
    """Raised when model is not available or fails to respond."""
    pass


class OllamaTimeoutError(OllamaClientError):
    """Raised when request times out."""
    pass


class OllamaClient:
    """
    Client for communicating with Ollama local LLM API.
    
    This client handles:
    - HTTP communication with Ollama server
    - Automatic retries with exponential backoff
    - Timeout handling
    - Error handling and classification
    - Configuration of model parameters
    
    Ollama must be running locally before using this client.
    Verify with: `curl http://localhost:11434/api/tags`
    
    Example:
        >>> config = OllamaConfig(
        ...     model_name="llama2",
        ...     temperature=0.7,
        ...     max_tokens=1024
        ... )
        >>> client = OllamaClient(config)
        >>> response = client.generate("Explain property rights.")
        >>> print(response)
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or OllamaConfig()
        self.generate_url = f"{self.config.base_url}/api/generate"
        
        logger.info(
            f"Initialized OllamaClient with model '{self.config.model_name}' "
            f"at {self.config.base_url}"
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt using Ollama.
        
        This method:
        1. Sends prompt to Ollama API with configured parameters
        2. Retries on transient failures (max 2 retries)
        3. Returns generated text response
        
        Args:
            prompt: Input text prompt for the LLM
        
        Returns:
            str: Generated text response from the model
        
        Raises:
            OllamaConnectionError: Cannot connect to Ollama server
            OllamaModelError: Model not available or generation failed
            OllamaTimeoutError: Request timed out
            OllamaClientError: Other client errors
        
        Example:
            >>> response = client.generate("What are legal clauses in a deed?")
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        last_exception = None
        
        # Retry loop with exponential backoff
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: 1s, 2s, 4s...
                    backoff_time = 2 ** (attempt - 1)
                    logger.info(
                        f"Retrying Ollama request (attempt {attempt + 1}/"
                        f"{self.config.max_retries + 1}) after {backoff_time}s"
                    )
                    time.sleep(backoff_time)
                
                response_text = self._make_request(prompt)
                
                logger.info(
                    f"Successfully generated response "
                    f"({len(response_text)} chars) on attempt {attempt + 1}"
                )
                
                return response_text
            
            except (ConnectionError, Timeout) as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}"
                )
                
                # Don't retry on last attempt
                if attempt == self.config.max_retries:
                    break
            
            except OllamaModelError as e:
                # Model errors should not be retried
                logger.error(f"Model error (no retry): {str(e)}")
                raise
        
        # All retries exhausted
        if isinstance(last_exception, ConnectionError):
            raise OllamaConnectionError(
                f"Failed to connect to Ollama after {self.config.max_retries + 1} attempts. "
                f"Is Ollama running at {self.config.base_url}?"
            ) from last_exception
        elif isinstance(last_exception, Timeout):
            raise OllamaTimeoutError(
                f"Request timed out after {self.config.max_retries + 1} attempts "
                f"(timeout: {self.config.timeout}s)"
            ) from last_exception
        else:
            raise OllamaClientError(
                f"Request failed after {self.config.max_retries + 1} attempts"
            ) from last_exception
    
    def _make_request(self, prompt: str) -> str:
        """
        Make a single HTTP request to Ollama API.
        
        Args:
            prompt: Input text prompt
        
        Returns:
            str: Generated text response
        
        Raises:
            ConnectionError: Cannot connect to server
            Timeout: Request timed out
            OllamaModelError: Model error or invalid response
        """
        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,  # Get full response at once
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,  # Ollama's param name
            }
        }
        
        logger.debug(
            f"Sending request to Ollama: model={self.config.model_name}, "
            f"prompt_length={len(prompt)}"
        )
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.config.timeout
            )
            
            # Check HTTP status
            if response.status_code == 404:
                raise OllamaModelError(
                    f"Model '{self.config.model_name}' not found. "
                    f"Pull it with: ollama pull {self.config.model_name}"
                )
            
            if response.status_code != 200:
                raise OllamaModelError(
                    f"Ollama returned status {response.status_code}: "
                    f"{response.text}"
                )
            
            # Parse JSON response
            try:
                response_data = response.json()
            except ValueError as e:
                raise OllamaModelError(
                    f"Invalid JSON response from Ollama: {response.text}"
                ) from e
            
            # Extract generated text
            if "response" not in response_data:
                raise OllamaModelError(
                    f"Missing 'response' field in Ollama output: {response_data}"
                )
            
            generated_text = response_data["response"]
            
            if not generated_text:
                logger.warning("Ollama returned empty response")
                return ""
            
            return generated_text.strip()
        
        except Timeout:
            logger.error(f"Request timed out after {self.config.timeout}s")
            raise
        
        except ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            raise
        
        except RequestException as e:
            # Other requests library errors
            raise OllamaClientError(f"HTTP request failed: {str(e)}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama server is accessible and model is available.
        
        Returns:
            dict: Health check results with 'status', 'model', 'available'
        
        Example:
            >>> health = client.health_check()
            >>> if health['available']:
            ...     print("Ollama is ready")
        """
        try:
            # Try to list available models
            tags_url = f"{self.config.base_url}/api/tags"
            response = requests.get(tags_url, timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m["name"] for m in models_data.get("models", [])]
                
                model_available = self.config.model_name in available_models
                
                return {
                    "status": "healthy",
                    "model": self.config.model_name,
                    "available": model_available,
                    "available_models": available_models,
                    "base_url": self.config.base_url
                }
            else:
                return {
                    "status": "unhealthy",
                    "model": self.config.model_name,
                    "available": False,
                    "error": f"HTTP {response.status_code}",
                    "base_url": self.config.base_url
                }
        
        except Exception as e:
            return {
                "status": "unreachable",
                "model": self.config.model_name,
                "available": False,
                "error": str(e),
                "base_url": self.config.base_url
            }
    
    def __repr__(self) -> str:
        return (
            f"OllamaClient(model='{self.config.model_name}', "
            f"base_url='{self.config.base_url}')"
        )
