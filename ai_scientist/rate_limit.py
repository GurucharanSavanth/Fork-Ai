"""Rate limit handling for AI-Scientist API calls."""
import time
import logging
from typing import Optional, Callable, Any
from functools import wraps
import backoff
from queue import Queue, Empty
from threading import Lock

import openai
import anthropic
import google.api_core.exceptions
import requests

class RateLimitHandler:
    """Handles rate limiting across different API providers."""

    def __init__(self):
        self._request_queues = {}  # Per-provider request queues
        self._locks = {}  # Per-provider locks
        self._last_request_time = {}  # Per-provider last request timestamps
        self._min_request_interval = {
            'openai': 1.0,  # 1 request per second
            'anthropic': 0.5,  # 2 requests per second
            'google': 1.0,  # 1 request per second
            'xai': 1.0,  # 1 request per second
            'semantic_scholar': 1.0,  # 1 request per second
            'default': 1.0  # Default fallback
        }
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('rate_limit_handler')

    def _get_provider_key(self, model: str) -> str:
        """Map model name to provider key."""
        if 'gpt' in model or model.startswith('o1-'):
            return 'openai'
        elif 'claude' in model:
            return 'anthropic'
        elif 'gemini' in model:
            return 'google'
        elif 'grok' in model:
            return 'xai'
        return 'default'

    def _ensure_provider_initialized(self, provider: str):
        """Initialize provider-specific resources if not already done."""
        if provider not in self._request_queues:
            self._request_queues[provider] = Queue()
        if provider not in self._locks:
            self._locks[provider] = Lock()
        if provider not in self._last_request_time:
            self._last_request_time[provider] = 0

    def handle_rate_limit(self, model: str) -> Callable:
        """Decorator for handling rate limits for specific models."""
        provider = self._get_provider_key(model)
        self._ensure_provider_initialized(provider)

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            @backoff.on_exception(
                backoff.expo,
                (
                    openai.RateLimitError,
                    anthropic.RateLimitError,
                    google.api_core.exceptions.ResourceExhausted,
                    requests.exceptions.HTTPError
                ),
                max_tries=5,
                on_backoff=lambda details: self.logger.warning(
                    f"Rate limit hit for {model} ({provider}). "
                    f"Backing off {details['wait']:.1f}s after {details['tries']} tries "
                    f"calling {details['target'].__name__} at {time.strftime('%X')}"
                )
            )
            async def wrapper(*args, **kwargs):
                with self._locks[provider]:
                    current_time = time.time()
                    min_interval = self._min_request_interval.get(
                        provider, self._min_request_interval['default']
                    )

                    # Enforce minimum interval between requests
                    time_since_last = current_time - self._last_request_time[provider]
                    if time_since_last < min_interval:
                        wait_time = min_interval - time_since_last
                        self.logger.debug(
                            f"Enforcing minimum interval for {provider}, "
                            f"waiting {wait_time:.1f}s"
                        )
                        time.sleep(wait_time)

                    try:
                        result = await func(*args, **kwargs)
                        self._last_request_time[provider] = time.time()
                        return result
                    except Exception as e:
                        if any(
                            err_type.__name__ in str(type(e))
                            for err_type in (
                                openai.RateLimitError,
                                anthropic.RateLimitError,
                                google.api_core.exceptions.ResourceExhausted
                            )
                        ):
                            self.logger.warning(
                                f"Rate limit error for {provider}: {str(e)}"
                            )
                        raise
            return wrapper
        return decorator

rate_limit_handler = RateLimitHandler()
