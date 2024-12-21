import time
import random
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from threading import Lock

@dataclass
class RateLimitConfig:
    requests_per_minute: int
    burst_limit: int
    min_delay: float = 0.1
    max_delay: float = 1.0

class APIRateLimiter:
    def __init__(self):
        self.rate_limits: Dict[str, RateLimitConfig] = {
            "anthropic": RateLimitConfig(45, 3),
            "openai": RateLimitConfig(60, 5),
            "semantic_scholar": RateLimitConfig(30, 2),
            "scopus": RateLimitConfig(20, 2),
            "taylor_francis": RateLimitConfig(15, 1)
        }
        self.last_request_time: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = {}
        self._locks: Dict[str, Lock] = {}
        self.logger = logging.getLogger(__name__)

    def _get_lock(self, provider: str) -> Lock:
        """Get or create a lock for a provider."""
        if provider not in self._locks:
            self._locks[provider] = Lock()
        return self._locks[provider]

    def _generate_delay(self, config: RateLimitConfig) -> float:
        """Generate delay using flipped lognormal distribution."""
        # Calculate mu and sigma for the lognormal distribution
        # We use the fact that for X ~ LogNormal(mu, sigma),
        # P(X <= x) = Phi((ln(x) - mu)/sigma)
        # We want P(X <= max_delay) = 0.99
        sigma = 0.5  # Chosen to give a reasonable spread
        mu = math.log(config.max_delay) - sigma * 2.326  # 2.326 is z-score for 0.99

        # Generate a random value from the lognormal distribution
        raw_delay = random.lognormvariate(mu, sigma)

        # Clip the delay to our bounds and flip it
        delay = max(config.min_delay, min(config.max_delay, raw_delay))
        flipped_delay = config.max_delay - delay + config.min_delay

        return flipped_delay

    def _update_request_count(self, provider: str) -> None:
        """Update the request count for rate limiting."""
        current_time = time.time()
        if provider not in self.request_counts:
            self.request_counts[provider] = 0
            self.last_request_time[provider] = current_time

        # Reset counter if a minute has passed
        if current_time - self.last_request_time[provider] >= 60:
            self.request_counts[provider] = 0
            self.last_request_time[provider] = current_time

        self.request_counts[provider] += 1

    def _check_rate_limit(self, provider: str) -> Tuple[bool, float]:
        """Check if we're rate limited and calculate delay if needed."""
        config = self.rate_limits.get(provider)
        if not config:
            self.logger.warning(f"No rate limit config for provider: {provider}")
            return False, 0.0

        current_time = time.time()
        last_time = self.last_request_time.get(provider, 0)
        count = self.request_counts.get(provider, 0)

        # Check if we're within the burst limit
        if count >= config.burst_limit:
            delay_needed = 60.0 / config.requests_per_minute
            time_passed = current_time - last_time
            if time_passed < delay_needed:
                return True, delay_needed - time_passed

        # Generate random delay for general rate limiting
        return False, self._generate_delay(config)

    def handle_request(self, provider: str) -> None:
        """Handle a request with proper rate limiting."""
        with self._get_lock(provider):
            is_limited, delay = self._check_rate_limit(provider)
            if is_limited:
                self.logger.info(f"Rate limited for {provider}, waiting {delay:.2f}s")
                time.sleep(delay)
            elif delay > 0:
                self.logger.debug(f"Adding delay of {delay:.2f}s for {provider}")
                time.sleep(delay)
            self._update_request_count(provider)

    def handle_backoff(self, details: Dict) -> None:
        """Handle backoff events from the @backoff decorator."""
        provider = details.get('provider', 'unknown')
        self.logger.warning(
            f"Backing off {details['wait']:0.1f}s after {details['tries']} tries "
            f"calling function {details['target'].__name__} for provider {provider}"
        )
        # Reset request count and update last request time for this provider
        with self._get_lock(provider):
            self.request_counts[provider] = 0
            self.last_request_time[provider] = time.time()
