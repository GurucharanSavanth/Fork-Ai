import unittest
from unittest.mock import Mock, patch
import time
import threading
from dataclasses import dataclass

from ai_scientist.utils.rate_limiter import APIRateLimiter, RateLimitConfig

class TestRateLimiter(unittest.TestCase):
    def setUp(self):
        self.rate_limiter = APIRateLimiter()
        # Set a minimum delay for consistent testing
        self.min_delay = 0.1  # 100ms
        self.rate_limiter.default_config = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=2,
            min_delay=self.min_delay
        )

    def test_rate_limit_config(self):
        """Test rate limit configuration."""
        config = RateLimitConfig(60, 5)
        self.assertEqual(config.requests_per_minute, 60)
        self.assertEqual(config.burst_limit, 5)
        self.assertEqual(config.min_delay, 0.1)
        self.assertEqual(config.max_delay, 1.0)

    def test_provider_specific_limits(self):
        """Test provider-specific rate limits."""
        self.assertEqual(self.rate_limiter.rate_limits["anthropic"].requests_per_minute, 45)
        self.assertEqual(self.rate_limiter.rate_limits["openai"].requests_per_minute, 60)
        self.assertEqual(self.rate_limiter.rate_limits["semantic_scholar"].requests_per_minute, 30)

    def test_burst_limit_handling(self):
        """Test burst limit handling."""
        provider = "test_provider"
        # Configure rate limiter with specific settings for test
        self.rate_limiter.rate_limits[provider] = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=2,
            min_delay=self.min_delay
        )

        # First two requests should be quick
        self.rate_limiter.handle_request(provider)
        self.rate_limiter.handle_request(provider)

        # Third request should be delayed
        start_time = time.time()
        self.rate_limiter.handle_request(provider)
        delayed_time = time.time() - start_time

        # Ensure delayed time is at least the minimum delay
        self.assertGreaterEqual(delayed_time, self.min_delay)

    def test_concurrent_requests(self):
        """Test concurrent request handling."""
        provider = "test_provider"
        self.rate_limiter.rate_limits[provider] = RateLimitConfig(60, 1)

        def make_request():
            self.rate_limiter.handle_request(provider)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that request count is correct
        self.assertEqual(self.rate_limiter.request_counts.get(provider, 0), 3)

    def test_backoff_handling(self):
        """Test backoff event handling."""
        provider = "test_provider"
        self.rate_limiter.rate_limits[provider] = RateLimitConfig(60, 5)

        # Simulate backoff event
        details = {
            'provider': provider,
            'wait': 1.0,
            'tries': 3,
            'target': Mock(__name__='test_function')
        }
        self.rate_limiter.handle_backoff(details)

        # Check that request count was reset
        self.assertEqual(self.rate_limiter.request_counts.get(provider, 0), 0)

if __name__ == '__main__':
    unittest.main()
