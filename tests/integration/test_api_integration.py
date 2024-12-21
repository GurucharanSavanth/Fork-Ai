import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai_scientist.utils.rate_limiter import APIRateLimiter

@pytest.fixture
def rate_limiter():
    return APIRateLimiter()

def test_concurrent_api_requests(rate_limiter):
    """Test concurrent API requests with rate limiting."""
    provider = "semantic_scholar"
    request_count = 10
    results = []

    def make_request():
        start_time = time.time()
        rate_limiter.handle_request(provider)
        return time.time() - start_time

    # Make concurrent requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(request_count)]
        results = [future.result() for future in as_completed(futures)]

    # Verify rate limiting worked
    assert len(results) == request_count
    # Check that not all requests were immediate (some were delayed)
    assert any(delay > rate_limiter.rate_limits[provider].min_delay for delay in results)

def test_provider_specific_rate_limits(rate_limiter):
    """Test that different providers have different rate limits."""
    providers = ["anthropic", "openai", "semantic_scholar"]
    delays = {}

    for provider in providers:
        start_time = time.time()
        # Make multiple requests to trigger rate limiting
        for _ in range(rate_limiter.rate_limits[provider].burst_limit + 1):
            rate_limiter.handle_request(provider)
        delays[provider] = time.time() - start_time

    # Verify different providers have different delays
    assert len(set(delays.values())) > 1, "Different providers should have different delays"

def test_backoff_integration(rate_limiter):
    """Test backoff mechanism with rate limiter."""
    provider = "openai"

    # Simulate multiple failed requests
    details = {
        'provider': provider,
        'wait': 1.0,
        'tries': 3,
        'target': lambda: None
    }

    # Record request count before backoff
    initial_count = rate_limiter.request_counts.get(provider, 0)

    # Handle backoff
    rate_limiter.handle_backoff(details)

    # Verify request count was reset
    assert rate_limiter.request_counts.get(provider, 0) == 0

    # Make new request after backoff
    start_time = time.time()
    rate_limiter.handle_request(provider)
    delay = time.time() - start_time

    # Verify delay was applied
    assert delay >= rate_limiter.rate_limits[provider].min_delay

def test_burst_limit_integration(rate_limiter):
    """Test burst limit behavior with real timing."""
    provider = "semantic_scholar"
    config = rate_limiter.rate_limits[provider]

    # Make requests up to burst limit
    start_time = time.time()
    for _ in range(config.burst_limit):
        rate_limiter.handle_request(provider)
    burst_time = time.time() - start_time

    # Make one more request that should be delayed
    start_time = time.time()
    rate_limiter.handle_request(provider)
    delayed_time = time.time() - start_time

    # Verify burst requests were faster than delayed request
    assert delayed_time > burst_time / config.burst_limit

if __name__ == '__main__':
    pytest.main([__file__])
