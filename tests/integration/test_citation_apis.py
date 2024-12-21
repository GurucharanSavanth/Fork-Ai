import pytest
import os
import time
from unittest.mock import patch, MagicMock
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from ai_scientist.utils.citation_api import CitationAPIManager, SemanticScholarAPI, ScopusAPI, TaylorFrancisAPI
from ai_scientist.utils.citation_db import CitationDB
from ai_scientist.perform_writeup import verify_citation

@pytest.fixture
def citation_manager():
    """Create CitationManager with real APIs when available."""
    # Check which APIs are available
    available_apis = []

    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        available_apis.append("semantic_scholar")
    if os.getenv("SCOPUS_API_KEY"):
        available_apis.append("scopus")
    if os.getenv("TAYLOR_FRANCIS_API_KEY"):
        available_apis.append("taylor_francis")

    if not available_apis:
        pytest.skip("No API keys available for testing")

    try:
        # Try to create manager with real APIs
        return CitationAPIManager()
    except ValueError as e:
        # If specific API initialization fails, log warning and skip
        pytest.skip(f"API initialization failed: {str(e)}")

@pytest.fixture
def citation_db():
    # Create temporary database for testing
    temp_dir = tempfile.mkdtemp()
    os.environ['CITATION_DB_PATH'] = os.path.join(temp_dir, 'test_citations.db')
    db = CitationDB()
    yield db
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

def test_real_doi_lookup(citation_manager):
    """Test DOI lookup with available APIs."""
    # Use a known DOI for testing
    doi = "10.48550/arXiv.1706.03762"  # "Attention Is All You Need"
    results = citation_manager.search_all_by_doi(doi)

    # Check results from each available API
    found_valid_result = False
    for api_name, result in results.items():
        if result is not None:
            found_valid_result = True
            # Verify data structure
            assert "title" in result, f"{api_name} missing title"
            assert "authors" in result, f"{api_name} missing authors"
            assert isinstance(result["authors"], list), f"{api_name} authors not a list"
            assert len(result["authors"]) > 0, f"{api_name} has no authors"

            # Verify specific fields for known paper
            if api_name == "semantic_scholar":
                assert "Attention Is All You Need" in result["title"]
                assert any("Vaswani" in author.get("name", "") for author in result["authors"])

    # Ensure at least one API returned valid results
    assert found_valid_result, "No API returned valid results"

def test_concurrent_api_requests(citation_manager):
    """Test concurrent API requests to different providers."""
    dois = [
        "10.48550/arXiv.1706.03762",  # Attention Is All You Need
        "10.48550/arXiv.1810.04805",  # BERT
        "10.48550/arXiv.2005.14165",  # GPT-3
    ]

    def lookup_doi(doi):
        return citation_manager.search_all_by_doi(doi)

    # Make concurrent requests
    with ThreadPoolExecutor(max_workers=len(dois)) as executor:
        futures = [executor.submit(lookup_doi, doi) for doi in dois]
        results = [future.result() for future in futures]

    # Verify all requests completed
    assert len(results) == len(dois)
    # Verify at least some results were found
    assert any(any(r is not None for r in result.values()) for result in results)

def test_citation_verification_integration(citation_manager, citation_db):
    """Test complete citation verification flow with real APIs."""
    cite_key = "attention2017"
    bib_text = """@article{attention2017,
        title={Attention Is All You Need},
        author={Vaswani, Ashish and others},
        journal={arXiv preprint arXiv:1706.03762},
        year={2017},
        doi={10.48550/arXiv.1706.03762}
    }"""

    # Verify citation
    assert verify_citation(cite_key, bib_text)

    # Check it was added to database
    citation = citation_db.get_citation(cite_key)
    assert citation is not None
    assert citation.verified
    assert citation.doi == "10.48550/arXiv.1706.03762"

def test_api_error_handling(citation_manager):
    """Test error handling with invalid DOI using real APIs."""
    # Test with completely invalid DOI
    invalid_doi = "10.1234/invalid-doi-12345"
    results = citation_manager.search_all_by_doi(invalid_doi)

    # All APIs should handle invalid DOI gracefully
    for api_name, result in results.items():
        assert result is None, f"{api_name} did not return None for invalid DOI"

    # Test with malformed DOI
    malformed_doi = "not.a.doi/format"
    results = citation_manager.search_all_by_doi(malformed_doi)

    # All APIs should handle malformed DOI gracefully
    for api_name, result in results.items():
        assert result is None, f"{api_name} did not return None for malformed DOI"

    # Test with empty DOI
    empty_results = citation_manager.search_all_by_doi("")
    for api_name, result in empty_results.items():
        assert result is None, f"{api_name} did not return None for empty DOI"

def test_rate_limit_handling(citation_manager):
    """Test rate limiting behavior with real APIs."""
    doi = "10.48550/arXiv.1706.03762"  # Attention Is All You Need

    # Track response times to detect rate limiting
    response_times = []
    start_time = None

    # Make multiple rapid requests to trigger rate limiting
    for i in range(10):
        start_time = time.time()
        results = citation_manager.search_all_by_doi(doi)
        response_time = time.time() - start_time
        response_times.append(response_time)

        # Verify results are still valid despite rate limiting
        found_valid_result = False
        for api_name, result in results.items():
            if result is not None:
                found_valid_result = True
                assert "title" in result
                assert "authors" in result

        # Ensure at least one API returned valid results
        assert found_valid_result

    # Verify rate limiting behavior
    # Later requests should take longer due to rate limiting
    early_avg = sum(response_times[:3]) / 3
    late_avg = sum(response_times[-3:]) / 3
    assert late_avg > early_avg

if __name__ == '__main__':
    pytest.main([__file__])
