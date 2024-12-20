import pytest
import os
from unittest.mock import patch, MagicMock
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from ai_scientist.utils.citation_api import CitationAPIManager, SemanticScholarAPI, ScopusAPI, TaylorFrancisAPI
from ai_scientist.utils.citation_db import CitationDB
from ai_scientist.perform_writeup import verify_citation

@pytest.fixture
def citation_manager():
    """Create a CitationManager that works with mocked APIs for integration tests."""
    # Always use mocked APIs for integration tests
    with patch('ai_scientist.utils.citation_api.SemanticScholarAPI') as mock_semantic, \
         patch('ai_scientist.utils.citation_api.ScopusAPI') as mock_scopus, \
         patch('ai_scientist.utils.citation_api.TaylorFrancisAPI') as mock_tf:

        mock_response = {
            "title": "Attention Is All You Need",
            "authors": [{"name": "Vaswani, Ashish"}, {"name": "Others"}],
            "year": 2017,
            "abstract": "Test abstract"
        }

        for mock_api in [mock_semantic.return_value, mock_scopus.return_value, mock_tf.return_value]:
            mock_api.search_by_doi = MagicMock(return_value=mock_response)

        return CitationAPIManager()

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
    """Test looking up a DOI using mocked APIs."""
    doi = "10.48550/arXiv.1706.03762"  # "Attention Is All You Need"
    results = citation_manager.search_all_by_doi(doi)

    # All mocked APIs should return results
    assert all(result is not None for result in results.values())

    # Verify returned data structure
    for api_name, result in results.items():
        assert "title" in result
        assert "authors" in result
        assert result["title"] == "Attention Is All You Need"
        assert len(result["authors"]) == 2
        assert isinstance(result["authors"], list)
        assert all(isinstance(author, dict) for author in result["authors"])
        assert all("name" in author for author in result["authors"])

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
    """Test handling of API errors and fallbacks."""
    # Configure APIs to return None for invalid DOI
    for api in citation_manager.apis.values():
        api.search_by_doi = MagicMock(return_value=None)

    # Test with invalid DOI
    results = citation_manager.search_all_by_doi("invalid-doi")

    # Should return None results but not raise exceptions
    assert all(result is None for result in results.values())

def test_rate_limit_handling(citation_manager):
    """Test handling of rate limits across APIs."""
    doi = "10.48550/arXiv.1706.03762"

    # Make multiple rapid requests
    results = []
    for _ in range(5):
        result = citation_manager.search_all_by_doi(doi)
        results.append(result)

    # Verify all requests completed without errors
    assert len(results) == 5
    # At least some results should be successful
    assert any(any(r is not None for r in result.values()) for result in results)

if __name__ == '__main__':
    pytest.main([__file__])
