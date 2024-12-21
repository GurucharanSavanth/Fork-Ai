import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import requests
from sqlalchemy.orm import Session

from ai_scientist.utils.citation_db import CitationDB, Citation
from ai_scientist.utils.citation_api import (
    CitationAPIManager, SemanticScholarAPI, ScopusAPI, TaylorFrancisAPI
)
from ai_scientist.utils.rate_limiter import APIRateLimiter

class TestCitationSystem(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'SCOPUS_API_KEY': 'mock_scopus_key',
            'SEMANTIC_SCHOLAR_API_KEY': 'mock_semantic_key',
            'TAYLOR_FRANCIS_API_KEY': 'mock_tf_key'
        })
        self.env_patcher.start()

        # Initialize rate limiter and mock response data
        self.rate_limiter = APIRateLimiter()
        self.mock_response = {
            "title": "Test Paper",
            "authors": [{"name": "Test Author"}],
            "year": 2024,
            "abstract": "Test abstract"
        }

        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'citations.db')
        self.citation_db = CitationDB(db_url=f'sqlite:///{self.db_path}')

        # Create mock API instances
        self.mock_semantic = MagicMock(spec=SemanticScholarAPI)
        self.mock_scopus = MagicMock(spec=ScopusAPI)
        self.mock_tf = MagicMock(spec=TaylorFrancisAPI)

        # Configure mock responses and rate limiters
        for mock_api in [self.mock_semantic, self.mock_scopus, self.mock_tf]:
            mock_api.rate_limiter = self.rate_limiter
            mock_api.search_by_doi = Mock(return_value=self.mock_response)

        # Patch the API classes
        self.semantic_patcher = patch('ai_scientist.utils.citation_api.SemanticScholarAPI', return_value=self.mock_semantic)
        self.scopus_patcher = patch('ai_scientist.utils.citation_api.ScopusAPI', return_value=self.mock_scopus)
        self.tf_patcher = patch('ai_scientist.utils.citation_api.TaylorFrancisAPI', return_value=self.mock_tf)

        # Start all patches
        self.semantic_patcher.start()
        self.scopus_patcher.start()
        self.tf_patcher.start()

        # Initialize citation manager (will use our mocked API classes)
        self.citation_manager = CitationAPIManager()

    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.env_patcher.stop()
        self.semantic_patcher.stop()
        self.scopus_patcher.stop()
        self.tf_patcher.stop()

        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

        # Close any open database sessions
        self.citation_db.Session.close_all()

    def test_add_citation(self):
        """Test adding a citation to the database."""
        citation = self.citation_db.add_citation(
            cite_key="test_paper",
            title="Test Paper",
            authors="Test Author",
            doi="10.1234/test"
        )
        self.assertIsNotNone(citation)
        self.assertEqual(citation.title, "Test Paper")
        self.assertEqual(citation.doi, "10.1234/test")
        self.assertFalse(citation.verified)

    def test_get_citation(self):
        """Test retrieving a citation from the database."""
        # Add a citation first
        self.citation_db.add_citation(
            cite_key="test_paper",
            title="Test Paper",
            authors="Test Author",
            doi="10.1234/test"
        )
        # Retrieve it
        citation = self.citation_db.get_citation("10.1234/test")
        self.assertIsNotNone(citation)
        self.assertEqual(citation.title, "Test Paper")

    def test_verify_citation(self):
        """Test citation verification using both DOI and cite_key."""
        # Add an unverified citation
        self.citation_db.add_citation(
            cite_key="test_paper",
            title="Test Paper",
            authors="Test Author",
            doi="10.1234/test"
        )
        # Verify using DOI
        result = self.citation_db.verify_citation("10.1234/test")
        self.assertTrue(result)
        # Check that it's marked as verified
        citation = self.citation_db.get_citation("10.1234/test")
        self.assertTrue(citation.verified)

        # Verify using cite_key
        citation2 = self.citation_db.add_citation(
            cite_key="another_paper",
            title="Another Paper",
            authors="Another Author",
            doi="10.1234/test2"
        )
        result2 = self.citation_db.verify_citation("another_paper")
        self.assertTrue(result2)
        citation2 = self.citation_db.get_citation("another_paper")
        self.assertTrue(citation2.verified)

    def test_verify_nonexistent_citation(self):
        """Test verification of non-existent citation."""
        result = self.citation_db.verify_citation("nonexistent")
        self.assertFalse(result)

    def test_paper_search(self):
        """Test paper search functionality."""
        # Test searching for a paper
        results = self.citation_manager.search_all_by_doi("10.1234/test")

        # Verify each API was called
        self.mock_semantic.search_by_doi.assert_called_once_with("10.1234/test")
        self.mock_scopus.search_by_doi.assert_called_once_with("10.1234/test")
        self.mock_tf.search_by_doi.assert_called_once_with("10.1234/test")

        # Verify results
        self.assertIn("semantic_scholar", results)
        self.assertEqual(results["semantic_scholar"], self.mock_response)
        self.assertEqual(results["scopus"], self.mock_response)
        self.assertEqual(results["taylor_francis"], self.mock_response)

    def test_metadata_handling(self):
        """Test citation metadata handling."""
        # Test adding citation with metadata
        citation = self.citation_db.add_citation(
            cite_key="test_paper",
            title="Complex Paper Title",
            authors="Author One; Author Two",
            doi="10.1234/test",
            full_text_hash="abc123",
            verified=False
        )

        # Verify all metadata fields
        self.assertEqual(citation.title, "Complex Paper Title")
        self.assertEqual(citation.authors, "Author One; Author Two")
        self.assertEqual(citation.doi, "10.1234/test")
        self.assertEqual(citation.full_text_hash, "abc123")
        self.assertFalse(citation.verified)

        # Test updating metadata
        session = self.citation_db.Session()
        try:
            citation = session.query(Citation).filter(Citation.doi == "10.1234/test").first()
            citation.verified = True
            session.add(citation)
            session.commit()
            session.refresh(citation)

            # Verify updates
            updated = session.query(Citation).filter(Citation.doi == "10.1234/test").first()
            self.assertTrue(updated.verified)
            self.assertEqual(updated.full_text_hash, "abc123")
        finally:
            session.close()

if __name__ == '__main__':
    unittest.main()
