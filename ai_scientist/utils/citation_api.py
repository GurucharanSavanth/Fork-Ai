from typing import Optional, Dict, List
import os
import logging
from abc import ABC, abstractmethod

import requests

from ai_scientist.utils.rate_limiter import APIRateLimiter

logger = logging.getLogger(__name__)

class BaseCitationAPI(ABC):
    """Base class for citation API implementations."""
    def __init__(self, rate_limiter: APIRateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()

    @abstractmethod
    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """Search for a paper by DOI."""
        pass

    @abstractmethod
    def get_full_text(self, doi: str) -> Optional[str]:
        """Retrieve full text of a paper if available."""
        pass

class SemanticScholarAPI(BaseCitationAPI):
    """Semantic Scholar API implementation."""
    def __init__(self, rate_limiter: APIRateLimiter):
        super().__init__(rate_limiter)
        self.base_url = "https://api.semanticscholar.org/v1"
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """Search for a paper by DOI on Semantic Scholar."""
        self.rate_limiter.handle_request("semantic_scholar")
        try:
            response = self.session.get(f"{self.base_url}/paper/{doi}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error searching Semantic Scholar for DOI {doi}: {e}")
            return None

    def get_full_text(self, doi: str) -> Optional[str]:
        """Get full text if available through Semantic Scholar."""
        self.rate_limiter.handle_request("semantic_scholar")
        try:
            response = self.session.get(f"{self.base_url}/paper/{doi}/fulltext")
            response.raise_for_status()
            return response.json().get("fullText")
        except requests.RequestException as e:
            logger.error(f"Error getting full text from Semantic Scholar for DOI {doi}: {e}")
            return None

class ScopusAPI(BaseCitationAPI):
    """Scopus API implementation."""
    def __init__(self, rate_limiter: APIRateLimiter):
        super().__init__(rate_limiter)
        self.base_url = "https://api.elsevier.com/content"
        api_key = os.getenv("SCOPUS_API_KEY")
        if not api_key:
            raise ValueError("SCOPUS_API_KEY environment variable is required")
        self.session.headers.update({
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        })

    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """Search for a paper by DOI on Scopus."""
        self.rate_limiter.handle_request("scopus")
        try:
            response = self.session.get(
                f"{self.base_url}/abstract/doi/{doi}",
                params={"view": "FULL"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error searching Scopus for DOI {doi}: {e}")
            return None

    def get_full_text(self, doi: str) -> Optional[str]:
        """Get full text if available through Scopus."""
        self.rate_limiter.handle_request("scopus")
        try:
            response = self.session.get(
                f"{self.base_url}/article/doi/{doi}",
                params={"view": "FULL"}
            )
            response.raise_for_status()
            return response.json().get("full-text-retrieval-response", {}).get("originalText")
        except requests.RequestException as e:
            logger.error(f"Error getting full text from Scopus for DOI {doi}: {e}")
            return None

class TaylorFrancisAPI(BaseCitationAPI):
    """Taylor & Francis API implementation."""
    def __init__(self, rate_limiter: APIRateLimiter):
        super().__init__(rate_limiter)
        self.base_url = "https://api.taylorfrancis.com/v2"
        api_key = os.getenv("TAYLOR_FRANCIS_API_KEY")
        if not api_key:
            raise ValueError("TAYLOR_FRANCIS_API_KEY environment variable is required")
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """Search for a paper by DOI on Taylor & Francis."""
        self.rate_limiter.handle_request("taylor_francis")
        try:
            response = self.session.get(
                f"{self.base_url}/articles",
                params={"doi": doi}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error searching Taylor & Francis for DOI {doi}: {e}")
            return None

    def get_full_text(self, doi: str) -> Optional[str]:
        """Get full text if available through Taylor & Francis."""
        self.rate_limiter.handle_request("taylor_francis")
        try:
            response = self.session.get(
                f"{self.base_url}/articles/{doi}/full-text"
            )
            response.raise_for_status()
            return response.json().get("fullText")
        except requests.RequestException as e:
            logger.error(f"Error getting full text from Taylor & Francis for DOI {doi}: {e}")
            return None

class CitationAPIManager:
    """Manager class for handling multiple citation APIs."""
    def __init__(self):
        self.rate_limiter = APIRateLimiter()
        self.apis = {
            "semantic_scholar": SemanticScholarAPI(self.rate_limiter),
            "scopus": ScopusAPI(self.rate_limiter),
            "taylor_francis": TaylorFrancisAPI(self.rate_limiter)
        }

    def search_all_by_doi(self, doi: str) -> Dict[str, Optional[Dict]]:
        """Search for a paper across all available APIs."""
        results = {}
        for api_name, api in self.apis.items():
            results[api_name] = api.search_by_doi(doi)
        return results

    def get_full_text(self, doi: str) -> Optional[str]:
        """Try to get full text from any available API."""
        for api in self.apis.values():
            if text := api.get_full_text(doi):
                return text
        return None
