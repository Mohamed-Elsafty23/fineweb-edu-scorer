"""
Web text extraction using trafilatura.
This module replicates the methodology from the FineWeb paper (Section 3.2).
"""

import trafilatura
from typing import Optional, Dict
import requests


class WebExtractor:
    """Extract clean text from web URLs using trafilatura."""
    
    def __init__(self, max_text_length: int = 50000):
        """
        Initialize the web extractor.
        
        Args:
            max_text_length: Maximum length of extracted text in characters
        """
        self.max_text_length = max_text_length
    
    def extract_from_url(self, url: str) -> Dict[str, Optional[str]]:
        """
        Extract text from a URL.
        
        Args:
            url: URL to extract text from
            
        Returns:
            Dictionary containing:
                - 'url': Original URL
                - 'raw_html': Raw HTML (first 1000 chars for preview)
                - 'text': Cleaned extracted text
                - 'error': Error message if extraction failed
        """
        result = {
            'url': url,
            'raw_html': None,
            'text': None,
            'error': None
        }
        
        try:
            # Fetch the URL
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                result['error'] = "Failed to download URL"
                return result
            
            # Store raw HTML preview
            result['raw_html'] = downloaded[:1000] if len(downloaded) > 1000 else downloaded
            
            # Extract clean text using trafilatura
            # This removes boilerplate (menus, ads, navigation) as described in the paper
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            if not text:
                result['error'] = "No text could be extracted from the page"
                return result
            
            # Limit text length
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            result['text'] = text
            
        except requests.exceptions.RequestException as e:
            result['error'] = f"Network error: {str(e)}"
        except Exception as e:
            result['error'] = f"Extraction error: {str(e)}"
        
        return result
    
    def extract_from_html(self, html: str) -> Optional[str]:
        """
        Extract text from raw HTML string.
        
        Args:
            html: Raw HTML string
            
        Returns:
            Cleaned extracted text or None if extraction failed
        """
        try:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            if text and len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            return text
        except Exception as e:
            print(f"Error extracting from HTML: {str(e)}")
            return None


def test_web_extractor():
    """Test the web extractor with sample URLs."""
    print("Testing Web Extractor...")
    
    extractor = WebExtractor()
    
    # Test URLs with different expected quality
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",  # High educational value
        "https://www.python.org/about/",  # Medium educational value
    ]
    
    for url in test_urls:
        print(f"\n\nTesting URL: {url}")
        result = extractor.extract_from_url(url)
        
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            print(f"Successfully extracted {len(result['text'])} characters")
            print(f"Preview: {result['text'][:200]}...")


if __name__ == "__main__":
    test_web_extractor()

