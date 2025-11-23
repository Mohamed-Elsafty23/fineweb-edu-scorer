import trafilatura
from typing import Optional, Dict
import requests


class WebExtractor:
    
    def __init__(self, max_text_length: int = 50000):
        self.max_text_length = max_text_length
    
    def extract_from_url(self, url: str) -> Dict[str, Optional[str]]:
        result = {
            'url': url,
            'raw_html': None,
            'text': None,
            'error': None
        }
        
        try:
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                result['error'] = "Failed to download URL"
                return result
            
            result['raw_html'] = downloaded[:1000] if len(downloaded) > 1000 else downloaded
            
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            if not text:
                result['error'] = "No text could be extracted from the page"
                return result
            
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            result['text'] = text
            
        except requests.exceptions.RequestException as e:
            result['error'] = f"Network error: {str(e)}"
        except Exception as e:
            result['error'] = f"Extraction error: {str(e)}"
        
        return result
    
    def extract_from_html(self, html: str) -> Optional[str]:
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
    print("Testing Web Extractor...")
    
    extractor = WebExtractor()
    
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.python.org/about/",
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
