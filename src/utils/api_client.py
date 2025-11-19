"""
Unified API client for interacting with LLM and embedding models.
Handles communication with llama-3.3-70b-instruct (Teacher) and multilingual-e5-large-instruct (Embeddings).
"""

from openai import OpenAI
from typing import List, Dict, Optional
import numpy as np
import config


class APIClient:
    """Unified client for LLM and embedding API calls."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
        """
        self.api_key = api_key or config.API_KEY
        self.base_url = base_url or config.BASE_URL
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
    def get_llm_response(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        model: str = None,
        temperature: float = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Get response from LLM model.
        
        Args:
            prompt: User prompt to send to the model
            system_message: System message to set context
            model: Model name (defaults to TEACHER_MODEL from config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            String response from the model
        """
        model = model or config.TEACHER_MODEL
        temperature = temperature or config.TEMPERATURE
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error getting LLM response: {str(e)}")
    
    def get_embeddings(
        self, 
        texts: List[str], 
        model: str = None
    ) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: Embedding model name (defaults to EMBEDDING_MODEL from config)
            
        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        model = model or config.EMBEDDING_MODEL
        
        # Handle single text by converting to list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=model
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}")
    
    def get_single_embedding(self, text: str, model: str = None) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Text string to embed
            model: Embedding model name
            
        Returns:
            1D numpy array of embedding
        """
        embeddings = self.get_embeddings([text], model=model)
        return embeddings[0]


# Create a global instance for easy importing
api_client = APIClient()


def test_api_client():
    """Test function to verify API client works."""
    print("Testing API Client...")
    
    # Test LLM
    print("\n1. Testing LLM response:")
    response = api_client.get_llm_response("What is 2+2?")
    print(f"Response: {response[:100]}...")
    
    # Test embeddings
    print("\n2. Testing embeddings:")
    texts = ["Hello world", "Machine learning is fascinating"]
    embeddings = api_client.get_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
    
    print("\n✓ API Client tests passed!")


if __name__ == "__main__":
    test_api_client()

