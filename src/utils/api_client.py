from openai import OpenAI
from typing import List, Dict, Optional
import numpy as np
import config


class APIClient:
    
    def __init__(self, api_key: str = None, base_url: str = None):
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
        model = model or config.EMBEDDING_MODEL
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=model
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}")
    
    def get_single_embedding(self, text: str, model: str = None) -> np.ndarray:
        embeddings = self.get_embeddings([text], model=model)
        return embeddings[0]


api_client = APIClient()


def test_api_client():
    print("Testing API Client...")
    
    print("\n1. Testing LLM response:")
    response = api_client.get_llm_response("What is 2+2?")
    print(f"Response: {response[:100]}...")
    
    print("\n2. Testing embeddings:")
    texts = ["Hello world", "Machine learning is fascinating"]
    embeddings = api_client.get_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
    
    print("\nAPI Client tests passed!")


if __name__ == "__main__":
    test_api_client()
