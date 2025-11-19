"""Configuration file for API credentials and model settings."""
import streamlit as st
import os

# Try to get from Streamlit secrets first, then fall back to environment variables
def get_config(key, default=None):
    """Get configuration from Streamlit secrets or environment variables."""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets') and key in st.secrets.get("api", {}):
            return st.secrets["api"][key]
        if hasattr(st, 'secrets') and key in st.secrets.get("models", {}):
            return st.secrets["models"][key]
        if hasattr(st, 'secrets') and key in st.secrets.get("thresholds", {}):
            return st.secrets["thresholds"][key]
    except:
        pass
    
    # Fall back to environment variable
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    # Return default
    return default

# API Configuration
API_KEY = get_config("OPENAI_API_KEY", "your-api-key-here")
BASE_URL = get_config("OPENAI_BASE_URL", "https://api.aimlapi.com/v1")

# Model names
TEACHER_MODEL = get_config("LLM_MODEL", "llama-3.3-70b-instruct")
EMBEDDING_MODEL = get_config("EMBEDDING_MODEL", "multilingual-e5-large-instruct")

# Model parameters
MAX_TOKENS = 2000  # Maximum tokens to process per text
TEMPERATURE = 0.3  # Lower temperature for consistent scoring

# Paths
MODELS_DIR = "models"
DATA_DIR = "data"
TRAINING_DATA_PATH = "data/training_data.csv"
EMBEDDINGS_PATH = "data/embeddings_and_scores.pkl"
STUDENT_MODEL_PATH = "models/student_classifier.pkl"

# Educational scoring threshold
EDUCATIONAL_THRESHOLD = get_config("EDUCATIONAL_THRESHOLD", 3)  # Score >= 3 is considered educational

