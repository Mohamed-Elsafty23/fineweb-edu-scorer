import streamlit as st
import os

def get_config(key, default=None):
    try:
        if hasattr(st, 'secrets') and key in st.secrets.get("api", {}):
            return st.secrets["api"][key]
        if hasattr(st, 'secrets') and key in st.secrets.get("models", {}):
            return st.secrets["models"][key]
        if hasattr(st, 'secrets') and key in st.secrets.get("thresholds", {}):
            return st.secrets["thresholds"][key]
    except:
        pass
    
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    return default

API_KEY = get_config("OPENAI_API_KEY", "your-api-key-here")
BASE_URL = get_config("OPENAI_BASE_URL", "https://api.aimlapi.com/v1")

TEACHER_MODEL = get_config("LLM_MODEL", "llama-3.3-70b-instruct")
EMBEDDING_MODEL = get_config("EMBEDDING_MODEL", "multilingual-e5-large-instruct")

MAX_TOKENS = 2000
TEMPERATURE = 0.3

MODELS_DIR = "models"
DATA_DIR = "data"
TRAINING_DATA_PATH = "data/training_data.csv"
EMBEDDINGS_PATH = "data/embeddings_and_scores.pkl"
STUDENT_MODEL_PATH = "models/student_classifier.pkl"

EDUCATIONAL_THRESHOLD = get_config("EDUCATIONAL_THRESHOLD", 3)
