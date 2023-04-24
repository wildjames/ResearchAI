import os
from dataclasses import dataclass

@dataclass
class Config:
    """Doesn't do anything, just holds configuration variables."""
    DEBUG = os.environ.get("DEBUG", False)
    OPENAI_API_TOKEN = os.environ.get("OPENAI_API_TOKEN", None)
    GOOGLE_TOKEN = os.environ.get("GOOGLE_API_TOKEN", None)
    GOOGLE_CUSTOM_SEARCH_TOKEN = os.environ.get("GOOGLE_CUSTOM_SEARCH_TOKEN", None)
    FAST_MODEL = "gpt-3.5-turbo"
    SMART_MODEL = "gpt-4"
    
    temperature = 0.0
