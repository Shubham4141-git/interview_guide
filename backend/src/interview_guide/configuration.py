

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file, if present
load_dotenv()

def _first(*keys: str) -> Optional[str]:
    """
    Return the value of the first environment variable found in keys.
    """
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
    return None

class Settings(BaseModel):
    llm_provider: Optional[str] = os.getenv("LLM_PROVIDER", "google")
    llm_model: Optional[str] = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    google_api_key: Optional[str] = _first("GOOGLE_API_KEY", "GEMINI_API_KEY")
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    debug: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")


# Singleton instance for app-wide settings
settings = Settings()