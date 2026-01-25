

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file, if present
_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_ROOT / ".env")

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
    llm_provider: Optional[str] = os.getenv("LLM_PROVIDER", "openai")
    llm_model_default: Optional[str] = (
        os.getenv("LLM_MODEL_DEFAULT")
        or os.getenv("LLM_MODEL")
        or "gpt-5-nano"
    )
    llm_model_eval: Optional[str] = os.getenv("LLM_MODEL_EVAL") or "gpt-4o-mini"
    openai_api_key: Optional[str] = _first("OPENAI_API_KEY")
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    total_questions_from_jd: int = int(os.getenv("TOTAL_QUESTIONS_FROM_JD", "10"))
    debug: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")


# Singleton instance for app-wide settings
settings = Settings()
