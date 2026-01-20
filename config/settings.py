#filepath: config/settings.py

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Load from .env file
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    # --- Core LLM (Gemini) ---
    GOOGLE_API_KEY: str = Field(default="")
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE_SEARCH: float = 0.5
    GEMINI_MAX_TOKENS_SEARCH: int = 8192

    # Provider selection for the Search Tool (defaults to 'google' per your request)
    LLM_PROVIDER: str = "google"
    SEARCH_TOOL_LLM_PROVIDER: str = "google"

    # --- E2B Sandbox ---
    E2B_API_KEY: str = Field(default="")
    # Leave blank to use the default "code-interpreter" template in the E2B handler.
    E2B_TEMPLATE_ID: str = Field(default="w31jz51oudo4ahby7svb")


    # --- Tavily Search ---
    TAVILY_API_KEY: str = Field(default="")

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Optional: Compatibility for the provided SearchingTool reference ---
    # These are included so your searching_tool.py doesn't crash on import
    # even if you are only using Gemini.
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL_NAME: str = "claude-3-opus-20240229"
    ANTHROPIC_TEMPERATURE_SEARCH: float = 0.5
    ANTHROPIC_MAX_TOKENS_SEARCH: int = 1024

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4-turbo"
    OPENAI_TEMPERATURE_SEARCH: float = 0.5
    OPENAI_MAX_TOKENS_SEARCH: int = 256

    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_API_BASE: str = ""
    DEEPSEEK_MODEL_NAME: str = "deepseek-chat"
    DEEPSEEK_TEMPERATURE_SEARCH: float = 0.5
    DEEPSEEK_MAX_TOKENS_SEARCH: int = 256

settings = Settings()
