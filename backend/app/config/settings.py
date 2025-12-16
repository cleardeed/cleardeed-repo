"""
Application Configuration Management

This module manages all application settings using Pydantic BaseSettings.
Configuration can be loaded from environment variables or .env file.

Responsibilities:
- Load and validate environment variables
- Provide typed configuration access
- Manage API keys, file paths, and service URLs
- Configure processing limits and timeouts
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf"]
    
    # PDF Processing Configuration
    MAX_PDF_PAGES: int = 500
    SUPPORTED_LANGUAGES: List[str] = ["en", "ta"]  # English, Tamil
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gap-english"  # Custom GAP analysis model
    OLLAMA_TIMEOUT: int = 300  # 5 minutes
    
    # LM Studio Configuration (Alternative)
    LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LM_STUDIO_ENABLED: bool = False
    
    # Processing Configuration
    ASYNC_PROCESSING: bool = True
    MAX_CONCURRENT_JOBS: int = 5
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_upload_path() -> str:
    """Get absolute path for upload directory"""
    upload_path = os.path.abspath(settings.UPLOAD_DIR)
    os.makedirs(upload_path, exist_ok=True)
    return upload_path
