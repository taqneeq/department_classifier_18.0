import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration with environment variable support"""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False
    
    # Classification settings - FIXED FOR PROPER QUESTIONING
    CONFIDENCE_THRESHOLD: float = 0.65  # Lowered from 0.85 to allow more questions
    SECONDARY_THRESHOLD: float = 0.65   # Lowered from 0.70
    EARLY_TERMINATION_THRESHOLD: float = 0.70  # Lowered from 0.80 - disabled early termination
    MAX_QUESTIONS: int = 15
    MIN_QUESTIONS: int = 4   # Minimum seed questions
    MIN_ADAPTIVE_QUESTIONS: int = 8  # Increased from 2 - force at least 6 adaptive questions
    LEARNING_RATE: float = 0.4  # Increased from 0.3 for faster learning
    
    # RAG settings
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    ENABLE_RAG: bool = True
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_CHUNKS_PER_DEPT: int = 5
    
    # Optional external API keys
    HF_TOKEN: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    
    # Data paths (relative to backend directory)
    DATA_DIR: str = "app/data"
    DEPARTMENTS_FILE: str = "app/data/departments.json"
    QUESTIONS_FILE: str = "app/data/question_bank.json"
    PDF_FILE: str = "app/data/departments.pdf"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*"  # Allow all origins in development
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

settings = Settings()