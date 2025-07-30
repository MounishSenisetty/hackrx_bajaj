"""
Configuration settings for the LLM-Powered Intelligent Query-Retrieval System
"""

import os
from typing import Dict, Any

class Config:
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_WORKERS = int(os.getenv("API_WORKERS", 1))
    
    # Authentication
    BEARER_TOKEN = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBBi1RQgXXxh4CvbByxAdTB9yhZqxIqyBQ")
    DEFAULT_LLM_MODEL = "gpt-4-turbo-preview"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.1
    
    # Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Document Processing
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Vector Search
    MAX_RELEVANT_CHUNKS = 5
    SIMILARITY_THRESHOLD = 0.3
    
    # Performance
    REQUEST_TIMEOUT = 60.0
    DOCUMENT_FETCH_TIMEOUT = 30.0
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_system_config(cls) -> Dict[str, Any]:
        """Get system configuration as dictionary."""
        return {
            "api_host": cls.API_HOST,
            "api_port": cls.API_PORT,
            "embedding_model": cls.EMBEDDING_MODEL,
            "max_chunk_size": cls.MAX_CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "max_relevant_chunks": cls.MAX_RELEVANT_CHUNKS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "max_document_size": cls.MAX_DOCUMENT_SIZE,
            "request_timeout": cls.REQUEST_TIMEOUT
        }
