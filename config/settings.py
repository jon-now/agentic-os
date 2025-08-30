import os
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings
import json

class Settings(BaseSettings):
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Application Settings
    APP_NAME: str = "Agentic Orchestrator"
    DEBUG: bool = True
    HOST: str = "localhost"
    PORT: int = 8000

    # File Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    CREDENTIALS_PATH: Path = PROJECT_ROOT / "credentials"
    LOGS_PATH: Path = PROJECT_ROOT / "logs"
    DATA_PATH: Path = PROJECT_ROOT / "data"
    MEMORY_PATH: Path = PROJECT_ROOT / "memory"

    # Gmail API
    GMAIL_CREDENTIALS_FILE: str = "credentials.json"
    GMAIL_TOKEN_FILE: str = "gmail_token.pickle"

    # Browser Settings
    BROWSER_HEADLESS: bool = True
    BROWSER_TIMEOUT: int = 30

    # LLM Settings
    LLM_TIMEOUT: int = 30
    MAX_CONTEXT_LENGTH: int = 4096

    # Vector Database Settings
    VECTOR_DB_TYPE: str = "chromadb"  # chromadb or faiss
    CHROMADB_PATH: Path = PROJECT_ROOT / "memory" / "chromadb"

    # Voice Settings
    WHISPER_MODEL: str = "base"
    TTS_MODEL: str = "piper"

    # RTX 3050 6GB GPU Optimizations
    GPU_MEMORY_FRACTION: float = 0.7
    USE_QUANTIZED_MODELS: bool = True
    VOICE_MODEL_SIZE: str = "small"
    ENABLE_MEMORY_MONITORING: bool = True
    MAX_CONCURRENT_GPU_TASKS: int = 1

    class Config:
        env_file = ".env"
        case_sensitive = True

    def create_directories(self):
        """Create necessary directories"""
        for path in [self.CREDENTIALS_PATH, self.LOGS_PATH, self.DATA_PATH, self.MEMORY_PATH, self.CHROMADB_PATH]:
            path.mkdir(exist_ok=True, parents=True)

settings = Settings()
settings.create_directories()
