import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VECTORDB_",
        extra="ignore"
    )

    APP_NAME: str = "AegisVector DB"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", os.environ.get("VECTORDB_PORT", "8000")))
    DATA_DIR: Path = Path("./data")
    AUTO_PERSIST: bool = True
    API_KEY: Optional[str] = None  # If set, requires 'X-API-Key' header on API endpoints
    DEFAULT_INDEX_TYPE: str = "hnsw"
    DEFAULT_METRIC: str = "cosine"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "*"

settings = Settings()

# Ensure data directory exists
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
