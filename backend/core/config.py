"""
Configuration Management
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_TITLE: str = "AutoML Pipeline Builder"
    API_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB
    UPLOAD_DIR: Path = Path("tmp/uploads")

    # ML Settings
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_RANDOM_STATE: int = 42

    # Supported formats
    SUPPORTED_FORMATS: set = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
