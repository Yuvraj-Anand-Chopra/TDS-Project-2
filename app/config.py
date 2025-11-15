import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application configuration settings"""

    # Email and Authentication
    EMAIL: str = os.getenv("EMAIL", "")
    SECRET: str = os.getenv("SECRET", "")

    # API Configuration
    API_ENDPOINT: str = os.getenv("API_ENDPOINT", "http://localhost:8000")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")

    # Google Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Security
    SYSTEM_PROMPT_MAX_CHARS: int = 100
    USER_PROMPT_MAX_CHARS: int = 100

    # Timeouts (in seconds)
    REQUEST_TIMEOUT: int = 180  # 3 minutes as per spec
    BROWSER_TIMEOUT: int = 30000  # milliseconds
    PDF_TIMEOUT: int = 60
    API_TIMEOUT: int = 30

    # Retry Configuration
    MAX_RETRIES: int = 3
    INITIAL_RETRY_DELAY: float = 1.0
    BACKOFF_FACTOR: float = 2.0
    MAX_BACKOFF_DELAY: float = 30.0

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Development
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENV: str = os.getenv("ENV", "development")

settings = Settings()
