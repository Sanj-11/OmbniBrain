import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    UPLOAD_DIR: Path = Path("uploads")

    class Config:
        env_file = ".env"

settings = Settings()