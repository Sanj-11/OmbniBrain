import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    UPLOAD_DIR: Path = Path("uploads")

    # Upper bound (in megabytes) for a single uploaded dataset. The upload is
    # streamed to disk and aborted as soon as this many bytes have been read,
    # so an oversized upload never fills the disk. Set MAX_UPLOAD_SIZE_MB=0 in
    # the environment to disable the size check.
    MAX_UPLOAD_SIZE_MB: float = 100.0

    class Config:
        env_file = ".env"

settings = Settings()