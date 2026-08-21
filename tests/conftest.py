import pathlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import crud
import models
import security
from config import settings
from database import get_db
from main import app


@pytest.fixture()
def api(tmp_path, monkeypatch):
    """Provide a TestClient wired to a throwaway database and upload folder.

    Returns a small context object with convenience helpers:

    * ``create_user(username)``        - creates a user in the test DB
    * ``auth_headers(username)``       - valid Bearer token header
    * ``create_dataset_record(...)``   - inserts a dataset row directly
    """
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'test_omnibrain.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    models.Base.metadata.create_all(bind=test_engine)

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", pathlib.Path(upload_dir))

    def override_get_db():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    class ApiContext:
        def __init__(self, client: TestClient):
            self.client = client

        def create_user(self, username: str, password: str = "secret123") -> models.User:
            session = TestSession()
            try:
                return crud.create_user(
                    session, {"username": username, "password": password}
                )
            finally:
                session.close()

        def auth_headers(self, username: str = "alice") -> dict:
            token = security.create_access_token(data={"sub": username})
            return {"Authorization": f"Bearer {token}"}

        def create_dataset_record(
            self,
            *,
            owner_id: int,
            original_filename: str,
            file_path: pathlib.Path,
            rows: int = 1,
            columns: int = 1,
            file_type: str = "text/csv",
        ) -> models.Dataset:
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
            except FileNotFoundError:
                # The missing-file test intentionally points at a path that
                # does not exist; the database record is still created.
                file_size_mb = 0.0
            session = TestSession()
            try:
                return crud.create_dataset(
                    session,
                    {
                        "original_filename": original_filename,
                        "stored_filename": f"test_{uuid.uuid4().hex}_{original_filename}",
                        "file_path": str(file_path),
                        "file_type": file_type,
                        "file_size": file_size_mb,
                        "rows": rows,
                        "columns": columns,
                    },
                    user_id=owner_id,
                )
            finally:
                session.close()

    client = TestClient(app)
    api_context = ApiContext(client)
    yield api_context

    app.dependency_overrides.clear()
    client.close()
    test_engine.dispose()