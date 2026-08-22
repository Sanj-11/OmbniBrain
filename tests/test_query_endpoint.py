"""Query endpoint validation tests (Asynchronous Query Processing – Part 1).

Covers:
- Valid query (with and without dataset_id)
- Empty query
- Invalid dataset/document ID
- Unauthorized request
- Invalid request body
"""

import pathlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import crud
import models
import schemas
from config import settings
from database import get_db
from main import app


@pytest.fixture()
def api(tmp_path, monkeypatch):
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
                return crud.create_user(session, {"username": username, "password": password})
            finally:
                session.close()

        def auth_headers(self, username: str = "alice") -> dict:
            from security import create_access_token
            token = create_access_token(data={"sub": username})
            return {"Authorization": f"Bearer {token}"}

        def create_dataset_record(
            self,
            *,
            owner_id: int,
            original_filename: str,
            file_path: pathlib.Path,
            rows: int = 4,
            columns: int = 5,
            file_type: str = "text/csv",
        ) -> models.Dataset:
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
            except FileNotFoundError:
                file_size_mb = 0.0
            session = TestSession()
            try:
                return crud.create_dataset(
                    session,
                    {
                        "original_filename": original_filename,
                        "stored_filename": f"test_{pathlib.Path(original_filename).name}",
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


class TestQueryEndpoint:
    def test_valid_query_without_dataset(self, api):
        api.create_user("alice")
        resp = api.client.post(
            "/query/",
            json={"text": "show me sales data"},
            headers=api.auth_headers("alice"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "General query: show me sales data"
        assert body["dataset_id"] is None
        assert body["rows_analyzed"] == 0
        assert body["columns_analyzed"] == 0

    def test_valid_query_with_owned_dataset(self, api, tmp_path):
        user = api.create_user("bob")
        ds_file = tmp_path / "data.csv"
        ds_file.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")
        dataset = api.create_dataset_record(
            owner_id=user.id,
            original_filename="data.csv",
            file_path=ds_file,
            rows=4,
            columns=2,
        )

        resp = api.client.post(
            "/query/",
            json={"text": "summarize", "dataset_id": dataset.id},
            headers=api.auth_headers("bob"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["dataset_id"] == dataset.id
        assert body["rows_analyzed"] == 4
        assert body["columns_analyzed"] == 2
        assert "summarize" in body["answer"]

    def test_empty_query_returns_400(self, api):
        api.create_user("alice")
        resp = api.client.post(
            "/query/",
            json={"text": "   "},
            headers=api.auth_headers("alice"),
        )
        assert resp.status_code == 400
        assert "cannot be empty" in resp.json()["detail"].lower()

    def test_missing_text_field_returns_422(self, api):
        api.create_user("alice")
        resp = api.client.post(
            "/query/",
            json={"dataset_id": 1},
            headers=api.auth_headers("alice"),
        )
        assert resp.status_code == 422

    def test_invalid_dataset_id_returns_404(self, api):
        api.create_user("alice")
        resp = api.client.post(
            "/query/",
            json={"text": "hello", "dataset_id": 9999999},
            headers=api.auth_headers("alice"),
        )
        assert resp.status_code == 404
        assert "Dataset not found" in resp.json()["detail"]

    def test_unauthorized_request_returns_401(self, api):
        resp = api.client.post("/query/", json={"text": "hello"})
        assert resp.status_code == 401

    def test_cross_user_dataset_access_returns_403(self, api, tmp_path):
        owner = api.create_user("owner")
        intruder = api.create_user("intruder")
        ds_file = tmp_path / "secret.csv"
        ds_file.write_text("a,b\n1,2\n")
        dataset = api.create_dataset_record(
            owner_id=owner.id,
            original_filename="secret.csv",
            file_path=ds_file,
        )

        resp = api.client.post(
            "/query/",
            json={"text": "steal data", "dataset_id": dataset.id},
            headers=api.auth_headers("intruder"),
        )
        assert resp.status_code == 403
        assert "Not authorized" in resp.json()["detail"]
