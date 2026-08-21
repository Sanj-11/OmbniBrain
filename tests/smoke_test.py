import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_TMP = tempfile.TemporaryDirectory()
_TMP_PATH = pathlib.Path(_TMP.name)

os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_PATH / 'smoke.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP_PATH / "uploads")

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from database import engine  # noqa: E402

with TestClient(app) as client:
    # 1. Register and authenticate
    reg = client.post(
        "/register", json={"username": "alice", "password": "secret123"}
    )
    assert reg.status_code == 200, reg.text
    print("register:", reg.status_code, reg.json()["username"])

    token = client.post(
        "/token",
        data={"username": "alice", "password": "secret123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert token.status_code == 200, token.text
    headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
    print("token issued")

    # 2. Upload a CSV through the Part 2A endpoint
    csv_content = (
        "name,age,salary,active\n"
        "Anna,28,1100.5,True\n"
        "Bob,32,950.25,False\n"
        "Cara,41,2250.75,True\n"
    )
    upload = client.post(
        "/datasets/upload",
        files={"file": ("people.csv", csv_content.encode("utf-8"), "text/csv")},
        headers=headers,
    )
    assert upload.status_code == 200, upload.text
    dataset = upload.json()
    print("uploaded dataset id:", dataset["id"], "rows:", dataset["rows"])

    # 3. Call the new Part 2B-1 structure endpoint
    structure = client.get(f"/datasets/{dataset['id']}/structure", headers=headers)
    assert structure.status_code == 200, structure.text
    body = structure.json()
    print("\n=== Part 2B-1 response ===")
    for key in ("rows", "columns", "column_names", "data_types"):
        print(f"  {key}: {body[key]}")
    assert body["rows"] == 3
    assert body["columns"] == 4
    assert body["column_names"] == ["name", "age", "salary", "active"]
    assert body["data_types"]["age"] == "int64"
    assert body["data_types"]["salary"] == "float64"
    assert body["data_types"]["active"] == "bool"

    # 4. Regression: existing Part 2A endpoints still work
    listing = client.get("/datasets/", headers=headers)
    assert listing.status_code == 200
    assert listing.json()[0]["id"] == dataset["id"]
    print("\nPart 2A list endpoint OK")

    details = client.get(f"/datasets/{dataset['id']}", headers=headers)
    assert details.status_code == 200
    assert details.json()["rows"] == 3
    print("Part 2A details endpoint OK")

    # 5. Unauthorized user cannot access another user's dataset
    client.post("/register", json={"username": "bob", "password": "secret123"})
    bob_token = client.post(
        "/token",
        data={"username": "bob", "password": "secret123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    ).json()["access_token"]
    bob_headers = {"Authorization": f"Bearer {bob_token}"}
    forbidden = client.get(
        f"/datasets/{dataset['id']}/structure", headers=bob_headers
    )
    assert forbidden.status_code == 403, forbidden.text
    print("\nCross-user access correctly denied (403)")

print("\nSMOKE TEST PASSED")

engine.dispose()
_TMP.cleanup()