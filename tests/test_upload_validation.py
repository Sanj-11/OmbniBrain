"""Document upload validation & preparation (Document Upload-2, part 1).

Exercises the upload-time rules of ``POST /datasets/upload`` before a file is
saved or processed: file presence, filename, allowed file type/extension,
empty uploads, and streaming saves. Query processing is explicitly out of
scope for this part and is not touched here.
"""

import io

import pandas as pd

import crud

from config import settings

USERNAME = "alice"
PASSWORD = "secret123"

VALID_CSV = (
    "name,age,salary,active\n"
    "Anna,28,1100.5,True\n"
    "Bob,32,950.25,False\n"
)

XLSX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


def build_xlsx_bytes(rows: list):
    """Turn a list of dicts into in-memory ``.xlsx`` bytes (openpyxl)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False)
    return buffer.getvalue()


def upload(api, filename, content_bytes, content_type="text/csv"):
    """Create the user and POST a single ``file`` part with auth headers."""
    api.create_user(USERNAME, PASSWORD)
    return api.client.post(
        "/datasets/upload",
        files={"file": (filename, content_bytes, content_type)},
        headers=api.auth_headers(USERNAME),
    )


def upload_raw_part_without_content_type(api, filename, content_bytes):
    """POST a hand-built multipart body whose part has no Content-Type header.

    A multipart part is allowed to omit ``Content-Type``; Starlette then
    reports ``UploadFile.content_type`` as ``None``, which the endpoint has to
    cope with because ``Dataset.file_type`` is a NOT NULL column.
    """
    api.create_user(USERNAME, PASSWORD)
    boundary = "----omnibrain-test-boundary"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            b"\r\n",
            content_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    headers = api.auth_headers(USERNAME)
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    return api.client.post("/datasets/upload", content=body, headers=headers)


# ---------------------------------------------------------------------------
# Valid uploads
# ---------------------------------------------------------------------------


def test_valid_csv_is_accepted(api):
    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["original_filename"] == "people.csv"
    assert body["rows"] == 2
    assert body["columns"] == 4
    assert body["id"] > 0


def test_valid_excel_is_accepted(api):
    content = build_xlsx_bytes(
        [{"name": "Alice", "score": 10}, {"name": "Bob", "score": 20}]
    )
    response = upload(api, "scores.xlsx", content, XLSX_CONTENT_TYPE)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 2
    assert body["columns"] == 2


def test_mixed_case_extension_is_accepted(api):
    response = upload(api, "PEOPLE.CSV", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text


def test_hostile_filename_is_sanitized(api):
    # Path-like filenames must never escape the upload directory.
    response = upload(api, "../../evil.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text

    stored = list(settings.UPLOAD_DIR.iterdir())
    assert len(stored) == 1
    assert stored[0].parent == settings.UPLOAD_DIR
    assert ".." not in stored[0].name
    assert stored[0].read_bytes().decode("utf-8") == VALID_CSV


def test_valid_upload_persists_one_stored_file(api):
    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text

    stored = list(settings.UPLOAD_DIR.iterdir())
    assert len(stored) == 1
    assert stored[0].read_bytes().decode("utf-8") == VALID_CSV


def test_repeated_uploads_of_same_name_do_not_overwrite(api):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)
    ids = []
    for _ in range(2):
        response = api.client.post(
            "/datasets/upload",
            files={"file": ("people.csv", VALID_CSV.encode("utf-8"), "text/csv")},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        ids.append(response.json()["id"])

    assert len(set(ids)) == 2
    assert len(list(settings.UPLOAD_DIR.iterdir())) == 2


def test_surrounding_whitespace_in_filename_is_normalized(api):
    # " people.csv " is a valid CSV name once trimmed; the trimmed name is what
    # gets recorded, so later parser choices keyed off the extension still work.
    response = upload(api, "  people.csv  ", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text
    assert response.json()["original_filename"] == "people.csv"


def test_overlong_filename_is_stored_safely(api):
    # A pathological 300-character name must not blow past the OS path limit.
    response = upload(api, "a" * 300 + ".csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text

    stored = list(settings.UPLOAD_DIR.iterdir())
    assert len(stored) == 1
    assert len(stored[0].name) < 200
    assert stored[0].name.endswith(".csv")


def test_extension_only_filename_keeps_extension_on_disk(api):
    # ".csv" sanitises to a name without an extension; the validated extension
    # is re-attached so the stored file stays identifiable.
    response = upload(api, ".csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text

    stored = list(settings.UPLOAD_DIR.iterdir())
    assert len(stored) == 1
    assert stored[0].name.endswith(".csv")


def test_missing_part_content_type_falls_back_to_extension(api):
    # ``file_type`` is NOT NULL, so a part without Content-Type must not 500.
    response = upload_raw_part_without_content_type(
        api, "people.csv", VALID_CSV.encode("utf-8")
    )
    assert response.status_code == 200, response.text
    assert response.json()["file_type"] == "text/csv"


def test_upload_within_size_limit_is_accepted(api, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1.0)
    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text


def test_size_limit_can_be_disabled(api, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0)
    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text

# ---------------------------------------------------------------------------
# Invalid uploads
# ---------------------------------------------------------------------------


def test_dotless_filename_is_rejected(api):
    response = upload(api, "csv_without_extension", VALID_CSV.encode("utf-8"))
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type"


def test_unsupported_extension_is_rejected(api):
    response = upload(api, "notes.txt", b"hello")
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type"


def test_empty_file_is_rejected(api):
    response = upload(api, "empty.csv", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_empty_excel_file_is_rejected(api):
    response = upload(api, "empty.xlsx", b"", XLSX_CONTENT_TYPE)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_header_only_csv_is_rejected(api):
    # A file carrying a header but no data rows is an empty dataset.
    response = upload(api, "headers.csv", b"name,age\n")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_missing_file_part_returns_422(api):
    # ``file`` is a required UploadFile parameter, so an absent part is a
    # request-validation error (422), not a business-rule error.
    api.create_user(USERNAME, PASSWORD)
    response = api.client.post(
        "/datasets/upload", headers=api.auth_headers(USERNAME)
    )
    assert response.status_code == 422


def test_file_part_without_valid_filename_is_rejected(api):
    # A multipart part with a missing or empty filename is not a valid file
    # upload for FastAPI's required ``UploadFile`` parameter, so request
    # validation rejects it with 422 (a client bug, caught before the
    # endpoint's own business-rule guards run).
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)
    for filename in (None, ""):
        response = api.client.post(
            "/datasets/upload",
            files={"file": (filename, VALID_CSV.encode("utf-8"), "text/csv")},
            headers=headers,
        )
        assert response.status_code == 422


def test_corrupt_excel_payload_is_rejected(api):
    response = upload(api, "corrupt.xlsx", b"definitely not an xlsx", XLSX_CONTENT_TYPE)
    assert response.status_code == 400
    assert "could not be processed" in response.json()["detail"].lower()


def test_zip_like_corrupt_excel_payload_is_rejected(api):
    # ZIP magic bytes make pandas pick the xlsx engine, which then fails on the
    # truncated archive; that must be a clean 400, not a 500.
    response = upload(
        api, "corrupt.xlsx", b"PK\x03\x04" + b"\x00" * 64, XLSX_CONTENT_TYPE
    )
    assert response.status_code == 400
    assert "could not be processed" in response.json()["detail"].lower()
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_unreadable_binary_csv_is_rejected(api):
    response = upload(api, "junk.csv", bytes(range(256)) * 8)
    assert response.status_code == 400
    assert "could not be processed" in response.json()["detail"].lower()
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_unparsable_legacy_xls_is_rejected(api):
    # OLE2 magic bytes make pandas select the legacy .xls engine. Whether or
    # not the optional engine is installed, an unparsable payload must produce
    # a 400 and leave nothing on disk (never a 500).
    payload = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512
    response = upload(api, "legacy.xls", payload, "application/vnd.ms-excel")
    assert response.status_code == 400
    assert "could not be processed" in response.json()["detail"].lower()
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_whitespace_only_filename_is_rejected(api):
    response = upload(api, "   ", VALID_CSV.encode("utf-8"))
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid filename"
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_oversized_upload_is_rejected(api, monkeypatch):
    # 1 KB cap versus a ~200 KB payload: the stream is aborted, the partial
    # file removed, and no dataset row created.
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0.001)
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)
    big_csv = ("name,age\n" + "Anna,28\n" * 25_000).encode("utf-8")

    response = api.client.post(
        "/datasets/upload",
        files={"file": ("big.csv", big_csv, "text/csv")},
        headers=headers,
    )
    assert response.status_code == 413, response.text
    assert "maximum upload size" in response.json()["detail"].lower()
    assert list(settings.UPLOAD_DIR.iterdir()) == []

    listing = api.client.get("/datasets/", headers=headers)
    assert listing.status_code == 200
    assert listing.json() == []


def test_unauthenticated_upload_returns_401(api):
    response = api.client.post(
        "/datasets/upload",
        files={"file": ("x.csv", VALID_CSV.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 401
    # Authentication is enforced before the payload is written anywhere.
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_rejected_uploads_leave_no_files_behind(api):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)

    # Unsupported extension
    api.client.post(
        "/datasets/upload",
        files={"file": ("bad.txt", b"x", "text/plain")},
        headers=headers,
    )
    # Empty file
    api.client.post(
        "/datasets/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
        headers=headers,
    )
    # Header-only file
    api.client.post(
        "/datasets/upload",
        files={"file": ("headers.csv", b"a,b\n", "text/csv")},
        headers=headers,
    )
    # Corrupt excel
    api.client.post(
        "/datasets/upload",
        files={"file": ("corrupt.xlsx", b"junk", XLSX_CONTENT_TYPE)},
        headers=headers,
    )
    # Whitespace-only filename
    api.client.post(
        "/datasets/upload",
        files={"file": ("  ", VALID_CSV.encode("utf-8"), "text/csv")},
        headers=headers,
    )

    assert list(settings.UPLOAD_DIR.iterdir()) == []
    # ... and no dataset rows were created either.
    listing = api.client.get("/datasets/", headers=headers)
    assert listing.status_code == 200
    assert listing.json() == []


def test_larger_valid_file_is_accepted(api):
    # A ~2 MB CSV must be streamed to disk without loading it all into memory
    # at once and must remain well below the default 100 MB cap.
    big_csv = ("name,age,salary\n" + "Anna,28,1100.5\n" * 70_000).encode("utf-8")
    response = upload(api, "big.csv", big_csv)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 70_000
    assert body["columns"] == 3

    stored = list(settings.UPLOAD_DIR.iterdir())
    assert len(stored) == 1
    assert stored[0].read_bytes() == big_csv


def test_multiple_sequential_uploads_are_all_accepted(api):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)
    for i in range(3):
        response = api.client.post(
            "/datasets/upload",
            files={"file": (f"data_{i}.csv", VALID_CSV.encode("utf-8"), "text/csv")},
            headers=headers,
        )
        assert response.status_code == 200, response.text

    stored = list(settings.UPLOAD_DIR.iterdir())
    assert len(stored) == 3
    assert len(set(f.name for f in stored)) == 3

    listing = api.client.get("/datasets/", headers=headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 3


def test_file_write_failure_returns_500_without_paths(api, monkeypatch, tmp_path):
    # Simulate a permission error or disk-full condition by making open()
    # raise OSError for files inside the upload directory.
    original_open = open

    def failing_open(path, mode="rb", *args, **kwargs):
        if str(settings.UPLOAD_DIR) in str(path):
            raise OSError(13, "Permission denied", str(path))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)

    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 500, response.text
    body = response.json()
    assert "path" not in body.get("detail", "").lower()
    assert "uploaded file" in body.get("detail", "").lower()
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_corrupt_file_error_does_not_expose_paths(api):
    response = upload(api, "junk.csv", bytes(range(256)) * 8)
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "path" not in detail.lower()
    assert "could not be processed" in detail.lower()
    assert list(settings.UPLOAD_DIR.iterdir()) == []


def test_database_failure_cleans_up_file_and_leaves_no_record(api, monkeypatch):
    # Simulate a database failure after the file has been saved to disk.
    # The endpoint must delete the saved file and return 500 without exposing
    # any internal details.
    original_create_dataset = crud.create_dataset

    def failing_create_dataset(db, dataset_data, user_id):
        raise Exception("database is down")

    monkeypatch.setattr(crud, "create_dataset", failing_create_dataset)

    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 500, response.text
    detail = response.json().get("detail", "")
    assert "path" not in detail.lower()
    assert list(settings.UPLOAD_DIR.iterdir()) == []

    headers = api.auth_headers(USERNAME)
    listing = api.client.get("/datasets/", headers=headers)
    assert listing.status_code == 200
    assert listing.json() == []


def test_successful_upload_creates_metadata_in_db(api):
    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text
    body = response.json()
    dataset_id = body["id"]

    # The file must exist on disk.
    stored = list(settings.UPLOAD_DIR.iterdir())
    assert len(stored) == 1
    assert stored[0].read_bytes().decode("utf-8") == VALID_CSV

    # The database record must contain the correct metadata.
    assert body["original_filename"] == "people.csv"
    assert body["rows"] == 2
    assert body["columns"] == 4
    assert body["file_type"] == "text/csv"
    assert body["file_size"] > 0

    # The record must be retrievable via the dataset endpoint.
    headers = api.auth_headers(USERNAME)
    details = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details.status_code == 200
    assert details.json()["id"] == dataset_id


def test_rejected_upload_does_not_create_db_record(api):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)

    for filename, content, content_type in [
        ("bad.txt", b"hello", "text/plain"),
        ("empty.csv", b"", "text/csv"),
        ("headers.csv", b"name,age\n", "text/csv"),
        ("corrupt.xlsx", b"junk", XLSX_CONTENT_TYPE),
    ]:
        api.client.post(
            "/datasets/upload",
            files={"file": (filename, content, content_type)},
            headers=headers,
        )

    assert list(settings.UPLOAD_DIR.iterdir()) == []
    listing = api.client.get("/datasets/", headers=headers)
    assert listing.status_code == 200
    assert listing.json() == []


def test_existing_endpoints_still_work_after_upload(api):
    response = upload(api, "people.csv", VALID_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text
    dataset_id = response.json()["id"]
    headers = api.auth_headers(USERNAME)

    # Listing
    listing = api.client.get("/datasets/", headers=headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    # Details
    details = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details.status_code == 200
    assert details.json()["rows"] == 2

    # Delete
    delete = api.client.delete(f"/datasets/{dataset_id}", headers=headers)
    assert delete.status_code == 204

    listing_after = api.client.get("/datasets/", headers=headers)
    assert listing_after.status_code == 200
    assert listing_after.json() == []


def test_docs_endpoint_works(api):
    response = api.client.get("/docs")
    assert response.status_code == 200
