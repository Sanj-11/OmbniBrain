import datetime as dt
import io

import pandas as pd

USERNAME = "alice"
PASSWORD = "secret123"

SAMPLE_CSV = (
    "name,age,salary,active\n"
    "Anna,28,1100.5,True\n"
    "Bob,32,950.25,False\n"
    "Cara,41,2250.75,True\n"
    "Dan,19,430.0,True\n"
)


def upload_csv(api, content, filename="sample.csv"):
    """Upload a CSV through the Part 2A endpoint and return the dataset id."""
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)
    response = api.client.post(
        "/datasets/upload",
        files={"file": (filename, content.encode("utf-8"), "text/csv")},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def profile(api, dataset_id, headers):
    return api.client.get(f"/datasets/{dataset_id}/structure", headers=headers)


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------


def test_valid_dataset_returns_correct_row_count(api):
    dataset_id = upload_csv(api, SAMPLE_CSV)
    response = profile(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200
    assert response.json()["rows"] == 4


def test_valid_dataset_returns_correct_column_count(api):
    dataset_id = upload_csv(api, SAMPLE_CSV)
    response = profile(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200
    assert response.json()["columns"] == 4


def test_valid_dataset_returns_correct_column_names(api):
    dataset_id = upload_csv(api, SAMPLE_CSV)
    response = profile(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200
    assert response.json()["column_names"] == ["name", "age", "salary", "active"]


def test_valid_dataset_returns_real_pandas_data_types(api):
    dataset_id = upload_csv(api, SAMPLE_CSV)
    response = profile(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200
    assert response.json()["data_types"] == {
        "name": "object",
        "age": "int64",
        "salary": "float64",
        "active": "bool",
    }
def test_datetime_data_type_is_reported_for_excel_files(api):
    """Datetime columns of an Excel file must come through unchanged."""
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "name": ["Anna", "Bob"],
                "joined": [
                    dt.datetime(2024, 5, 6, 10, 30),
                    dt.datetime(2024, 7, 1, 9, 0),
                ],
            }
        ).to_excel(writer, index=False)

    response = api.client.post(
        "/datasets/upload",
        files={
            "file": (
                "dates.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    dataset_id = response.json()["id"]

    result = profile(api, dataset_id, headers).json()
    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["column_names"] == ["name", "joined"]
    assert result["data_types"]["joined"] == "datetime64[ns]"
    assert result["data_types"]["name"] == "object"


def test_response_contains_all_required_fields(api):
    dataset_id = upload_csv(api, SAMPLE_CSV)
    body = profile(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert {"rows", "columns", "column_names", "data_types"} <= set(body.keys())
    assert body["id"] == dataset_id


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = profile(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = profile(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_missing_file_returns_404(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    missing = tmp_path / "gone.csv"
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="gone.csv",
        file_path=missing,
    )
    response = profile(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "file not found" in response.json()["detail"].lower()
def test_corrupted_file_returns_400(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"this is definitely not a valid xlsx or csv payload")
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="corrupt.xlsx",
        file_path=corrupt,
        file_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response = profile(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400


def test_empty_dataset_is_handled(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    # A header-only file never passes the upload endpoint (it is rejected
    # there), so insert the record directly to exercise this path.
    empty = tmp_path / "empty.csv"
    empty.write_text("name,age\n", encoding="utf-8")
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="empty.csv",
        file_path=empty,
        rows=0,
        columns=2,
    )
    response = profile(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_still_rejects_empty_dataset(api):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)
    response = api.client.post(
        "/datasets/upload",
        files={"file": ("empty.csv", "".encode("utf-8"), "text/csv")},
        headers=headers,
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Authentication / ownership
# ---------------------------------------------------------------------------


def test_unauthenticated_request_returns_401(api):
    response = api.client.get("/datasets/1/structure")
    assert response.status_code == 401


def test_cannot_profile_another_users_dataset(api, tmp_path):
    api.create_user(USERNAME, PASSWORD)
    bob = api.create_user("bob", PASSWORD)

    bob_file = tmp_path / "bobs.csv"
    bob_file.write_text("a,b\n1,2\n", encoding="utf-8")
    bob_dataset = api.create_dataset_record(
        owner_id=bob.id,
        original_filename="bobs.csv",
        file_path=bob_file,
        rows=1,
        columns=2,
    )

    response = profile(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_profile_own_dataset(api):
    dataset_id = upload_csv(api, SAMPLE_CSV)
    assert profile(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


# ---------------------------------------------------------------------------
# Part 2A regression guard: existing dataset endpoints keep working
# ---------------------------------------------------------------------------


def test_parta_endpoints_still_work(api):
    headers = api.auth_headers(USERNAME)
    dataset_id = upload_csv(api, SAMPLE_CSV)

    list_response = api.client.get("/datasets/", headers=headers)
    assert list_response.status_code == 200
    assert [d["id"] for d in list_response.json()] == [dataset_id]

    details_response = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details_response.status_code == 200
    assert details_response.json()["rows"] == 4
    assert details_response.json()["columns"] == 4
