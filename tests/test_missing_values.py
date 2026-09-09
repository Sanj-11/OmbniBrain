import datetime as dt
import io

import pandas as pd

USERNAME = "alice"
PASSWORD = "secret123"

# 4 rows / 5 columns; each of "name", "age", "department" and "active" has
# exactly one missing value, while "salary" has none.  Total missing = 4.
MISSING_CSV = (
    "name,age,salary,department,active\n"
    "Anna,28,1100.5,Engineering,True\n"
    "Bob,32,950.25,,False\n"
    "Cara,,2250.75,Marketing,True\n"
    ",19,430.0,Marketing,\n"
)

NO_MISSING_CSV = (
    "name,age,salary\n"
    "Anna,28,1100.5\n"
    "Bob,32,950.25\n"
    "Cara,41,2250.75\n"
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


def missing_profile(api, dataset_id, headers):
    return api.client.get(f"/datasets/{dataset_id}/missing-values", headers=headers)
# ---------------------------------------------------------------------------
# Dataset with missing values
# ---------------------------------------------------------------------------


def test_missing_values_are_reported_correctly(api):
    dataset_id = upload_csv(api, MISSING_CSV)
    response = missing_profile(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["total_rows"] == 4
    assert body["total_missing_values"] == 4

    expected = {
        "name": {"missing_count": 1, "missing_percentage": 25.0},
        "age": {"missing_count": 1, "missing_percentage": 25.0},
        "salary": {"missing_count": 0, "missing_percentage": 0.0},
        "department": {"missing_count": 1, "missing_percentage": 25.0},
        "active": {"missing_count": 1, "missing_percentage": 25.0},
    }
    for column, profile in expected.items():
        assert column in body["columns"], column
        assert body["columns"][column]["missing_count"] == profile["missing_count"]
        assert (
            body["columns"][column]["missing_percentage"]
            == profile["missing_percentage"]
        )


def test_missing_values_response_has_all_required_fields(api):
    dataset_id = upload_csv(api, MISSING_CSV)
    body = missing_profile(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert {"total_rows", "total_missing_values", "columns"} <= set(body.keys())
    assert set(body["columns"]) == {"name", "age", "salary", "department", "active"}


# ---------------------------------------------------------------------------
# Dataset with no missing values
# ---------------------------------------------------------------------------


def test_dataset_without_missing_values_succeeds(api):
    dataset_id = upload_csv(api, NO_MISSING_CSV)
    response = missing_profile(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200
    body = response.json()

    assert body["total_missing_values"] == 0
    for column in ("name", "age", "salary"):
        assert body["columns"][column]["missing_count"] == 0
        assert body["columns"][column]["missing_percentage"] == 0.0
# ---------------------------------------------------------------------------
# Different data types
# ---------------------------------------------------------------------------


def test_missing_values_detected_across_data_types(api):
    """NaN/None/NaT must be detected in numeric, string, bool and datetime."""
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)

    frame = pd.DataFrame(
        {
            "numeric": [1.5, None, 3.5, 4.5],
            "string": ["a", None, "c", "d"],
            "boolean": [True, False, None, True],
            "datetime": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 2, 1),
                None,
                dt.datetime(2024, 4, 1),
            ],
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)

    response = api.client.post(
        "/datasets/upload",
        files={
            "file": (
                "typed.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    dataset_id = response.json()["id"]

    body = missing_profile(api, dataset_id, headers).json()
    assert body["total_rows"] == 4
    assert body["total_missing_values"] == 4
    for column in ("numeric", "string", "boolean", "datetime"):
        assert body["columns"][column]["missing_count"] == 1
        assert body["columns"][column]["missing_percentage"] == 25.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = missing_profile(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = missing_profile(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_missing_file_returns_404(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    missing = tmp_path / "gone.csv"
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="gone.csv",
        file_path=missing,
    )
    response = missing_profile(api, dataset.id, api.auth_headers(USERNAME))
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
    response = missing_profile(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400


def test_empty_dataset_is_rejected(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    empty = tmp_path / "empty.csv"
    empty.write_text("name,age\n", encoding="utf-8")
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="empty.csv",
        file_path=empty,
        rows=0,
        columns=2,
    )
    response = missing_profile(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
# ---------------------------------------------------------------------------
# Authentication / ownership
# ---------------------------------------------------------------------------


def test_unauthenticated_request_returns_401(api):
    response = api.client.get("/datasets/1/missing-values")
    assert response.status_code == 401


def test_cannot_profile_another_users_dataset(api, tmp_path):
    api.create_user(USERNAME, PASSWORD)
    bob = api.create_user("bob", PASSWORD)

    bob_file = tmp_path / "bobs.csv"
    bob_file.write_text("a,b\n1,2\n" * 3, encoding="utf-8")
    bob_dataset = api.create_dataset_record(
        owner_id=bob.id,
        original_filename="bobs.csv",
        file_path=bob_file,
        rows=3,
        columns=2,
    )

    response = missing_profile(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_profile_own_dataset(api):
    dataset_id = upload_csv(api, MISSING_CSV)
    assert missing_profile(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


# ---------------------------------------------------------------------------
# Regression guards: Part 2A and Part 2B-1 keep working
# ---------------------------------------------------------------------------


def test_parta_and_structure_endpoints_still_work(api):
    dataset_id = upload_csv(api, MISSING_CSV)
    headers = api.auth_headers(USERNAME)

    list_response = api.client.get("/datasets/", headers=headers)
    assert list_response.status_code == 200
    assert [d["id"] for d in list_response.json()] == [dataset_id]

    details_response = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details_response.status_code == 200

    structure_response = api.client.get(
        f"/datasets/{dataset_id}/structure", headers=headers
    )
    assert structure_response.status_code == 200
    assert structure_response.json()["rows"] == 4