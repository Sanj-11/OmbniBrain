import datetime as dt
import io

import pandas as pd

USERNAME = "alice"
PASSWORD = "secret123"


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


def quality(api, dataset_id, headers):
    return api.client.get(f"/datasets/{dataset_id}/quality", headers=headers)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_dataset_with_one_duplicate_row(api):
    # A,10,Sales appears twice: the second occurrence is a duplicate of the
    # first, so exactly one duplicate row is expected.
    content = (
        "name,age,department\n"
        "A,10,Sales\n"
        "B,20,HR\n"
        "A,10,Sales\n"
        "C,30,IT\n"
    )
    dataset_id = upload_csv(api, content)
    body = quality(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert body["duplicate_rows"] == 1


def test_dataset_with_no_duplicates(api):
    content = (
        "name,age,department\n"
        "A,10,Sales\n"
        "B,20,HR\n"
        "C,30,IT\n"
    )
    dataset_id = upload_csv(api, content)
    response = quality(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200
    assert response.json()["duplicate_rows"] == 0


def test_multiple_duplicate_occurrences_follow_pandas_semantics(api):
    # Three identical rows: pandas marks the 2nd and 3rd as duplicates, so the
    # first occurrence must NOT be counted -> duplicate_rows == 2.
    content = (
        "name,age,department\n"
        "A,10,Sales\n"
        "A,10,Sales\n"
        "A,10,Sales\n"
    )
    dataset_id = upload_csv(api, content)
    assert quality(api, dataset_id, api.auth_headers(USERNAME)).json()[
        "duplicate_rows"
    ] == 2


def test_duplicate_rows_are_not_removed_from_dataset(api):
    # The quality endpoint is read-only: the underlying dataset must keep its
    # duplicate rows and the structure endpoint must still report all of them.
    content = (
        "name,age\n"
        "A,10\n"
        "A,10\n"
        "B,20\n"
    )
    dataset_id = upload_csv(api, content)
    headers = api.auth_headers(USERNAME)

    assert quality(api, dataset_id, headers).json()["duplicate_rows"] == 1

    structure = api.client.get(f"/datasets/{dataset_id}/structure", headers=headers)
    assert structure.status_code == 200
    assert structure.json()["rows"] == 3

    details = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details.status_code == 200
    assert details.json()["rows"] == 3


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------


def test_integer_and_float_columns_are_numerical(api):
    content = "name,age,salary\nAnna,28,1100.5\nBob,32,950.25\n"
    dataset_id = upload_csv(api, content)
    body = quality(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert "age" in body["numerical_columns"]     # int64
    assert "salary" in body["numerical_columns"]  # float64
    assert "name" not in body["numerical_columns"]


def test_string_object_and_category_columns_are_categorical(api):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)

    frame = pd.DataFrame(
        {
            "name": ["a", "b", "c"],                  # object/string
            "role": pd.Categorical(["x", "y", "x"]),  # category
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

    classification = quality(api, dataset_id, headers).json()
    assert set(classification["categorical_columns"]) == {"name", "role"}
    assert classification["numerical_columns"] == []


def test_mixed_dataset_classification(api):
    """int, float, string, category, bool and datetime are all classified."""
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)

    frame = pd.DataFrame(
        {
            "age": [28, 32, 41],                      # int64      -> numerical
            "score": [1.5, 2.5, 3.5],                 # float64    -> numerical
            "name": ["a", "b", "c"],                  # object     -> categorical
            "role": pd.Categorical(["x", "y", "x"]),  # category   -> categorical
            "active": [True, False, True],            # bool       -> categorical
            "joined": [                                # datetime64 -> datetime
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 2, 1),
                dt.datetime(2024, 3, 1),
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
                "mixed.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    dataset_id = response.json()["id"]

    classification = quality(api, dataset_id, headers).json()
    assert set(classification["numerical_columns"]) == {"age", "score"}
    assert set(classification["categorical_columns"]) == {"name", "role", "active"}
    assert classification["datetime_columns"] == ["joined"]


def test_response_contains_all_required_fields(api):
    content = "name,age,department\nA,10,Sales\nB,20,HR\n"
    dataset_id = upload_csv(api, content)
    body = quality(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert {"duplicate_rows", "numerical_columns", "categorical_columns"} <= set(
        body.keys()
    )
    assert "datetime_columns" in body
    assert body["duplicate_rows"] == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = quality(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = quality(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_missing_file_returns_404(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    missing = tmp_path / "gone.csv"
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="gone.csv",
        file_path=missing,
    )
    response = quality(api, dataset.id, api.auth_headers(USERNAME))
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
    response = quality(api, dataset.id, api.auth_headers(USERNAME))
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
    response = quality(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Authentication / ownership
# ---------------------------------------------------------------------------


def test_unauthenticated_request_returns_401(api):
    response = api.client.get("/datasets/1/quality")
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

    response = quality(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_profile_own_dataset(api):
    content = "name,age\nA,10\nA,10\n"
    dataset_id = upload_csv(api, content)
    assert quality(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


# ---------------------------------------------------------------------------
# Regression guards: Part 2A, Part 2B-1 and Part 2B-2 keep working
# ---------------------------------------------------------------------------


def test_parta_and_previous_profiling_endpoints_still_work(api):
    content = (
        "name,age,salary,active\n"
        "Anna,28,1100.5,True\n"
        "Bob,32,950.25,False\n"
        "Bob,32,950.25,False\n"
    )
    dataset_id = upload_csv(api, content)
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
    assert structure_response.json()["rows"] == 3

    missing_response = api.client.get(
        f"/datasets/{dataset_id}/missing-values", headers=headers
    )
    assert missing_response.status_code == 200