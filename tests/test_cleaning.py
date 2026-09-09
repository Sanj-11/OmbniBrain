import io

import pandas as pd

from cleaning import clean_missing_values, detect_missing_values

USERNAME = "alice"
PASSWORD = "secret123"

# Age is an int column read with one NaN; the median of [20, 25, 30] is 25.
NUMERICAL_CSV = "Age\n20\n25\nNaN\n30\n"
# City is a string column; the mode ("Hyderabad") fills the missing value.
CATEGORICAL_CSV = "City\nHyderabad\nDelhi\nNaN\nHyderabad\n"
# "Empty" is fully missing across all rows and must be dropped.
EMPTY_COLUMN_CSV = (
    "Name,Age,Empty\n"
    "A,20,NaN\n"
    "B,25,NaN\n"
    "C,30,NaN\n"
)
# Mixed: age (1), salary (1) and city (1) are each missing exactly once.
MIXED_CSV = (
    "name,age,salary,city\n"
    "Alice,30,1100.5,NewYork\n"
    "Bob,,1500.75,Delhi\n"
    "Cara,32,,Pune\n"
    "Dan,40,2000.5,\n"
)
NO_MISSING_CSV = (
    "name,age\n"
    "Anna,28\n"
    "Bob,32\n"
    "Cara,41\n"
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


def clean(api, dataset_id, headers):
    return api.client.post(f"/datasets/{dataset_id}/clean/missing-values", headers=headers)


# ---------------------------------------------------------------------------
# Reusable service functions (proved on bare DataFrames, no server required)
# ---------------------------------------------------------------------------


def test_service_detects_missing_values_per_column():
    df = pd.read_csv(io.StringIO(MIXED_CSV))
    counts = detect_missing_values(df)
    assert counts["age"] == 1
    assert counts["salary"] == 1
    assert counts["city"] == 1
    assert counts["name"] == 0


def test_service_fills_numerical_with_median():
    df = pd.read_csv(io.StringIO(NUMERICAL_CSV))
    result = clean_missing_values(df)
    assert result.missing_values_before == 1
    assert result.missing_values_after == 0
    assert result.cleaning_applied is True
    # Median of [20, 25, 30] == 25.
    assert result.cleaned_df["Age"].iloc[2] == 25
    report = result.columns_processed["Age"]
    assert report.strategy == "median"
    assert report.values_filled == 1


def test_service_fills_categorical_with_mode():
    df = pd.read_csv(io.StringIO(CATEGORICAL_CSV))
    result = clean_missing_values(df)
    assert result.missing_values_before == 1
    assert result.missing_values_after == 0
    assert result.cleaned_df["City"].iloc[2] == "Hyderabad"
    report = result.columns_processed["City"]
    assert report.strategy == "mode"
    assert report.values_filled == 1


def test_service_drops_completely_empty_column():
    df = pd.read_csv(io.StringIO(EMPTY_COLUMN_CSV))
    result = clean_missing_values(df)
    assert result.removed_empty_columns == ["Empty"]
    assert "Empty" not in result.cleaned_df.columns
    assert result.missing_values_before == 3
    assert result.missing_values_after == 0
    assert "Empty" not in result.columns_processed
    assert list(result.cleaned_df.columns) == ["Name", "Age"]


def test_service_mixed_dataset_uses_appropriate_strategy():
    df = pd.read_csv(io.StringIO(MIXED_CSV))
    result = clean_missing_values(df)
    assert result.missing_values_after == 0
    assert result.columns_processed["age"].strategy == "median"
    assert result.columns_processed["salary"].strategy == "median"
    assert result.columns_processed["city"].strategy == "mode"


def test_service_no_missing_values_requires_no_cleaning():
    df = pd.read_csv(io.StringIO(NO_MISSING_CSV))
    result = clean_missing_values(df)
    assert result.missing_values_before == 0
    assert result.missing_values_after == 0
    assert result.cleaning_applied is False
    assert result.columns_processed == {}


def test_service_does_not_modify_the_original_dataframe():
    original = pd.read_csv(io.StringIO(NUMERICAL_CSV))
    snapshot = original.copy(deep=True)
    clean_missing_values(original)
    assert original.equals(snapshot)


def test_service_handles_dataset_with_only_numerical_columns():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
    result = clean_missing_values(df)
    assert result.missing_values_after == 0
    assert result.columns_processed["a"].strategy == "median"


def test_service_handles_dataset_with_only_categorical_columns():
    df = pd.DataFrame({"x": ["p", None, "q", "p"], "y": ["m", "m", "n", None]})
    result = clean_missing_values(df)
    assert result.missing_values_after == 0
    assert result.columns_processed["x"].strategy == "mode"
    assert result.cleaned_df["x"].isna().sum() == 0


def test_service_fills_categorical_without_a_dominant_mode():
    # A categorical column whose non-null values are all distinct still has a
    # mode in pandas terms (ties resolve to the first encountered value), so
    # cleaning must succeed and leave no missing values. The "Unknown" fill
    # used by :mod:`cleaning` is a defensive fallback for the (unusual) case
    # where a categorical column reports no usable mode at all.
    df = pd.DataFrame({"z": ["r", "s", "t", None]})
    result = clean_missing_values(df)
    assert result.missing_values_after == 0
    assert result.columns_processed["z"].strategy == "mode"
    assert result.cleaned_df["z"].isna().sum() == 0


# ---------------------------------------------------------------------------
# API endpoint - detection & cleaning report
# ---------------------------------------------------------------------------


def test_endpoint_returns_correct_report_for_missing_values(api):
    dataset_id = upload_csv(api, MIXED_CSV)
    response = clean(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["missing_values_before"] == 3
    assert body["missing_values_after"] == 0
    assert body["cleaning_applied"] is True
    assert body["columns_processed"]["age"]["strategy"] == "median"
    assert body["columns_processed"]["salary"]["strategy"] == "median"
    assert body["columns_processed"]["city"]["strategy"] == "mode"
    assert body["columns_processed"]["age"]["values_filled"] == 1


# ---------------------------------------------------------------------------
# Test 1 - No missing values
# ---------------------------------------------------------------------------


def test_endpoint_no_missing_values(api):
    dataset_id = upload_csv(api, NO_MISSING_CSV)
    response = clean(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["missing_values_before"] == 0
    assert body["missing_values_after"] == 0
    assert body["cleaning_applied"] is False
    assert body["columns_processed"] == {}
    assert body["removed_empty_columns"] == []


# ---------------------------------------------------------------------------
# Test 2 - Numerical missing values
# ---------------------------------------------------------------------------


def test_endpoint_numerical_missing_values(api):
    dataset_id = upload_csv(api, NUMERICAL_CSV)
    response = clean(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["missing_values_before"] == 1
    assert body["missing_values_after"] == 0
    assert body["columns_processed"]["Age"]["strategy"] == "median"
    assert body["columns_processed"]["Age"]["values_filled"] == 1


# ---------------------------------------------------------------------------
# Test 3 - Categorical missing values
# ---------------------------------------------------------------------------


def test_endpoint_categorical_missing_values(api):
    dataset_id = upload_csv(api, CATEGORICAL_CSV)
    response = clean(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["missing_values_before"] == 1
    assert body["missing_values_after"] == 0
    assert body["columns_processed"]["City"]["strategy"] == "mode"
    assert body["columns_processed"]["City"]["values_filled"] == 1


# ---------------------------------------------------------------------------
# Test 4 - Completely empty column
# ---------------------------------------------------------------------------


def test_endpoint_completely_empty_column_is_recorded(api):
    dataset_id = upload_csv(api, EMPTY_COLUMN_CSV)
    response = clean(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["removed_empty_columns"] == ["Empty"]
    assert body["missing_values_before"] == 3
    assert body["missing_values_after"] == 0
    assert "Empty" not in body["columns_processed"]


# ---------------------------------------------------------------------------
# Test 5 - Mixed dataset (no-missing, numerical, categorical combined)
# ---------------------------------------------------------------------------


def test_endpoint_mixed_dataset(api):
    content = (
        "name,age,salary,city,note\n"
        "Alice,30,1100.5,Hyderabad,ok\n"
        "Bob,,2000,Sales,\n"
        "Cara,32,,Pune,ok\n"
    )
    result = upload_csv(api, content)
    body = clean(api, result, api.auth_headers(USERNAME)).json()
    assert body["missing_values_before"] == 3
    assert body["missing_values_after"] == 0
    # Numerical -> median, categorical -> mode.
    assert body["columns_processed"]["age"]["strategy"] == "median"
    assert body["columns_processed"]["salary"]["strategy"] == "median"
    assert body["columns_processed"]["note"]["strategy"] == "mode"
    # Columns without missing values are not listed.
    assert "name" not in body["columns_processed"]
    assert "city" not in body["columns_processed"]


# ---------------------------------------------------------------------------
# The stored/original dataset is never modified
# ---------------------------------------------------------------------------


def test_original_dataset_remains_unchanged_by_cleaning(api):
    dataset_id = upload_csv(api, MIXED_CSV)
    headers = api.auth_headers(USERNAME)
    clean(api, dataset_id, headers)

    structure = api.client.get(f"/datasets/{dataset_id}/structure", headers=headers)
    assert structure.status_code == 200
    assert structure.json()["rows"] == 4
    assert structure.json()["columns"] == 4

    details = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details.status_code == 200
    assert details.json()["rows"] == 4
    assert details.json()["columns"] == 4


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_missing_file_returns_404(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    missing = tmp_path / "gone.csv"
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="gone.csv",
        file_path=missing,
    )
    response = clean(api, dataset.id, api.auth_headers(USERNAME))
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
    response = clean(api, dataset.id, api.auth_headers(USERNAME))
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
    response = clean(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Authentication / ownership
# ---------------------------------------------------------------------------


def test_unauthenticated_request_returns_401(api):
    response = api.client.post("/datasets/1/clean/missing-values")
    assert response.status_code == 401


def test_cannot_clean_another_users_dataset(api, tmp_path):
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

    response = clean(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_clean_own_dataset(api):
    dataset_id = upload_csv(api, NUMERICAL_CSV)
    assert clean(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


# ---------------------------------------------------------------------------
# Regression guards: Part 2A and Part 2B keep working
# ---------------------------------------------------------------------------


def test_parta_and_previous_profiling_endpoints_still_work(api):
    content = (
        "name,age,salary,active\n"
        "Anna,28,1100.5,True\n"
        "Bob,,950.25,False\n"
    )
    dataset_id = upload_csv(api, content)
    headers = api.auth_headers(USERNAME)

    list_response = api.client.get("/datasets/", headers=headers)
    assert list_response.status_code == 200

    details_response = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details_response.status_code == 200

    for path in ("structure", "missing-values", "quality", "summary"):
        resp = api.client.get(f"/datasets/{dataset_id}/{path}", headers=headers)
        assert resp.status_code == 200, path