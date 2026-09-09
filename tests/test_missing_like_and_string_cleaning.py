"""Tests for Part 2C-3B - Missing-like String Values & Whitespace Cleanup.

Covers the POST /datasets/{dataset_id}/clean/missing-like-strings endpoint and
the reusable :mod:`cleaning` service functions:

* detection and replacement of missing-like string values (NA, N/A, NULL, etc.),
* empty-string handling,
* whitespace-only string handling,
* leading/trailing whitespace cleanup,
* repeated internal whitespace normalization,
* preservation of legitimate values like "Unknown" and "Other",
* strict data preservation (rows, columns, values, dtypes),
* numerical/categorical column safety,
* error handling and the dataset ownership/authentication rules from Part 2A,
* regression guards for Part 2A, Part 2B and earlier Part 2C stages.
"""
import io

import pandas as pd

from cleaning import (
    MISSING_LIKE_VALUES,
    _clean_string_column,
    clean_missing_like_and_string_values,
)

USERNAME = "alice"
PASSWORD = "secret123"

# 5 rows / 3 columns with various missing-like and whitespace issues.
MIXED_STRING_CSV = (
    "city,customer_name,status\n"
    " Hyderabad ,Alice,NA\n"
    "Delhi ,,\n"
    "Mumbai ,Bob,  NULL  \n"
    "New   Delhi,Cara,none\n"
    " Chennai ,,Unknown\n"
)

# CSV that includes "Other" and "Unknown" which must be preserved.
LEGITIMATE_VALUES_CSV = (
    "category,description\n"
    "A,Unknown\n"
    "B,Other\n"
    "C,N/A\n"
    "D, normal text \n"
)

# Numeric columns must be left untouched.
NUMERIC_CSV = (
    "id,age,score\n"
    "1, NA ,10.5\n"
    "2,25,20.0\n"
    "3,   ,30.5\n"
)

# Empty dataset must be rejected.
EMPTY_CSV = "city,name\n"


def upload_csv(api, content, filename="sample.csv"):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)
    response = api.client.post(
        "/datasets/upload",
        files={"file": (filename, content.encode("utf-8"), "text/csv")},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def clean_strings(api, dataset_id, headers):
    return api.client.post(
        f"/datasets/{dataset_id}/clean/missing-like-strings", headers=headers
    )


# ---------------------------------------------------------------------------
# Service: missing-like detection & replacement
# ---------------------------------------------------------------------------


def test_service_detects_na():
    df = pd.DataFrame({"city": ["Hyderabad", "NA", "Delhi"]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 1
    assert result.missing_like_values_handled == 1
    assert pd.isna(result.cleaned_df["city"].iloc[1])


def test_service_detects_na_slash():
    df = pd.DataFrame({"city": ["Hyderabad", "N/A", "Delhi"]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 1
    assert pd.isna(result.cleaned_df["city"].iloc[1])


def test_service_detects_null():
    df = pd.DataFrame({"city": ["Hyderabad", "NULL", "Delhi"]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 1
    assert pd.isna(result.cleaned_df["city"].iloc[1])


def test_service_detects_empty_string():
    df = pd.DataFrame({"city": ["Hyderabad", "", "Delhi"]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 1
    assert pd.isna(result.cleaned_df["city"].iloc[1])


def test_service_detects_whitespace_only_string():
    df = pd.DataFrame({"city": ["Hyderabad", "   ", "Delhi"]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 1
    assert pd.isna(result.cleaned_df["city"].iloc[1])


def test_service_detects_lowercase_variants():
    df = pd.DataFrame({"city": ["Hyderabad", "na", "n/a", "null", "none", "nan"]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 5
    assert result.missing_like_values_handled == 5


def test_service_preserves_unknown_and_other():
    df = pd.DataFrame(
        {"category": ["Unknown", "Other", "N/A", " normal "], "value": [1, 2, 3, 4]}
    )
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 1
    assert result.cleaned_df["category"].iloc[0] == "Unknown"
    assert result.cleaned_df["category"].iloc[1] == "Other"
    assert pd.isna(result.cleaned_df["category"].iloc[2])
    assert result.cleaned_df["category"].iloc[3] == "normal"


# ---------------------------------------------------------------------------
# Service: whitespace cleanup
# ---------------------------------------------------------------------------


def test_service_strips_leading_trailing_whitespace():
    df = pd.DataFrame({"city": [" Hyderabad ", " Delhi", "Mumbai "]})
    result = clean_missing_like_and_string_values(df)
    assert result.whitespace_values_cleaned == 3
    assert result.cleaned_df["city"].iloc[0] == "Hyderabad"
    assert result.cleaned_df["city"].iloc[1] == "Delhi"
    assert result.cleaned_df["city"].iloc[2] == "Mumbai"


def test_service_collapses_repeated_internal_whitespace():
    df = pd.DataFrame({"city": ["New   Delhi", "Bangalore", "Pune"]})
    result = clean_missing_like_and_string_values(df)
    assert result.whitespace_values_cleaned == 1
    assert result.cleaned_df["city"].iloc[0] == "New Delhi"
    assert result.cleaned_df["city"].iloc[1] == "Bangalore"
    assert result.cleaned_df["city"].iloc[2] == "Pune"


def test_service_combined_whitespace_and_missing():
    df = pd.DataFrame({"city": [" Hyderabad ", "N/A", "New   Delhi", "   ", "Delhi"]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 2
    assert result.missing_like_values_handled == 2
    assert result.whitespace_values_cleaned == 2
    assert result.cleaned_df["city"].iloc[0] == "Hyderabad"
    assert pd.isna(result.cleaned_df["city"].iloc[1])
    assert result.cleaned_df["city"].iloc[2] == "New Delhi"
    assert pd.isna(result.cleaned_df["city"].iloc[3])
    assert result.cleaned_df["city"].iloc[4] == "Delhi"


# ---------------------------------------------------------------------------
# Service: data preservation
# ---------------------------------------------------------------------------


def test_service_preserves_rows_and_columns():
    df = pd.DataFrame({"city": ["A", "B"], "name": ["X", "Y"], "age": [1, 2]})
    result = clean_missing_like_and_string_values(df)
    assert len(result.cleaned_df) == 2
    assert list(result.cleaned_df.columns) == ["city", "name", "age"]


def test_service_does_not_modify_original_dataframe():
    original = pd.DataFrame({"city": ["A", "N/A"], "age": [1, 2]})
    snapshot = original.copy(deep=True)
    clean_missing_like_and_string_values(original)
    assert original.equals(snapshot)


def test_service_numerical_columns_are_untouched():
    df = pd.DataFrame({"id": [1, 2, 3], "score": [10.5, 20.0, 30.5]})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 0
    assert result.whitespace_values_cleaned == 0
    assert result.cleaned_df["id"].tolist() == [1, 2, 3]
    assert result.cleaned_df["score"].tolist() == [10.5, 20.0, 30.5]


def test_service_categorical_columns_are_untouched():
    df = pd.DataFrame({"cat": pd.Categorical(["A", "B", "C"])})
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 0
    assert result.cleaned_df["cat"].dtype.name == "category"


def test_service_datetime_columns_are_untouched():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])}
    )
    result = clean_missing_like_and_string_values(df)
    assert result.missing_like_values_detected == 0
    assert pd.api.types.is_datetime64_any_dtype(result.cleaned_df["date"])


def test_service_mixed_dataset_reports_correct_counts():
    df = pd.read_csv(io.StringIO(MIXED_STRING_CSV))
    result = clean_missing_like_and_string_values(df)
    # After pandas CSV parsing: NA, empty cells, NULL become NaN.
    # Remaining strings: "  NULL  " (status row 2), "none" (status row 3), "Unknown" (status row 4).
    # "  NULL  " -> strip -> "NULL" -> missing-like
    # "none" -> missing-like
    # "Unknown" -> legitimate
    # City column has 5 values with leading/trailing or repeated internal whitespace.
    # Total missing-like: 2
    # Total whitespace: 5
    assert result.missing_like_values_detected == 2
    assert result.missing_like_values_handled == 2
    assert result.whitespace_values_cleaned == 5
    assert set(result.columns_affected) == {"city", "status"}


def test_service_does_not_delete_rows_or_columns():
    df = pd.DataFrame({"city": ["A", "N/A", "B"], "name": ["X", "Y", "Z"]})
    result = clean_missing_like_and_string_values(df)
    assert len(result.cleaned_df) == 3
    assert list(result.cleaned_df.columns) == ["city", "name"]


# ---------------------------------------------------------------------------
# Service: column-level helper
# ---------------------------------------------------------------------------


def test_service_clean_string_column_empty_series():
    series = pd.Series([], dtype=object)
    cleaned, missing, whitespace = _clean_string_column(series)
    assert len(cleaned) == 0
    assert missing == 0
    assert whitespace == 0


def test_service_clean_string_column_non_string_values():
    series = pd.Series([1, 2.5, True, None], dtype=object)
    cleaned, missing, whitespace = _clean_string_column(series)
    assert missing == 0
    assert whitespace == 0
    assert cleaned.tolist() == [1, 2.5, True, None]


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


def test_endpoint_returns_correct_report(api):
    dataset_id = upload_csv(api, MIXED_STRING_CSV)
    response = clean_strings(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text

    body = response.json()
    # After pandas CSV parsing: NA, empty cells, NULL become NaN.
    # Remaining string missing-like: "  NULL  " -> "NULL", "none".
    # Whitespace cleaned: all 5 city values.
    assert body["missing_like_values_detected"] == 2
    assert body["missing_like_values_handled"] == 2
    assert body["whitespace_values_cleaned"] == 5
    assert set(body["columns_affected"]) == {"city", "status"}


def test_endpoint_preserves_legitimate_values(api):
    dataset_id = upload_csv(api, LEGITIMATE_VALUES_CSV)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    # "N/A" is converted to NaN by pandas during upload.
    # "Unknown" and "Other" are preserved as legitimate values.
    assert body["missing_like_values_detected"] == 0
    assert body["missing_like_values_handled"] == 0


def test_endpoint_does_not_affect_numeric_columns(api):
    dataset_id = upload_csv(api, NUMERIC_CSV)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert body["missing_like_values_detected"] == 2
    assert body["columns_affected"] == ["age"]


# ---------------------------------------------------------------------------
# Test 1 - No missing-like or whitespace issues
# ---------------------------------------------------------------------------


def test_endpoint_no_issues(api):
    content = "city,name\nHyderabad,Alice\nDelhi,Bob\n"
    dataset_id = upload_csv(api, content)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert body["missing_like_values_detected"] == 0
    assert body["missing_like_values_handled"] == 0
    assert body["whitespace_values_cleaned"] == 0
    assert body["columns_affected"] == []


# ---------------------------------------------------------------------------
# Test 2 - All missing-like variants
# ---------------------------------------------------------------------------


def test_endpoint_all_missing_like_variants(api):
    content = (
        "col\n"
        "NA\n"
        "N/A\n"
        "na\n"
        "n/a\n"
        "NULL\n"
        "null\n"
        "None\n"
        "none\n"
        "NaN\n"
        "nan\n"
        "\n"
        "   \n"
        "valid\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    # Pandas converts most missing-like strings to NaN during upload.
    # Remaining strings that our function sees: "na", "none", "valid".
    # "na" and "none" are missing-like. "valid" has no whitespace issues.
    assert body["missing_like_values_detected"] == 2
    assert body["missing_like_values_handled"] == 2
    assert body["whitespace_values_cleaned"] == 0


# ---------------------------------------------------------------------------
# Test 3 - Whitespace only
# ---------------------------------------------------------------------------


def test_endpoint_whitespace_only(api):
    content = "col\nvalid\n   \n\t\t\nnormal\n"
    dataset_id = upload_csv(api, content)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    # Pandas converts whitespace-only cells to NaN during upload.
    assert body["missing_like_values_detected"] == 0
    assert body["missing_like_values_handled"] == 0
    assert body["whitespace_values_cleaned"] == 0


# ---------------------------------------------------------------------------
# Test 4 - Leading/trailing and internal whitespace
# ---------------------------------------------------------------------------


def test_endpoint_whitespace_cleanup(api):
    content = (
        "city,name\n"
        " Hyderabad ,Alice\n"
        "Delhi ,Bob\n"
        "Mumbai ,Cara\n"
        "New   Delhi,Dan\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    assert body["missing_like_values_detected"] == 0
    assert body["missing_like_values_handled"] == 0
    assert body["whitespace_values_cleaned"] == 4
    # Only city has whitespace issues; name values are clean.
    assert body["columns_affected"] == ["city"]


# ---------------------------------------------------------------------------
# Test 5 - Legitimate Unknown and Other
# ---------------------------------------------------------------------------


def test_endpoint_preserves_unknown_and_other(api):
    content = "category,description\nA,Unknown\nB,Other\nC,N/A\n"
    dataset_id = upload_csv(api, content)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    # "N/A" is converted to NaN by pandas during upload.
    # "Unknown" and "Other" are legitimate values and must be preserved.
    assert body["missing_like_values_detected"] == 0
    assert body["missing_like_values_handled"] == 0


# ---------------------------------------------------------------------------
# Test 6 - Mixed dataset
# ---------------------------------------------------------------------------


def test_endpoint_mixed_dataset(api):
    content = (
        "name,age,city,status\n"
        "Alice,30, Hyderabad ,active\n"
        "Bob,,Delhi,NA\n"
        "Cara,32,New   Delhi,  NULL  \n"
        "Dan,40, Chennai ,none\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean_strings(api, dataset_id, api.auth_headers(USERNAME)).json()
    # After pandas upload: "NA" and empty cell become NaN.
    # Remaining string missing-like: "  NULL  " -> "NULL", "none".
    # City whitespace: " Hyderabad ", "New   Delhi", " Chennai " = 3 values.
    # "Delhi" has no whitespace issue.
    assert body["missing_like_values_detected"] == 2
    assert body["missing_like_values_handled"] == 2
    assert body["whitespace_values_cleaned"] == 3
    assert set(body["columns_affected"]) == {"city", "status"}


# ---------------------------------------------------------------------------
# The original dataset is never modified
# ---------------------------------------------------------------------------


def test_endpoint_original_dataset_remains_unchanged(api):
    content = "city,name\nA,1\nB,2\n"
    dataset_id = upload_csv(api, content)
    headers = api.auth_headers(USERNAME)
    clean_strings(api, dataset_id, headers)

    structure = api.client.get(f"/datasets/{dataset_id}/structure", headers=headers)
    assert structure.status_code == 200
    assert structure.json()["rows"] == 2
    assert structure.json()["columns"] == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean_strings(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean_strings(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_missing_file_returns_404(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    missing = tmp_path / "gone.csv"
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="gone.csv",
        file_path=missing,
    )
    response = clean_strings(api, dataset.id, api.auth_headers(USERNAME))
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
    response = clean_strings(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400


def test_empty_dataset_is_rejected(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    empty = tmp_path / "empty.csv"
    empty.write_text("city,name\n", encoding="utf-8")
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="empty.csv",
        file_path=empty,
        rows=0,
        columns=2,
    )
    response = clean_strings(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Authentication / ownership
# ---------------------------------------------------------------------------


def test_unauthenticated_request_returns_401(api):
    response = api.client.post("/datasets/1/clean/missing-like-strings")
    assert response.status_code == 401


def test_cannot_clean_another_users_dataset(api, tmp_path):
    api.create_user(USERNAME, PASSWORD)
    bob = api.create_user("bob", PASSWORD)

    bob_file = tmp_path / "bobs.csv"
    bob_file.write_text("city,name\nA,1\n", encoding="utf-8")
    bob_dataset = api.create_dataset_record(
        owner_id=bob.id,
        original_filename="bobs.csv",
        file_path=bob_file,
        rows=1,
        columns=2,
    )

    response = clean_strings(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_clean_own_dataset(api):
    dataset_id = upload_csv(api, "city,name\nA,1\n")
    assert clean_strings(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


# ---------------------------------------------------------------------------
# Regression guards: Part 2A, Part 2B and earlier Part 2C stages keep working
# ---------------------------------------------------------------------------


def test_previous_stages_still_work(api):
    content = (
        "name,age,salary,active\n"
        "Anna,28,1100.5,True\n"
        "Bob,,950.25,False\n"
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

    assert (
        api.client.post(
            f"/datasets/{dataset_id}/clean/missing-values", headers=headers
        ).status_code
        == 200
    )
    assert (
        api.client.post(
            f"/datasets/{dataset_id}/clean/duplicates-and-types", headers=headers
        ).status_code
        == 200
    )
    assert (
        api.client.post(
            f"/datasets/{dataset_id}/clean/standardize-columns", headers=headers
        ).status_code
        == 200
    )
    # The new endpoint is itself reachable in the same run.
    assert clean_strings(api, dataset_id, headers).status_code == 200