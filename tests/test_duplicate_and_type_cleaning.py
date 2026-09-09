import io
import json

import pandas as pd
import pytest

from cleaning import (
    ACTION_CONVERTED_NUMERIC,
    ACTION_SKIPPED,
    ACTION_UNCHANGED,
    clean_duplicates_and_types,
    column_dtype_category,
    correct_column_dtypes,
    detect_duplicate_rows,
    remove_duplicate_rows,
)

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


def clean2(api, dataset_id, headers):
    return api.client.post(
        f"/datasets/{dataset_id}/clean/duplicates-and-types", headers=headers
    )


# ---------------------------------------------------------------------------
# Test 1 - No duplicates
# ---------------------------------------------------------------------------


def test_service_reports_no_duplicates():
    df = pd.DataFrame({"ID": [1, 2, 3], "Name": ["A", "B", "C"]})
    stats = detect_duplicate_rows(df)
    assert stats["rows_before"] == 3
    assert stats["duplicate_rows_found"] == 0
    assert stats["unique_rows"] == 3
    assert stats["rows_after"] == 3
    assert stats["duplicate_percentage"] == 0.0

    result = clean_duplicates_and_types(df)
    assert result.duplicates_cleaned is False
    assert result.duplicate_rows_removed == 0
    assert result.rows_after == result.rows_before
    assert len(result.cleaned_df) == 3


# ---------------------------------------------------------------------------
# Test 2 - Duplicate rows
# ---------------------------------------------------------------------------


def test_service_removes_duplicate_rows_keeping_first_occurrence():
    df = pd.DataFrame(
        {"ID": [1, 2, 2, 3], "Name": ["A", "B", "B", "C"], "Age": [20, 25, 25, 30]}
    )
    result = clean_duplicates_and_types(df)

    assert result.rows_before == 4
    assert result.duplicate_rows_found == 1
    assert result.duplicate_rows_removed == 1
    assert result.rows_after == 3
    assert result.duplicate_percentage == 25.0
    assert result.duplicates_cleaned is True

    assert list(result.cleaned_df.columns) == ["ID", "Name", "Age"]
    assert result.cleaned_df.to_dict("records") == [
        {"ID": 1, "Name": "A", "Age": 20},
        {"ID": 2, "Name": "B", "Age": 25},
        {"ID": 3, "Name": "C", "Age": 30},
    ]
    # The index is reset to a clean 0..n-1 range after removal.
    assert result.cleaned_df.index.tolist() == [0, 1, 2]


def test_service_duplicate_percentage_is_computed():
    df = pd.DataFrame({"x": [1, 1, 1, 2]})
    stats = detect_duplicate_rows(df)
    assert stats["duplicate_rows_found"] == 2
    assert stats["duplicate_percentage"] == 50.0


def test_service_all_duplicate_rows_reduce_to_one():
    df = pd.DataFrame({"Name": ["A", "A", "A"], "Age": [10, 10, 10]})
    result = clean_duplicates_and_types(df)
    assert result.rows_before == 3
    assert result.duplicate_rows_found == 2
    assert result.rows_after == 1
    assert len(result.cleaned_df) == 1


def test_service_single_row_dataset():
    df = pd.DataFrame({"Age": pd.Series(["20"], dtype=object)})
    result = clean_duplicates_and_types(df)
    assert result.duplicate_rows_found == 0
    assert result.rows_after == 1
    assert result.columns_processed["Age"].action == ACTION_CONVERTED_NUMERIC


def test_service_does_not_modify_the_original_dataframe():
    original = pd.DataFrame(
        {"ID": [1, 2, 2], "Amount": pd.Series(["10", "20", "20"], dtype=object)}
    )
    snapshot = original.copy(deep=True)
    clean_duplicates_and_types(original)
    assert original.equals(snapshot)
# ---------------------------------------------------------------------------
# Test 3 - Numeric values stored as strings
# ---------------------------------------------------------------------------


def test_service_converts_numeric_strings_to_numeric():
    df = pd.DataFrame({"Age": pd.Series(["20", "25", "30"], dtype=object)})
    result = clean_duplicates_and_types(df)

    report = result.columns_processed["Age"]
    assert report.original_dtype == "object"
    assert report.action == ACTION_CONVERTED_NUMERIC
    assert pd.api.types.is_numeric_dtype(result.cleaned_df["Age"])
    assert pd.api.types.is_integer_dtype(result.cleaned_df["Age"])
    assert result.dtype_changes == 1
    assert result.conversions_skipped == []


def test_service_converts_decimal_strings_to_numeric():
    df = pd.DataFrame(
        {"Amount": pd.Series(["10.5", "20.75", "30.0"], dtype=object)}
    )
    result = clean_duplicates_and_types(df)
    assert result.columns_processed["Amount"].action == ACTION_CONVERTED_NUMERIC
    assert pd.api.types.is_float_dtype(result.cleaned_df["Amount"])


def test_service_remove_duplicate_rows_resets_index():
    df = pd.DataFrame({"x": [1, 1, 2]})
    cleaned = remove_duplicate_rows(df)
    assert cleaned.index.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Test 4 - Mixed numeric/text values: conversion must be skipped safely
# ---------------------------------------------------------------------------


def test_service_mixed_numeric_text_skips_conversion():
    df = pd.DataFrame(
        {"Amount": pd.Series(["100", "250", "Unknown", "300"], dtype=object)}
    )
    result = clean_duplicates_and_types(df)

    assert "Amount" in result.conversions_skipped
    report = result.columns_processed["Amount"]
    assert report.action == ACTION_SKIPPED
    assert report.new_dtype == "object"
    # "Unknown" is preserved: it is not silently destroyed.
    assert "Unknown" in result.cleaned_df["Amount"].values
    assert "100" in result.cleaned_df["Amount"].values
    # The skipped column still keeps the original representation.
    assert result.dtype_changes == 0


def test_service_conversion_threshold_boundary():
    # 3 out of 4 values numeric = 75% < 80% threshold -> skip.
    skipped = pd.DataFrame(
        {"v": pd.Series(["1", "2", "3", "x"], dtype=object)}
    )
    result = clean_duplicates_and_types(skipped)
    assert result.conversions_skipped == ["v"]

    # 5 out of 6 values numeric = 83.3% >= 80% threshold -> convert.
    converted = pd.DataFrame(
        {"v": pd.Series(["1", "2", "3", "4", "5", "x"], dtype=object)}
    )
    result = clean_duplicates_and_types(converted)
    assert result.conversions_skipped == []
    assert result.columns_processed["v"].action == ACTION_CONVERTED_NUMERIC


# ---------------------------------------------------------------------------
# Test 5 - Date column
# ---------------------------------------------------------------------------


def test_service_converts_date_strings_to_datetime():
    df = pd.DataFrame(
        {"Date": pd.Series(["2026-01-10", "2026-02-15", "2026-03-20"], dtype=object)}
    )
    result = clean_duplicates_and_types(df)

    report = result.columns_processed["Date"]
    assert report.original_dtype == "object"
    assert report.action == "converted_to_datetime"
    assert pd.api.types.is_datetime64_any_dtype(result.cleaned_df["Date"])
    assert result.dtype_changes == 1


# ---------------------------------------------------------------------------
# Test 6 - Invalid date values
# ---------------------------------------------------------------------------


def test_service_invalid_date_values_skip_conversion():
    df = pd.DataFrame(
        {"Date": pd.Series(["2026-01-10", "Invalid", "2026-03-20"], dtype=object)}
    )
    result = clean_duplicates_and_types(df)

    assert "Date" in result.conversions_skipped
    report = result.columns_processed["Date"]
    assert report.action == ACTION_SKIPPED
    assert report.new_dtype == "object"
    # The invalid value must not be silently destroyed.
    assert "Invalid" in result.cleaned_df["Date"].values


# ---------------------------------------------------------------------------
# Test 7 - Normal text column
# ---------------------------------------------------------------------------


def test_service_leaves_normal_text_column_unchanged():
    df = pd.DataFrame(
        {"Name": pd.Series(["Anusha", "Rahul", "Priya"], dtype=object)}
    )
    result = clean_duplicates_and_types(df)

    report = result.columns_processed["Name"]
    assert report.action == ACTION_UNCHANGED
    assert report.new_dtype == "object"
    assert "Name" not in result.conversions_skipped
    assert result.dtype_changes == 0
# ---------------------------------------------------------------------------
# Boolean / already-correct dtypes / dtype categories
# ---------------------------------------------------------------------------


def test_service_converts_boolean_like_columns():
    df = pd.DataFrame(
        {"flag": pd.Series(["True", "False", "True"], dtype=object)}
    )
    result = clean_duplicates_and_types(df)
    assert result.columns_processed["flag"].action == "converted_to_boolean"
    assert "boolean" in str(result.cleaned_df["flag"].dtype)


def test_service_constant_boolean_column_is_not_converted():
    # Only one variant present -> ambiguous (could be a label column), so it is
    # reported as a skipped conversion and the original values are preserved.
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "flag": pd.Series(["True", "True", "True"], dtype=object),
        }
    )
    result = clean_duplicates_and_types(df)
    assert result.columns_processed["flag"].action == ACTION_SKIPPED
    assert "flag" in result.conversions_skipped
    assert result.cleaned_df["flag"].tolist() == ["True", "True", "True"]


def test_service_already_numeric_columns_untouched():
    df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.5, 2.5, 3.5]})
    result = clean_duplicates_and_types(df)
    assert result.columns_processed["int_col"].action == ACTION_UNCHANGED
    assert result.columns_processed["float_col"].action == ACTION_UNCHANGED
    assert result.dtype_changes == 0


def test_service_column_dtype_categories():
    frame = pd.DataFrame(
        {
            "i": [1, 2],
            "f": [1.0, 2.0],
            "b": [True, False],
            "s": ["x", "y"],
            "d": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "c": pd.Categorical(["x", "y"]),
        }
    )
    assert column_dtype_category(frame["i"].dtype) == "integer"
    assert column_dtype_category(frame["f"].dtype) == "float"
    assert column_dtype_category(frame["b"].dtype) == "boolean"
    assert column_dtype_category(frame["s"].dtype) == "object"
    assert column_dtype_category(frame["d"].dtype) == "datetime"
    assert column_dtype_category(frame["c"].dtype) == "categorical"


def test_service_correct_column_dtypes_reports_every_column():
    df = pd.DataFrame(
        {
            "Name": pd.Series(["A", "B"], dtype=object),
            "Age": pd.Series(["20", "30"], dtype=object),
        }
    )
    _, columns_processed, dtype_changes, conversions_skipped = correct_column_dtypes(df)
    assert set(columns_processed) == {"Name", "Age"}
    assert dtype_changes == 1
    assert conversions_skipped == []
    assert columns_processed["Age"].action == ACTION_CONVERTED_NUMERIC
    assert columns_processed["Name"].action == ACTION_UNCHANGED
# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


def test_endpoint_no_duplicates(api):
    content = "ID,Name\n1,A\n2,B\n3,C\n"
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["rows_before"] == 3
    assert body["duplicate_rows_found"] == 0
    assert body["duplicate_rows_removed"] == 0
    assert body["rows_after"] == 3
    assert body["duplicates_cleaned"] is False
    assert body["columns_processed"]["Name"]["action"] == ACTION_UNCHANGED


def test_endpoint_removes_duplicates(api):
    content = "ID,Name,Age\n1,A,20\n2,B,25\n2,B,25\n3,C,30\n"
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["rows_before"] == 4
    assert body["duplicate_rows_found"] == 1
    assert body["duplicate_rows_removed"] == 1
    assert body["rows_after"] == 3
    assert body["duplicate_percentage"] == 25.0
    assert body["duplicates_cleaned"] is True


def test_endpoint_all_duplicate_rows(api):
    content = "name,age\nA,10\nA,10\nA,10\n"
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["duplicate_rows_found"] == 2
    assert body["rows_after"] == 1
    assert body["duplicate_percentage"] == pytest.approx(66.67)


def test_endpoint_converts_numeric_strings_in_object_column(api):
    # A truly object numeric column, because one token is not numeric.
    content = (
        "id,label,score\n"
        "1,A,10\n"
        "1,A,10\n"
        "2,B,20\n"
        "3,C,30\n"
        "4,D,50\n"
        "5,E,65\n"
        "6,F,unknown\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["rows_before"] == 7
    assert body["duplicate_rows_found"] == 1
    score = body["columns_processed"]["score"]
    assert score["original_dtype"] == "object"
    assert score["action"] == ACTION_CONVERTED_NUMERIC
    assert score["new_dtype"] in ("int64", "float64")
def test_endpoint_converts_date_column(api):
    content = (
        "name,joined\n"
        "Anusha,2026-01-10\n"
        "Rahul,2026-02-15\n"
        "Priya,2026-03-20\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    joined = body["columns_processed"]["joined"]
    assert joined["original_dtype"] == "object"
    assert joined["action"] == "converted_to_datetime"
    assert joined["new_dtype"] == "datetime64[ns]"
    assert body["columns_processed"]["name"]["action"] == ACTION_UNCHANGED


def test_endpoint_invalid_dates_are_skipped(api):
    content = (
        "name,joined\n"
        "Anusha,2026-01-10\n"
        "Rahul,Invalid\n"
        "Priya,2026-03-20\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert "joined" in body["conversions_skipped"]
    assert body["columns_processed"]["joined"]["action"] == ACTION_SKIPPED


def test_endpoint_text_column_unchanged(api):
    content = "name,city\nAnusha,Hyderabad\nRahul,Delhi\nPriya,Pune\n"
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["columns_processed"]["name"]["action"] == ACTION_UNCHANGED
    assert body["columns_processed"]["city"]["action"] == ACTION_UNCHANGED
    assert body["conversions_skipped"] == []
    assert body["dtype_changes"] == 0


def test_endpoint_report_has_all_required_fields(api):
    content = "ID,Name\n1,A\n1,A\n"
    dataset_id = upload_csv(api, content)
    response = clean2(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text

    payload = response.json()
    assert {
        "rows_before",
        "duplicate_rows_found",
        "duplicate_rows_removed",
        "rows_after",
        "columns_processed",
        "dtype_changes",
        "conversions_skipped",
    } <= set(payload.keys())
    # Every column appears in columns_processed (untouched ones included).
    assert set(payload["columns_processed"]) == {"ID", "Name"}
    # Ready for JSON serialisation (no raw numpy/pandas objects remain).
    json.dumps(payload)


def test_endpoint_chains_missing_value_cleaning(api):
    # The Part 2C-1 missing-value stage runs first; this endpoint must accept
    # datasets that had missing values without re-implementing that logic.
    content = (
        "name,age,salary\n"
        "Alice,30,1100.5\n"
        "Bob,,1500.75\n"
        "Bob,,1500.75\n"
        "Cara,32,2000.5\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["rows_before"] == 4
    assert body["duplicate_rows_found"] == 1
    assert body["rows_after"] == 3
    assert body["columns_processed"]["age"]["action"] == ACTION_UNCHANGED


def test_endpoint_original_dataset_remains_unchanged(api):
    content = "ID,Name\n1,A\n1,A\n2,B\n"
    dataset_id = upload_csv(api, content)
    headers = api.auth_headers(USERNAME)
    clean2(api, dataset_id, headers)

    structure = api.client.get(f"/datasets/{dataset_id}/structure", headers=headers)
    assert structure.status_code == 200
    assert structure.json()["rows"] == 3

    details = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details.status_code == 200
    assert details.json()["rows"] == 3


def test_endpoint_mixed_numeric_text_is_skipped(api):
    content = "name,amount\nA,100\nB,250\nC,Unknown\nD,300\n"
    dataset_id = upload_csv(api, content)
    body = clean2(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert "amount" in body["conversions_skipped"]
    assert body["columns_processed"]["amount"]["action"] == ACTION_SKIPPED
    assert body["columns_processed"]["amount"]["new_dtype"] == "object"
# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean2(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean2(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_missing_file_returns_404(api, tmp_path):
    user = api.create_user(USERNAME, PASSWORD)
    missing = tmp_path / "gone.csv"
    dataset = api.create_dataset_record(
        owner_id=user.id, original_filename="gone.csv", file_path=missing
    )
    response = clean2(api, dataset.id, api.auth_headers(USERNAME))
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
    response = clean2(api, dataset.id, api.auth_headers(USERNAME))
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
    response = clean2(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_unauthenticated_request_returns_401(api):
    response = api.client.post("/datasets/1/clean/duplicates-and-types")
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

    response = clean2(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_clean_own_dataset(api):
    dataset_id = upload_csv(api, "ID,Name\n1,A\n2,B\n")
    assert clean2(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


# ---------------------------------------------------------------------------
# Regression guards: Part 2A, Part 2B and Part 2C-1 keep working
# ---------------------------------------------------------------------------


def test_parta_partb_and_partc1_endpoints_still_work(api):
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

    details_response = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details_response.status_code == 200

    for path in ("structure", "missing-values", "quality", "summary"):
        resp = api.client.get(f"/datasets/{dataset_id}/{path}", headers=headers)
        assert resp.status_code == 200, path

    missing_clean = api.client.post(
        f"/datasets/{dataset_id}/clean/missing-values", headers=headers
    )
    assert missing_clean.status_code == 200

    this_clean = clean2(api, dataset_id, headers)
    assert this_clean.status_code == 200
    assert this_clean.json()["duplicate_rows_found"] == 1