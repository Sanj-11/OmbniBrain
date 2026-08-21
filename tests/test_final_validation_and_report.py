"""Tests for Part 2C-3D - Final Value Validation & Cleaning Report.

Covers the reusable :mod:`cleaning` service functions and the
POST /datasets/{dataset_id}/clean/final-validation endpoint:

* the combined report merging the Part 2C-3A/2C-3B/2C-3C outcomes,
* the final value validation (remaining missing-like, whitespace, numeric and
  infinity values),
* the before/after data-integrity checks (rows, columns, names, values),
* the ``passed`` / ``warning`` / ``failed`` status computation,
* error handling and the ownership/authentication rules from Part 2A,
* regression guards for Part 2A, Part 2B and the earlier Part 2C stages.
"""
import pandas as pd

from cleaning import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_WARNING,
    _evaluate_status,
    validate_and_report,
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


def final_validate(api, dataset_id, headers):
    return api.client.post(
        f"/datasets/{dataset_id}/clean/final-validation", headers=headers
    )


# ---------------------------------------------------------------------------
# Status computation
# ---------------------------------------------------------------------------


def test_status_passed_when_all_checks_hold():
    assert (
        _evaluate_status(True, True, True, True, True, 0, 0, 0, 0, 0)
        == STATUS_PASSED
    )


def test_status_warning_when_minor_issues_remain():
    # Remaining missing-like values are a quality issue, not structural.
    assert _evaluate_status(True, True, True, True, True, 0, 0, 1, 0, 0) == (
        STATUS_WARNING
    )
    # As are remaining whitespace-only or invalid-numeric values.
    assert _evaluate_status(True, True, True, True, True, 0, 0, 0, 2, 0) == (
        STATUS_WARNING
    )
    assert _evaluate_status(True, True, True, True, True, 0, 0, 0, 0, 3) == (
        STATUS_WARNING
    )


def test_status_failed_on_structural_problems():
    # Unexpected row removal is structural.
    assert _evaluate_status(False, True, True, True, True, 0, 0, 0, 0, 0) == (
        STATUS_FAILED
    )
    # Duplicate / empty / non-standardized names are structural.
    assert _evaluate_status(True, True, False, True, True, 0, 0, 0, 0, 0) == (
        STATUS_FAILED
    )
    assert _evaluate_status(True, True, True, False, True, 0, 0, 0, 0, 0) == (
        STATUS_FAILED
    )
    # Any infinity that survived the numeric stage is structural.
    assert _evaluate_status(True, True, True, True, True, 1, 0, 0, 0, 0) == (
        STATUS_FAILED
    )


# ---------------------------------------------------------------------------
# Service: clean dataset
# ---------------------------------------------------------------------------


def test_service_clean_dataset_passes_validation():
    df = pd.DataFrame(
        {"name": ["Anna", "Bob"], "age": [28, 35], "active": [True, False]}
    )
    result = validate_and_report(df)
    assert result.report.status == STATUS_PASSED
    v = result.report.validation
    assert v.row_count_unchanged is True
    assert v.column_count_unchanged is True
    assert v.unique_column_names is True
    assert v.no_empty_column_names is True
    assert v.standardized_column_names is True
    assert v.remaining_missing_like_values == 0
    assert v.remaining_whitespace_only_values == 0
    assert v.remaining_invalid_numeric_values == 0
    assert v.remaining_pos_infinity == 0
    assert v.remaining_neg_infinity == 0


# ---------------------------------------------------------------------------
# Service: messy column names (fixed by 2C-3A, confirmed by 2C-3D)
# ---------------------------------------------------------------------------


def test_service_messy_column_names_confirmed_valid():
    df = pd.DataFrame({"Customer Name": [1, 2], "customer_name": [3, 4]})
    result = validate_and_report(df)
    assert list(result.cleaned_df.columns) == ["customer_name", "customer_name_1"]
    v = result.report.validation
    assert v.unique_column_names is True
    assert v.no_empty_column_names is True
    assert v.standardized_column_names is True
    assert result.report.column_standardization.duplicate_names_handled == 1
    assert result.report.status == STATUS_PASSED


def test_service_empty_column_names_confirmed_filled():
    df = pd.DataFrame({"": [1, 2], "Name": ["a", "b"]})
    result = validate_and_report(df)
    assert result.report.column_standardization.empty_names_handled == 1
    assert result.report.validation.no_empty_column_names is True
    assert result.report.validation.standardized_column_names is True

# ---------------------------------------------------------------------------
# Service: missing-like values (handled by 2C-3B, validated by 2C-3D)
# ---------------------------------------------------------------------------


def test_service_missing_like_values_handled_and_validated():
    df = pd.DataFrame({"city": ["A", "N/A", "NULL", "  B  "]})
    result = validate_and_report(df)
    assert result.report.string_cleaning.missing_like_values_handled == 2
    assert result.report.string_cleaning.whitespace_values_cleaned >= 1
    v = result.report.validation
    assert v.remaining_missing_like_values == 0
    assert v.remaining_whitespace_only_values == 0
    assert result.report.status == STATUS_PASSED


def test_service_warns_when_missing_like_value_remains_in_categorical():
    # Categorical columns are deliberately left untouched by 2C-3B, so a
    # missing-like token surviving there is reported as a warning (not failed).
    df = pd.DataFrame({"cat": pd.Categorical(["A", "NA", "B"])})
    result = validate_and_report(df)
    assert result.report.status == STATUS_WARNING
    assert result.report.validation.remaining_missing_like_values == 1


# ---------------------------------------------------------------------------
# Service: numeric formatting (handled by 2C-3C, validated by 2C-3D)
# ---------------------------------------------------------------------------


def test_service_numeric_formatting_handled_and_validated():
    df = pd.DataFrame({"amount": ["$1,000", "2,500", "300"]}, dtype=object)
    result = validate_and_report(df)
    assert result.report.numeric_cleaning.numeric_values_normalized == 2
    assert result.report.numeric_cleaning.currency_values_handled == 1
    assert result.cleaned_df["amount"].tolist() == ["1000", "2500", "300"]
    assert result.report.validation.remaining_invalid_numeric_values == 0
    assert result.report.status == STATUS_PASSED


def test_service_preserves_unusual_but_valid_values():
    # Ambiguous / non-numeric text is preserved - it must not fail validation.
    df = pd.DataFrame({"category": ["Unknown", "not available", "Other"]})
    result = validate_and_report(df)
    assert result.report.status == STATUS_PASSED
    assert result.cleaned_df["category"].tolist() == [
        "Unknown",
        "not available",
        "Other",
    ]


# ---------------------------------------------------------------------------
# Service: infinity (handled by 2C-3C, confirmed by 2C-3D)
# ---------------------------------------------------------------------------


def test_service_string_infinity_handled_and_confirmed():
    df = pd.DataFrame({"value": ["inf", "-inf", "Infinity", 1.5]}, dtype=object)
    result = validate_and_report(df)
    assert result.report.numeric_cleaning.positive_infinity_handled == 2
    assert result.report.numeric_cleaning.negative_infinity_handled == 1
    v = result.report.validation
    assert v.remaining_pos_infinity == 0
    assert v.remaining_neg_infinity == 0
    assert result.report.status == STATUS_PASSED


def test_service_real_float_infinity_handled_and_confirmed():
    df = pd.DataFrame({"value": [float("inf"), float("-inf"), 1.5]})
    result = validate_and_report(df)
    v = result.report.validation
    assert v.remaining_pos_infinity == 0
    assert v.remaining_neg_infinity == 0
    assert result.report.status == STATUS_PASSED


# ---------------------------------------------------------------------------
# Service: data integrity (no unexpected rows/columns removed, values kept)
# ---------------------------------------------------------------------------


def test_service_preserves_rows_columns_and_valid_values():
    original = pd.DataFrame(
        {"Customer Name": ["A", "B"], "Total Sales": [10, 20]}
    )
    snapshot = original.copy(deep=True)
    result = validate_and_report(original)

    # The original DataFrame is never modified.
    assert original.equals(snapshot)

    v = result.report.validation
    assert v.rows_before == v.rows_after == 2
    assert v.columns_before == v.columns_after == 2
    assert v.row_count_unchanged is True
    assert v.column_count_unchanged is True
    assert v.total_values_before == v.total_values_after == 4

    # Valid data is preserved (only the column names are standardized).
    assert result.cleaned_df["customer_name"].tolist() == ["A", "B"]
    assert result.cleaned_df["total_sales"].tolist() == [10, 20]


def test_service_original_dataframe_never_modified():
    df = pd.DataFrame({"City": ["A", "N/A"], "Amount": ["$1", "2"]})
    snapshot = df.copy(deep=True)
    validate_and_report(df)
    assert df.equals(snapshot)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


def test_endpoint_returns_combined_report_with_status(api):
    content = "Customer Name,Total Sales,Order-Date\nAnusha,100,2026-01-01\nRahul,200,2026-02-01\n"
    dataset_id = upload_csv(api, content)
    response = final_validate(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == STATUS_PASSED
    assert body["column_standardization"]["columns_processed"] == 3
    assert body["column_standardization"]["final_column_names"] == [
        "customer_name",
        "total_sales",
        "order_date",
    ]
    assert body["validation"]["row_count_unchanged"] is True
    assert body["validation"]["column_count_unchanged"] is True
    assert body["validation"]["unique_column_names"] is True
    assert body["validation"]["remaining_invalid_numeric_values"] == 0


def test_endpoint_handles_messy_column_names_and_cleaning(api):
    content = "Customer Name,Status,Value\nA,none,inf\nB,ok,5\n"
    dataset_id = upload_csv(api, content)
    body = final_validate(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["status"] == STATUS_PASSED
    # Column names are standardized and unique.
    assert body["column_standardization"]["final_column_names"] == [
        "customer_name",
        "status",
        "value",
    ]
    assert body["validation"]["unique_column_names"] is True
    # The N/A cell was cleaned as a missing-like value.
    assert body["string_cleaning"]["missing_like_values_handled"] >= 1
    # The inf token was cleaned as infinity; none remains.
    assert body["numeric_cleaning"]["positive_infinity_handled"] >= 1
    assert body["validation"]["remaining_pos_infinity"] == 0


def test_endpoint_report_is_json_serializable(api):
    import json

    content = "name,age\nAnna,28\nBob,35\n"
    dataset_id = upload_csv(api, content)
    response = final_validate(api, dataset_id, api.auth_headers(USERNAME))
    json.dumps(response.json())


def test_endpoint_original_dataset_remains_unchanged(api):
    content = "Customer Name,Total Sales\nA,1\nB,2\n"
    dataset_id = upload_csv(api, content)
    headers = api.auth_headers(USERNAME)
    final_validate(api, dataset_id, headers)

    structure = api.client.get(f"/datasets/{dataset_id}/structure", headers=headers)
    assert structure.status_code == 200
    assert "Customer Name" in structure.json()["column_names"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = final_validate(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = final_validate(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_unauthenticated_request_returns_401(api):
    response = api.client.post("/datasets/1/clean/final-validation")
    assert response.status_code == 401


def test_cannot_validate_another_users_dataset(api, tmp_path):
    api.create_user(USERNAME, PASSWORD)
    bob = api.create_user("bob", PASSWORD)
    bob_file = tmp_path / "bobs.csv"
    bob_file.write_text("name,age\nBob,30\n", encoding="utf-8")
    bob_dataset = api.create_dataset_record(
        owner_id=bob.id,
        original_filename="bobs.csv",
        file_path=bob_file,
        rows=1,
        columns=2,
    )
    response = final_validate(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


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
    response = final_validate(api, dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


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

    for clean_path in (
        "missing-values",
        "duplicates-and-types",
        "standardize-columns",
        "missing-like-strings",
        "numeric-special",
        "final-validation",
    ):
        assert (
            api.client.post(
                f"/datasets/{dataset_id}/clean/{clean_path}", headers=headers
            ).status_code
            == 200
        ), clean_path

