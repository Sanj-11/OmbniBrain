"""Tests for Part 2C-3A - Column Name Standardization.

Covers the reusable :mod:`cleaning` service functions and the
POST /datasets/{dataset_id}/clean/standardize-columns endpoint:

* whitespace trimming, lowercasing, space-to-underscore conversion,
* separator normalization (``-`` ``.`` ``/`` and other special characters),
* collapsing of repeated and leading/trailing underscores,
* empty/unnamed column handling (``""``, whitespace, ``None``),
* duplicate column-name detection and suffix resolution,
* strict data preservation (rows, columns, values, dtypes, order),
* error handling, ownership/authentication and regression guards.
"""
import io

import pandas as pd

from cleaning import (
    EMPTY_COLUMN_PREFIX,
    _assign_unique_names,
    _normalize_column_name,
    standardize_column_names,
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


def standardize(api, dataset_id, headers):
    return api.client.post(
        f"/datasets/{dataset_id}/clean/standardize-columns", headers=headers
    )


# ---------------------------------------------------------------------------
# Single-name normalization rules
# ---------------------------------------------------------------------------


def test_normalize_trims_whitespace():
    assert _normalize_column_name(" Customer Name ") == "customer_name"


def test_normalize_lowercases():
    assert _normalize_column_name("Customer Name") == "customer_name"


def test_normalize_replaces_spaces_with_underscores():
    assert _normalize_column_name("customer name") == "customer_name"


def test_normalize_dash_separator():
    assert _normalize_column_name("Order-Date") == "order_date"


def test_normalize_dot_separator():
    assert _normalize_column_name("Order.Date") == "order_date"


def test_normalize_slash_separator():
    assert _normalize_column_name("Order/Date") == "order_date"


def test_normalize_collapses_repeated_underscores():
    assert _normalize_column_name("customer___name") == "customer_name"


def test_normalize_strips_leading_and_trailing_underscores():
    assert _normalize_column_name("_customer_name_") == "customer_name"


def test_normalize_other_special_characters():
    # Other special characters also collapse into a single underscore.
    assert _normalize_column_name("Total # Sales @ (USD)") == "total_sales_usd"


def test_normalize_preserves_digits():
    assert _normalize_column_name("col 2026") == "col_2026"


def test_normalize_empty_and_none():
    assert _normalize_column_name("") == ""
    assert _normalize_column_name("   ") == ""
    assert _normalize_column_name(None) == ""


def test_unique_names_resolves_collisions():
    assert _assign_unique_names(
        ["customer_name", "customer_name", "customer_name"]
    ) == ["customer_name", "customer_name_1", "customer_name_2"]


def test_unique_names_never_overwrites_on_suffix_clash():
    # A real column already named "customer_name_1" must not be overwritten.
    assert _assign_unique_names(
        ["customer_name", "customer_name", "customer_name_1"]
    ) == ["customer_name", "customer_name_1", "customer_name_1_1"]


# ---------------------------------------------------------------------------
# Service: full standardization
# ---------------------------------------------------------------------------


def test_service_standardizes_normal_names():
    df = pd.DataFrame(
        {
            "Customer Name": ["A", "B"],
            "Total Sales": [10, 20],
            "Order-Date": ["2026-01-01", "2026-02-01"],
        }
    )
    result = standardize_column_names(df)
    assert list(result.cleaned_df.columns) == [
        "customer_name",
        "total_sales",
        "order_date",
    ]
    assert result.columns_processed == 3
    assert result.columns_changed == 3
    assert result.column_changes == {
        "Customer Name": "customer_name",
        "Total Sales": "total_sales",
        "Order-Date": "order_date",
    }
    assert result.empty_names_handled == 0
    assert result.duplicate_names_handled == 0


def test_service_handles_whitespace_and_capitalization():
    df = pd.DataFrame({"  Customer NAME ": ["x"], "PRODUCT.ID": ["y"]})
    result = standardize_column_names(df)
    assert list(result.cleaned_df.columns) == ["customer_name", "product_id"]


def test_service_handles_empty_names():
    df = pd.DataFrame({"": [1, 2], "   ": [3, 4], "Name": ["a", "b"]})
    result = standardize_column_names(df)
    assert list(result.cleaned_df.columns) == [
        f"{EMPTY_COLUMN_PREFIX}1",
        f"{EMPTY_COLUMN_PREFIX}2",
        "name",
    ]
    assert result.empty_names_handled == 2
    # Empty columns are not deleted.
    assert result.columns_processed == 3


def test_service_handles_none_column_name():
    df = pd.DataFrame({None: [1, 2], "Value": [3, 4]})
    result = standardize_column_names(df)
    assert list(result.cleaned_df.columns) == [
        f"{EMPTY_COLUMN_PREFIX}1",
        "value",
    ]
    assert result.empty_names_handled == 1


def test_service_resolves_duplicate_column_names():
    df = pd.DataFrame(
        {
            "Customer Name": [1],
            "customer_name": [2],
            "Customer-Name": [3],
        }
    )
    result = standardize_column_names(df)
    assert list(result.cleaned_df.columns) == [
        "customer_name",
        "customer_name_1",
        "customer_name_2",
    ]
    assert result.duplicate_names_handled == 1
    assert result.columns_processed == 3


def test_service_does_not_modify_original_dataframe():
    original = pd.DataFrame({"Customer Name": ["A", "B"], "Total Sales": [1, 2]})
    snapshot = original.copy(deep=True)
    standardize_column_names(original)
    assert original.equals(snapshot)


def test_service_preserves_data_rows_columns_values_and_order():
    df = pd.DataFrame(
        {
            "Customer Name": ["Anusha", "Rahul"],
            "Order Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "Total Sales": pd.Series([100.5, 200.25], dtype="float64"),
            "Active": [True, False],
        }
    )
    result = standardize_column_names(df)

    standardized = result.cleaned_df
    # Counts and order unchanged.
    assert len(standardized) == len(df)
    assert len(standardized.columns) == len(df.columns)
    assert list(standardized.columns) == [
        "customer_name",
        "order_date",
        "total_sales",
        "active",
    ]
    # Values and data types untouched.
    assert standardized["customer_name"].tolist() == ["Anusha", "Rahul"]
    assert standardized["total_sales"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(standardized["order_date"])
    assert standardized["active"].dtype == bool
    # Original dtypes are preserved exactly.
    assert standardized.dtypes.tolist() == df.dtypes.tolist()


def test_service_chains_from_previous_stage_output():
    # The standardized DataFrame is returned for chaining into Part 2C-3B.
    df = pd.DataFrame({"Customer Name": ["A"], "Total Sales": [1]})
    result = standardize_column_names(df)
    assert result.cleaned_df is not None
    assert isinstance(result.cleaned_df, pd.DataFrame)
    assert "customer_name" in result.final_column_names


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


def test_endpoint_returns_correct_report(api):
    content = "Customer Name,Total Sales,Order-Date\nAnusha,100,2026-01-01\nRahul,200,2026-02-01\n"
    dataset_id = upload_csv(api, content)
    response = standardize(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["columns_processed"] == 3
    assert body["columns_changed"] == 3
    assert body["final_column_names"] == [
        "customer_name",
        "total_sales",
        "order_date",
    ]
    assert body["empty_names_handled"] == 0
    assert body["duplicate_names_handled"] == 0
    assert body["column_changes"] == {
        "Customer Name": "customer_name",
        "Total Sales": "total_sales",
        "Order-Date": "order_date",
    }


def test_endpoint_handles_duplicate_names(api):
    # A CSV whose header repeats a name (after case/separator differences) keeps
    # all three columns; the endpoint must standardize and de-duplicate them.
    content = "Customer Name,customer_name,Customer-Name\n1,3,5\n2,4,6\n"
    dataset_id = upload_csv(api, content)
    body = standardize(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["columns_processed"] == 3
    assert body["duplicate_names_handled"] == 1
    assert body["final_column_names"] == [
        "customer_name",
        "customer_name_1",
        "customer_name_2",
    ]


def test_endpoint_normalizes_special_characters_in_upload(api):
    # A header containing special characters is normalized through the upload
    # path; pandas only mangles truly-empty headers, so this exercises the
    # separator-normalization rule end to end.
    content = "Total # Sales,Customer/ID\n10,1\n20,2\n"
    dataset_id = upload_csv(api, content)
    body = standardize(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["final_column_names"] == ["total_sales", "customer_id"]


def test_endpoint_report_is_json_serializable(api):
    import json

    content = "Customer Name,Total Sales\nA,1\nB,2\n"
    dataset_id = upload_csv(api, content)
    response = standardize(api, dataset_id, api.auth_headers(USERNAME))
    json.dumps(response.json())


def test_endpoint_original_dataset_remains_unchanged(api):
    content = "Customer Name,Total Sales\nA,1\nB,2\n"
    dataset_id = upload_csv(api, content)
    headers = api.auth_headers(USERNAME)
    standardize(api, dataset_id, headers)

    structure = api.client.get(f"/datasets/{dataset_id}/structure", headers=headers)
    assert structure.status_code == 200
    # The stored dataset keeps its original (un-standardized) column names.
    assert "Customer Name" in structure.json()["column_names"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = standardize(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = standardize(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_unauthenticated_request_returns_401(api):
    response = api.client.post("/datasets/1/clean/standardize-columns")
    assert response.status_code == 401


def test_cannot_standardize_another_users_dataset(api, tmp_path):
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
    response = standardize(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_standardize_own_dataset(api):
    dataset_id = upload_csv(api, "Customer Name,Total Sales\nA,1\n")
    assert (
        standardize(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200
    )


# ---------------------------------------------------------------------------
# Regression guards: Part 2A, Part 2B and earlier Part 2C stages keep working
# ---------------------------------------------------------------------------


def test_previous_partc_stages_still_work(api):
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
    # The new standardization endpoint is itself reachable in the same run.
    assert standardize(api, dataset_id, headers).status_code == 200
