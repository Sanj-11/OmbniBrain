"""Tests for Part 2C-3C - Numeric & Special Value Handling.

Covers the POST /datasets/{dataset_id}/clean/numeric-special endpoint and the
reusable :mod:`cleaning` service functions:

* numeric-format normalization (currency symbols, thousands separators),
* unambiguous-only currency handling (``$1,000`` -> ``1000``, ``₹2,500`` -> ``2500``),
* thousands-separator conversion (``1,500`` -> ``1500``),
* preservation of normal numeric values,
* preservation of ambiguous values (``Unknown``, ``not available``, ``100-200``),
* positive / negative infinity detection and missing-value handling,
* unsafe-conversion rejection (mixed columns are preserved, not coerced),
* strict data preservation (rows, columns, names, values),
* error handling and the dataset ownership/authentication rules from Part 2A,
* regression guards for Part 2A, Part 2B and earlier Part 2C stages.
"""
import io

import pandas as pd

from cleaning import (
    _normalize_numeric_string,
    clean_numeric_and_special_values,
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


def clean_numeric(api, dataset_id, headers):
    return api.client.post(
        f"/datasets/{dataset_id}/clean/numeric-special", headers=headers
    )


# ---------------------------------------------------------------------------
# Service: normalization helper
# ---------------------------------------------------------------------------


def test_normalize_currency_dollar():
    parsed = _normalize_numeric_string("$1,000")
    assert parsed == ("1000", True, True)


def test_normalize_currency_rupee():
    parsed = _normalize_numeric_string("₹2,500")
    assert parsed == ("2500", True, True)


def test_normalize_thousands_separator():
    parsed = _normalize_numeric_string("1,500")
    assert parsed == ("1500", False, True)


def test_normalize_plain_number():
    assert _normalize_numeric_string("123") == ("123", False, False)
    assert _normalize_numeric_string("12.5") == ("12.5", False, False)
    assert _normalize_numeric_string("+42") == ("+42", False, False)


def test_normalize_rejects_infinity_tokens():
    # Infinity is a special value handled separately, never a plain number.
    assert _normalize_numeric_string("inf") is None
    assert _normalize_numeric_string("-inf") is None
    assert _normalize_numeric_string("Infinity") is None


def test_normalize_rejects_ambiguous_values():
    assert _normalize_numeric_string("100-200") is None
    assert _normalize_numeric_string("ABC123") is None
    assert _normalize_numeric_string("Unknown") is None
    assert _normalize_numeric_string("not available") is None
    assert _normalize_numeric_string("") is None
    assert _normalize_numeric_string("   ") is None


# ---------------------------------------------------------------------------
# Service: numeric normalization
# ---------------------------------------------------------------------------


def test_service_normalizes_currency_and_thousands():
    df = pd.DataFrame({"amount": ["$1,000", "₹2,500", "1,500"]}, dtype=object)
    result = clean_numeric_and_special_values(df)
    assert result.numeric_values_normalized == 3
    assert result.currency_values_handled == 2
    assert result.thousands_separators_handled == 3
    assert result.cleaned_df["amount"].tolist() == ["1000", "2500", "1500"]


def test_service_preserves_plain_numeric_values():
    df = pd.DataFrame({"value": ["42", "3.14", "7"]}, dtype=object)
    result = clean_numeric_and_special_values(df)
    # Plain numbers are kept exactly as-is (no formatting to change).
    assert result.numeric_values_normalized == 0
    assert result.cleaned_df["value"].tolist() == ["42", "3.14", "7"]


def test_service_preserves_unknown_and_not_available():
    df = pd.DataFrame(
        {"category": ["Unknown", "not available", "Other"]}, dtype=object
    )
    result = clean_numeric_and_special_values(df)
    assert result.numeric_values_normalized == 0
    assert result.cleaned_df["category"].tolist() == [
        "Unknown",
        "not available",
        "Other",
    ]


def test_service_preserves_range_value():
    df = pd.DataFrame({"range": ["100-200"]}, dtype=object)
    result = clean_numeric_and_special_values(df)
    assert result.numeric_values_normalized == 0
    assert result.cleaned_df["range"].iloc[0] == "100-200"


def test_service_preserves_alphanumeric_value():
    df = pd.DataFrame({"code": ["ABC123"]}, dtype=object)
    result = clean_numeric_and_special_values(df)
    assert result.numeric_values_normalized == 0
    assert result.cleaned_df["code"].iloc[0] == "ABC123"


# ---------------------------------------------------------------------------
# Service: infinity handling
# ---------------------------------------------------------------------------


def test_service_handles_positive_infinity_string():
    df = pd.DataFrame({"value": ["inf"]}, dtype=object)
    result = clean_numeric_and_special_values(df)
    assert result.positive_infinity_handled == 1
    assert result.negative_infinity_handled == 0
    assert pd.isna(result.cleaned_df["value"].iloc[0])


def test_service_handles_negative_infinity_string():
    df = pd.DataFrame({"value": ["-inf"]}, dtype=object)
    result = clean_numeric_and_special_values(df)
    assert result.positive_infinity_handled == 0
    assert result.negative_infinity_handled == 1
    assert pd.isna(result.cleaned_df["value"].iloc[0])


def test_service_handles_infinity_variants():
    df = pd.DataFrame(
        {"value": ["inf", "Infinity", "-inf", "-Infinity"]}, dtype=object
    )
    result = clean_numeric_and_special_values(df)
    assert result.positive_infinity_handled == 2
    assert result.negative_infinity_handled == 2
    assert result.cleaned_df["value"].isna().sum() == 4


def test_service_handles_real_float_infinities():
    df = pd.DataFrame({"value": [float("inf"), float("-inf"), 1.5]})
    result = clean_numeric_and_special_values(df)
    assert result.positive_infinity_handled == 1
    assert result.negative_infinity_handled == 1
    assert result.cleaned_df["value"].iloc[0] != float("inf")
    assert result.cleaned_df["value"].iloc[1] != float("-inf")
    assert result.cleaned_df["value"].iloc[2] == 1.5


# ---------------------------------------------------------------------------
# Service: unsafe conversion is rejected and data preserved
# ---------------------------------------------------------------------------


def test_service_rejects_unsafe_conversion_and_preserves_values():
    # A column that is partly numeric formatting and partly junk text. The
    # success ratio (3/4 = 75%) is below the safe threshold, so the conversion
    # is rejected and every value - including the junk token - is preserved.
    df = pd.DataFrame(
        {"amount": ["1,000", "2,000", "3,000", "Unknown"]}, dtype=object
    )
    result = clean_numeric_and_special_values(df)
    assert result.unsafe_conversions_skipped == 1
    assert "amount" in result.conversion_skipped_columns
    assert result.cleaned_df["amount"].tolist() == [
        "1000",
        "2000",
        "3000",
        "Unknown",
    ]


def test_service_safe_conversion_is_not_flagged_unsafe():
    # Entirely numeric strings are normalized but the conversion is safe (no
    # values destroyed), so nothing is skipped.
    df = pd.DataFrame({"amount": ["1,000", "2,000", "3,000"]}, dtype=object)
    result = clean_numeric_and_special_values(df)
    assert result.unsafe_conversions_skipped == 0
    assert result.cleaned_df["amount"].tolist() == ["1000", "2000", "3000"]


# ---------------------------------------------------------------------------
# Service: data preservation
# ---------------------------------------------------------------------------


def test_service_preserves_rows_columns_and_names():
    df = pd.DataFrame(
        {
            "amount": ["$1,000", "$2,000", "1,500"],
            "status": ["paid", "pending", "Unknown"],
            "id": [1, 2, 3],
        }
    )
    before_rows = len(df)
    before_cols = len(df.columns)
    before_names = list(df.columns)
    before_amount = df["amount"].copy(deep=True)
    original = df.copy(deep=True)

    result = clean_numeric_and_special_values(df)
    assert len(result.cleaned_df) == before_rows
    assert len(result.cleaned_df.columns) == before_cols
    assert list(result.cleaned_df.columns) == before_names
    # The original DataFrame was never modified.
    assert df["amount"].equals(before_amount)
    assert df.equals(original)


# ---------------------------------------------------------------------------
# Endpoint: numeric normalization
# ---------------------------------------------------------------------------


def test_endpoint_normalizes_numeric_and_special_values(api):
    content = (
        'amount,note\n'
        '"$1,000",inf\n'
        '"₹2,500",-inf\n'
        '"1,500",Unknown\n'
        '42,not available\n'
    )
    dataset_id = upload_csv(api, content)
    body = clean_numeric(api, dataset_id, api.auth_headers(USERNAME)).json()

    # Two currency symbols ($ and ₹) were handled; three values had thousands
    # separators; three values had their numeric formatting normalized.
    assert body["currency_values_handled"] == 2
    assert body["thousands_separators_handled"] == 3
    assert body["numeric_values_normalized"] == 3
    # Two infinities (inf / -inf) became missing values in the note column.
    assert body["positive_infinity_handled"] == 1
    assert body["negative_infinity_handled"] == 1


def test_endpoint_rejects_unsafe_conversion_and_preserves_data(api):
    # The amount column is a mix of thousands-separated numbers and one junk
    # token. After normalization 3/4 values are numeric (75% < 80% threshold),
    # so the conversion is rejected and the junk token is preserved, never
    # coerced to a number or silently dropped.
    content = (
        'amount,category\n'
        '"1,000",apple\n'
        '"2,000",orange\n'
        '"3,500",pear\n'
        'unknown,banana\n'
    )
    dataset_id = upload_csv(api, content)
    body = clean_numeric(api, dataset_id, api.auth_headers(USERNAME)).json()

    assert body["unsafe_conversions_skipped"] == 1
    assert "amount" in body["conversion_skipped_columns"]
    # Rows / columns / names are all preserved by the endpoint.
    assert body["rows_before"] == body["rows_after"] == 4
    assert body["columns_before"] == body["columns_after"] == 2
    assert body["column_names_unchanged"] is True


def test_endpoint_preserves_ambiguous_values(api):
    content = (
        "code,tag\n"
        "ABC123,100-200\n"
        "X42,Unknown\n"
        "23,not available\n"
    )
    dataset_id = upload_csv(api, content)
    body = clean_numeric(api, dataset_id, api.auth_headers(USERNAME)).json()
    # Nothing meets unambiguously-numeric criteria, so nothing is normalized.
    assert body["numeric_values_normalized"] == 0
    assert body["currency_values_handled"] == 0


# ---------------------------------------------------------------------------
# Endpoint: original dataset remains unchanged
# ---------------------------------------------------------------------------


def test_endpoint_original_dataset_remains_unchanged(api):
    content = 'amount,note\n"$1,000",100\n'
    dataset_id = upload_csv(api, content)
    headers = api.auth_headers(USERNAME)

    clean_numeric(api, dataset_id, headers)

    details = api.client.get(f"/datasets/{dataset_id}", headers=headers)
    assert details.status_code == 200
    preview = details.json()["preview"]
    # The stored upload keeps its original (un-normalized) value.
    assert preview[0]["amount"] == "$1,000"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_dataset_returns_404(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean_numeric(api, 999_999, api.auth_headers(USERNAME))
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_dataset_id_returns_422(api):
    api.create_user(USERNAME, PASSWORD)
    response = clean_numeric(api, "not-an-integer", api.auth_headers(USERNAME))
    assert response.status_code == 422


def test_unauthenticated_request_returns_401(api):
    response = api.client.post("/datasets/1/clean/numeric-special")
    assert response.status_code == 401


def test_cannot_clean_another_users_dataset(api, tmp_path):
    api.create_user(USERNAME, PASSWORD)
    bob = api.create_user("bob", PASSWORD)
    bob_file = tmp_path / "bobs.csv"
    bob_file.write_text("amount\n\"$1,000\"\n", encoding="utf-8")
    bob_dataset = api.create_dataset_record(
        owner_id=bob.id,
        original_filename="bobs.csv",
        file_path=bob_file,
        rows=1,
        columns=1,
    )
    response = clean_numeric(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_clean_own_dataset(api):
    dataset_id = upload_csv(api, 'amount\n"$1,000"\n')
    assert clean_numeric(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


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
    response = clean_numeric(api, dataset.id, api.auth_headers(USERNAME))
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
    ):
        assert (
            api.client.post(
                f"/datasets/{dataset_id}/clean/{clean_path}", headers=headers
            ).status_code
            == 200
        )
    # The new endpoint is itself reachable in the same run.
    assert clean_numeric(api, dataset_id, headers).status_code == 200