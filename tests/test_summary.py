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


def summary(api, dataset_id, headers):
    return api.client.get(f"/datasets/{dataset_id}/summary", headers=headers)


# ---------------------------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------------------------


def test_numerical_summary_returns_correct_statistics(api):
    content = (
        "name,age,salary\n"
        "Anna,28,1100.5\n"
        "Bob,32,950.25\n"
        "Cara,41,2250.75\n"
        "Dan,19,430.0\n"
    )
    dataset_id = upload_csv(api, content)
    body = summary(api, dataset_id, api.auth_headers(USERNAME)).json()

    numerical = body["numerical_summary"]
    assert set(numerical) == {"age", "salary"}

    df = pd.read_csv(io.StringIO(content))
    expected = df["age"].describe()
    age = numerical["age"]
    assert age["count"] == int(expected["count"])
    assert age["mean"] == float(expected["mean"])
    assert age["std"] == float(expected["std"])
    assert age["min"] == float(expected["min"])
    assert age["25%"] == float(expected["25%"])
    assert age["50%"] == float(expected["50%"])
    assert age["75%"] == float(expected["75%"])
    assert age["max"] == float(expected["max"])


# ---------------------------------------------------------------------------
# Categorical summary
# ---------------------------------------------------------------------------


def test_categorical_summary_returns_correct_statistics(api):
    content = (
        "name,department\n"
        "A,Sales\n"
        "B,HR\n"
        "C,Sales\n"
        "D,Sales\n"
        "E,IT\n"
    )
    dataset_id = upload_csv(api, content)
    body = summary(api, dataset_id, api.auth_headers(USERNAME)).json()

    categorical = body["categorical_summary"]
    assert set(categorical) == {"name", "department"}
    dept = categorical["department"]
    assert dept["top"] == "Sales"
    assert dept["frequency"] == 3


# ---------------------------------------------------------------------------
# Mixed dataset (integer, float, string, category, boolean, datetime)
# ---------------------------------------------------------------------------


def test_mixed_dataset_summary_generates_without_errors(api):
    api.create_user(USERNAME, PASSWORD)
    headers = api.auth_headers(USERNAME)

    frame = pd.DataFrame(
        {
            "age": [28, 31, 41],                       # int64      -> numerical
            "score": [1.5, 2.5, 3.5],              # float64    -> numerical
            "name": ["a", "b", "c"],                     # object     -> categorical
            "role": pd.Categorical(["x", "y", "x"]),  # category -> categorical
            "active": [True, False, True],            # bool       -> categorical
            "joined": [                                # datetime   -> separate
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

    body = summary(api, dataset_id, headers).json()
    # int/float are numerical; object/category/bool are categorical; the
    # datetime column is kept separate and excluded from both summaries.
    assert set(body["numerical_summary"]) == {"age", "score"}
    assert set(body["categorical_summary"]) == {"name", "role", "active"}
    assert "joined" not in body["numerical_summary"]
    assert "joined" not in body["categorical_summary"]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def test_missing_values_do_not_crash_summary(api):
    content = (
        "name,age,salary,department,active\n"
        "Anna,28,1100.5,Engineering,True\n"
        "Bob,32,950.25,,False\n"
        "Cara,,2250.75,Marketing,True\n"
        ",19,430.0,Marketing,\n"
    )
    dataset_id = upload_csv(api, content)
    response = summary(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()

    # describe() uses the non-null count and ignores NaN for the stats.
    age = body["numerical_summary"]["age"]
    assert age["count"] == 3
    # value_counts() drops NaN, so a categorical summary with missing values
    # must still succeed and report non-null counts.
    dept = body["categorical_summary"]["department"]
    assert dept["count"] == 3
    assert dept["unique"] == 2
    assert "top" in dept


# ---------------------------------------------------------------------------
# Constant / single-value columns
# ---------------------------------------------------------------------------


def test_constant_value_columns_do_not_error(api):
    content = (
        "department,salary\n"
        "Sales,100\n"
        "Sales,100\n"
        "Sales,100\n"
        "Sales,100\n"
    )
    dataset_id = upload_csv(api, content)
    response = summary(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200, response.text
    body = response.json()

    dept = body["categorical_summary"]["department"]
    assert dept["count"] == 4
    assert dept["unique"] == 1
    assert dept["top"] == "Sales"
    assert dept["frequency"] == 4

    sal = body["numerical_summary"]["salary"]
    # A constant numerical column has std == 0 and min == max.
    assert sal["min"] == 100.0
    assert sal["max"] == 100.0
    assert sal["mean"] == 100.0
    assert sal["std"] == 0.0


# ---------------------------------------------------------------------------
# Empty dataset
# ---------------------------------------------------------------------------


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
    response = summary(api, dataset.id, api.auth_headers(USERNAME))
    # The project rejects empty datasets at upload time, so the summary raises
    # the same validation error rather than crashing on zero rows.
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_response_is_json_serializable(api):
    content = (
        "name,age,salary\n"
        "Anna,28,1100.5\n"
        "Bob,32,950.25\n"
        "Cara,41,2250.75\n"
    )
    dataset_id = upload_csv(api, content)
    response = summary(api, dataset_id, api.auth_headers(USERNAME))
    assert response.status_code == 200
    # JSON-encoding the parsed response proves the payload contains no raw
    # NumPy/pandas objects that FastAPI could not have serialised.
    import json

    payload = response.json()
    json.dumps(payload)
    assert "numerical_summary" in payload
    assert "categorical_summary" in payload


# ---------------------------------------------------------------------------
# Authentication / ownership
# ---------------------------------------------------------------------------


def test_unauthenticated_request_returns_401(api):
    response = api.client.get("/datasets/1/summary")
    assert response.status_code == 401


def test_cannot_summary_another_users_dataset(api, tmp_path):
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
    response = summary(api, bob_dataset.id, api.auth_headers(USERNAME))
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


def test_owner_can_summary_own_dataset(api):
    content = "name,age\nA,10\nB,20\n"
    dataset_id = upload_csv(api, content)
    assert summary(api, dataset_id, api.auth_headers(USERNAME)).status_code == 200


# ---------------------------------------------------------------------------
# Regression guards: Part 2A, 2B-1, 2B-2 and 2B-3 keep working
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

    for path in ("structure", "missing-values", "quality"):
        resp = api.client.get(f"/datasets/{dataset_id}/{path}", headers=headers)
        assert resp.status_code == 200, path