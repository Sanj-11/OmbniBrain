"""End-to-end Week 1 Team C Module 2 test.

Verifies the two pieces of Module 2 as one user flow:
1. upload a supported dataset through POST /datasets/upload;
2. use the returned dataset_id with POST /query/.

This intentionally stops at dataset metadata/row-column analysis. Semantic
retrieval, embeddings and RAG belong to later weeks.
"""

import io

import pandas as pd


VALID_CSV = (
    "product,region,sales\n"
    "Laptop,North,1200\n"
    "Phone,South,900\n"
    "Tablet,West,700\n"
)


def test_upload_then_query_end_to_end(api):
    """A successfully uploaded dataset can immediately be queried by its owner."""
    api.create_user("module2_user")
    headers = api.auth_headers("module2_user")

    upload_response = api.client.post(
        "/datasets/upload",
        files={"file": ("sales.csv", VALID_CSV.encode("utf-8"), "text/csv")},
        headers=headers,
    )

    assert upload_response.status_code == 200, upload_response.text
    dataset = upload_response.json()
    assert dataset["original_filename"] == "sales.csv"
    assert dataset["rows"] == 3
    assert dataset["columns"] == 3
    assert dataset["id"] > 0

    query_response = api.client.post(
        "/query/",
        json={
            "text": "summarize sales",
            "dataset_id": dataset["id"],
        },
        headers=headers,
    )

    assert query_response.status_code == 200, query_response.text
    result = query_response.json()
    assert result["dataset_id"] == dataset["id"]
    assert result["rows_analyzed"] == 3
    assert result["columns_analyzed"] == 3
    assert "summarize sales" in result["answer"]


def test_uploaded_excel_can_be_queried_end_to_end(api):
    """The same upload -> query flow works for the supported Excel format."""
    api.create_user("excel_user")
    headers = api.auth_headers("excel_user")

    buffer = io.BytesIO()
    pd.DataFrame(
        {"name": ["A", "B"], "score": [90, 80]}
    ).to_excel(buffer, index=False, engine="openpyxl")

    upload_response = api.client.post(
        "/datasets/upload",
        files={
            "file": (
                "scores.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        headers=headers,
    )

    assert upload_response.status_code == 200, upload_response.text
    dataset_id = upload_response.json()["id"]

    query_response = api.client.post(
        "/query/",
        json={"text": "analyze scores", "dataset_id": dataset_id},
        headers=headers,
    )

    assert query_response.status_code == 200, query_response.text
    result = query_response.json()
    assert result["rows_analyzed"] == 2
    assert result["columns_analyzed"] == 2
