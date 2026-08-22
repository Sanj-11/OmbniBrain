"""Async query processing validation tests (Asynchronous Query Processing - Part 2).

These tests verify that the query endpoint keeps blocking work off the
FastAPI event loop while preserving all previous behaviour:

- Normal queries (with and without a dataset)
- Large dataset query (real pandas CSV parse)
- Multiple concurrent dataset queries
- The event loop stays responsive while a slow dataset query is in flight
- Invalid / missing / corrupted dataset handling stays correct
- Database-failure exceptions from the worker thread become clean 500s
- Ownership/security checks are unchanged
"""

import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

import query as query_module


def _write_csv(tmp_path: pathlib.Path, name: str, rows: int, columns: int) -> pathlib.Path:
    """Write a deterministic ``rows x columns`` integer CSV and return its path."""
    assert rows >= 1 and columns >= 1
    header = ",".join(f"col{c}" for c in range(columns))
    lines = [
        ",".join(str((r * 31 + c * 7) % 97) for c in range(columns))
        for r in range(rows)
    ]
    path = tmp_path / name
    path.write_text(f"{header}\n" + "\n".join(lines))
    return path


def test_normal_query_without_dataset(api):
    api.create_user("alice")
    resp = api.client.post(
        "/query/",
        json={"text": "show me sales data"},
        headers=api.auth_headers("alice"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "General query: show me sales data"
    assert body["dataset_id"] is None
    assert body["rows_analyzed"] == 0
    assert body["columns_analyzed"] == 0


def test_normal_query_with_owned_dataset(api, tmp_path):
    user = api.create_user("bob")
    ds_file = _write_csv(tmp_path, "data.csv", rows=4, columns=2)
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="data.csv",
        file_path=ds_file,
        rows=4,
        columns=2,
    )
    resp = api.client.post(
        "/query/",
        json={"text": "summarize", "dataset_id": dataset.id},
        headers=api.auth_headers("bob"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == dataset.id
    assert body["rows_analyzed"] == 4
    assert body["columns_analyzed"] == 2
    assert "summarize" in body["answer"]


def test_large_dataset_query_returns_correct_counts(api, tmp_path):
    user = api.create_user("alice")
    n_rows, n_cols = 50_000, 12
    ds_file = _write_csv(tmp_path, "big.csv", rows=n_rows, columns=n_cols)
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="big.csv",
        file_path=ds_file,
        rows=n_rows,
        columns=n_cols,
    )

    start = time.monotonic()
    resp = api.client.post(
        "/query/",
        json={"text": "analyze big data", "dataset_id": dataset.id},
        headers=api.auth_headers("alice"),
    )
    elapsed = time.monotonic() - start

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rows_analyzed"] == n_rows
    assert body["columns_analyzed"] == n_cols
    assert "big data" in body["answer"]
    assert elapsed < 30.0, f"large query took too long: {elapsed:.2f}s"


def test_multiple_concurrent_queries_all_succeed(api, tmp_path):
    user = api.create_user("alice")
    n_rows, n_cols = 20_000, 6
    ds_file = _write_csv(tmp_path, "mid.csv", rows=n_rows, columns=n_cols)
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="mid.csv",
        file_path=ds_file,
        rows=n_rows,
        columns=n_cols,
    )
    headers = api.auth_headers("alice")

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(
                api.client.post,
                "/query/",
                json={"text": f"q{i}", "dataset_id": dataset.id},
                headers=headers,
            )
            for i in range(8)
        ]
        responses = [f.result(timeout=60) for f in futures]

    assert all(r.status_code == 200 for r in responses)
    for r in responses:
        body = r.json()
        assert body["rows_analyzed"] == n_rows
        assert body["columns_analyzed"] == n_cols


def test_slow_dataset_query_does_not_block_event_loop(api, tmp_path, monkeypatch):
    """A slow blocking dataset load must not stall /health served on the loop."""
    user = api.create_user("alice")
    ds_file = _write_csv(tmp_path, "tiny.csv", rows=1, columns=2)
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="tiny.csv",
        file_path=ds_file,
        rows=1,
        columns=2,
    )

    def slow_load_dataset(file_path, filename):
        time.sleep(0.7)
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    monkeypatch.setattr(query_module, "load_dataset_file", slow_load_dataset)
    headers = api.auth_headers("alice")

    with ThreadPoolExecutor(max_workers=2) as pool:
        query_future = pool.submit(
            api.client.post,
            "/query/",
            json={"text": "slow", "dataset_id": dataset.id},
            headers=headers,
        )
        # Let the query request land and begin processing, then probe /health.
        time.sleep(0.15)
        h0 = time.monotonic()
        health = api.client.get("/health")
        health_elapsed = time.monotonic() - h0
        query_resp = query_future.result(timeout=15)

    assert health.status_code == 200
    assert query_resp.status_code == 200, query_resp.text
    body = query_resp.json()
    assert body["rows_analyzed"] == 3
    assert body["columns_analyzed"] == 2
    # The health probe must be served while the slow query is still running:
    # if the query blocked the event loop, this round-trip would take ~0.7s.
    assert health_elapsed < 0.5, (
        f"event loop was blocked: /health took {health_elapsed:.3f}s"
    )


def test_invalid_dataset_id_returns_404(api):
    api.create_user("alice")
    resp = api.client.post(
        "/query/",
        json={"text": "hello", "dataset_id": 9999999},
        headers=api.auth_headers("alice"),
    )
    assert resp.status_code == 404
    assert "Dataset not found" in resp.json()["detail"]


def test_missing_dataset_file_returns_404(api, tmp_path):
    user = api.create_user("alice")
    missing = tmp_path / "gone.csv"  # never created on disk
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="gone.csv",
        file_path=missing,
        rows=1,
        columns=1,
    )
    resp = api.client.post(
        "/query/",
        json={"text": "hello", "dataset_id": dataset.id},
        headers=api.auth_headers("alice"),
    )
    assert resp.status_code == 404
    assert "file not found" in resp.json()["detail"].lower()


def test_corrupted_dataset_file_returns_500(api, tmp_path):
    user = api.create_user("alice")
    corrupt = tmp_path / "bad.csv"
    corrupt.write_bytes(b"\x00\x01\x02\xff" * 32)  # binary garbage in a CSV file
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="bad.csv",
        file_path=corrupt,
        rows=1,
        columns=1,
    )
    resp = api.client.post(
        "/query/",
        json={"text": "hello", "dataset_id": dataset.id},
        headers=api.auth_headers("alice"),
    )
    assert resp.status_code == 500
    assert "could not process dataset file" in resp.json()["detail"].lower()


def test_database_failure_returns_500(api, tmp_path, monkeypatch):
    user = api.create_user("alice")
    ds_file = _write_csv(tmp_path, "data.csv", rows=2, columns=2)
    dataset = api.create_dataset_record(
        owner_id=user.id,
        original_filename="data.csv",
        file_path=ds_file,
        rows=2,
        columns=2,
    )

    def failing_lookup(db, dataset_id):
        raise RuntimeError("simulated database outage")

    monkeypatch.setattr(query_module.crud, "get_dataset_by_id", failing_lookup)

    resp = api.client.post(
        "/query/",
        json={"text": "hello", "dataset_id": dataset.id},
        headers=api.auth_headers("alice"),
    )
    assert resp.status_code == 500
    assert "database" in resp.json()["detail"].lower()


def test_cross_user_dataset_access_returns_403(api, tmp_path):
    owner = api.create_user("owner")
    intruder = api.create_user("intruder")
    ds_file = _write_csv(tmp_path, "secret.csv", rows=2, columns=2)
    dataset = api.create_dataset_record(
        owner_id=owner.id,
        original_filename="secret.csv",
        file_path=ds_file,
        rows=2,
        columns=2,
    )
    resp = api.client.post(
        "/query/",
        json={"text": "steal", "dataset_id": dataset.id},
        headers=api.auth_headers("intruder"),
    )
    assert resp.status_code == 403
    assert "Not authorized" in resp.json()["detail"]