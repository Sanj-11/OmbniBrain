import asyncio
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_TMP = tempfile.TemporaryDirectory(prefix="omnibrain_qint_")
_TMP_PATH = pathlib.Path(_TMP.name)

# Environment MUST be set before importing main / database / config.
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_PATH / 'qint.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP_PATH / "uploads")

import httpx  # noqa: E402
import uvicorn  # noqa: E402

import models  # noqa: E402
import query as query_module  # noqa: E402
from database import SessionLocal, engine  # noqa: E402
from main import app  # noqa: E402

PORT = 8723
BASE = f"http://127.0.0.1:{PORT}"

REPORT = []  # (name, expected, actual_status, ok, note)


def record(name: str, expected: str, actual, ok: bool, note: str = ""):
    REPORT.append((name, expected, str(actual), ok, note))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: expected {expected}, got {actual} {note}")


def _matrix_csv(rows: int, cols: int) -> bytes:
    """Deterministic int matrix CSV kept compact enough to build quickly."""
    header = ",".join(f"c{c}" for c in range(cols))
    lines = [
        ",".join(str((r * 131 + c * 17) % 1000) for c in range(cols))
        for r in range(rows)
    ]
    return f"{header}\n".encode() + "\n".join(lines).encode()


async def run_suite(client: httpx.AsyncClient):
    # ------------------------------------------------------------------
    # 1. Startup surface: /, /health, /docs, OpenAPI with the /query route
    # ------------------------------------------------------------------
    r = await client.get("/")
    record("GET / (server running)", "200", r.status_code, r.status_code == 200)

    r = await client.get("/health")
    record("GET /health", "200", r.status_code, r.status_code == 200)

    r = await client.get("/docs")
    ok = r.status_code == 200 and "swagger-ui" in r.text
    record("GET /docs (Swagger UI)", "200 + swagger-ui", r.status_code, ok)

    r = await client.get("/openapi.json")
    schema = r.json() if r.text else {}
    paths = schema.get("paths", {})
    ok = (r.status_code == 200 and "/query/" in paths
          and "post" in paths.get("/query/", {}))
    record("OpenAPI exposes POST /query/", "200 with path+method", r.status_code, ok)

    # ------------------------------------------------------------------
    # 2. Authentication: two users
    # ------------------------------------------------------------------
    r = await client.post("/register", json={"username": "ua", "password": "pw12345"})
    ua_id = r.json().get("id") if r.status_code == 200 else None
    record("register ua", "200 {id}", r.status_code, isinstance(ua_id, int))

    r = await client.post("/register", json={"username": "ub", "password": "pw12345"})
    ub_id = r.json().get("id") if r.status_code == 200 else None
    record("register ub", "200 {id}", r.status_code, isinstance(ub_id, int))

    r = await client.post(
        "/token",
        data={"username": "ua", "password": "pw12345"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    ua_headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r = await client.post(
        "/token",
        data={"username": "ub", "password": "pw12345"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    ub_headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    # ------------------------------------------------------------------
    # 3. Uploads: one small dataset per user + one big shared-shape dataset
    # ------------------------------------------------------------------
    a_small = "name,age,city\nAlice,28,London\nBob,32,Paris\n".encode()
    r = await client.post(
        "/datasets/upload", files={"file": ("a.csv", a_small, "text/csv")},
        headers=ua_headers,
    )
    a_id = r.json().get("id") if r.status_code == 200 else None
    record("upload a.csv (ua)", "200 rows=2 cols=3", r.status_code,
           r.status_code == 200 and r.json().get("rows") == 2
           and r.json().get("columns") == 3)

    b_small = "x,y,z\n1,2,3\n4,5,6\n7,8,9\n".encode()
    r = await client.post(
        "/datasets/upload", files={"file": ("b.csv", b_small, "text/csv")},
        headers=ub_headers,
    )
    b_id = r.json().get("id") if r.status_code == 200 else None
    record("upload b.csv (ub)", "200 rows=3 cols=3", r.status_code,
           r.status_code == 200 and r.json().get("rows") == 3
           and r.json().get("columns") == 3)

    BIG_ROWS, BIG_COLS = 120_000, 8
    big_csv = _matrix_csv(BIG_ROWS, BIG_COLS)
    r = await client.post(
        "/datasets/upload", files={"file": ("big.csv", big_csv, "text/csv")},
        headers=ua_headers,
    )
    big_id = r.json().get("id") if r.status_code == 200 else None
    record(f"upload big.csv ({len(big_csv)//1024} KiB)", f"200 rows={BIG_ROWS}",
           r.status_code,
           r.status_code == 200 and r.json().get("rows") == BIG_ROWS)

    await _run_query_and_error_checks(client, ua_id, ub_id, a_id, b_id,
                                      big_id, BIG_ROWS, BIG_COLS,
                                      ua_headers, ub_headers)


async def _run_query_and_error_checks(client, ua_id, ub_id, a_id, b_id,
                                      big_id, BIG_ROWS, BIG_COLS,
                                      ua_headers, ub_headers):
    big_cols = BIG_COLS
# ------------------------------------------------------------------
    # 4. Normal queries -- the query forms the system supports
    # ------------------------------------------------------------------
    r = await client.post("/query/", json={"text": "what were total sales?"},
                          headers=ua_headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200
          and body.get("answer") == "General query: what were total sales?")
    record("general query (no dataset)", "200 general answer", r.status_code, ok)

    r = await client.post("/query/", json={"text": "SELECT * FROM anything"},
                          headers=ua_headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200
          and body.get("answer") == "General query: SELECT * FROM anything")
    record("query without dataset (SQL-like text)", "200 general answer",
           r.status_code, ok)

    r = await client.post(
        "/query/", json={"text": "summarize", "dataset_id": a_id}, headers=ua_headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("rows_analyzed") == 2
          and body.get("columns_analyzed") == 3 and body.get("dataset_id") == a_id
          and "summarize" in body.get("answer", ""))
    record("dataset query a.csv", "200 rows=2 cols=3", r.status_code, ok)

    r = await client.post(
        "/query/", json={"text": "   padded   ", "dataset_id": a_id}, headers=ua_headers)
    body = r.json() if r.text else {}
    ok = r.status_code == 200 and "padded" in body.get("answer", "")
    record("query text trimmed before use", "200 trimmed", r.status_code, ok)

    r = await client.post(
        "/query/", json={"text": "big", "dataset_id": big_id}, headers=ua_headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("rows_analyzed") == BIG_ROWS
          and body.get("columns_analyzed") == big_cols)
    record("dataset query big.csv", "200 counts match upload", r.status_code, ok)

    # ------------------------------------------------------------------
    # 5. Error handling (incl. processing failure and database failure)
    # ------------------------------------------------------------------
    r = await client.post("/query/", json={"text": "   "}, headers=ua_headers)
    ok = (r.status_code == 400
          and "cannot be empty" in r.json().get("detail", "").lower())
    record("empty (whitespace) query", "400 empty", r.status_code, ok)

    r = await client.post("/query/", json={}, headers=ua_headers)
    record("missing text field", "422", r.status_code, r.status_code == 422)

    r = await client.post("/query/", json={"text": ""}, headers=ua_headers)
    record("empty string text", "422", r.status_code, r.status_code == 422)

    r = await client.post("/query/", json={"text": 42}, headers=ua_headers)
    record("non-string text", "422", r.status_code, r.status_code == 422)

    r = await client.post(
        "/query/", content="this is not json",
        headers={"Content-Type": "application/json",
                 "Authorization": ua_headers["Authorization"]},
    )
    record("invalid JSON body", "422", r.status_code, r.status_code == 422)

    r = await client.post("/query/", json={"text": "hello"})
    record("unauthorized query", "401", r.status_code, r.status_code == 401)

    r = await client.post("/query/", json={"text": "hello", "dataset_id": 99999999},
                          headers=ua_headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("invalid dataset id", "404", r.status_code, ok)

    r = await client.post("/query/", json={"text": "steal", "dataset_id": a_id},
                          headers=ub_headers)
    ok = r.status_code == 403 and "Not authorized" in r.json().get("detail", "")
    record("cross-user dataset query blocked", "403", r.status_code, ok)

    # Missing file -> 404
    session = SessionLocal()
    try:
        rec = session.query(models.Dataset).filter(models.Dataset.id == a_id).first()
        a_path = pathlib.Path(rec.file_path)
    finally:
        session.close()
    os.remove(a_path)
    r = await client.post("/query/", json={"text": "hi", "dataset_id": a_id},
                          headers=ua_headers)
    ok = (r.status_code == 404
          and "file not found" in r.json().get("detail", "").lower())
    record("query on dataset whose file was deleted", "404 file not found",
           r.status_code, ok)
# Corrupted file -> processing failure (500)
    corrupt = _TMP_PATH / "corrupt.csv"
    corrupt.write_bytes(b"\x00\x01\x02\xff" * 64)
    session = SessionLocal()
    try:
        rec = models.Dataset(
            original_filename="corrupt.csv",
            stored_filename="direct_corrupt.csv",
            file_path=str(corrupt),
            file_type="text/csv",
            file_size=0.0,
            rows=1,
            columns=1,
            owner_id=ua_id,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        corrupt_id = rec.id
    finally:
        session.close()
    r = await client.post("/query/", json={"text": "hi", "dataset_id": corrupt_id},
                          headers=ua_headers)
    ok = (r.status_code == 500
          and "could not process dataset file" in r.json().get("detail", "").lower())
    record("corrupted file -> processing failure", "500", r.status_code, ok)

    # Database failure: simulated outage inside the worker thread -> clean 500.
    _orig_lookup = query_module.crud.get_dataset_by_id
    def failing_lookup(db, dataset_id_):
        raise RuntimeError("simulated database outage")
    query_module.crud.get_dataset_by_id = failing_lookup
    try:
        r = await client.post(
            "/query/", json={"text": "hello", "dataset_id": big_id}, headers=ua_headers)
        ok = (r.status_code == 500
              and "database".lower() in r.json().get("detail", "").lower())
        record("database failure -> clean 500", "500", r.status_code, ok)
    finally:
        query_module.crud.get_dataset_by_id = _orig_lookup

    # ------------------------------------------------------------------
    # 6. Data correctness / isolation (no shared state between users)
    # ------------------------------------------------------------------
    # ub queries their own dataset right after ua ran a BIG query; results
    # must still be ub's rows/columns, not the last-seen values.
    r = await client.post("/query/", json={"text": "own", "dataset_id": b_id},
                          headers=ub_headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("rows_analyzed") == 3
          and body.get("columns_analyzed") == 3)
    record("isolation: ub sees own counts after ua big query", "200 rows=3",
           r.status_code, ok)

    # ------------------------------------------------------------------
    # 7. Concurrency & performance (non-blocking request handling)
    # ------------------------------------------------------------------
    # 7a. The event loop must stay responsive while a heavy parse is in flight:
    #     requests are RECEIVED and SERVED while another query is processed.
    import time as _time_mod
    _orig_load = query_module.load_dataset_file
    def _slow_load_dataset_file(file_path, filename):
        _time_mod.sleep(1.4)  # simulate genuinely heavy processing
        return _orig_load(file_path, filename)
    query_module.load_dataset_file = _slow_load_dataset_file
    try:
        t0 = time.perf_counter()
        q_task = asyncio.create_task(client.post(
            "/query/", json={"text": "slow", "dataset_id": big_id},
            headers=ua_headers))
        await asyncio.sleep(0.4)  # let the query land on the worker thread
        p0 = time.perf_counter()
        polls = await asyncio.gather(*[client.get("/health") for _ in range(8)])
        lst = await client.get("/datasets/", headers=ua_headers)
        p1 = time.perf_counter()
        q = await q_task
        t1 = time.perf_counter()
        polls_ok = (all(p.status_code == 200 for p in polls)
                    and lst.status_code == 200
                    and q.status_code == 200
                    and q.json().get("rows_analyzed") == BIG_ROWS)
        ok = polls_ok and (p1 - p0) < 1.0
        record(f"responsiveness during 1.4s parse: 8 polls + list in "
               f"{(p1 - p0)*1000:.0f} ms (query total {(t1 - t0):.2f}s)",
               "all 200, polls<1s", "ok" if ok else "FAIL", ok)
    finally:
        query_module.load_dataset_file = _orig_load

    # 7b. Single-query wall-clock (average of 3) on the same dataset.
    query_module.load_dataset_file = _orig_load
    await client.post("/query/", json={"text": "warmup", "dataset_id": big_id},
                     headers=ua_headers)  # warm the file cache
    t0 = time.perf_counter()
    n_single = 3
    for i in range(n_single):
        r = await client.post("/query/", json={"text": f"single {i}",
                               "dataset_id": big_id}, headers=ua_headers)
        assert r.status_code == 200, r.text
    single_total = time.perf_counter() - t0
    record(f"{n_single} sequential queries on big.csv",
           f"{single_total:.2f}s total ({single_total/n_single:.2f}s each)",
           f"{single_total:.2f}s", True)

    # 7c. Multiple simultaneous queries, same dataset.
    t0 = time.perf_counter()
    rs = await asyncio.gather(*[
        client.post("/query/", json={"text": f"c{i}", "dataset_id": big_id},
                    headers=ua_headers)
        for i in range(8)
    ])
    conc_total = time.perf_counter() - t0
    good = [r for r in rs if r.status_code == 200
            and r.json().get("rows_analyzed") == BIG_ROWS
            and r.json().get("columns_analyzed") == BIG_COLS]
    ok = len(good) == 8
    record(f"8 concurrent queries on big.csv ({conc_total:.2f}s)",
           "all 200 + correct counts", f"{len(good)}/8", ok)

    # 7d. Mixed-user concurrent queries: no shared-state bleed between
    #.    different users/datasets running at the same time.
    rs = await asyncio.gather(*[
        client.post("/query/", json={"text": f"ua{i}", "dataset_id": big_id},
                    headers=ua_headers) for i in range(4)
    ] + [
        client.post("/query/", json={"text": f"ub{i}", "dataset_id": b_id},
                    headers=ub_headers) for i in range(4)
    ])
    assert len(rs) == 8
    a_ok = [r for r in rs[:4] if r.status_code == 200
            and r.json().get("rows_analyzed") == BIG_ROWS]
    b_ok = [r for r in rs[4:] if r.status_code == 200
            and r.json().get("rows_analyzed") == 3]
    ok = len(a_ok) == 4 and len(b_ok) == 4
    record("4 ua queries + 4 ub queries concurrently", "all correct per user",
           f"{len(a_ok)}/{4} ua, {len(b_ok)}/{4} ub", ok)

    # 7e. Final health after all load.
    r = await client.get("/health")
    record("health after all load", "200", r.status_code, r.status_code == 200)


async def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT,
                            log_level="warning", timeout_graceful_shutdown=2)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)

    if not server.started:
        print("FATAL: uvicorn did not start")
        sys.exit(2)

    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=60.0) as client:
            await run_suite(client)
    finally:
        server.should_exit = True
        await task
        engine.dispose()

    failed = [row for row in REPORT if not row[3]]
    total = len(REPORT)
    print("\n" + "=" * 68)
    print(f"QUERY INTEGRATION VALIDATION: {total - len(failed)}/{total} passed")
    if failed:
        print("FAILED:")
        for name, exp, act, ok, note in failed:
            print(f"  - {name}: expected {exp}, got {act} {note}")
    else:
        print("ALL INTEGRATION CHECKS PASSED")
    print("=" * 68)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        _TMP.cleanup()