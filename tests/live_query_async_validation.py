"""Live-server validation of the Async Query Processing (Part 2) for OmniBrain.

Starts the real FastAPI application with uvicorn on a private port and drives
it over real HTTP with httpx:

- normal / empty / unauthorized queries
- large-dataset query (real pandas parse) -- results must stay correct
- the event loop must remain responsive (raced /health polls) while a large
  dataset is being parsed by the query endpoint
- 8 concurrent dataset queries all succeed
- invalid dataset (404), cross-user access (403), missing file (404),
  corrupted file (500)
- the server stays responsive after all the load

Run from the repository root:

    .venv\\Scripts\\python.exe tests\\live_query_async_validation.py
"""

import asyncio
import os
import pathlib
import sys
import tempfile
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_TMP = tempfile.TemporaryDirectory(prefix="omnibrain_query_live_")
_TMP_PATH = pathlib.Path(_TMP.name)

# Environment MUST be set before importing main / database / config.
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_PATH / 'live.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP_PATH / "uploads")

import httpx  # noqa: E402
import uvicorn  # noqa: E402

import models  # noqa: E402
from database import SessionLocal, engine  # noqa: E402
from main import app  # noqa: E402

PORT = 8722
BASE = f"http://127.0.0.1:{PORT}"

REPORT = []  # (name, expected, actual_status, ok, note)


def record(name: str, expected: str, actual, ok: bool, note: str = ""):
    REPORT.append((name, expected, str(actual), ok, note))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: expected {expected}, got {actual} {note}")


def _big_csv(rows: int, cols: int) -> bytes:
    """Deterministic int matrix CSV kept compact enough to build quickly."""
    header = ",".join(f"c{c}" for c in range(cols))
    lines = [
        ",".join(str((r * 131 + c * 17) % 1000) for c in range(cols))
        for r in range(rows)
    ]
    return f"{header}\n".encode() + "\n".join(lines).encode()


def _insert_dataset_row(
    *,
    owner_id: int,
    original_filename: str,
    file_path: pathlib.Path,
    rows: int,
    columns: int,
) -> int:
    """Insert a dataset record directly (for missing/corrupt file scenarios)."""
    session = SessionLocal()
    try:
        rec = models.Dataset(
            original_filename=original_filename,
            stored_filename=f"direct_{uuid.uuid4().hex}_{original_filename}",
            file_path=str(file_path),
            file_type="text/csv",
            file_size=0.0,
            rows=rows,
            columns=columns,
            owner_id=owner_id,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec.id
    finally:
        session.close()


async def run_suite(client: httpx.AsyncClient):
    # ------------------------------------------------------------------
    # 1. Register users and authenticate
    # ------------------------------------------------------------------
    r = await client.post("/register", json={"username": "quser", "password": "pw12345"})
    owner_id = r.json().get("id") if r.status_code == 200 else None
    record("POST /register (quser)", "200 {id}", r.status_code,
           r.status_code == 200 and isinstance(owner_id, int))

    r = await client.post("/token",
                          data={"username": "quser", "password": "pw12345"},
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
    ok = r.status_code == 200 and bool(r.json().get("access_token"))
    record("POST /token", "200 bearer token", r.status_code, ok)
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    await client.post("/register", json={"username": "other", "password": "pw12345"})
    r = await client.post("/token",
                          data={"username": "other", "password": "pw12345"},
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
    other_headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    # ------------------------------------------------------------------
    # 2. Baseline query behaviour (no dataset)
    # ------------------------------------------------------------------
    r = await client.post("/query/", json={"text": "hello"}, headers=headers)
    body = r.json() if r.text else {}
    ok = r.status_code == 200 and body.get("answer") == "General query: hello"
    record("POST /query/ (valid, no dataset)", "200 general query", r.status_code, ok)

    r = await client.post("/query/", json={"text": "   "}, headers=headers)
    ok = r.status_code == 400 and "cannot be empty" in r.json().get("detail", "").lower()
    record("POST /query/ (empty text)", "400", r.status_code, ok)

    r = await client.post("/query/", json={"text": "hello"})
    record("POST /query/ (no auth)", "401", r.status_code, r.status_code == 401)

    # ------------------------------------------------------------------
    # 3. Upload a small and a large dataset
    # ------------------------------------------------------------------
    small = ("a,b\n1,2\n3,4\n5,6\n7,8\n").encode()
    r = await client.post("/datasets/upload",
                          files={"file": ("small.csv", small, "text/csv")},
                          headers=headers)
    small_id = r.json().get("id") if r.status_code == 200 else None
    record("POST /datasets/upload (small.csv)", "200 {id}",
           r.status_code, r.status_code == 200 and small_id is not None)

    BIG_ROWS, BIG_COLS = 200_000, 10
    big_csv = _big_csv(BIG_ROWS, BIG_COLS)
    r = await client.post("/datasets/upload",
                          files={"file": ("big.csv", big_csv, "text/csv")},
                          headers=headers)
    big_id = r.json().get("id") if r.status_code == 200 else None
    record(f"POST /datasets/upload (big.csv, {len(big_csv)//1024} KiB)",
           f"200 rows={BIG_ROWS}", r.status_code,
           r.status_code == 200 and r.json().get("rows") == BIG_ROWS)

    # ------------------------------------------------------------------
    # 4. Dataset query correctness (small + big)
    # ------------------------------------------------------------------
    r = await client.post("/query/", json={"text": "summarize", "dataset_id": small_id},
                          headers=headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("rows_analyzed") == 4
          and body.get("columns_analyzed") == 2 and "summarize" in body.get("answer", ""))
    record("POST /query/ (owned small dataset)", "200 rows=4 cols=2", r.status_code, ok)

    r = await client.post("/query/", json={"text": "big", "dataset_id": big_id},
                          headers=headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("rows_analyzed") == BIG_ROWS
          and body.get("columns_analyzed") == BIG_COLS)
    record("POST /query/ (large dataset, correct counts)",
           f"200 rows={BIG_ROWS} cols={BIG_COLS}", r.status_code, ok)

    # ------------------------------------------------------------------
    # 5. Failure cases for dataset queries
    # ------------------------------------------------------------------
    r = await client.post("/query/", json={"text": "hi", "dataset_id": 9999999},
                          headers=headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("POST /query/ (invalid dataset id)", "404", r.status_code, ok)

    r = await client.post("/query/", json={"text": "steal", "dataset_id": small_id},
                          headers=other_headers)
    ok = r.status_code == 403 and "Not authorized" in r.json().get("detail", "")
    record("POST /query/ (cross-user access)", "403", r.status_code, ok)


    # Missing file: delete the stored file for the small dataset.
    session = SessionLocal()
    try:
        small_rec = session.query(models.Dataset).filter(models.Dataset.id == small_id).first()
        small_path = pathlib.Path(small_rec.file_path)
    finally:
        session.close()
    os.remove(small_path)
    r = await client.post("/query/", json={"text": "hi", "dataset_id": small_id},
                          headers=headers)
    ok = r.status_code == 404 and "file not found" in r.json().get("detail", "").lower()
    record("POST /query/ (dataset file deleted)", "404 file not found", r.status_code, ok)

    # Corrupted file: record pointing at binary-garbage CSV.
    corrupt_path = _TMP_PATH / "corrupt.csv"
    corrupt_path.write_bytes(b"\x00\x01\x02\xff" * 64)
    corrupt_id = _insert_dataset_row(
        owner_id=owner_id, original_filename="corrupt.csv",
        file_path=corrupt_path, rows=1, columns=1)
    r = await client.post("/query/", json={"text": "hi", "dataset_id": corrupt_id},
                          headers=headers)
    ok = (r.status_code == 500
          and "could not process dataset file" in r.json().get("detail", "").lower())
    record("POST /query/ (corrupted dataset file)", "500", r.status_code, ok)

    # ------------------------------------------------------------------
    # 6. Event-loop responsiveness while a slow dataset query is being
    #    processed.  We simulate a genuinely heavy parse (2s) INSIDE the
    #    running uvicorn process by wrapping query.load_dataset_file, then
    #    race 8 /health polls against it. With the thread-pool offload the
    #    polls must complete in well under a second; a blocking endpoint
    #    would stall the single uvicorn event loop for ~2s and fail this.
    # ------------------------------------------------------------------
    import query as _query_mod
    import time as _time_mod

    _orig_load_dataset_file = _query_mod.load_dataset_file

    def _slow_load_dataset_file(file_path, filename):
        _time_mod.sleep(2.0)  # simulate genuinely heavy processing
        return _orig_load_dataset_file(file_path, filename)

    _query_mod.load_dataset_file = _slow_load_dataset_file
    try:
        async def slow_query_and_polls():
            t0 = asyncio.get_event_loop().time()
            q_task = asyncio.create_task(client.post(
                "/query/", json={"text": "slow", "dataset_id": big_id},
                headers=headers))
            await asyncio.sleep(0.4)  # let the query land on the loop
            hp0 = asyncio.get_event_loop().time()
            polls = await asyncio.gather(*[client.get("/health") for _ in range(8)])
            hp1 = asyncio.get_event_loop().time()
            q = await q_task
            t1 = asyncio.get_event_loop().time()
            return q, polls, hp1 - hp0, t1 - t0

        q, polls, polls_elapsed, query_total = await slow_query_and_polls()
        all_200 = all(p.status_code == 200 for p in polls)
        q_ok = (q.status_code == 200 and q.json().get("rows_analyzed") == BIG_ROWS)
        ok = all_200 and q_ok and polls_elapsed < 1.0
        record(f"slow parse (2s) + 8 /health polls (polls {polls_elapsed:.2f}s, query {query_total:.2f}s)",
               "query 200 + polls 200 + polls<1s", q.status_code, ok)
    finally:
        _query_mod.load_dataset_file = _orig_load_dataset_file

    # ------------------------------------------------------------------
    # 7. Multiple concurrent large dataset queries
    # ------------------------------------------------------------------
    t0 = asyncio.get_event_loop().time()
    rs = await asyncio.gather(*[
        client.post("/query/", json={"text": f"c{i}", "dataset_id": big_id},
                    headers=headers)
        for i in range(8)
    ])
    t1 = asyncio.get_event_loop().time()
    good = [r for r in rs if r.status_code == 200
            and r.json().get("rows_analyzed") == BIG_ROWS]
    ok = len(good) == 8
    record(f"8 concurrent big queries ({t1 - t0:.2f}s)", "all 200 + correct counts",
           f"{len(good)}/8", ok)

    # ------------------------------------------------------------------
    # 8. Final responsiveness check
    # ------------------------------------------------------------------
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
        async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as client:
            await run_suite(client)
    finally:
        server.should_exit = True
        await task
        engine.dispose()

    failed = [row for row in REPORT if not row[3]]
    total = len(REPORT)
    print("\n" + "=" * 68)
    print(f"LIVE QUERY ASYNC VALIDATION: {total - len(failed)}/{total} passed")
    if failed:
        print("FAILED:")
        for name, exp, act, ok, note in failed:
            print(f"  - {name}: expected {exp}, got {act} {note}")
    else:
        print("ALL LIVE QUERY TESTS PASSED")
    print("=" * 68)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        _TMP.cleanup()
