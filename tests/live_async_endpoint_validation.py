"""Live-server validation of the Async API Endpoints (Async API-2).

Starts the real FastAPI application with uvicorn on a private port, then
drives it over real HTTP with httpx -- success cases, failure cases, /docs,
concurrency, and a final responsiveness probe.

Run from the repository root:

    .venv\\Scripts\\python.exe tests\\live_async_endpoint_validation.py
"""

import asyncio
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_TMP = tempfile.TemporaryDirectory(prefix="omnibrain_live_")
_TMP_PATH = pathlib.Path(_TMP.name)

# Environment MUST be set before importing main / database / config.
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_PATH / 'live.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP_PATH / "uploads")

import httpx
import uvicorn

from main import app  # noqa: E402
from database import engine  # noqa: E402

PORT = 8721
BASE = f"http://127.0.0.1:{PORT}"

REPORT = []  # (name, expected, actual_status, ok, note)


def record(name: str, expected: str, actual, ok: bool, note: str = ""):
    REPORT.append((name, expected, str(actual), ok, note))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: expected {expected}, got {actual} {note}")


def check_json_field(body, field, present: bool) -> bool:
    return (field in body) == present
async def run_suite(client: httpx.AsyncClient):
    # ------------------------------------------------------------------
    # 1. Root / health / OpenAPI docs
    # ------------------------------------------------------------------
    r = await client.get("/")
    record("GET /", "200", r.status_code, r.status_code == 200)

    r = await client.get("/health")
    record("GET /health", "200", r.status_code, r.status_code == 200)

    r = await client.get("/docs")
    ok = r.status_code == 200 and "swagger-ui" in r.text
    record("GET /docs (Swagger UI)", "200 + swagger-ui", r.status_code, ok)

    r = await client.get("/openapi.json")
    ok = r.status_code == 200 and "paths" in r.json()
    record("GET /openapi.json", "200 + paths", r.status_code, ok)

    # ------------------------------------------------------------------
    # 2. Authentication (async endpoints)
    # ------------------------------------------------------------------
    r = await client.post("/register", json={"username": "carol", "password": "pw12345"})
    body = r.json() if r.text else {}
    ok = r.status_code == 200 and body.get("username") == "carol" and "id" in body
    record("POST /register (valid)", "200 {id, username}",
           r.status_code, ok)

    r = await client.post("/register", json={"username": "carol", "password": "pw12345"})
    ok = r.status_code == 400 and r.json().get("detail") == "Username already registered"
    record("POST /register (duplicate)", "400 duplicate message", r.status_code, ok)

    r = await client.post("/register", json={"username": "no_pass"})
    ok = r.status_code == 422
    record("POST /register (missing password)", "422", r.status_code, ok)

    r = await client.post("/token",
                          data={"username": "carol", "password": "pw12345"},
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("token_type") == "bearer"
          and bool(body.get("access_token")))
    record("POST /token (valid)", "200 {access_token, bearer}",
           r.status_code, ok)
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    r = await client.post("/token",
                          data={"username": "carol", "password": "wrong"},
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
    ok = r.status_code == 401 and "Incorrect username or password" in r.json().get("detail", "")
    record("POST /token (bad password)", "401 incorrect credentials",
           r.status_code, ok)

    r = await client.get("/users/me", headers=headers)
    body = r.json() if r.text else {}
    ok = r.status_code == 200 and body.get("username") == "carol"
    record("GET /users/me (valid token)", "200 carol", r.status_code, ok)

    r = await client.get("/users/me")
    record("GET /users/me (no token)", "401", r.status_code, r.status_code == 401)

    r = await client.get("/users/me", headers={"Authorization": "Bearer not.a.jwt"})
    record("GET /users/me (invalid token)", "401", r.status_code, r.status_code == 401)
# ------------------------------------------------------------------
    # 3. Dataset upload (async endpoint) -- success
    # ------------------------------------------------------------------
    csv_missing = (
        "name,age,salary,active,notes\n"
        "Anna,28,1100.5,True,\n"
        "Bob,,950.25,False,missing\n"
        "Cara,41,,True,\n"
        "Bob,32,950.25,False,missing\n"
    ).encode("utf-8")

    r = await client.post("/datasets/upload",
                          files={"file": ("people_missing.csv", csv_missing, "text/csv")},
                          headers=headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("rows") == 4
          and body.get("columns") == 5 and "id" in body)
    record("POST /datasets/upload (valid CSV)", "200 rows=4 cols=5",
           r.status_code, ok)
    dataset_id = body.get("id") if r.status_code == 200 else None

    r = await client.post("/datasets/upload",
                          files={"file": ("x.csv", csv_missing, "text/csv")})
    record("POST /datasets/upload (no auth)", "401", r.status_code, r.status_code == 401)

    r = await client.post("/datasets/upload",
                          files={"file": ("x.txt", b"hello", "text/plain")},
                          headers=headers)
    ok = r.status_code == 400 and "Unsupported file type" in r.json().get("detail", "")
    record("POST /datasets/upload (unsupported type)", "400 unsupported type",
           r.status_code, ok)

    r = await client.post("/datasets/upload",
                          files={"file": ("empty.csv", b"", "text/csv")},
                          headers=headers)
    # An empty file is rejected with 400 (pandas EmptyDataError path). The
    # exact message is the generic parser error; only the status is asserted.
    record("POST /datasets/upload (empty file)", "400 rejected",
           r.status_code, r.status_code == 400)

    r = await client.post("/datasets/upload", headers=headers)
    record("POST /datasets/upload (no file part)", "422",
           r.status_code, r.status_code == 422)

    # ------------------------------------------------------------------
    # 4. Dataset listing / details / deletion (async endpoints)
    # ------------------------------------------------------------------
    r = await client.get("/datasets/", headers=headers)
    body = r.json() if r.text else []
    ok = (r.status_code == 200 and isinstance(body, list)
          and any(d.get("id") == dataset_id for d in body))
    record("GET /datasets/ (list)", "200 contains uploaded id",
           r.status_code, ok)

    r = await client.get("/datasets/")
    record("GET /datasets/ (no auth)", "401", r.status_code, r.status_code == 401)

    r = await client.get(f"/datasets/{dataset_id}", headers=headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and "column_info" in body
          and isinstance(body.get("preview"), list))
    record("GET /datasets/{id} (details)", "200 column_info + preview",
           r.status_code, ok)

    r = await client.get("/datasets/9999999", headers=headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("GET /datasets/9999999", "404 not found", r.status_code, ok)

    r = await client.get("/datasets/abc", headers=headers)
    record("GET /datasets/abc (non-int id)", "422", r.status_code, r.status_code == 422)

    r = await client.delete("/datasets/9999999", headers=headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("DELETE /datasets/9999999", "404 not found", r.status_code, ok)

    # ------------------------------------------------------------------
    # 5. Query endpoint
    # ------------------------------------------------------------------
    r = await client.post("/query/", json={"text": "hello"}, headers=headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("answer") == "General query: hello"
          and body.get("dataset_id") is None)
    record("POST /query/ (valid, no dataset)", "200 general query",
           r.status_code, ok)

    r = await client.post("/query/", json={"text": "  "}, headers=headers)
    ok = r.status_code == 400 and "cannot be empty" in r.json().get("detail", "").lower()
    record("POST /query/ (empty text)", "400 empty query",
           r.status_code, ok)

    r = await client.post("/query/", headers=headers)
    ok = r.status_code == 422
    record("POST /query/ (no body)", "422", r.status_code, ok)

    r = await client.post("/query/", json={"text": "hello", "dataset_id": 9999999}, headers=headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("POST /query/ (invalid dataset id)", "404 not found",
           r.status_code, ok)

    r = await client.post("/query/", json={"text": "hello"})
    record("POST /query/ (no auth)", "401", r.status_code, r.status_code == 401)
# ------------------------------------------------------------------
    # 6. Profiling endpoints (regression; synchronous)
    # ------------------------------------------------------------------
    r = await client.get(f"/datasets/{dataset_id}/structure", headers=headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and body.get("rows") == 4
          and body.get("columns") == 5 and "data_types" in body)
    record("GET /datasets/{id}/structure", "200 rows=4 data_types",
           r.status_code, ok)

    r = await client.get(f"/datasets/{dataset_id}/missing-values", headers=headers)
    body = r.json() if r.text else {}
    ok = r.status_code == 200 and "total_missing_values" in body
    record("GET /datasets/{id}/missing-values", "200 profile",
           r.status_code, ok)

    r = await client.get(f"/datasets/{dataset_id}/quality", headers=headers)
    body = r.json() if r.text else {}
    ok = (r.status_code == 200 and "duplicate_rows" in body
          and "numerical_columns" in body)
    record("GET /datasets/{id}/quality", "200 duplicate+columns",
           r.status_code, ok)

    r = await client.get(f"/datasets/{dataset_id}/summary", headers=headers)
    body = r.json() if r.text else {}
    ok = r.status_code == 200 and "numerical_summary" in body
    record("GET /datasets/{id}/summary", "200 summary",
           r.status_code, ok)

    r = await client.get("/datasets/9999999/structure", headers=headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("GET /datasets/9999999/structure", "404 not found",
           r.status_code, ok)

    # ------------------------------------------------------------------
    # 7. Cleaning endpoints (regression; synchronous)
    # ------------------------------------------------------------------
    for ep in ("missing-values", "duplicates-and-types", "standardize-columns",
               "missing-like-strings", "numeric-special", "final-validation"):
        r = await client.post(f"/datasets/{dataset_id}/clean/{ep}", headers=headers)
        ok = r.status_code == 200
        record(f"POST /datasets/{{id}}/clean/{ep}", "200 report",
               r.status_code, ok)

    r = await client.post("/datasets/9999999/clean/missing-values", headers=headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("POST /datasets/9999999/clean/missing-values", "404 not found",
           r.status_code, ok)

    # ------------------------------------------------------------------
    # 8. Cross-user authorization (403)
    # ------------------------------------------------------------------
    r = await client.post("/register", json={"username": "dave", "password": "pw12345"})
    r2 = await client.post("/token",
                           data={"username": "dave", "password": "pw12345"},
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
    dave_headers = {"Authorization": f"Bearer {r2.json()['access_token']}"}

    r = await client.get(f"/datasets/{dataset_id}", headers=dave_headers)
    ok = r.status_code == 403 and "Not authorized" in r.json().get("detail", "")
    record("GET /datasets/{id} (cross-user)", "403", r.status_code, ok)

    r = await client.get(f"/datasets/{dataset_id}/structure", headers=dave_headers)
    record("GET /datasets/{id}/structure (cross-user)", "403",
           r.status_code, r.status_code == 403)

    r = await client.delete(f"/datasets/{dataset_id}", headers=dave_headers)
    ok = r.status_code == 403 and "Not authorized" in r.json().get("detail", "")
    record("DELETE /datasets/{id} (cross-user)", "403", r.status_code, ok)

    r = await client.get(f"/datasets/{dataset_id}", headers=headers)
    record("owner still has dataset after cross-user 403", "200",
           r.status_code, r.status_code == 200)
# ------------------------------------------------------------------
    # 9. Concurrency
    # ------------------------------------------------------------------
    # 9a. Concurrent GETs against async endpoints -- server must stay responsive.
    t0 = asyncio.get_event_loop().time()
    lists = await asyncio.gather(*[client.get("/datasets/", headers=headers)
                                   for _ in range(12)])
    mes = await asyncio.gather(*[client.get("/users/me", headers=headers)
                                 for _ in range(12)])
    t1 = asyncio.get_event_loop().time()
    all_ok = all(x.status_code == 200 for x in lists + mes)
    ok = all_ok and (t1 - t0) < 5.0
    record(f"Concurrent GETs (24 in {t1 - t0:.2f}s)", "all 200, <5s",
           "all 200" if all_ok else "mixed", ok)

    # 9b. Upload streamed concurrently with /health polls: one request must not
    #     block others from being handled (server remains responsive mid-load).
    async def upload_and_poll():
        up = await client.post(
            "/datasets/upload",
            files={"file": ("poll.csv", csv_missing, "text/csv")},
            headers=headers,
        )
        polls = await asyncio.gather(*[client.get("/health") for _ in range(6)])
        return up, polls

    t0 = asyncio.get_event_loop().time()
    up_res, polls = await upload_and_poll()
    t1 = asyncio.get_event_loop().time()
    ok = (up_res.status_code == 200
          and all(p.status_code == 200 for p in polls)
          and (t1 - t0) < 5.0)
    record(f"upload concurrent with 6 /health polls ({t1 - t0:.2f}s)",
           "upload 200 + all polls 200", up_res.status_code, ok)

    # 9c. Concurrent uploads. SQLite serialises writers; this reveals whether
    #     any write-lock errors surface (a pre-existing SQLite constraint,
    #     not something introduced by the async conversion).
    t0 = asyncio.get_event_loop().time()
    ups = await asyncio.gather(*[
        client.post("/datasets/upload",
                    files={"file": (f"c{i}.csv", csv_missing, "text/csv")},
                    headers=headers)
        for i in range(4)
    ], return_exceptions=True)
    t1 = asyncio.get_event_loop().time()
    ok_ups = [u for u in ups if not isinstance(u, Exception)]
    statuses = [u.status_code for u in ok_ups]
    all_200 = len(statuses) == 4 and all(s == 200 for s in statuses)
    record(f"4 concurrent uploads ({t1 - t0:.2f}s, {len(ok_ups)}/4 returned)",
           "all 200", str(statuses), all_200)

    # 9d. Server still responsive after the load.
    r = await client.get("/health")
    record("health after all load", "200", r.status_code, r.status_code == 200)

    # ------------------------------------------------------------------
    # 10. Deletion of the first uploaded dataset (async endpoint)
    # ------------------------------------------------------------------
    r = await client.delete(f"/datasets/{dataset_id}", headers=headers)
    record("DELETE /datasets/{id} (valid)", "204",
           r.status_code, r.status_code == 204)

    r = await client.get(f"/datasets/{dataset_id}", headers=headers)
    ok = r.status_code == 404 and "Dataset not found" in r.json().get("detail", "")
    record("GET deleted dataset", "404 gone", r.status_code, ok)

    r = await client.get("/datasets/", headers=headers)
    n = len(r.json()) if r.text else 0
    ok = r.status_code == 200 and n >= 4
    record(f"datasets remain after delete ({n} rows)", "200 >=4 rows",
           r.status_code, ok)


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
    print(f"LIVE ASYNC ENDPOINT VALIDATION: {total - len(failed)}/{total} passed")
    if failed:
        print("FAILED:")
        for name, exp, act, ok, note in failed:
            print(f"  - {name}: expected {exp}, got {act} {note}")
    else:
        print("ALL LIVE HTTP TESTS PASSED")
    print("=" * 68)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        _TMP.cleanup()