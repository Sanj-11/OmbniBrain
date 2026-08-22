"""Live concurrency and non-blocking validation for OmniBrain.

Starts the real FastAPI application with uvicorn on a private port and drives
it over real HTTP with httpx:

- Multiple query requests fired at (approximately) the same time, same user.
- Queries mixed across different datasets and different users (isolation).
- Multiple upload requests fired simultaneously (including same-name collisions).
- One longer-running request while unrelated simple requests are sent
  (health probes, no-dataset query, dataset list) - both with an injected
  slow parse and with a real large-CSV parse.
- Result correctness and per-user/per-dataset isolation checks.
- Log capture so unhandled asyncio tasks and exceptions surface.

Run from the repository root:

    .venv\\Scripts\\python.exe tests\\concurrency_nonblocking_validation.py
"""

import asyncio
import logging
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_TMP = tempfile.TemporaryDirectory(prefix="omnibrain_concurrency_")
_TMP_PATH = pathlib.Path(_TMP.name)
_SERVER_LOG = _TMP_PATH / "server.log"

# Environment MUST be set before importing main / database / config.
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_PATH / 'concurrency.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP_PATH / "uploads")

import httpx  # noqa: E402
import uvicorn  # noqa: E402

import models  # noqa: E402
import query as query_module  # noqa: E402
from database import engine  # noqa: E402
from main import app  # noqa: E402

PORT = 8724
BASE = f"http://127.0.0.1:{PORT}"

# Capture uvicorn / asyncio / root logger output so "Task exception was never
# retrieved", unexpected tracebacks and SQLite errors can be checked after.
HANDLER = logging.FileHandler(_SERVER_LOG, encoding="utf-8")
HANDLER.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
logging.getLogger().addHandler(HANDLER)

REPORT = []  # (name, ok, note)
METRICS = []  # (name, measured-notes)


def record(name: str, ok: bool, note: str = ""):
    REPORT.append((name, ok, note))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" - {note}" if note else ""))


def metric(name: str, note: str):
    METRICS.append((name, note))
    print(f"  [metric] {name}: {note}")


def _matrix_csv(rows: int, cols: int) -> bytes:
    """Deterministic int matrix CSV, compact enough to build quickly."""
    header = ",".join(f"c{c}" for c in range(cols))
    lines = [
        ",".join(str((r * 131 + c * 17) % 1000) for c in range(cols))
        for r in range(rows)
    ]
    return f"{header}\n".encode() + "\n".join(lines).encode()


def _small_csv(rows: int) -> bytes:
    """Small 2-column CSV with ``rows`` rows and recognisable values."""
    header = "a,b\n"
    body = "\n".join(f"{i},{i * 10}" for i in range(1, rows + 1))
    return f"{header}{body}\n".encode()


async def _timed_gather(coros):
    """Run coroutines concurrently, each returns (index, response, elapsed)."""
    async def _wrapped(index, coro):
        t0 = time.perf_counter()
        resp = await coro
        return index, resp, time.perf_counter() - t0

    return await asyncio.gather(*[_wrapped(i, c) for i, c in enumerate(coros)])


def _overlap_factor(elapsed: list) -> float:
    """sum(durations)/max(duration): >1 means requests overlapped in execution."""
    wall = max(elapsed) if elapsed else 0.0
    return (sum(elapsed) / wall) if wall > 0 else 1.0


async def _register_and_login(client, username: str) -> dict:
    """Create a user through the real API and return auth headers."""
    r = await client.post("/register", json={"username": username, "password": "pw12345"})
    if r.status_code != 200:
        raise RuntimeError(f"register {username}: {r.text}")
    r = await client.post(
        "/token",
        data={"username": username, "password": "pw12345"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if r.status_code != 200:
        raise RuntimeError(f"login {username}: {r.text}")
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def run_suite(client: httpx.AsyncClient):
    # ------------------------------------------------------------------
    # 0. Server surface
    # ------------------------------------------------------------------
    r = await client.get("/health")
    record("server up: GET /health", r.status_code == 200)

    # ------------------------------------------------------------------
    # 1. Setup: two users, small datasets, one genuinely large dataset
    # ------------------------------------------------------------------
    ua_headers = await _register_and_login(client, "ua")
    ub_headers = await _register_and_login(client, "ub")

    BIG_ROWS, BIG_COLS = 120_000, 8
    big_csv = _matrix_csv(BIG_ROWS, BIG_COLS)

    r = await client.post(
        "/datasets/upload", files={"file": ("small_a.csv", _small_csv(3), "text/csv")},
        headers=ua_headers)
    small_a_id = r.json().get("id") if r.status_code == 200 else None
    record("upload small_a.csv (ua)", r.status_code == 200 and r.json().get("rows") == 3,
           f"id={small_a_id}")

    r = await client.post(
        "/datasets/upload", files={"file": ("small_b.csv", _small_csv(5), "text/csv")},
        headers=ub_headers)
    small_b_id = r.json().get("id") if r.status_code == 200 else None
    record("upload small_b.csv (ub)", r.status_code == 200 and r.json().get("rows") == 5,
           f"id={small_b_id}")

    r = await client.post(
        "/datasets/upload", files={"file": ("big.csv", big_csv, "text/csv")},
        headers=ua_headers)
    big_id = r.json().get("id") if r.status_code == 200 else None
    record(f"upload big.csv ({len(big_csv) // 1024} KiB, {BIG_ROWS} rows)",
           r.status_code == 200 and r.json().get("rows") == BIG_ROWS, f"id={big_id}")

    # ------------------------------------------------------------------
    # 2. Multiple query requests at approximately the same time.
    #    12 concurrent dataset queries (same user, same dataset).
    # ------------------------------------------------------------------
    coros = [client.post("/query/", json={"text": f"q{i}", "dataset_id": big_id},
                         headers=ua_headers) for i in range(12)]
    out = await _timed_gather(coros)
    rs = [resp for _, resp, _ in out]
    elapsed = [el for _, _, el in out]
    ok = all(r.status_code == 200 and r.json().get("rows_analyzed") == BIG_ROWS
             for r in rs)
    record("12 concurrent queries (ua, big.csv) - same user/same dataset",
           ok, f"{sum(1 for r in rs if r.status_code == 200)}/{len(rs)} ok")
    metric("12 concurrent queries wall",
           f"max={max(elapsed):.2f}s sum={sum(elapsed):.2f}s "
           f"overlap_f={_overlap_factor(elapsed):.2f}")

    # ------------------------------------------------------------------
    # 3. Concurrent queries across different datasets / different users.
    #    Every response must carry its own dataset's row count (isolation).
    # ------------------------------------------------------------------
    jobs = (
        [client.post("/query/", json={"text": f"mxa{i}", "dataset_id": big_id},
                     headers=ua_headers) for i in range(4)]
        + [client.post("/query/", json={"text": f"mxs{i}", "dataset_id": small_a_id},
                       headers=ua_headers) for i in range(4)]
        + [client.post("/query/", json={"text": f"mxb{i}", "dataset_id": small_b_id},
                       headers=ub_headers) for i in range(4)]
    )
    out = await _timed_gather(jobs)
    expected = [BIG_ROWS] * 4 + [3] * 4 + [5] * 4
    correct = sum(
        1 for (exp, (_, resp, _)) in zip(expected, out)
        if resp.status_code == 200 and resp.json().get("rows_analyzed") == exp
    )
    record("12 concurrent queries across 3 different datasets (ua/ub)",
           correct == 12, f"{correct}/12 datasets returned their own row counts")
    elapsed = [el for _, _, el in out]
    metric("cross-dataset concurrent queries",
           f"overlap_f={_overlap_factor(elapsed):.2f} max={max(elapsed):.2f}s")

    # ------------------------------------------------------------------
    # 4. Multiple upload requests at the same time (same user, same name).
    #    Unique stored names must not collide; every upload keeps its own
    #    content (row counts 5..10 mixed in one gather).
    # ------------------------------------------------------------------
    uploads = [
        client.post("/datasets/upload",
                    files={"file": ("same.csv", _small_csv(5 + i), "text/csv")},
                    headers=ua_headers)
        for i in range(6)
    ]
    out = await asyncio.gather(*uploads, return_exceptions=True)
    good = upok = 0
    ids = set()
    for i, r in enumerate(out):
        if isinstance(r, Exception):
            continue
        if r.status_code != 200:
            continue
        upok += 1
        if r.json().get("rows") == 5 + i:
            good += 1
        ids.add(r.json().get("id"))
    ok = upok == 6 and good == 6 and len(ids) == 6
    record("6 concurrent uploads, same filename, distinct contents (ua)",
           ok, f"{upok}/6 uploaded, {good}/6 row counts correct, {len(ids)} unique ids")

    # ------------------------------------------------------------------
    # 5. One long-running request + commonplace requests in the same window.
    #    The dataset parse is deliberately slowed (sleep inside the worker
    #    thread) to prove the event loop stays unblocked: health probes, a
    #    no-dataset query (handled on the event loop itself) and the dataset
    #    list must all be served while the slow query is in flight.
    # ------------------------------------------------------------------
    _orig_load = query_module.load_dataset_file
    def _slow_load_dataset_file(file_path, filename):
        if "big.csv" in str(file_path):
            time.sleep(1.5)  # only the BIG dataset parse is slowed
        return _orig_load(file_path, filename)

    query_module.load_dataset_file = _slow_load_dataset_file
    try:
        slow_task = asyncio.create_task(
            client.post("/query/", json={"text": "slow", "dataset_id": big_id},
                        headers=ua_headers))
        await asyncio.sleep(0.4)  # let the slow query land on the worker thread

        t0 = time.perf_counter()
        polls = await asyncio.gather(*[client.get("/health") for _ in range(8)])
        t1 = time.perf_counter()
        gen = await client.post("/query/", json={"text": "general"}, headers=ua_headers)
        t2 = time.perf_counter()
        lst = await client.get("/datasets/", headers=ua_headers)
        t3 = time.perf_counter()
        small = await client.post("/query/", json={"text": "s",
                                                   "dataset_id": small_a_id},
                                  headers=ua_headers)
        t4 = time.perf_counter()
        slow = await slow_task

        polls_ms = (t1 - t0) * 1000.0
        gen_ms = (t2 - t1) * 1000.0
        lst_ms = (t3 - t2) * 1000.0
        small_ms = (t4 - t3) * 1000.0

        good_polls = all(p.status_code == 200 for p in polls)
        gen_ok = gen.status_code == 200 and gen.json().get("answer") == "General query: general"
        lst_ok = lst.status_code == 200
        small_ok = small.status_code == 200 and small.json().get("rows_analyzed") == 3
        slow_ok = slow.status_code == 200 and slow.json().get("rows_analyzed") == BIG_ROWS

        # The three event-loop-only probes must complete well under the 1.5 s
        # parse; the small worker-thread query must also stay under 1 s.
        max_loop_probe = max(polls_ms, gen_ms, lst_ms)
        ok = (good_polls and gen_ok and lst_ok and small_ok and slow_ok
              and max_loop_probe < 1000.0 and small_ms < 1000.0)
        record("slow query (1.5s injected parse on big.csv) + 8 health + general query + list + small query",
               ok,
               f"polls={polls_ms:.0f}ms gen={gen_ms:.0f}ms list={lst_ms:.0f}ms "
               f"small={small_ms:.0f}ms (loop probes max {max_loop_probe:.0f}ms)")
        metric("event-loop responsiveness during 1.5s injected parse",
               f"8 health={polls_ms:.0f}ms gen-query={gen_ms:.0f}ms list={lst_ms:.0f}ms "
               f"small-query={small_ms:.0f}ms - all served while the parse slept 1.5s")
    finally:
        query_module.load_dataset_file = _orig_load
# ------------------------------------------------------------------
    # 6. Real large parse (no artificial sleep) + fast requests at the same
    #    time, and a real large upload concurrent with fast requests.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    big_task = asyncio.create_task(
        client.post("/query/", json={"text": "real big", "dataset_id": big_id},
                    headers=ua_headers))
    await asyncio.sleep(0.15)
    t1 = time.perf_counter()
    polls = await asyncio.gather(*[client.get("/health") for _ in range(6)])
    gen = await client.post("/query/", json={"text": "gen2"}, headers=ua_headers)
    fast_elapsed = time.perf_counter() - t1
    big_resp = await big_task
    big_query_total = time.perf_counter() - t0
    ok = (big_resp.status_code == 200
          and big_resp.json().get("rows_analyzed") == BIG_ROWS
          and all(p.status_code == 200 for p in polls)
          and gen.status_code == 200)
    record("real big parse (120k rows) + 6 health polls + general query",
           ok, f"fast window {fast_elapsed*1000:.0f} ms, big query total "
               f"{big_query_total:.2f}s")
    metric("responsiveness during real 120k-row parse",
           f"6 health + general query in {fast_elapsed*1000:.0f} ms")

    big2 = _matrix_csv(90_000, 10)  # ~1.5 MB, real upload + real parse
    up_task = asyncio.create_task(
        client.post("/datasets/upload", files={"file": ("big2.csv", big2, "text/csv")},
                    headers=ua_headers))
    await asyncio.sleep(0.1)
    t1 = time.perf_counter()
    polls2 = await asyncio.gather(*[client.get("/health") for _ in range(6)])
    gen2 = await client.post("/query/", json={"text": "gen3"}, headers=ua_headers)
    l2 = await client.get("/datasets/", headers=ua_headers)
    up_elapsed = time.perf_counter() - t1
    up_resp = await up_task
    ok = (up_resp.status_code == 200 and up_resp.json().get("rows") == 90_000
          and all(p.status_code == 200 for p in polls2)
          and gen2.status_code == 200 and l2.status_code == 200)
    record("large upload (90k rows) streaming + parsing concurrent with 8 fast requests",
           ok, f"fast window {up_elapsed*1000:.0f} ms, upload id={up_resp.json().get('id')}")

    # ------------------------------------------------------------------
    # 7. Mixed users + mixed datasets at the same time; isolation checks.
    # ------------------------------------------------------------------
    rs = await asyncio.gather(
        client.post("/query/", json={"text": "ua-big", "dataset_id": big_id},
                    headers=ua_headers),
        client.post("/query/", json={"text": "ub-small", "dataset_id": small_b_id},
                    headers=ub_headers),
        client.post("/query/", json={"text": "ua-small", "dataset_id": small_a_id},
                    headers=ua_headers),
        client.post("/query/", json={"text": "ub-big", "dataset_id": big_id},
                    headers=ub_headers),  # ub must NOT access ua's dataset (403)
        client.get("/datasets/", headers=ub_headers),
        client.get("/datasets/", headers=ua_headers),
    )
    a_ok = (rs[0].status_code == 200 and rs[0].json().get("rows_analyzed") == BIG_ROWS
            and rs[2].status_code == 200 and rs[2].json().get("rows_analyzed") == 3)
    b_ok = (rs[1].status_code == 200 and rs[1].json().get("rows_analyzed") == 5
            and rs[3].status_code == 403)
    lists_ok = rs[4].status_code == 200 and rs[5].status_code == 200
    record("mixed-user concurrency with isolation (ua+ub, per-user dataset lists, 403 on foreign)",
           a_ok and b_ok and lists_ok,
           f"ua-big={rs[0].status_code} ua-small={rs[2].status_code} "
           f"ub-small={rs[1].status_code} ub->ua(403)={rs[3].status_code} "
           f"lists={rs[4].status_code}/{rs[5].status_code}")

    # Same user mixing different request kinds in one burst.
    burst = await asyncio.gather(
        client.post("/query/", json={"text": "burst", "dataset_id": big_id},
                    headers=ua_headers),
        client.post("/query/", json={"text": "burst gen"}, headers=ua_headers),
        client.get("/datasets/", headers=ua_headers),
        client.get(f"/datasets/{small_a_id}", headers=ua_headers),
        client.get("/users/me", headers=ua_headers),
    )
    ok = (burst[0].status_code == 200 and burst[1].status_code == 200
          and burst[2].status_code == 200 and burst[3].status_code == 200
          and burst[4].status_code == 200)
    record("same user, mixed request types in one concurrent burst (query/query/list/details/me)",
           ok, f"statuses={[r.status_code for r in burst]}")

    # ------------------------------------------------------------------
    # 8. Final responsiveness after all the load.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    r = await client.get("/health")
    health_elapsed = time.perf_counter() - t0
    record("health after all load", r.status_code == 200,
           f"latency {(health_elapsed*1000):.0f} ms")

    # ------------------------------------------------------------------
    # 9. Scan captured server log for unhandled tasks / exceptions.
    # ------------------------------------------------------------------
    issues = _scan_server_log()
    record("server log clean (no unhandled task/exceptions/locked-DB)",
           len(issues) == 0, "; ".join(issues) if issues else "no entries found")


def _scan_server_log() -> list:
    """Return problematic lines captured from the server log, if any."""
    patterns = (
        "Task exception was never retrieved",
        "Exception in callback",
        "Traceback (most recent call last)",
        "database is locked",
        "OperationalError",
        "IntegrityError",
        "Unhandled error",
    )
    issues = []
    if _SERVER_LOG.exists():
        for line in _SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            lowered = line.lower()
            if any(p.lower() in lowered for p in patterns):
                issues.append(line.strip())
    return issues


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
        logging.getLogger().removeHandler(HANDLER)
        HANDLER.close()
        engine.dispose()

    failed = [row for row in REPORT if not row[1]]
    total = len(REPORT)
    print("\n" + "=" * 72)
    print(f"CONCURRENCY / NON-BLOCKING VALIDATION: {total - len(failed)}/{total} passed")
    if failed:
        print("FAILED:")
        for name, ok, note in failed:
            print(f"  - {name}: {note}")
    else:
        print("ALL CONCURRENCY CHECKS PASSED")
    print("=" * 72)

    if METRICS:
        print("\nMEASURED (approx, not synthetic):")
        for name, note in METRICS:
            print(f"  {name}: {note}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        _TMP.cleanup()