# OmniBrain – Week 1, Team C, Module 2

## Scope

Module 2 covers:

1. Document Upload API
2. Query API
3. API validation and error handling
4. Automated endpoint testing

This module is intentionally independent of Week 2 embeddings/vector-store work. The query endpoint currently provides an integration-ready dataset analysis response (rows/columns) rather than implementing retrieval or RAG.

## 1. Document Upload API

**Endpoint:** `POST /datasets/upload`

**Authentication:** Bearer token required.

**Supported formats:** `.csv`, `.xlsx`, `.xls`

The upload flow:

- validates that the file and filename are present;
- validates the file extension;
- sanitizes the client filename to prevent path traversal;
- creates a unique stored filename using the user ID and UUID;
- streams the file to disk in 1 MB chunks;
- enforces `MAX_UPLOAD_SIZE_MB` while streaming;
- rejects empty uploads;
- parses CSV/Excel using pandas;
- rejects empty or corrupted datasets;
- records filename, file type, size, row count, column count and storage path in the database;
- removes partial files when validation or database persistence fails.

### Success example

`POST /datasets/upload` with `people.csv` returns dataset metadata including:

```json
{
  "id": 1,
  "original_filename": "people.csv",
  "file_type": "text/csv",
  "file_size": 0.001,
  "rows": 2,
  "columns": 4,
  "upload_date": "..."
}
```

## 2. Query API

**Endpoint:** `POST /query/`

**Request:**

```json
{
  "text": "summarize the dataset",
  "dataset_id": 1
}
```

The endpoint:

- validates the query text;
- supports queries without a dataset;
- verifies that a referenced dataset exists;
- verifies dataset ownership;
- loads CSV/Excel data;
- returns rows and columns analyzed;
- returns clean `400`, `401`, `403`, `404` and `500` responses;
- dispatches dataset-backed blocking work to the thread-pool executor so the async event loop remains responsive.

## 3. Testing

Run the full suite with:

```bash
pytest -q
```

Module 2 coverage includes:

### Upload tests

- valid CSV and Excel uploads
- mixed-case extensions
- filename sanitization
- repeated filenames without overwrite
- whitespace/long/extension-only filenames
- missing multipart content type fallback
- empty files and empty datasets
- unsupported/dotless extensions
- corrupt Excel/CSV payloads
- upload-size limits
- file-system and database failure cleanup
- metadata persistence
- existing dataset endpoints after upload
- `/docs` availability

### Query tests

- general queries without a dataset
- owned dataset queries
- large dataset queries
- concurrent queries
- non-blocking `/health` behavior during a slow query
- invalid dataset IDs
- missing dataset files
- corrupted dataset files
- database failures
- cross-user dataset access

## 4. Integration with Module 1

Module 1 provides the FastAPI application scaffold and router registration. Module 2 adds/uses the `/datasets/upload` and `/query/` routes. Both are included by `main.py`, so the modules can be run together through the same FastAPI application and Swagger UI at `/docs`.

## 5. Week 1 boundary

Do **not** add embeddings, vector databases, semantic retrieval, or RAG generation to this module. Those belong to the later Week 2/3 integration work.

## 6. End-to-end Module 2 validation

`tests/test_module2_upload_query_integration.py` validates the complete Week 1
flow through the public API rather than inserting a dataset directly into the
test database. It verifies:

1. an authenticated user uploads a CSV and receives a `dataset_id`;
2. the returned `dataset_id` is accepted by the query endpoint;
3. the query response reports the uploaded dataset's row/column counts;
4. the same flow works for `.xlsx` files.

This provides a clear integration boundary between the document-upload and
query parts of Team C's Module 2.
