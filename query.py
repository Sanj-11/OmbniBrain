import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional

import crud
import models
import schemas
from auth import get_current_user
from database import get_db
from datasets import load_dataset_file

router = APIRouter(prefix="/query", tags=["Query"])


def _get_owned_dataset_or_raise(
    dataset_id: int,
    db: Session,
    current_user: models.User,
) -> models.Dataset:
    """Look up a dataset and enforce per-user ownership.

    ``crud.get_dataset_by_id`` is a blocking, synchronous SQLAlchemy database
    query; this helper therefore only ever runs inside the worker thread used
    by :func:`execute_query_service`. Ownership checks (404 for unknown, 403
    for another user's dataset) are unchanged.
    """
    try:
        db_dataset = crud.get_dataset_by_id(db, dataset_id=dataset_id)
    except Exception:
        # Database failure: surface a clean 500 instead of leaking the
        # underlying error. Raised inside the executor thread and propagated
        # unchanged through the awaited future.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while accessing the dataset",
        )

    if db_dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    if db_dataset.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this dataset",
        )
    return db_dataset


def execute_query_service(
    dataset_id: Optional[int],
    query_text: str,
    current_user: models.User,
    db: Session,
) -> schemas.QueryResponse:
    """Execute a query and return a structured response.

    This is the *blocking processing unit* of the query pipeline
    (Asynchronous Query Processing - Part 2). It is intentionally synchronous
    and is never invoked directly on the asyncio event loop:

    * a dataset-backed query is scheduled on the default thread-pool executor
      by the async endpoint via ``loop.run_in_executor``;
    * a query without a dataset performs no I/O/DB work and is kept directly on
      the event loop (lightweight request handling).

    Blocking operations confined to this unit (each runs on the worker thread):

    - ``_get_owned_dataset_or_raise`` -> ``crud.get_dataset_by_id``: synchronous
      database query.
    - ``load_dataset_file``: synchronous pandas CSV/Excel read from disk.
    - ``len(df)`` / ``len(df.columns)``: synchronous pandas frame-size calls
      (and any future DataFrame inspection such as ``df.head()``,
      ``df.describe()``, ``df.query()``).

    Exceptions raised here - including the ``HTTPException`` mapped from
    missing/corrupted files and from database access failures - are propagated
    unchanged from the executor thread, re-raised at the ``await`` point in the
    async endpoint, and converted by FastAPI into the matching API responses.
    """
    if dataset_id is not None:
        db_dataset = _get_owned_dataset_or_raise(dataset_id, db, current_user)
        try:
            df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset file not found on server",
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not process dataset file",
            )

        rows = len(df)
        columns = len(df.columns)
        answer = f"Query '{query_text}' analyzed {rows} rows and {columns} columns."

        return schemas.QueryResponse(
            answer=answer,
            dataset_id=dataset_id,
            rows_analyzed=rows,
            columns_analyzed=columns,
        )

    return schemas.QueryResponse(
        answer=f"General query: {query_text}",
        dataset_id=None,
        rows_analyzed=0,
        columns_analyzed=0,
    )


@router.post("/", response_model=schemas.QueryResponse)
async def query_dataset(
    request: schemas.QueryRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Handle one query request.

    Only lightweight work happens in the async flow here: request validation
    and dispatch. A query that targets a dataset is processed by the blocking
    :func:`execute_query_service` on the thread-pool executor, so a slow
    database lookup or a large CSV/Excel parse can never occupy the event loop.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty",
        )

    query_text = request.text.strip()

    if request.dataset_id is None:
        # No dataset means no I/O: keep this trivially fast path in the async
        # flow instead of paying for a pointless thread-hop.
        return execute_query_service(None, query_text, current_user, db)

    # Dataset query: offload the blocking processing unit to the thread-pool
    # executor so the event loop stays free to keep serving the rest of the
    # API (including /health) while the work is in flight.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        execute_query_service,
        request.dataset_id,
        query_text,
        current_user,
        db,
    )
