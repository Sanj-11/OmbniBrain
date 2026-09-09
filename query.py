import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import crud
import models
import schemas
from auth import get_current_user
from database import get_db
from datasets import load_dataset_file
from agents.python_agent import PythonAgent


router = APIRouter(prefix="/query", tags=["Query"])

# Create Python Agent
python_agent = PythonAgent()


def _get_owned_dataset_or_raise(
    dataset_id: int,
    db: Session,
    current_user: models.User,
) -> models.Dataset:
    """
    Get the dataset from the database and make sure
    it belongs to the current user.
    """

    try:
        db_dataset = crud.get_dataset_by_id(
            db,
            dataset_id=dataset_id,
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while accessing the dataset",
        )

    # Dataset does not exist
    if db_dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    # Dataset belongs to another user
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
    """
    Process the user's query using the Python Agent.
    """

    # =========================================================
    # 1. GENERAL QUERY WITHOUT DATASET
    # =========================================================

    if dataset_id is None:

        return schemas.QueryResponse(
            answer=f"General query: {query_text}",
            dataset_id=None,
            rows_analyzed=0,
            columns_analyzed=0,
        )

    # =========================================================
    # 2. GET DATASET
    # =========================================================

    db_dataset = _get_owned_dataset_or_raise(
        dataset_id,
        db,
        current_user,
    )

    # =========================================================
    # 3. LOAD DATASET
    # =========================================================

    try:

        df = load_dataset_file(
            db_dataset.file_path,
            db_dataset.original_filename,
        )

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

    # =========================================================
    # 4. CHECK DATASET
    # =========================================================

    if df.empty:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty",
        )

    # Number of rows and columns
    rows = len(df)
    columns = len(df.columns)

    # =========================================================
    # 5. SEND QUESTION TO PYTHON AGENT
    # =========================================================

    try:

        answer = python_agent.answer_question(
            df,
            query_text,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Python Agent error: {str(exc)}",
        )

    # =========================================================
    # 6. RETURN RESPONSE
    # =========================================================

    return schemas.QueryResponse(
        answer=answer,
        dataset_id=dataset_id,
        rows_analyzed=rows,
        columns_analyzed=columns,
    )


@router.post(
    "/",
    response_model=schemas.QueryResponse,
)
async def query_dataset(
    request: schemas.QueryRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Handle a user's dataset query.
    """

    # =========================================================
    # 1. VALIDATE QUESTION
    # =========================================================

    if not request.text or not request.text.strip():

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty",
        )

    query_text = request.text.strip()

    # =========================================================
    # 2. QUERY WITHOUT DATASET
    # =========================================================

    if request.dataset_id is None:

        return execute_query_service(
            None,
            query_text,
            current_user,
            db,
        )

    # =========================================================
    # 3. QUERY WITH DATASET
    #
    # Pandas + database operations are blocking,
    # so run them in a thread pool.
    # =========================================================

    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        None,
        execute_query_service,
        request.dataset_id,
        query_text,
        current_user,
        db,
    )