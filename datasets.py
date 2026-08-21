import os
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
import uuid
from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename
from typing import List

import crud
import models
import schemas
from database import get_db
from auth import get_current_user
from config import settings
router = APIRouter(prefix="/datasets", tags=["Datasets"])

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_dataset_file(file_path: str, filename: str) -> pd.DataFrame:
    """Read an uploaded dataset file into a pandas DataFrame.

    The parser is chosen from the original filename extension: CSV files are
    read with ``pd.read_csv`` and every other supported format (xlsx/xls) is
    read with ``pd.read_excel``.
    """
    if filename.lower().endswith('.csv'):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)

@router.post("/upload", response_model=schemas.Dataset)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename to prevent overwrites
    safe_original_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4()
    stored_filename = f"{current_user.id}_{unique_id}_{safe_original_filename}"
    file_path = settings.UPLOAD_DIR / stored_filename

    try:
        # Save the file first
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Use pandas to validate the file content and get metadata
        df = load_dataset_file(file_path, file.filename)

        if df.empty:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")

        dataset_data = {
            "original_filename": file.filename,
            "stored_filename": stored_filename,
            "file_path": str(file_path),
            "file_type": file.content_type,
            "file_size": file_path.stat().st_size / (1024 * 1024),  # in MB
            "rows": len(df),
            "columns": len(df.columns),
        }
        
        # If all validation passes, create the database record
        return crud.create_dataset(db=db, dataset_data=dataset_data, user_id=current_user.id)

    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
        # Handle corrupted, empty, or invalid format files
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Could not process file: {e}")
    except HTTPException:
        # Re-raise HTTPExceptions to avoid them being caught by the generic exception handler
        raise
    except Exception as e:
        # Catch any other unexpected errors
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@router.get("/", response_model=List[schemas.Dataset])
def get_user_datasets(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return crud.get_datasets_by_user(db, user_id=current_user.id)

@router.get("/{dataset_id}", response_model=schemas.DatasetDetails)
def get_dataset_details(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_dataset = crud.get_dataset_by_id(db, dataset_id=dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if db_dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)

        return {
            "id": db_dataset.id,
            "original_filename": db_dataset.original_filename,
            "file_type": db_dataset.file_type,
            "file_size": db_dataset.file_size,
            "upload_date": db_dataset.upload_date,
            "rows": db_dataset.rows,
            "columns": db_dataset.columns,
            "column_info": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(10).to_dict(orient='records'),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found on server")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process dataset file: {e}")

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_dataset = crud.get_dataset_by_id(db, dataset_id=dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if db_dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this dataset")

    # Delete physical file
    try:
        if os.path.exists(db_dataset.file_path):
            os.remove(db_dataset.file_path)
    except Exception as e:
        pass # Log this error in a real app, but proceed to delete DB record

    # Delete database record
    if not crud.delete_dataset(db, dataset_id=dataset_id):
        # This case should be rare if the first check passed
        raise HTTPException(status_code=500, detail="Failed to delete dataset from database")

    return