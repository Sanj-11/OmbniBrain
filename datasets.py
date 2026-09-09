import asyncio
import os
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
import uuid
from pathlib import Path
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

# Fallback MIME types, keyed by the validated extension. A multipart part is
# allowed to arrive without a Content-Type header, in which case
# ``UploadFile.content_type`` is ``None``; ``Dataset.file_type`` is a NOT NULL
# column, so a sensible type is derived from the extension instead.
_EXTENSION_CONTENT_TYPES = {
    'csv': 'text/csv',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'xls': 'application/vnd.ms-excel',
}

# The uploaded file is streamed to disk in chunks of this size (1 MB). This
# keeps memory usage bounded no matter how large the uploaded dataset is, and
# means an arbitrarily large file is never loaded into memory as one blob.
_UPLOAD_CHUNK_SIZE = 1024 * 1024

# The sanitised part of a stored filename is capped so that
# ``<user id>_<uuid4>_<name>.<ext>`` stays comfortably inside the operating
# system's path-length limit (Windows rejects long paths with OSError).
_MAX_STORED_NAME_LENGTH = 100


def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _file_extension(filename: str) -> str:
    """Return the lower-cased extension of ``filename`` without the dot."""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def _discard_file(file_path) -> None:
    """Delete a partially written / rejected upload, ignoring absent files."""
    try:
        os.remove(file_path)
    except OSError:
        pass


def _validate_upload(file: UploadFile) -> tuple:
    """Validate an incoming upload *before* anything is written to disk.

    Checks, in order, that a file part is present, that it carries a usable
    filename, and that the filename's extension is one of the supported
    dataset formats. Returns the whitespace-normalised filename together with
    its validated extension.
    """
    # -- 1. File presence / filename ------------------------------------------
    # Starlette returns a zero-length UploadFile when a multipart ``file`` part
    # is sent without a filename, so guard on both the part and its filename.
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Surrounding whitespace is normalised away so that a name such as
    # " people.csv " is treated as "people.csv" everywhere: for the extension
    # check, for the stored record, and for the parser choice made later from
    # the recorded filename.
    filename = file.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # -- 2. File type / extension ---------------------------------------------
    if not allowed_file(filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return filename, _file_extension(filename)


def _build_stored_filename(user_id: int, filename: str, extension: str) -> str:
    """Build a unique, filesystem-safe name for the stored copy of an upload.

    The client-supplied name is sanitised (defeating path traversal such as
    ``../../evil.csv``), length-capped, and combined with the owner id and a
    UUID4 so that concurrent uploads of the same name never overwrite each
    other. The validated extension is always preserved.
    """
    # A filename made only of unsafe characters sanitises to an empty string
    # (e.g. a wholly-whitespace or dot-only name); reject it rather than store
    # a meaningless stored filename.
    safe_original_filename = secure_filename(filename)
    if not safe_original_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Sanitising can strip the extension (".csv" becomes "csv"), so rebuild the
    # name from its stem plus the extension validated above.
    stem = Path(safe_original_filename).stem or safe_original_filename
    stem = stem[:_MAX_STORED_NAME_LENGTH].strip(' .') or "dataset"

    return f"{user_id}_{uuid.uuid4()}_{stem}.{extension}"


async def _stream_upload_to_disk(file: UploadFile, file_path) -> int:
    """Stream an upload to ``file_path`` in chunks and return its size in bytes.

    The payload is never buffered in memory as a whole: it is read in
    fixed-size chunks and written straight to disk. An empty upload is
    rejected before the file is created, and an upload exceeding
    ``settings.MAX_UPLOAD_SIZE_MB`` is aborted mid-stream with the partial file
    removed, so neither is ever left on disk.

    Blocking file I/O is offloaded to the default thread-pool executor so the
    asyncio event loop remains responsive while large files are saved.
    """
    max_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)

    # -- 3. Reject empty uploads ----------------------------------------------
    # Peek at the first chunk before creating the file so a zero-byte upload
    # never produces a stored file at all.
    chunk = await file.read(_UPLOAD_CHUNK_SIZE)
    if not chunk:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    total_bytes = 0
    buffer = None
    loop = asyncio.get_running_loop()
    try:
        # Offload the blocking open() call to a thread so the event loop
        # is never blocked by filesystem operations.
        buffer = await loop.run_in_executor(None, open, file_path, "wb")
        try:
            while chunk:
                total_bytes += len(chunk)

                # -- 4. Enforce the size limit while streaming ----------------
                if max_bytes > 0 and total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "File too large. The maximum upload size is "
                            f"{settings.MAX_UPLOAD_SIZE_MB:g} MB."
                        ),
                    )

                # Offload the blocking write() call to a thread.
                await loop.run_in_executor(None, buffer.write, chunk)
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        finally:
            if buffer is not None:
                await loop.run_in_executor(None, buffer.close)
    except OSError:
        _discard_file(file_path)
        raise HTTPException(
            status_code=500,
            detail="Failed to save the uploaded file. Please try again later.",
        )
    except Exception:
        _discard_file(file_path)
        raise

    return total_bytes


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
    # Validate presence, filename and file type before touching the disk.
    filename, extension = _validate_upload(file)

    # Prepare a unique, sanitised destination for the upload.
    stored_filename = _build_stored_filename(current_user.id, filename, extension)

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = settings.UPLOAD_DIR / stored_filename

    # Stream the upload to disk (empty and oversized uploads are rejected here
    # and leave nothing behind).
    total_bytes = await _stream_upload_to_disk(file, file_path)

    # -- 5. Validate the content and collect metadata -------------------------
    # Offload the blocking pandas I/O to a thread so the asyncio event loop
    # stays responsive while large files are parsed.
    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(None, load_dataset_file, file_path, filename)
    except Exception:
        _discard_file(file_path)
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be processed. It may be corrupted or in an unsupported format.",
        )

    if df.empty:
        _discard_file(file_path)
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")

    dataset_data = {
        "original_filename": filename,
        "stored_filename": stored_filename,
        "file_path": str(file_path),
        "file_type": (file.content_type or "").strip()
        or _EXTENSION_CONTENT_TYPES.get(extension, "application/octet-stream"),
        "file_size": total_bytes / (1024 * 1024),  # in MB
        "rows": len(df),
        "columns": len(df.columns),
    }

    # If all validation passes, create the database record
    try:
        return crud.create_dataset(db=db, dataset_data=dataset_data, user_id=current_user.id)
    except Exception:
        _discard_file(file_path)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while saving the dataset record.",
        )

@router.get("/", response_model=List[schemas.Dataset])
async def get_user_datasets(
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
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The dataset file could not be processed. It may be corrupted.",
        )

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
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