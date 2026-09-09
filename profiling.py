import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import crud
import models
import schemas
from auth import get_current_user
from database import get_db
from datasets import load_dataset_file

try:
    from openpyxl import exceptions as openpyxl_exceptions
    _OPENPYXL_ERRORS: tuple = (openpyxl_exceptions.InvalidFileException,)
except ImportError:  # pragma: no cover - openpyxl is declared in requirements.txt
    _OPENPYXL_ERRORS = ()


def _json_safe(value: Any) -> Any:
    """Convert a pandas/NumPy scalar into a plain JSON-compatible Python value.

    Handles the NumPy and pandas scalar types that can appear in computed
    summaries (``numpy.int64``, ``numpy.float64``, ``numpy.bool_``,
    ``pandas.Timestamp`` and naive ``NaN``) so that FastAPI can serialise the
    response without errors. Missing values are mapped to ``None`` (JSON
    ``null``), which is always JSON-safe.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # pd.isna raises for some non-scalar/list-like inputs; those are left
        # to the type-specific handling below.
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float, bool)):
        return value
    if hasattr(value, "item"):
        # numpy/Pandas scalars with .item() (including pd.Timestamp) expose a
        # plain Python primitive via item().
        return _json_safe(value.item())
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _nullable_float(value: Any) -> Optional[float]:
    """Convert a summary statistic to ``float`` or ``None`` when it is missing."""
    clean = _json_safe(value)
    if clean is None:
        return None
    return float(clean)


def classify_columns(df: pd.DataFrame):
    """Classify the columns of ``df`` into numerical, categorical and datetime.

    This is the single shared classifier behind Part 2B-3 and Part 2B-4 so the
    two profiling endpoints agree on which columns are numerical, categorical
    or datetime. The rules match the real pandas dtypes:

    * datetime columns (``datetime64``) are kept separate;
    * boolean columns are categorical (checked before the numeric test because
      pandas reports ``bool`` as numeric);
    * numeric dtypes (int/float and other pandas numeric dtypes) are numerical;
    * everything else (object/string, pandas ``Categorical``, any other
      non-numeric dtype) is categorical.

    Returns three parallel lists of column names.
    """
    numerical_columns: List[str] = []
    categorical_columns: List[str] = []
    datetime_columns: List[str] = []

    for column in df.columns:
        col_name = str(column)
        dtype = df[column].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_columns.append(col_name)
        elif pd.api.types.is_bool_dtype(dtype):
            categorical_columns.append(col_name)
        elif pd.api.types.is_numeric_dtype(dtype):
            numerical_columns.append(col_name)
        else:
            categorical_columns.append(col_name)

    return numerical_columns, categorical_columns, datetime_columns


router = APIRouter(prefix="/datasets", tags=["Dataset Profiling"])


def get_owned_dataset_or_raise(
    dataset_id: int,
    db: Session,
    current_user: models.User,
) -> models.Dataset:
    """Return the dataset owned by ``current_user`` or raise 404/403.

    Mirrors the dataset access rules already used across Part 2A: a user may
    only work with their own datasets.
    """
    db_dataset = crud.get_dataset_by_id(db, dataset_id=dataset_id)
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


@router.get("/{dataset_id}/structure", response_model=schemas.DatasetStructure)
def get_dataset_structure(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Return the structural profile (rows, columns, names, types) of a dataset.

    The dataset is loaded from its stored file with the same pandas-based
    logic used elsewhere in the API, and the reported data types are the real
    pandas dtypes detected by pandas (e.g. ``int64``, ``float64``, ``object``,
    ``bool``, ``datetime64[ns]``). Nothing is converted to make the response
    easier.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        zipfile.BadZipFile,
        *_OPENPYXL_ERRORS,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file",
        )

    if len(df.columns) == 0 or len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be profiled.",
        )

    column_names: List[str] = [str(column) for column in df.columns]
    data_types: Dict[str, str] = {
        str(column): str(dtype) for column, dtype in df.dtypes.items()
    }

    return {
        "id": db_dataset.id,
        "original_filename": db_dataset.original_filename,
        "file_type": db_dataset.file_type,
        "file_size": db_dataset.file_size,
        "upload_date": db_dataset.upload_date,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": column_names,
        "data_types": data_types,
    }


@router.get(
    "/{dataset_id}/missing-values",
    response_model=schemas.DatasetMissingValueProfile,
)
def get_dataset_missing_values(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Return the missing-value analysis (Part 2B-2) of a dataset.

    The dataset is loaded from its stored file with the same pandas-based
    logic used elsewhere in the API. Every column is scanned with pandas'
    ``isna()`` detection so that NaN, None and NaT values are all recognised
    in numerical, object/string, boolean, categorical and datetime columns.

    The reported dataset is not modified in any way: this is a read-only
    profiling operation and missing values are neither filled nor dropped.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        zipfile.BadZipFile,
        *_OPENPYXL_ERRORS,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file",
        )

    if len(df.columns) == 0 or len(df) == 0:
        # The project treats an empty dataset as invalid and rejects it at
        # upload time; report the same validation error here for consistency.
        # Guarding on row/column count also avoids a division-by-zero when
        # computing percentages.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be profiled.",
        )

    total_rows: int = len(df)
    total_missing_values: int = int(df.isna().sum().sum())

    column_profiles: Dict[str, schemas.ColumnMissingValue] = {}
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = (
            round((missing_count / total_rows) * 100, 2) if total_rows else 0.0
        )
        column_profiles[str(column)] = schemas.ColumnMissingValue(
            missing_count=missing_count,
            missing_percentage=missing_percentage,
        )

    return schemas.DatasetMissingValueProfile(
        total_rows=total_rows,
        total_missing_values=total_missing_values,
        columns=column_profiles,
    )


@router.get("/{dataset_id}/quality", response_model=schemas.DatasetQualityProfile)
def get_dataset_quality(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Return the duplicate count and column classification (Part 2B-3).

    The dataset is loaded from its stored file with the same pandas-based
    logic used elsewhere in the API.

    * ``duplicate_rows`` uses pandas' ``DataFrame.duplicated().sum()`` so every
      row that repeats an earlier row is counted exactly once and the first
      occurrence of an identical row is never counted. Duplicate rows are not
      deleted; the stored dataset is left unchanged.
    * Column classification is driven by the real pandas dtypes: numeric
      columns (int/float and other numeric dtypes) are ``numerical_columns``,
      non-numeric columns (object/string, ``category``, ``bool`` and other
      non-numeric dtypes) are ``categorical_columns``, and ``datetime`` columns
      are reported separately in ``datetime_columns`` rather than being forced
      into either category. No column names are hardcoded.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        zipfile.BadZipFile,
        *_OPENPYXL_ERRORS,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file",
        )

    if len(df.columns) == 0 or len(df) == 0:
        # The project treats an empty dataset as invalid and rejects it at
        # upload time; report the same validation error here for consistency.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be profiled.",
        )

    # Count rows that are duplicates of an earlier row. Series.duplicated()
    # marks the second and later occurrences of an identical row, so the
    # first occurrence is never included, and the sum is the duplicate count.
    duplicate_rows: int = int(df.duplicated().sum())

    numerical_columns, categorical_columns, datetime_columns = classify_columns(df)

    return schemas.DatasetQualityProfile(
        duplicate_rows=duplicate_rows,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )


@router.get("/{dataset_id}/summary", response_model=schemas.DatasetSummary)
def get_dataset_summary(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Return the basic dataset summary (Part 2B-4) of a dataset.

    The dataset is loaded from its stored file with the same pandas-based
    logic used elsewhere in the API, and the same column classifier as Part
    2B-3 decides which columns are numerical, categorical or datetime.

    * Every numerical column is summarised with pandas' ``describe()``: count,
      mean, standard deviation, min, the 25/50/75 percentiles and max.
    * Every categorical column is summarised with its non-null count, the
      number of unique non-null values, the most frequent value and how often
      that value occurs.
    * Datetime columns are kept separate (as in Part 2B-3) and are not part of
      either summary.

    The summary is read-only: missing values are neither filled nor dropped,
    duplicates are not removed, and the stored dataset is never modified.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        zipfile.BadZipFile,
        *_OPENPYXL_ERRORS,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file",
        )

    if len(df.columns) == 0 or len(df) == 0:
        # The project treats an empty dataset as invalid and rejects it at
        # upload time; report the same validation error here for consistency.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be profiled.",
        )

    numerical_columns, categorical_columns, _ = classify_columns(df)

    numerical_summary: Dict[str, schemas.NumericalColumnSummary] = {}
    for column in numerical_columns:
        stats = df[column].describe()
        numerical_summary[str(column)] = schemas.NumericalColumnSummary(
            count=_json_safe(stats.get("count")) or 0,
            mean=_nullable_float(stats.get("mean")),
            std=_nullable_float(stats.get("std")),
            min=_nullable_float(stats.get("min")),
            p25=_nullable_float(stats.get("25%")),
            p50=_nullable_float(stats.get("50%")),
            p75=_nullable_float(stats.get("75%")),
            max=_nullable_float(stats.get("max")),
        )

    categorical_summary: Dict[str, schemas.CategoricalColumnSummary] = {}
    for column in categorical_columns:
        series = df[column]
        counts = series.value_counts()
        if counts.empty:
            top: Any = None
            frequency: int = 0
        else:
            top = _json_safe(counts.index[0])
            frequency = int(counts.iloc[0])
        categorical_summary[str(column)] = schemas.CategoricalColumnSummary(
            count=int(series.notna().sum()),
            unique=int(series.nunique()),
            top=top,
            frequency=frequency,
        )

    return schemas.DatasetSummary(
        numerical_summary=numerical_summary,
        categorical_summary=categorical_summary,
    )