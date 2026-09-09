import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

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
from profiling import classify_columns, get_owned_dataset_or_raise

UNKNOWN_FILL = "Unknown"

# Strategies used for the ``strategy`` field of the cleaning report.
STRATEGY_MEDIAN = "median"
STRATEGY_MODE = "mode"

# Shared parser-error types translated into a 400 response below.
_PARSING_ERRORS: tuple = (
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    ValueError,
)

router = APIRouter(prefix="/datasets", tags=["Dataset Cleaning"])


def detect_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Return the number of missing values in every column of ``df``.

    Uses pandas' ``isna()`` detection so ``NaN``, ``None`` and ``NaT`` are all
    recognised in numerical, object/string, boolean, categorical and datetime
    columns. The returned mapping uses the string form of each column name as
    its key so the result is stable regardless of the underlying column labels.
    """
    return {str(column): int(df[column].isna().sum()) for column in df.columns}


@dataclass
class MissingValueCleaningResult:
    """The outcome of one missing-value cleaning run (Part 2C-1).

    Holds both the cleaned DataFrame (ready for the next cleaning stage) and
    the run statistics that map onto :class:`schemas.MissingValueCleaningReport`.
    """

    cleaned_df: pd.DataFrame
    missing_values_before: int
    missing_values_after: int
    columns_processed: Dict[str, schemas.ColumnCleaningReport]
    removed_empty_columns: List[str]
    cleaning_applied: bool


def clean_missing_values(df: pd.DataFrame) -> MissingValueCleaningResult:
    """Fill the missing values of ``df`` using the Part 2C-1 defaults.

    The operation is performed on a ``copy()`` of ``df`` so the original
    DataFrame (and therefore the uploaded file it was loaded from) is never
    modified. Completely-empty columns are dropped; every other column that has
    missing values is filled with the median (numerical) or the mode /
    ``"Unknown"`` (categorical, datetime and boolean). The returned result
    contains the cleaned copy plus a per-column report.
    """
    # Always work on a copy: the caller/stored dataset is never modified.
    cleaned_df = df.copy()
    missing_values_before = int(cleaned_df.isna().sum().sum())
    total_rows = len(cleaned_df)

    missing_counts = detect_missing_values(cleaned_df)

    # A completely-empty column has no median/mode to compute. Detect those
    # columns up front and drop them, recording the removal in the report.
    removed_empty_columns = [
        column
        for column in cleaned_df.columns
        if total_rows > 0 and missing_counts[column] == total_rows
    ]
    if removed_empty_columns:
        cleaned_df = cleaned_df.drop(columns=removed_empty_columns)

    # Reuse the Part 2B column classifier so cleaning agrees with profiling on
    # which columns are numerical, categorical or datetime.
    numerical_columns, categorical_columns, datetime_columns = classify_columns(cleaned_df)
    strategy_for_column: Dict[str, str] = {
        str(column): STRATEGY_MEDIAN for column in numerical_columns
    }
    for column in list(categorical_columns) + list(datetime_columns):
        strategy_for_column[str(column)] = STRATEGY_MODE

    columns_processed: Dict[str, schemas.ColumnCleaningReport] = {}

    for column in cleaned_df.columns:
        col_name = str(column)
        missing_count = missing_counts.get(col_name, 0)
        if missing_count <= 0:
            continue

        strategy = strategy_for_column.get(col_name, STRATEGY_MODE)
        series = cleaned_df[column]

        if strategy == STRATEGY_MEDIAN:
            fill_value = series.median()
            # A partially-null numerical column always has a usable median;
            # this guard is only a defensive fallback.
            if pd.isna(fill_value):
                fill_value = 0.0
            cleaned_df[column] = series.fillna(fill_value)
        else:  # mode strategy
            mode_values = series.mode(dropna=True)
            if mode_values.empty:
                fill_value = UNKNOWN_FILL
            else:
                fill_value = mode_values.iloc[0]
            cleaned_df[column] = series.fillna(fill_value)

        columns_processed[col_name] = schemas.ColumnCleaningReport(
            strategy=strategy,
            values_filled=missing_count,
        )

    missing_values_after = int(cleaned_df.isna().sum().sum())
    cleaning_applied = missing_values_before > 0

    return MissingValueCleaningResult(
        cleaned_df=cleaned_df,
        missing_values_before=missing_values_before,
        missing_values_after=missing_values_after,
        columns_processed=columns_processed,
        removed_empty_columns=removed_empty_columns,
        cleaning_applied=cleaning_applied,
    )


@router.post(
    "/{dataset_id}/clean/missing-values",
    response_model=schemas.MissingValueCleaningReport,
)
def clean_dataset_missing_values(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.MissingValueCleaningReport:
    """Clean the missing values of an uploaded dataset (Part 2C-1).

    The dataset is loaded from its stored file with the same pandas-based logic
    used elsewhere in the API, cleaned on an in-memory copy with
    :func:`clean_missing_values`, and a report of the actions taken is
    returned. The uploaded file itself is never modified.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except _PARSING_ERRORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file during cleaning",
        )

    if len(df.columns) == 0 or len(df) == 0:
        # The project treats an empty dataset as invalid at upload time; report
        # the same validation error here for consistency with profiling.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be cleaned.",
        )

    result = clean_missing_values(df)

    return schemas.MissingValueCleaningReport(
        missing_values_before=result.missing_values_before,
        missing_values_after=result.missing_values_after,
        columns_processed=result.columns_processed,
        removed_empty_columns=result.removed_empty_columns,
        cleaning_applied=result.cleaning_applied,
    )


# ---------------------------------------------------------------------------
# Part 2C-2 - Duplicate removal & safe data type correction
# ---------------------------------------------------------------------------

# Categories used by :func:`column_dtype_category` for the report and for the
# decision of whether a conversion attempt is allowed.
DTYPE_CATEGORY_INTEGER = "integer"
DTYPE_CATEGORY_FLOAT = "float"
DTYPE_CATEGORY_BOOLEAN = "boolean"
DTYPE_CATEGORY_DATETIME = "datetime"
DTYPE_CATEGORY_CATEGORICAL = "categorical"
DTYPE_CATEGORY_OBJECT = "object"
DTYPE_CATEGORY_OTHER = "other"

# Conversion safety thresholds. A column in ``object``/``string`` dtype is only
# converted to numeric or datetime when at least this fraction of its non-null
# values parse successfully. Boolean conversion is stricter: every non-null
# value must be a recognised boolean token AND both truthy and falsy values
# must be present (so a constant label column is never mistaken for boolean).
# If the thresholds are not met the conversion is skipped and the column keeps
# its original representation. These values are the documented default; the
# numeric/datetime 0.8 threshold means a few junk tokens are tolerated, while
# a column mostly-or-wholly text is left alone.
NUMERIC_CONVERSION_THRESHOLD = 0.8
DATETIME_CONVERSION_THRESHOLD = 0.8
BOOLEAN_CONVERSION_THRESHOLD = 1.0

# Per-column ``action`` values used in the Part 2C-2 cleaning report.
ACTION_UNCHANGED = "unchanged"
ACTION_CONVERTED_NUMERIC = "converted_to_numeric"
ACTION_CONVERTED_DATETIME = "converted_to_datetime"
ACTION_CONVERTED_BOOLEAN = "converted_to_boolean"
ACTION_SKIPPED = "conversion_skipped"


@dataclass
class DuplicateAndTypeCleaningResult:
    """The outcome of one Part 2C-2 cleaning run.

    Holds the cleaned DataFrame (ready for Part 2C-3), the duplicate
    statistics and the per-column dtype report which maps onto
    :class:`schemas.DataCleaningReport`.
    """

    cleaned_df: pd.DataFrame
    rows_before: int
    duplicate_rows_found: int
    duplicate_rows_removed: int
    rows_after: int
    duplicate_percentage: float
    duplicates_cleaned: bool
    columns_processed: Dict[str, schemas.ColumnDtypeCleaningReport]
    dtype_changes: int
    conversions_skipped: List[str]


def detect_duplicate_rows(df: pd.DataFrame) -> Dict[str, float]:
    """Return duplicate statistics for ``df`` without modifying it.

    A row is a duplicate when its values match an earlier row in *all*
    columns (pandas ``duplicated(keep="first")`` semantics, matching the
    read-only Part 2B-3 quality endpoint). The first occurrence of an
    identical row is never counted as a duplicate.
    """
    rows_before = int(len(df))
    duplicate_rows = int(df.duplicated(keep="first").sum())
    unique_rows = rows_before - duplicate_rows
    duplicate_percentage = (
        round(duplicate_rows / rows_before * 100.0, 2) if rows_before > 0 else 0.0
    )
    return {
        "rows_before": rows_before,
        "duplicate_rows_found": duplicate_rows,
        "unique_rows": unique_rows,
        "duplicate_percentage": duplicate_percentage,
        "rows_after": unique_rows,
    }


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` without duplicate rows.

    The first occurrence of each identical row is kept, the original column
    order is preserved, the index is reset to a clean ``0..n-1`` range, and
    the original DataFrame is never modified. Datasets with no duplicates are
    returned unchanged (apart from the reset index copy).
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(keep="first")
    return cleaned.reset_index(drop=True)


def column_dtype_category(dtype) -> str:
    """Classify a pandas dtype into one of the Part 2C-2 categories.

    Returns ``"integer"``, ``"float"``, ``"boolean"``, ``"datetime"``,
    ``"categorical"``, ``"object"`` (plain object or pandas ``"string"``
    columns, i.e. text that may hide numbers/dates) or ``"other"``.
    """
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return DTYPE_CATEGORY_DATETIME
    if pd.api.types.is_bool_dtype(dtype):
        return DTYPE_CATEGORY_BOOLEAN
    if pd.api.types.is_integer_dtype(dtype):
        return DTYPE_CATEGORY_INTEGER
    if pd.api.types.is_float_dtype(dtype):
        return DTYPE_CATEGORY_FLOAT
    if pd.api.types.is_numeric_dtype(dtype):
        return DTYPE_CATEGORY_FLOAT
    if isinstance(dtype, pd.CategoricalDtype):
        return DTYPE_CATEGORY_CATEGORICAL
    if pd.api.types.is_string_dtype(dtype):
        return DTYPE_CATEGORY_OBJECT
    return DTYPE_CATEGORY_OTHER


def _conversion_success_ratio(original: pd.Series, converted: pd.Series) -> float:
    """Fraction of originally non-null values that converted successfully.

    Only originally non-null values count: values that were already missing
    stay missing and are not interpreted as data loss.
    """
    non_null = int(original.notna().sum())
    if non_null == 0:
        return 0.0
    successful = int((converted.notna() & original.notna()).sum())
    return successful / non_null


def _numeric_conversion_attempt(series: pd.Series, threshold: float):
    """Try to parse ``series`` as numbers; ``(converted_series, ratio)``.

    Returns ``None`` for the converted series when the success ratio is below
    ``threshold`` so the caller can keep the original representation instead
    of silently destroying values that do not parse.
    """
    if int(series.notna().sum()) == 0:
        return None, 0.0
    converted = pd.to_numeric(series, errors="coerce")
    ratio = _conversion_success_ratio(series, converted)
    if ratio >= threshold:
        return converted, ratio
    return None, ratio


def _datetime_conversion_attempt(series: pd.Series, threshold: float):
    """Try to parse ``series`` as datetimes; ``(converted_series, ratio)``.

    Returns ``None`` when the parse ratio is below ``threshold`` so a column
    with only a handful of date-looking values is not treated as a date
    column. Ordinary numeric columns are handled by the numeric conversion
    step first and never reach here.
    """
    if int(series.notna().sum()) == 0:
        return None, 0.0
    # Probing an arbitrary text column with ``pd.to_datetime`` intentionally
    # emits pandas' "could not infer format" UserWarning; the warning is noise
    # here because the result is discarded unless the parse ratio is high.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        converted = pd.to_datetime(series, errors="coerce")
    ratio = _conversion_success_ratio(series, converted)
    if ratio >= threshold:
        return converted, ratio
    return None, ratio


def _boolean_value(value):
    """Map a scalar to ``True``/``False`` or ``None`` when not boolean-like."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "y", "1"):
            return True
        if normalized in ("false", "no", "n", "0"):
            return False
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def _boolean_conversion_attempt(series: pd.Series, threshold: float):
    """Try to parse ``series`` as booleans; ``(converted_series, ratio)``.

    Conservative: conversion only happens when every non-null value is a
    recognised boolean token (``True``/``False``, ``1``/``0``, ``yes``/``no``
    and friends) *and* both truthy and falsy values are present, so a
    constant label column is never misread as a boolean column.
    """
    if int(series.notna().sum()) == 0:
        return None, 0.0
    mapped = series.map(_boolean_value)
    recognized = mapped.notna() & series.notna()
    ratio = int(recognized.sum()) / int(series.notna().sum())
    if ratio < threshold:
        return None, ratio
    if not (bool(mapped.eq(True).any()) and bool(mapped.eq(False).any())):
        return None, ratio
    return mapped.astype("boolean"), ratio


def correct_column_dtypes(df: pd.DataFrame):
    """Safely correct the pandas dtypes of ``df`` on a copy.

    Only ``object``/``string`` columns are considered for conversion. A column
    is converted to numeric, datetime or boolean only when the conversion
    satisfies the configured safety threshold (see the module-level
    ``*_CONVERSION_THRESHOLD`` constants); otherwise the column keeps its
    original values and dtype. Already-numeric, datetime, boolean, categorical
    and other exotic columns are left as-is.

    Returns ``(corrected_df, columns_processed, dtype_changes,
    conversions_skipped)`` where ``columns_processed`` maps every column name
    to its :class:`schemas.ColumnDtypeCleaningReport`.
    """
    working = df.copy()
    columns_processed: Dict[str, schemas.ColumnDtypeCleaningReport] = {}
    conversions_skipped: List[str] = []
    dtype_changes = 0

    for column in working.columns:
        col_name = str(column)
        series = working[column]
        original_dtype = str(series.dtype)
        category = column_dtype_category(series.dtype)
        action = ACTION_UNCHANGED
        considered_ratio = 0.0

        if category == DTYPE_CATEGORY_OBJECT:
            # 1. numeric first (a mostly-numeric column is never read as dates)
            numeric = _numeric_conversion_attempt(series, NUMERIC_CONVERSION_THRESHOLD)
            if numeric[0] is not None:
                working[column] = numeric[0]
                action = ACTION_CONVERTED_NUMERIC
            else:
                considered_ratio = max(considered_ratio, numeric[1])
                # 2. datetime
                datetime_attempt = _datetime_conversion_attempt(
                    series, DATETIME_CONVERSION_THRESHOLD
                )
                if datetime_attempt[0] is not None:
                    working[column] = datetime_attempt[0]
                    action = ACTION_CONVERTED_DATETIME
                else:
                    considered_ratio = max(considered_ratio, datetime_attempt[1])
                    # 3. boolean
                    boolean_attempt = _boolean_conversion_attempt(
                        series, BOOLEAN_CONVERSION_THRESHOLD
                    )
                    if boolean_attempt[0] is not None:
                        working[column] = boolean_attempt[0]
                        action = ACTION_CONVERTED_BOOLEAN
                    else:
                        considered_ratio = max(considered_ratio, boolean_attempt[1])

        new_dtype = str(working[column].dtype)

        if action in (
            ACTION_CONVERTED_NUMERIC,
            ACTION_CONVERTED_DATETIME,
            ACTION_CONVERTED_BOOLEAN,
        ):
            dtype_changes += 1
        elif (
            action == ACTION_UNCHANGED
            and category == DTYPE_CATEGORY_OBJECT
            and considered_ratio > 0.0
        ):
            # A conversion was clearly considered (some values looked numeric
            # or date-like) but the safety threshold was not met: report it in
            # ``conversions_skipped`` and keep the original representation.
            action = ACTION_SKIPPED
            conversions_skipped.append(col_name)

        columns_processed[col_name] = schemas.ColumnDtypeCleaningReport(
            original_dtype=original_dtype,
            new_dtype=new_dtype,
            action=action,
        )

    return working, columns_processed, dtype_changes, conversions_skipped


def clean_duplicates_and_types(df: pd.DataFrame) -> DuplicateAndTypeCleaningResult:
    """Run the Part 2C-2 pipeline: duplicate removal then type correction.

    Both stages operate on copies so ``df`` - and the stored uploaded file it
    was loaded from - is never modified. The returned
    :class:`DuplicateAndTypeCleaningResult` exposes the cleaned DataFrame for
    the next cleaning stage (Part 2C-3) together with the full report. A
    dataset with no duplicates is returned with no rows removed.
    """
    duplicate_stats = detect_duplicate_rows(df)

    working = remove_duplicate_rows(df)
    working, columns_processed, dtype_changes, conversions_skipped = correct_column_dtypes(
        working
    )

    return DuplicateAndTypeCleaningResult(
        cleaned_df=working,
        rows_before=int(duplicate_stats["rows_before"]),
        duplicate_rows_found=int(duplicate_stats["duplicate_rows_found"]),
        duplicate_rows_removed=int(duplicate_stats["duplicate_rows_found"]),
        rows_after=int(duplicate_stats["rows_after"]),
        duplicate_percentage=float(duplicate_stats["duplicate_percentage"]),
        duplicates_cleaned=bool(duplicate_stats["duplicate_rows_found"] > 0),
        columns_processed=columns_processed,
        dtype_changes=dtype_changes,
        conversions_skipped=conversions_skipped,
    )
@router.post(
    "/{dataset_id}/clean/duplicates-and-types",
    response_model=schemas.DataCleaningReport,
)
def clean_dataset_duplicates_and_types(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.DataCleaningReport:
    """Clean a dataset's duplicates and data types (Part 2C-2).

    The dataset is loaded from its stored file with the same pandas-based
    logic used elsewhere in the API and cleaned with the Part 2C-1
    missing-value pipeline (:func:`clean_missing_values`). The result is then
    passed through Part 2C-2 (:func:`clean_duplicates_and_types`), which
    removes duplicate rows (keeping the first occurrence) and safely corrects
    column data types. A structured cleaning report is returned; the uploaded
    file itself is never modified.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except _PARSING_ERRORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file during cleaning",
        )

    if len(df.columns) == 0 or len(df) == 0:
        # The project treats an empty dataset as invalid at upload time; report
        # the same validation error here for consistency with profiling.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be cleaned.",
        )

    # Part 2C-1 fills missing values first; this stage runs on that output so
    # Part 2C-2 never re-implements the missing-value logic.
    missing_result = clean_missing_values(df)
    result = clean_duplicates_and_types(missing_result.cleaned_df)

    return schemas.DataCleaningReport(
        rows_before=result.rows_before,
        duplicate_rows_found=result.duplicate_rows_found,
        duplicate_rows_removed=result.duplicate_rows_removed,
        rows_after=result.rows_after,
        duplicate_percentage=result.duplicate_percentage,
        duplicates_cleaned=result.duplicates_cleaned,
        columns_processed=result.columns_processed,
        dtype_changes=result.dtype_changes,
        conversions_skipped=result.conversions_skipped,
    )


# ---------------------------------------------------------------------------
# Part 2C-3A - Column name standardization
# ---------------------------------------------------------------------------

# Names given to columns that have no usable name (empty string, whitespace
# only, or None). They are numbered sequentially so they remain unique and one
# empty column never overwrites another.
EMPTY_COLUMN_PREFIX = "unnamed_column_"


def _normalize_column_name(name) -> str:
    """Return the standardized form of a single column ``name``.

    Applies, in order:

    * ``None`` / ``NaN`` become an empty string (handled by the caller as an
      unnamed column);
    * trim surrounding whitespace;
    * lowercase the text;
    * replace any run of non-alphanumeric characters (spaces, ``-``, ``.``,
      ``/``, ``#``, ``@``, ``()`` and friends) with a single underscore, which
      also collapses repeated underscores;
    * strip any leading/trailing underscores.

    Meaningful alphanumerics are preserved, so ``"Customer Name"``,
    ``"Order-Date"`` and ``"Order/Date"`` all standardize cleanly.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _assign_unique_names(pre_names: List[str]) -> List[str]:
    """Make every name in ``pre_names`` unique with ``_1``/``_2`` suffixes.

    The first occurrence of a name keeps it unchanged; later occurrences get an
    incrementing numeric suffix. A suffix collision with an already-assigned
    name (for example a pre-existing ``customer_name_1``) is resolved by
    continuing to increment, so the output is guaranteed unique regardless of
    the input. ``pre_names`` itself is not modified.
    """
    seen: Dict[str, int] = {}
    used: set = set()
    final: List[str] = []

    for name in pre_names:
        seen[name] = seen.get(name, 0) + 1
        candidate = name if seen[name] == 1 else f"{name}_{seen[name] - 1}"
        while candidate in used:
            seen[name] += 1
            candidate = f"{name}_{seen[name] - 1}"
        used.add(candidate)
        final.append(candidate)

    return final


@dataclass
class ColumnStandardizationResult:
    """The outcome of one Part 2C-3A standardization run.

    Holds the standardized DataFrame (ready for Part 2C-3B) and the run
    statistics that map onto
    :class:`schemas.ColumnNameStandardizationReport`.
    """

    cleaned_df: pd.DataFrame
    columns_processed: int
    columns_changed: int
    column_changes: Dict[str, str]
    empty_names_handled: int
    duplicate_names_handled: int
    final_column_names: List[str]


def standardize_column_names(df: pd.DataFrame) -> ColumnStandardizationResult:
    """Standardize the column names of ``df`` on a copy.

    The operation is performed on a ``copy()`` of ``df`` (so the caller/stored
    dataset is never modified) and only renames columns: row count, column
    count, values, data types and column order are all preserved. The returned
    ``cleaned_df`` exposes the standardized columns for the next cleaning stage
    (Part 2C-3B). Only the names change.
    """
    working = df.copy()
    original_names = [str(column) for column in working.columns]

    # 1. Normalize every name, then record how many had no usable content.
    normalized = [_normalize_column_name(column) for column in working.columns]
    empty_names_handled = sum(1 for base in normalized if base == "")

    # 2. Replace empty/unnamed columns with unique placeholders before the
    #    uniqueness pass so two empty columns never collide.
    pre_names: List[str] = []
    unnamed_counter = 0
    for base in normalized:
        if base == "":
            unnamed_counter += 1
            pre_names.append(f"{EMPTY_COLUMN_PREFIX}{unnamed_counter}")
        else:
            pre_names.append(base)

    # 3. Resolve any remaining collisions (duplicate normalized names) with
    #    suffixes, guaranteeing a unique final name per column.
    final_names = _assign_unique_names(pre_names)
    duplicate_names_handled = sum(
        1 for name, count in Counter(pre_names).items() if count > 1
    )

    # 4. Build the rename mapping and apply it.
    rename_map = dict(zip(working.columns, final_names))
    working = working.rename(columns=rename_map)

    column_changes = {
        original: final
        for original, final in zip(original_names, final_names)
        if original != final
    }

    return ColumnStandardizationResult(
        cleaned_df=working,
        columns_processed=len(original_names),
        columns_changed=len(column_changes),
        column_changes=column_changes,
        empty_names_handled=empty_names_handled,
        duplicate_names_handled=duplicate_names_handled,
        final_column_names=list(working.columns),
    )


@router.post(
    "/{dataset_id}/clean/standardize-columns",
    response_model=schemas.ColumnNameStandardizationReport,
)
def standardize_dataset_columns(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.ColumnNameStandardizationReport:
    """Standardize the column names of an uploaded dataset (Part 2C-3A).

    The dataset is loaded from its stored file with the same pandas-based logic
    used elsewhere in the API and its column names are standardized in-memory
    with :func:`standardize_column_names`. Only the column names are rewritten
    (trimmed, lowercased, separator/whitespace normalized, unnamed columns
    filled and duplicates suffixed); the data itself is untouched and the
    uploaded file is never modified. The standardized DataFrame is returned in
    the report for the next cleaning stage (Part 2C-3B).
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except _PARSING_ERRORS:
        raise HTTPException(  # pragma: no cover - defensive
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file during cleaning",
        )

    if len(df.columns) == 0 or len(df) == 0:
        # The project treats an empty dataset as invalid at upload time; report
        # the same validation error here for consistency with the other stages.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be cleaned.",
        )

    result = standardize_column_names(df)

    return schemas.ColumnNameStandardizationReport(
        columns_processed=result.columns_processed,
        columns_changed=result.columns_changed,
        column_changes=result.column_changes,
        empty_names_handled=result.empty_names_handled,
        duplicate_names_handled=result.duplicate_names_handled,
        final_column_names=result.final_column_names,
    )


# ---------------------------------------------------------------------------
# Part 2C-3B - Missing-like string values & whitespace cleanup
# ---------------------------------------------------------------------------

MISSING_LIKE_VALUES = frozenset(
    {
        "NA",
        "N/A",
        "na",
        "n/a",
        "NULL",
        "null",
        "None",
        "none",
        "NaN",
        "nan",
        "",
    }
)


@dataclass
class MissingAndStringCleaningResult:
    """The outcome of one Part 2C-3B cleaning run.

    Holds the cleaned DataFrame (ready for Part 2C-3C) together with the run
    statistics that map onto :class:`schemas.MissingAndStringCleaningReport`.
    """

    cleaned_df: pd.DataFrame
    missing_like_values_detected: int
    missing_like_values_handled: int
    whitespace_values_cleaned: int
    columns_affected: List[str]


def _is_string_column(series: pd.Series) -> bool:
    """Return True when ``series`` is a string/object column we should process."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return False
    return pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
        series.dtype
    )


def _clean_string_column(series: pd.Series) -> tuple[pd.Series, int, int]:
    """Clean a single string column and return (cleaned_series, missing_count, whitespace_count).

    * missing-like strings and empty/whitespace-only strings are replaced with ``pd.NA``.
    * remaining string values have leading/trailing whitespace stripped and
      repeated internal whitespace collapsed to a single space.
    """
    result = series.copy()
    missing_count = 0
    whitespace_count = 0

    mask = series.apply(lambda x: isinstance(x, str))
    if not mask.any():
        return result, 0, 0

    str_series = series[mask]

    stripped = str_series.str.strip()
    missing_mask = (stripped == "") | stripped.isin(MISSING_LIKE_VALUES)
    missing_count = int(missing_mask.sum())

    non_missing = str_series[~missing_mask]
    if len(non_missing) > 0:
        stripped_non_missing = non_missing.str.strip()
        needs_clean = (
            (non_missing != stripped_non_missing)
            | non_missing.str.contains(r"\s{2,}", regex=True, na=False)
        )
        whitespace_count = int(needs_clean.sum())

    if missing_count > 0:
        missing_indices = str_series[missing_mask].index
        result.loc[missing_indices] = pd.NA

    if whitespace_count > 0:
        clean_indices = non_missing[needs_clean].index
        to_clean = non_missing[needs_clean]
        cleaned_vals = to_clean.str.strip().str.replace(r"\s+", " ", regex=True)
        result.loc[clean_indices] = cleaned_vals

    return result, missing_count, whitespace_count


def clean_missing_like_and_string_values(
    df: pd.DataFrame,
) -> MissingAndStringCleaningResult:
    """Clean missing-like strings and whitespace in ``df`` on a copy.

    The operation is performed on a ``copy()`` of ``df`` so the original
    DataFrame is never modified. Only string/object (non-categorical) columns
    are processed; numerical, datetime, boolean and categorical columns are
    left untouched.

    For every string value that matches :data:`MISSING_LIKE_VALUES`, is empty,
    or is whitespace-only, the value is replaced with ``pd.NA`` so downstream
    missing-value handlers can fill it. Legitimate text values such as
    ``"Unknown"`` or ``"Other"`` are preserved.

    Remaining string values have leading/trailing whitespace stripped and
    repeated internal whitespace collapsed to a single space.
    """
    working = df.copy()
    total_missing_like = 0
    total_whitespace = 0
    columns_affected: List[str] = []

    for column in working.columns:
        if not _is_string_column(working[column]):
            continue

        cleaned_series, missing_count, whitespace_count = _clean_string_column(
            working[column]
        )

        if missing_count > 0 or whitespace_count > 0:
            columns_affected.append(str(column))
            working[column] = cleaned_series

        total_missing_like += missing_count
        total_whitespace += whitespace_count

    return MissingAndStringCleaningResult(
        cleaned_df=working,
        missing_like_values_detected=total_missing_like,
        missing_like_values_handled=total_missing_like,
        whitespace_values_cleaned=total_whitespace,
        columns_affected=columns_affected,
    )


@router.post(
    "/{dataset_id}/clean/missing-like-strings",
    response_model=schemas.MissingAndStringCleaningReport,
)
def clean_dataset_missing_like_strings(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.MissingAndStringCleaningReport:
    """Clean missing-like string values and whitespace in an uploaded dataset (Part 2C-3B).

    The dataset is loaded from its stored file, standardized with the Part 2C-3A
    pipeline (:func:`standardize_column_names`), then cleaned on an in-memory copy
    with :func:`clean_missing_like_and_string_values`. A report of the actions
    taken is returned; the uploaded file itself is never modified. The cleaned
    DataFrame is passed to Part 2C-3C for further processing.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except _PARSING_ERRORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file during cleaning",
        )

    if len(df.columns) == 0 or len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be cleaned.",
        )

    # Part 2C-3A standardizes column names first; this stage runs on that output
    # so Part 2C-3B never re-implements the name-standardization logic. The
    # cleaned DataFrame is returned for chaining into Part 2C-3C.
    standardized = standardize_column_names(df)
    result = clean_missing_like_and_string_values(standardized.cleaned_df)

    return schemas.MissingAndStringCleaningReport(
        missing_like_values_detected=result.missing_like_values_detected,
        missing_like_values_handled=result.missing_like_values_handled,
        whitespace_values_cleaned=result.whitespace_values_cleaned,
        columns_affected=result.columns_affected,
    )


# ---------------------------------------------------------------------------
# Part 2C-3C - Numeric & special value handling
# ---------------------------------------------------------------------------

# Single-character currency symbols that unambiguously mark a number when they
# sit at the very start or very end of a value. A symbol embedded in a longer
# text (e.g. "price$abc") is never treated as currency.
CURRENCY_SYMBOLS = frozenset({"$", "€", "£", "¥", "₹", "₩", "₽", "₴", "₦", "¢"})

# Infinity-like tokens, matched case-insensitively after trimming.
POSITIVE_INFINITY_TOKENS = frozenset(
    {"inf", "+inf", "infinity", "+infinity", "∞", "+∞"}
)
NEGATIVE_INFINITY_TOKENS = frozenset({"-inf", "-infinity", "-∞"})

# A plain number with an optional sign and decimal part: "123", "12.5", ".5".
_PLAIN_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
# A number whose commas group digits in threes: "1,000", "12,345,678".
_GROUPED_NUMERIC_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _is_positive_infinity(value) -> bool:
    """Return True when ``value`` is a positive-infinity token like ``inf``."""
    return isinstance(value, str) and value.strip().lower() in POSITIVE_INFINITY_TOKENS


def _is_negative_infinity(value) -> bool:
    """Return True when ``value`` is a negative-infinity token like ``-inf``."""
    return isinstance(value, str) and value.strip().lower() in NEGATIVE_INFINITY_TOKENS


def _normalize_numeric_string(value: str) -> Optional[tuple]:
    """Normalize an unambiguously numeric string; return ``None`` when ambiguous.

    Handles a single leading/trailing currency symbol and 3-digit thousands
    separators, e.g. ``"$1,000" -> ("1000", True, True)``,
    ``"₹2,500" -> ("2500", True, True)`` and ``"1,500" -> ("1500", False, True)``.

    Any value that cannot be interpreted unambiguously as a plain number
    returns ``None`` - ``"ABC123"``, ``"100-200"`` (a range), ``"Unknown"``,
    ``"not available"``, ``"1e5"``, decimal-comma ``"1,5"``, empty strings and
    infinity tokens all fall through so the caller preserves the original text
    instead of inventing a number.
    """
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if (
        lowered in POSITIVE_INFINITY_TOKENS
        or lowered in NEGATIVE_INFINITY_TOKENS
    ):
        # Infinity is a special value handled separately, never a plain number.
        return None

    digits_part = text
    currency = False
    if len(text) > 1:
        head = text[0]
        tail = text[-1]
        if head in CURRENCY_SYMBOLS and tail not in CURRENCY_SYMBOLS:
            currency = True
            digits_part = text[1:]
        elif tail in CURRENCY_SYMBOLS and head not in CURRENCY_SYMBOLS:
            currency = True
            digits_part = text[:-1]

    # A plain number must not contain letters, so "ABC123", "1e5", "12abc"
    # and anything similarly malformed is rejected here.
    if re.search(r"[A-Za-z]", digits_part):
        return None

    if _GROUPED_NUMERIC_RE.match(digits_part):
        return digits_part.replace(",", ""), currency, True
    if _PLAIN_NUMERIC_RE.match(digits_part):
        return digits_part, currency, False
    return None


@dataclass
class NumericAndSpecialValueCleaningResult:
    """The outcome of one Part 2C-3C cleaning run.

    Holds the cleaned DataFrame (ready for Part 2C-3D) together with the run
    statistics that map onto :class:`schemas.NumericAndSpecialValueCleaningReport`.
    """

    cleaned_df: pd.DataFrame
    numeric_values_normalized: int
    currency_values_handled: int
    thousands_separators_handled: int
    positive_infinity_handled: int
    negative_infinity_handled: int
    unsafe_conversions_skipped: int
    conversion_skipped_columns: List[str]
    columns_processed: Dict[str, schemas.NumericColumnCleaningReport]


def clean_numeric_and_special_values(
    df: pd.DataFrame,
) -> NumericAndSpecialValueCleaningResult:
    """Normalize obvious numeric and special values in ``df`` on a copy (Part 2C-3C).

    The operation is performed on a ``copy()`` so the original DataFrame - and
    the uploaded file it was loaded from - is never modified. Only the value
    representation changes; row/column counts, column names and column order
    are all preserved.

    * Numeric-looking strings such as ``"$1,000"``, ``"₹2,500"`` and ``"1,500"``
      are normalized to their plain numeric form (``"1000"``, ``"2500"``,
      ``"1500"``) when the interpretation is unambiguous.
    * Values that cannot be interpreted unambiguously (``"ABC123"``,
      ``"100-200"``, ``"Unknown"``, ``"not available"``) are preserved exactly.
    * ``inf``/``-inf``/``Infinity``/``-Infinity`` - as strings or real floats -
      are turned into missing values so the existing missing-value fill can
      handle them; no rows are ever deleted.
    * No dtype coercion is performed. A column whose conversion to numbers
      would destroy values is measured, rejected and reported in
      ``conversion_skipped_columns`` with its original representation preserved.
    """
    working = df.copy()
    numeric_values_normalized = 0
    currency_values_handled = 0
    thousands_separators_handled = 0
    positive_infinity_handled = 0
    negative_infinity_handled = 0
    unsafe_conversions_skipped = 0
    conversion_skipped_columns: List[str] = []
    columns_processed: Dict[str, schemas.NumericColumnCleaningReport] = {}

    for column in working.columns:
        col_name = str(column)
        series = working[column]
        original_dtype = str(series.dtype)
        dtype = series.dtype

        values_normalized = 0
        currency_handled = 0
        thousands_handled = 0
        pos_inf = 0
        neg_inf = 0
        unsafe = False

        if pd.api.types.is_bool_dtype(dtype):
            # Boolean columns are categorical: never treated as numeric.
            columns_processed[col_name] = schemas.NumericColumnCleaningReport(
                original_dtype=original_dtype,
                new_dtype=original_dtype,
            )
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            if pd.api.types.is_float_dtype(dtype):
                # Real infinities in a float column become missing values.
                pos_mask = series == float("inf")
                neg_mask = series == float("-inf")
                pos_inf = int(pos_mask.sum())
                neg_inf = int(neg_mask.sum())
                if pos_inf > 0 or neg_inf > 0:
                    working[column] = series.mask(pos_mask | neg_mask, np.nan)
        elif _is_string_column(series):
            rebuilt = []
            for value in series.tolist():
                if isinstance(value, str) and _is_positive_infinity(value):
                    rebuilt.append(pd.NA)
                    pos_inf += 1
                    continue
                if isinstance(value, str) and _is_negative_infinity(value):
                    rebuilt.append(pd.NA)
                    neg_inf += 1
                    continue
                if isinstance(value, float) and math.isnan(value):
                    # NaN stays a missing value; it is not an infinity.
                    rebuilt.append(pd.NA)
                    continue
                if isinstance(value, float) and not math.isfinite(value):
                    # A real float infinity sitting inside an object column.
                    if value > 0:
                        pos_inf += 1
                    else:
                        neg_inf += 1
                    rebuilt.append(pd.NA)
                    continue
                if isinstance(value, str):
                    parsed = _normalize_numeric_string(value)
                    if parsed is not None:
                        cleaned, has_currency, has_thousands = parsed
                        rebuilt.append(cleaned)
                        if cleaned != value.strip():
                            values_normalized += 1
                        if has_currency:
                            currency_handled += 1
                        if has_thousands:
                            thousands_handled += 1
                        continue
                # Ambiguous or non-numeric text: preserved untouched.
                rebuilt.append(value)

            if (
                values_normalized > 0
                or currency_handled > 0
                or thousands_handled > 0
                or pos_inf > 0
                or neg_inf > 0
            ):
                working[column] = pd.Series(rebuilt, index=series.index, dtype=object)

            # Numeric safety: measure how many non-null values would survive a
            # numeric conversion of the (now normalized) column. When there is
            # real numeric content but junk tokens would be destroyed, reject
            # the conversion and keep every value exactly as it is.
            try:
                converted = pd.to_numeric(working[column], errors="coerce")
            except (TypeError, ValueError, OverflowError):
                converted = pd.Series(
                    [np.nan] * len(working[column]), index=working[column].index
                )
            ratio = _conversion_success_ratio(working[column], converted)
            numeric_count = int(converted.notna().sum())
            invalid_count = int(working[column].notna().sum()) - numeric_count
            if (
                numeric_count > 0
                and invalid_count > 0
                and ratio < NUMERIC_CONVERSION_THRESHOLD
            ):
                unsafe = True
                unsafe_conversions_skipped += 1
                conversion_skipped_columns.append(col_name)

        numeric_values_normalized += values_normalized
        currency_values_handled += currency_handled
        thousands_separators_handled += thousands_handled
        positive_infinity_handled += pos_inf
        negative_infinity_handled += neg_inf

        columns_processed[col_name] = schemas.NumericColumnCleaningReport(
            original_dtype=original_dtype,
            new_dtype=str(working[column].dtype),
            values_normalized=values_normalized,
            positive_infinity_handled=pos_inf,
            negative_infinity_handled=neg_inf,
            unsafe_conversion_skipped=unsafe,
        )

    return NumericAndSpecialValueCleaningResult(
        cleaned_df=working,
        numeric_values_normalized=numeric_values_normalized,
        currency_values_handled=currency_values_handled,
        thousands_separators_handled=thousands_separators_handled,
        positive_infinity_handled=positive_infinity_handled,
        negative_infinity_handled=negative_infinity_handled,
        unsafe_conversions_skipped=unsafe_conversions_skipped,
        conversion_skipped_columns=conversion_skipped_columns,
        columns_processed=columns_processed,
    )


@router.post(
    "/{dataset_id}/clean/numeric-special",
    response_model=schemas.NumericAndSpecialValueCleaningReport,
)
def clean_dataset_numeric_special_values(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.NumericAndSpecialValueCleaningReport:
    """Clean numeric formatting and special values in an uploaded dataset (Part 2C-3C).

    The dataset is loaded from its stored file, standardized with the Part 2C-3A
    pipeline (:func:`standardize_column_names`), run through the Part 2C-3B
    missing-like-string stage (:func:`clean_missing_like_and_string_values`) and
    finally cleaned with :func:`clean_numeric_and_special_values`. A report of
    the actions taken is returned; the uploaded file itself is never modified.
    The cleaned DataFrame is passed to Part 2C-3D for further processing.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except _PARSING_ERRORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file during cleaning",
        )

    if len(df.columns) == 0 or len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be cleaned.",
        )

    # Part 2C-3A standardizes column names, then Part 2C-3B cleans missing-like
    # strings; this stage runs on that output so it never re-implements the
    # earlier logic. The cleaned DataFrame is returned for chaining into 2C-3D.
    standardized = standardize_column_names(df)
    missing_like = clean_missing_like_and_string_values(standardized.cleaned_df)
    result = clean_numeric_and_special_values(missing_like.cleaned_df)

    # Preservation verification is measured for THIS stage only: the input to
    # ``clean_numeric_and_special_values`` is the 2C-3B output DataFrame.
    before_df = missing_like.cleaned_df
    after_df = result.cleaned_df

    return schemas.NumericAndSpecialValueCleaningReport(
        numeric_values_normalized=result.numeric_values_normalized,
        currency_values_handled=result.currency_values_handled,
        thousands_separators_handled=result.thousands_separators_handled,
        positive_infinity_handled=result.positive_infinity_handled,
        negative_infinity_handled=result.negative_infinity_handled,
        unsafe_conversions_skipped=result.unsafe_conversions_skipped,
        conversion_skipped_columns=result.conversion_skipped_columns,
        columns_processed=result.columns_processed,
        rows_before=int(len(before_df)),
        rows_after=int(len(after_df)),
        columns_before=int(len(before_df.columns)),
        columns_after=int(len(after_df.columns)),
        column_names_unchanged=list(before_df.columns) == list(after_df.columns),
    )
# ---------------------------------------------------------------------------
# Part 2C-3D - Final value validation & cleaning report
# ---------------------------------------------------------------------------

# Validation statuses used across the project's cleaning flow.
STATUS_PASSED = "passed"
STATUS_WARNING = "warning"
STATUS_FAILED = "failed"


@dataclass
class FinalValidationResult:
    """The outcome of the final Part 2C-3 validation run.

    Holds the fully cleaned DataFrame (ready for Part 2C-4A) together with the
    combined :class:`schemas.FinalCleaningReport` that merges the Part 2C-3A,
    2C-3B and 2C-3C outcomes with the final validation checks.
    """

    cleaned_df: pd.DataFrame
    report: schemas.FinalCleaningReport


def _count_remaining_string_issues(df: pd.DataFrame) -> tuple[int, int]:
    """Count missing-like tokens and whitespace-only strings still present.

    Scans every column (including categorical columns that the Part 2C-3B
    string stage deliberately leaves untouched) for string values that look
    like missing data or that are whitespace-only. Actual missing cells
    (``NaN`` / ``pd.NA`` / ``None``) are not counted - they are the normal
    representation of a missing value.
    """
    missing_like = 0
    whitespace_only = 0
    for column in df.columns:
        for value in df[column].tolist():
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped == "":
                whitespace_only += 1
            elif stripped.lower() in MISSING_LIKE_VALUES:
                missing_like += 1
    return missing_like, whitespace_only


def _count_remaining_non_finite(df: pd.DataFrame) -> tuple[int, int]:
    """Count positive and negative infinity values still remaining.

    Considers real float infinities in numeric columns as well as string
    infinity tokens and real float infinities nested inside object/categorical
    columns. ``NaN`` is not an infinity and is ignored.
    """
    pos = 0
    neg = 0
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_float_dtype(series.dtype):
            pos += int((series == float("inf")).sum())
            neg += int((series == float("-inf")).sum())
        else:
            for value in series.tolist():
                if (
                    isinstance(value, float)
                    and not math.isfinite(value)
                    and not math.isnan(value)
                ):
                    if value > 0:
                        pos += 1
                    else:
                        neg += 1
                elif isinstance(value, str):
                    if _is_positive_infinity(value):
                        pos += 1
                    elif _is_negative_infinity(value):
                        neg += 1
    return pos, neg


def _evaluate_status(
    row_count_unchanged: bool,
    column_count_unchanged: bool,
    unique_names: bool,
    no_empty_names: bool,
    standardized_names: bool,
    remaining_pos_inf: int,
    remaining_neg_inf: int,
    remaining_missing: int,
    remaining_whitespace: int,
    remaining_invalid_numeric: int,
) -> str:
    """Derive the overall Part 2C-3 validation status.

    * ``failed``     - a structural problem makes the dataset unsafe to continue
      (unexpected row/column removal, invalid or duplicate column names, or
      infinity values that the numeric stage was supposed to clean).
    * ``warning``    - minor quality issues remain (missing-like, whitespace or
      invalid-numeric values) but the dataset can still proceed to Part 2C-4A.
    * ``passed``     - the cleaned DataFrame satisfies every expected rule.
    """
    structural_problem = (
        not row_count_unchanged
        or not column_count_unchanged
        or not unique_names
        or not no_empty_names
        or not standardized_names
        or remaining_pos_inf > 0
        or remaining_neg_inf > 0
    )
    if structural_problem:
        return STATUS_FAILED
    if (
        remaining_missing > 0
        or remaining_whitespace > 0
        or remaining_invalid_numeric > 0
    ):
        return STATUS_WARNING
    return STATUS_PASSED
def validate_and_report(df: pd.DataFrame) -> FinalValidationResult:
    """Run the final Part 2C-3 validation over the whole 2C-3 pipeline.

    Chains the existing Part 2C-3A (column-name standardization), 2C-3B
    (missing-like/string) and 2C-3C (numeric/special) stages on in-memory
    copies - reusing their logic rather than re-implementing any cleaning - then
    performs the final, read-only value validation and the before/after
    data-integrity checks. Any problem that remains is reported but never
    automatically modified here, and no rows or columns are removed.

    Returns a :class:`FinalValidationResult` exposing the fully cleaned
    DataFrame (ready for Part 2C-4A) together with the combined report.
    """
    before = df.copy()

    # Chain the three Part 2C-3 cleaning stages in order; each reuses the
    # output of the previous one so no cleaning logic is duplicated.
    standardized = standardize_column_names(before)
    missing_like = clean_missing_like_and_string_values(standardized.cleaned_df)
    numeric = clean_numeric_and_special_values(missing_like.cleaned_df)
    after = numeric.cleaned_df

    # ---- Final value validation (read-only; nothing is modified here) ----
    remaining_missing, remaining_whitespace = _count_remaining_string_issues(after)
    remaining_pos_inf, remaining_neg_inf = _count_remaining_non_finite(after)
    remaining_invalid_numeric = remaining_pos_inf + remaining_neg_inf

    # ---- Final column validation (names are standardized by 2C-3A; never
    #      renamed again in this stage) ----
    final_names = list(after.columns)
    unique_names = len(set(final_names)) == len(final_names)
    no_empty_names = all(str(name).strip() != "" for name in final_names)
    standardized_names = all(
        str(name).strip() != "" and _normalize_column_name(name) == str(name)
        for name in final_names
    )

    # ---- Before/after data integrity ----
    rows_before, rows_after = int(len(before)), int(len(after))
    columns_before, columns_after = int(len(before.columns)), int(len(after.columns))
    row_count_unchanged = rows_before == rows_after
    column_count_unchanged = columns_before == columns_after
    total_values_before = rows_before * columns_before
    total_values_after = rows_after * columns_after

    status = _evaluate_status(
        row_count_unchanged,
        column_count_unchanged,
        unique_names,
        no_empty_names,
        standardized_names,
        remaining_pos_inf,
        remaining_neg_inf,
        remaining_missing,
        remaining_whitespace,
        remaining_invalid_numeric,
    )

    return FinalValidationResult(
        cleaned_df=after,
        report=schemas.FinalCleaningReport(
            status=status,
            column_standardization=_build_column_standardization_summary(standardized),
            string_cleaning=_build_string_cleaning_summary(missing_like),
            numeric_cleaning=_build_numeric_cleaning_summary(numeric),
            validation=_build_validation_summary(
                status,
                rows_before,
                rows_after,
                columns_before,
                columns_after,
                row_count_unchanged,
                column_count_unchanged,
                unique_names,
                no_empty_names,
                standardized_names,
                total_values_before,
                total_values_after,
                remaining_missing,
                remaining_whitespace,
                remaining_invalid_numeric,
                remaining_pos_inf,
                remaining_neg_inf,
            ),
        ),
    )

def _build_column_standardization_summary(
    result: ColumnStandardizationResult,
) -> schemas.ColumnStandardizationSummary:
    """Summarize a Part 2C-3A result for the combined Part 2C-3 report."""
    return schemas.ColumnStandardizationSummary(
        columns_processed=result.columns_processed,
        columns_changed=result.columns_changed,
        empty_names_handled=result.empty_names_handled,
        duplicate_names_handled=result.duplicate_names_handled,
        final_column_names=result.final_column_names,
    )


def _build_string_cleaning_summary(
    result: MissingAndStringCleaningResult,
) -> schemas.StringCleaningSummary:
    """Summarize a Part 2C-3B result for the combined Part 2C-3 report."""
    return schemas.StringCleaningSummary(
        missing_like_values_handled=result.missing_like_values_handled,
        whitespace_values_cleaned=result.whitespace_values_cleaned,
        columns_affected=result.columns_affected,
    )


def _build_numeric_cleaning_summary(
    result: NumericAndSpecialValueCleaningResult,
) -> schemas.NumericCleaningSummary:
    """Summarize a Part 2C-3C result for the combined Part 2C-3 report."""
    return schemas.NumericCleaningSummary(
        numeric_values_normalized=result.numeric_values_normalized,
        currency_values_handled=result.currency_values_handled,
        thousands_separators_handled=result.thousands_separators_handled,
        positive_infinity_handled=result.positive_infinity_handled,
        negative_infinity_handled=result.negative_infinity_handled,
        unsafe_conversions_skipped=result.unsafe_conversions_skipped,
    )


def _build_validation_summary(
    status: str,
    rows_before: int,
    rows_after: int,
    columns_before: int,
    columns_after: int,
    row_count_unchanged: bool,
    column_count_unchanged: bool,
    unique_names: bool,
    no_empty_names: bool,
    standardized_names: bool,
    total_values_before: int,
    total_values_after: int,
    remaining_missing: int,
    remaining_whitespace: int,
    remaining_invalid_numeric: int,
    remaining_pos_inf: int,
    remaining_neg_inf: int,
) -> schemas.ValidationSummary:
    """Build the final validation summary for the combined Part 2C-3 report."""
    return schemas.ValidationSummary(
        status=status,
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=columns_before,
        columns_after=columns_after,
        row_count_unchanged=row_count_unchanged,
        column_count_unchanged=column_count_unchanged,
        unique_column_names=unique_names,
        no_empty_column_names=no_empty_names,
        standardized_column_names=standardized_names,
        total_values_before=total_values_before,
        total_values_after=total_values_after,
        remaining_missing_like_values=remaining_missing,
        remaining_whitespace_only_values=remaining_whitespace,
        remaining_invalid_numeric_values=remaining_invalid_numeric,
        remaining_pos_infinity=remaining_pos_inf,
        remaining_neg_infinity=remaining_neg_inf,
    )


@router.post(
    "/{dataset_id}/clean/final-validation",
    response_model=schemas.FinalCleaningReport,
)
def validate_dataset_final(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.FinalCleaningReport:
    """Run the final Part 2C-3 validation and cleaning report (Part 2C-3D).

    The dataset is loaded from its stored file, run through the Part 2C-3A,
    2C-3B and 2C-3C stages, and then validated with
    :func:`validate_and_report`. The returned report combines the three stage
    outcomes with the final value validation and before/after data-integrity
    checks. No rows or columns are removed and nothing is modified during this
    validation; the cleaned DataFrame is provided in-memory for chaining into
    Part 2C-4A and the uploaded file itself is never modified.
    """
    db_dataset = get_owned_dataset_or_raise(dataset_id, db, current_user)

    try:
        df = load_dataset_file(db_dataset.file_path, db_dataset.original_filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )
    except _PARSING_ERRORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty, corrupted, or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process dataset file during cleaning",
        )

    if len(df.columns) == 0 or len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is empty and cannot be cleaned.",
        )

    return validate_and_report(df).report

