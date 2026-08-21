from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import List, Dict, Any, Optional


# User Schemas
class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int

    class Config:
        from_attributes = True


# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# Dataset Schemas
class DatasetBase(BaseModel):
    original_filename: str
    file_type: str
    file_size: float
    rows: int
    columns: int


class Dataset(DatasetBase):
    id: int
    upload_date: datetime

    class Config:
        from_attributes = True


class DatasetDetails(Dataset):
    column_info: Dict[str, str]
    preview: List[Dict[str, Any]]

    class Config:
        from_attributes = True


class DatasetStructure(Dataset):
    """Structural profile (Part 2B-1) of a single uploaded dataset.

    Exposes the number of rows, the number of columns, the column names and
    the pandas data type of every column, as detected from the stored file.
    """
    column_names: List[str]
    data_types: Dict[str, str]

    class Config:
        from_attributes = True


# Missing-value analysis (Part 2B-2)
class ColumnMissingValue(BaseModel):
    """Missing-value counts for a single column (Part 2B-2)."""
    missing_count: int
    missing_percentage: float


class DatasetMissingValueProfile(BaseModel):
    """Missing-value analysis (Part 2B-2) of a single uploaded dataset.

    Reports the total row count, the total number of missing values across the
    whole dataset and, for every column, how many values are missing and their
    percentage of the column.
    """
    total_rows: int
    total_missing_values: int
    columns: Dict[str, ColumnMissingValue]


# Duplicate detection & column classification (Part 2B-3)
class DatasetQualityProfile(BaseModel):
    """Duplicate and column-type profile (Part 2B-3) of a single uploaded dataset.

    Reports two read-only quality signals:

    * ``duplicate_rows`` - the number of rows that are duplicates of an earlier
      row in the dataset (pandas' ``Series.duplicated()`` semantics, so the
      first occurrence of an identical row is never counted).
    * The column classifications, based on the real pandas dtypes: numeric
      columns (int/float and other pandas numeric dtypes) are listed as
      ``numerical_columns``, non-numeric columns (object/string, ``category``,
      ``bool`` and other non-numeric dtypes) are listed as ``categorical_columns``,
      and any ``datetime`` columns are kept separate in ``datetime_columns`` so
      they are not forced into either category.

    Datasets without datetime columns simply return an empty ``datetime_columns``
    list. Nothing here modifies the underlying dataset; this is read-only.
    """
    duplicate_rows: int
    numerical_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str] = []


# Basic dataset summary (Part 2B-4)
class NumericalColumnSummary(BaseModel):
    """Basic descriptive statistics for a single numerical column (Part 2B-4).

    Mirrors pandas' ``Series.describe()`` on a numerical column. The quartiles
    are exposed in the JSON response under the ``25%``, ``50%`` and ``75%``
    keys (matching the percentile labels pandas produces). Fields are optional
    because the statistics are ``None`` when a column has no usable values
    (for example, a column whose values are all missing).
    """

    model_config = ConfigDict(populate_by_name=True)

    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = Field(alias="25%", default=None)
    p50: Optional[float] = Field(alias="50%", default=None)
    p75: Optional[float] = Field(alias="75%", default=None)
    max: Optional[float] = None


class CategoricalColumnSummary(BaseModel):
    """Basic summary of a single categorical column (Part 2B-4).

    Reports the non-null count, the number of unique non-null values, the most
    frequent value and how often that most frequent value occurs. ``top`` is
    ``None`` when the column has no non-null values.
    """

    count: int
    unique: int
    top: Optional[Any] = None
    frequency: int


class DatasetSummary(BaseModel):
    """Basic dataset summary (Part 2B-4) of a single uploaded dataset.

    Separately reports the summary of every numerical column (count, mean,
    standard deviation, min/max and the quartiles) and the summary of every
    categorical column (count, unique, most frequent value and its frequency).
    Consistent with Partition 2B-3, datetime columns are kept separate and are
    therefore not included in either summary.
    """

    numerical_summary: Dict[str, NumericalColumnSummary]
    categorical_summary: Dict[str, CategoricalColumnSummary]
# Missing-value handling (Part 2C-1)
class ColumnCleaningReport(BaseModel):
    """Cleaning action applied to a single column (Part 2C-1).

    ``strategy`` records how a column's missing values were filled
    (``"median"`` for numerical columns, ``"mode"`` for categorical/text
    columns with a usable mode, and ``"Unknown"`` for categorical columns
    without a mode). ``values_filled`` is the number of missing cells that
    were filled in that column.
    """
    strategy: str
    values_filled: int


class MissingValueCleaningReport(BaseModel):
    """Result of the missing-value cleaning operation (Part 2C-1).

    Reports the total number of missing values before and after cleaning, the
    per-column action taken for every affected column, and the names of any
    completely-empty columns that were removed. Datasets with no missing
    values simply return this with ``cleaning_applied`` set to ``False`` so a
    caller can tell at a glance that no cleaning was needed.
    """
    missing_values_before: int
    missing_values_after: int
    columns_processed: Dict[str, ColumnCleaningReport]
    removed_empty_columns: List[str] = []
    cleaning_applied: bool = False


# Duplicate removal & data type correction (Part 2C-2)
class ColumnDtypeCleaningReport(BaseModel):
    """Data type outcome for a single column (Part 2C-2).

    ``original_dtype`` is the pandas dtype of the column before any
    correction (for example ``object`` for numeric strings read from a CSV),
    ``new_dtype`` is the dtype after the cleaning step, and ``action`` is one
    of:

    * ``"unchanged"``           - the column kept its values and dtype;
    * ``"converted_to_numeric"``- numeric strings were parsed to int/float;
    * ``"converted_to_datetime"`` - date-like strings became ``datetime64``;
    * ``"converted_to_boolean"``  - boolean-like values became a bool dtype;
    * ``"conversion_skipped"``  - a conversion was considered but skipped to
      avoid destroying values (the original representation was preserved).
    """

    original_dtype: str
    new_dtype: str
    action: str


class DataCleaningReport(BaseModel):
    """Result of Part 2C-2: duplicate removal and safe data type correction.

    Reports the row counts before and after duplicate removal, the number of
    duplicate rows found and removed, the duplicate percentage, and, for every
    column, the original and new pandas dtype plus the action taken.
    ``dtype_changes`` is the count of columns whose dtype actually changed and
    ``conversions_skipped`` lists every column where a conversion was
    considered but skipped to avoid data loss.
    """

    rows_before: int
    duplicate_rows_found: int
    duplicate_rows_removed: int
    rows_after: int
    duplicate_percentage: float = 0.0
    duplicates_cleaned: bool = False
    columns_processed: Dict[str, ColumnDtypeCleaningReport]
    dtype_changes: int = 0
    conversions_skipped: List[str] = []


# Column name standardization (Part 2C-3A)
class ColumnNameStandardizationReport(BaseModel):
    """Result of the column-name standardization operation (Part 2C-3A).

    Every column name is trimmed, lowercased, had its whitespace and special
    characters normalized to underscores, and had any repeated or leading/trailing
    underscores collapsed, producing dashboard-friendly names. Empty/unnamed
    columns are given unique ``unnamed_column_N`` placeholders and collisions
    after normalization are resolved with ``_1``, ``_2`` ... suffixes so no column
    ever overwrites another. Only the names change: row/column counts, values,
    data types and column order are all preserved.

    ``column_changes`` maps each original column name (as a string) to its new
    name for every column whose name was modified. ``duplicate_names_handled``
    is the number of distinct (pre-suffix) normalized names that collided with an
    earlier column, i.e. the number of duplicate-name groups resolved.
    """

    columns_processed: int
    columns_changed: int
    column_changes: Dict[str, str]
    empty_names_handled: int = 0
    duplicate_names_handled: int = 0
    final_column_names: List[str]


# Missing-like & string value cleaning (Part 2C-3B)
class MissingAndStringCleaningReport(BaseModel):
    """Result of the missing-like string and whitespace cleaning operation (Part 2C-3B).

    Reports how many string values that look like missing data were detected and
    replaced with ``NaN``, how many whitespace-only values were handled, and how
    many legitimate string values had their leading/trailing/repeated internal
    whitespace cleaned. ``columns_affected`` lists every column that was modified.
    """

    missing_like_values_detected: int
    missing_like_values_handled: int
    whitespace_values_cleaned: int
    columns_affected: List[str]


# Numeric & special value cleaning (Part 2C-3C)
class NumericColumnCleaningReport(BaseModel):
    """Numeric/special outcome for a single column (Part 2C-3C).

    ``original_dtype`` is the pandas dtype of the column before this stage and
    ``new_dtype`` is the dtype after it. Because this stage only normalizes
    numeric *formatting* (currency symbols, thousands separators) and special
    values (infinities) on a copy, the dtype usually stays unchanged.
    ``values_normalized`` is the number of cells whose numeric representation
    was rewritten, ``positive_infinity_handled`` / ``negative_infinity_handled``
    are the counts of each infinity turned into a missing value, and
    ``unsafe_conversion_skipped`` is ``True`` when a numeric conversion was
    considered for the column but rejected because it would have destroyed
    values (the original representation was preserved).
    """

    original_dtype: str
    new_dtype: str
    values_normalized: int = 0
    positive_infinity_handled: int = 0
    negative_infinity_handled: int = 0
    unsafe_conversion_skipped: bool = False


class NumericAndSpecialValueCleaningReport(BaseModel):
    """Result of the numeric & special value cleaning operation (Part 2C-3C).

    Reports how many values had their numeric formatting normalized, how many
    currency-symbol values and thousands separators were handled, how many
    positive/negative infinities became missing values, and how many numeric
    conversions were considered but skipped to avoid destroying data. The
    ``*_before``/``*_after`` fields and ``column_names_unchanged`` verify that
    row/column counts and the column names were all preserved unchanged.
    """

    numeric_values_normalized: int = 0
    currency_values_handled: int = 0
    thousands_separators_handled: int = 0
    positive_infinity_handled: int = 0
    negative_infinity_handled: int = 0
    unsafe_conversions_skipped: int = 0
    conversion_skipped_columns: List[str] = []
    columns_processed: Dict[str, NumericColumnCleaningReport]
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    column_names_unchanged: bool = True


# Final validation & cleaning report (Part 2C-3D)
class ColumnStandardizationSummary(BaseModel):
    """Summarized outcome of the Part 2C-3A column-name standardization.

    ``columns_processed`` is the number of columns considered, ``columns_changed``
    how many were actually renamed, and ``empty_names_handled`` /
    ``duplicate_names_handled`` how many unnamed and colliding names were
    resolved. ``final_column_names`` lists the standardized, unique names that
    the pipeline produced.
    """

    columns_processed: int
    columns_changed: int
    empty_names_handled: int = 0
    duplicate_names_handled: int = 0
    final_column_names: List[str] = []


class StringCleaningSummary(BaseModel):
    """Summarized outcome of the Part 2C-3B missing-like/string cleaning.

    ``missing_like_values_handled`` is how many missing-like strings were
    turned into missing values and ``whitespace_values_cleaned`` how many
    legitimate strings had whitespace cleaned. ``columns_affected`` lists every
    column that was modified.
    """

    missing_like_values_handled: int = 0
    whitespace_values_cleaned: int = 0
    columns_affected: List[str] = []


class NumericCleaningSummary(BaseModel):
    """Summarized outcome of the Part 2C-3C numeric/special cleaning.

    Reports how many values had numeric formatting normalized and how many
    currency symbols / thousands separators were handled, plus how many
    positive/negative infinities became missing values and how many numeric
    conversions were considered but skipped to avoid destroying data.
    """

    numeric_values_normalized: int = 0
    currency_values_handled: int = 0
    thousands_separators_handled: int = 0
    positive_infinity_handled: int = 0
    negative_infinity_handled: int = 0
    unsafe_conversions_skipped: int = 0


class ValidationSummary(BaseModel):
    """Final Part 2C-3 validation and before/after data-integrity checks.

    ``rows_before`` / ``rows_after`` and ``columns_before`` / ``columns_after``
    record the counts before and after the Part 2C-3 pipeline; the
    ``*_unchanged`` flags confirm that no rows or columns were unexpectedly
    removed. ``unique_column_names``, ``no_empty_column_names`` and
    ``standardized_column_names`` confirm the column names are valid and use the
    standardized convention. The ``remaining_*`` fields count any missing-like,
    whitespace-only, invalid-numeric or infinity values that survived cleaning,
    and ``status`` is one of ``passed`` / ``warning`` / ``failed``.
    """

    status: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    row_count_unchanged: bool
    column_count_unchanged: bool
    unique_column_names: bool
    no_empty_column_names: bool
    standardized_column_names: bool
    total_values_before: int = 0
    total_values_after: int = 0
    remaining_missing_like_values: int = 0
    remaining_whitespace_only_values: int = 0
    remaining_invalid_numeric_values: int = 0
    remaining_pos_infinity: int = 0
    remaining_neg_infinity: int = 0


class FinalCleaningReport(BaseModel):
    """Combined Part 2C-3 report: final validation + the 2C-3A/2C-3B/2C-3C reports.

    Merges the column-standardization, string-cleaning and numeric-cleaning
    summaries with the final validation checks and an overall ``status`` of
    ``passed``, ``warning`` or ``failed``. A ``failed`` dataset has structural
    problems that make it unsafe to continue to Part 2C-4A.
    """

    status: str
    column_standardization: ColumnStandardizationSummary
    string_cleaning: StringCleaningSummary
    numeric_cleaning: NumericCleaningSummary
    validation: ValidationSummary