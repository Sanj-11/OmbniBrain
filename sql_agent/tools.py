import pandas as pd


def load_data(file_path: str, original_filename: str):
    """Load CSV or Excel dataset into a pandas DataFrame."""

    if original_filename.lower().endswith(".csv"):
        return pd.read_csv(file_path)

    if original_filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    raise ValueError("Unsupported file type")


def get_dataset_schema(df: pd.DataFrame) -> str:
    """Return a simple schema description of the dataset."""

    schema_lines = []

    for column in df.columns:
        schema_lines.append(
            f"{column}: {df[column].dtype}"
        )

    return "\n".join(schema_lines)


def execute_sql_query(df: pd.DataFrame, sql_query: str):
    """Execute a SQL query on the DataFrame."""

    import sqlite3

    conn = sqlite3.connect(":memory:")

    try:
        # Make DataFrame available as a SQL table named 'data'
        df.to_sql("data", conn, index=False, if_exists="replace")

        # Execute the SQL query
        result = pd.read_sql_query(sql_query, conn)

        return result

    finally:
        conn.close()