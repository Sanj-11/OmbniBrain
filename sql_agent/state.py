from typing import Any, Optional
from typing_extensions import TypedDict


class SQLAgentState(TypedDict, total=False):
    question: str
    dataset_id: int

    file_path: str
    original_filename: str

    schema: str
    sql_query: str
    query_result: Any

    answer: str
    error: Optional[str]