from typing import Literal


# Defines the two agents that the Supervisor can route a question to
AgentType = Literal["sql", "python"]


def route_question(question: str) -> AgentType:
    """
    Identifies the type of user question and selects the appropriate agent.
    """
    question = question.lower().strip()

    # Keywords related to calculations and data analysis are handled by Python Agent
    python_keywords = [
        "calculate",
        "average",
        "mean",
        "median",
        "maximum",
        "minimum",
        "sum",
        "statistics",
        "analyze",
        "analysis",
    ]

    # Keywords related to database queries are handled by SQL Agent
    sql_keywords = [
        "database",
        "table",
        "sql",
        "record",
        "records",
        "employee",
        "employees",
    ]

    # Route calculation and analysis questions to the Python Agent
    if any(keyword in question for keyword in python_keywords):
        return "python"

    # Route database-related questions to the SQL Agent
    if any(keyword in question for keyword in sql_keywords):
        return "sql"

    # Use Python Agent as the default when the question is not clearly identified
    return "python"