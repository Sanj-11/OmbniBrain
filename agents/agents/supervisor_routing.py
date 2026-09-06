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
    # Count matching keywords for each agent
    python_score = sum(keyword in question for keyword in python_keywords)
    sql_score = sum(keyword in question for keyword in sql_keywords)

    # Route to the agent with the higher matching score
    if python_score > sql_score:
        return "python"

    if sql_score > python_score:
        return "sql"

    # Use Python Agent when there is no clear winner
    return "python"
