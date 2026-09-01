from typing import Literal

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


class AgentState(TypedDict):
    question: str
    next_agent: str
    answer: str


def supervisor(state: AgentState):
    """Decide which agent should handle the user's question."""

    question = state["question"].lower()

    # Questions that require database/SQL operations
    sql_keywords = [
        "database",
        "sql",
        "table",
        "rows",
        "records",
        "sales",
        "revenue",
        "salary",
        "count",
        "total",
        "average",
        "maximum",
        "minimum",
    ]

    # Questions that require Python/data analysis
    python_keywords = [
        "calculate",
        "calculate",
        "mean",
        "median",
        "standard deviation",
        "plot",
        "graph",
        "chart",
        "predict",
        "prediction",
        "analyze",
        "analysis",
    ]

    if any(keyword in question for keyword in sql_keywords):
        return {"next_agent": "sql"}

    if any(keyword in question for keyword in python_keywords):
        return {"next_agent": "python"}

    # Default route
    return {"next_agent": "python"}


def route_question(state: AgentState) -> Literal["sql", "python"]:
    """Route the question based on the supervisor's decision."""

    return state["next_agent"]


# Create LangGraph workflow
workflow = StateGraph(AgentState)

# Add supervisor node
workflow.add_node("supervisor", supervisor)

# Start with supervisor
workflow.set_entry_point("supervisor")

# Route to the selected agent
workflow.add_conditional_edges(
    "supervisor",
    route_question,
    {
        "sql": END,
        "python": END,
    },
)

# Compile graph
supervisor_graph = workflow.compile()

if __name__ == "__main__":
    result1 = supervisor_graph.invoke({
        "question": "What is the total sales?",
        "next_agent": "",
        "answer": ""
    })

    print("Question: What is the total sales?")
    print("Selected Agent:", result1["next_agent"])

    result2 = supervisor_graph.invoke({
        "question": "Calculate the average salary",
        "next_agent": "",
        "answer": ""
    })

    print("Question: Calculate the average salary")
    print("Selected Agent:", result2["next_agent"])