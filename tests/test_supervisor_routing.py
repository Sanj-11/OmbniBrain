from agents.agents.supervisor_routing import route_question


def test_python_calculation_question():
    assert route_question("Calculate the average salary") == "python"


def test_sql_database_question():
    assert route_question("Show employee records from the table") == "sql"


def test_python_analysis_question():
    assert route_question("Analyze the sales data") == "python"


def test_sql_query_question():
    assert route_question("Query the employee database") == "sql"


def test_ambiguous_question_defaults_to_python():
    assert route_question("Tell me something") == "python"