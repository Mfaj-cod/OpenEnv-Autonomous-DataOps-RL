def get_task_data(task_id: str):

    # EASY
    if task_id == "missing_values_easy":
        return {
            "data": [
                {"name": "Alice", "age": 25, "salary": None, "department": "Engineering"},
                {"name": "Bob", "age": None, "salary": 50000, "department": "Sales"},
                {"name": "Charlie", "age": 30, "salary": None, "department": "Engineering"},
                {"name": "David", "age": None, "salary": 70000, "department": "HR"},
            ],
            "schema": {
                "name": "str",
                "age": "int",
                "salary": "int",
                "department": "str"
            },
            "visible_errors": [
                "Missing values present"
            ],
            "hidden_errors": []
        }

    # MEDIUM
    elif task_id == "schema_fix_medium":
        return {
            "data": [
                {"name": "Alice", "age": "25", "salary": "50000"},
                {"name": "Bob", "age": "thirty", "salary": "60000"},
                {"name": "Charlie", "age": "30", "salary": None},
                {"name": "David", "age": "28", "salary": "invalid"},
            ],
            "schema": {
                "name": "str",
                "age": "int",
                "salary": "int"
            },
            "visible_errors": [
                "Potential type issues detected"
            ],
            "hidden_errors": [
                "Type mismatch in value column",
                "Non-numeric values present",
                "Missing values present"
            ]
        }

    # HARD
    elif task_id == "pipeline_debug_hard":
        return {
            "data": [
                {"id": 1, "value": "100", "category": "A"},
                {"id": 1, "value": "100", "category": "A"},  # duplicate
                {"id": 2, "value": None, "category": "B"},   # missing
                {"id": 3, "value": "invalid", "category": ""},  # type + missing
                {"id": 4, "value": "250", "category": "C"},
            ],
            "schema": {
                "id": "int",
                "value": "int",
                "category": "str"
            },
            "visible_errors": [
                "Minor issues detected in dataset"
            ],
            "hidden_errors": [
                "Duplicate rows exist",
                "Missing values present",
                "Type mismatch in value column"
            ]
        }

    else:
        raise ValueError(f"Unknown task: {task_id}")


# TASK LIST
def list_tasks():
    return [
        {
            "id": "missing_values_easy",
            "difficulty": "easy",
            "description": "Handle missing values in structured dataset"
        },
        {
            "id": "schema_fix_medium",
            "difficulty": "medium",
            "description": "Fix type inconsistencies and invalid entries"
        },
        {
            "id": "pipeline_debug_hard",
            "difficulty": "hard",
            "description": "Debug full pipeline with duplicates, missing values, and type issues"
        }
    ]