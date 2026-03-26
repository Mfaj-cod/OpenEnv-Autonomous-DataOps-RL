from copy import deepcopy
from typing import Any, Dict, List

DEFAULT_MAX_STEPS = 15

TASK_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "missing_values_easy": {
        "id": "missing_values_easy",
        "difficulty": "easy",
        "description": "Resolve missing values and run a clean pipeline pass.",
        "objective": "Impute missing values and reach high completeness quickly.",
        "max_steps": 12,
        "success_threshold": 0.95,
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
            "department": "str",
        },
        "visible_errors": ["Missing values present"],
        "hidden_errors": [],
        "grader_focus": [
            "completeness",
            "pipeline_success",
            "efficiency",
        ],
    },
    "schema_fix_medium": {
        "id": "schema_fix_medium",
        "difficulty": "medium",
        "description": "Repair invalid numeric fields and restore schema compliance.",
        "objective": "Convert malformed numeric columns and clear missing values.",
        "max_steps": DEFAULT_MAX_STEPS,
        "success_threshold": 0.93,
        "data": [
            {"name": "Alice", "age": "25", "salary": "50000"},
            {"name": "Bob", "age": "thirty", "salary": "60000"},
            {"name": "Charlie", "age": "30", "salary": None},
            {"name": "David", "age": "28", "salary": "invalid"},
        ],
        "schema": {
            "name": "str",
            "age": "int",
            "salary": "int",
        },
        "visible_errors": ["Potential type issues detected"],
        "hidden_errors": [
            "Type mismatch in value column",
            "Non-numeric values present",
            "Missing values present",
        ],
        "grader_focus": [
            "schema_alignment",
            "data_quality",
            "pipeline_success",
        ],
    },
    "pipeline_debug_hard": {
        "id": "pipeline_debug_hard",
        "difficulty": "hard",
        "description": "Debug a realistic broken pipeline with duplicates, nulls, and type failures.",
        "objective": "Apply multi-step cleanup and finish with a successful pipeline run.",
        "max_steps": DEFAULT_MAX_STEPS,
        "success_threshold": 0.90,
        "data": [
            {"id": 1, "value": "100", "category": "A"},
            {"id": 1, "value": "100", "category": "A"},
            {"id": 2, "value": None, "category": "B"},
            {"id": 3, "value": "invalid", "category": ""},
            {"id": 4, "value": "250", "category": "C"},
        ],
        "schema": {
            "id": "int",
            "value": "int",
            "category": "str",
        },
        "visible_errors": ["Minor issues detected in dataset"],
        "hidden_errors": [
            "Duplicate rows exist",
            "Missing values present",
            "Type mismatch in value column",
            "Non-numeric values present",
        ],
        "grader_focus": [
            "data_quality",
            "schema_alignment",
            "hidden_resolution",
            "pipeline_success",
        ],
    },
}


def get_task_data(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task: {task_id}")
    return deepcopy(TASK_DEFINITIONS[task_id])


def get_task_ids() -> List[str]:
    return list(TASK_DEFINITIONS.keys())


def get_task_metadata(task_id: str) -> Dict[str, Any]:
    task = get_task_data(task_id)
    return {
        "id": task["id"],
        "difficulty": task["difficulty"],
        "description": task["description"],
        "objective": task["objective"],
        "max_steps": task.get("max_steps", DEFAULT_MAX_STEPS),
        "success_threshold": task.get("success_threshold", 0.95),
        "grader_focus": task.get("grader_focus", []),
    }


def list_tasks() -> List[Dict[str, Any]]:
    task_list: List[Dict[str, Any]] = []
    for task_id in get_task_ids():
        task_list.append(get_task_metadata(task_id))
    return task_list
