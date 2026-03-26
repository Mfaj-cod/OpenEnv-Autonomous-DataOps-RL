from fastapi.testclient import TestClient

import server


def test_required_endpoints_and_shapes():
    client = TestClient(server.app)

    reset_response = client.post("/reset?task_id=missing_values_easy")
    assert reset_response.status_code == 200
    reset_body = reset_response.json()
    assert reset_body["status"] == "success"
    assert "observation" in reset_body

    step_response = client.post(
        "/step",
        json={"action_type": "use_tool", "tool_name": "profile_data"},
    )
    assert step_response.status_code == 200
    step_body = step_response.json()
    assert "reward" in step_body
    assert "done" in step_body

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json()["status"] == "success"

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    tasks_body = tasks_response.json()
    assert "tasks" in tasks_body
    assert "action_schema" in tasks_body
    assert "required_action_fields" in tasks_body
    assert "conditional_requirements" in tasks_body

    grader_response = client.get("/grader")
    assert grader_response.status_code == 200
    grader_body = grader_response.json()
    assert set(["score", "done", "task_id", "components"]).issubset(grader_body.keys())
    assert 0.0 <= grader_body["score"] <= 1.0

    baseline_response = client.get("/baseline?force_fallback=true")
    assert baseline_response.status_code == 200
    baseline_body = baseline_response.json()
    assert baseline_body["status"] == "success"
    assert "average_score" in baseline_body["result"]
