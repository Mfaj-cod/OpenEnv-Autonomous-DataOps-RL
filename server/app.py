import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from env.baseline import run_baseline
from env.environment import DataOpsEnv
from env.grader import grade_report
from env.models import (
    ACTION_TYPE_VALUES,
    CONDITIONAL_ACTION_REQUIREMENTS,
    TOOL_NAME_VALUES,
    Action,
)
from env.tasks import list_tasks

app = FastAPI(title="DataOps RL Environment API")
env = DataOpsEnv()


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_schema(model):
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


@app.get("/")
def root():
    return {
        "message": "DataOps RL Environment is running",
        "endpoints": ["/reset", "/step", "/state", "/grader", "/tasks", "/baseline", "/debug"],
    }


@app.post("/reset")
def reset(
    task_id: str = Query("missing_values_easy"),
    data_path: Optional[str] = Query(None),
):
    try:
        obs = env.reset(task_id=task_id, data_path=data_path)
        return {
            "status": "success",
            "task_id": task_id,
            "observation": _model_to_dict(obs),
            "done": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(action: Action):
    if env.state_data is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )

    try:
        obs, reward, done, info = env.step(action)
        return {
            "status": "success",
            "task_id": env.current_task_id,
            "observation": _model_to_dict(obs),
            "reward": _model_to_dict(reward),
            "done": done,
            "info": info,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
def state():
    return {
        "status": "success",
        "state": env.state(),
    }


@app.get("/tasks")
def tasks():
    action_schema = _model_schema(Action)
    return {
        "status": "success",
        "tasks": list_tasks(),
        "action_schema": action_schema,
        "required_action_fields": action_schema.get("required", []),
        "conditional_requirements": CONDITIONAL_ACTION_REQUIREMENTS,
        "action_types": ACTION_TYPE_VALUES,
        "tool_names": TOOL_NAME_VALUES,
    }


@app.get("/grader")
def grader():
    if env.state_data is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )

    try:
        report = grade_report(env.state(), task_id=env.current_task_id)
        return {
            "status": "success",
            "task_id": report["task_id"],
            "done": env.done,
            "score": round(float(report["score"]), 4),
            "components": {
                name: round(float(value), 4)
                for name, value in report["components"].items()
            },
            "weights": report["weights"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/baseline")
def baseline(force_fallback: bool = Query(False)):
    try:
        result = run_baseline(force_policy="fallback" if force_fallback else None)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/debug")
def debug():
    return {
        "status": "success",
        "internal_state": {
            "task_id": env.current_task_id,
            "step_count": env.step_count,
            "done": env.done,
            "hidden_errors": env.hidden_errors,
            "revealed_errors": env.revealed_errors,
            "state_data": env.state_data,
            "prev_grade_report": env.prev_grade_report,
        },
    }


def main():
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
