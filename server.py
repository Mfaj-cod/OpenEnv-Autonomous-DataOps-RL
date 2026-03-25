from fastapi import FastAPI, Query, HTTPException
from typing import Optional

from env.environment import DataOpsEnv
from env.models import Action
from env.tasks import list_tasks
from env.grader import grade
from env.baseline import run_baseline


app = FastAPI(title="DataOps RL Environment API")

env = DataOpsEnv()


# ROOT
@app.get("/")
def root():
    return {
        "message": "DataOps RL Environment is running",
        "endpoints": [
            "/reset",
            "/step",
            "/state",
            "/grader",
            "/tasks",
            "/baseline",
            "/debug"
        ]
    }


# RESET
@app.post("/reset")
def reset(
    task_id: str = Query("missing_values_easy"),
    data_path: Optional[str] = Query(None)
):
    try:
        obs = env.reset(task_id=task_id, data_path=data_path)
        return {
            "status": "success",
            "observation": obs.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# STEP
@app.post("/step")
def step(action: Action):
    if env.state_data is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    try:
        obs, reward, done, info = env.step(action)

        return {
            "status": "success",
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# STATE
@app.get("/state")
def state():
    return {
        "status": "success",
        "state": env.state()
    }


# TASKS
@app.get("/tasks")
def tasks():
    return {
        "status": "success",
        "tasks": list_tasks()
    }


# GRADER
@app.get("/grader")
def grader():
    if env.state_data is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    try:
        score = grade(env.state())
        return {
            "status": "success",
            "score": round(score, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# BASELINE
@app.get("/baseline")
def baseline():
    try:
        result = run_baseline()
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# DEBUG (VERY USEFUL)
@app.get("/debug")
def debug():
    return {
        "status": "success",
        "internal_state": {
            "step_count": env.step_count,
            "done": env.done,
            "hidden_errors": env.hidden_errors,
            "revealed_errors": env.revealed_errors,
            "state_data": env.state_data
        }
    }