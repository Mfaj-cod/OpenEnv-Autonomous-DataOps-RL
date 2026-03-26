import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from env.environment import DataOpsEnv
from env.grader import grade_report
from env.models import Action
from env.tasks import list_tasks

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when dependency is missing at runtime
    OpenAI = None

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = Any

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def deterministic_policy(obs) -> Action:
    normalized_errors = [error.lower() for error in obs.visible_errors]
    error_text = " ".join(normalized_errors)

    if obs.step_count == 0:
        return Action(action_type="use_tool", tool_name="profile_data")

    if any(error.startswith("numeric parsing issues in columns:") for error in normalized_errors):
        return Action(action_type="convert_type")

    if any(error.startswith("missing values in columns:") for error in normalized_errors):
        return Action(action_type="fill_missing")

    if any(error.startswith("duplicate rows detected:") for error in normalized_errors):
        return Action(action_type="remove_duplicates")
    if "duplicate columns:" in error_text:
        return Action(action_type="remove_duplicates")

    if obs.data_quality_score < 0.92 and obs.step_count < 5:
        return Action(action_type="use_tool", tool_name="view_logs")

    return Action(action_type="run_pipeline")


def _resolve_api_key() -> Tuple[Optional[str], Optional[str]]:
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return groq_key, "GROQ_API_KEY"

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return openai_key, "OPENAI_API_KEY"

    return None, None


def _extract_json_blob(raw_content: str) -> str:
    stripped = raw_content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("No JSON object found in model response")


def _action_from_model_output(raw_content: str) -> Action:
    payload = json.loads(_extract_json_blob(raw_content))
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object")

    payload.setdefault("parameters", {})
    return Action(**payload)


def _llm_policy(
    obs,
    task_id: str,
    client: OpenAIClient,
    model: str,
) -> Action:
    system_prompt = (
        "You are an action planner for a DataOps RL environment. "
        "Return only one JSON object with keys: action_type, column, tool_name, parameters. "
        "Valid action_type values: inspect_column, fill_missing, convert_type, remove_duplicates, run_pipeline, use_tool. "
        "For use_tool set tool_name to one of query_sql, view_logs, profile_data. "
        "For inspect_column set column. "
        "If no fix is needed, choose run_pipeline."
    )

    user_prompt = (
        f"Task: {task_id}\n"
        f"Step: {obs.step_count}\n"
        f"Visible errors: {obs.visible_errors}\n"
        f"Schema: {obs.data_schema}\n"
        f"Data quality score: {obs.data_quality_score}\n"
        "Respond with JSON only."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    content = response.choices[0].message.content or "{}"
    return _action_from_model_output(content)


def run_baseline(force_policy: Optional[str] = None) -> Dict[str, Any]:
    api_key, key_source = _resolve_api_key()
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    use_llm = (
        force_policy != "fallback"
        and OpenAI is not None
        and api_key is not None
    )

    client = None
    if use_llm:
        client = OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
        )

    policy_mode = "llm" if use_llm else "fallback"
    tasks = list_tasks()

    total_score = 0.0
    total_steps = 0
    llm_fallback_steps = 0
    task_results: Dict[str, Dict[str, Any]] = {}

    for task in tasks:
        env = DataOpsEnv()
        task_id = task["id"]
        max_steps = int(task.get("max_steps", 15))
        obs = env.reset(task_id=task_id)

        for _ in range(max_steps):
            if client is None:
                action = deterministic_policy(obs)
            else:
                try:
                    action = _llm_policy(obs, task_id, client, model)
                except Exception:
                    action = deterministic_policy(obs)
                    llm_fallback_steps += 1

            obs, _, done, _ = env.step(action)
            if done:
                break

        final_report = grade_report(env.state(), task_id=task_id)
        final_score = float(final_report["score"])
        total_score += final_score
        total_steps += obs.step_count

        task_results[task_id] = {
            "final_score": round(final_score, 4),
            "steps_used": int(obs.step_count),
            "done": bool(env.done),
            "components": {
                name: round(value, 4)
                for name, value in final_report["components"].items()
            },
        }

    average_score = total_score / max(len(tasks), 1)

    return {
        "average_score": round(float(average_score), 4),
        "policy_mode": policy_mode,
        "provider": "groq-openai-compatible" if use_llm else "deterministic-fallback",
        "model": model if use_llm else "deterministic_policy",
        "key_source": key_source if use_llm else "none",
        "fallback_steps": int(llm_fallback_steps),
        "total_steps": int(total_steps),
        "task_results": task_results,
    }


if __name__ == "__main__":
    print(json.dumps(run_baseline(), indent=2))
