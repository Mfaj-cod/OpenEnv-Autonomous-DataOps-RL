import argparse
import json
import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from openai import OpenAI

from env.baseline import deterministic_policy
from env.environment import DataOpsEnv
from env.grader import grade_report
from env.models import Action
from env.tasks import get_task_ids, get_task_metadata

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = OpenAI

HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
BENCHMARK = os.getenv("BENCHMARK", "dataops_rl_env")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))

SYSTEM_PROMPT = (
    "You are acting inside a DataOps cleaning environment. "
    "Choose exactly one valid next action as a JSON object with keys: "
    "action_type, column, tool_name, parameters. "
    "Valid action_type values are inspect_column, fill_missing, convert_type, "
    "remove_duplicates, run_pipeline, use_tool. "
    "For use_tool set tool_name to query_sql, view_logs, or profile_data. "
    "For inspect_column set column. "
    "Prefer actions that improve data quality and complete the pipeline."
)


def log_start(task: str, env_name: str, model_name: str) -> None:
    print(f"[START] task={task} env={env_name} model={model_name}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json_blob(raw_content: str) -> str:
    stripped = raw_content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return match.group(0)


def _action_from_model_output(raw_content: str) -> Action:
    payload = json.loads(_extract_json_blob(raw_content))
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object")

    payload.setdefault("parameters", {})
    return Action(**payload)


def _format_action(action: Action) -> str:
    if action.action_type == "use_tool":
        return f"use_tool({action.tool_name})"
    if action.action_type == "inspect_column":
        return f"inspect_column({action.column})"
    return action.action_type


def _build_user_prompt(task_id: str, observation, history: List[str]) -> str:
    recent_history = "\n".join(history[-5:]) if history else "None"
    return (
        f"Task: {task_id}\n"
        f"Visible errors: {observation.visible_errors}\n"
        f"Schema: {observation.data_schema}\n"
        f"Data quality score: {observation.data_quality_score:.4f}\n"
        f"Step count: {observation.step_count}\n"
        f"Recent history:\n{recent_history}\n"
        "Return one action JSON object only."
    )


def _choose_action(
    client: Optional["OpenAIClient"],
    task_id: str,
    observation,
    history: List[str],
) -> Tuple[Action, Optional[str]]:
    if client is None:
        return deterministic_policy(observation), None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(task_id, observation, history),
                },
            ],
        )
        raw_content = response.choices[0].message.content or "{}"
        return _action_from_model_output(raw_content), None
    except Exception:
        return deterministic_policy(observation), None


def _run_task(task_id: str, client: Optional["OpenAIClient"]) -> None:
    env = DataOpsEnv()
    task_meta = get_task_metadata(task_id)
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    final_score = 1e-6
    success = False

    log_start(task=task_id, env_name=BENCHMARK, model_name=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_id)

        for step in range(1, int(task_meta["max_steps"]) + 1):
            action, action_error = _choose_action(client, task_id, observation, history)
            action_repr = _format_action(action)

            try:
                observation, reward, done, _ = env.step(action)
                reward_value = float(reward.score)
                error_value = action_error
            except Exception as exc:
                reward_value = 0.0
                done = True
                error_value = str(exc)

            rewards.append(reward_value)
            steps_taken = step
            history.append(f"step={step} action={action_repr} reward={reward_value:.2f}")

            log_step(
                step=step,
                action=action_repr,
                reward=reward_value,
                done=done,
                error=error_value,
            )

            if done:
                break

        report = grade_report(env.state(), task_id=task_id)
        final_score = float(max(1e-6, min(float(report["score"]), 1.0 - 1e-6)))
        success = bool(final_score >= float(task_meta["success_threshold"]))
    finally:
        if hasattr(env, "close"):
            env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenEnv inference episodes.")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip model calls and use the deterministic local policy.",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Run only the specified task id. Repeat to run multiple tasks.",
    )
    args = parser.parse_args()

    client: Optional["OpenAIClient"] = None
    if not args.force_fallback and HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    task_ids = args.tasks or get_task_ids()
    for task_id in task_ids:
        _run_task(task_id, client)


if __name__ == "__main__":
    main()
