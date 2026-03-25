from env.environment import DataOpsEnv
from env.models import Action
from env.grader import grade


# SMART POLICY
def smarter_policy(obs):
    errors = " ".join(obs.visible_errors).lower()

    # STEP 0: always inspect first
    if obs.step_count == 0:
        return Action(action_type="use_tool", tool_name="profile_data")

    # PRIORITY 1: type issues
    if "numeric values stored as strings" in errors or "type" in errors:
        return Action(action_type="convert_type")

    # PRIORITY 2: missing values
    if "missing values" in errors:
        return Action(action_type="fill_missing")

    # PRIORITY 3: duplicates
    if "duplicate" in errors:
        return Action(action_type="remove_duplicates")

    # FALLBACK: explore if unsure
    if obs.data_quality_score < 0.9 and obs.step_count < 5:
        return Action(action_type="use_tool", tool_name="view_logs")

    # FINAL: run pipeline
    return Action(action_type="run_pipeline")


# BASELINE RUNNER
def run_baseline():
    env = DataOpsEnv()
    tasks = [
        "missing_values_easy",
        "schema_fix_medium",
        "pipeline_debug_hard"
    ]

    total_score = 0
    results = {}

    for task in tasks:
        obs = env.reset(task)

        for _ in range(15):
            action = smarter_policy(obs)
            obs, reward, done, _ = env.step(action)

            if done:
                break

        final_score = grade(env.state())
        total_score += final_score

        results[task] = {
            "final_score": round(final_score, 4),
            "steps_used": obs.step_count
        }

    avg_score = total_score / len(tasks)

    return {
        "average_score": round(avg_score, 4),
        "task_results": results
    }