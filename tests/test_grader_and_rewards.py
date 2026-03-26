from env.baseline import run_baseline
from env.environment import DataOpsEnv
from env.grader import grade_report
from env.models import Action
from env.tasks import get_task_ids


def test_grader_scores_are_bounded_for_all_tasks():
    env = DataOpsEnv()

    for task_id in get_task_ids():
        env.reset(task_id=task_id)
        report = grade_report(env.state(), task_id=task_id)
        assert 0.0 <= report["score"] <= 1.0


def test_reward_has_partial_progress_and_repeat_penalty():
    env = DataOpsEnv()
    env.reset(task_id="missing_values_easy")

    _, first_reward, _, _ = env.step(Action(action_type="fill_missing"))
    _, second_reward, _, _ = env.step(Action(action_type="fill_missing"))

    assert first_reward.components["quality_delta"] >= 0.0
    assert second_reward.components["repeat_penalty"] <= 0.0


def test_baseline_fallback_is_reproducible(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    first = run_baseline(force_policy="fallback")
    second = run_baseline(force_policy="fallback")

    assert first == second
