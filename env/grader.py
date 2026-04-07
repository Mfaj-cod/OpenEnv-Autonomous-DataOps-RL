from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from env.tasks import DEFAULT_MAX_STEPS, get_task_data

EPS = 1e-6  # ensures strict (0, 1)

TASK_GRADER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "missing_values_easy": {
        "completeness": 0.35,
        "data_quality": 0.25,
        "pipeline_success": 0.15,
        "hidden_resolution": 0.10,
        "efficiency": 0.15,
    },
    "schema_fix_medium": {
        "schema_alignment": 0.30,
        "data_quality": 0.20,
        "pipeline_success": 0.20,
        "hidden_resolution": 0.15,
        "completeness": 0.10,
        "efficiency": 0.05,
    },
    "pipeline_debug_hard": {
        "data_quality": 0.25,
        "schema_alignment": 0.20,
        "uniqueness": 0.10,
        "pipeline_success": 0.20,
        "hidden_resolution": 0.15,
        "efficiency": 0.10,
    },
}


def _safe(score: float) -> float:
    if score is None or np.isnan(score) or np.isinf(score):
        return EPS
    return score


def _clamp(score: float) -> float:
    score = _safe(score)
    return float(max(EPS, min(score, 1.0 - EPS)))


def _soft_bool(x: bool) -> float:
    return 1.0 - EPS if x else EPS


def _compute_schema_alignment(
    df: pd.DataFrame,
    expected_schema: Optional[Dict[str, str]] = None,
) -> float:
    if df.empty:
        return EPS

    if not expected_schema:
        return 1.0 - EPS

    column_scores = []

    for column, expected_type in expected_schema.items():
        if column not in df.columns:
            column_scores.append(EPS)
            continue

        series = df[column]
        string_values = series.astype(str).str.strip()
        non_empty = series[~series.isnull() & (string_values != "")]

        if non_empty.empty:
            column_scores.append(1.0 - EPS)
            continue

        normalized_expected_type = expected_type.lower()
        if normalized_expected_type in {"int", "float", "number"}:
            numeric = pd.to_numeric(non_empty, errors="coerce")
            column_scores.append(float(numeric.notnull().mean()))
        else:
            column_scores.append(1.0 - EPS)

    if not column_scores:
        return 1.0 - EPS

    return _clamp(np.mean(column_scores))


def compute_quality_signals(
    data: Any,
    expected_schema: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    if not data:
        return {
            "data_quality": EPS,
            "completeness": EPS,
            "uniqueness": EPS,
            "schema_alignment": EPS,
            "type_consistency": EPS,
        }

    df = pd.DataFrame(data)
    if df.empty or df.size == 0:
        return {
            "data_quality": EPS,
            "completeness": EPS,
            "uniqueness": EPS,
            "schema_alignment": EPS,
            "type_consistency": EPS,
        }

    total_cells = max(int(df.size), 1)
    missing_mask = df.isnull() | (df.astype(str).apply(lambda col: col.str.strip()) == "")
    missing_ratio = float(missing_mask.sum().sum()) / total_cells
    completeness = _clamp(1.0 - missing_ratio)

    duplicate_ratio = float(df.duplicated().sum()) / max(1, len(df))
    uniqueness = _clamp(1.0 - duplicate_ratio)

    schema_alignment = _compute_schema_alignment(df, expected_schema)
    type_consistency = schema_alignment if expected_schema else uniqueness

    data_quality = _clamp(
        (0.45 * completeness) + (0.25 * uniqueness) + (0.30 * type_consistency)
    )

    return {
        "data_quality": data_quality,
        "completeness": completeness,
        "uniqueness": uniqueness,
        "schema_alignment": schema_alignment,
        "type_consistency": _clamp(type_consistency),
    }


def compute_data_quality(
    data: Any,
    expected_schema: Optional[Dict[str, str]] = None,
) -> float:
    return compute_quality_signals(data, expected_schema)["data_quality"]


def grade_report(env_state: Dict[str, Any], task_id: Optional[str] = None) -> Dict[str, Any]:
    resolved_task_id = task_id or env_state.get("task_id") or "missing_values_easy"

    task_schema: Dict[str, str] = {}
    task_hidden_count = 0
    try:
        task_config = get_task_data(resolved_task_id)
        task_schema = task_config.get("schema", {})
        task_hidden_count = len(task_config.get("hidden_errors", []))
    except ValueError:
        pass

    expected_schema = env_state.get("expected_schema") or task_schema
    signals = compute_quality_signals(env_state.get("data", []), expected_schema)

    total_hidden = int(env_state.get("total_hidden", task_hidden_count))
    hidden_resolved = int(env_state.get("hidden_resolved", 0))
    if total_hidden <= 0:
        hidden_resolution = 1.0 - EPS
    else:
        hidden_resolution = _clamp(hidden_resolved / float(total_hidden))

    max_steps = int(env_state.get("max_steps", DEFAULT_MAX_STEPS))
    steps = int(env_state.get("step_count", 0))
    efficiency = _clamp(1.0 - (steps / float(max(1, max_steps))))

    components = {
        "data_quality": signals["data_quality"],
        "completeness": signals["completeness"],
        "uniqueness": signals["uniqueness"],
        "schema_alignment": signals["schema_alignment"],
        "type_consistency": signals["type_consistency"],
        "pipeline_success": _soft_bool(env_state.get("pipeline_success", 0)),
        "hidden_resolution": hidden_resolution,
        "efficiency": efficiency,
    }

    weights = TASK_GRADER_WEIGHTS.get(
        resolved_task_id,
        TASK_GRADER_WEIGHTS["pipeline_debug_hard"],
    )

    score = 0.0
    for component, weight in weights.items():
        value = _safe(components.get(component, EPS))
        score += weight * value

    score = _clamp(_safe(score))

    return {
        "task_id": resolved_task_id,
        "score": score,
        "components": {name: _clamp(_safe(value)) for name, value in components.items()},
        "weights": weights,
    }


def grade(env_state: Dict[str, Any], task_id: Optional[str] = None) -> float:
    return grade_report(env_state, task_id=task_id)["score"]