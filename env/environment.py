import copy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from env.grader import compute_data_quality, compute_quality_signals, grade_report
from env.models import Action, Observation, Reward
from env.tasks import DEFAULT_MAX_STEPS, get_task_data

ACTION_COST = {
    "inspect_column": 0.015,
    "fill_missing": 0.02,
    "convert_type": 0.035,
    "remove_duplicates": 0.02,
    "run_pipeline": 0.06,
    "use_tool": 0.015,
}

DYNAMIC_ERROR_PREFIXES = (
    "Missing values in columns:",
    "Numeric parsing issues in columns:",
    "Duplicate rows detected:",
)


class DataOpsEnv:
    def __init__(self):
        self.state_data = None
        self.hidden_errors: List[str] = []
        self.revealed_errors: List[str] = []
        self.step_count = 0
        self.done = False
        self.current_task_id = "missing_values_easy"
        self.max_steps = DEFAULT_MAX_STEPS
        self.success_threshold = 0.95
        self.prev_grade_report: Dict[str, object] = {
            "score": 0.0,
            "components": {
                "data_quality": 0.0,
                "hidden_resolution": 0.0,
            },
        }

    def state(self) -> Dict:
        if self.state_data is None:
            return {
                "task_id": None,
                "data": [],
                "schema": {},
                "expected_schema": {},
                "pipeline_success": 0,
                "schema_valid": 0,
                "hidden_resolved": 0,
                "total_hidden": 0,
                "max_steps": DEFAULT_MAX_STEPS,
                "action_history": [],
                "step_count": 0,
                "done": False,
            }

        return {
            **self.state_data,
            "step_count": self.step_count,
            "done": self.done,
            "hidden_remaining": len(self.hidden_errors),
        }

    def reset(self, task_id: str = "missing_values_easy", data_path: str = None) -> Observation:
        task = get_task_data(task_id)
        self.current_task_id = task_id
        self.max_steps = int(task.get("max_steps", DEFAULT_MAX_STEPS))
        self.success_threshold = float(task.get("success_threshold", 0.95))

        if data_path:
            df = pd.read_csv(data_path)
            df = self._sanitize_df(df)
            data = df.to_dict(orient="records")
            schema = {col: str(df[col].dtype) for col in df.columns}
            expected_schema = copy.deepcopy(task["schema"])
        else:
            data = copy.deepcopy(task["data"])
            schema = copy.deepcopy(task["schema"])
            expected_schema = copy.deepcopy(task["schema"])

        self.hidden_errors = list(task["hidden_errors"])
        self.revealed_errors = []
        self.step_count = 0
        self.done = False

        self.state_data = {
            "task_id": task_id,
            "data": self._sanitize_data(data),
            "schema": schema,
            "expected_schema": expected_schema,
            "pipeline_success": 0,
            "schema_valid": 0,
            "hidden_resolved": 0,
            "total_hidden": len(task["hidden_errors"]),
            "max_steps": self.max_steps,
            "action_history": [],
            "info_gain": 0,
        }

        df = self._to_df()
        hints = [f"Hint: {message}" for message in task.get("visible_errors", [])]
        self.revealed_errors = self._merge_unique(hints + self._generate_visible_errors(df))

        self.prev_grade_report = grade_report(self.state(), task_id=self.current_task_id)
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.state_data is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        if self.done:
            reward = Reward(score=0.0, components={"episode_finished": 1.0})
            return self._get_observation(), reward, True, {
                "message": "Episode already finished. Call reset() for a new episode."
            }

        self.state_data["info_gain"] = 0
        self.step_count += 1

        action_signature = self._action_signature(action)
        repeat_penalty = self._repeat_penalty(action_signature)

        self._apply_action(action)
        self.state_data["action_history"].append(action_signature)

        current_report = grade_report(self.state(), task_id=self.current_task_id)
        previous_components = self.prev_grade_report.get("components", {})
        current_components = current_report.get("components", {})

        score_delta = float(current_report["score"]) - float(self.prev_grade_report["score"])
        quality_delta = float(current_components.get("data_quality", 0.0)) - float(
            previous_components.get("data_quality", 0.0)
        )
        hidden_delta = float(current_components.get("hidden_resolution", 0.0)) - float(
            previous_components.get("hidden_resolution", 0.0)
        )

        pipeline_bonus = 0.0
        if action.action_type == "run_pipeline":
            pipeline_bonus = 0.08 if self.state_data.get("pipeline_success", 0) else -0.03

        tool_bonus = 0.0
        if action.action_type == "use_tool":
            tool_bonus = 0.02 if self.state_data.get("info_gain", 0) else -0.01

        action_cost = ACTION_COST.get(action.action_type, 0.02)
        no_op_penalty = -0.01 if abs(score_delta) < 1e-9 and abs(quality_delta) < 1e-9 else 0.0

        reward_value = (
            score_delta
            + (0.40 * quality_delta)
            + (0.20 * hidden_delta)
            + pipeline_bonus
            + tool_bonus
            + repeat_penalty
            + no_op_penalty
            - action_cost
        )
        reward_value = float(np.clip(reward_value, -1.0, 1.0))

        self.prev_grade_report = current_report

        if (
            float(current_report["score"]) >= self.success_threshold
            and not self.hidden_errors
            and bool(self.state_data.get("pipeline_success", 0))
        ):
            self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        reward = Reward(
            score=reward_value,
            components={
                "score_delta": float(score_delta),
                "quality_delta": float(quality_delta),
                "hidden_progress": float(hidden_delta),
                "pipeline_bonus": float(pipeline_bonus),
                "tool_bonus": float(tool_bonus),
                "repeat_penalty": float(repeat_penalty),
                "no_op_penalty": float(no_op_penalty),
                "action_cost": float(action_cost),
            },
        )

        info = {
            "task_id": self.current_task_id,
            "hidden_remaining": len(self.hidden_errors),
            "grader": current_report,
        }

        return self._get_observation(), reward, self.done, info

    def _get_observation(self) -> Observation:
        return Observation(
            dataset_preview=self._sanitize_data(self.state_data["data"][:3]),
            data_schema=self.state_data["schema"],
            visible_errors=self.revealed_errors,
            available_tools=["query_sql", "view_logs", "profile_data"],
            data_quality_score=float(self._safe_quality()),
            step_count=self.step_count,
        )

    def _to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.state_data["data"])

    def _update_state(self, df: pd.DataFrame):
        df = self._sanitize_df(df)
        self.state_data["data"] = df.to_dict(orient="records")
        self.state_data["schema"] = {col: str(df[col].dtype) for col in df.columns}
        self.state_data["schema_valid"] = int(self._schema_alignment_score() >= 0.95)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace(["", " ", "NULL", "null"], None)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        return df

    def _sanitize_data(self, data):
        clean = []
        for row in data:
            clean_row = {}
            for key, value in row.items():
                clean_row[key] = self._safe_number(value)
            clean.append(clean_row)
        return clean

    def _safe_number(self, value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return None
        return value

    def _safe_quality(self) -> float:
        try:
            return compute_data_quality(
                self.state_data["data"],
                self.state_data.get("expected_schema", {}),
            )
        except Exception:
            return 0.0

    def _schema_alignment_score(self) -> float:
        signals = compute_quality_signals(
            self.state_data["data"],
            self.state_data.get("expected_schema", {}),
        )
        return float(signals["schema_alignment"])

    def _generate_visible_errors(self, df: pd.DataFrame) -> List[str]:
        errors: List[str] = []

        null_columns: List[str] = []
        for column in df.columns:
            column_values = df[column]
            missing_mask = column_values.isnull() | (column_values.astype(str).str.strip() == "")
            if missing_mask.any():
                null_columns.append(column)
        if null_columns:
            errors.append(f"Missing values in columns: {null_columns}")

        type_issues: List[str] = []
        expected_schema = self.state_data.get("expected_schema", {})
        for column, expected_type in expected_schema.items():
            if column not in df.columns:
                continue
            if expected_type.lower() not in {"int", "float", "number"}:
                continue

            series = df[column]
            non_empty = series[~series.isnull() & (series.astype(str).str.strip() != "")]
            if non_empty.empty:
                continue

            numeric = pd.to_numeric(non_empty, errors="coerce")
            if float(numeric.notnull().mean()) < 1.0:
                type_issues.append(column)
        if type_issues:
            errors.append(f"Numeric parsing issues in columns: {type_issues}")

        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            errors.append(f"Duplicate rows detected: {duplicate_count}")

        return errors

    def _merge_unique(self, values: List[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                merged.append(value)
        return merged

    def _apply_action(self, action: Action):
        if action.action_type == "fill_missing":
            self._handle_missing()
        elif action.action_type == "remove_duplicates":
            self._remove_duplicates()
        elif action.action_type == "convert_type":
            self._handle_type_conversion()
        elif action.action_type == "run_pipeline":
            self._run_pipeline()
        elif action.action_type == "use_tool":
            self._use_tool(action.tool_name)
        elif action.action_type == "inspect_column":
            self._inspect_column(action.column)

        df = self._to_df()
        persistent_messages = [
            item
            for item in self.revealed_errors
            if not item.startswith(DYNAMIC_ERROR_PREFIXES)
        ]
        self.revealed_errors = self._merge_unique(
            persistent_messages + self._generate_visible_errors(df)
        )

    def _handle_missing(self):
        df = self._to_df()
        expected_schema = self.state_data.get("expected_schema", {})

        for column in df.columns:
            expected_type = expected_schema.get(column, "").lower()
            if expected_type in {"int", "float", "number"}:
                numeric = pd.to_numeric(df[column], errors="coerce")
                fill_value = numeric.median()
                if pd.isna(fill_value):
                    fill_value = 0
                df[column] = numeric.fillna(fill_value)
            else:
                if not df[column].mode().empty:
                    df[column] = df[column].fillna(df[column].mode().iloc[0])
                else:
                    df[column] = df[column].fillna("missing")

        self._update_state(df)
        self._resolve_hidden("Missing values present")

    def _handle_type_conversion(self):
        df = self._to_df()
        expected_schema = self.state_data.get("expected_schema", {})

        for column in df.columns:
            expected_type = expected_schema.get(column, "").lower()
            if expected_type in {"int", "float", "number"}:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        self._update_state(df)

        if self._schema_alignment_score() >= 0.90:
            self._resolve_hidden("Type mismatch in value column")
            self._resolve_hidden("Non-numeric values present")

    def _remove_duplicates(self):
        df = self._to_df().drop_duplicates()
        self._update_state(df)
        self._resolve_hidden("Duplicate rows exist")

    def _run_pipeline(self):
        df = self._to_df()
        has_missing = bool((df.isnull() | (df.astype(str).apply(lambda col: col.str.strip() == ""))).sum().sum())
        has_duplicates = bool(df.duplicated().sum())
        schema_alignment = self._schema_alignment_score()

        if not has_missing and not has_duplicates and schema_alignment >= 0.95:
            self._resolve_all_hidden()
            self.state_data["pipeline_success"] = 1
            self.state_data["schema_valid"] = 1
        else:
            self.state_data["pipeline_success"] = 0

    def _use_tool(self, tool_name: str):
        before = len(self.revealed_errors)

        if tool_name == "view_logs":
            self.revealed_errors = self._merge_unique(self.revealed_errors + self.hidden_errors)
        elif tool_name == "profile_data":
            df = self._to_df()
            profile = {
                "nulls": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated().sum()),
                "columns": list(df.columns),
            }
            self.revealed_errors.append(f"Profile: {profile}")
            self.revealed_errors = self._merge_unique(self.revealed_errors)
        elif tool_name == "query_sql":
            df = self._to_df()
            duplicate_columns = [column for column in df.columns if df[column].duplicated().any()]
            self.revealed_errors.append(f"Duplicate columns: {duplicate_columns}")
            self.revealed_errors = self._merge_unique(self.revealed_errors)

        after = len(self.revealed_errors)
        if after > before:
            self.state_data["info_gain"] = 1

    def _inspect_column(self, column_name: str):
        df = self._to_df()
        if not column_name or column_name not in df.columns:
            self.revealed_errors.append("Inspect failed: unknown column")
            self.revealed_errors = self._merge_unique(self.revealed_errors)
            return

        series = df[column_name]
        summary = {
            "column": column_name,
            "nulls": int(series.isnull().sum()),
            "unique": int(series.nunique(dropna=True)),
            "sample": [self._safe_number(value) for value in series.head(3).tolist()],
        }
        self.revealed_errors.append(f"Inspect: {summary}")
        self.revealed_errors = self._merge_unique(self.revealed_errors)
        self.state_data["info_gain"] = 1

    def _resolve_hidden(self, issue: str):
        if issue in self.hidden_errors:
            self.hidden_errors.remove(issue)
            self.state_data["hidden_resolved"] += 1

    def _resolve_all_hidden(self):
        self.state_data["hidden_resolved"] += len(self.hidden_errors)
        self.hidden_errors = []

    def _action_signature(self, action: Action) -> str:
        return f"{action.action_type}|{action.column or ''}|{action.tool_name or ''}"

    def _repeat_penalty(self, action_signature: str) -> float:
        history = self.state_data.get("action_history", [])
        if len(history) >= 2 and history[-1] == action_signature and history[-2] == action_signature:
            return -0.03
        if len(history) >= 1 and history[-1] == action_signature:
            return -0.01
        return 0.0
