import copy
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

from env.models import Observation, Action, Reward
from env.tasks import get_task_data
from env.grader import compute_data_quality, grade


ACTION_COST = {
    "inspect_column": 0.02,
    "fill_missing": 0.02,
    "convert_type": 0.05,
    "remove_duplicates": 0.02,
    "run_pipeline": 0.1,
    "use_tool": 0.02,
}


class DataOpsEnv:
    def __init__(self):
        self.state_data = None
        self.hidden_errors = []
        self.revealed_errors = []
        self.step_count = 0
        self.done = False
        self.prev_score = 0.0

    # STATE
    def state(self):
        if self.state_data is None:
            return {
                "data": [],
                "schema": {},
                "pipeline_success": 0,
                "schema_valid": 0,
                "hidden_resolved": 0,
                "step_count": 0
            }

        return {
            **self.state_data,
            "step_count": self.step_count
        }

    # RESET
    def reset(self, task_id: str = "missing_values_easy", data_path: str = None) -> Observation:
        task = get_task_data(task_id)

        if data_path:
            df = pd.read_csv(data_path)
            df = self._sanitize_df(df)
            data = df.to_dict(orient="records")
            schema = {col: str(df[col].dtype) for col in df.columns}
        else:
            data = copy.deepcopy(task["data"])
            schema = copy.deepcopy(task["schema"])

        self.state_data = {
            "data": self._sanitize_data(data),
            "schema": schema,
            "pipeline_success": 0,
            "schema_valid": 0,
            "hidden_resolved": 0,
        }

        self.hidden_errors = list(task["hidden_errors"])

        df = pd.DataFrame(self.state_data["data"])
        self.revealed_errors = self._generate_visible_errors(df)

        self.step_count = 0
        self.done = False
        self.prev_score = grade(self.state())

        return self._get_observation()

    # STEP
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        self.state_data["pipeline_success"] = 0
        self.state_data["info_gain"] = 0
        self.step_count += 1

        self._apply_action(action)

        current_score = grade(self.state())
        reward_value = current_score - self.prev_score

        if abs(reward_value) < 1e-6:
            reward_value -= 0.005 # small penalty for no improvement

        # reward shaping
        if action.action_type == "use_tool":
            if self.state_data.get("info_gain", 0):
                reward_value += 0.03
            else:
                reward_value -= 0.01

        elif action.action_type == "run_pipeline":
            if self.state_data.get("pipeline_success", 0):
                reward_value += 0.1   # strong reward
            else:
                reward_value -= 0.02  # penalize premature execution

        reward_value -= ACTION_COST.get(action.action_type, 0)

        if action.action_type == "fill_missing" and reward_value > 0:
            reward_value += 0.03

        self.prev_score = current_score

        # termination
        if current_score >= 0.95 and not self.hidden_errors:
            self.done = True

        if self.step_count >= 15:
            self.done = True

        reward = Reward(
            score=float(self._safe_number(reward_value)),
            components={
                "score_delta": float(self._safe_number(reward_value))
            }
        )

        return self._get_observation(), reward, self.done, {}

    # OBS
    def _get_observation(self) -> Observation:
        return Observation(
            dataset_preview=self._sanitize_data(self.state_data["data"][:3]),
            data_schema=self.state_data["schema"],
            visible_errors=self.revealed_errors,
            available_tools=["query_sql", "view_logs", "profile_data"],
            data_quality_score=float(self._safe_quality()),
            step_count=self.step_count,
        )

    # HELPERS

    def _to_df(self):
        return pd.DataFrame(self.state_data["data"])

    def _update_state(self, df: pd.DataFrame):
        df = self._sanitize_df(df)
        self.state_data["data"] = df.to_dict(orient="records")
        self.state_data["schema"] = {col: str(df[col].dtype) for col in df.columns}

    def _sanitize_df(self, df: pd.DataFrame):
        df = df.replace(["", " ", "NULL", "null"], None)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        return df

    def _sanitize_data(self, data):
        clean = []
        for row in data:
            clean_row = {}
            for k, v in row.items():
                clean_row[k] = self._safe_number(v)
            clean.append(clean_row)
        return clean

    def _safe_number(self, x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x) or np.isinf(x):
                return None
        return x

    def _safe_quality(self):
        try:
            q = compute_data_quality(self.state_data["data"])
            return 0.0 if np.isnan(q) else float(q)
        except:
            return 0.0

    # ERROR GENERATION
    def _generate_visible_errors(self, df):
        errors = []

        # missing detection
        null_cols = []
        for col in df.columns:
            col_data = df[col]
            missing_mask = col_data.isnull() | (col_data.astype(str).str.strip() == "")
            if missing_mask.any():
                null_cols.append(col)

        if null_cols:
            errors.append(f"Missing values in columns: {null_cols}")

        # type detection
        type_issues = []
        for col in df.columns:
            if df[col].dtype == "object":
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    numeric_like = pd.to_numeric(non_null, errors="coerce")
                    if numeric_like.notnull().mean() > 0.7:
                        type_issues.append(col)

        if type_issues:
            errors.append(f"Numeric values stored as strings: {type_issues}")

        return errors

    # ACTIONS

    def _apply_action(self, action: Action):

        if action.action_type == "fill_missing":
            self._handle_missing()

        elif action.action_type == "remove_duplicates":
            df = self._to_df().drop_duplicates()
            self._update_state(df)
            self._resolve_hidden("Duplicate rows exist")

        elif action.action_type == "convert_type":
            self._handle_type_conversion()

        elif action.action_type == "run_pipeline":
            self._run_pipeline()

        elif action.action_type == "use_tool":
            self._use_tool(action.tool_name)

        # refresh errors
        df = self._to_df()
        self.revealed_errors = self._generate_visible_errors(df)

    # LOGIC

    def _handle_missing(self):
        df = self._to_df()

        for col in df.columns:
            if df[col].dtype == "object":
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                else:
                    df[col] = df[col].fillna("missing")
            else:
                median = df[col].median()
                if pd.isna(median):
                    median = 0
                df[col] = df[col].fillna(median)

        self._update_state(df)
        self._resolve_hidden("Missing values present")

    def _handle_type_conversion(self):
        df = self._to_df()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._update_state(df)
        self.state_data["schema_valid"] = 1

        self._resolve_hidden("Type mismatch in value column")

    def _run_pipeline(self):
        df = self._to_df()

        if df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0:
            self._resolve_all_hidden()
            self.state_data["pipeline_success"] = 1   # ✅ CRITICAL

    # TOOLS

    def _use_tool(self, tool_name):
        before = len(self.revealed_errors)

        if tool_name == "view_logs":
            self.revealed_errors.extend(self.hidden_errors)

        elif tool_name == "profile_data":
            df = self._to_df()
            profile = {
                "nulls": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated().sum()),
                "columns": list(df.columns)
            }
            self.revealed_errors.append(f"Profile: {profile}")

        elif tool_name == "query_sql":
            df = self._to_df()
            dup_cols = [col for col in df.columns if df[col].duplicated().any()]
            self.revealed_errors.append(f"Duplicate columns: {dup_cols}")

        after = len(self.revealed_errors)
        if after > before:
            self.state_data["info_gain"] = 1

    # HIDDEN
    def _resolve_hidden(self, issue):
        if issue in self.hidden_errors:
            self.hidden_errors.remove(issue)
            self.state_data["hidden_resolved"] += 1

    def _resolve_all_hidden(self):
        self.state_data["hidden_resolved"] += len(self.hidden_errors)
        self.hidden_errors = []