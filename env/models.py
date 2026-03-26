from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from pydantic import model_validator
    HAS_MODEL_VALIDATOR = True
except ImportError:  # pragma: no cover - only used with Pydantic v1
    from pydantic import root_validator
    HAS_MODEL_VALIDATOR = False

ACTION_TYPE_VALUES = [
    "inspect_column",
    "fill_missing",
    "convert_type",
    "remove_duplicates",
    "run_pipeline",
    "use_tool",
]

TOOL_NAME_VALUES = [
    "query_sql",
    "view_logs",
    "profile_data",
]

CONDITIONAL_ACTION_REQUIREMENTS = {
    "inspect_column": ["column"],
    "use_tool": ["tool_name"],
}


class Observation(BaseModel):
    dataset_preview: List[Dict[str, Any]]
    data_schema: Dict[str, str]
    visible_errors: List[str]
    available_tools: List[str]
    data_quality_score: float
    step_count: int


class Action(BaseModel):
    action_type: Literal[
        "inspect_column",
        "fill_missing",
        "convert_type",
        "remove_duplicates",
        "run_pipeline",
        "use_tool",
    ]
    column: Optional[str] = None
    tool_name: Optional[Literal["query_sql", "view_logs", "profile_data"]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    if HAS_MODEL_VALIDATOR:
        @model_validator(mode="after")
        def validate_action_payload(self):
            if self.action_type == "use_tool" and self.tool_name is None:
                raise ValueError("tool_name is required when action_type is 'use_tool'")

            if self.action_type != "use_tool" and self.tool_name is not None:
                raise ValueError("tool_name can only be provided for action_type 'use_tool'")

            if self.action_type == "inspect_column" and not self.column:
                raise ValueError("column is required when action_type is 'inspect_column'")

            return self
    else:  # pragma: no cover - Pydantic v1 fallback
        @root_validator(skip_on_failure=True)
        def validate_action_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            action_type = values.get("action_type")
            column = values.get("column")
            tool_name = values.get("tool_name")

            if action_type == "use_tool" and tool_name is None:
                raise ValueError("tool_name is required when action_type is 'use_tool'")

            if action_type != "use_tool" and tool_name is not None:
                raise ValueError("tool_name can only be provided for action_type 'use_tool'")

            if action_type == "inspect_column" and not column:
                raise ValueError("column is required when action_type is 'inspect_column'")

            return values


class Reward(BaseModel):
    score: float
    components: Dict[str, float]


class HiddenState(BaseModel):
    hidden_issues: List[str]
    revealed: List[str]
