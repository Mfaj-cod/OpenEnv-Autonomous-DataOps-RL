from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any


# OBSERVATION
class Observation(BaseModel):
    dataset_preview: List[Dict[str, Any]]
    data_schema: Dict[str, str]   # renamed from 'schema'
    visible_errors: List[str]
    available_tools: List[str]
    data_quality_score: float
    step_count: int


# ACTION
class Action(BaseModel):
    action_type: Literal[
        "inspect_column",
        "fill_missing",
        "convert_type",
        "remove_duplicates",
        "run_pipeline",
        "use_tool"
    ]

    # optional: enables future column-level actions
    column: Optional[str] = None

    tool_name: Optional[Literal[
        "query_sql",
        "view_logs",
        "profile_data"
    ]] = None

    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


# REWARD
class Reward(BaseModel):
    score: float
    components: Dict[str, float]


# OPTIONAL: DEBUG / EXTENSION
class HiddenState(BaseModel):
    hidden_issues: List[str]
    revealed: List[str]