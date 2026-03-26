---
title: DataOps RL Environment
emoji: "\U0001F9F9"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
  - agent-evaluation
---

# DataOps RL Environment (OpenEnv Hackathon Submission)

This project is a real-world OpenEnv environment where an agent acts like a DataOps engineer: it diagnoses tabular data issues, applies transformations, and decides when to run the pipeline.

## Why this environment is useful

Real teams repeatedly perform this workflow in production ETL/ELT systems:
- detect missing values and schema mismatches
- repair numeric parsing and duplicate records
- validate whether a data pipeline run is safe and successful

The environment is designed for trajectory learning and evaluation, not one-shot classification.

## OpenEnv API compliance

Environment class: `env.environment:DataOpsEnv`

Implemented API:
- `reset(task_id, data_path=None) -> Observation`
- `step(action: Action) -> (Observation, Reward, done, info)`
- `state() -> dict`

Typed Pydantic models:
- `Observation` in `env/models.py`
- `Action` in `env/models.py`
- `Reward` in `env/models.py`

Manifest:
- `openenv.yaml` includes entrypoint, schemas, actions/tools, reward signals, task metadata, and endpoints.

## Action / observation / reward spaces

### Observation
- `dataset_preview`: first rows of working data
- `data_schema`: inferred schema after each step
- `visible_errors`: currently detectable issues and tool findings
- `available_tools`: `query_sql`, `view_logs`, `profile_data`
- `data_quality_score`: scalar quality signal
- `step_count`: current timestep

### Action
- `action_type` (required): one of
  - `inspect_column`
  - `fill_missing`
  - `convert_type`
  - `remove_duplicates`
  - `run_pipeline`
  - `use_tool`
- `tool_name` is required only when `action_type=use_tool`
- `column` is required only when `action_type=inspect_column`
- `parameters` optional dictionary

### Reward
Dense reward with partial progress and penalties:
- positive components: `score_delta`, `quality_delta`, `hidden_progress`, `pipeline_bonus`, `tool_bonus`
- penalties: `repeat_penalty`, `no_op_penalty`, `action_cost`
- final `reward.score` is clipped to `[-1.0, 1.0]`

## Tasks (easy -> medium -> hard)

1. `missing_values_easy`
- Objective: impute null values and run a clean pipeline pass
- Difficulty: easy
- Success threshold: `grader_score >= 0.95`

2. `schema_fix_medium`
- Objective: repair malformed numeric fields and resolve missing values
- Difficulty: medium
- Success threshold: `grader_score >= 0.93`

3. `pipeline_debug_hard`
- Objective: fix duplicates + nulls + type issues and complete the pipeline
- Difficulty: hard
- Success threshold: `grader_score >= 0.90`

## Deterministic grader (0.0 to 1.0)

`/grader` returns:
- `score`
- `done`
- `task_id`
- `components` (data_quality, completeness, uniqueness, schema_alignment, pipeline_success, hidden_resolution, efficiency)
- `weights` (task-specific deterministic weighting)

All scores are clamped to `[0.0, 1.0]`.

## Required endpoints

Implemented endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks` (includes `action_schema`, required fields, and conditional requirements)
- `GET /grader`
- `GET /baseline`

## Baseline inference (Groq via OpenAI SDK)

`env/baseline.py` uses the OpenAI Python SDK with Groq-compatible base URL:
- `base_url=https://api.groq.com/openai/v1`
- key resolution:
  1. `GROQ_API_KEY` (preferred)
  2. `OPENAI_API_KEY` (fallback compatibility)
- model:
  - default: `llama-3.3-70b-versatile`
  - override: `GROQ_MODEL`

If API key/model call is unavailable, baseline automatically falls back to a deterministic policy so `/baseline` remains functional.

## Reproducible baseline scores (fallback mode)

Generated with:
```bash
python -c "from env.baseline import run_baseline; print(run_baseline(force_policy='fallback'))"
```

Current scores:
- `missing_values_easy`: `0.9625`
- `schema_fix_medium`: `0.9867`
- `pipeline_debug_hard`: `0.9667`
- average: `0.9719`

## Setup

### Local
```bash
pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t dataops-openenv .
docker run -p 8000:8000 dataops-openenv
```

### Optional UI
```bash
streamlit run ui.py
```

## Validation checklist

Recommended pre-submission checks:
```bash
pytest -q tests
openenv validate
```

`openenv validate` is optional locally if the OpenEnv CLI is not installed; required in CI/submission.

## Repository structure

```text
env/
  environment.py
  grader.py
  models.py
  tasks.py
  baseline.py
server/
  app.py
openenv.yaml
Dockerfile
ui.py
```

## License

MIT
