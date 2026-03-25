# DataOps Reinforcement Learning Environment

## Overview

This project presents a reinforcement learning (RL) environment designed to simulate real-world data operations (DataOps) workflows. The system enables agents to clean, debug, and optimize datasets through sequential decision-making, mimicking the behavior of a data engineer.

It combines:
- A structured environment for dataset manipulation
- A multi-dimensional data quality evaluation system
- A baseline intelligent agent
- A REST API for interaction
- An interactive Streamlit-based user interface

The objective is to demonstrate how intelligent agents can autonomously improve data quality through iterative actions.

---

## Key Features

### 1. DataOps Environment
A custom environment where agents can:
- Handle missing values
- Fix type inconsistencies
- Remove duplicates
- Execute pipeline validation
- Use diagnostic tools

### 2. Multi-Dimensional Data Quality Evaluation
The grading system evaluates:
- Completeness (missing values)
- Uniqueness (duplicates)
- Consistency (type correctness)
- Efficiency (steps taken)
- Pipeline success and debugging behavior

### 3. Reinforcement Learning Compatibility
- Reward function aligned with final grading
- Supports sequential decision-making
- Designed for future integration with RL algorithms (DQN, PPO)

### 4. Intelligent Baseline Agent
A rule-based agent that:
- Diagnoses issues using observations
- Prioritizes actions logically
- Achieves strong performance across tasks

### 5. REST API (FastAPI)
Endpoints:
- `/reset` — initialize environment
- `/step` — perform action
- `/state` — get full environment state
- `/grader` — evaluate current performance
- `/tasks` — list available tasks
- `/baseline` — run baseline agent
- `/debug` — inspect internal state

### 6. Streamlit Dashboard
Interactive UI for:
- Running tasks
- Executing actions step-by-step
- Visualizing dataset changes
- Monitoring data quality metrics
- Viewing action logs and reward trends

---

## Project Structure
```bash

├── env/
│ ├── environment.py # Core RL environment
│ ├── grader.py # Data quality and scoring logic
│ ├── models.py # Pydantic schemas
│ ├── tasks.py # Task definitions
│ ├── baseline.py # Baseline intelligent agent
│
├── server.py # FastAPI server
├── app.py # Streamlit UI
├── openenv.yaml # Environment configuration
├── testing_sample.csv # Example dataset
├── Dockerfile

```

---

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd <repo-name>
```


### 2. Install Dependencies
```bash
pip install -r requirements.txt
```


Recommended packages:
- fastapi
- uvicorn
- pandas
- numpy
- pydantic
- streamlit
- requests

---

## Running the System

### Start Backend API
```bash
uvicorn server:app --reload
```

API available at: http://127.0.0.1:8000/docs


### Start Streamlit UI

```bash
streamlit run app.py
```

---

## How It Works

### Step 1: Reset Environment
Initialize a task or load a dataset.

### Step 2: Observe State
Agent receives:
- Dataset preview
- Schema
- Visible errors
- Data quality score

### Step 3: Take Actions
Available actions:
- fill_missing
- convert_type
- remove_duplicates
- run_pipeline
- use_tool

### Step 4: Receive Reward
Reward is based on improvement in overall data quality score.

### Step 5: Iterate Until Completion
The agent continues until:
- Data quality is high
- All issues are resolved
- Step limit is reached

---

## Tasks

### Easy
- Missing value handling
- Clear and direct problem

### Medium
- Type inconsistencies
- Invalid entries
- Requires reasoning

### Hard
- Mixed issues:
  - Duplicates
  - Missing values
  - Type mismatches
- Requires multi-step strategy

---

## Baseline Agent

The baseline agent:
- Uses diagnostic signals from visible errors
- Prioritizes fixes in correct order
- Achieves strong performance without learning

Typical strategy:
1. Profile data
2. Fix type issues
3. Handle missing values
4. Remove duplicates
5. Run pipeline validation

---

## Evaluation

Final score is computed using:

- Data quality (40%)
- Schema validity (20%)
- Pipeline success (20%)
- Hidden issue resolution (10%)
- Efficiency (10%)

Score range: 0.0 to 1.0


---

## Logs and Visualization

The UI provides:
- Action history
- Reward tracking over time
- Dataset preview updates
- Error diagnostics

This enables:
- Debugging agent behavior
- Understanding decision flow
- Evaluating improvements step-by-step

---

## Extensibility

The system is designed to support:
- Column-level actions
- Advanced validation rules
- Learned reward models
- Integration with RL frameworks
- Additional tools and diagnostics

---

## Use Cases

- Reinforcement learning experimentation
- Data cleaning automation research
- Simulation of real-world data engineering workflows
- Educational tool for data systems and AI

---

## Future Improvements

- Integration with RL algorithms (PPO, DQN)
- Advanced schema validation (e.g., Great Expectations-style)
- Action explainability
- Dataset diff visualization
- Multi-agent collaboration

---

## License

MIT License

---

## Author

Developed as part of a DataOps + Reinforcement Learning system for hackathon submission.