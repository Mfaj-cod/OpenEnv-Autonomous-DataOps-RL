import os
import tempfile
import time

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TASK_OPTIONS = [
    "missing_values_easy",
    "schema_fix_medium",
    "pipeline_debug_hard",
]
ACTION_OPTIONS = [
    "inspect_column",
    "fill_missing",
    "convert_type",
    "remove_duplicates",
    "run_pipeline",
    "use_tool",
]
TOOL_OPTIONS = ["profile_data", "view_logs", "query_sql"]

st.set_page_config(page_title="DataOps RL UI", layout="wide")

st.markdown(
    """
<style>
header[data-testid="stHeader"] {
    background: #08121f !important;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stToolbar"] {
    background: #08121f !important;
}

.block-container {
    padding-top: 2rem;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(35, 115, 202, 0.24), transparent 35%),
        radial-gradient(circle at top right, rgba(28, 184, 125, 0.20), transparent 30%),
        linear-gradient(180deg, #08121f 0%, #050913 100%) !important;
    color: #f5f7fb;
}

[data-testid="stSidebar"] {
    background: rgba(5, 9, 19, 0.94) !important;
}

.stButton button {
    background: linear-gradient(90deg, #2373ca, #1cb87d);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 700;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


def _api_request(method: str, url: str, **kwargs):
    try:
        response = requests.request(method=method, url=url, timeout=45, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _save_uploaded_csv(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


if "logs" not in st.session_state:
    st.session_state.logs = []
if "obs" not in st.session_state:
    st.session_state.obs = None
if "upload_path" not in st.session_state:
    st.session_state.upload_path = None
if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None
if "grader_result" not in st.session_state:
    st.session_state.grader_result = None

st.title("DataOps RL Command Center")
st.caption("Manual frontend for the current OpenEnv API. The submission script lives in inference.py.")

st.sidebar.header("Control Panel")
api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")
task = st.sidebar.selectbox("Task", TASK_OPTIONS)
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if st.sidebar.button("Reset Environment"):
    with st.spinner("Resetting environment..."):
        time.sleep(0.2)
        if uploaded_file is not None:
            st.session_state.upload_path = _save_uploaded_csv(uploaded_file)
            payload_url = f"{api_url}/reset?task_id={task}&data_path={st.session_state.upload_path}"
        else:
            st.session_state.upload_path = None
            payload_url = f"{api_url}/reset?task_id={task}"

        data, error = _api_request("POST", payload_url)

    if error:
        st.error(error)
    else:
        st.session_state.obs = data["observation"]
        st.session_state.logs = []
        st.session_state.grader_result = None
        st.success("Environment ready")

st.sidebar.subheader("Actions")
action_type = st.sidebar.selectbox("Action", ACTION_OPTIONS)

current_columns = []
if st.session_state.obs:
    current_columns = list(st.session_state.obs.get("data_schema", {}).keys())

column_name = None
if action_type == "inspect_column":
    if current_columns:
        column_name = st.sidebar.selectbox("Column", current_columns)
    else:
        column_name = st.sidebar.text_input("Column", value="")

tool_name = None
if action_type == "use_tool":
    tool_name = st.sidebar.selectbox("Tool", TOOL_OPTIONS)

if st.sidebar.button("Execute Step"):
    payload = {"action_type": action_type}
    if tool_name:
        payload["tool_name"] = tool_name
    if column_name:
        payload["column"] = column_name

    progress = st.sidebar.progress(0)
    for i in range(100):
        time.sleep(0.003)
        progress.progress(i + 1)

    data, error = _api_request("POST", f"{api_url}/step", json=payload)

    if error:
        st.error(error)
    else:
        st.session_state.obs = data["observation"]
        st.session_state.logs.append(
            {
                "action": payload,
                "reward": data["reward"]["score"],
                "done": data["done"],
            }
        )
        st.sidebar.success("Step executed")

baseline_force_fallback = st.sidebar.checkbox("Force fallback baseline", value=False)
if st.sidebar.button("Run Baseline Agent"):
    with st.spinner("Running baseline..."):
        query = "?force_fallback=true" if baseline_force_fallback else ""
        data, error = _api_request("GET", f"{api_url}/baseline{query}")

    if error:
        st.error(error)
    else:
        st.session_state.baseline_result = data["result"]
        result = st.session_state.baseline_result
        st.sidebar.success(
            f"Avg Score: {result['average_score']} ({result['policy_mode']})"
        )

if st.sidebar.button("Fetch Grader"):
    data, error = _api_request("GET", f"{api_url}/grader")
    if error:
        st.error(error)
    else:
        st.session_state.grader_result = data
        st.sidebar.success(f"Score: {data['score']}")

if st.session_state.obs:
    obs = st.session_state.obs
    tabs = st.tabs(["Data", "Errors", "Metrics", "Logs", "Baseline"])

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.subheader("Dataset Preview")
            st.dataframe(pd.DataFrame(obs["dataset_preview"]), width='stretch')
        with right:
            st.subheader("Schema")
            st.json(obs["data_schema"])

    with tabs[1]:
        st.subheader("Detected Issues")
        if obs["visible_errors"]:
            for error in obs["visible_errors"]:
                st.warning(error)
        else:
            st.success("No errors detected")

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Quality Score", round(obs["data_quality_score"], 4))
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Step Count", obs["step_count"])
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.grader_result:
            st.subheader("Grader")
            st.json(st.session_state.grader_result)

    with tabs[3]:
        if st.session_state.logs:
            log_df = pd.DataFrame(st.session_state.logs)
            st.dataframe(log_df, width='stretch')
            st.subheader("Reward Trend")
            st.line_chart([log["reward"] for log in st.session_state.logs])
        else:
            st.info("No actions yet")

    with tabs[4]:
        if st.session_state.baseline_result:
            st.subheader("Baseline Result")
            st.json(st.session_state.baseline_result)
        else:
            st.info("Run the baseline agent to inspect the current baseline output.")
else:
    st.info("Initialize the environment to begin.")
