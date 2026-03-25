import streamlit as st
import requests
import pandas as pd
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DataOps RL UI", layout="wide")

# GLOBAL CSS FIX
st.markdown("""
<style>

/* ===== REMOVE STREAMLIT GRADIENT NAVBAR ===== */
header[data-testid="stHeader"] {
    background: #000000 !important;
    border-bottom: 1px solid #222;
}

/* Fix toolbar (Deploy button area) */
[data-testid="stToolbar"] {
    background: #000000 !important;
}

/* Fix overlap issue */
.block-container {
    padding-top: 2rem;
}

/* FULL BLACK BACKGROUND */
.stApp {
    background: #000000 !important;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: #000000 !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    border-radius: 12px;
    color: white;
    font-weight: bold;
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 10px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# REAL TITLE (NOT FAKE HEADER)
st.title("🚀 DataOps RL Command Center")

# SESSION STATE
if "logs" not in st.session_state:
    st.session_state.logs = []

# SIDEBAR
st.sidebar.header("⚙️ Control Panel")

task = st.sidebar.selectbox(
    "Task",
    ["missing_values_easy", "schema_fix_medium", "pipeline_debug_hard"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# RESET
if st.sidebar.button("🔄 Reset Environment"):
    with st.spinner("Resetting environment..."):
        time.sleep(0.8)

        if uploaded_file:
            with open("temp.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            response = requests.post(f"{API_URL}/reset?task_id={task}&data_path=testing_sample.csv")
        else:
            response = requests.post(f"{API_URL}/reset?task_id={task}")

    if response.status_code == 200:
        st.session_state.obs = response.json()["observation"]
        st.session_state.logs = []
        st.success("Environment Ready")
        # st.balloons()
    else:
        st.error(response.text)

# ACTIONS
st.sidebar.subheader("🎮 Actions")

action_type = st.sidebar.selectbox(
    "Action",
    ["fill_missing", "convert_type", "remove_duplicates", "run_pipeline", "use_tool"]
)

tool_name = None
if action_type == "use_tool":
    tool_name = st.sidebar.selectbox(
        "Tool",
        ["profile_data", "view_logs", "query_sql"]
    )

if st.sidebar.button("▶ Execute Step"):
    payload = {"action_type": action_type}
    if tool_name:
        payload["tool_name"] = tool_name

    progress = st.sidebar.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    response = requests.post(f"{API_URL}/step", json=payload)

    if response.status_code == 200:
        data = response.json()
        st.session_state.obs = data["observation"]

        st.session_state.logs.append({
            "action": payload,
            "reward": data["reward"]["score"],
            "done": data["done"]
        })

        st.sidebar.success("Step Executed")
    else:
        st.error(response.text)

# BASELINE
if st.sidebar.button("🤖 Run Baseline Agent"):
    with st.spinner("Running baseline..."):
        response = requests.get(f"{API_URL}/baseline")

    if response.status_code == 200:
        result = response.json()["result"]
        st.sidebar.success(f"Avg Score: {result['average_score']}")
    else:
        st.error(response.text)

# MAIN UI
if "obs" in st.session_state:

    obs = st.session_state.obs

    tabs = st.tabs(["📊 Data", "⚠️ Errors", "📈 Metrics", "📜 Logs"])

    # -------- DATA TAB --------
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Preview")
            df = pd.DataFrame(obs["dataset_preview"])
            st.dataframe(df, use_container_width=True)

        with col2:
            st.subheader("Schema")
            st.json(obs["data_schema"])

    # -------- ERRORS TAB --------
    with tabs[1]:
        st.subheader("Detected Issues")
        if obs["visible_errors"]:
            for err in obs["visible_errors"]:
                st.warning(err)
        else:
            st.success("No errors detected")

    # -------- METRICS TAB --------
    with tabs[2]:
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Quality Score", round(obs["data_quality_score"], 4))
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Step Count", obs["step_count"])
            st.markdown('</div>', unsafe_allow_html=True)

    # -------- LOGS TAB --------
    with tabs[3]:
        if st.session_state.logs:
            log_df = pd.DataFrame(st.session_state.logs)
            st.dataframe(log_df, use_container_width=True)

            st.subheader("Reward Trend")
            rewards = [log["reward"] for log in st.session_state.logs]
            st.line_chart(rewards)
        else:
            st.info("No actions yet")

else:
    st.info("Initialize environment to begin")