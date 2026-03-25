import pandas as pd
import numpy as np


# =========================
# DATA QUALITY METRIC
# =========================
def compute_data_quality(data):
    if not data:
        return 0.0

    df = pd.DataFrame(data)

    # safety guard
    if df.empty or df.size == 0:
        return 0.0

    total_cells = df.size

    # -------- 1. COMPLETENESS --------
    missing_mask = df.isnull() | (df.astype(str).apply(lambda x: x.str.strip()) == "")
    missing_ratio = missing_mask.sum().sum() / total_cells

    # -------- 2. UNIQUENESS --------
    duplicate_ratio = df.duplicated().sum() / max(1, len(df))

    # -------- 3. TYPE CONSISTENCY --------
    type_penalty = 0

    for col in df.columns:
        if df[col].dtype == "object":
            non_null = df[col].dropna()

            if len(non_null) == 0:
                continue

            numeric_like = pd.to_numeric(non_null, errors="coerce")
            valid_ratio = numeric_like.notnull().mean()

            # if mostly numeric but stored as object → bad
            if valid_ratio > 0.7:
                type_penalty += 1

    type_penalty = type_penalty / len(df.columns)

    # -------- 4. VALIDITY (basic numeric sanity) --------
    validity_penalty = 0

    for col in df.columns:
        if df[col].dtype != "object":
            if df[col].isnull().all():
                continue

            # detect extreme anomalies (very rough)
            try:
                std = df[col].std()
                if std == 0 or np.isnan(std):
                    continue
            except:
                continue

    # -------- FINAL SCORE --------
    score = 1.0

    score -= 0.4 * missing_ratio
    score -= 0.3 * duplicate_ratio
    score -= 0.3 * type_penalty

    return max(0.0, min(score, 1.0))


# =========================
# FINAL GRADER
# =========================
def grade(env_state):
    data = env_state.get("data", [])

    quality = compute_data_quality(data)

    schema_valid = env_state.get("schema_valid", 0)
    pipeline_success = env_state.get("pipeline_success", 0)
    hidden_resolved = env_state.get("hidden_resolved", 0)
    steps = env_state.get("step_count", 1)

    # -------- NORMALIZATION --------
    hidden_score = min(hidden_resolved / 3, 1.0)

    # stronger efficiency pressure
    efficiency = max(0.0, 1 - steps / 12)

    # -------- FINAL SCORE --------
    score = 0.0

    # core signal (most important)
    score += 0.4 * quality

    # correctness
    score += 0.2 * schema_valid
    score += 0.2 * pipeline_success

    # debugging behavior
    score += 0.1 * hidden_score

    # efficiency
    score += 0.1 * efficiency

    return max(0.0, min(score, 1.0))