"""
=======================================================
  STUDENT TASK DELAY PREDICTOR — STREAMLIT UI
=======================================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task Delay Predictor",
    page_icon="🎓",
    layout="centered"
)

# ─────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        background-color: #4C72B0;
        color: white;
        font-size: 18px;
        padding: 12px 40px;
        border-radius: 10px;
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #3a5a8f;
        color: white;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    .result-title {
        font-size: 14px;
        color: #888;
        margin-bottom: 5px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .result-value {
        font-size: 32px;
        font-weight: 800;
        margin: 5px 0;
    }
    .risk-low      { color: #27ae60; }
    .risk-medium   { color: #f39c12; }
    .risk-high     { color: #e67e22; }
    .risk-critical { color: #e74c3c; }
    .header-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .header-sub {
        text-align: center;
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #2c3e50;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #4C72B0;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        models = {
            "scaler"       : joblib.load("scaler.pkl"),
            "le_task"      : joblib.load("le_task.pkl"),
            "le_difficulty": joblib.load("le_difficulty.pkl"),
            "m_risk"       : joblib.load("model_risk.pkl"),
            "le_risk"      : joblib.load("le_risk.pkl"),
            "m_stress"     : joblib.load("model_stress.pkl"),
            "le_stress"    : joblib.load("le_stress.pkl"),
            "m_grade"      : joblib.load("model_grade.pkl"),
            "le_grade"     : joblib.load("le_grade.pkl"),
            "m_completion" : joblib.load("model_completion.pkl"),
            "le_completion": joblib.load("le_completion.pkl"),
        }
        return models, None
    except Exception as e:
        return None, str(e)

models, error = load_models()

# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
st.markdown('<div class="header-title">🎓 Task Delay Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Enter your task details below to predict delay risk and outcomes</div>', unsafe_allow_html=True)

if error:
    st.error(f"⚠️ Could not load models: {error}\n\nMake sure you ran **train_model.py** first and all .pkl files are in the same folder.")
    st.stop()

# ─────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Task Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    task = st.selectbox(
        "📚 Task Type",
        options=["assignment", "lab_experiment", "exam", "project"],
        format_func=lambda x: {
            "assignment"    : "📝 Assignment",
            "lab_experiment": "🔬 Lab Experiment",
            "exam"          : "📄 Exam",
            "project"       : "💻 Project"
        }[x]
    )

    difficulty = st.selectbox(
        "⚡ Difficulty Level",
        options=["easy", "medium", "hard", "very_hard"],
        format_func=lambda x: {
            "easy"     : "🟢 Easy",
            "medium"   : "🟡 Medium",
            "hard"     : "🟠 Hard",
            "very_hard": "🔴 Very Hard"
        }[x]
    )

    preparation = st.selectbox(
        "📖 Preparation Level",
        options=["low", "medium", "high"],
        format_func=lambda x: {
            "low"   : "🔴 Low",
            "medium": "🟡 Medium",
            "high"  : "🟢 High"
        }[x]
    )

with col2:
    delay_days = st.number_input(
        "⏰ Delay Days",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
        help="How many days have you already delayed this task?"
    )

    remaining_days = st.number_input(
        "📅 Remaining Days Until Deadline",
        min_value=0,
        max_value=60,
        value=10,
        step=1,
        help="How many days are left before the deadline?"
    )

    task_name = st.text_input(
        "✏️ Task Name (optional)",
        placeholder="e.g. Final Year Project, Mid-Sem Exam...",
        help="Just for display — doesn't affect prediction"
    )

# ─────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Now")

# ─────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────
PREP_MAP   = {"low": 20, "medium": 50, "high": 80}
GRADE_MAP  = {"0-5%": "5%",  "6-12%": "12%", "13-20%": "18%", "21-30%": "25%"}
COMP_MAP   = {"<40%": "35%", "40-60%": "55%","60-80%": "72%", ">80%":   "85%"}

def run_prediction(task, delay_days, difficulty, remaining_days, preparation):
    prep_percent = PREP_MAP.get(preparation, 50)

    # Estimate risk score
    risk = 0
    if remaining_days <= 3:    risk += 25
    elif remaining_days <= 7:  risk += 15
    elif remaining_days <= 14: risk += 8
    diff_map = {"easy": 0, "medium": 5, "hard": 12, "very_hard": 20}
    risk += diff_map.get(difficulty, 5)
    risk += min(delay_days * 2, 20)
    risk += (1 - prep_percent / 100) * 20
    risk  = float(min(max(risk, 0), 100))

    input_df = pd.DataFrame([{
        "task_type"          : models["le_task"].transform([task])[0],
        "delay_days"         : delay_days,
        "difficulty_level"   : models["le_difficulty"].transform([difficulty])[0],
        "remaining_days"     : remaining_days,
        "progress_percent"   : prep_percent,
        "risk_score"         : risk,
        "past_delay_history" : 3,
        "required_hours"     : 20,
        "current_cgpa"       : 7.0,
    }])

    inp_scaled   = models["scaler"].transform(input_df)
    risk_level   = models["le_risk"].inverse_transform(models["m_risk"].predict(inp_scaled))[0]
    stress_level = models["le_stress"].inverse_transform(models["m_stress"].predict(inp_scaled))[0]
    grade_bucket = models["le_grade"].inverse_transform(models["m_grade"].predict(inp_scaled))[0]
    comp_bucket  = models["le_completion"].inverse_transform(models["m_completion"].predict(inp_scaled))[0]

    return {
        "risk_level"   : risk_level,
        "stress_level" : stress_level,
        "grade_reduction": GRADE_MAP[grade_bucket],
        "completion_prob": COMP_MAP[comp_bucket],
        "risk_score"   : round(risk, 1),
    }

# ─────────────────────────────────────────────────────
# SHOW RESULTS
# ─────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Analyzing your task..."):
        result = run_prediction(task, delay_days, difficulty, remaining_days, preparation)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

    if task_name:
        st.markdown(f"**Task:** {task_name}")

    # Color mapping
    risk_color_map = {
        "Low"     : "risk-low",
        "Medium"  : "risk-medium",
        "High"    : "risk-high",
        "Critical": "risk-critical"
    }
    stress_color_map = {
        "Low"     : "risk-low",
        "Medium"  : "risk-medium",
        "High"    : "risk-high",
        "Critical": "risk-critical"
    }

    risk_class   = risk_color_map.get(result["risk_level"], "risk-medium")
    stress_class = stress_color_map.get(result["stress_level"], "risk-medium")

    # 4 result cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">🚨 Risk Level</div>
            <div class="result-value {risk_class}">{result['risk_level']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">📉 Grade Reduction</div>
            <div class="result-value risk-high">{result['grade_reduction']}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">✅ Completion Prob</div>
            <div class="result-value risk-low">{result['completion_prob']}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">😓 Stress Level</div>
            <div class="result-value {stress_class}">{result['stress_level']}</div>
        </div>""", unsafe_allow_html=True)

    # Risk Score Progress Bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**🎯 Overall Risk Score: {result['risk_score']} / 100**")
    st.progress(int(result["risk_score"]) / 100)

    # Advice Box
    st.markdown("<br>", unsafe_allow_html=True)
    risk = result["risk_level"]
    if risk == "Low":
        st.success("✅ **You're on track!** Keep up the good work and submit on time.")
    elif risk == "Medium":
        st.warning("⚠️ **Moderate risk detected.** Try to increase your preparation and reduce delays.")
    elif risk == "High":
        st.error("🔴 **High risk!** You should start working immediately and manage your time carefully.")
    elif risk == "Critical":
        st.error("🚨 **Critical risk!** Seek help immediately, talk to your professor, and prioritize this task above everything else.")

    # Input Summary
    with st.expander("📋 View Your Input Summary"):
        summary = {
            "Task Type"       : task,
            "Delay Days"      : f"{delay_days} days",
            "Difficulty"      : difficulty,
            "Remaining Days"  : f"{remaining_days} days",
            "Preparation"     : preparation,
        }
        for k, v in summary.items():
            st.write(f"**{k}:** {v}")

# ─────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#aaa; font-size:13px;'>Built with Logistic Regression · Trained on 10,000 student task records</center>",
    unsafe_allow_html=True
)
