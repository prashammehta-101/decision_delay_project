"""
=======================================================
  STUDENT TASK DELAY PREDICTOR
  RBAC: Admin + Student | Interactive Graphs
=======================================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings("ignore")

st.set_page_config(page_title="TaskIQ — Delay Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); border-right: 1px solid rgba(99,102,241,0.3); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-tile { background:#fff; border-radius:14px; padding:20px; text-align:center; border:1px solid #e2e8f0; }
.metric-num { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700; margin:4px 0; }
.metric-label { font-size:12px; color:#94a3b8; font-weight:600; letter-spacing:1px; text-transform:uppercase; }
.result-card { border-radius:14px; padding:22px 18px; text-align:center; margin-bottom:8px; }
.rc-low    { background:linear-gradient(135deg,#d1fae5,#a7f3d0); border:2px solid #6ee7b7; }
.rc-medium { background:linear-gradient(135deg,#fef9c3,#fde68a); border:2px solid #fbbf24; }
.rc-high   { background:linear-gradient(135deg,#ffedd5,#fed7aa); border:2px solid #fb923c; }
.rc-critical { background:linear-gradient(135deg,#fee2e2,#fecaca); border:2px solid #f87171; }
.rc-blue   { background:linear-gradient(135deg,#dbeafe,#bfdbfe); border:2px solid #60a5fa; }
.rc-purple { background:linear-gradient(135deg,#ede9fe,#ddd6fe); border:2px solid #a78bfa; }
.rc-val  { font-family:'Space Mono',monospace; font-size:1.9rem; font-weight:700; color:#1e293b; }
.rc-lbl  { font-size:11px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:#475569; margin-top:4px; }
.sec-title { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; color:#1e293b; border-left:4px solid #6366f1; padding-left:12px; margin:24px 0 16px; }
.stButton > button { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:white !important; border:none !important; border-radius:12px !important; padding:14px 32px !important; font-size:16px !important; font-weight:600 !important; width:100% !important; box-shadow:0 4px 15px rgba(99,102,241,0.4) !important; }
.alert-success { background:#d1fae5; border:1px solid #6ee7b7; border-radius:10px; padding:14px 18px; color:#065f46; font-weight:500; }
.alert-warning { background:#fef9c3; border:1px solid #fbbf24; border-radius:10px; padding:14px 18px; color:#92400e; font-weight:500; }
.alert-danger  { background:#fee2e2; border:1px solid #fca5a5; border-radius:10px; padding:14px 18px; color:#991b1b; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ── USERS ──
USERS = {
    "admin":    {"password": "admin123", "role": "admin",   "name": "Admin"},
    "student1": {"password": "pass123",  "role": "student", "name": "Arihant"},
    "student2": {"password": "pass456",  "role": "student", "name": "Prasham"},
    "student3": {"password": "pass789",  "role": "student", "name": "Riya"},
}

# ── SESSION STATE ──
for k, v in [("logged_in",False),("username",""),("role",""),("name",""),("history",[]),("last_result",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── LOAD MODELS ──
@st.cache_resource
def load_models():
    try:
        return {k: joblib.load(f) for k, f in [
            ("scaler","scaler.pkl"),("le_task","le_task.pkl"),("le_difficulty","le_difficulty.pkl"),
            ("m_risk","model_risk.pkl"),("le_risk","le_risk.pkl"),
            ("m_stress","model_stress.pkl"),("le_stress","le_stress.pkl"),
            ("m_grade","model_grade.pkl"),("le_grade","le_grade.pkl"),
            ("m_completion","model_completion.pkl"),("le_completion","le_completion.pkl"),
        ]}, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_dataset():
    try:    return pd.read_csv("student_task_delay_dataset.csv"), None
    except Exception as e: return None, str(e)

models, model_error = load_models()
df_data, data_error = load_dataset()

PREP_MAP  = {"low":20,"medium":50,"high":80}
GRADE_MAP = {"0-5%":"5%","6-12%":"12%","13-20%":"18%","21-30%":"25%"}
COMP_MAP  = {"<40%":"35%","40-60%":"55%","60-80%":"72%",">80%":"85%"}
GRADE_NUM = {"0-5%":5,"6-12%":12,"13-20%":18,"21-30%":25}
COMP_NUM  = {"<40%":35,"40-60%":55,"60-80%":72,">80%":85}

def run_prediction(task, delay_days, difficulty, remaining_days, preparation):
    prep_percent = PREP_MAP.get(preparation, 50)
    risk = 0
    if remaining_days <= 3: risk += 25
    elif remaining_days <= 7: risk += 15
    elif remaining_days <= 14: risk += 8
    risk += {"easy":0,"medium":5,"hard":12,"very_hard":20}.get(difficulty,5)
    risk += min(delay_days * 2, 20)
    risk += (1 - prep_percent/100) * 20
    risk = float(min(max(risk,0),100))
    inp = pd.DataFrame([{"task_type":models["le_task"].transform([task])[0],"delay_days":delay_days,
        "difficulty_level":models["le_difficulty"].transform([difficulty])[0],"remaining_days":remaining_days,
        "progress_percent":prep_percent,"risk_score":risk,"past_delay_history":3,"required_hours":20,"current_cgpa":7.0}])
    s = models["scaler"].transform(inp)
    gb = models["le_grade"].inverse_transform(models["m_grade"].predict(s))[0]
    cb = models["le_completion"].inverse_transform(models["m_completion"].predict(s))[0]
    return {
        "risk_level":   models["le_risk"].inverse_transform(models["m_risk"].predict(s))[0],
        "stress_level": models["le_stress"].inverse_transform(models["m_stress"].predict(s))[0],
        "grade_reduction": GRADE_MAP[gb], "completion_prob": COMP_MAP[cb],
        "risk_score": round(risk,1), "grade_num": GRADE_NUM[gb], "comp_num": COMP_NUM[cb],
    }

# ── LOGIN ──
def show_login():
    _, cc, _ = st.columns([1,1.1,1])
    with cc:
        st.markdown("""
        <div style='text-align:center;padding:40px 0 20px'>
            <div style='font-size:52px'>🎓</div>
            <div style='font-family:Space Mono,monospace;font-size:1.8rem;font-weight:700;color:#1e293b'>TaskIQ</div>
            <div style='color:#64748b;font-size:14px;margin-top:6px'>Student Task Delay Intelligence System</div>
        </div>""", unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown("**Username**")
            username = st.text_input("", placeholder="Enter username", label_visibility="collapsed")
            st.markdown("**Password**")
            password = st.text_input("", type="password", placeholder="Enter password", label_visibility="collapsed")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign In →", use_container_width=True)
        if submitted:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.update(logged_in=True, username=username,
                    role=USERS[username]["role"], name=USERS[username]["name"])
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        st.markdown("""
        <div style='background:#f8fafc;border-radius:12px;padding:16px;margin-top:16px;border:1px solid #e2e8f0'>
            <div style='font-size:12px;font-weight:700;color:#64748b;margin-bottom:8px'>DEMO CREDENTIALS</div>
            <div style='font-size:13px;color:#475569'>
                👑 <b>admin</b> / admin123 — Admin Dashboard<br>
                🎓 <b>student1</b> / pass123 — Student View
            </div>
        </div>""", unsafe_allow_html=True)

# ── SIDEBAR ──
def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:20px 0 10px'>
            <div style='font-family:Space Mono,monospace;font-size:1.3rem;font-weight:700;color:#e2e8f0'>🎓 TaskIQ</div>
            <div style='font-size:12px;color:#94a3b8;margin-top:3px'>Delay Intelligence System</div>
        </div>
        <hr style='border-color:rgba(99,102,241,0.3);margin:10px 0'>
        <div style='padding:10px 0'>
            <div style='font-size:13px;color:#94a3b8'>Signed in as</div>
            <div style='font-size:16px;font-weight:600;color:#e2e8f0;margin-top:2px'>{st.session_state.name}</div>
            <span style='background:{"#4c1d95" if st.session_state.role=="admin" else "#1e3a5f"};
                color:{"#c4b5fd" if st.session_state.role=="admin" else "#93c5fd"};
                font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;display:inline-block;margin-top:6px'>
                {"👑 ADMIN" if st.session_state.role=="admin" else "🎓 STUDENT"}
            </span>
        </div>
        <hr style='border-color:rgba(99,102,241,0.3);margin:10px 0'>
        """, unsafe_allow_html=True)
        if st.button("🚪  Sign Out", use_container_width=True):
            st.session_state.update(logged_in=False,username="",role="",name="",history=[],last_result=None)
            st.rerun()

# ── STUDENT PAGE ──
def show_student():
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:1.5rem;font-weight:700;color:#1e293b;margin-bottom:6px'>
        Welcome, {st.session_state.name} 👋
    </div>
    <div style='color:#64748b;font-size:14px;margin-bottom:24px'>
        Enter your task details to get an instant delay risk prediction with graphs.
    </div>""", unsafe_allow_html=True)

    if model_error:
        st.error(f"⚠️ Models not found. Run train_model.py first.\n\n`{model_error}`"); return

    st.markdown('<div class="sec-title">📋 Task Details</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            task_name = st.text_input("✏️ Task Name", placeholder="e.g. Final Year Project")
            task      = st.selectbox("📚 Task Type", ["assignment","lab_experiment","exam","project"],
                format_func=lambda x: {"assignment":"📝 Assignment","lab_experiment":"🔬 Lab Experiment","exam":"📄 Exam","project":"💻 Project"}[x])
            difficulty = st.selectbox("⚡ Difficulty", ["easy","medium","hard","very_hard"],
                format_func=lambda x: {"easy":"🟢 Easy","medium":"🟡 Medium","hard":"🟠 Hard","very_hard":"🔴 Very Hard"}[x])
        with c2:
            delay_days     = st.slider("⏰ Days Already Delayed", 0, 30, 3)
            remaining_days = st.slider("📅 Days Until Deadline", 0, 60, 10)
        with c3:
            preparation = st.selectbox("📖 Preparation Level", ["low","medium","high"],
                format_func=lambda x: {"low":"🔴 Low","medium":"🟡 Medium","high":"🟢 High"}[x])
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Predict Delay Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing..."):
            r = run_prediction(task, delay_days, difficulty, remaining_days, preparation)
            r.update(task_name=task_name or task, task_type=task, difficulty=difficulty,
                     delay_days=delay_days, remaining_days=remaining_days,
                     preparation=preparation, timestamp=datetime.now().strftime("%H:%M:%S"))
            st.session_state.last_result = r
            st.session_state.history.append(r)

    if st.session_state.last_result:
        r = st.session_state.last_result
        st.markdown("---")
        st.markdown('<div class="sec-title">📊 Prediction Results</div>', unsafe_allow_html=True)

        risk_cls   = {"Low":"rc-low","Medium":"rc-medium","High":"rc-high","Critical":"rc-critical"}.get(r["risk_level"],"rc-medium")
        stress_cls = {"Low":"rc-low","Medium":"rc-medium","High":"rc-high","Critical":"rc-critical"}.get(r["stress_level"],"rc-medium")

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="result-card {risk_cls}"><div class="rc-val">{r["risk_level"]}</div><div class="rc-lbl">🚨 Risk Level</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="result-card rc-high"><div class="rc-val">{r["grade_reduction"]}</div><div class="rc-lbl">📉 Grade Reduction</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="result-card rc-blue"><div class="rc-val">{r["completion_prob"]}</div><div class="rc-lbl">✅ Completion Prob</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="result-card {stress_cls}"><div class="rc-val">{r["stress_level"]}</div><div class="rc-lbl">😓 Stress Level</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)

        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=r["risk_score"],
                title={"text":"Overall Risk Score","font":{"size":16}}, delta={"reference":50},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#6366f1","thickness":0.3},
                       "steps":[{"range":[0,30],"color":"#d1fae5"},{"range":[30,55],"color":"#fef9c3"},
                                 {"range":[55,75],"color":"#ffedd5"},{"range":[75,100],"color":"#fee2e2"}],
                       "threshold":{"line":{"color":"#ef4444","width":4},"thickness":0.75,"value":75}}
            ))
            fig_g.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"))
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            cats  = ["Delay Impact","Difficulty","Deadline Urgency","Prep Gap","Risk Score"]
            vals  = [min(r["delay_days"]/14*100,100),
                     {"easy":20,"medium":40,"hard":70,"very_hard":90}.get(r["difficulty"],50),
                     max(0,min(100,(1-r["remaining_days"]/30)*100)),
                     (1-PREP_MAP.get(r["preparation"],50)/100)*100,
                     r["risk_score"]]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                fillcolor="rgba(99,102,241,0.2)", line=dict(color="#6366f1",width=2)))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                showlegend=False, height=280, margin=dict(t=40,b=10,l=40,r=40),
                paper_bgcolor="rgba(0,0,0,0)", title=dict(text="Risk Profile Radar",font=dict(size=16)),
                font=dict(family="DM Sans"))
            st.plotly_chart(fig_r, use_container_width=True)

        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(x=["Grade Reduction (%)","Completion Prob (%)"],
            y=[r["grade_num"],r["comp_num"]], marker_color=["#fb923c","#34d399"],
            text=[f'{r["grade_num"]}%',f'{r["comp_num"]}%'], textposition="outside", width=0.35))
        fig_b.update_layout(title="Outcome Breakdown", yaxis=dict(range=[0,115],gridcolor="#f1f5f9"),
            xaxis=dict(showgrid=False), height=280, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50,b=20,l=20,r=20), font=dict(family="DM Sans"))
        st.plotly_chart(fig_b, use_container_width=True)

        advice = {"Low":("alert-success","✅ You're on track! Keep up the consistency and submit on time."),
                  "Medium":("alert-warning","⚠️ Moderate risk detected. Increase study hours and reduce further delays."),
                  "High":("alert-danger","🔴 High risk! Start working immediately — every day counts now."),
                  "Critical":("alert-danger","🚨 Critical risk! Contact your professor and prioritize this above everything else.")}
        cls, msg = advice.get(r["risk_level"],("alert-info",""))
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

        if len(st.session_state.history) > 1:
            st.markdown('<div class="sec-title">🕓 Your Prediction History</div>', unsafe_allow_html=True)
            hdf = pd.DataFrame([{"Time":h["timestamp"],"Task":h["task_name"],"Risk":h["risk_level"],
                "Grade Cut":h["grade_reduction"],"Completion":h["completion_prob"],"Stress":h["stress_level"],
                "Score":h["risk_score"]} for h in st.session_state.history])
            st.dataframe(hdf, use_container_width=True, hide_index=True)

# ── ADMIN PAGE ──
def show_admin():
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:1.5rem;font-weight:700;color:#1e293b;margin-bottom:6px'>
        👑 Admin Dashboard
    </div>
    <div style='color:#64748b;font-size:14px;margin-bottom:24px'>
        Full overview of dataset statistics, model accuracy, and live prediction.
    </div>""", unsafe_allow_html=True)

    if data_error: st.error(f"Dataset not found: {data_error}"); return

    df = df_data.copy()
    total = len(df); will_delay = df["will_delay"].sum(); wont = total - will_delay
    avg_risk = round(df["risk_score"].mean(),1); avg_delay = round(df[df["delay_days"]>0]["delay_days"].mean(),1)

    st.markdown('<div class="sec-title">📈 Dataset Overview</div>', unsafe_allow_html=True)
    for col, num, lbl, color in zip(st.columns(5),
        [f"{total:,}",f"{will_delay:,}",f"{wont:,}",f"{avg_risk}",f"{avg_delay}d"],
        ["Total Records","Will Delay","On Time","Avg Risk Score","Avg Delay Days"],
        ["#6366f1","#ef4444","#10b981","#f59e0b","#3b82f6"]):
        with col:
            st.markdown(f'<div class="metric-tile"><div class="metric-num" style="color:{color}">{num}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    tabs = st.tabs(["📊 Dataset Analytics","🤖 Model Accuracy","🔮 Live Prediction"])

    with tabs[0]:
        r1, r2 = st.columns(2)
        with r1:
            dc = df["will_delay"].value_counts()
            fig = go.Figure(go.Pie(labels=["Will Delay","On Time"],values=[dc.get(1,0),dc.get(0,0)],
                hole=0.55, marker_colors=["#ef4444","#10b981"], textinfo="percent+label"))
            fig.update_layout(title="Delay Distribution",height=320,paper_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),margin=dict(t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            tc = df["task_type"].value_counts().reset_index(); tc.columns=["Task Type","Count"]
            fig2 = px.bar(tc,x="Task Type",y="Count",color="Count",color_continuous_scale="Purples",title="Tasks by Type")
            fig2.update_layout(height=320,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),showlegend=False,yaxis=dict(gridcolor="#f1f5f9"),coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        r3, r4 = st.columns(2)
        with r3:
            fig3 = px.histogram(df,x="risk_score",nbins=30,color_discrete_sequence=["#6366f1"],title="Risk Score Distribution")
            fig3.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),yaxis=dict(gridcolor="#f1f5f9"),bargap=0.05)
            st.plotly_chart(fig3, use_container_width=True)
        with r4:
            dd = df.groupby("difficulty_level")["will_delay"].mean().reset_index()
            dd.columns=["Difficulty","Delay Rate"]; dd["Delay Rate"]=(dd["Delay Rate"]*100).round(1)
            dd["Difficulty"]=pd.Categorical(dd["Difficulty"],["easy","medium","hard","very_hard"],ordered=True)
            dd=dd.sort_values("Difficulty")
            fig4 = px.bar(dd,x="Difficulty",y="Delay Rate",color="Delay Rate",color_continuous_scale="Reds",title="Delay Rate by Difficulty (%)")
            fig4.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),yaxis=dict(gridcolor="#f1f5f9"),coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

        fig5 = px.scatter(df.sample(1000,random_state=42),x="risk_score",y="delay_days",color="difficulty_level",
            size="remaining_days",title="Risk Score vs Delay Days (1000 sample)",
            color_discrete_map={"easy":"#10b981","medium":"#f59e0b","hard":"#f97316","very_hard":"#ef4444"})
        fig5.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),yaxis=dict(gridcolor="#f1f5f9"),xaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig5, use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="sec-title">🤖 Model Performance</div>', unsafe_allow_html=True)
        acc_info = [("Risk Level",97.45,97.52,"rc-low"),("Completion Prob",97.35,97.39,"rc-blue"),
                    ("Grade Reduction",67.35,68.71,"rc-medium"),("Stress Level",41.20,39.74,"rc-high")]
        for col,(name,acc,f1,cls) in zip(st.columns(4),acc_info):
            with col:
                st.markdown(f'<div class="result-card {cls}" style="padding:16px"><div style="font-size:11px;font-weight:700;letter-spacing:1px;color:#475569;text-transform:uppercase">{name}</div><div style="font-family:Space Mono,monospace;font-size:1.8rem;font-weight:700;color:#1e293b;margin:6px 0">{acc}%</div><div style="font-size:12px;color:#64748b">F1: {f1}%</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        ml = ["Risk Level","Stress Level","Grade Reduction","Completion Prob"]
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(name="Accuracy",x=ml,y=[97.45,41.20,67.35,97.35],marker_color="#6366f1",text=["97.45%","41.20%","67.35%","97.35%"],textposition="outside"))
        fig_acc.add_trace(go.Bar(name="F1 Score",x=ml,y=[97.52,39.74,68.71,97.39],marker_color="#34d399",text=["97.52%","39.74%","68.71%","97.39%"],textposition="outside"))
        fig_acc.add_hline(y=80,line_dash="dash",line_color="#ef4444",annotation_text="80% threshold",annotation_position="right")
        fig_acc.update_layout(barmode="group",title="Accuracy vs F1 Score",height=380,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),yaxis=dict(range=[0,115],gridcolor="#f1f5f9"),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig_acc, use_container_width=True)

        cv_d = {"Risk Level":[97.2,97.0,97.1,96.9,97.25],"Completion Prob":[97.9,97.1,97.5,97.6,97.95],"Grade Reduction":[66.8,69.6,67.2,68.5,67.5],"Stress Level":[40.2,43.1,41.0,40.8,42.5]}
        fig_cv = go.Figure()
        for (mn,scores),color in zip(cv_d.items(),["#6366f1","#10b981","#f59e0b","#ef4444"]):
            fig_cv.add_trace(go.Scatter(x=[f"Fold {i+1}" for i in range(5)],y=scores,mode="lines+markers",name=mn,line=dict(color=color,width=2),marker=dict(size=8)))
        fig_cv.update_layout(title="5-Fold Cross Validation",height=320,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),yaxis=dict(gridcolor="#f1f5f9",title="Accuracy (%)"),xaxis=dict(gridcolor="#f1f5f9"),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig_cv, use_container_width=True)

    with tabs[2]:
        st.markdown('<div class="sec-title">🔮 Live Prediction (Admin)</div>', unsafe_allow_html=True)
        if model_error: st.error("Models not loaded."); return
        with st.form("admin_predict"):
            ac1,ac2,ac3 = st.columns(3)
            with ac1:
                a_task = st.selectbox("Task Type",["assignment","lab_experiment","exam","project"])
                a_diff = st.selectbox("Difficulty",["easy","medium","hard","very_hard"])
            with ac2:
                a_delay = st.slider("Delay Days",0,30,5)
                a_rem   = st.slider("Remaining Days",0,60,15)
            with ac3:
                a_prep  = st.selectbox("Preparation",["low","medium","high"])
            go_btn = st.form_submit_button("▶ Run Prediction", use_container_width=True)

        if go_btn:
            ar = run_prediction(a_task,a_delay,a_diff,a_rem,a_prep)
            risk_cls = {"Low":"rc-low","Medium":"rc-medium","High":"rc-high","Critical":"rc-critical"}.get(ar["risk_level"],"rc-medium")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(f'<div class="result-card {risk_cls}"><div class="rc-val">{ar["risk_level"]}</div><div class="rc-lbl">Risk Level</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="result-card rc-high"><div class="rc-val">{ar["grade_reduction"]}</div><div class="rc-lbl">Grade Reduction</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="result-card rc-blue"><div class="rc-val">{ar["completion_prob"]}</div><div class="rc-lbl">Completion Prob</div></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="result-card rc-purple"><div class="rc-val">{ar["stress_level"]}</div><div class="rc-lbl">Stress Level</div></div>', unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(mode="gauge+number",value=ar["risk_score"],title={"text":"Risk Score / 100"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#6366f1"},
                       "steps":[{"range":[0,30],"color":"#d1fae5"},{"range":[30,55],"color":"#fef9c3"},
                                 {"range":[55,75],"color":"#ffedd5"},{"range":[75,100],"color":"#fee2e2"}]}))
            fig_g.update_layout(height=260,paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=40,b=0,l=30,r=30),font=dict(family="DM Sans"))
            st.plotly_chart(fig_g, use_container_width=True)

# ── ROUTER ──
if not st.session_state.logged_in:
    show_login()
else:
    show_sidebar()
    if st.session_state.role == "admin": show_admin()
    else: show_student()
