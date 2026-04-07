"""
=======================================================
  STUDENT TASK DELAY — LOGISTIC REGRESSION MODEL
=======================================================
INPUT  : Task, Delay, Difficulty, Remaining Time, Preparation
OUTPUT : Risk Level, Grade Reduction, Completion Probability, Stress Level

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────
print("=" * 55)
print("  LOADING DATASET")
print("=" * 55)

df = pd.read_csv("student_task_delay_dataset.csv")
print(f"✅ Rows loaded     : {len(df)}")
print(f"✅ Missing values  : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────
# STEP 2 — CREATE OUTPUT LABELS FROM EXISTING COLUMNS
# ─────────────────────────────────────────────────────
def assign_risk_level(row):
    s = row["risk_score"]
    if s < 30:   return "Low"
    elif s < 55: return "Medium"
    elif s < 75: return "High"
    else:        return "Critical"

def assign_grade_reduction(row):
    d = max(row["delay_days"], 0)
    w = row["task_weight"]
    reduction = round((d / 14) * (w / 50) * 30, 1)
    return min(reduction, 30.0)

def assign_completion_prob(row):
    base = 100 - row["risk_score"]
    prog_bonus = row["progress_percent"] * 0.1
    prob = base + prog_bonus
    return round(min(max(prob, 5), 98), 1)

df["risk_level"]      = df.apply(assign_risk_level, axis=1)
df["grade_reduction"] = df.apply(assign_grade_reduction, axis=1)
df["completion_prob"] = df.apply(assign_completion_prob, axis=1)
df["stress_output"]   = df["stress_level"].str.capitalize()

# Bucket numeric outputs into categories for classification
df["grade_bucket"] = pd.cut(df["grade_reduction"],
                              bins=[-1, 5, 12, 20, 30],
                              labels=["0-5%", "6-12%", "13-20%", "21-30%"])
df["completion_bucket"] = pd.cut(df["completion_prob"],
                                  bins=[0, 40, 60, 80, 100],
                                  labels=["<40%", "40-60%", "60-80%", ">80%"])

print("\n✅ Output labels created:")
print("   Risk Level        :", df["risk_level"].value_counts().to_dict())
print("   Stress Output     :", df["stress_output"].value_counts().to_dict())

# ─────────────────────────────────────────────────────
# STEP 3 — FEATURE SELECTION & ENCODING
# ─────────────────────────────────────────────────────
PREP_MAP = {"low": 20, "medium": 50, "high": 80}

features = [
    "task_type", "delay_days", "difficulty_level",
    "remaining_days", "progress_percent",
    "risk_score", "past_delay_history",
    "required_hours", "current_cgpa"
]

X = df[features].copy()

le_task       = LabelEncoder().fit(["assignment", "exam", "lab_experiment", "project"])
le_difficulty = LabelEncoder().fit(["easy", "hard", "medium", "very_hard"])

X["task_type"]        = le_task.transform(X["task_type"])
X["difficulty_level"] = le_difficulty.transform(X["difficulty_level"])

joblib.dump(le_task,       "le_task.pkl")
joblib.dump(le_difficulty, "le_difficulty.pkl")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# ─────────────────────────────────────────────────────
# STEP 4 — TRAIN 4 LOGISTIC REGRESSION MODELS
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING 4 LOGISTIC REGRESSION MODELS")
print("=" * 55)

def train_lr_model(X_scaled, y_series, name):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    model = LogisticRegression(max_iter=1000, solver="lbfgs",
                                class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    cv  = cross_val_score(model, X_scaled, y_enc,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="accuracy")
    print(f"\n  [{name}]")
    print(f"    Test Accuracy   : {acc*100:.2f}%")
    print(f"    Cross-Val Acc   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    print(f"    Classes         : {list(le.classes_)}")
    return model, le

m_risk,       le_risk       = train_lr_model(X_scaled, df["risk_level"],         "Risk Level")
m_stress,     le_stress     = train_lr_model(X_scaled, df["stress_output"],       "Stress Level")
m_grade,      le_grade      = train_lr_model(X_scaled, df["grade_bucket"],        "Grade Reduction")
m_completion, le_completion = train_lr_model(X_scaled, df["completion_bucket"],   "Completion Prob")

joblib.dump(m_risk,       "model_risk.pkl")
joblib.dump(le_risk,      "le_risk.pkl")
joblib.dump(m_stress,     "model_stress.pkl")
joblib.dump(le_stress,    "le_stress.pkl")
joblib.dump(m_grade,      "model_grade.pkl")
joblib.dump(le_grade,     "le_grade.pkl")
joblib.dump(m_completion, "model_completion.pkl")
joblib.dump(le_completion,"le_completion.pkl")

print("\n✅ All 4 models saved!")

# ─────────────────────────────────────────────────────
# STEP 5 — EVALUATION CHARTS
# ─────────────────────────────────────────────────────
print("\n⏳ Generating evaluation charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Logistic Regression — 4 Output Models Evaluation", fontsize=14, fontweight="bold")

model_info = [
    (m_risk,       le_risk,       df["risk_level"],         "Risk Level"),
    (m_stress,     le_stress,     df["stress_output"],       "Stress Level"),
    (m_grade,      le_grade,      df["grade_bucket"],        "Grade Reduction"),
    (m_completion, le_completion, df["completion_bucket"],   "Completion Probability"),
]

for ax, (model, le, y_col, title) in zip(axes.flat, model_info):
    y_enc = le.transform(y_col)
    _, X_test, _, y_test = train_test_split(X_scaled, y_enc, test_size=0.2,
                                             random_state=42, stratify=y_enc)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f"{title}  |  Acc: {accuracy_score(y_test, y_pred)*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
print("✅ Evaluation chart saved → model_evaluation.png")

# ─────────────────────────────────────────────────────
# STEP 6 — PREDICTION FUNCTION
# ─────────────────────────────────────────────────────
GRADE_MAP = {"0-5%": "5%", "6-12%": "12%", "13-20%": "18%", "21-30%": "25%"}
COMP_MAP  = {"<40%": "35%", "40-60%": "55%", "60-80%": "72%", ">80%": "85%"}

def predict(task, delay_days, difficulty, remaining_days, preparation):
    """
    task           : 'assignment' / 'lab_experiment' / 'exam' / 'project'
    delay_days     : number  e.g. 8
    difficulty     : 'easy' / 'medium' / 'hard' / 'very_hard'
    remaining_days : number  e.g. 25
    preparation    : 'low' / 'medium' / 'high'
    """
    prep_percent = PREP_MAP.get(preparation.lower(), 50)

    # Estimate risk_score from user inputs
    risk = 0
    if remaining_days <= 3:    risk += 25
    elif remaining_days <= 7:  risk += 15
    elif remaining_days <= 14: risk += 8
    diff_map = {"easy": 0, "medium": 5, "hard": 12, "very_hard": 20}
    risk += diff_map.get(difficulty.lower(), 5)
    risk += min(delay_days * 2, 20)
    risk += (1 - prep_percent / 100) * 20
    risk  = float(min(max(risk, 0), 100))

    input_df = pd.DataFrame([{
        "task_type"          : le_task.transform([task.lower()])[0],
        "delay_days"         : delay_days,
        "difficulty_level"   : le_difficulty.transform([difficulty.lower()])[0],
        "remaining_days"     : remaining_days,
        "progress_percent"   : prep_percent,
        "risk_score"         : risk,
        "past_delay_history" : 3,
        "required_hours"     : 20,
        "current_cgpa"       : 7.0,
    }])

    inp_scaled   = scaler.transform(input_df)
    risk_level   = le_risk.inverse_transform(m_risk.predict(inp_scaled))[0]
    stress_level = le_stress.inverse_transform(m_stress.predict(inp_scaled))[0]
    grade_bucket = le_grade.inverse_transform(m_grade.predict(inp_scaled))[0]
    comp_bucket  = le_completion.inverse_transform(m_completion.predict(inp_scaled))[0]

    print("\n" + "=" * 45)
    print("  📥 INPUT")
    print("=" * 45)
    print(f"  Task              : {task}")
    print(f"  Delay             : {delay_days} days")
    print(f"  Difficulty        : {difficulty}")
    print(f"  Remaining Time    : {remaining_days} days")
    print(f"  Preparation       : {preparation}")
    print("=" * 45)
    print("  📤 OUTPUT")
    print("=" * 45)
    print(f"  Risk Level        : {risk_level}")
    print(f"  Grade Reduction   : {GRADE_MAP[grade_bucket]}")
    print(f"  Completion Prob   : {COMP_MAP[comp_bucket]}")
    print(f"  Stress Level      : {stress_level}")
    print("=" * 45)

    return {
        "risk_level"       : risk_level,
        "grade_reduction"  : GRADE_MAP[grade_bucket],
        "completion_prob"  : COMP_MAP[comp_bucket],
        "stress_level"     : stress_level,
    }

# ─────────────────────────────────────────────────────
# STEP 7 — DEMO  (your exact example)
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DEMO PREDICTION")
print("=" * 55)

print("\n" + "=" * 45)
print("  ENTER YOUR TASK DETAILS")
print("=" * 45)

task           = input("Task (assignment/lab_experiment/exam/project) : ")
delay_days     = int(input("Delay days (0, 3, 5, 8 etc)                  : "))
difficulty     = input("Difficulty (easy/medium/hard/very_hard)      : ")
remaining_days = int(input("Remaining days (5, 10, 25 etc)               : "))
preparation    = input("Preparation (low/medium/high)                 : ")

predict(task, delay_days, difficulty, remaining_days, preparation)

print("\n🎉 Done! All files saved in your folder.")
