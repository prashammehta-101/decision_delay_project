"""
=======================================================
  MODEL ACCURACY CHECKER
  Checks accuracy of all 4 trained models in detail
=======================================================
Run: python check_accuracy.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET & REBUILD LABELS
# ─────────────────────────────────────────────────────
print("=" * 60)
print("  LOADING DATA & TRAINED MODELS")
print("=" * 60)

df = pd.read_csv("student_task_delay_dataset.csv")

# Rebuild output labels (same as training)
def assign_risk_level(row):
    s = row["risk_score"]
    if s < 30:   return "Low"
    elif s < 55: return "Medium"
    elif s < 75: return "High"
    else:        return "Critical"

def assign_grade_reduction(row):
    d = max(row["delay_days"], 0)
    w = row["task_weight"]
    return min(round((d / 14) * (w / 50) * 30, 1), 30.0)

def assign_completion_prob(row):
    base = 100 - row["risk_score"]
    prob = base + row["progress_percent"] * 0.1
    return round(min(max(prob, 5), 98), 1)

df["risk_level"]      = df.apply(assign_risk_level, axis=1)
df["grade_reduction"] = df.apply(assign_grade_reduction, axis=1)
df["completion_prob"] = df.apply(assign_completion_prob, axis=1)
df["stress_output"]   = df["stress_level"].str.capitalize()

df["grade_bucket"] = pd.cut(df["grade_reduction"],
                              bins=[-1, 5, 12, 20, 30],
                              labels=["0-5%", "6-12%", "13-20%", "21-30%"])
df["completion_bucket"] = pd.cut(df["completion_prob"],
                                  bins=[0, 40, 60, 80, 100],
                                  labels=["<40%", "40-60%", "60-80%", ">80%"])

# Load encoders & scaler
le_task       = joblib.load("le_task.pkl")
le_difficulty = joblib.load("le_difficulty.pkl")
scaler        = joblib.load("scaler.pkl")

features = [
    "task_type", "delay_days", "difficulty_level",
    "remaining_days", "progress_percent",
    "risk_score", "past_delay_history",
    "required_hours", "current_cgpa"
]

X = df[features].copy()
X["task_type"]        = le_task.transform(X["task_type"])
X["difficulty_level"] = le_difficulty.transform(X["difficulty_level"])
X_scaled = scaler.transform(X)

print("✅ Dataset loaded  :", df.shape[0], "rows")
print("✅ Models loading  ...")

# Load all 4 models + their encoders
m_risk,       le_risk       = joblib.load("model_risk.pkl"),       joblib.load("le_risk.pkl")
m_stress,     le_stress     = joblib.load("model_stress.pkl"),     joblib.load("le_stress.pkl")
m_grade,      le_grade      = joblib.load("model_grade.pkl"),      joblib.load("le_grade.pkl")
m_completion, le_completion = joblib.load("model_completion.pkl"), joblib.load("le_completion.pkl")

print("✅ All 4 models loaded!")

# ─────────────────────────────────────────────────────
# STEP 2 — ACCURACY CHECK FUNCTION
# ─────────────────────────────────────────────────────
def check_accuracy(model, le, y_series, model_name):
    y_enc = le.transform(y_series)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cv = cross_val_score(
        model, X_scaled, y_enc,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy"
    )

    print(f"\n{'─'*60}")
    print(f"  MODEL : {model_name}")
    print(f"{'─'*60}")
    print(f"  ✅ Test Accuracy       : {acc*100:.2f}%")
    print(f"  ✅ Precision           : {precision*100:.2f}%")
    print(f"  ✅ Recall              : {recall*100:.2f}%")
    print(f"  ✅ F1 Score            : {f1*100:.2f}%")
    print(f"  ✅ Cross-Val Accuracy  : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    print(f"  ✅ Best CV Score       : {cv.max()*100:.2f}%")
    print(f"  ✅ Worst CV Score      : {cv.min()*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_,
                                 zero_division=0))

    return y_test, y_pred, acc, f1, cv.mean()


# ─────────────────────────────────────────────────────
# STEP 3 — RUN ACCURACY CHECK ON ALL 4 MODELS
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DETAILED ACCURACY REPORT — ALL 4 MODELS")
print("=" * 60)

y_test_risk,   y_pred_risk,   acc_risk,   f1_risk,   cv_risk   = check_accuracy(m_risk,       le_risk,       df["risk_level"],        "RISK LEVEL")
y_test_stress, y_pred_stress, acc_stress, f1_stress, cv_stress = check_accuracy(m_stress,     le_stress,     df["stress_output"],      "STRESS LEVEL")
y_test_grade,  y_pred_grade,  acc_grade,  f1_grade,  cv_grade  = check_accuracy(m_grade,      le_grade,      df["grade_bucket"],       "GRADE REDUCTION")
y_test_comp,   y_pred_comp,   acc_comp,   f1_comp,   cv_comp   = check_accuracy(m_completion, le_completion, df["completion_bucket"],  "COMPLETION PROBABILITY")

# ─────────────────────────────────────────────────────
# STEP 4 — SUMMARY TABLE
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY TABLE")
print("=" * 60)
print(f"  {'Model':<28} {'Accuracy':>10} {'F1 Score':>10} {'Cross-Val':>10}")
print(f"  {'─'*56}")
print(f"  {'Risk Level':<28} {acc_risk*100:>9.2f}% {f1_risk*100:>9.2f}% {cv_risk*100:>9.2f}%")
print(f"  {'Stress Level':<28} {acc_stress*100:>9.2f}% {f1_stress*100:>9.2f}% {cv_stress*100:>9.2f}%")
print(f"  {'Grade Reduction':<28} {acc_grade*100:>9.2f}% {f1_grade*100:>9.2f}% {cv_grade*100:>9.2f}%")
print(f"  {'Completion Probability':<28} {acc_comp*100:>9.2f}% {f1_comp*100:>9.2f}% {cv_comp*100:>9.2f}%")
print(f"  {'─'*56}")
avg_acc = (acc_risk + acc_stress + acc_grade + acc_comp) / 4
avg_f1  = (f1_risk  + f1_stress  + f1_grade  + f1_comp)  / 4
avg_cv  = (cv_risk  + cv_stress  + cv_grade  + cv_comp)   / 4
print(f"  {'OVERALL AVERAGE':<28} {avg_acc*100:>9.2f}% {avg_f1*100:>9.2f}% {avg_cv*100:>9.2f}%")
print("=" * 60)

# ─────────────────────────────────────────────────────
# STEP 5 — CONFUSION MATRIX CHARTS
# ─────────────────────────────────────────────────────
print("\n⏳ Generating confusion matrix charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Confusion Matrices — All 4 Models", fontsize=15, fontweight="bold")

plots = [
    (y_test_risk,   y_pred_risk,   le_risk,       f"Risk Level  (Acc: {acc_risk*100:.1f}%)"),
    (y_test_stress, y_pred_stress, le_stress,     f"Stress Level  (Acc: {acc_stress*100:.1f}%)"),
    (y_test_grade,  y_pred_grade,  le_grade,      f"Grade Reduction  (Acc: {acc_grade*100:.1f}%)"),
    (y_test_comp,   y_pred_comp,   le_completion, f"Completion Prob  (Acc: {acc_comp*100:.1f}%)"),
]

for ax, (y_test, y_pred, le, title) in zip(axes.flat, plots):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("accuracy_report.png", dpi=150, bbox_inches="tight")
print("✅ Chart saved → accuracy_report.png")

# ─────────────────────────────────────────────────────
# STEP 6 — BAR CHART: MODEL COMPARISON
# ─────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 5))

models    = ["Risk Level", "Stress Level", "Grade Reduction", "Completion Prob"]
acc_vals  = [acc_risk*100, acc_stress*100, acc_grade*100, acc_comp*100]
f1_vals   = [f1_risk*100,  f1_stress*100,  f1_grade*100,  f1_comp*100]

x     = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, acc_vals, width, label="Accuracy",  color="#4C72B0")
bars2 = ax.bar(x + width/2, f1_vals,  width, label="F1 Score",  color="#55A868")

ax.set_title("Model Accuracy vs F1 Score Comparison", fontsize=13, fontweight="bold")
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend()
ax.axhline(y=80, color="red", linestyle="--", linewidth=1, label="80% threshold")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Chart saved → model_comparison.png")

print("\n" + "=" * 60)
print("  ACCURACY CHECK COMPLETE!")
print("=" * 60)
print("  Files saved:")
print("    📊 accuracy_report.png   ← Confusion matrices")
print("    📊 model_comparison.png  ← Accuracy vs F1 bar chart")
print("=" * 60)
