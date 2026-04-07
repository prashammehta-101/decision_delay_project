"""
Dataset Generator: Student Task Delay Prediction
Generates 10,000 unique rows for ML training.

Features:
    delay_days, days_until_deadline, task_name, task_type,
    current_cgpa, task_weight, difficulty_level, progress_percent,
    required_hours, remaining_days, stress_level, past_delay_history,
    dependency_count, weightage_marks, risk_score

Target:
    will_delay (0 = No, 1 = Yes)

Requirements:
    pip install pandas numpy faker
"""

import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TASK_TYPES = ["assignment", "lab_experiment", "exam", "project"]

TASK_TEMPLATES = {
    "assignment": [
        "Data Structures Assignment", "OS Assignment", "DBMS Assignment",
        "Networking Assignment", "Algorithms Assignment", "Math Assignment",
        "Software Engineering Assignment", "Web Dev Assignment",
        "Machine Learning Assignment", "Computer Graphics Assignment",
    ],
    "lab_experiment": [
        "OS Lab - Process Scheduling", "DBMS Lab - SQL Queries",
        "Network Lab - Packet Analysis", "DS Lab - Linked List",
        "AI Lab - Search Algorithms", "Compiler Lab - Lexical Analysis",
        "Embedded Systems Lab", "IOT Lab Experiment",
        "Digital Electronics Lab", "Computer Vision Lab",
    ],
    "exam": [
        "Mid-Semester Exam", "End-Semester Exam", "Unit Test",
        "Internal Assessment", "Viva Voce", "Practical Exam",
        "Online Quiz", "Weekly Test", "Module Exam", "Surprise Test",
    ],
    "project": [
        "Final Year Project", "Mini Project", "Research Project",
        "Group Project - Web App", "ML Model Project", "IoT Based Project",
        "Mobile App Development", "Capstone Project",
        "Open Source Contribution", "Industry Internship Project",
    ],
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard", "very_hard"]
STRESS_LEVELS     = ["low", "medium", "high", "critical"]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def pick_task():
    task_type = random.choice(TASK_TYPES)
    base_name = random.choice(TASK_TEMPLATES[task_type])
    # Add slight variation so names feel unique
    suffix = random.choice(["", f" #{random.randint(1,5)}", f" (Sem {random.randint(1,8)})"])
    return task_type, base_name + suffix


def compute_risk_score(row: dict) -> float:
    """
    Deterministic risk score (0–100) based on domain logic.
    Higher = more likely to delay.
    """
    score = 0.0

    # Low progress + high required hours
    score += (1 - row["progress_percent"] / 100) * 20

    # Proximity to deadline
    if row["remaining_days"] <= 1:
        score += 25
    elif row["remaining_days"] <= 3:
        score += 15
    elif row["remaining_days"] <= 7:
        score += 8

    # Difficulty
    diff_map = {"easy": 0, "medium": 5, "hard": 12, "very_hard": 20}
    score += diff_map[row["difficulty_level"]]

    # Stress
    stress_map = {"low": 0, "medium": 5, "high": 12, "critical": 20}
    score += stress_map[row["stress_level"]]

    # Past delays
    score += min(row["past_delay_history"] * 3, 15)

    # Low CGPA → less likely to manage time
    if row["current_cgpa"] < 6.0:
        score += 10
    elif row["current_cgpa"] < 7.5:
        score += 5

    # Dependencies
    score += min(row["dependency_count"] * 2, 10)

    # Task weight
    score += row["task_weight"] * 0.1

    # Clamp
    return round(min(max(score, 0), 100), 2)


def compute_will_delay(risk_score: float, noise: float = 0.05) -> int:
    """
    Label: will_delay = 1 if task is likely to be delayed.
    Threshold is 45 with small random noise for realism.
    """
    threshold = 45 + np.random.normal(0, 5)
    return int(risk_score > threshold)


# ─────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────

def generate_dataset(n: int = 10_000) -> pd.DataFrame:
    rows = []

    for _ in range(n):
        task_type, task_name = pick_task()

        remaining_days      = random.randint(0, 30)
        days_until_deadline = remaining_days + random.randint(0, 5)  # slight variation
        required_hours      = round(random.uniform(1, 80), 1)
        progress_percent    = random.randint(0, 100)
        current_cgpa        = round(random.uniform(4.0, 10.0), 2)
        task_weight         = random.randint(5, 50)          # marks/weight out of total
        weightage_marks     = random.randint(10, 100)
        difficulty_level    = random.choices(
            DIFFICULTY_LEVELS, weights=[20, 35, 30, 15]
        )[0]
        stress_level        = random.choices(
            STRESS_LEVELS, weights=[20, 35, 30, 15]
        )[0]
        past_delay_history  = random.randint(0, 10)          # number of past delays
        dependency_count    = random.randint(0, 6)           # tasks this depends on

        row = dict(
            task_name           = task_name,
            task_type           = task_type,
            days_until_deadline = days_until_deadline,
            remaining_days      = remaining_days,
            required_hours      = required_hours,
            progress_percent    = progress_percent,
            current_cgpa        = current_cgpa,
            task_weight         = task_weight,
            weightage_marks     = weightage_marks,
            difficulty_level    = difficulty_level,
            stress_level        = stress_level,
            past_delay_history  = past_delay_history,
            dependency_count    = dependency_count,
        )

        risk_score  = compute_risk_score(row)
        will_delay  = compute_will_delay(risk_score)

        # delay_days: if will_delay=1, a positive delay; else 0 or slightly negative (early)
        if will_delay:
            delay_days = random.randint(1, 14)
        else:
            delay_days = random.randint(-3, 0)   # negative = submitted early

        row["risk_score"]  = risk_score
        row["delay_days"]  = delay_days
        row["will_delay"]  = will_delay

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    cols = [
        "task_name", "task_type",
        "days_until_deadline", "remaining_days",
        "required_hours", "progress_percent",
        "current_cgpa", "task_weight", "weightage_marks",
        "difficulty_level", "stress_level",
        "past_delay_history", "dependency_count",
        "risk_score", "delay_days",
        "will_delay",   # ← target label
    ]
    return df[cols]


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 10,000 rows...")
    df = generate_dataset(10_000)

    output_file = "student_task_delay_dataset.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Dataset saved → {output_file}")
    print(f"Shape          : {df.shape}")
    print(f"\nClass balance (will_delay):\n{df['will_delay'].value_counts()}")
    print(f"\nSample rows:")
    print(df.head(5).to_string(index=False))
