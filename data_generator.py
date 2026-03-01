import numpy as np
import pandas as pd

np.random.seed(42)
n_students = 1000

data = pd.DataFrame({
    "LMS_Login_Frequency": np.random.randint(1, 15, n_students),
    "Avg_Submission_Delay": np.random.randint(0, 10, n_students),
    "Attendance_Percentage": np.random.randint(40, 100, n_students),
    "Sentiment_Score": np.random.uniform(-1, 1, n_students),
    "Activity_Variance": np.random.uniform(0, 1, n_students),
    "Study_Hours": np.random.randint(1, 30, n_students),
    "Previous_GPA": np.random.uniform(5, 10, n_students)
})

def generate_risk(row):
    score = 0
    
    if row["LMS_Login_Frequency"] < 5:
        score += 2
    if row["Avg_Submission_Delay"] > 5:
        score += 2
    if row["Attendance_Percentage"] < 60:
        score += 2
    if row["Sentiment_Score"] < -0.3:
        score += 1
    if row["Study_Hours"] < 5:
        score += 1

    if score <= 2:
        return 0
    elif score <= 5:
        return 1
    else:
        return 2

data["Burnout_Risk"] = data.apply(generate_risk, axis=1)

data["Dropout"] = data["Burnout_Risk"].apply(
    lambda x: 1 if x == 2 else 0
)

data.to_csv("student_data.csv", index=False)

print("Dataset Generated Successfully!")