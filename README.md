Early Detection of Student Burnout & Dropout Risk

AI-Based Behavioral Analytics System

Project Overview

This project is an AI-driven early warning system designed to detect student burnout and predict dropout risk at an early stage using behavioral analytics and machine learning. Universities often identify academic issues only after performance drops significantly. By that time, intervention becomes difficult. This system proactively analyzes behavioral indicators such as LMS usage, assignment delays, attendance patterns, sentiment scores, and study habits to generate predictive insights.

The system provides:

Burnout Risk Level (Low / Medium / High)

Risk Score (0–100)

Dropout Probability (%)

Key Behavioral Triggers

Recommended Intervention Strategy

Interactive Streamlit Dashboard

Tech Stack

Python 3.11

Scikit-learn

Streamlit

Pandas

NumPy

Matplotlib

Plotly

Joblib

Project Structure

student-burnout-detector/

data_generator.py
train_model.py
app.py
requirements.txt
student_data.csv
model.pkl
dropout_model.pkl
scaler.pkl
README.md

How to Run the Project (Step-by-Step Procedure)

Step 1: Install Python
Install Python 3.11 (Recommended for ML compatibility).
Check version using:
python --version

Step 2: Open Project Folder
Navigate to the project directory in terminal:

cd student-burnout-detector

Step 3: Create Virtual Environment (Recommended)

python -m venv venv

Activate virtual environment:

For Windows:
venv\Scripts\activate

For Mac/Linux:
source venv/bin/activate

You should now see (venv) in your terminal.

Step 4: Install Dependencies

pip install -r requirements.txt

If requirements.txt is unavailable, run:

pip install numpy pandas scikit-learn streamlit matplotlib plotly joblib

Step 5: Generate Dataset

Run:

python data_generator.py

This creates a synthetic dataset named student_data.csv.

You should see:
Dataset Generated Successfully!

Step 6: Train Machine Learning Models

Run:

python train_model.py

This will:

Train Random Forest model for Burnout Risk

Train Logistic Regression model for Dropout Probability

Save model.pkl

Save dropout_model.pkl

Save scaler.pkl

You should see:
Models Trained and Saved!

Step 7: Run the Streamlit Application

Run:

streamlit run app.py

If Streamlit command does not work, run:

python -m streamlit run app.py

The browser will open automatically and display the interactive dashboard.

How the System Works

User inputs student behavioral parameters through sliders.

Random Forest predicts Burnout Risk Level.

Logistic Regression calculates Dropout Probability.

Risk Score is derived from prediction confidence.

Feature Importance identifies key behavioral triggers.

Dashboard displays:

Risk Classification

Risk Score Gauge

Dropout Probability

Behavioral Insights

Burnout Trend Graph

Intervention Recommendations

Smart Alerts (for high risk)

Machine Learning Models Used

Random Forest Classifier
Used for multi-class burnout risk classification (Low / Medium / High).
Provides feature importance for explainability.

Logistic Regression
Used for binary dropout prediction.
Provides probability-based output.

Risk Classification Criteria

Risk Score 0–30 → Low Risk
Risk Score 31–60 → Medium Risk
Risk Score 61–100 → High Risk

Dashboard Features

Interactive input sliders

Burnout Risk Level display

Risk Score Gauge visualization

Dropout Probability percentage

Feature Importance chart

Burnout Trend simulation

Recommended Intervention Strategy

Smart alert system for high-risk students

Behavioral Insights Identified

Low LMS login frequency strongly correlates with burnout risk.

High assignment delays indicate disengagement.

Attendance below threshold predicts higher dropout probability.

Negative sentiment score signals academic stress.

Reduced study hours reflect burnout trends.

Practical Impact

Enables early intervention before academic decline.

Reduces student dropout rates.

Supports mental health monitoring.

Helps faculty make data-driven decisions.

Improves institutional retention metrics.

Future Enhancements

Integration with real LMS databases.

Time-series prediction using LSTM.

SHAP-based Explainable AI.

Automated email/SMS alert system.

Cloud deployment for institutional use.

Multi-user admin dashboard.

Troubleshooting

If model files are missing:
Run python train_model.py again.

If modules are missing:
Run pip install -r requirements.txt.

If Streamlit is not recognized:
Run python -m streamlit run app.py.

Demo Flow (For Presentation)

Enter student behavioral data.

Click “Predict Risk”.

Display Burnout Risk Level.

Show Risk Score and Dropout Probability.

Explain behavioral triggers.

Display recommended intervention.

Developed as a Hackathon Project
Integrated M.Tech CSE (Business Analytics)
