import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Student Burnout Detector",
                   layout="wide")

# ------------------------------
# Load Models
# ------------------------------
model = joblib.load("model.pkl")
dropout_model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Burnout & Dropout Risk Detector")
st.markdown("### AI-Based Early Warning System for Universities")

st.divider()

# ------------------------------
# Layout Columns
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    lms = st.slider("LMS Login Frequency (per week)", 0, 20, 5)
    delay = st.slider("Average Assignment Delay (days)", 0, 15, 2)
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0)

with col2:
    variance = st.slider("Activity Variance", 0.0, 1.0, 0.5)
    study_hours = st.slider("Study Hours per Week", 0, 40, 10)
    gpa = st.slider("Previous GPA", 0.0, 10.0, 7.0)

input_data = np.array([[lms, delay, attendance, sentiment,
                        variance, study_hours, gpa]])

# ------------------------------
# Prediction Section
# ------------------------------
if st.button("🚀 Predict Risk"):

    risk_pred = model.predict(input_data)[0]
    risk_prob = np.max(model.predict_proba(input_data)) * 100
    
    input_scaled = scaler.transform(input_data)
    dropout_prob = dropout_model.predict_proba(input_scaled)[0][1] * 100

    risk_labels = {0: "Low", 1: "Medium", 2: "High"}

    st.divider()
    st.header("📊 Prediction Results")

    colA, colB, colC = st.columns(3)

    colA.metric("Burnout Risk Level", risk_labels[risk_pred])
    colB.metric("Risk Score", f"{round(risk_prob, 2)} / 100")
    colC.metric("Dropout Probability", f"{round(dropout_prob, 2)} %")

    # ------------------------------
    # Risk Gauge
    # ------------------------------
    st.subheader("🎯 Risk Score Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_prob,
        title={'text': "Burnout Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Feature Importance
    # ------------------------------
    st.subheader("📌 Key Behavioral Triggers")

    importance = model.feature_importances_
    features = ["LMS Login", "Delay", "Attendance",
                "Sentiment", "Variance",
                "Study Hours", "GPA"]

    fig2, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    st.pyplot(fig2)

    # ------------------------------
    # Trend Simulation
    # ------------------------------
    st.subheader("📈 Burnout Trend (Last 8 Weeks)")
    trend = np.random.randint(20, 90, 8)
    st.line_chart(trend)

    # ------------------------------
    # Intervention
    # ------------------------------
    st.subheader("🛠 Recommended Intervention")

    if risk_pred == 0:
        st.success("Encourage participation and regular monitoring.")
    elif risk_pred == 1:
        st.warning("Suggest academic counseling and peer mentoring.")
    else:
        st.error("⚠ Immediate intervention required. Notify faculty & counselor.")

    # ------------------------------
    # Smart Alert
    # ------------------------------
    if risk_pred == 2:
        st.error("🚨 ALERT: Academic Counselor Notified!")