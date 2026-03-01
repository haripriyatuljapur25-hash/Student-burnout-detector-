import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("student_data.csv")

X = data.drop(["Burnout_Risk", "Dropout"], axis=1)
y_risk = data["Burnout_Risk"]
y_dropout = data["Dropout"]

# Split
X_train, X_test, y_train_risk, y_test_risk = train_test_split(
    X, y_risk, test_size=0.2, random_state=42
)

# Train Burnout Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_risk)

# Train Dropout Model
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

log_model = LogisticRegression()
log_model.fit(X_scaled, y_dropout)

# Save models
joblib.dump(rf_model, "model.pkl")
joblib.dump(log_model, "dropout_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models Trained and Saved!")