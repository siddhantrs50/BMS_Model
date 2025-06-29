import pandas as pd
import joblib
import json
import datetime
from recommenders.recommendation_engine import recommend_action
import os
os.makedirs("output", exist_ok=True)


# Load trained models and preprocessing tools
clf = joblib.load("models/bms_status_classifier.pkl")
reg = joblib.load("models/failure_predictor.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Read TCU log and assign correct headers (even if file has no header row)
df = pd.read_csv("data/tcu_logs_today.csv", header=None)
df.columns = [
    "Total Pack Voltage", "Max Cell Voltage", "Min Cell Reading",
    "Max Cell Temp", "Average Cell Temp", "Internal Resistance",
    "Estimated Range", "Cell Balancing Status"
]

# Convert categorical to numeric
df["Cell Balancing Status"] = df["Cell Balancing Status"].map({"Inactive": 0, "Active": 1})

# Extract features
X = df[[
    "Total Pack Voltage", "Max Cell Voltage", "Min Cell Reading",
    "Max Cell Temp", "Average Cell Temp", "Internal Resistance",
    "Estimated Range", "Cell Balancing Status"
]]
X_scaled = scaler.transform(X)

# Run predictions
df["Predicted Status"] = encoder.inverse_transform(clf.predict(X_scaled))
df["Predicted Failure Probability"] = reg.predict(X_scaled)

# Assign recommendations
df["Recommendation"] = df.apply(recommend_action, axis=1)

# Summarize
summary = {
    "date": datetime.date.today().isoformat(),
    "summary": df["Predicted Status"].value_counts().to_dict(),
    "avg_failure_prob": round(df["Predicted Failure Probability"].mean(), 3),
    "recommendation": df["Recommendation"].value_counts().idxmax()
}

# Save output JSON
output_path = f"output/battery_summary_{summary['date']}.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"Daily summary saved to {output_path}")
