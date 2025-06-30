import os
import pandas as pd
import joblib
import json
import datetime
import requests
from recommenders.recommendation_engine import recommend_action

# ✅ Step 1: Ensure output folder exists
os.makedirs("output", exist_ok=True)

# ✅ Step 2: Fetch today's data from your API
fetch_url = "http://13.60.195.136:5050/bms-data"
response = requests.get(fetch_url)

if response.status_code != 200:
    print("Failed to fetch BMS data. Status code:", response.status_code)
    exit()

data = response.json()
df = pd.DataFrame(data)

# ✅ Step 3: Preprocess
df["Cell Balancing Status"] = df["Cell Balancing Status"].map({"Inactive": 0, "Active": 1})

# ✅ Step 4: Load trained ML models
clf = joblib.load("models/bms_status_classifier.pkl")
reg = joblib.load("models/failure_predictor.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

X = df[[
    "Total Pack Voltage", "Max Cell Voltage", "Min Cell Reading",
    "Max Cell Temp", "Average Cell Temp", "Internal Resistance",
    "Estimated Range", "Cell Balancing Status"
]]
X_scaled = scaler.transform(X)

# Step 5: Predict and recommend
df["Predicted Status"] = encoder.inverse_transform(clf.predict(X_scaled))
df["Predicted Failure Probability"] = reg.predict(X_scaled)
df["Recommendation"] = df.apply(recommend_action, axis=1)

# Step 6: Aggregate result
summary = {
    "date": datetime.date.today().isoformat(),
    "summary": df["Predicted Status"].value_counts().to_dict(),
    "avg_failure_prob": round(df["Predicted Failure Probability"].mean(), 3),
    "recommendation": df["Recommendation"].value_counts().idxmax()
}

# Step 7: Save locally
output_path = f"output/battery_summary_{summary['date']}.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"Battery summary saved to {output_path}")

# Step 8: Send to Android IVI App (API must be ready to receive JSON POST)
android_api_url = "http://13.60.195.136:8080/battery-summary"

try:
    response = requests.post(android_api_url, json=summary)
    if response.status_code == 200:
        print("Summary successfully sent to Android IVI app!")
    else:
        print(f"Failed to send summary. Status: {response.status_code}, Response: {response.text}")
except Exception as e:
    print("Error posting to Android app backend:", str(e))
