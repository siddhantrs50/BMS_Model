import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import json
from datetime import datetime
import numpy as np
import boto3
import os

# --- Database connection ---
db_url = "postgresql://postgres:Root22012003@bms-database.c94sysco8nc8.eu-north-1.rds.amazonaws.com:5432/bmsdb"
table_name = "bms"

# --- AWS S3 settings ---
bucket_name = "bms-model"
s3 = boto3.client('s3')

# --- Date range for today's records ---
today = datetime.now().date()
start_time = datetime.combine(today, datetime.min.time())
end_time = datetime.combine(today, datetime.max.time())

# --- Connect and fetch today's data ---
engine = create_engine(db_url)
query = f"""
SELECT * FROM {table_name}
WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
"""
df = pd.read_sql(query, engine)

# --- Analyze BMS status and estimate failure probability ---
if not df.empty:
    if "Cell Balancing Status" in df.columns:
        df["Cell Balancing Status"] = df["Cell Balancing Status"].map({"Inactive": 0, "Active": 1})
    
    df["failure_prob"] = df["BMS Status"].map({"Healthy": 0.1, "Warning": 0.5, "Critical": 0.9}) + np.random.normal(0, 0.05, len(df))
    df["failure_prob"] = df["failure_prob"].clip(0, 1)

    summary = {
        "date": str(today),
        "summary": df["BMS Status"].value_counts().to_dict(),
        "avg_failure_prob": round(df["failure_prob"].mean(), 3),
        "recommendation": (
            "Reduce aggressive driving and monitor cell temps closely."
            if df["BMS Status"].eq("Warning").sum() > 20
            else "Battery healthy. No action needed."
        )
    }
else:
    summary = {
        "date": str(today),
        "summary": {},
        "avg_failure_prob": None,
        "recommendation": "No data available for the day."
    }

# --- Save JSON locally ---
summary_json = json.dumps(summary, indent=4)
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_filename = f"battery_summary_{today}.json"
output_path = os.path.join(output_folder, output_filename)

with open(output_path, "w") as f:
    f.write(summary_json)

print(f"Summary saved to: {output_path}")

# --- Upload to S3 ---
s3 = boto3.client("s3")

try:
    s3.upload_file(output_path, bucket_name, f"bms_summaries/{output_filename}")
    print(f" Uploaded to S3 bucket '{bucket_name}' as 'bms_summaries/{output_filename}'")
except Exception as e:
    print(f" S3 Upload failed: {str(e)}")
