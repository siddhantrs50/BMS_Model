import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate 1440 rows (simulating 1-min interval over 24 hours)
n_rows = 1440

def generate_row():
    return {
        "Total Pack Voltage": round(np.random.uniform(320, 370), 2),
        "Max Cell Voltage": round(np.random.uniform(4.0, 4.2), 3),
        "Min Cell Reading": round(np.random.uniform(3.2, 3.6), 3),
        "Max Cell Temp": round(np.random.uniform(50, 70), 1),
        "Average Cell Temp": round(np.random.uniform(45, 65), 1),
        "Internal Resistance": round(np.random.uniform(0.0025, 0.005), 5),
        "Estimated Range": round(np.random.uniform(180, 250), 1),
        "Cell Balancing Status": random.choice(["Active", "Inactive"])
    }

# Create DataFrame
df = pd.DataFrame([generate_row() for _ in range(n_rows)])

# Save without header (as expected by your ML scripts)
df.to_csv("data/tcu_logs_today.csv", index=False, header=False)
print("tcu_logs_today.csv generated with 1440 records.")
