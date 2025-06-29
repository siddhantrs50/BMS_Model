import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from recommenders.recommendation_engine import recommend_action

data = pd.read_csv('data/battery_dataset.csv')

data['Cell Balancing Status'] = data['Cell Balancing Status'].map({'Inactive': 0, 'Active': 1})
label_encoder = LabelEncoder()
data['BMS Status Encoded'] = label_encoder.fit_transform(data['BMS Status'])

X = data[['Total Pack Voltage', 'Max Cell Voltage', 'Min Cell Reading',
          'Max Cell Temp', 'Average Cell Temp', 'Internal Resistance',
          'Estimated Range', 'Cell Balancing Status']]
y = data['BMS Status Encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier()
clf.fit(X_scaled, y)
joblib.dump(clf, 'models/bms_status_classifier.pkl')

data['FailureProbability'] = data['BMS Status'].map({'Healthy': 0.1, 'Warning': 0.5, 'Critical': 0.9})
reg = LinearRegression()
reg.fit(X_scaled, data['FailureProbability'])
joblib.dump(reg, 'models/failure_predictor.pkl')

joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

predicted_status_encoded = clf.predict(X_scaled)
predicted_status = label_encoder.inverse_transform(predicted_status_encoded)
predicted_failure_prob = reg.predict(X_scaled)

data['Predicted Status'] = predicted_status
data['Predicted Failure Probability'] = predicted_failure_prob
data['Recommendation'] = data.apply(recommend_action, axis=1)

output = data[['Total Pack Voltage', 'Max Cell Temp', 'Internal Resistance',
               'Predicted Status', 'Predicted Failure Probability', 'Recommendation']]

output.to_json('output_for_android.json', orient='records', indent=4)
print(" Output saved to output_for_android.json")
