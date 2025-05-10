import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'flood_past.csv')
df = pd.read_csv(data_path)

# Drop rows with missing critical values
df = df.dropna(subset=['PRECIPITATION', 'SEVERITY', 'TERRAIN', 'CLIMATE', 'QUARTER'])

# Encode categorical features
le_terrain = LabelEncoder()
le_climate = LabelEncoder()
le_quarter = LabelEncoder()
le_severity = LabelEncoder()

df['TERRAIN_ENC'] = le_terrain.fit_transform(df['TERRAIN'])
df['CLIMATE_ENC'] = le_climate.fit_transform(df['CLIMATE'])
df['QUARTER_ENC'] = le_quarter.fit_transform(df['QUARTER'])
df['LABEL'] = le_severity.fit_transform(df['SEVERITY'])

# Feature matrix and target
X = df[['PRECIPITATION', 'SEVERITY1', 'TERRAIN_ENC', 'CLIMATE_ENC', 'QUARTER_ENC']]
y = df['LABEL']

# Train-test split (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Optional: evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in le_severity.classes_]))

# Define model save path outside current folder
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
os.makedirs(model_dir, exist_ok=True)

# Save model and encoders
joblib.dump(model, os.path.join(model_dir, "xgb_drought_model.pkl"))
joblib.dump({
    'TERRAIN': le_terrain,
    'CLIMATE': le_climate,
    'QUARTER': le_quarter,
    'SEVERITY': le_severity
}, os.path.join(model_dir, "xgb_drought_encoders.pkl"))

print("✅ Model and encoders saved in the '../model/' folder.")
