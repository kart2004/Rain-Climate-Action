import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/avg_ke.csv")

# Features and target
X = df[['State', 'Year']]
y = df['Average_KE_MJ_per_ha_mm']

# Preprocessing: One-hot encode the State column
categorical_features = ['State']
numerical_features = ['Year']

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Create a pipeline with Random Forest Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 🔍 Prediction function
def predict_ke(state, year):
    input_df = pd.DataFrame({'State': [state], 'Year': [year]})
    prediction = model.predict(input_df)[0]
    return round(prediction, 3)

def predict_r_factor(state, year, precipitation_mm):
    ke = predict_ke(state, year)  # Predict kinetic energy
    r = ke * precipitation_mm     # R-factor = KE × Precipitation
    return round(r, 2)

