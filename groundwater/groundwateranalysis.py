import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the pre-merged dataset
merged = pd.read_csv(r'<path_to_dataset>\groundwater_dataset.csv')

# Step 1: Calculate annual rainfall
merged['Annual Rainfall'] = merged[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].sum(axis=1)

# Step 2: Calculate groundwater recharge for each state (using a simple formula)
merged['Groundwater Recharge'] = merged['Annual Rainfall'] * (1 - merged['Avg R'] / 10000)

# Display the results
print(merged[['STATE', 'Annual Rainfall', 'Avg R', 'Groundwater Recharge']])

# Visualization: Annual Rainfall vs Groundwater Recharge
sns.set(style="whitegrid")

# 1. Annual Rainfall vs Groundwater Recharge
plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged, x='Annual Rainfall', y='Groundwater Recharge', hue='STATE')
plt.title('Annual Rainfall vs Groundwater Recharge')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Groundwater Recharge')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Avg R vs Groundwater Recharge
plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged, x='Avg R', y='Groundwater Recharge', hue='STATE')
plt.title('Average R Factor vs Groundwater Recharge')
plt.xlabel('Average R Factor')
plt.ylabel('Groundwater Recharge')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3. Combined Plot: Use size for one variable and color for another
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=merged,
    x='Annual Rainfall',
    y='Groundwater Recharge',
    size='Avg R',
    hue='STATE',
    sizes=(40, 400),
    palette='tab20'
)
plt.title('Rainfall, R Factor & Recharge (Bubble Plot)')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Groundwater Recharge')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Correlation Matrix
correlation = merged[['Annual Rainfall', 'Avg R', 'Groundwater Recharge']].corr()
print(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Train a linear regression model
X = merged[['Annual Rainfall']]
y = merged['Groundwater Recharge']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
import numpy as np
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# KMeans Clustering
# Select features for clustering
cluster_data = merged[['Annual Rainfall', 'Avg R', 'Groundwater Recharge']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters (2D plot using first two features)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged, x='Annual Rainfall', y='Groundwater Recharge', hue='Cluster', palette='Set2', s=100)
plt.title('KMeans Clustering of States')
plt.xlabel('Annual Rainfall')
plt.ylabel('Groundwater Recharge')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Optional: View which state is in which cluster
print(merged[['STATE', 'Annual Rainfall', 'Avg R', 'Groundwater Recharge', 'Cluster']].sort_values(by='Cluster'))

# Summarize each cluster
cluster_summary = merged.groupby('Cluster')[['Annual Rainfall', 'Avg R', 'Groundwater Recharge']].mean()
print("Cluster Summary:")
print(cluster_summary)

# Save the final dataset
merged.to_csv("groundwater_dataset.csv", index=False)
