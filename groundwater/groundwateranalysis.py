import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Global variable to store the merged data
merged = None

def initialize_groundwater_data():
    """Initialize and process groundwater data"""
    global merged
    merged = pd.read_csv('groundwater/groundwater_dataset.csv')
    
    # Calculate annual rainfall and groundwater recharge
    merged['Annual Rainfall'] = merged[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                      'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].sum(axis=1)
    merged['Groundwater Recharge'] = merged['Annual Rainfall'] * (1 - merged['Avg R'] / 10000)
    
    # Perform clustering
    cluster_data = merged[['Annual Rainfall', 'Avg R', 'Groundwater Recharge']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    merged['Cluster'] = kmeans.fit_predict(scaled_data)
    
    return merged

def get_risk_level(groundwater_recharge):
    """Determine risk level based on groundwater recharge"""
    if groundwater_recharge < 500:
        return "High Risk", 90, "Critical groundwater depletion risk. Immediate conservation needed."
    elif groundwater_recharge < 1000:
        return "Moderate Risk", 60, "Moderate groundwater stress. Conservation measures recommended."
    else:
        return "Low Risk", 30, "Healthy groundwater levels. Continue sustainable practices."

def analyze_groundwater_risk(state_code):
    """Analyze groundwater risk for a given state"""
    try:
        if merged is None:
            initialize_groundwater_data()
            
        # Get state data
        state_data = merged[merged['STATE'] == state_code]
        
        if state_data.empty:
            raise ValueError(f"No data found for state code: {state_code}")
            
        # Get values for the state
        annual_rainfall = state_data['Annual Rainfall'].iloc[0]
        groundwater_recharge = state_data['Groundwater Recharge'].iloc[0]
        cluster = state_data['Cluster'].iloc[0]
        
        # Get risk assessment
        risk_level, risk_percentage, summary = get_risk_level(groundwater_recharge)
        
        return {
            'risk_level': risk_level,
            'risk_percentage': risk_percentage,
            'annual_rainfall': round(annual_rainfall, 2),
            'recharge_rate': round(groundwater_recharge, 2),
            'cluster': int(cluster),
            'summary': summary
        }
        
    except Exception as e:
        return {
            'risk_level': "Unknown",
            'risk_percentage': 0,
            'annual_rainfall': 0,
            'recharge_rate': 0,
            'cluster': -1,
            'summary': f"Unable to analyze groundwater risk: {str(e)}"
        }

def create_visualizations():
    """Create and display all visualizations"""
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

    # 3. Correlation Heatmap
    correlation = merged[['Annual Rainfall', 'Avg R', 'Groundwater Recharge']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # 4. Cluster Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged, x='Annual Rainfall', y='Groundwater Recharge', 
                   hue='Cluster', palette='Set2', s=100)
    plt.title('KMeans Clustering of States')
    plt.xlabel('Annual Rainfall')
    plt.ylabel('Groundwater Recharge')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

def train_model():
    """Train and evaluate the linear regression model"""
    X = merged[['Annual Rainfall']]
    y = merged['Groundwater Recharge']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse

if __name__ == "__main__":
    # Initialize data
    merged = initialize_groundwater_data()
    
    # Display initial results
    print("\nInitial Data Summary:")
    print(merged[['STATE', 'Annual Rainfall', 'Avg R', 'Groundwater Recharge']])
    
    # Train and evaluate model
    model, r2, rmse = train_model()
    print(f"\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Create visualizations
    create_visualizations()
    
    # Display cluster summary
    cluster_summary = merged.groupby('Cluster')[['Annual Rainfall', 'Avg R', 'Groundwater Recharge']].mean()
    print("\nCluster Summary:")
    print(cluster_summary)
    
    # Save the processed dataset
    merged.to_csv("groundwater/groundwater_dataset.csv", index=False)