import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    log_loss, cohen_kappa_score
)
import tensorflow as tf
import os
import sys

# Import your config module
try:
    from flood.config import STATE_MAPPING, MONTH_MAPPING, labelencoder_X_1, onehotencoder, sc_X
except ImportError:
    print("Warning: Could not import from flood.config, will create placeholders.")
    # Create placeholder objects if import fails
    labelencoder_X_1 = None
    onehotencoder = None
    sc_X = None

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

def process_and_split_data(data_path='data/flood_past.csv'):
    """Process flood data and split into train/test sets"""
    print("Loading dataset...")
    try:
        dataset = pd.read_csv(data_path)
        print(f"Dataset loaded successfully with shape: {dataset.shape}")
        
        # Filter dataset
        dataset = dataset[dataset['YEAR'] > 1980].dropna()
        print(f"Dataset after filtering: {dataset.shape}")
        
        # Extract features and target
        X = dataset[['SUBDIVISION', 'TERRAIN', 'PRECIPITATION']].copy()
        y = dataset['SEVERITY'].copy()
        
        print("Creating encoders...")
        # Handle state encoding
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        # Create new encoders if needed
        state_encoder = LabelEncoder()
        X_state_encoded = state_encoder.fit_transform(X['SUBDIVISION'])
        
        # Encode terrain with a fresh encoder
        terrain_encoder = LabelEncoder()
        X_terrain_encoded = terrain_encoder.fit_transform(X['TERRAIN'])
        
        # Create one hot encoders
        state_ohe = OneHotEncoder(sparse_output=False)
        terrain_ohe = OneHotEncoder(sparse_output=False)
        
        # Apply one-hot encoding
        state_onehot = state_ohe.fit_transform(X_state_encoded.reshape(-1, 1))
        terrain_onehot = terrain_ohe.fit_transform(X_terrain_encoded.reshape(-1, 1))
        
        # Combine features
        X_processed = np.hstack([
            state_onehot, 
            terrain_onehot, 
            X['PRECIPITATION'].values.reshape(-1, 1)
        ])
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y.values, test_size=0.2, random_state=42
        )
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, dataset
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def load_or_create_model(X_train, y_train, input_shape):
    """Load existing model or create a new one if loading fails"""
    try:
        print("Attempting to load model...")
        # Try to import model initialization function
        sys.path.append(os.path.abspath("./"))
        from flood.load import init
        model, graph = init()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a simple model instead...")
        
        # Create a simple model for evaluation purposes
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.utils import to_categorical
        
        # Convert target to one-hot
        num_classes = int(max(y_train)) + 1
        y_train_onehot = to_categorical(y_train, num_classes)
        
        # Create model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        print("Training the model...")
        model.fit(
            X_train, y_train_onehot,
            epochs=10,  # Reduced for testing
            batch_size=32,
            verbose=1
        )
        
        return model

def evaluate_flood_model(model, X_test, y_test):
    """Comprehensive evaluation of flood prediction model"""
    # Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Define severity levels for reporting
    class_names = [f"Severity {i}" for i in range(6)]  # 0-5 severity levels
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Handle potential value errors for metrics that require certain conditions
    try:
        metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"Warning: Error calculating precision/recall metrics: {e}")
        metrics['precision_weighted'] = 0
        metrics['recall_weighted'] = 0
        metrics['f1_weighted'] = 0
        metrics['precision_macro'] = 0
        metrics['recall_macro'] = 0
        metrics['f1_macro'] = 0
    
    # Classification report
    try:
        metrics['class_report'] = classification_report(y_test, y_pred, 
                                                    target_names=class_names, 
                                                    output_dict=True,
                                                    zero_division=0)
    except Exception as e:
        print(f"Warning: Error generating classification report: {e}")
        metrics['class_report'] = {}
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(6))
    metrics['confusion_matrix'] = cm
    
    # Calculate class-specific metrics
    try:
        from sklearn.preprocessing import label_binarize
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=range(6))
        
        # Calculate AUC for each class
        roc_auc = {}
        pr_auc = {}
        for i in range(6):  # For each severity level
            if i < y_pred_probs.shape[1]:  # Check if this class exists in predictions
                # Only calculate if this class exists in the test set
                if np.sum(y_test_bin[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
                    roc_auc[class_names[i]] = auc(fpr, tpr)
                    
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
                    pr_auc[class_names[i]] = auc(recall, precision)
                else:
                    roc_auc[class_names[i]] = 0
                    pr_auc[class_names[i]] = 0
        
        metrics['roc_auc'] = roc_auc
        metrics['pr_auc'] = pr_auc
        
        # Calculate average AUCs
        if roc_auc:
            metrics['roc_auc_macro'] = np.mean(list(roc_auc.values()))
        else:
            metrics['roc_auc_macro'] = 0
            
        if pr_auc:
            metrics['pr_auc_macro'] = np.mean(list(pr_auc.values()))
        else:
            metrics['pr_auc_macro'] = 0
            
    except Exception as e:
        print(f"Warning: Error calculating AUC metrics: {e}")
        metrics['roc_auc'] = {}
        metrics['pr_auc'] = {}
        metrics['roc_auc_macro'] = 0
        metrics['pr_auc_macro'] = 0
    
    # Calculate specialized metrics for flood prediction
    
    # Ordinal accuracy (within +/- 1 level)
    metrics['ordinal_accuracy'] = np.mean(np.abs(y_test - y_pred) <= 1)
    
    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
    
    # Calculate class distribution
    metrics['class_distribution_actual'] = np.bincount(y_test, minlength=6) / len(y_test)
    metrics['class_distribution_pred'] = np.bincount(y_pred, minlength=6) / len(y_pred)
    
    # Cost-weighted metrics - create cost matrix with higher costs for under-prediction
    cost_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i > j:  # Under-prediction (actual > predicted)
                cost_matrix[i, j] = (i - j) ** 2  # Quadratic cost
            elif i < j:  # Over-prediction (actual < predicted)
                cost_matrix[i, j] = 0.5 * (j - i)  # Linear cost
    
    # Calculate total weighted error
    total_cost = 0
    for i in range(len(y_test)):
        total_cost += cost_matrix[y_test[i], y_pred[i]]
    metrics['weighted_error'] = total_cost / len(y_test)
    
    # High severity recall (detecting severe floods correctly)
    high_severity_idx = y_test >= 3  # Severity 3 and above
    if np.any(high_severity_idx):
        high_y_test = y_test[high_severity_idx]
        high_y_pred = y_pred[high_severity_idx]
        metrics['high_severity_recall'] = np.mean(high_y_pred >= 3)
    else:
        metrics['high_severity_recall'] = np.nan
    
    # Calculate quadratic weighted kappa
    metrics['quadratic_weighted_kappa'] = calculate_qwk(y_test, y_pred, 6)
    
    return metrics, y_pred, y_pred_probs

def calculate_qwk(y_true, y_pred, num_classes):
    """Calculate Quadratic Weighted Kappa"""
    # Create weight matrix for quadratic weighting
    w_mat = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w_mat[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Calculate expected matrix (outer product of marginals)
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)
    expected = np.outer(row_sum, col_sum) / np.sum(row_sum)
    
    # Calculate kappa
    numerator = np.sum(w_mat * cm)
    denominator = np.sum(w_mat * expected)
    
    # Check for division by zero
    if denominator == 0:
        return 0
    
    return 1.0 - (numerator / denominator)

def plot_evaluation_results(metrics, save_path='results'):
    """Plot evaluation results and save to specified path"""
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 1. Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        
        # Only normalize if the confusion matrix is not empty
        if np.sum(cm) > 0:
            cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-10)
            cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with 0
        else:
            cm_norm = cm
            
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix\n(annotations show counts, colors show percentages)')
        plt.ylabel('True Severity')
        plt.xlabel('Predicted Severity')
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot class distribution
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics['class_distribution_actual']))
        width = 0.35
        
        plt.bar(x - width/2, metrics['class_distribution_actual'], width, label='Actual')
        plt.bar(x + width/2, metrics['class_distribution_pred'], width, label='Predicted')
        
        plt.xlabel('Severity Class')
        plt.ylabel('Proportion')
        plt.title('Class Distribution: Actual vs Predicted')
        plt.xticks(x)
        plt.legend()
        plt.savefig(f"{save_path}/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Create summary metrics figure
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        key_metrics = {
            'Accuracy': metrics['accuracy'],
            'Ordinal Accuracy (±1)': metrics['ordinal_accuracy'],
            'Quadratic Weighted Kappa': metrics['quadratic_weighted_kappa'],
            'Cohen\'s Kappa': metrics['cohen_kappa'],
            'Weighted Error': metrics['weighted_error'],
            'F1 (weighted)': metrics['f1_weighted'],
            'F1 (macro)': metrics['f1_macro'],
        }
        
        if 'high_severity_recall' in metrics and not np.isnan(metrics['high_severity_recall']):
            key_metrics['High Severity Recall'] = metrics['high_severity_recall']
        
        # Create table
        cell_text = [[f"{key}", f"{value:.4f}"] for key, value in key_metrics.items()]
        
        plt.table(cellText=cell_text, 
                colLabels=['Metric', 'Value'],
                loc='center',
                cellLoc='center',
                colWidths=[0.6, 0.3])
        
        plt.title('Key Evaluation Metrics', pad=20)
        plt.tight_layout()
        plt.savefig(f"{save_path}/summary_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Per-class metrics
        if 'class_report' in metrics and metrics['class_report']:
            plt.figure(figsize=(12, 6))
            
            # Extract metrics for each class
            classes = []
            precision_values = []
            recall_values = []
            f1_values = []
            
            for cls, values in metrics['class_report'].items():
                if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                    classes.append(cls)
                    precision_values.append(values.get('precision', 0))
                    recall_values.append(values.get('recall', 0))
                    f1_values.append(values.get('f1-score', 0))
            
            if classes:  # Only plot if we have classes
                # Create bar chart
                x = np.arange(len(classes))
                width = 0.25
                
                plt.bar(x - width, precision_values, width, label='Precision')
                plt.bar(x, recall_values, width, label='Recall')
                plt.bar(x + width, f1_values, width, label='F1-score')
                
                plt.xlabel('Severity Class')
                plt.ylabel('Score')
                plt.title('Precision, Recall, and F1-score by Class')
                plt.xticks(x, classes)
                plt.legend()
                plt.savefig(f"{save_path}/class_metrics.png", dpi=300, bbox_inches='tight')
                plt.close()
                
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()

def save_metrics_to_json(metrics, save_path='results/flood_model_metrics.json'):
    """Save metrics to a JSON file"""
    try:
        import json
        
        # Convert numpy arrays and other non-serializable types to lists/primitives
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_metrics = convert_to_serializable(metrics)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        print(f"Metrics saved to {save_path}")
    except Exception as e:
        print(f"Error saving metrics to JSON: {e}")

def run_evaluation_pipeline():
    """Main function to run the complete evaluation pipeline"""
    try:
        print("Starting evaluation pipeline...")
        
        # Process and split data
        X_train, X_test, y_train, y_test, dataset = process_and_split_data()
        
        if X_train is None:
            print("Error: Data processing failed.")
            return
        
        # Load or create model
        model = load_or_create_model(X_train, y_train, X_train.shape[1])
        
        # Evaluate the model
        print("Evaluating model...")
        metrics, y_pred, y_pred_probs = evaluate_flood_model(model, X_test, y_test)
        
        # Plot results
        print("Generating plots...")
        plot_evaluation_results(metrics)
        
        # Save metrics
        save_metrics_to_json(metrics)
        
        # Print key metrics
        print("\n===== FLOOD PREDICTION MODEL EVALUATION =====")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Ordinal Accuracy (±1 level): {metrics['ordinal_accuracy']:.4f}")
        print(f"Quadratic Weighted Kappa: {metrics['quadratic_weighted_kappa']:.4f}")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"Weighted Error: {metrics['weighted_error']:.4f}")
        
        # Print class-specific metrics if available
        if 'class_report' in metrics and metrics['class_report']:
            print("\n--- Per-Class Metrics ---")
            for cls, values in metrics['class_report'].items():
                if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                    print(f"{cls}: Precision={values.get('precision', 0):.3f}, " +
                          f"Recall={values.get('recall', 0):.3f}, F1={values.get('f1-score', 0):.3f}, " +
                          f"Support={values.get('support', 0)}")
        
        print("\nEvaluation complete. Results saved to 'results' directory.")
        return metrics
    
    except Exception as e:
        print(f"Error during evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    metrics = run_evaluation_pipeline()