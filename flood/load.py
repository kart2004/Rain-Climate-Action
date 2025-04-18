from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import os
import pickle
from sklearn.exceptions import NotFittedError

# Global variables to store encoders and scalers
label_encoder_terrain = None
terrain_onehotencoder = None

def create_model(input_shape):
    """Create an improved model with regularization and dropout"""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,),
                          kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),  # Add dropout to prevent overfitting
        keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(6, activation='softmax')  # 6 outputs for severity 0-5
    ])
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

def retrain_model():
    """Retrain model with state, terrain, and precipitation features"""
    global label_encoder_terrain, terrain_onehotencoder
    
    # Load dataset
    dataset = pd.read_csv('data/flood_past.csv')
    dataset = dataset[dataset['YEAR'] > 1980]
    dataset = dataset.dropna()
    
    # Extract features: SUBDIVISION, TERRAIN, PRECIPITATION 
    X = dataset.iloc[:, [0, 6, 4]].values
    y = dataset.iloc[:, 8].values  # SEVERITY
    
    # Encode state (SUBDIVISION) - Use existing encoder
    from flood.config import labelencoder_X_1, onehotencoder
    labelencoder_X_1.fit(X[:, 0])  # Ensure the encoder is fitted
    X[:, 0] = labelencoder_X_1.transform(X[:, 0])
    
    # Fit the onehotencoder with the transformed state values
    X_encoded = X[:, 0].reshape(-1, 1)
    onehotencoder.fit(X_encoded)  # Fit the encoder here
    state_onehot = onehotencoder.transform(X_encoded)
    
    # Encode terrain
    label_encoder_terrain = LabelEncoder()
    X[:, 1] = label_encoder_terrain.fit_transform(X[:, 1])
    
    # One-hot encode terrain
    terrain_onehotencoder = OneHotEncoder(sparse_output=False)
    terrain_encoded = X[:, 1].reshape(-1, 1)
    terrain_onehot = terrain_onehotencoder.fit_transform(terrain_encoded)
    
    # Combine features with precipitation
    precipitation = X[:, 2].reshape(-1, 1)
    X_processed = np.hstack((state_onehot, terrain_onehot, precipitation))
    
    # Encode target (SEVERITY)
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    
    # Split data for training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=0)
    
    # Scale features
    from flood.config import sc_X
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    # One-hot encode target
    onehotencoder_y = OneHotEncoder(sparse_output=False)
    y_train_reshaped = y_train.reshape(-1, 1)
    y_train_onehot = onehotencoder_y.fit_transform(y_train_reshaped)
    
    y_test_reshaped = y_test.reshape(-1, 1)
    y_test_onehot = onehotencoder_y.transform(y_test_reshaped)
    
    # Handle class imbalance with balanced weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_train), 
                                        y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Create model with proper input shape
    input_shape = X_train.shape[1]
    model = create_model(input_shape)
    
    # Train with early stopping and class weights
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    model.fit(X_train, y_train_onehot, 
              epochs=100, 
              batch_size=32, 
              validation_split=0.2,
              callbacks=[early_stopping], 
              class_weight=class_weight_dict)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    # Save model weights
    model.save_weights('./model/flood_model.weights.h5')
    
    # Save terrain encoder for future use
    with open('./model/terrain_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder_terrain, f)
    with open('./model/terrain_onehotencoder.pkl', 'wb') as f:
        pickle.dump(terrain_onehotencoder, f)
    
    return model

# Change the init() function near line 140
def init():
    """Initialize model with enhanced features"""
    global label_encoder_terrain, terrain_onehotencoder
    
    # First ensure the encoders are ready
    from flood.config import onehotencoder, labelencoder_X_1
    
    # Check if the onehotencoder is fitted, if not, create a simple one for initialization
    fitted_encoder = False
    try:
        # Try a simple transform to check if it's fitted
        onehotencoder.transform(np.array([[0]]))
        fitted_encoder = True
    except (NotFittedError, Exception):
        # If onehotencoder is not fitted, we'll use default dimensions
        fitted_encoder = False
    
    # Try to load terrain encoders if they exist
    try:
        if os.path.exists('./model/terrain_encoder.pkl'):
            with open('./model/terrain_encoder.pkl', 'rb') as f:
                label_encoder_terrain = pickle.load(f)
            with open('./model/terrain_onehotencoder.pkl', 'rb') as f:
                terrain_onehotencoder = pickle.load(f)
    except Exception as e:
        print(f"Note: Terrain encoders not loaded: {e}")
    
    # Determine dimensions without relying on the encoders
    state_dim = 30  # Default: assume we have 30 states
    terrain_dim = 5  # Default: 5 terrain types
    
    # Use encoder dimensions if they're already fitted
    if fitted_encoder:
        sample_state = np.array([[0]])
        state_dim = onehotencoder.transform(sample_state).shape[1]
    
    if terrain_onehotencoder is not None:
        sample_terrain = np.array([[0]])
        terrain_dim = terrain_onehotencoder.transform(sample_terrain).shape[1]
    
    # Total input shape: one-hot encoded state + one-hot encoded terrain + precipitation
    input_shape = state_dim + terrain_dim + 1
    
    # Create model with the right input dimension
    model = create_model(input_shape)
    
    # Try to load weights
    try:
        model.load_weights("./model/flood_model.weights.h5")
        print("Loaded model weights from disk")
    except Exception as e:
        print(f"Error loading weights: {e} - Using untrained model")
        # Don't retrain here to avoid circular dependency
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.compat.v1.get_default_graph()
    
    return model, graph
