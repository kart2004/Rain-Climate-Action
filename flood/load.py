from tensorflow import keras
import tensorflow as tf
from flood.retrain_model import retrain_model

def init():
    # Recreate the model architecture
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(31,), 
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)),
        keras.layers.Dense(16, activation='relu',
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)),
        keras.layers.Dense(6, activation='softmax',
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05))
    ])
    
    # Try to load weights directly
    try:
        model.load_weights("./model/flood_model.weights.h5")
        print("Loaded model weights from disk")
    except Exception as e:
        print(f"Error loading weights: {e} - Retraining the model...")
        model = retrain_model()  # Retrain the model if weights are missing or fail to load
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.compat.v1.get_default_graph()
    return model, graph