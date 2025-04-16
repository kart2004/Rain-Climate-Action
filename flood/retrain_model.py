from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def retrain_model():
    # Load the dataset
    dataset = pd.read_csv('data/flood_past.csv')
    dataset = dataset[dataset['YEAR'] > 1980]
    dataset = dataset.dropna()

    # Encode the SUBDIVISION column
    labelencoder_X_1 = LabelEncoder()
    all_states = dataset['SUBDIVISION'].unique()
    labelencoder_X_1.fit(all_states)
    X = dataset.iloc[:, [0, 4]].values  # SUBDIVISION and PRECIPITATION
    y = dataset.iloc[:, 8].values       # SEVERITY

    X[:, 0] = labelencoder_X_1.transform(X[:, 0])  # Encode SUBDIVISION
    X_encoded = X[:, 0].reshape(-1, 1)
    onehotencoder = OneHotEncoder(sparse_output=False)
    encoded_features = onehotencoder.fit_transform(X_encoded)
    X = np.column_stack((encoded_features, X[:, 1]))  # Combine encoded SUBDIVISION and PRECIPITATION

    # Encode the target variable (SEVERITY)
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale the features
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # One-hot encode the target variable
    onehotencoder_2 = OneHotEncoder(sparse_output=False)
    y_train = np.reshape(y_train, (-1, 1))
    y_train = onehotencoder_2.fit_transform(y_train)

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)),
        keras.layers.Dense(16, activation='relu',
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)),
        keras.layers.Dense(y_train.shape[1], activation='softmax',
                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05))
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Save the model weights
    model.save_weights('./model/flood_model.weights.h5')
    print("Model retrained and weights saved.")

    # Evaluate the model on the test set
    y_test = np.reshape(y_test, (-1, 1))
    y_test = onehotencoder_2.transform(y_test)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model