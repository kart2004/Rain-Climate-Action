import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def create_rainfall_model(input_dim):
    """
    Create a neural network model for rainfall prediction.
    """
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001))(x)
    outputs = keras.layers.Dense(1, activation='linear')(x)  # regression
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def generate_future_rainfall():
    # 1) Load & clean
    df = pd.read_csv('data/flood_past.csv').dropna()

    # 2) Prepare features & target
    X = df[['SUBDIVISION', 'YEAR', 'QUARTER', 'TERRAIN']].copy()
    y = df['PRECIPITATION'].values

    # 3) Encode categorical
    le_state   = LabelEncoder().fit(X['SUBDIVISION'])
    le_terrain = LabelEncoder().fit(X['TERRAIN'])
    X['SUBDIVISION'] = le_state.transform(X['SUBDIVISION'])
    X['TERRAIN']     = le_terrain.transform(X['TERRAIN'])

    ohe_q = OneHotEncoder(sparse_output=False,  
                          categories=[sorted(X['QUARTER'].unique())]) \
                  .fit(X[['QUARTER']])
    q_encoded = ohe_q.transform(X[['QUARTER']])
    X = X.drop('QUARTER', axis=1)
    X = np.hstack([X.values, q_encoded])   # now shape = (n_samples, 7)

    # 4) Scale
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # 6) Build & train model
    model = create_rainfall_model(X_train.shape[1])
    model.fit(
        X_train, y_train,
        epochs=50, batch_size=32,
        validation_split=0.2, verbose=1
    )

    # 7) Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.2f}, Test MAE: {mae:.2f}")

    # 8) Prepare ALL future samples at once
    future_years  = np.arange(2016, 2031)
    subdivisions  = np.unique(df['SUBDIVISION'])
    quarters      = ohe_q.categories_[0]
    terrains_full = df[['SUBDIVISION','TERRAIN']].drop_duplicates().set_index('SUBDIVISION')['TERRAIN']

    all_samples = []
    for year in future_years:
        for sub in subdivisions:
            terr = terrains_full.loc[sub]
            s_enc = le_state.transform([sub])[0]
            t_enc = le_terrain.transform([terr])[0]
            for q in quarters:
                q_enc = ohe_q.transform([[q]])[0]
                row = [s_enc, year, t_enc, *q_enc]
                all_samples.append(row)

    all_samples = np.array(all_samples)              # shape (N_future, 7)
    all_scaled  = scaler.transform(all_samples)      # same 7 dims

    # 9) Batch predict
    preds = model.predict(all_scaled, verbose=0).flatten()

    # 10) Build DataFrame & save
    recs = []
    idx = 0
    for year in future_years:
        for sub in subdivisions:
            terr = terrains_full.loc[sub]
            for q in quarters:
                recs.append([sub, year, q, terr, float(preds[idx])])
                idx += 1

    future_df = pd.DataFrame(
        recs,
        columns=['SUBDIVISION','YEAR','QUARTER','TERRAIN','PREDICTED_PRECIPITATION']
    )
    future_df.to_csv('data/flood_gen_future.csv', index=False)
    print("Saved future rainfall to data/flood_gen_future.csv")

if __name__ == "__main__":
    # Force TensorFlow to use CPU only (optional)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.get_logger().setLevel('ERROR')
    generate_future_rainfall()
