import numpy as np
import os
import random
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    data_dir = 'data/'
    models_dir = 'models/'
    docs_data_dir = 'docs/data/'

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_data_dir, exist_ok=True)

    # 1. Load Preprocessed Data
    print("Loading preprocessed data...")
    X_train_scaled = np.load(os.path.join(data_dir, 'X_train_scaled.npy'))
    X_test_scaled = np.load(os.path.join(data_dir, 'X_test_scaled.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # 2. Build Artificial Neural Network (ANN)
    print("Building Artificial Neural Network (ANN)...")
    ann_model = Sequential()
    ann_model.add(Dense(100, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    ann_model.add(Dense(80, activation='relu'))
    ann_model.add(Dense(1, activation='linear'))

    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 3. Train the Model
    print("Training started. This may take a minute or two...\n")
    history = ann_model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # 4. Evaluate the Model
    y_pred_ann = ann_model.predict(X_test_scaled)
    r2_ann = r2_score(y_test, y_pred_ann)
    rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))

    print("\n[SUCCESS] ANN Training Complete!")
    print(f"ANN R² Score: {r2_ann:.4f} (Aiming for >0.97)")
    print(f"ANN RMSE:     {rmse_ann:.6f}")

    # 5. Export Model and History
    ann_path = os.path.join(models_dir, 'ann_model.keras')
    ann_model.save(ann_path)
    print(f"Saved ANN to: {ann_path}")

    # Save training history for frontend
    history_dict = {
        "loss": [float(val) for val in history.history['loss']],
        "val_loss": [float(val) for val in history.history['val_loss']]
    }
    history_path = os.path.join(docs_data_dir, 'ann_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"Saved ANN training history to: {history_path}")

if __name__ == "__main__":
    main()