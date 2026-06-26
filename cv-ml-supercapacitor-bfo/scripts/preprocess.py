import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Define paths (Update these to your local or server paths)
    file_path = 'Data/CV_DATASET.xlsx'
    models_dir = 'models/'
    data_dir = 'data/'
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 2. Load the Excel data
    print(f"Loading dataset from {file_path}...\n")
    df = pd.read_excel(file_path)

    print("--- Data Inspection ---")
    print(f"Dataset Shape (Rows, Columns): {df.shape}")
    
    # 3. Separate Target (y) and Features (X)
    target_col = 'Current'
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 5. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save the Scaler
    joblib.dump(scaler, os.path.join(models_dir, 'standard_scaler.pkl'))
    print(f"Saved scaler to {models_dir}standard_scaler.pkl")

    # 7. Save intermediate processed data for the training scripts
    np.save(os.path.join(data_dir, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(data_dir, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test.to_numpy())

    print("\n[SUCCESS] Preprocessing Complete! Data is split, scaled, and ready for modeling.")

if __name__ == "__main__":
    main()