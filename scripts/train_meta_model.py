import pandas as pd
import numpy as np
import os
import joblib
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def main():
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

    # 2. Load Final Base Models (trained on full training set)
    print("Loading final Base Models...")
    ann_model = load_model(os.path.join(models_dir, 'ann_model.keras'))
    rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))

    # 3. Generate Out-of-Fold (OOF) Predictions to train the Meta-Model (No Leakage!)
    print("\nGenerating Out-of-Fold (OOF) predictions via 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_pred_ann = np.zeros(len(X_train_scaled))
    oof_pred_rf = np.zeros(len(X_train_scaled))
    oof_pred_xgb = np.zeros(len(X_train_scaled))

    fold = 1
    for train_idx, val_idx in kf.split(X_train_scaled):
        print(f"--- Training Fold {fold}/5 ---")
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Fold ANN
        fold_ann = Sequential()
        fold_ann.add(Dense(100, activation='relu', input_shape=(X_train_scaled.shape[1],)))
        fold_ann.add(Dense(80, activation='relu'))
        fold_ann.add(Dense(1, activation='linear'))
        fold_ann.compile(optimizer='adam', loss='mse')
        fold_ann.fit(X_tr, y_tr, epochs=50, batch_size=32, verbose=0)

        # Fold RF
        fold_rf = RandomForestRegressor(n_estimators=40, max_depth=11, random_state=42, n_jobs=-1)
        fold_rf.fit(X_tr, y_tr)

        # Fold XGBoost
        fold_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        fold_xgb.fit(X_tr, y_tr)

        # Predict on validation fold
        oof_pred_ann[val_idx] = fold_ann.predict(X_val, verbose=0).flatten()
        oof_pred_rf[val_idx] = fold_rf.predict(X_val)
        oof_pred_xgb[val_idx] = fold_xgb.predict(X_val)
        fold += 1

    X_meta_train = pd.DataFrame({
        'ANN': oof_pred_ann,
        'RF': oof_pred_rf,
        'XGB': oof_pred_xgb
    })

    # 4. Generate Predictions from the Testing Data (using final full-set models)
    print("\nGenerating final predictions on Test Set...")
    test_pred_ann = ann_model.predict(X_test_scaled, verbose=0).flatten()
    test_pred_rf = rf_model.predict(X_test_scaled)
    test_pred_xgb = xgb_model.predict(X_test_scaled)

    X_meta_test = pd.DataFrame({
        'ANN': test_pred_ann,
        'RF': test_pred_rf,
        'XGB': test_pred_xgb
    })

    # 5. Train the Meta-Model (Ridge Regression) on OOF Training Features
    print("Fitting RidgeCV Meta-Model...")
    meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta_model.fit(X_meta_train, y_train)

    # 6. Final Evaluation
    y_pred_meta = meta_model.predict(X_meta_test)
    r2_meta = r2_score(y_test, y_pred_meta)
    rmse_meta = np.sqrt(mean_squared_error(y_test, y_pred_meta))

    print("\n[SUCCESS] Meta-Model Training Complete!")
    print("--- FINAL ACCURACY (OOF Stacking) ---")
    print(f"Meta-Model R² Score: {r2_meta:.4f}")
    print(f"Meta-Model RMSE:     {rmse_meta:.6f}")
    print(f"Ridge Alpha chosen:  {meta_model.alpha_}")

    # 7. Save Meta-Model
    joblib.dump(meta_model, os.path.join(models_dir, 'meta_stacked_model.pkl'))
    print(f"Saved Meta-Model to {models_dir}meta_stacked_model.pkl")

    # Export scatter data for the frontend Chart.js (sample 100 points for visualization)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(y_test), size=min(100, len(y_test)), replace=False)
    scatter_data = [{"x": float(y_test[i]), "y": float(y_pred_meta[i])} for i in indices]
    
    scatter_path = os.path.join(docs_data_dir, 'test_scatter.json')
    with open(scatter_path, 'w') as f:
        json.dump(scatter_data, f, indent=4)
    print(f"Saved test scatter plot sample points to {scatter_path}")

if __name__ == "__main__":
    main()