import numpy as np
import os
import joblib
import json
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

    # ==========================================
    # Random Forest Model
    # ==========================================
    print("\nBuilding Random Forest (RF) Model...")
    rf_model = RandomForestRegressor(
        n_estimators=40,
        max_depth=11,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training RF started...")
    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print("[SUCCESS] Random Forest Training Complete!")
    print(f"RF R² Score: {r2_rf:.4f}")
    print(f"RF RMSE:     {rmse_rf:.6f}")
    
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))

    # Export RF feature importances
    rf_importance_dict = {
        "labels": ["Potential", "Scan Rate", "Oxidation State", "Dopant 1", "Dopant 2", "Dopant 3"],
        "importances": [float(val) for val in rf_model.feature_importances_]
    }
    rf_importance_path = os.path.join(docs_data_dir, 'rf_importance.json')
    with open(rf_importance_path, 'w') as f:
        json.dump(rf_importance_dict, f, indent=4)
    print(f"Saved RF importances to {rf_importance_path}")

    # ==========================================
    # XGBoost Model
    # ==========================================
    print("\nBuilding XGBoost (XGB) Model...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    print("Training XGB started...")
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )

    y_pred_xgb = xgb_model.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    print("[SUCCESS] XGBoost Training Complete!")
    print(f"XGB R² Score: {r2_xgb:.4f}")
    print(f"XGB RMSE:     {rmse_xgb:.6f}")
    
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
    print(f"\nSaved RF and XGB models to {models_dir}")

    # Export XGBoost error history
    evals_result = xgb_model.evals_result()
    # validation_0 is train, validation_1 is test/validation
    xgb_history_dict = {
        "rounds": list(range(len(evals_result['validation_0']['rmse']))),
        "train_rmse": [float(val) for val in evals_result['validation_0']['rmse']],
        "test_rmse": [float(val) for val in evals_result['validation_1']['rmse']]
    }
    xgb_history_path = os.path.join(docs_data_dir, 'xgb_history.json')
    with open(xgb_history_path, 'w') as f:
        json.dump(xgb_history_dict, f, indent=4)
    print(f"Saved XGB history to {xgb_history_path}")

if __name__ == "__main__":
    main()