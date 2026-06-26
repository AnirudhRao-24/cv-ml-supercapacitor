import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error
from scipy.integrate import trapezoid

def calculate_specific_capacitance(potential, current, scan_rate_mv, mass_g=0.001):
    # Convert scan rate from mV/s to V/s
    scan_rate_v = scan_rate_mv / 1000.0
    
    # Calculate potential window width
    potential_window = np.max(potential) - np.min(potential)
    
    # Calculate area under the CV curve (integral of absolute current w.r.t potential)
    # Using absolute differences in potential to prevent cancellation on cyclic sweeps
    diff_pot = np.abs(np.diff(potential))
    avg_current = (np.abs(current[1:]) + np.abs(current[:-1])) / 2.0
    area = np.sum(avg_current * diff_pot)
    
    # Divide total area by the number of cycles to get area per cycle
    total_sweep_length = np.sum(diff_pot)
    num_cycles = total_sweep_length / (2.0 * potential_window)
    area_per_cycle = area / num_cycles
    
    # Specific capacitance formula: C = area_per_cycle / (2 * mass * scan_rate * potential_window)
    specific_capacitance = area_per_cycle / (2.0 * mass_g * scan_rate_v * potential_window)
    return specific_capacitance

def main():
    models_dir = 'models/'
    outputs_dir = 'outputs/'
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Load the out-of-sample test data
    print("Loading completely unseen 60mV Dataset...")
    unseen_file_path = 'Data/60MV_CV.xlsx'
    df_unseen = pd.read_excel(unseen_file_path)

    target_col = 'Current'
    y_unseen = df_unseen[target_col]
    X_unseen = df_unseen.drop(target_col, axis=1)

    # 2. Load Models and Scaler
    print("Loading Scaler and ML Pipeline...")
    scaler = joblib.load(os.path.join(models_dir, 'standard_scaler.pkl'))
    ann_model = load_model(os.path.join(models_dir, 'ann_model.keras'))
    rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))
    meta_model = joblib.load(os.path.join(models_dir, 'meta_stacked_model.pkl'))

    # 3. Scale the unseen data
    X_unseen_scaled = scaler.transform(X_unseen)

    # 4. Generate Base Model Predictions
    print("Generating predictions from base models...")
    unseen_pred_ann = ann_model.predict(X_unseen_scaled, verbose=0).flatten()
    unseen_pred_rf = rf_model.predict(X_unseen_scaled)
    unseen_pred_xgb = xgb_model.predict(X_unseen_scaled)

    # 5. Stack the Predictions for the Meta-Model
    X_meta_unseen = pd.DataFrame({
        'ANN': unseen_pred_ann,
        'RF': unseen_pred_rf,
        'XGB': unseen_pred_xgb
    })

    # 6. Final Meta-Model Prediction
    print("Generating final Meta-Model predictions...")
    final_predictions = meta_model.predict(X_meta_unseen)

    # 7. Calculate Final Out-of-Sample Metrics
    final_r2 = r2_score(y_unseen, final_predictions)
    final_rmse = np.sqrt(mean_squared_error(y_unseen, final_predictions))

    # Calculate specific capacitance (scan rate = 60 mV/s, active mass = 0.001 g / 1 mg)
    sc_exp = calculate_specific_capacitance(df_unseen['Potential'].to_numpy(), y_unseen.to_numpy(), 60.0)
    sc_pred = calculate_specific_capacitance(df_unseen['Potential'].to_numpy(), final_predictions, 60.0)
    sc_error = abs(sc_exp - sc_pred) / sc_exp * 100

    print("\n--- REAL WORLD TEST RESULTS ---")
    print(f"Unseen Data R² Score: {final_r2:.4f}")
    print(f"Unseen Data RMSE:     {final_rmse:.6f}")
    print(f"Experimental Specific Capacitance: {sc_exp:.5f} F/g")
    print(f"Predicted Specific Capacitance:    {sc_pred:.5f} F/g (Error: {sc_error:.2f}%)")

    # 8. Plot the actual vs predicted CV Curve
    plt.figure(figsize=(10, 6))

    # Plot the real experimental curve in blue
    plt.plot(df_unseen['Potential'], y_unseen, label='Actual CV Curve (Experimental)', color='#1f77b4', linewidth=2)

    # Plot our machine learning prediction in red dashes
    plt.plot(df_unseen['Potential'], final_predictions, label='Predicted CV Curve (Meta-Model)', color='#d62728', linestyle='--', linewidth=2)

    plt.xlabel('Potential (V)', fontsize=12)
    plt.ylabel('Current (A)', fontsize=12)
    plt.title('Validation: Actual vs Predicted CV Curve at 60mV/s', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Save the plot
    plot_path = os.path.join(outputs_dir, 'validation_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved validation curve plot to: {plot_path}")

    print("Evaluation plot saved. Skipping interactive display to prevent console blocking.")

if __name__ == "__main__":
    main()