"""
train_baseline.py

Purpose: Train and honestly evaluate baseline models (Random Forest, XGBoost)
to predict polymer Tg from our featurized data.

Uses 5-fold cross-validation so every row gets used for testing exactly
once, giving a stable performance estimate.

Run: python3 src/train_baseline.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

DATA_PATH = "data/processed/featurized.csv"
MODEL_DIR = "models"

def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df["Tg_celsius"].values
    # Fingerprints-only beat descriptors-only AND the combined set in our
    # earlier 50-row experiment (debug_feature_count.py). Using that result.
    fingerprint_cols = [c for c in df.columns if c.startswith("fp_")]
    X = df[fingerprint_cols].values
    names = df["polymer_name"].values
    return X, y, names

def evaluate_model(model, X, y, model_name):
    """
    cross_val_predict trains the model K times (K=5), each time on 4/5 of
    the data, predicting on the held-out 1/5. Stitches together predictions
    for EVERY row, each made by a model that never saw that row in training.
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kfold)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"\n{model_name} (5-fold cross-validated):")
    print(f"  MAE  = {mae:.1f} °C   (average absolute prediction error)")
    print(f"  RMSE = {rmse:.1f} °C  (penalizes large errors more)")
    print(f"  R²   = {r2:.3f}       (1.0 = perfect, 0.0 = no better than guessing the mean)")

    return y_pred, {"mae": mae, "rmse": rmse, "r2": r2}

def main():
    X, y, names = load_data()
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # --- Model 1: Random Forest ---
    # max_depth increased from 6 (safe for 50 rows) to 16, now that 7,367
    # rows make deeper trees safe from overfitting.
    rf = RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
    rf_pred, rf_metrics = evaluate_model(rf, X, y, "Random Forest")

    # --- Model 2: XGBoost ---
    xgb = XGBRegressor(
        n_estimators=400, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1
    )
    xgb_pred, xgb_metrics = evaluate_model(xgb, X, y, "XGBoost")

    # --- Baseline sanity check ---
    mean_pred = np.full_like(y, y.mean(), dtype=float)
    mean_mae = mean_absolute_error(y, mean_pred)
    print(f"\nDummy baseline (always predict mean Tg = {y.mean():.1f}°C):")
    print(f"  MAE  = {mean_mae:.1f} °C   <- both models above must beat this")

    rf.fit(X, y)
    xgb.fit(X, y)
    joblib.dump(rf, f"{MODEL_DIR}/random_forest_tg.joblib")
    joblib.dump(xgb, f"{MODEL_DIR}/xgboost_tg.joblib")
    print(f"\nSaved trained models to {MODEL_DIR}/")

    results_df = pd.DataFrame({
        "polymer_name": names,
        "actual_Tg": y,
        "rf_predicted_Tg": rf_pred,
        "xgb_predicted_Tg": xgb_pred,
    })
    results_df.to_csv("results/cv_predictions.csv", index=False)
    print("Saved cross-validated predictions to results/cv_predictions.csv")

if __name__ == "__main__":
    main()