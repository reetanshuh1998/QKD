#!/usr/bin/env python3
"""
SHAP explainability analysis for the Hybrid Autoencoder + XGBoost model.

Generates:
  - models/plots/paper_figures/shap_summary_bar.png   (global feature importance)
  - models/plots/paper_figures/shap_beeswarm.png      (per-sample feature effects)

Requires: the full training pipeline to have been executed first
(generate_qkd_dataset → feature_engineering → model_training → hyperparameter_tuning).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs('../models/plots/paper_figures', exist_ok=True)


def main():
    print("Loading data and models for SHAP analysis...")

    # --- Load data (identical split to training pipeline) ---
    df = pd.read_csv('../data/processed/qkd_processed.csv')
    X = df.drop(columns=['Label']).values
    y = df['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Load models ---
    scaler = joblib.load('../models/scaler.pkl')
    label_encoder = joblib.load('../models/label_encoder.pkl')
    autoencoder = load_model('../models/autoencoder.keras')
    encoder = load_model('../models/encoder.keras')

    # Prefer tuned model; fall back to base hybrid
    try:
        xgb_model = joblib.load('../models/xgboost_hybrid_tuned.pkl')
        print("Using tuned hybrid XGBoost model.")
    except FileNotFoundError:
        xgb_model = joblib.load('../models/xgboost_hybrid.pkl')
        print("Using base hybrid XGBoost model.")

    # --- Build hybrid feature matrix (must match training pipeline) ---
    X_test_scaled = scaler.transform(X_test)
    latent_test = encoder.predict(X_test_scaled, verbose=0)
    test_reconst = autoencoder.predict(X_test_scaled, verbose=0)
    mse_test = np.mean(np.square(X_test_scaled - test_reconst), axis=1).reshape(-1, 1)

    X_test_hybrid = np.hstack((X_test_scaled, latent_test, mse_test))

    # --- Construct feature names ---
    base_features = df.drop(columns=['Label']).columns.tolist()
    latent_features = [f'AE_Latent_{i}' for i in range(latent_test.shape[1])]
    feature_names = base_features + latent_features + ['Reconstruction_MSE']

    print(f"Hybrid feature dimension: {X_test_hybrid.shape[1]}")
    print(f"Computing SHAP values (TreeExplainer)...")

    # --- SHAP TreeExplainer ---
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_hybrid)

    # --- Bar summary plot (global importance) ---
    print("Generating SHAP bar summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values, X_test_hybrid,
        feature_names=feature_names,
        plot_type='bar',
        class_names=label_encoder.classes_,
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig('../models/plots/paper_figures/shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: models/plots/paper_figures/shap_summary_bar.png")

    # --- Beeswarm plot (per-class, pick the most interesting class) ---
    # For multi-class, shap_values is a list of arrays. Show class 0 (combined)
    # or the global view if shap handles it.
    print("Generating SHAP beeswarm plot...")
    plt.figure()
    if isinstance(shap_values, list):
        # Multi-class: show average absolute SHAP across all classes
        shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        shap.summary_plot(
            shap_abs_mean, X_test_hybrid,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
    else:
        shap.summary_plot(
            shap_values, X_test_hybrid,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
    plt.tight_layout()
    plt.savefig('../models/plots/paper_figures/shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: models/plots/paper_figures/shap_beeswarm.png")

    print("SHAP analysis complete.")


if __name__ == '__main__':
    main()
