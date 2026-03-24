import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
# Suppress heavy TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_autoencoder(input_dim, latent_dim):
    # Encoder - Deeper architecture for 29 dimensions
    inputs = Input(shape=(input_dim,))
    
    x = Dense(32)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dense(16)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    latent = Dense(latent_dim, activation='relu', name='latent_layer')(x)
    
    # Decoder
    x = Dense(16)(latent)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    outputs = Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, latent)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def main():
    print("Loading processed QKD data...")
    df = pd.read_csv('../data/processed/qkd_processed.csv')
    
    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    
    # Split the dataset 80/20 first to prevent feature data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Data POST-split exclusively
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, '../models/scaler.pkl')
    
    print(f"Data Split shapes: X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
    
    input_dim = X_train_scaled.shape[1] # Use scaled data for input_dim
    latent_dim = 8  # Expanded feature dimension to capture WCP variances
    
    print("\n--- Training Deep Learning Autoencoder ---")
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    
    # 4. Filter Normal Traffic for Autoencoder Training (Anomaly Detection paradigm)
    try:
        normal_label = label_encoder.transform(['normal'])[0]
    except Exception:
        # Fallback if label encoding is different or 'normal' not found
        # This assumes 'normal' is typically the most frequent class or a known numerical value
        # For robust fallback, one might inspect unique values in y_str or y
        print("Warning: 'normal' label not found or label_encoder issue. Falling back to label 3 for normal traffic.")
        normal_label = 3 # Assuming '3' is the numerical representation for 'normal'
        
    mask_normal = (y_train == normal_label)
    X_train_normal = X_train_scaled[mask_normal]

    print(f"Training Autoencoder on {len(X_train_normal)} Normal Traffic Samples...")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save the models
    autoencoder.save('../models/autoencoder.keras')
    encoder.save('../models/encoder.keras')
    print("Autoencoder models saved.")
    
    print("\n--- Extracting Latent Representations ---")
    latent_train = encoder.predict(X_train)
    latent_test = encoder.predict(X_test)
    
    # Calculate reconstruction error as an additional feature
    train_reconstructions = autoencoder.predict(X_train)
    test_reconstructions = autoencoder.predict(X_test)
    
    mse_train = np.mean(np.square(X_train - train_reconstructions), axis=1).reshape(-1, 1)
    mse_test = np.mean(np.square(X_test - test_reconstructions), axis=1).reshape(-1, 1)
    
    # Form the Hybrid Dataset: Original Features + Latent Vector + MSE Reconstruct Error
    X_train_hybrid = np.hstack((X_train, latent_train, mse_train))
    X_test_hybrid = np.hstack((X_test, latent_test, mse_test))
    
    print(f"Hybrid Features Shape: {X_train_hybrid.shape}")
    
    print("\n--- Training Hybrid XGBoost Model ---")
    # Using XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, 
        max_depth=8, 
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42
    )
    
    xgb.fit(X_train_hybrid, y_train)
    
    # Save XGBoost
    joblib.dump(xgb, '../models/xgboost_hybrid.pkl')
    print("Hybrid XGBoost model saved.")
    
    print("\n--- Evaluating Hybrid Model on Test Set ---")
    y_pred = xgb.predict(X_test_hybrid)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n======================================")
    print(f"OVERALL HYBRID MODEL ACCURACY: {acc:.4f}")
    print(f"======================================\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save classification report to file for paper prep
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv('../models/evaluation_metrics.csv')
    
    print("Model development complete! Metrics saved to models/evaluation_metrics.csv.")

if __name__ == '__main__':
    main()
