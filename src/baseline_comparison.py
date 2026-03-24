import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
import joblib

def main():
    print("--- Running Comparative Architecture Analysis ---")
    
    # 1. Load the preprocessed dataset
    df = pd.read_csv('../data/processed/qkd_processed.csv')
    
    # Exclude Label to get raw normalized features
    X_raw = df.drop(columns=['Label']).values
    y = df['Label'].values
    
    # Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nSecuring isolated post-split feature scaling...")
    from sklearn.preprocessing import StandardScaler
    try:
        scaler = joblib.load('../models/scaler.pkl')
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # 2. Train Standalone XGBoost (Baseline Model)
    print("\nTraining Baseline Standalone XGBoost...")
    baseline_xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, n_estimators=100, max_depth=5, learning_rate=0.05, n_jobs=-1)
    baseline_xgb.fit(X_train_scaled, y_train)
    y_pred_base = baseline_xgb.predict(X_test_scaled)
    baseline_accuracy = accuracy_score(y_test, y_pred_base)
    print(f"Standlone XGBoost Accuracy: {baseline_accuracy*100:.2f}%")
    
    # 3. Train Hybrid (Autoencoder + XGBoost)
    print("\nExtracting Hybrid Features...")
    try:
        encoder = load_model('../models/encoder.keras')
    except Exception as e:
        print("Autoencoder model not found. Please run model_training.py first.")
        return
        
    X_train_latent = encoder.predict(X_train_scaled)
    X_test_latent = encoder.predict(X_test_scaled)
    
    # Recreate MSE features structurally to match training pipeline
    X_train_mse = np.mean(np.square(X_train_scaled - load_model('../models/autoencoder.keras').predict(X_train_scaled)), axis=1).reshape(-1, 1)
    X_test_mse = np.mean(np.square(X_test_scaled - load_model('../models/autoencoder.keras').predict(X_test_scaled)), axis=1).reshape(-1, 1)
    
    X_train_hybrid = np.hstack((X_train_scaled, X_train_latent, X_train_mse))
    X_test_hybrid = np.hstack((X_test_scaled, X_test_latent, X_test_mse))
    
    print("\nTraining Hybrid Autoencoder + XGBoost...")
    hybrid_xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, n_estimators=250, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    hybrid_xgb.fit(X_train_hybrid, y_train)
    y_pred_hybrid = hybrid_xgb.predict(X_test_hybrid)
    hybrid_accuracy = accuracy_score(y_test, y_pred_hybrid)
    print(f"Hybrid Model Accuracy: {hybrid_accuracy*100:.2f}%")
    
    # 4. Train Deep Neural Network (DNN)
    print("\nTraining Deep Neural Network (DNN)...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import LabelBinarizer
    
    lb = LabelBinarizer()
    y_train_ohe = lb.fit_transform(y_train)
    y_test_ohe = lb.transform(y_test)
    
    dnn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(y_train_ohe.shape[1], activation='softmax')
    ])
    
    dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop_dnn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    dnn.fit(X_train_scaled, y_train_ohe, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop_dnn], verbose=0)
    
    y_pred_dnn_prob = dnn.predict(X_test_scaled)
    y_pred_dnn = np.argmax(y_pred_dnn_prob, axis=1)
    
    dnn_accuracy = accuracy_score(y_test, y_pred_dnn)
    print(f"Deep Neural Network (DNN) Accuracy: {dnn_accuracy*100:.2f}%\n")
    
    # 5. Graphical Comparison
    models = ['XGBoost\nBaseline', 'Hybrid\n(AE+XGB)', 'Deep Neural\nNetwork (DNN)']
    accuracies = [baseline_accuracy * 100, hybrid_accuracy * 100, dnn_accuracy * 100]
    
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=models, y=accuracies, palette="coolwarm")
    plt.ylim(min(accuracies) - 5, 100)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Advanced Architecture Performance Comparison', fontsize=14, pad=15)
    
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    import os
    os.makedirs('../models/plots', exist_ok=True)
    plt.savefig('../models/plots/architecture_comparison.png', dpi=300)
    print("\nSaved comparison graphic to models/plots/architecture_comparison.png")

if __name__ == '__main__':
    main()
