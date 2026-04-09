import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Ensure plots directory exists
os.makedirs('../models/plots', exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("Loading datasets and models...")
    df = pd.read_csv('../data/processed/qkd_processed.csv')
    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    
    # Load Label Encoder to get class names
    label_encoder = joblib.load('../models/label_encoder.pkl')
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Autoencoder and Encoder
    autoencoder = load_model('../models/autoencoder.keras')
    encoder = load_model('../models/encoder.keras')
    
    print("Generating Hybrid Features...")
    scaler = joblib.load('../models/scaler.pkl')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    latent_train = encoder.predict(X_train_scaled, verbose=0)
    latent_test = encoder.predict(X_test_scaled, verbose=0)
    
    train_reconst = autoencoder.predict(X_train_scaled, verbose=0)
    test_reconst = autoencoder.predict(X_test_scaled, verbose=0)
    
    mse_train = np.mean(np.square(X_train_scaled - train_reconst), axis=1).reshape(-1, 1)
    mse_test = np.mean(np.square(X_test_scaled - test_reconst), axis=1).reshape(-1, 1)
    
    X_train_hybrid = np.hstack((X_train_scaled, latent_train, mse_train))
    X_test_hybrid = np.hstack((X_test_scaled, latent_test, mse_test))
    
    print("Starting Hyperparameter Tuning on XGBoost (RandomizedSearchCV)...")
    # Define Parameter Grid
    param_grid = {
        'n_estimators': [150, 300, 400, 500],
        'max_depth': [5, 7, 9, 12, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    xgb_base = XGBClassifier(eval_metric='mlogloss', random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Using 10 iterations to rigorously explore deeply orthogonal boundaries safely natively
    random_search = RandomizedSearchCV(
        xgb_base, param_grid, n_iter=10, 
        scoring='accuracy', cv=cv, verbose=2, 
        random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train_hybrid, y_train)
    best_xgb = random_search.best_estimator_
    print(f"Best XGBoost Parameters: {random_search.best_params_}")
    
    # Save the tuned model
    joblib.dump(best_xgb, '../models/xgboost_hybrid_tuned.pkl')
    
    print("Evaluating Best Model...")
    y_pred = best_xgb.predict(X_test_hybrid)
    y_pred_proba = best_xgb.predict_proba(X_test_hybrid)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n======================================")
    print(f"NEW TUNED MODEL ACCURACY: {acc:.4%}")
    print(f"======================================\n")
    
    # --- Generate Paper Plots ---
    print("Generating Visualizations for the Paper...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Hybrid Model, Acc: {acc:.2%})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../models/plots/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 2. Multi-class / Binary ROC Curve
    plt.figure(figsize=(10, 8))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[1]} (AUC = {roc_auc:.3f})')
    else:
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 9})
    plt.tight_layout()
    plt.savefig('../models/plots/roc_curves.png', dpi=300)
    plt.close()
    
    # 3. Learning Curve (Using eval_set on a fresh training of the best model)
    eval_set = [(X_train_hybrid, y_train), (X_test_hybrid, y_test)]
    # Use early stopping logic on a separate model to extract learning curves
    best_xgb_learning = XGBClassifier(**random_search.best_params_, eval_metric='mlogloss', random_state=42)
    best_xgb_learning.fit(X_train_hybrid, y_train, eval_set=eval_set, verbose=False)
    results = best_xgb_learning.evals_result()
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['mlogloss'], label='Train Loss')
    plt.plot(results['validation_1']['mlogloss'], label='Validation Loss')
    plt.ylabel('Log Loss', fontsize=12)
    plt.xlabel('Trees (Epochs)', fontsize=12)
    plt.title('Hybrid Model Learning Curve', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../models/plots/learning_curve.png', dpi=300)
    plt.close()

    # 4. Feature Importance for the Hybrid Model
    base_features = df.drop(columns=['Label']).columns.tolist()
    latent_features = [f'Autoencoder_Latent_{i}' for i in range(latent_train.shape[1])]
    hybrid_feature_names = base_features + latent_features + ['Reconstruction_MSE']
    
    importance = best_xgb.feature_importances_
    feat_imp = pd.DataFrame({'Feature': hybrid_feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Feature Importance (Tuned Hybrid Autoencoder+XGBoost)', fontsize=14)
    plt.xlabel('XGBoost Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('../models/plots/hybrid_feature_importance.png', dpi=300)
    plt.close()
    
    print("Hyperparameter tuning completed. All plots saved to models/plots/ directory!")

if __name__ == '__main__':
    main()
