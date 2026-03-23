import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

def main():
    print("Loading raw dataset...")
    df = pd.read_csv('../data/raw/custom_qkd_dataset.csv')

    # Separate features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save the label mapping for reference
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label Mapping:", label_mapping)
    joblib.dump(label_encoder, '../models/label_encoder.pkl')

    print("Scaling numerical features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert scaled features back to a DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    joblib.dump(scaler, '../models/scaler.pkl')

    print("Extracting Feature Importance using XGBoost...")
    # Train a quick XGBoost model just to get feature importance
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb.fit(X_scaled_df, y_encoded)
    
    importance = xgb.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='magma')
    plt.title('XGBoost Initial Feature Importance')
    plt.tight_layout()
    plt.savefig('../notebooks/feature_importance.png')
    print("Saved feature importance plot to notebooks/feature_importance.png")

    print("Saving processed dataset...")
    # FIX: Export strictly RAW features to prevent Data Leakage. Scaling MUST happen post train-test-split.
    processed_df = X.copy()
    processed_df['Label'] = y_encoded
    
    processed_df.to_csv('../data/processed/qkd_processed.csv', index=False)
    print("Processed dataset saved to data/processed/qkd_processed.csv")
    print("Feature Engineering completed successfully.")

if __name__ == '__main__':
    main()
