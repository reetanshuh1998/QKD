"""
Feature engineering for QKD attack detection.

Responsibilities (intentionally minimal to prevent data leakage):
  1. Load raw CSV from the simulator.
  2. Drop Distance_km (analysis-only column, not an ML feature).
  3. Encode string labels to integers via LabelEncoder.
  4. Save raw (UNSCALED) features + encoded labels to processed CSV.

StandardScaler fitting is deliberately deferred to model_training.py,
where it is applied exclusively on the training split.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def main():
    print("Loading raw dataset...")
    df = pd.read_csv('../data/raw/custom_qkd_dataset.csv')

    # Separate features and target
    drop_cols = ['Label']
    if 'Distance_km' in df.columns:
        drop_cols.append('Distance_km')
    X = df.drop(columns=drop_cols)

    y = df['Label']

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save the label mapping for reference
    label_mapping = dict(zip(label_encoder.classes_,
                             label_encoder.transform(label_encoder.classes_)))
    print("Label Mapping:", label_mapping)
    os.makedirs('../models', exist_ok=True)
    joblib.dump(label_encoder, '../models/label_encoder.pkl')

    # Save raw (unscaled) features + encoded labels.
    # Scaling is done post-split in model_training.py to prevent data leakage.
    print("Saving processed dataset (raw features, no scaling)...")
    processed_df = X.copy()
    processed_df['Label'] = y_encoded

    os.makedirs('../data/processed', exist_ok=True)
    processed_df.to_csv('../data/processed/qkd_processed.csv', index=False)
    print(f"Processed dataset saved: {processed_df.shape[0]} rows × {processed_df.shape[1]} cols")
    print("Feature engineering completed successfully.")


if __name__ == '__main__':
    main()

