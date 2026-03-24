import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from tensorflow.keras.models import load_model

os.makedirs('../models/plots', exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("Loading data and models for core publication plots...")
    
    # 1. Load Raw Data
    df_raw = pd.read_csv('../data/raw/custom_qkd_dataset.csv')
    
    distance = df_raw['Distance_km'].values
    y_raw = df_raw['Label'].values
    
    label_encoder = joblib.load('../models/label_encoder.pkl')
    y_encoded = label_encoder.transform(y_raw)
    class_names = label_encoder.classes_
    
    drop_cols = ['Label', 'Distance_km'] if 'Distance_km' in df_raw.columns else ['Label']
    X_raw = df_raw.drop(columns=drop_cols)
    feature_names_base = X_raw.columns.tolist()
    
    # Stratified Split identical to training pipeline preventing data leak
    idx = np.arange(len(df_raw))
    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X_raw.values, y_encoded, idx, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    distance_test = distance[idx_test]
    
    # Prepare scaled features on existing scaler
    scaler = joblib.load('../models/scaler.pkl')
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Load Models
    autoencoder = load_model('../models/autoencoder.keras')
    encoder = load_model('../models/encoder.keras')
    
    # Target tuned model, default backward if missing
    try:
        xgb_model = joblib.load('../models/xgboost_hybrid_tuned.pkl')
    except Exception:
        xgb_model = joblib.load('../models/xgboost_hybrid.pkl')
        
    latent_test = encoder.predict(X_test_scaled, verbose=0)
    test_reconst = autoencoder.predict(X_test_scaled, verbose=0)
    mse_test = np.mean(np.square(X_test_scaled - test_reconst), axis=1) # Shape (N,)
    
    # AE Output evaluation dataframe
    df_test_results = pd.DataFrame({
        'Distance_km': distance_test,
        'Label_Idx': y_test,
        'Label': label_encoder.inverse_transform(y_test),
        'AE_MSE': mse_test
    })
    
    # Recreate the 33-dimensional hybrid pipeline input matrix natively
    X_test_hybrid = np.hstack((X_test_scaled, latent_test, mse_test.reshape(-1, 1)))
    y_pred = xgb_model.predict(X_test_hybrid)
    df_test_results['Predicted_Idx'] = y_pred
    
    # ==========================================
    # 1. Gain vs distance_normal.png
    # ==========================================
    print("Generating gain_vs_distance_normal.png...")
    df_normal = df_raw[df_raw['Label'] == 'normal']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Distance_km', y='Q_mu', data=df_normal, label='Signal (Q_mu)', alpha=0.5, color='blue')
    sns.scatterplot(x='Distance_km', y='Q_nu', data=df_normal, label='Decoy (Q_nu)', alpha=0.5, color='orange')
    sns.scatterplot(x='Distance_km', y='Q_0', data=df_normal, label='Vacuum (Q_0)', alpha=0.5, color='green')
    plt.yscale('log')
    plt.title('Session-level gains decay with distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Detection Yield (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('../models/plots/gain_vs_distance_normal.png', dpi=300)
    plt.close()
    
    # ==========================================
    # 2. decoy_consistency_qnu_vs_qmu.png
    # ==========================================
    print("Generating decoy_consistency_qnu_vs_qmu.png...")
    mask_att = df_raw['Label'].isin(['normal', 'pns_attack', 'combined_attack'])
    df_att = df_raw[mask_att]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Q_mu', y='Q_nu', hue='Label', data=df_att, palette='Set1', alpha=0.6)
    plt.title('Decoy Consistency: Normal vs PNS Distortion')
    plt.xlabel('Signal Yield (Q_mu)')
    plt.ylabel('Decoy Yield (Q_nu)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../models/plots/decoy_consistency_qnu_vs_qmu.png', dpi=300)
    plt.close()

    # ==========================================
    # 3. ae_mse_distribution_by_class.png
    # ==========================================
    print("Generating ae_mse_distribution_by_class.png...")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Label', y='AE_MSE', data=df_test_results, palette='Set2')
    plt.yscale('log')
    plt.title('AE Reconstruction Error (MSE) by Class')
    plt.xlabel('Attack Label')
    plt.ylabel('Mean Squared Error (Log Scale)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../models/plots/ae_mse_distribution_by_class.png', dpi=300)
    plt.close()
    
    # ==========================================
    # 4. macro_f1_vs_distance_bins.png
    # ==========================================
    print("Generating macro_f1_vs_distance_bins.png...")
    bins = [5, 15, 25, 35, 50]
    labels = ['5-15', '15-25', '25-35', '35-50']
    df_test_results['Dist_Bin'] = pd.cut(df_test_results['Distance_km'], bins=bins, labels=labels, include_lowest=True)
    
    f1_scores = []
    for b in labels:
        subset = df_test_results[df_test_results['Dist_Bin'] == b]
        if len(subset) > 0:
            f1 = f1_score(subset['Label_Idx'], subset['Predicted_Idx'], average='macro')
            f1_scores.append(f1)
        else:
            f1_scores.append(np.nan)
            
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=labels, y=f1_scores, marker='o', color='purple', linewidth=2)
    plt.title('Model maintains stable performance across metro distances')
    plt.xlabel('Distance Bins (km)')
    plt.ylabel('Macro-F1 Score')
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../models/plots/macro_f1_vs_distance_bins.png', dpi=300)
    plt.close()
    
    # ==========================================
    # 5. recall_heatmap_vs_distance.png
    # ==========================================
    print("Generating recall_heatmap_vs_distance.png...")
    recall_matrix = np.zeros((len(class_names), len(labels)))
    
    for i, c_name in enumerate(class_names):
        c_idx = label_encoder.transform([c_name])[0]
        subset_class = df_test_results[df_test_results['Label_Idx'] == c_idx]
        
        for j, b in enumerate(labels):
            subset_bin = subset_class[subset_class['Dist_Bin'] == b]
            if len(subset_bin) > 0:
                rec = recall_score(subset_bin['Label_Idx'], subset_bin['Predicted_Idx'], labels=[c_idx], average='macro', zero_division=0)
                recall_matrix[i, j] = rec
            else:
                recall_matrix[i, j] = np.nan
                
    plt.figure(figsize=(10, 8))
    sns.heatmap(recall_matrix, annot=True, fmt='.2f', cmap='YlGnBu', 
                xticklabels=labels, yticklabels=class_names, vmin=0, vmax=1)
    plt.title('Per-Class Recall vs Distance Bins')
    plt.xlabel('Distance Bins (km)')
    plt.ylabel('Attack Label')
    plt.tight_layout()
    plt.savefig('../models/plots/recall_heatmap_vs_distance.png', dpi=300)
    plt.close()
    
    print("Done! Core publication plots safely saved within models/plots/")

if __name__ == '__main__':
    main()
