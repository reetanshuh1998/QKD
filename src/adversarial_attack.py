import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    print("--- Phase VI: Adversarial Machine Learning (The AI Eavesdropper) ---")
    
    # 1. Load Data to perfectly recreate the Test Subset boundaries
    df = pd.read_csv('../data/processed/qkd_processed.csv')
    X_raw = df.drop(columns=['Label']).values
    y = df['Label'].values
    
    # Duplicate the identical train_test_split to explicitly isolate the Test set
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Load Pretrained Core Utilities and Models
    try:
        scaler = joblib.load('../models/scaler.pkl')
        label_encoder = joblib.load('../models/label_encoder.pkl')
        xgb_model = joblib.load('../models/xgboost_hybrid.pkl')
        autoencoder = load_model('../models/autoencoder.keras', compile=False)
        encoder = load_model('../models/encoder.keras', compile=False)
    except Exception as e:
        print(f"Error loading models: {e}. Ensure prior scripts have been executed.")
        return
        
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Identify 'normal' label natively for dynamic adaptation
    normal_idx = label_encoder.transform(['normal'])[0]
    
    # 3. Filter Eve's Intercept Traffic exclusively
    eve_mask = (y_test != normal_idx)
    X_eve = X_test_scaled[eve_mask]
    y_eve_true = y_test[eve_mask]
    
    print(f"\nExtracted Eve's Explicit Intrusion Matrix: {X_eve.shape[0]} intercept vectors isolated.")
    
    # --- BEFORE EVASION (BENCHMARK) ---
    latent_eve_initial = encoder.predict(X_eve, verbose=0)
    recon_eve_initial = autoencoder.predict(X_eve, verbose=0)
    mse_eve_initial = np.mean(np.square(X_eve - recon_eve_initial), axis=1).reshape(-1, 1)
    
    X_hybrid_initial = np.hstack((X_eve, latent_eve_initial, mse_eve_initial))
    
    y_pred_before = xgb_model.predict(X_hybrid_initial)
    # Detection Rate = Total vectors that the model correctly flagged as NOT normal.
    detected_before = np.sum(y_pred_before != normal_idx)
    detection_rate_before = (detected_before / len(X_eve)) * 100
    
    print(f"BASELINE: The Hybrid Autoencoder cleanly caught {detection_rate_before:.2f}% of Eve's intrusions natively.")
    
    # --- AI EVASION (GRADIENT DESCEND ATTACK) ---
    print("\nInitializing Artificial Intelligence Gradient spoofing algorithms...")
    print("Eve is actively tuning thermodynamic intensities minimizing latent anomaly constraints...")
    
    X_adv_tensor = tf.Variable(X_eve, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.08)
    
    epochs = 40
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # We want to minimize the MSE. Eve forces her MSE to drop implicitly mimicking 'Normal' distributions!
            reconstructions = autoencoder(X_adv_tensor)
            # Calculating pure Mean Squared Error across physical bounds
            mse_loss = tf.reduce_mean(tf.square(X_adv_tensor - reconstructions), axis=1)
            
            # Constraint Penalty: Eve shouldn't warp dimensions she inherently cannot control globally 
            # (e.g. Sifted Key Length/Initial Key shouldn't radically shift out of thermodynamic laws)
            # Imposing absolute regularizations limits physics deviation bounds perfectly.
            # Dimensions 0 & 1 = Initial & Sifted length. 
            deviation_penalty = tf.reduce_mean(tf.square(X_adv_tensor[:, 0:2] - X_eve[:, 0:2])) * 0.5
            
            total_loss = tf.reduce_mean(mse_loss) + deviation_penalty
            
        gradients = tape.gradient(total_loss, X_adv_tensor)
        optimizer.apply_gradients([(gradients, X_adv_tensor)])
    
    print("Optimization completed!")
    
    # --- AFTER EVASION ---
    X_eve_adv_np = X_adv_tensor.numpy()
    
    latent_eve_adv = encoder.predict(X_eve_adv_np, verbose=0)
    recon_eve_adv = autoencoder.predict(X_eve_adv_np, verbose=0)
    mse_eve_adv = np.mean(np.square(X_eve_adv_np - recon_eve_adv), axis=1).reshape(-1, 1)
    
    X_hybrid_adv = np.hstack((X_eve_adv_np, latent_eve_adv, mse_eve_adv))
    
    y_pred_after = xgb_model.predict(X_hybrid_adv)
    # Detection Rate after Evasion
    detected_after = np.sum(y_pred_after != normal_idx)
    detection_rate_after = (detected_after / len(X_eve)) * 100
    
    print(f"ATTACK SUCCESS: After Eve inherently deployed Adversarial ML, the Hybrid Autoencoder's detection dropped to {detection_rate_after:.2f}%!")
    
    # --- Structural Visual Render ---
    print("\nRendering catastrophic detection drop metrics graphically...")
    scenarios = ['Pre-Evasion\n(Autoencoder Benchmark)', 'Post-Evasion\n(Adversarial Artificial Intelligence)']
    rates = [detection_rate_before, detection_rate_after]
    
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="darkgrid")
    ax = sns.barplot(x=scenarios, y=rates, palette="Reds_r")
    
    plt.ylim(0, 100)
    plt.ylabel('Eve Intrusion Detection Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Hybrid ML Vulnerability to Targeted Physical Adversarial Subsets', fontsize=13, pad=15)
    
    for i, v in enumerate(rates):
        ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=13)
        
    plt.tight_layout()
    import os
    os.makedirs('../models/plots', exist_ok=True)
    plt.savefig('../models/plots/adversarial_evasion.png', dpi=300)
    print("Saved Adversarial graphic correctly perfectly to models/plots/adversarial_evasion.png")

if __name__ == '__main__':
    main()
