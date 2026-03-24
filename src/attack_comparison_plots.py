import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure plots directory exists
os.makedirs('../models/plots', exist_ok=True)

print("Loading raw data...")
df = pd.read_csv('../data/raw/custom_qkd_dataset.csv')
key_features = ['Q_mu', 'E_mu', 'Rx_Power_Mean']
    
plt.figure(figsize=(18, 5))
for i, feature in enumerate(key_features):
    plt.subplot(1, 3, i+1)
    
    # Plot 'normal' distribution
    sns.kdeplot(data=df[df['Label'] == 'normal'], x=feature, label='Normal', color='blue', fill=True, linewidth=2)
    
    # Plot distinct attacks
    sns.kdeplot(data=df[df['Label'] == 'pns_attack'], x=feature, label='PNS Attack', color='red', linestyle='--', linewidth=2)
    sns.kdeplot(data=df[df['Label'] == 'mitm_attack'], x=feature, label='MITM Attack', color='orange', linestyle=':', linewidth=2)
    sns.kdeplot(data=df[df['Label'] == 'detector_blinding_attack'], x=feature, label='Blinding Attack', color='purple', linestyle='-.', linewidth=2)
    
    plt.title(f'{feature}: Normal vs Attacks')
    plt.legend()
plt.tight_layout()
plt.savefig('../models/plots/attack_comparison_density.png', dpi=300)
plt.close()
print("Saved attack_comparison_density.png")

# Aggregated boxplot
df['Aggregated_Label'] = df['Label'].apply(lambda x: 'Normal' if x == 'normal' else 'Attack')
plt.figure(figsize=(12, 6))

df_temp_melt = df[['Aggregated_Label'] + key_features].copy()
for col in key_features:
    df_temp_melt[col] = (df_temp_melt[col] - df_temp_melt[col].mean()) / df_temp_melt[col].std()

sns.boxplot(data=df_temp_melt.melt(id_vars=['Aggregated_Label'], value_vars=key_features), 
            x='variable', y='value', hue='Aggregated_Label', palette='Set2')
plt.title('Z-Score Normalized Key Features: Normal vs. All Attacks (Aggregated)')
plt.ylabel('Standard Deviations from Mean')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('../models/plots/attack_comparison_boxplot.png', dpi=300)
plt.close()
print("Saved attack_comparison_boxplot.png")
