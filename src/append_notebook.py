import nbformat as nbf
import os

nb_path = '../notebooks/01_EDA.ipynb'
with open(nb_path, 'r') as f:
    nb = nbf.read(f, as_version=4)

code = """
# Compare 'normal' vs specific attacks using KDE (Density) plots for key features
key_features = ['QBER', 'Measurement_entropy', 'Arrival_var']
    
plt.figure(figsize=(18, 5))
for i, feature in enumerate(key_features):
    plt.subplot(1, 3, i+1)
    
    # Plot 'normal' distribution
    sns.kdeplot(data=df[df['Label'] == 'normal'], x=feature, label='Normal', color='blue', fill=True, linewidth=2)
    
    # Plot a few distinct attacks to show the separation
    sns.kdeplot(data=df[df['Label'] == 'pns_attack'], x=feature, label='PNS Attack', color='red', linestyle='--', linewidth=2)
    sns.kdeplot(data=df[df['Label'] == 'mitm_attack'], x=feature, label='MITM Attack', color='orange', linestyle=':', linewidth=2)
    sns.kdeplot(data=df[df['Label'] == 'detector_blinding_attack'], x=feature, label='Blinding Attack', color='purple', linestyle='-.', linewidth=2)
    
    plt.title(f'{feature}: Normal vs Specific Attacks')
    plt.legend()
plt.tight_layout()
plt.show()

# Also a grouped boxplot specifically isolating Normal vs all Attacks
df['Aggregated_Label'] = df['Label'].apply(lambda x: 'Normal' if x == 'normal' else 'Attack')
plt.figure(figsize=(12, 6))
# To standardize scales visually, let's normalize the 3 key features for this plot only
df_temp_melt = df[['Aggregated_Label'] + key_features].copy()
for col in key_features:
    df_temp_melt[col] = (df_temp_melt[col] - df_temp_melt[col].mean()) / df_temp_melt[col].std()

sns.boxplot(data=df_temp_melt.melt(id_vars=['Aggregated_Label'], value_vars=key_features), 
            x='variable', y='value', hue='Aggregated_Label', palette='Set2')
plt.title('Z-Score Normalized Key Features: Normal vs. All Attacks (Aggregated)')
plt.ylabel('Standard Deviations from Mean')
plt.xlabel('Features')
plt.tight_layout()
plt.show()
"""

markdown_cell = nbf.v4.new_markdown_cell("## 4. Normal vs. Attack Comparison (Density & Aggregation)\nTo visibly understand why the classification ROC AUC reaches 1.000 for attacks like PNS and Detector Blinding, we can plot the feature density distributions. Notice how perfectly separated the peaks for attacks are compared to the **Normal** traffic signal.")
code_cell = nbf.v4.new_code_cell(code)

nb['cells'].extend([markdown_cell, code_cell])

with open(nb_path, 'w') as f:
    nbf.write(nb, f)
print("Successfully appended comparative feature plotting to 01_EDA.ipynb")
