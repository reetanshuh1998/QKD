import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../models/plots', exist_ok=True)
print("Loading Custom QKD Dataset for Masterpiece Plotting...")
df = pd.read_csv('../data/raw/custom_qkd_dataset.csv')

# Derived Features Calculation (Mathematically canceling transmission distance variance)
df['R_Q'] = (df['Q_nu'] + 1e-10) / (df['Q_mu'] + 1e-10)
df['Delta_Q_mu'] = df['Q_mu_Z'] - df['Q_mu_X']
df['Delta_E_mu'] = df['E_mu_Z'] - df['E_mu_X']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sns.set_theme(style='whitegrid')

# distinct 8-color palette
palette = sns.color_palette("tab10", 8)

# (A) Scatter Q_nu vs Q_mu (All classes)
sns.scatterplot(x='Q_mu', y='Q_nu', hue='Label', data=df, ax=axes[0, 0], alpha=0.5, palette=palette, s=25, legend='full')
axes[0, 0].set_title('(A) Decoy Consistency Scatter: $Q_{\\nu}$ vs $Q_{\\mu}$', fontsize=14)
axes[0, 0].set_xlabel('Signal Yield ($Q_{\\mu}$)', fontsize=12)
axes[0, 0].set_ylabel('Decoy Yield ($Q_{\\nu}$)', fontsize=12)

# Extract global handles for the grid and remove local legend to save space
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[0, 0].get_legend().remove()

# (B) KDE of R_Q (All classes)
# Log scale applied to visually isolate ratio magnitudes
sns.kdeplot(data=df, x='R_Q', hue='Label', ax=axes[0, 1], fill=True, common_norm=False, palette=palette, alpha=0.3, log_scale=True, legend=False)
axes[0, 1].set_title('(B) Decoy Yield Ratio KDE: $R_Q = Q_{\\nu} / Q_{\\mu}$', fontsize=14)
axes[0, 1].set_xlabel('Decoy/Signal Yield Ratio $R_Q$ (Log Scale)', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)

# (C) Scatter Rx_Power_Mean vs Monitor_Alarm_Rate (All classes)
sns.scatterplot(x='Rx_Power_Mean', y='Monitor_Alarm_Rate', hue='Label', data=df, ax=axes[1, 0], alpha=0.5, palette=palette, s=25, legend=False)
axes[1, 0].set_title('(C) Optical Power Intrusion Signature', fontsize=14)
axes[1, 0].set_xlabel('Receiver Optical Power Mean (Amplitude)', fontsize=12)
axes[1, 0].set_ylabel('Hardware Monitor Alarm Rate', fontsize=12)

# (D) Scatter Timing_Mean_us vs Timing_Std_us (All classes)
sns.scatterplot(x='Timing_Mean_us', y='Timing_Std_us', hue='Label', data=df, ax=axes[1, 1], alpha=0.5, palette=palette, s=25, legend=False)
axes[1, 1].set_title('(D) Active Interception Timing Signature', fontsize=14)
axes[1, 1].set_xlabel('Detection Arrival Timing Mean ($\\mu s$)', fontsize=12)
axes[1, 1].set_ylabel('Detection Arrival Timing Std. Variance ($\\mu s$)', fontsize=12)

# Improve overall layout formatting seamlessly
plt.suptitle('Mechanistic Attack Signatures across Physical Observables (All Attacks)', fontsize=18, fontweight='bold', y=0.98)
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.05, 1, 0.95], pad=2.0)
plt.savefig('../models/plots/attack_signatures_across_observables.png', dpi=300)
plt.close()
print("Masterpiece securely generated compiling ALL classes: models/plots/attack_signatures_across_observables.png")
