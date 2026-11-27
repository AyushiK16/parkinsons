import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("parkinsons_updrs.data")

print("=" * 60)
print("1. TARGET VARIABLE SUMMARY (total_UPDRS)")
print("=" * 60)
print(df['total_UPDRS'].describe())

print("\n" + "=" * 60)
print("2. PATIENT RECORDING COUNT DISTRIBUTION")
print("=" * 60)
recordings_per_patient = df['subject#'].value_counts()
print(f"Min recordings per patient: {recordings_per_patient.min()}")
print(f"Max recordings per patient: {recordings_per_patient.max()}")
print(f"Mean recordings per patient: {recordings_per_patient.mean():.1f}")

# Set plotting style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 5))

# Plot 1: Target Variable Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['total_UPDRS'], kde=True, color='teal', bins=30)
plt.title("Distribution of Total UPDRS Score", fontsize=12, fontweight='bold')
plt.xlabel("total_UPDRS Score")
plt.ylabel("Recording Count")

# Plot 2: total_UPDRS Range per Patient
plt.subplot(1, 2, 2)
sns.boxplot(x='subject#', y='total_UPDRS', data=df, color='lightblue')
plt.title("total_UPDRS Distribution Across 42 Patients", fontsize=12, fontweight='bold')
plt.xlabel("Subject ID (Patient)")
plt.ylabel("total_UPDRS Score")
plt.xticks(rotation=90, fontsize=8)

plt.tight_layout()
plt.savefig("eda_updrs_distribution.png", dpi=300)
print("\nPlot saved successfully as 'eda_updrs_distribution.png'!")
plt.show()