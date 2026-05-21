import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('palmerpenguins_original.csv')

num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

df_clean = df[num_cols].dropna()

means = df_clean.mean()
variances = df_clean.var()
std_devs = df_clean.std()
ranges = df_clean.max() - df_clean.min()

cov_matrix = df_clean.cov()
corr_matrix = df_clean.corr()

print("Basic Descriptive Statistics:")
print(pd.DataFrame({'Mean': means, 'Variance': variances, 'Range': ranges}))

print("\nStandard Deviation:\n")
print(std_devs)

print("\nCovariance Matrix:\n")
print(cov_matrix)

print("\nCorrelation Matrix:\n")
print(corr_matrix)

# Min-Max Scaling: (x - min) / (max - min)
df_minmax = (df_clean - df_clean.min()) / (df_clean.max() - df_clean.min())

# Standardization: (x - mean) / std
df_std = (df_clean - df_clean.mean()) / df_clean.std()

# L2 Normalization: x / sqrt(sum(x^2))
l2_norms = np.sqrt((df_clean ** 2).sum(axis=1))
df_norm_l2 = df_clean.div(l2_norms, axis=0)

# L1 Normalization: x / sum(|x|)
l1_norms = df_clean.abs().sum(axis=1)
df_norm_l1 = df_clean.div(l1_norms, axis=0)

def compare_plots(original, transformed, method_name, col):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(original[col], bins=20, color='blue', alpha=0.6)
    plt.title(f'Original: {col}')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(transformed[col], bins=20, color='red', alpha=0.6)
    plt.title(f'{method_name}: {col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.show()

methods = [
    (df_minmax, "Min-Max Scaling"),
    (df_std, "Standardization"),
    (df_norm_l1, "L1 Normalization"),
    (df_norm_l2, "L2 Normalization")
]

for data, name in methods:
    compare_plots(df_clean, data, name, 'body_mass_g')

# Summary statistics of the cleaned DataFrame
print(df_clean.describe().T)
