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

print("Podstawowe statystyki:")
print(pd.DataFrame({'Średnia': means, 'Wariancja': variances, 'Zakres': ranges}))

print("\nOdchylenie standardowe:\n")
print(std_devs)

print("\nMacierz kowariancji:\n")
print(cov_matrix)

print("\nMacierz korelacji:\n")
print(corr_matrix)

#SKALOWANIE MIN-MAX: (x - min) / (max - min)
df_minmax = (df_clean - df_clean.min()) / (df_clean.max() - df_clean.min())

#STANDARYZACJA: (x - mean) / std
df_std = (df_clean - df_clean.mean()) / df_clean.std()

# NORMALIZACJA (wierszowa - każda próbka/pingwin osobno)
# Normalizacja L2: x / sqrt(sum(x^2))
l2_norms = np.sqrt((df_clean ** 2).sum(axis=1))
df_norm_l2 = df_clean.div(l2_norms, axis=0)

# Normalizacja L1: x / sum(|x|)
l1_norms = df_clean.abs().sum(axis=1)
df_norm_l1 = df_clean.div(l1_norms, axis=0)

#Funkcja do wizualizacji
def compare_plots(original, transformed, method_name, col):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(original[col], bins=20, color='blue', alpha=0.6)
    plt.title(f'Oryginał: {col}')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(transformed[col], bins=20, color='red', alpha=0.6)
    plt.title(f'{method_name}: {col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.show()

#Porównanie rozkładów przed i po transformacji
methods = [
    (df_minmax, "Min-Max"),
    (df_std, "Standaryzacja"),
    (df_norm_l1, "Normalizacja L1"),
    (df_norm_l2, "Normalizacja L2")
]

for data, name in methods:
    compare_plots(df_clean, data, name, 'body_mass_g')

#Statystyki opisowe wyczyszczonego DataFrame'u
print(df_clean.describe().T)
