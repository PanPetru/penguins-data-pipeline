import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = sns.load_dataset('penguins')

target_col = 'bill_length_mm'

df_mean = df.copy()
df_median = df.copy()
df_mode = df.copy()
df_other = df.copy()

df_mean[target_col] = df[target_col].fillna(df[target_col].mean())
df_median[target_col] = df[target_col].fillna(df[target_col].median())
df_mode[target_col] = df[target_col].fillna(df[target_col].mode()[0])
df_other[target_col] = df.groupby('species')[target_col].transform(lambda x: x.fillna(x.mean()))

print(df_mean[target_col])
print(df_median[target_col])
print(df_mode[target_col])

print("\nPorównanie -------------------------------------------------------")
comparison = pd.DataFrame({
    'Oryginał': df[target_col].describe(),
    'Średnia': df_mean[target_col].describe(),
    'Mediana': df_median[target_col].describe(),
    'Najczęstsza': df_mode[target_col].describe(),
    'Grupowa': df_other[target_col].describe()
})
print(comparison)

print("\nLiczba braków po imputacji:")
print(df_other[target_col].isnull().sum())

plt.figure(figsize=(10, 6))

# Porównujemy oryginał, średnią i metodę grupową (najciekawsze różnice)
sns.kdeplot(df[target_col], label='Oryginał', color='black', lw=2)
sns.kdeplot(df_mean[target_col], label='Imputacja średnią', ls='--')
sns.kdeplot(df_other[target_col], label='Imputacja grupową', ls='-.')

plt.title('Wpływ imputacji na rozkład bill_length_mm')
plt.legend()
plt.show()