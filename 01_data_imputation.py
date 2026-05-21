import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('palmerpenguins_original.csv')

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

print("\nComparison -------------------------------------------------------")
comparison = pd.DataFrame({
    'Original': df[target_col].describe(),
    'Mean': df_mean[target_col].describe(),
    'Median': df_median[target_col].describe(),
    'Mode': df_mode[target_col].describe(),
    'Grouped': df_other[target_col].describe()
})
print(comparison)

print("\nNumber of missing values after imputation:")
print(df_other[target_col].isnull().sum())

plt.figure(figsize=(10, 6))

sns.kdeplot(df[target_col], label='Original', color='black', lw=2)
sns.kdeplot(df_mean[target_col], label='Mean Imputation', ls='--')
sns.kdeplot(df_other[target_col], label='Grouped Imputation', ls='-.')

plt.title('Impact of Imputation on bill_length_mm Distribution')
plt.legend()
plt.show()
