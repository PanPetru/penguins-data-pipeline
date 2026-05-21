import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

df = pd.read_csv('palmerpenguins_original.csv').dropna()

features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
x_data = df[features]
y_data = LabelEncoder().fit_transform(df['species'])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_data)


def validate_model(model, x_val, y_val, name):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_val, y_val, cv=kfold)

    print(f"Walidacja {name} ---------------------------")
    print(f"Wyniki poszczególnych prób: {scores}")
    print(f"Średnia celność: {np.mean(scores):.3f}")
    print(f"Odchylenie standardowe: {np.std(scores):.3f}\n")
    return scores


knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = svm.SVC(kernel='linear')

validation_results = [
    validate_model(knn_model, x_scaled, y_data, "kNN"),
    validate_model(svm_model, x_scaled, y_data, "SVM")
]

plt.figure(figsize=(8, 6))
plt.boxplot(validation_results, tick_labels=['kNN', 'SVM'])
plt.title('Porównanie stabilności modeli (Cross-Validation)')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()