import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('palmerpenguins_original.csv')
df = df.dropna()

features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]
y = LabelEncoder().fit_transform(df['species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print("Łączna zachowana wariancja PCA:")
print(np.sum(pca.explained_variance_ratio_))

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)


def evaluate(X_tr, X_te, y_tr, y_te, name):
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    svc = svm.SVC(kernel='linear').fit(X_tr, y_tr)

    knn_acc = accuracy_score(y_te, knn.predict(X_te))
    svc_acc = accuracy_score(y_te, svc.predict(X_te))

    return [name, f"{knn_acc:.3f}", f"{svc_acc:.3f}"]


results = []
results.append(evaluate(X_train_std, X_test_std, y_train, y_test, "Oryginał (4D)"))
results.append(evaluate(X_train_pca, X_test_pca, y_train, y_test, "Po PCA (2D)"))
results.append(evaluate(X_train_lda, X_test_lda, y_train, y_test, "Po LDA (2D)"))

res_df = pd.DataFrame(results, columns=["Metoda", "kNN Accuracy", "SVM Accuracy"])
print(res_df)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolors='k')
plt.title("Dane po PCA (Unsupervised)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis', edgecolors='k')
plt.title("Dane po LDA (Supervised)")
plt.xlabel("LD1")
plt.ylabel("LD2")

plt.tight_layout()
plt.show()

methods = res_df["Metoda"]

knn_scores = res_df["kNN Accuracy"].astype(float)
svm_scores = res_df["SVM Accuracy"].astype(float)

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, knn_scores, width, label='kNN')
plt.bar(x + width/2, svm_scores, width, label='SVM')

plt.xticks(x, methods)
plt.ylabel("Accuracy")
plt.title("Porównanie klasyfikatorów")
plt.legend()

plt.tight_layout()
plt.show()