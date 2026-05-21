import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm

df = pd.read_csv('palmerpenguins_original.csv')
df = df.dropna()

features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("-"*100)
print("STROJENIE KNN")
for k in range(1, 11):
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(X_train, y_train)
    pred_tmp = knn_tmp.predict(X_test)
    print(f"k={k}, accuracy={accuracy_score(y_test, pred_tmp):.3f}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("-"*100)
print("\nWYNIKI KNN")
print(f"Accuracy: {accuracy_score(y_test, knn_pred):.3f}")
print(classification_report(y_test, knn_pred, target_names=label_encoder.classes_))
print("Confusion Matrix KNN:")
print(confusion_matrix(y_test, knn_pred))

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
svm_pred = clf.predict(X_test)

print("-"*100)
print("\nWYNIKI SVM")
print(f"Accuracy: {accuracy_score(y_test, svm_pred):.3f}")
print(classification_report(y_test, svm_pred, target_names=label_encoder.classes_))
print("Confusion Matrix SVM:")
print(confusion_matrix(y_test, svm_pred))

print("-"*100)
print("\nPORÓWNANIE MODELI")
print(f"KNN Accuracy: {accuracy_score(y_test, knn_pred):.3f}")
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.3f}")

def plot_boundaries(model, X_data, y_data, title):
    X_vis = X_data[:, :2]

    h = .02
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    model.fit(X_vis, y_data)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_data, edgecolors='k')
    plt.title(title)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plot_boundaries(KNeighborsClassifier(n_neighbors=5), X_train, y_train, "KNN")

plt.subplot(1, 2, 2)
plot_boundaries(svm.SVC(kernel='linear'), X_train, y_train, "SVM")

plt.tight_layout()
plt.show()