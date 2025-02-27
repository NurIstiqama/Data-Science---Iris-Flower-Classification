import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset Iris dari scikit-learn
iris = datasets.load_iris()

# Konversi menjadi DataFrame agar lebih mudah dipahami
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Mapping angka target ke nama spesies
df['target'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Tampilkan 5 data pertama
print(df.head())

# Cek jumlah baris dan kolom
print("Jumlah baris dan kolom:", df.shape)

# Cek informasi dataset
print(df.info())

# Cek jumlah masing-masing kelas target
print(df['target'].value_counts())


# Pairplot untuk melihat hubungan antar fitur
sns.pairplot(df, hue='target', markers=["o", "s", "D"])
plt.show()


# Heatmap untuk melihat korelasi antar fitur
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi Antar Fitur")
plt.show()

# Pisahkan fitur (X) dan target (y)
X = iris.data
y = iris.target

# Bagi data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cek jumlah data training dan testing
print(f"Jumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")

# Standarisasi fitur agar model bekerja lebih baik
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Buat model KNN dengan k=3
model = KNeighborsClassifier(n_neighbors=3)

# Latih model dengan data training
model.fit(X_train, y_train)

# Prediksi data testing
y_pred = model.predict(X_test)

# Cek akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy:.2f}")


# Tampilkan classification report
print("Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))

# Tampilkan confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()