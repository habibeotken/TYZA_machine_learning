"""
Göğüs kanseri veri setini kullanarak KNN algoritması ile  classification yapma
Modelin doğruluk oranını hesapla ,farklı K değerleri için hiperparametre araması yapma


1-veri setini yükle
2-feature ve targetleri ayır
3-eğitim ve test veri setlerini ayır
4-öznitelikleri ölçeklendir
5-KNN modelini oluştur ve eğit
6-doğruluk oranı ve confusion matrix ile modelin performansını değerlendir
7-hiperparametre ayarlaması
8-sonuçların grafiksel olarak gösterilmesi

"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

#veri setini yükleme
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target
print (df.head())


#feature ve targetleri ayırma

X = cancer.data
y = cancer.target

#eğitim ve test veri setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#öznitelikleri ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#KNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)

#modelin test veri seti üzerindeki performansını değerlendirme
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

#KNN eğitimi 


knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print (f"Accuracy for k=12: {accuracy}")


#7-hiperparametre ayarlaması 8-sonuçların grafiksel olarak gösterilmesi
k_accuracy = []
k_values =[]


for k in range(3,15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
 
    k_accuracy.append(accuracy_score(y_pred, y_test))
    k_values.append(k)

plt.plot(k_values, k_accuracy)
plt.show()