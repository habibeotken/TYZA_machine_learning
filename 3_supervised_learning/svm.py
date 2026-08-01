"""
- Digits veri seti kullanarak SVM ile birlikte çok sınıflı bir sınıflandırma problemi çözme

"""

"""
1-veri setini yükle
2-veri setini görselleştir
3-feature ve targetleri ayır
4-eğitim ve test setlerini oluştur
5-svm modelini oluştur ve eğit 
6-modeli test set ile değerlendir
7-test verisi üzerinden tahmin yap
8-model performansını sınıflandırma raporu ile değerlendir

"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#veri setini yükleme
digits = load_digits()
print(digits.DESCR)

#veri setini görselleştirme
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray', interpolation='nearest')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')

plt.show()

#feature ve targetleri ayırma
X = digits.data
y = digits.target   

#eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#svm modelini oluşturma ve eğitme
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

#modeli test set ile değerlendirme
y_pred = svm_model.predict(X_test)  


#model performansını sınıflandırma raporu ile değerlendirme
cls_report =classification_report(y_test, y_pred)   
print(cls_report)