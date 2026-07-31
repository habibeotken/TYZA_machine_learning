"""
-UCI heart disease datasetini kullanarak lojistik regresyon modeli ile binary classification yapma
-model bir bireyin kalp hastalığı olup olmadığını tahmin etmeye çalışacak ve accuracy ile modelin performansını değerlendirecek

-veri seti yüke ve analizle yap
-veri seti içerisindeki eksik değerleri doldur veya temizle
-öznitelik ve hedef değişkeni ayır
-logistic regression modelini oluştur ve eğit
-modelin test veri seti üzerindeki performansını accuracy ile değerlendir
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

#veri setini yükleme ve temel analizleri yap

heart_disease = fetch_ucirepo(id=45)
df = pd.DataFrame(data = heart_disease.data.features)
df["targets"] = heart_disease.data.targets
df["targets"] = df["targets"].apply(lambda x: 1 if x == 1 else 0) #binary classification için hedef değişkeni 0 ve 1 olarak ayarla 
print(df.head())


#veri setindeki eksik değerleri doldurma veya temizleme

print("Temizlemeden önce eksik değer sayısı:", df.isna().sum().sum())

if df.isna().any().any():
    df = df.dropna() #eksik değerleri temizle
    print(f"veri setindeki eksik değerler temizlendi")
else:   
    print(f"veri setinde eksik değer bulunmamaktadır")

#öznitelik ve hedef değişkeni ayırma
X = df.drop("targets", axis=1).values #target sütununu düşürerek öznitelikleri al ki model tahmin yapabilsin. 
y = df.targets.values #df içindeki targets sütununu alır.Bunu NumPy dizisine çevirir.

#veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#logistic regression modelini oluşturma ve eğitme
log_reg_model = LogisticRegression(penalty='l2', C=1.0, max_iter=500)
log_reg_model.fit(X_train, y_train)

#modelin test veri seti üzerindeki performansını accuracy ile değerlendirme
accuracy = log_reg_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")


