"""
Verilen müşteri özelliklerinden müşterilerin ayrılıp ayrılmayacağını tahmin eden bir model oluşturma


Veri seti:
Veri setinde yaş, aylık gelir, eğitim, deneyim, şirkette çalışma süresi, performans, fazla mesai, izin kullanımı, iş memnuniyeti, iş-yaşam dengesi, terfi ve işe uzaklık gibi özellikler var. Hedef sütunumuz isten_ayrildi.




Kod Akışı:
1- Veri setini yükleme
2-Veri setindeki eksik değerleri kontrol etme ve gerekli işlemleri yapma
3-Veri setindeki kategorik değişkenleri one-hot encoding yöntemi ile sayısal verilere çevirme
4-Sayısal değişkenleri ölçeklendirme
5-Öznitelik üretme ve seçme
6-Veri setini train, test ve validation olarak ayırma
9- Logistic Regression, KNN ve Decision Tree modellerini oluşturma ve eğitme
10-Validation sonuçlarına göre modelleri karşılaştırma ve en iyi modeli seçme
11-Seçilen modeli test veri seti üzerinde değerlendirme
12- Test set için confusion matrix ,accuracy, precision, recall ve f1 score değerlerini hesaplama
13-Hangi modelin daha iyi performans gösterdiğini belirleme ve sonuçları yorumlama


"""

# Gerekli kütüphaneleri yükleme
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



# Veri setini yükleme
data = pd.read_csv('calisan_isten_ayrilma_100_eksik_verili.csv')
print(data.head())


# Eksik değerleri kontrol etme
print(f"Eksik değerler:\n{data.isnull().sum()}")

sayisal_sutunlar = [
    "yas",
    "performans_puani",
    "aylik_fazla_mesai_saati",
    "is_memnuniyeti",
    "is_yasam_dengesi",
    "son_terfiden_beri_yil",
    "ise_uzaklik_km",
    "aylik_gelir",
    "toplam_deneyim_yili",
    "sirkette_calisma_yili",
   
]

# Sayısal sütunlardaki eksik değerleri ortalama ile doldurma 
for sutun in sayisal_sutunlar:
    ortalama_degeri = data[sutun].mean()
    data[sutun] = data[sutun].fillna(ortalama_degeri)
    
    print(f"{sutun} -> doldurulan değer: {ortalama_degeri:.2f}")

#kalan eksik değerleri silme
print(f"Eksik değerler:\n{data.isnull().sum()}")
data = data.dropna()
print(f"\nEksik değerler silindikten sonra veri seti boyutu: {data.shape}")



# Kategorik değişkenleri one-hot encoding yöntemi ile sayısal verilere çevirme
label_encoder = OneHotEncoder( drop='first', sparse_output=False)
categorical_features = ['egitim','fazla_mesai']
y = label_encoder.fit_transform(data[categorical_features])
print(f"\none-hot encoding ile dönüştürülen kategorik değişkenler:\n{y[:5]}")






# Sayısal değişkenleri ölçeklendirme
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[sayisal_sutunlar])

# Öznitelik üretme ve seçme


# Yeni öznitelik üretme: şirkette kalma oranı
data["sirkette_kalma_orani"] = (
    data["sirkette_calisma_yili"] /
    (data["toplam_deneyim_yili"] + 1)
)





X = pd.concat([pd.DataFrame(X_numerical, columns=sayisal_sutunlar), pd.DataFrame(y, columns=label_encoder.get_feature_names_out(categorical_features))], axis=1)
y = data['isten_ayrildi']
print(y.value_counts())
sayisal_sutunlar.append("sirkette_kalma_orani")
#eklenen öznitekileri gösterme
print(f"\n şirkette kalma oranını görüntüleme:\n{data[['sirkette_kalma_orani']].head()}")

# Veri setini train, test ve validation olarak ayırma
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Logistic Regression modelini oluşturma ve eğitme
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# KNN modelini oluşturma ve eğitme
knn_model = KNeighborsClassifier(n_neighbors=9)
knn_model.fit(X_train, y_train)

#knn hiperparametrelerinden optimum k değerini bulmak 

for k in range(1,11):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_val_pred = knn_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    plt.scatter(k, accuracy, color='blue',marker='x')
plt.title("KNN Modeli için Doğruluk Değerleri")
plt.xlabel("k Değeri")
plt.ylabel("Doğruluk")
plt.show()  



# Decision Tree modelini oluşturma ve eğitme
decision_tree_model = DecisionTreeClassifier(random_state=42,max_depth=2,min_samples_split=15)
decision_tree_model.fit(X_train, y_train)


# Validation sonuçlarına göre modelleri karşılaştırma ve en iyi modeli seçme
models = {
    'Logistic Regression': logistic_model,
    'KNN': knn_model,
    'Decision Tree': decision_tree_model
}

accuracy_scores = {}
for model_name, model in models.items():
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    accuracy_scores[model_name] = accuracy
    print(f"{model_name} Validation Accuracy: {accuracy:.4f}")

# En iyi modeli seçme
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
print(f"\nEn iyi model: {best_model_name} with accuracy: {accuracy_scores[best_model_name]:.4f}")

# Seçilen modeli test veri seti üzerinde değerlendirme
best_model = models[best_model_name]
y_test_pred = best_model.predict(X_test)

# Test set için confusion matrix ,accuracy, precision, recall ve f1 score değerlerini hesaplama
conf_matrix = confusion_matrix(y_test, y_test_pred)
conf_matrix_table = pd.DataFrame(
    conf_matrix,
    index=["Gerçek: Hayır", "Gerçek: Evet"],
    columns=["Tahmin: Hayır", "Tahmin: Evet"],
)

accuracy = accuracy_score(y_test, y_test_pred)

precision = precision_score(
    y_test, y_test_pred, pos_label="Evet"
)
recall = recall_score(
    y_test, y_test_pred, pos_label="Evet"
)
f1 = f1_score(
    y_test, y_test_pred, pos_label="Evet"
)

print(
    f"\nTest Set Metrikleri:\nConfusion Matrix:\n{conf_matrix_table}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}"
)

# Hangi modelin daha iyi performans gösterdiğini belirleme ve sonuçları yorumlama
# En iyi performans gösteren modeli yorumlama
print(f"{best_model_name} modeli accuracy sonuçlarına göre en iyi performans gösteren qmodeldir.")
