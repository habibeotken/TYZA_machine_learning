"""
Göğüs kanseri veri seti üzerinde bir sınıflandrıma modeli eğitmek ve bunları lime ve shap kütüphaneleri ile açıklamak 


1-Veri setini yükle
2-test ve train olarak ayır
3-verinin özelliklerini ölçeklendir
4-random forest modelini eğit
5- model performansının değerlendirilmesi
6-Lime ile tek bir özellik açıklama
7-Shap ile özellik katkılarının açıklanması

"""
#kütüphaneleri import etme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


#veri setinin yüklenmesi
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

class_names = data.target_names
print(f"Veri seti boyutu: {X.shape}")
print("Sınıf isimleri:", class_names)

#veri setinin test ve train olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"eğitim veri boyutu: {X_train.shape}")
print(f"test veri boyutu: {X_test.shape}")

#veri setinin özelliklerini ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

#random forest modelini eğitme
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

#model performansının değerlendirilmesi
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluk oranı: {accuracy:.4f}")

#LİME ile tek bir özellik açıklama
sample_index = 0  # Açıklamak istediğiniz örnek indeksini seçin
sample = X_test_scaled[sample_index]
sample_original = X_test.iloc[sample_index]

prediction = rf_model.predict([sample])[0]
prediction_proba = rf_model.predict_proba([sample])[0]
print(f"açıklnacak örnek indeksi: {sample_original}")
print(f"Model tahmini: {class_names[prediction]} (olasılık: {prediction_proba[prediction]:.4f})")



lime_explainer = LimeTabularExplainer(X_train_scaled, feature_names=X.columns.to_list(), class_names=class_names, discretize_continuous=True, mode="classification")
lime_exp = lime_explainer.explain_instance(sample, rf_model.predict_proba, num_features=5)
print("LIME açıklaması:" )

for feature, contribution in lime_exp.as_list():
    print(f"{feature}: {contribution:.4f}") 

#SHAP ile özellik katkılarının açıklanması
shap_explainer = shap.TreeExplainer(rf_model)
shap_values = shap_explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):
    shap_values_class_1 = shap_values[1]
    expected_value_class_1 = shap_explainer.expected_value[1]
else:
    shap_values_class_1 = shap_values[:, :, 1]
    expected_value_class_1 = shap_explainer.expected_value[1]    

shap.summary_plot(shap_values_class_1, X_test_scaled_df)
plt.show()