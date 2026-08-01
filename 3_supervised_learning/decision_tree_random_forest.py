"""
-İris veri seti kullanılarak karar ağacı ve rastgele orman algoritmaları geliştirme
-karar ağaçları ile görselleştirme ve öznitelik önemini değerlendirme(feature selection)


1-veri setini yükle
2-feature ve targetleri ayır
3-eğitim ve test veri setlerini ayır
4-karar ağacı ve rastgele orman modellerini oluştur ve eğit
5-test verisi ile modelin performansını değerlendir
6-model başarısını accuracy ile değerlendirme
7-doğruluk oranı ve confusion matrix ile modelin performansını değerlendir
8-karar ağacının görselleştirilmesi
9-karar ağacının feature importance değerlerini görselleştir 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import decision_tree_random_forest as dt_rf


#veri setini yükleme
iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

print(df.head(10))

#feature ve targetleri ayırma
X = iris.data
y = iris.target

#eğitim ve test veri setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#karar ağacı ve rastgele orman modellerini oluşturma ve eğitme
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='gini')
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=2)

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

#test verisi ile modelin performansını değerlendirme
dt_y_pred = dt_model.predict(X_test)
rf_y_pred = rf_model.predict(X_test)


#model başarısını accuracy ile değerlendirme
dt_accuracy = accuracy_score(y_test, dt_y_pred)
rf_accuracy = accuracy_score(y_test, rf_y_pred)

print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

#doğruluk oranı ve confusion matrix ile modelin performansını değerlendirme

conf_matrix_dt = confusion_matrix(y_test, dt_y_pred)

plt.figure()
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#karar ağacının görselleştirilmesi

plt.figure()
plot_tree(dt_model, feature_names=iris.feature_names, class_names=list(iris.target_names))

plt.title("Decision Tree Visualization")
plt.show()

#feature importance değerleri

feature_importances = dt_model.feature_importances_
feature_names = iris.feature_names

#feature değerlerini sıralama ve görselleştirme

for name in feature_names:
    importance = feature_importances[feature_names.index(name)]
    print(f"Feature: {name}, Importance: {importance}")

plt.figure()
plt.barh(feature_names, feature_importances)
plt.title("Feature Importance in Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Features")
