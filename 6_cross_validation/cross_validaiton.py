"""
K-fold , Stratified K-fold, Leave-One-Out Cross-Validation 

1-veri setini yükle
2-basit bir sınıflandırma modeli oluştur
3-K-fold ile veri setini K parçaya böl
4-Stratified K-fold ile veri setini K parçaya böl
5-Leave-One-Out Cross-Validation ile veri setini K parçaya böl
6-Sonuçları yazdır
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.datasets import load_iris
from sklearn import tree as decision_tree


# 1-veri setini yükle
data = load_iris()
X, y = data.data, data.target

# 2-basit bir sınıflandırma modeli oluştur
model = decision_tree.DecisionTreeClassifier()


# 3-K-fold ile veri setini K parçaya böl
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model, X, y, cv=kfold)

# 4-Stratified K-fold ile veri setini K parçaya böl
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
stratified_kfold_scores = cross_val_score(model, X, y, cv=stratified_kfold)

# 5-Leave-One-Out Cross-Validation ile veri setini K parçaya böl
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo)


# 6-Sonuçları yazdır
print("K-fold Cross-Validation Scores:", kfold_scores)
print("Stratified K-fold Cross-Validation Scores:", stratified_kfold_scores)
print("Leave-One-Out Cross-Validation Scores:", loo_scores)

print("K-fold Cross-Validation Mean Score:", np.mean(kfold_scores))
print("K-fold Cross-Validation Std Score:", np.std(kfold_scores))
print("Stratified K-fold Cross-Validation Mean Score:", np.mean(stratified_kfold_scores))
print("Stratified K-fold Cross-Validation Std Score:", np.std(stratified_kfold_scores))
print("Leave-One-Out Cross-Validation Mean Score:", np.mean(loo_scores))
print("Leave-One-Out Cross-Validation Std Score:", np.std(loo_scores))