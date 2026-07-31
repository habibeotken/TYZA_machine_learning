"""
regresyon modelleri pratiği
1-sentetik veri seti oluşturma
2-doğrusal,polynomial,lasso ve ridge regresyon modelleri oluşturma
3-lasso ile feature selecetion yapma


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#sentetik veri seti oluşturma
np.random.seed(42)

#4 features

x1 = np.random.uniform(0, 10, 200)
x2 = np.random.uniform(0, 10, 200)
x3 = np.random.uniform(0, 10, 200)
x4 = np.random.uniform(0, 10, 200)

x = np.column_stack((x1, x2, x3, x4)) #bağımsız değişkenler
print(x)
print("Tüm eksik değerler (X):")
print(pd.isna(x).sum())

#x1 ve x2 ve x3 önemli features , x4 önemsiz features
y = (
    4
    +0.5 * x1
    +1.8 * x2
    + 0.15 * (x1 * 2)
    - 0.1 * (x2 ** 2)
    + 0.2 * x1 * x2
    +1.9 * x3
    + np.random.normal(0, 2, 200) #gürültü ekleme


)

#oluşturulan veri setini görselleştirme

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, y, c='blue', marker='o')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
fig.savefig("scatter_plot.png")
plt.close(fig)

#veriyi eğitim ve test olarak ayırma

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#modelleri oluşturma
 
linear_model = LinearRegression()
polynomial_model = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("linear_regression", LinearRegression())
    ]
)

lasso_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.1))
    ])


ridge_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.1))
    ])

#modelleri eğitme ve tahmin yapma

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
polynomial_model.fit(X_train, y_train)

#predictions

y_linear_pred = linear_model.predict(X_test)
y_ridge_pred = ridge_model.predict(X_test)
y_lasso_pred = lasso_model.predict(X_test)
y_polynomial_pred = polynomial_model.predict(X_test)

#modellerin performansını değerlendirme

print("Linear Regression MSE:", mean_squared_error(y_test, y_linear_pred), "R2 Score:", r2_score(y_test, y_linear_pred))
print("Ridge Regression MSE:", mean_squared_error(y_test, y_ridge_pred), "R2 Score:", r2_score(y_test, y_ridge_pred))
print("Lasso Regression MSE:", mean_squared_error(y_test, y_lasso_pred), "R2 Score:", r2_score(y_test, y_lasso_pred))
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_polynomial_pred), "R2 Score:", r2_score(y_test, y_polynomial_pred))

#lasso ile feature selection yapma

ozellik_names = np.array(['x1', 'x2', 'x3', 'x4'])

lasso_coef = lasso_model.named_steps['lasso'].coef_

for name, coef in zip(ozellik_names, lasso_coef):
    print(f"{name}: {coef}")


