"""
KNN, Decision Tree ve Logistic Regression modellerini grid ve random search ile hiperparametre optimizasyonu yapma
Farklı arama yöntemlerini kullanarak modellerin performansını karşılaştırma

1-veri setini yükle
2-test ve train olarak ayır
3-Modeller için hiperparametre uzayını belirle 
4-Grid Search ve Random Search ile hiperparametre optimizasyonu yap
5-Modellerin performansını karşılaştır

"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




# Veri setini yükle
data = load_breast_cancer()
X, y = data.data, data.target

# Test ve train olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modeller için hiperparametre uzayını belirle
models_and_parameters = {
    'KNN': {
        'pipeline': Pipeline([
            ("scaler", StandardScaler()),
            ("knn_model", KNeighborsClassifier())
        ]),
        "param_grid": {
            'knn_model__n_neighbors': [3, 5, 7, 9],
            'knn_model__metric': ["euclidean", "manhattan"]
        },
        "random_param_grid": {
            'knn_model__n_neighbors': [3, 5, 7,9],
            'knn_model__metric': ["euclidean", "manhattan"]
        }
    },
    'DecisionTree': {
        "pipeline": Pipeline([
            ("dt_model", DecisionTreeClassifier(random_state=42))
        ]),
        "param_grid": {
            'dt_model__max_depth': [None, 5, 10, 15],
            'dt_model__min_samples_split': [2, 5, 10],
            'dt_model__criterion': ['gini', 'entropy']
        },
        "random_param_grid": {
            'dt_model__max_depth': [None, 5, 10, 15],
            'dt_model__min_samples_split': [2, 5, 10],
            'dt_model__criterion': ['gini', 'entropy']
        }
    },
    'LogisticRegression': {
        "pipeline": Pipeline([
            ("lr_model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "param_grid": {
            'lr_model__C': [0.01, 0.1, 1, 10],
            'lr_model__penalty': ['l1', 'l2'],
        
        },
        "random_param_grid": {
            'lr_model__C': [0.01, 0.1, 1, 10],
            'lr_model__penalty': ['l1', 'l2'],
          
        }
    }
}

results = []

#grid ve random ile en iyi hiperparametreleri bul
#en iyi modeli test setinde değerlendir
for model_name, model_info in models_and_parameters.items():
    # Grid Search
    grid_search = GridSearchCV(
        estimator=model_info["pipeline"],
        param_grid=model_info["param_grid"],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_best_model = accuracy_score(y_test, grid_search.predict(X_test))

    # Random Search
    random_search = RandomizedSearchCV(
        estimator=model_info["pipeline"],
        param_distributions=model_info["random_param_grid"],
        n_iter=10,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    random_best_model = accuracy_score(y_test, random_search.predict(X_test))

    results.append({
        "model": model_name,
        "yontem": "grid_search",
        "cv en iyi skor": round(grid_search.best_score_, 2),
        "test skor": round(grid_best_model, 2),
        "en iyi parametreler": str(grid_search.best_params_)
    })
    
    results.append({
        "model": model_name,
        "yontem": "random_search",
        "cv en iyi skor": round(random_search.best_score_, 2),
        "test skor": round(random_best_model, 2),
        "en iyi parametreler": str(random_search.best_params_)
    })

    #Sonuçları özet olarak yazdır
    pd.set_option("display.max_colwidth", None)
    result_df = pd.DataFrame(results)
    print(result_df)

