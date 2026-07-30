
"""
öznitelik mühendisliğine giriş

Amaç: 
1-mevcut sütunlardan yeni öznitelikler türetmek
2-korelasyon üzerinden modele daha faydalı olacak öznitelikleri seçmek

"""
import pandas as pd

df = pd.read_csv("oznitelik.csv")

print(df)

df["deneyim_rate"] = df["deneyim_yili"] / df["yas"]
df["yillik_harcama"] = df["aylik_harcama"] * 12


print (df)

#hedef değişken ile öznitelikler arasındaki korelasyonu hesaplamak

sayisal_df = df.drop(["sehir"], axis=1)
correlation = sayisal_df.corr(numeric_only=True)["performans_puani"].sort_values(ascending=False)

print(correlation)

#korelasyon değerine göre öznitelik seçimi yapmak

selected_features = correlation[abs(correlation) > 0.5].index.tolist()
selected_features.remove("performans_puani")  # hedef değişkeni listeden çıkar

print("Seçilen öznitelikler:", selected_features)