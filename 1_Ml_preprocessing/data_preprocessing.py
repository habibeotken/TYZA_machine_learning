"""
makine öğrenmesi veri ön işleme pratikleri
"""

import pandas as pd
from sklearn.model_selection import train_test_split #eğitim ve test veriseti oluşturur
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler #veri ölçeklendirme


df =  pd.read_csv("musteri_verisi_ml_pratik.csv") #veri setini okuma
print(df.head()) #veri setinin ilk 5 satırını yazdırma
print(df.info()) #veri setinin özet bilgilerini yazdırma

print(df.isnull().sum()) #veri setindeki eksik değerleri yazdırma
df_dropna = df.dropna() #eksik değerleri silme
print(f"Veri setinin boyutu eksik değerler silindikten sonra: \n{df_dropna}") #eksik değerler silindikten sonra veri setinin boyutunu yazdırma

df_filled = df.copy() #veri setini kopyalama  

sayisal_sutunlar = ["yas", "deneyim_yili", "maas"] #sayısal sütunları belirleme 

#sayısal sütunlar için eksik değerleri medyan ile doldurma
for sutun in sayisal_sutunlar: #sayısal sütunlar için döngü
    medyan_degeri = df_filled[sutun].median() #medyan değerini hesaplama
    df_filled[sutun] = df_filled[sutun].fillna(medyan_degeri) #eksik değerleri medyan ile doldurma


#kategorik_sutunlarda eksik değerleri en sık görülen değer ile doldurma
df_filled["egitim"] = df_filled["egitim"].fillna(df_filled["egitim"].mode()[0]) #eksik değerleri en sık görülen değer ile doldurma  

print(f"Veri setinin boyutu eksik değerler doldurulduktan sonra: \n{df_filled}") #eksik değerler doldurulduktan sonra veri setinin boyutunu yazdırma

print("*********************************************************************************************")
#IQR yöntemi ile aykırı değerleri tespit etme 

aykiri_deger_maskesi = pd.Series(False, index=df_filled.index) #aykırı değer maskesi oluşturma

for sutun in sayisal_sutunlar: #sayısal sütunlar için döngü
    Q1 = df_filled[sutun].quantile(0.25) #1. çeyrek değerini hesaplama
    Q3 = df_filled[sutun].quantile(0.75) #3. çeyrek değerini hesaplama
    IQR = Q3 - Q1 #IQR değerini hesaplama
    alt_sinir = Q1 - 1.5 * IQR #alt sınırı hesaplama
    ust_sinir = Q3 + 1.5 * IQR #üst sınırı hesaplama
    sutun_maskesi = (df_filled[sutun] < alt_sinir) | (df_filled[sutun] > ust_sinir) #aykırı değer maskesini oluşturma
    aykiri_deger_maskesi |= sutun_maskesi #aykırı değer maskesini güncelleme

    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}") #aykırı değer sayısını yazdırma

if sutun_maskesi.any(): #aykırı değer maskesi varsa
        print(f"Aykırı değerler: \n{df_filled[sutun][sutun_maskesi]}") #aykırı değerleri yazdırma

print(f"en az bir aykırı değer içeren satırlar \n{df_filled[aykiri_deger_maskesi]}") #en az bir aykırı değer içeren satırları yazdırma


print("*********************************************************************************************")
#aykırı değerleri silme

df_clean = df_filled.loc[~aykiri_deger_maskesi].copy()   #aykırı değerleri silme
df_clean.reset_index(drop=True, inplace=True) #indeksleri sıfırlama
print(f"Veri setinin boyutu aykırı değerler silindikten sonra: \n{df_clean}") #aykırı değerler silindikten sonra veri setinin boyutunu yazdırma


print("*********************************************************************************************")

#label encoding ve one-hot encoding
#hedef değişkeni sayısal hale getir
label_encoder = LabelEncoder() #label encoder nesnesi oluşturma

y = label_encoder.fit_transform(df_clean["satin_aldi"]) #hedef değişkeni sayısal hale getirme , fit öğreniyor transform ise dönüştürüyor

print(f"hedef değişkenin sınıfları: \n{label_encoder.classes_}") #hedef değişkenin sayısal halini yazdırma
print(f"hedef değişkenin sayısal hali: \n{y}") #hedef değişkenin sayısal halini yazdırma

#hedef sütunu veri setinden çıkar 
x = df_clean.drop("satin_aldi", axis=1) #hedef sütunu veri setinden çıkarma
x = pd.get_dummies(x, columns=["egitim"], drop_first=True, dtype=int) #kategorik sütunları one-hot encoding ile sayısal hale getirme

print("*********************************************************************************************")
print(f"kategorik dönüşüm sonrası veri seti: \n{x}") #kategorik dönüşüm sonrası veri setini yazdırma

# veriyi train , testve validation olarak ayırma
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y) #veriyi train ve test olarak ayırma 

x_train, x_validation, y_train, y_validation = train_test_split(x_train_val, y_train_val, test_size=0.4, random_state=42, stratify =y_train_val) #veriyi train ve validation olarak ayırma

print(f"train veri setinin boyutu: {x_train.shape}, test veri setinin boyutu: {x_test.shape}, validation veri setinin boyutu: {x_validation.shape}") #veri setlerinin boyutlarını yazdırma  

#sayısal özelliklerde standardizasyon  

standard_scaler = StandardScaler() #standard scaler nesnesi oluşturma

x_train_standard = x_train.copy() #train veri setini kopyalama
x_validation_standard = x_validation.copy() #validation veri setini kopyalama   
x_test_standard = x_test.copy() #test veri setini kopyalama


#ölçekleyici sadece train veri seti üzerinde fit edilir ve transform edilir, validation ve test veri setleri ise sadece transform edilir. Bu sayede modelin gerçek performansı ölçülür.
x_train_standard[sayisal_sutunlar] = (
     standard_scaler.fit_transform(
          x_train[sayisal_sutunlar]
          ) #train veri setini standardizasyon ile dönüştürme
)

#validation ve test veri setleri sadece transform edilir
x_validation_standard[sayisal_sutunlar] = (
     standard_scaler.transform(
          x_validation[sayisal_sutunlar]
          ) #validation veri setini standardizasyon ile dönüştürme
)

x_test_standard[sayisal_sutunlar] = (
     standard_scaler.transform(
          x_test[sayisal_sutunlar]
          ) #test veri setini standardizasyon ile dönüştürme
)

print(f"train veri setinin standardizasyon sonrası boyutu: \n{x_train_standard}, test veri setinin standardizasyon sonrası boyutu: {x_test_standard.shape}, validation veri setinin standardizasyon sonrası boyutu: {x_validation_standard.shape}") #veri setlerinin boyutlarını yazdırma


print("*********************************************************************************************")
#normalizasyon işlemi
minmax_scaler = MinMaxScaler() #minmax scaler nesnesi oluşturma
x_train_minmax = x_train.copy() #train veri setini kopyalama
x_validation_minmax = x_validation.copy() #validation veri setini kopyalama
x_test_minmax = x_test.copy() #test veri setini kopyalama

x_train_minmax[sayisal_sutunlar] = (
     minmax_scaler.fit_transform(
          x_train[sayisal_sutunlar]
          ) #train veri setini normalizasyon ile dönüştürme
)

#validation ve test veri setleri sadece transform edilir
x_validation_minmax[sayisal_sutunlar] = (
     minmax_scaler.transform(
          x_validation[sayisal_sutunlar]
          ) #validation veri setini normalizasyon ile dönüştürme
)

x_test_minmax[sayisal_sutunlar] = (
     minmax_scaler.transform(
          x_test[sayisal_sutunlar]
          ) #test veri setini normalizasyon ile dönüştürme
)

print(f"train veri setinin normalizasyon sonrası boyutu: \n{x_train_minmax}, test veri setinin normalizasyon sonrası boyutu: {x_test_minmax.shape}, validation veri setinin normalizasyon sonrası boyutu: {x_validation_minmax.shape}") #veri setlerinin boyutlarını yazdırma