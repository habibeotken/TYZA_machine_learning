# Calisan Isten Ayrilma Tahmini

Bu proje, çalışanların işten ayrılıp ayrılmayacağını tahmin etmek için kurulan bir makine öğrenmesi akışını içerir. Veri setinde yaş, gelir, eğitim, deneyim, performans, fazla mesai, memnuniyet ve benzeri değişkenler kullanılır. Script; eksik değerleri işler, kategorik verileri sayısal hale getirir, yeni bir özellik üretir ve Logistic Regression, KNN ile Decision Tree modellerini karşılaştırır.

## Nasıl Çalıştırılır

1. Bu klasörde bir Python sanal ortamı aktif edin.
2. Gerekli paketleri kurun:

```bash
pip install -r requirement.txt
pip install scikit-learn matplotlib
```

3. Modeli çalıştırın:

```bash
python musteri_ayrilma.py
```

Veri dosyası olan `calisan_isten_ayrilma_100_eksik_verili.csv` script ile aynı klasörde bulunmalıdır.

## Kısa Sonuç Yorumu

Kod, validation set üzerindeki accuracy değerlerine göre en iyi modeli seçer ve test setinde confusion matrix, accuracy, precision, recall ve F1 score hesaplar. Genel amaç, hangi modelin çalışan ayrılmasını daha başarılı tahmin ettiğini görmektir. Çalıştırma sonucunda ekrana basılan metrikler üzerinden en iyi model yorumlanır.