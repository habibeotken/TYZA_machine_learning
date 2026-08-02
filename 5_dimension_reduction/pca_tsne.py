"""
digit veri setini pca ve tsne ile boyut indirgeme işlemlerini yapar ve görselleştirir.
4 boyuttan 2 boyuta indirger.


1- veri setini yükler
2-target ve feature olarak ayırır
3-veriler standardize edilir
4-pca modeli ile 2 boyuta indirger
5-pca ile indirgenmiş veriyi görselleştirir
6-tsne modeli ile 2 boyuta indirger
7-tsne ile indirgenmiş veriyi görselleştirir

"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

#veri setini yükle
digits = load_digits()

#target ve feature olarak ayır
X = digits.data
y = digits.target
plt.figure("feature göster")
for i in range(len(digits.feature_names)):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=digits.feature_names[i])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Digits Dataset')
plt.legend()
plt.show()

#verileri standardize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#pca modeli ile 2 boyutlu tanımla
pca = PCA(n_components=2)

#pca ile indirgenmiş veriyi oluştur
X_pca = pca.fit_transform(X_scaled)
print(f"X_scaled: {X_scaled}")
print()
print(f"X_pca: {X_pca}")


#pca ile indirgenmiş veriyi görselleştir
plt.figure()
for i in range(len (digits.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=digits.target_names[i])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Digits Dataset')
plt.legend()
plt.show()


#tsne modeli ile 2 boyutlu tanımla
tsne = TSNE(n_components=2, random_state=42)
#tsne ile indirgenmiş veriyi oluştur
X_tsne = tsne.fit_transform(X_scaled)
print(f"X_tsne: {X_tsne}")

#tsne ile indirgenmiş veriyi görselleştir
plt.figure()
for i in range(len(digits.target_names)):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=digits.target_names[i])
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE of Digits Dataset')
plt.legend()
plt.show()