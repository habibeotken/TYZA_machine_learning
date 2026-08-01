"""
 Sentetik veri oluştur ve  K means ile agglomerative clustering algoritmalarını inceleme 
 K means ile kümelemenin nasıl olduğunu görselleştir
 dendrogram ile birleştirme yapısını inceleme


 1-veri seti oluşturma
 2-veri setini görselleştirme
 3- k means modelini eğitme 
 4- her veri noktasının hangi kümeye ait olduğunu görselleştirme
 5-centroid ile küme merkezlerini görselleştirme
 6_aynı veri ile hiyerarşik kümeleme 
 7-dendrogram ile birleştirme yapısını görselleştirme

"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


#veri setini oluşturma
X, _ = make_blobs(n_samples=300, centers=6, cluster_std=0.7, random_state=42)

#veri setini görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=30, edgecolor='k', alpha=0.5)


# K means modelini eğitme
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(X)


# Her veri noktasının hangi kümeye ait olduğunu görselleştirme
cluster_labels = kmeans.labels_
print("K-means Küme Etiketleri:\n", cluster_labels)

print("-----------------------------------------------------------")

#centroid ile küme merkezlerini görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=30, edgecolor='k', alpha=0.5, c=cluster_labels, cmap='viridis', marker='o')



#küme merkezleri 
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.6, marker='X', label='Centroid')

plt.title('K-means Kümeleme Sonuçları')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# aynı veri ile hiyerarşik kümeleme
agglo = AgglomerativeClustering(n_clusters=6)
agglo_labels = agglo.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], s=30, edgecolor='k', alpha=0.5, c=agglo_labels, cmap='viridis')
plt.title('Hiyerarşik Kümeleme Sonuçları')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#dendrogram ile birleştirme yapısını görselleştirme
linked = linkage(X, method='ward')
plt.figure()
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Örnekler')
plt.ylabel('Mesafe')
plt.show()