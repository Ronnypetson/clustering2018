from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import homogeneity_score
#from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np

def medoid(cluster):
	n_ = len(cluster)
	dist_matrix = np.zeros((n_,n_))
	for i in range(n_):
		for j in range(i+1,n_):
			dist = np.linalg.norm(cluster[i]-cluster[j])
			dist_matrix[i,j] = dist
			dist_matrix[j,i] = dist
	med = np.argmin(dist_matrix.sum(axis=0))
	near = np.argmin(dist_matrix[med])
	return cluster[med], cluster[near]

k_max = 10
data_dir = 'word2vec.csv'
X = np.genfromtxt(data_dir,dtype=np.float32,delimiter=',',skip_header=0,encoding='ascii')
np.random.shuffle(X)

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

Js = []
Calinks = []
Davies = []
all_labels = {}

for k in range(2,k_max): # 2 à 300 (ronny), 301 à 500 (letícia)
	#kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=-1).fit(X)
	labels = kmeans.labels_
	all_labels[k] = labels

	#centroids = kmeans.cluster_centers_
	#J = np.mean([np.linalg.norm(X[i]-centroids[labels[i]]) for i in range(X.shape[0])])
	print(k)
	Js.append(kmeans.inertia_) # J
	#coef_homo = homogeneity_score(X_pca,labels)
	#print(coef_homo)
	#coef_silhueta = silhouette_score(X,labels,metric='euclidean')
	#print('Coef de silhueta para %d clusters: %f'%(k,coef_silhueta))
	calinks_score = metrics.calinski_harabaz_score(X, labels)
	Calinks.append(calinks_score)
	print('Calinks Score para %d clusters: %f'%(k,calinks_score))
	davies_score = metrics.davies_bouldin_score(X, labels)
	Davies.append(davies_score)
	print('Davies Bound Score para %d clusters: %f'%(k,davies_score))

#avaliar os clusters finais com metrica local

'''
# Get medoid
labels = os labels do k escolhido
clusters_inds = np.random.choice(k,min(k,3),replace=False)
for c in cluster_inds:
	cl = []
	for i in range(X.shape[0]):
		if labels[i] == c:
			cl.append(X[i])
	print(medoid(cl))
'''

plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.plot(range(2,k_max),Js)
plt.show()

plt.xlabel('Number of clusters')
plt.ylabel('Calinks Score')
plt.plot(range(2,k_max),Calinks)
plt.show()

plt.xlabel('Number of clusters')
plt.ylabel('Davies Bound Score')
plt.plot(range(2,k_max),Davies)
plt.show()

'''
# Aplicar o PCA só depois no melhor
pca = PCA(0.95) # 'mle' # n_components=2,svd_solver='full'
pca.fit(X)

#print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_)) # explained_variance_ratio_

X_pca = pca.transform(X)
#print(X_pca.shape)
#X_pca = pca.transform(X)
'''

