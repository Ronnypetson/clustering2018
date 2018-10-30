from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import homogeneity_score
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
	return cluster[np.argmin(dist_matrix.sum(axis=0))]

k_max = 20
data_dir = 'word2vec.csv'
X = np.genfromtxt(data_dir,dtype=np.float32,delimiter=',',skip_header=0,encoding='ascii')
np.random.shuffle(X)

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

# Aplicar o PCA só depois no melhor
pca = PCA(0.95) # 'mle' # n_components=2,svd_solver='full'
pca.fit(X)

#print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_)) # explained_variance_ratio_

X_pca = pca.transform(X)
#print(X_pca.shape)
#X_pca = pca.transform(X)

#exit()

Js = []
for k in range(2,k_max): # 2 à 300, 301 à 500. Distribuir nos núcleos
	#kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X_pca)
	labels = kmeans.labels_

	# Get medoid
	#c_0 = []
	#for i in range(X_pca.shape[0]):
	#	if labels[i] == 0:
	#		c_0.append(X_pca[i])
	#print(medoid(c_0))

	#centroids = kmeans.cluster_centers_
	#J = np.mean([np.linalg.norm(X[i]-centroids[labels[i]]) for i in range(X.shape[0])])
	Js.append(kmeans.inertia_) # J
	#coef_homo = homogeneity_score(X_pca,labels)
	#print(coef_homo)
	#coef_silhueta = silhouette_score(X,labels,metric='euclidean')
	#print('Coef de silhueta para %d clusters: %f'%(k,coef_silhueta))

plt.plot(range(2,k_max),Js)
plt.show()

