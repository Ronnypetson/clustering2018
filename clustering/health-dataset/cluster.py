# Usage: python3 cluster.py > out.txt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import homogeneity_score
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np

# Get medoid, nearest to medoid, and its "normalized distance" to the medoid
def medoid(cluster):
	n_ = len(cluster)
	dist_matrix = np.zeros((n_,n_))
	for i in range(n_):
		for j in range(i+1,n_):
			dist = np.linalg.norm(cluster[i]-cluster[j])
			dist_matrix[i,j] = dist
			dist_matrix[j,i] = dist
	med = np.argmin(dist_matrix.sum(axis=0))
	near = 1 if med == 0 else 0
	for i in range(n_):
		if i != med:
			if dist_matrix[med,i] < dist_matrix[med,near]:
				near = i
	rel_dist = np.linalg.norm(cluster[near]-cluster[med])/np.mean(dist_matrix[med])
	return cluster[med], cluster[near], rel_dist

# Search for the range of K
def search_k(X,k_max=301):
	Js = []
	Calinks = []
	Davies = []
	for k in range(2,k_max):
		kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=-1).fit(X)
		labels = kmeans.labels_

		print(k)
		Js.append(kmeans.inertia_)
		calinks_score = metrics.calinski_harabaz_score(X, labels)
		Calinks.append(calinks_score)
		print('Calinks Score para %d clusters: %f'%(k,calinks_score))
		davies_score = metrics.davies_bouldin_score(X, labels)
		Davies.append(davies_score)
		print('Davies Bound Score para %d clusters: %f'%(k,davies_score))

	# Plot metrics as function of k
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

def eval_elbow(X,elbow=[40,60,80]):
	# Evaluate some values of K close to the "elbow"
	Js = []
	Calinks = []
	Davies = []
	for k in elbow:
		kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=-1).fit(X)
		labels = kmeans.labels_
		print('%d clusters'%k)

		# Get random cluster
		cluster_id = np.random.choice(k)
		cluster = []
		for i in range(X.shape[0]):
			if labels[i] == cluster_id:
				cluster.append(X[i])

		# Get medoid and nearest
		med, near, rel_dist = medoid(cluster)
		print('Medoid: {}'.format(med))
		print('Nearest to medoid: {}'.format(near))
		print('Relative distance from medoid to nearest: %f'%rel_dist)

		# Get scores
		print('Inertia: %f'%kmeans.inertia_)
		davies_score = metrics.davies_bouldin_score(X, labels)
		print('Davies: %f'%davies_score)
		coef_silhueta = silhouette_score(X,labels,metric='euclidean')
		print('Silhueta: %f'%coef_silhueta)

def eval_chosen_pca(X,k=80):
	Js = []
	Calinks = []
	Davies = []
	# Evaluate with k groups by using the data from PCA and different variances
	for v in [0.80, 0.85, 0.90, 0.95, 0.99]:
		print('Variance = %f'%v)
		pca = PCA(v)
		pca.fit(X)

		print('Number of components: %d'%len(pca.explained_variance_ratio_))
		print('Explained variance: {}'.format(pca.explained_variance_ratio_))

		X_pca = pca.transform(X)

		kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=-1).fit(X_pca)
		labels = kmeans.labels_

		# Get random cluster
		print()
		cluster_ids = np.random.choice(k,3,replace=False) # 3 clusters chosen randomly
		for c in cluster_ids:
			print('Cluster %d:'%c)
			cluster = []
			for i in range(X.shape[0]):
				if labels[i] == c:
					cluster.append(X_pca[i])
			# Get medoid and nearest
			med, near, rel_dist = medoid(cluster)
			print('Medoid: {}'.format(med))
			print('Nearest to medoid: {}'.format(near))
			print('Relative distance from medoid to nearest: %f'%rel_dist)
		print()

		# Get scores
		print('Inertia: %f'%kmeans.inertia_)
		davies_score = metrics.davies_bouldin_score(X_pca, labels)
		print('Davies: %f'%davies_score)
		coef_silhueta = silhouette_score(X_pca,labels,metric='euclidean')
		print('Silhueta: %f'%coef_silhueta)
		print()

def load_and_normalize_data(data_dir='word2vec.csv'):
	# Load data
	X = np.genfromtxt(data_dir,dtype=np.float32,delimiter=',',skip_header=0,encoding='ascii')

	# Normalize the data
	scaler = StandardScaler()
	scaler.fit(X)
	return scaler.transform(X)

X = load_and_normalize_data()
search_k(X)
eval_elbow(X)
eval_chosen_pca(X)

'''
med = input().split(' ')
med_f = []
for m in med:
	if len(m) > 1:
		med_f.append(float(m))

med_f = np.array(med_f)

for i in range(X.shape[0]):
	if np.linalg.norm(X[i]-med_f) < 0.001: # np.array_equal(x,med_f)
		print('Line %d'%(i+2))
		break
'''
