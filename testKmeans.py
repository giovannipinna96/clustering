from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Data import create_blob
from plotUtils import plot_blob, plot_three_clusters

X, _ = create_blob(n_sample=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1)

plot_blob(X)
plt.title('Data')
plt.show()

km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=1)
y_km = km.fit_predict(X)

plot_three_clusters(X, y_km)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black',
            label='Centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.title('Kmenas clusters')
plt.show()

print('Distrosion: %.2f' % km.inertia_)

# now we used the kmenas ++ and not random
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()
# The perfect number of cluster in this case is 3

