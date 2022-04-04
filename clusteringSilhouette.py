from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Data import create_blob
from plotUtils import plot_silhouette

X, _ = create_blob(n_sample=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1)

# with three clusters
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plot_silhouette(X, y_km)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()

# with two clusters
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plot_silhouette(X, y_km)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()
