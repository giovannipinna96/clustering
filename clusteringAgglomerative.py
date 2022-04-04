from sklearn.cluster import AgglomerativeClustering
from Data import create_blob
from plotUtils import plot_three_clusters
import matplotlib.pyplot as plt

X, _ = create_blob()

ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)

plot_three_clusters(X, labels)
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.title('Kmenas clusters')
plt.show()

# ac = AgglomerativeClustering(n_clusters=2,
#                              affinity='euclidean',
#                              linkage='complete')
# labels = ac.fit_predict(X)
# print('Cluster labels: %s' % labels)
